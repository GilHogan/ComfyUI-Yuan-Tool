"""H3 运动上下文/裁剪节点：全面对齐官方 Add Guide for MiniMax H3 数据路径。

「上下文」把上一片段尾部媒体编码为关键帧（minimax_keyframes）固定到本片段开头；
「裁剪」把 AV 潜空间解码后从头部裁掉被固定窗口覆盖的帧并保存尾段供下一片段衔接。
像素域裁切无潜空间 VRF 相位约束，任意帧数均可干净裁掉。
"""

import hashlib
import os

import numpy as np
import folder_paths
import node_helpers
import torch
import torchaudio

import comfy.utils
from server import PromptServer
from aiohttp import web

from .Yuan_common import handle_chunk_upload

try:
    from safetensors.torch import load_file as _st_load
    from safetensors import safe_open as _st_safe_open
except ImportError:  # ComfyUI 总是自带 safetensors，此处仅是双保险
    _st_load = _st_safe_open = None


# ============================================================================
# H3 常量与潜空间工具函数
# ============================================================================

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FPS = 24  # H3 原生帧率；音频潜空间按 40Hz 采样


def _pixel_frames(latent_t):
    """latent_t 个潜空间步覆盖的像素帧数。"""
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(latent_t))


def _streams_from_latent(latent):
    """解包 H3 AV 潜空间为所含各流；NestedTensor 须用 unbind()——samples[0]
    会把索引广播进两流并剥掉批维，得不到单条流。"""
    samples = latent["samples"]
    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        parts = list(samples)
    else:
        raise ValueError(
            "h3_motion_context: expected a MiniMax H3 AV latent (a nested "
            "video/audio pair), got %r" % type(samples))
    if not parts:
        raise ValueError("h3_motion_context: AV latent contains no streams")
    return parts


def _snap_guide_frames(n):
    """向下吸附到合法引导长度（17k+5：5/22/39/56…，同官方 Add Guide 多帧批处理）；小于 5 帧返回 0 由调用方报错。"""
    n = int(n)
    if n < 5:
        return 0
    while n % 17 != 5:
        n -= 1
    return n


def _resize_guide(image, width, height):
    """把引导帧缩放到目标分辨率（与官方 Add Guide _resize(..., "center") 逐位一致：
    取前 3 通道、lanczos、保宽高比的居中裁剪）。"""
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos",
                                         "center")
    return samples.movedim(1, -1)


def _tail_audio_latent(audio_vae, waveform, sample_rate, frames):
    """取波形尾部 frames 帧对应样本，按官方 Add Guide _encode_ref_audio 同路径编码。

    样本数向下对齐 hop（800 样本/步≈25ms@32kHz），使包装器的编码前裁剪恰好
    无操作、窗口尾端不发生亚步偏移。"""
    vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if waveform.ndim == 2:
        waveform = waveform[None]
    if int(sample_rate) != vae_sr:
        waveform = torchaudio.functional.resample(
            waveform, int(sample_rate), vae_sr)
    try:
        hop = int(audio_vae.downscale_ratio)
    except (TypeError, ValueError, AttributeError):
        hop = 800
    if hop < 1:
        hop = 800
    need = int(round(frames / float(FPS) * vae_sr))
    need -= need % hop
    length = int(waveform.shape[-1])
    if need > length:
        need = length - (length % hop)
    if need < hop:
        raise ValueError(
            "h3_motion_context: 音频上下文窗口为空（可用音频不足 %d 样本）。"
            % hop)
    z = audio_vae.encode(waveform[:1, ..., length - need:].movedim(1, -1))
    if z.ndim == 3:
        z = z.unsqueeze(0)
    return z


# ============================================================================
# 上下文媒体标记与加载（自动索引 / 手动上传共用）
# ============================================================================

# 空标记键：片段序号为 0 / 文件未找到时，加载输出带此标记的空媒体，
# 运动上下文据此直通、不裁头
CONTEXT_EMPTY_MARKER = "_h3_motion_context_empty"
# 原因键：空标记的原因（first_clip / file_not_found）
CONTEXT_EMPTY_REASON = "_h3_motion_context_empty_reason"
# 细节键：原因细节（如未找到的片段序号）
CONTEXT_EMPTY_REASON_DETAIL = "_h3_motion_context_empty_reason_detail"
# 正常加载时携带的片段序号键，用于生成"已关联片段 N"提示
CONTEXT_CLIP_INDEX_KEY = "_h3_motion_context_clip_index"

# 尾段媒体文件名约定：存储位置目录下 clip_%05d.mp4（片段序号 2 → clip_00002.mp4）
CLIP_FILE_PREFIX = "clip"
CLIP_FILE_EXT = ".mp4"
# 加载上下文媒体时解码帧的最长边上限：超过该值先等比缩到最长边=1536 再参与
# 引导，降低解码量与后续 VAE 编码开销；未超限原样返回（字节不变），不破坏
# 链条内小分辨率尾段的逐字节精确。固定默认，不暴露参数。
CONTEXT_LOAD_MAX_SIDE = 1536


def _load_media_file(path, clip_index=None):
    """加载「H3 运动裁剪」保存的尾段媒体：.mp4 经 av 解码为像素帧+波形；
    旧版 .safetensors（uint8 帧+波形）仍兼容读取。

    返回 {"pixels": [t,H,W,3] float[0,1], "waveform": [1,2,L]（无音轨为 None）,
    "sample_rate": int}。
    """
    lower = (path or "").lower()
    if lower.endswith(".mp4"):
        return _read_mp4(path, clip_index=clip_index)
    return _read_st_media(path, clip_index=clip_index)


def _cap_longest_side(frames_u8):
    """把 uint8 视频帧等比缩到最长边 ≤ CONTEXT_LOAD_MAX_SIDE。

    仅当超上限时才缩放，未超限原样返回（字节不变）——链条内小分辨率尾段
    保持逐字节精确。支持单帧 [H,W,3] 与帧序列 [T,H,W,3]；解码循环里逐帧
    调用可在堆叠成大张量前先降内存。"""
    h, w = int(frames_u8.shape[-3]), int(frames_u8.shape[-2])
    if max(h, w) <= CONTEXT_LOAD_MAX_SIDE:
        return frames_u8
    ratio = CONTEXT_LOAD_MAX_SIDE / float(max(h, w))
    nh = max(int(h * ratio + 0.5), 2)
    nw = max(int(w * ratio + 0.5), 2)
    batched = frames_u8.ndim == 4
    x = frames_u8 if batched else frames_u8[None]
    x = x.float().div_(255.0).movedim(-1, 1)  # [*,3,H,W]
    y = comfy.utils.common_upscale(x, nw, nh, "area", "disabled")
    out = (y.movedim(1, -1).mul_(255.0).clamp_(0.0, 255.0)
           .round_().to(torch.uint8))
    return out if batched else out[0]


def _read_st_media(path, clip_index=None):
    """读取旧版 .safetensors 格式的尾段媒体（uint8 帧 + 音频波形）。"""
    if _st_load is None:
        raise RuntimeError("h3_motion_context: safetensors is not "
                           "available; cannot load context media.")
    data = _st_load(path)
    if "video" not in data or "audio" not in data:
        raise ValueError(
            "h3_motion_context: %s 不是「H3 运动裁剪」保存的上下文媒体文件"
            "（缺少 video/audio 数据）。" % path)
    video = data["video"]
    # 与主路径一致：4D [T,H,W,3] 即帧序列本身，只有 5D [B,T,H,W,3] 才去批维
    if video.ndim == 5:
        video = video[0]
    if video.ndim == 3:
        video = video[None]
    if video.dtype != torch.uint8:
        raise ValueError(
            "h3_motion_context: %s 是旧版潜空间格式（latent）。请用新版"
            "「H3 运动裁剪」重新生成分段媒体文件。" % path)
    video = _cap_longest_side(video)  # 解码即统一缩放（超上限才缩）
    pixels = video.float().div_(255.0)
    wave = data["audio"]
    if wave.ndim == 3:
        wave = wave[:1]
    elif wave.ndim == 2:
        wave = wave[None]
    sample_rate = 32000
    if _st_safe_open is not None:
        try:
            with _st_safe_open(path, framework="pt", device="cpu") as f:
                meta = f.metadata() or {}
                sample_rate = int(meta.get("sample_rate", 32000))
                if clip_index is None and meta.get("clip_index"):
                    clip_index = int(meta["clip_index"])
        except Exception:
            pass
    out = {"pixels": pixels, "waveform": wave, "sample_rate": sample_rate}
    if clip_index is not None:
        out[CONTEXT_CLIP_INDEX_KEY] = clip_index
    return out


def _read_mp4(path, clip_index=None):
    """把「H3 运动裁剪」保存的 .mp4（音视频一体）解码为像素帧与波形。

    视频帧 → uint8 → [0,1] float 的 [t,H,W,3]；音频重采样为 float32 立体声
    [2,L]（保留容器原始采样率，由调用方后续转 32kHz）；无音轨时 waveform 为 None。"""
    import av
    video_frames = []
    audio_parts = []
    sample_rate = 32000
    container = av.open(path)
    try:
        video_stream = next((s for s in container.streams if s.type == "video"), None)
        audio_stream = next((s for s in container.streams if s.type == "audio"), None)
        if video_stream is None:
            raise ValueError("h3_motion_context: %s 内没有视频轨。" % path)
        resampler = None
        if audio_stream is not None:
            sample_rate = int(audio_stream.codec_context.sample_rate or 48000)
            resampler = av.audio.resampler.AudioResampler(
                format="fltp", layout="stereo", rate=sample_rate)
        # 视频/音频须在同一次 decode 中交错取帧：先取完单流会把文件读到
        # EOF，后续再 decode 另一流将拿不到任何帧（av 18 实测返回空）
        targets = [s for s in (video_stream, audio_stream) if s is not None]
        for frame in container.decode(*targets):
            if isinstance(frame, av.VideoFrame):
                arr = frame.to_ndarray(format="rgb24")  # [H,W,3] uint8
                # 解码即统一缩放：逐帧等比缩到最长边 ≤ 1536（超上限才缩，
                # 在堆叠成大张量前先降内存），再参与引导
                video_frames.append(_cap_longest_side(
                    torch.from_numpy(np.ascontiguousarray(arr))))
            elif resampler is not None:
                for rf in resampler.resample(frame):
                    # to_ndarray 返回 [声道,样本]，且会按实际样本裁剪——
                    # 直接用 planes buffer 会连带对齐填充读到多余样本
                    nd = rf.to_ndarray()
                    audio_parts.append(torch.from_numpy(
                        np.ascontiguousarray(nd)))
        if resampler is not None:
            for rf in resampler.resample(None):  # 冲刷重采样器尾部
                nd = rf.to_ndarray()
                audio_parts.append(torch.from_numpy(
                    np.ascontiguousarray(nd)))
    finally:
        container.close()
    if not video_frames:
        raise ValueError("h3_motion_context: %s 未解码到任何视频帧。" % path)
    pixels = torch.stack(video_frames, 0).float().div_(255.0)
    waveform = None
    if audio_parts:
        wave = torch.cat(audio_parts, 1)  # [声道, L]
        if wave.shape[0] == 1:  # 单声道提升为立体声
            wave = wave.repeat(2, 1)
        elif wave.shape[0] > 2:
            wave = wave[:2]
        waveform = wave.unsqueeze(0)  # [1,2,L]
    out = {"pixels": pixels, "waveform": waveform, "sample_rate": sample_rate}
    if clip_index is not None:
        out[CONTEXT_CLIP_INDEX_KEY] = clip_index
    return out


def _load_context_media(存储位置, 片段序号=1):
    """按 存储位置+片段序号 加载本地尾段媒体：序号 0（首片段）返回 first_clip 空标记；
    >0 按 存储位置/clip_%05d.mp4 加载，未找到时返回 file_not_found 空标记。
    两种空标记调用方均直通、不裁头。"""
    try:
        idx = int(片段序号)
    except (TypeError, ValueError):
        raise ValueError("h3_motion_context: 片段序号必须是整数，得到 %r"
                         % (片段序号,))
    if idx == 0:
        return {CONTEXT_EMPTY_MARKER: True,
                CONTEXT_EMPTY_REASON: "first_clip"}
    try:
        path = _clip_file_path(存储位置, idx)
    except FileNotFoundError:
        # 兜底：主「存储位置」目录缺失（参数错位）时，扫描 output 目录实际保存位置
        path = _find_clip_in_output(idx)
        if path is None:
            return {CONTEXT_EMPTY_MARKER: True,
                    CONTEXT_EMPTY_REASON: "file_not_found",
                    CONTEXT_EMPTY_REASON_DETAIL: str(idx)}
    return _load_media_file(path, clip_index=idx)


def _context_latent_fingerprint(存储位置, 片段序号, 手动上传):
    """IS_CHANGED 缓存指纹：手动上传用文件指纹；片段序号常量 0 输出确定性 0；
    否则对存储目录全部尾段媒体做综合指纹（内容变→指纹变→下游重跑；同内容→命中）；
    异常统一返回 NaN 保守重跑。"""
    if (手动上传 or "").strip():
        try:
            path = _resolve_manual_media_path(手动上传)
            st = os.stat(path)
            return "manual:%s:%s:%s" % (path, st.st_mtime_ns, st.st_size)
        except Exception:
            return float("NaN")
    try:
        if int(片段序号) == 0:
            return 0
    except (TypeError, ValueError):
        pass
    try:
        fp = _dir_fingerprint(_build_load_path(存储位置))
    except Exception:
        return float("NaN")
    # 与加载时的兜底扫描保持一致：主存储位置目录不存在（参数错位）时，
    # 改扫 output 目录下实际保存的文件做指纹，避免缓存漏跑/误缓存
    if fp.startswith("missing"):
        alt = _find_clip_in_output(片段序号)
        if alt:
            try:
                st = os.stat(alt)
                return "found:%s:%d:%d" % (alt, st.st_mtime_ns, st.st_size)
            except OSError:
                return float("NaN")
    return fp


# ============================================================================
# H3 运动上下文：把上一片段尾部媒体固定为本片段的关键帧引导
# ============================================================================

class Yuan_H3MotionContext:
    """把上一片段尾部画面/音频固定为本片段开头：以 resolved_frame_index=0 锚定，
    与官方 Add Guide 同路径——画面经视频 VAE 编码、音频经音频 VAE 编码后追加进
    minimax_keyframes（可与官方引导节点自由混用）。上下文来源由「模式」决定：
    上传 / 端口 / 自动索引。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "条件化": ("CONDITIONING", {
                    "tooltip": "正向条件化。本节点向其追加关键帧引导后输出，"
                               "可与官方 Add Guide 等 H3 条件节点串联。"}),
                "潜空间": ("LATENT", {
                    "tooltip": "本片段的 H3 AV 潜空间（采样器或空 latent 节点"
                               "输出）。仅读取形状（时长/分辨率/音频轨长度），"
                               "不修改其内容。"}),
                "VAE": ("VAE", {
                    "tooltip": "H3 视频 VAE。把上一片段尾帧编码为关键帧"
                               " latent，与官方 Add Guide 连接 image 的路径"
                               "相同。"}),
                "模式": (["上传", "端口", "自动索引"], {
                    "default": "自动索引",
                    "tooltip": "上下文来源三选一：上传——仅用手动上传的媒体"
                               "文件；端口——仅用「上下文图像」「上下文音频」"
                               "端口的连线（如直接连上一片段「H3 运动裁剪」"
                               "的输出）；自动索引——按 存储位置+片段序号 "
                               "自动加载本地保存的尾段媒体。"}),
                "启用上下文": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "总开关。关闭时条件化直通、不固定任何引导，"
                               "输出 \"0:尾段长度\"——本片段完全独立生成，"
                               "但「运动裁剪」的尾段仍按两窗口较大值保存，"
                               "供下一片段衔接。"}),
                "存储位置": ("STRING", {
                    "default": "H3-Mubu",
                    "tooltip": "自动索引模式下加载的目录名（ComfyUI 输出"
                               "文件夹下的子目录）。与「H3 运动裁剪」的"
                               "「存储位置」一致即可对应加载。"}),
                "片段序号": ("INT", {
                    "default": 1, "min": 0, "max": 9999,
                    "tooltip": "自动索引模式下加载的片段序号：设为上一片段"
                               "「H3 运动裁剪」保存时使用的相同序号即可对应"
                               "加载（clip_00002.mp4 这类文件）。0 表示"
                               "链条第一个片段，不加载、直通。"}),
                "上下文长度": (["5", "22", "39", "56"], {
                    "default": "5",
                    "tooltip": "固定到本片段开头的画面帧数，必须是 H3 引导"
                               "片段的合法长度（17k+5：5/22/39/56，与官方"
                               " Add Guide 的多帧引导一致），其他值向下吸附。"
                               "「运动裁剪」将从交付部分裁掉同等帧数，接缝处"
                               "画面从引导末端无缝续接。锚窗越长自由帧越易被"
                               "拉向上一段画面并逐段污染，默认 5 帧短锚接缝"
                               "依然无缝、污染最小；长锚仅强延续需求时用。"}),
                "音频上下文长度": (["0", "5", "22", "39", "56"], {
                    "default": "5",
                    "tooltip": "从上一片段尾部固定的音频时长（按帧数换算），"
                               "经音频 VAE 编码后随关键帧锚定在本片段开头，"
                               "交付音频从固定窗口末端无缝续接。0=不固定音频"
                               "（模型自由生成，固定的画面可能带出上一片段"
                               "场景的声音）。建议与「上下文长度」保持一致"
                               "（超出时按画面窗收窄），大于 0 时需连接 "
                               "audio_vae。"}),
            },
            "optional": {
                "audio_vae": ("VAE", {
                    "tooltip": "H3 音频 VAE。「音频上下文长度」大于 0 时必须"
                               "连接，用于把上一片段尾部的音频波形编码为关键"
                               "帧音频潜空间。"}),
                "上下文图像": ("IMAGE", {
                    "tooltip": "端口模式下的上一片段尾部画面（如直接连上一"
                               "片段「H3 运动裁剪」的「图像」输出）。本节点"
                               "取其尾部「上下文长度」帧作引导。"}),
                "上下文音频": ("AUDIO", {
                    "tooltip": "端口模式下的上一片段尾部音频（如直接连上一"
                               "片段「H3 运动裁剪」的「音频」输出）。「音频"
                               "上下文长度」大于 0 时必须连接。"}),
                "手动上传": ("STRING", {
                    "default": "",
                    "tooltip": "上传模式下由「上传上下文」按钮写入的媒体文件"
                               "路径，也可手填（支持 input:/output:/temp: "
                               "前缀）。"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("条件化", "裁剪帧数")
    FUNCTION = "apply"
    CATEGORY = "Yuan Tool/MiniMax"
    DESCRIPTION = ("把上一片段尾部的视频图像与音频固定为本片段开头的"
                   "关键帧引导，全面对齐官方 Add Guide for MiniMax H3 的"
                   "数据路径：图像经视频 VAE 编码、音频经音频 VAE 编码，"
                   "锚定在第 0 帧并追加进正向条件化。支持 上传/端口/自动"
                   "索引 三种上下文来源。裁剪帧数输出为字符串\"状态:长度\""
                   "（如 1:22），供「H3 运动裁剪」解析。")

    def apply(self, 条件化, 潜空间, VAE, 启用上下文=True, 模式="自动索引",
              存储位置="H3-Mubu", 片段序号=1, 上下文长度="5",
              音频上下文长度="5", audio_vae=None,
              上下文图像=None, 上下文音频=None, 手动上传=""):
        # 引导窗口与音频窗口的请求值；无上下文/直通路径下，尾段保存长度
        # 仍按两窗口较大值输出，保证下一片段有可衔接的媒体文件
        g_req = _snap_guide_frames(int(上下文长度 or 0))
        a_req = int(音频上下文长度 or 0)
        idle_tail = max(g_req, a_req)
        if not 启用上下文:
            return {"result": (条件化, "0:%d" % idle_tail), "ui": {
                "h3_hint": "上下文已关闭，直通"}}
        # 上下文来源由「模式」决定：上传（只用上传文件）/ 端口（只用连线）/ 自动索引
        if 模式 == "上传":
            if not (手动上传 or "").strip():
                raise ValueError(
                    "h3_motion_context: 「上传」模式下需先通过「上传上下文」"
                    "按钮上传媒体文件（.mp4），或填写「手动上传」路径。")
            media = _load_media_file(_resolve_manual_media_path(手动上传),
                                     clip_index=None)
            hint = "已上传上下文文件"
        elif 模式 == "端口":
            if 上下文图像 is None:
                raise ValueError(
                    "h3_motion_context: 「端口」模式下「上下文图像」"
                    "端口必须连接。")
            waveform = None
            sample_rate = 32000
            if isinstance(上下文音频, dict):
                waveform = 上下文音频.get("waveform")
                sample_rate = int(上下文音频.get("sample_rate", 32000))
            media = {"pixels": 上下文图像, "waveform": waveform,
                     "sample_rate": sample_rate}
            hint = "已关联上下文"
        else:  # 自动索引
            media = _load_context_media(存储位置, 片段序号)
            if media.get(CONTEXT_EMPTY_MARKER):
                reason = media.get(CONTEXT_EMPTY_REASON, "first_clip")
                if reason == "file_not_found":
                    detail = media.get(CONTEXT_EMPTY_REASON_DETAIL, "?")
                    return {"result": (条件化, "0:%d" % idle_tail), "ui": {
                        "h3_hint": "未找到片段 %s 文件" % detail}}
                return {"result": (条件化, "0:%d" % idle_tail), "ui": {
                    "h3_hint": "片段\"0\"，直通"}}
            hint = "已关联片段 %s 文件" % media.get(
                CONTEXT_CLIP_INDEX_KEY, "?")

        # 仅读取本片段形状：按「通道数 24」从 AV 两流中识别视频流（不假设顺序），
        # 兼容 NestedTensor 与普通 (video, audio) 元组
        parts = _streams_from_latent(潜空间)
        video = None
        for pt in parts:
            v = pt if pt.ndim != 4 else pt.unsqueeze(0)
            if v.ndim == 5 and v.shape[1] == 24:
                video = v
                break
        if video is None:
            raise ValueError(
                "h3_motion_context: 未在 AV 潜空间中找到 24 通道视频流，流形状="
                "%s。请连接 MiniMax H3 采样器/空潜空间节点的输出。"
                % ([tuple(t.shape) for t in parts],))
        latent_t = int(video.shape[2])
        width = int(video.shape[4]) * 16
        height = int(video.shape[3]) * 16
        frame_count = _pixel_frames(latent_t)
        track_steps = None
        for pt in parts:
            if pt.ndim >= 1 and pt.shape[0] == 1 and pt.ndim >= 3 and pt is not video:
                try:
                    track_steps = int(pt.shape[-1])
                except (TypeError, ValueError):
                    pass

        pixels = media["pixels"]
        # 媒体像素约定 [T,H,W,3]；仅去掉真实的批维（5D [B,T,H,W,3]）。
        # 4D 就是帧序列本身，不能截断——早期把 4D 误当批维截成首帧，正是
        # 「裁剪保存的尾段加载后只剩 1 帧/引导断裂」的根因之一。
        if pixels.ndim == 5:
            pixels = pixels[0]
        elif pixels.ndim == 3:  # 单帧 [H,W,3] 补帧轴，交由下方帧数校验兜底
            pixels = pixels[None]
        available = int(pixels.shape[0])
        # 引导长度：min(请求,可用) 后吸附 17k+5（官方对多帧引导同款向下裁剪）
        g = _snap_guide_frames(min(g_req, available))
        if g < 5:
            raise ValueError(
                "h3_motion_context: 上下文媒体仅有 %d 帧画面，至少需要 5 帧"
                "才能固定引导。" % available)
        if g > frame_count:
            raise ValueError(
                "h3_motion_context: 引导片段 %d 帧超出本片段总长 %d 帧。"
                % (g, frame_count))

        # 引导帧：取尾部 g 帧，按官方 Add Guide 同款 center 裁剪缩放到
        # 本片段分辨率，经视频 VAE 编码为关键帧 latent
        guide = _resize_guide(pixels[available - g:], width, height)
        if width % 16 or height % 16:
            raise ValueError(
                "h3_motion_context: 本片段分辨率 %dx%d 不是 16 的倍数，无法"
                "编码引导。请调整工作流的目标分辨率。"
                % (width, height))
        try:
            keyframe = {"resolved_frame_index": 0,
                        "latent": VAE.encode(guide.contiguous())}
        except RuntimeError as e:
            raise RuntimeError(
                "h3_motion_context: VAE 编码引导片段失败。\n"
                "  引导帧形状 guide=%s（g=%d 帧, 目标分辨率 %dx%d）\n"
                "  本片段视频潜空间形状=%s（像素 %dx%d, 共 %d 帧）\n"
                "  原始错误：%s\n"
                "请检查「H3 运动上下文」的「潜空间」是否接自 H3 采样器输出、"
                "上下文的画面分辨率是否与片段一致（不一致时按本片段分辨率"
                "缩放）。"
                % (tuple(guide.shape), g, width, height,
                   tuple(video.shape), width, height, frame_count, e))

        # 音频引导：取尾部音频窗（帧数换算样本）经音频 VAE 编码，按官方规则裁到本片段音频轨剩余长度（frame_idx=0→全轨可用）
        a = 0
        waveform = media.get("waveform")
        if a_req > 0:
            if waveform is None or int(waveform.shape[-1]) < 1:
                raise ValueError(
                    "h3_motion_context: 「音频上下文长度」大于 0，但当前"
                    "上下文%s无音频可用。「端口」模式需连接「上下文音频」，"
                    "或把「音频上下文长度」设为 0。"
                    % ("" if 模式 == "端口" else "文件"))
            if audio_vae is None:
                raise ValueError(
                    "h3_motion_context: 「音频上下文长度」大于 0 时需连接 "
                    "audio_vae（H3 音频 VAE）。")
            if waveform.ndim == 2:
                waveform = waveform[None]
            sr = int(media.get("sample_rate") or 32000)
            avail_frames = int(waveform.shape[-1]) / float(sr) * FPS
            # 音频锚窗上限收窄到画面锚窗 g：音频引导不应覆盖出画面引导之外
            # （否则该区段只有音频、无画面可对应，且下方裁剪量 max(g,a) 会
            # 白白多裁 a-g 帧新画面）
            a = min(a_req, int(avail_frames), g)
            if a < 1:
                raise ValueError(
                    "h3_motion_context: 上下文音频可用时长不足 1 帧，无法固定"
                    "音频引导。")
            z = _tail_audio_latent(audio_vae, waveform, sr, a)
            if track_steps is not None and int(z.shape[-1]) > track_steps:
                z = z[..., :track_steps].clone()
            keyframe["audio_latent"] = z

        # 裁剪量：覆盖画面引导窗与音频窗的较大值——音频窗比画面窗长时只裁
        # 画面窗会让固定音频泄漏进交付部分
        cut = max(g, a)
        if cut >= frame_count:
            raise ValueError(
                "h3_motion_context: 裁剪量 %d 帧达到/超过本片段总长 %d 帧。"
                "请减小「上下文长度」或「音频上下文长度」。"
                % (cut, frame_count))

        # 追加进正向条件化（与官方 Add Guide 相同：读取已有列表、追加、
        # 整表写回，可与官方引导节点自由混用）
        keyframes = list(条件化[0][1].get("minimax_keyframes", []))
        keyframes.append(keyframe)
        out = node_helpers.conditioning_set_values(
            条件化, {"minimax_keyframes": keyframes})
        if a > 0:
            hint += "（含音频）"
        return {"result": (out, "1:%d" % cut), "ui": {
            "h3_hint": hint}}

    @classmethod
    def IS_CHANGED(cls, 模式="自动索引", 存储位置="H3-Mubu", 片段序号=1,
                   手动上传="", **kwargs):
        # IS_CHANGED 只能拿 widget 值、无法感知端口连线：端口模式返回 0、按输入数据
        # 变化正常重跑；上传/自动索引模式按本地文件指纹判定。**kwargs 兼容其余端口值。
        if 模式 == "端口":
            return 0
        return _context_latent_fingerprint(存储位置, 片段序号, 手动上传)


# ============================================================================
# H3 运动裁剪：解码 AV 潜空间为图像+音频，裁头输出并保存尾段媒体
# ============================================================================

class Yuan_H3MotionContextTrim:
    """把 H3 采样器的 AV 潜空间解码为图像+音频后两段式处理。

    1) 按「裁剪帧数」"状态:长度"从头部裁掉被固定窗口覆盖的帧——像素域裁切
       无 VRF 相位约束，任意帧数均可干净裁掉，音频按 24fps→采样率同步换算；
    2) 交付部分尾部再切一段（状态 0 时长度即字符串中的尾段长度），以 uint8 帧
       +float32 波形保存到本地供下一片段加载衔接（由「保存到本地」开关控制）。
    解码→重编码即官方 Add Guide 的条件来源路径，天然重置采样器原始潜空间
    逐链累积的亮度漂移。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "潜空间": ("LATENT", {
                    "tooltip": "H3 采样器的 AV 潜空间（同时含视频流与音频"
                               "流）。本节点将其解码为图像与音频后裁切。"}),
                "裁剪帧数": ("STRING", {
                    "default": "0:22",
                    "forceInput": True,
                    "tooltip": "强制输入端口（不可改），连接「H3 运动上下文」"
                               "的裁剪帧数输出：1:22=启用上下文（头部裁22帧、"
                               "尾段保存22帧）；0:22=未启用/首片段（不裁头、"
                               "尾段仍保存22帧供衔接）。也兼容纯数字（按启用"
                               "语义：裁头与尾段等长）。未连线时用默认 0:22。"}),
                "VAE": ("VAE", {
                    "tooltip": "H3 视频 VAE。把 AV 潜空间的视频流解码为像素"
                               "帧。"}),
                "audio_vae": ("VAE", {
                    "tooltip": "H3 音频 VAE。把 AV 潜空间的音频流解码为波形。"}),
                "片段序号": ("INT", {
                    "default": 1, "min": 1, "max": 9999,
                    "tooltip": "本片段在链条中的序号。设为2保存到 clip_00002.mp4，"
                               "重复生成覆盖原文件；下一片段「H3 运动上下文」"
                               "的「片段序号」设相同值即可对应加载。"}),
                "保存到本地": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "尾段保存总开关。开启按裁剪帧数长度保存尾段"
                               "（1:22→22帧，0:22→22帧，未启用也保存供"
                               "衔接）；关闭仅输出、不生成文件，不影响图像/"
                               "音频端口。"}),
                "存储位置": ("STRING", {
                    "default": "H3-Mubu",
                    "tooltip": "保存在 ComfyUI 输出文件夹下的子目录名。"
                               "下一片段「H3 运动上下文」使用相同的存储位置"
                               "即可对应加载。"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("图像", "音频")
    FUNCTION = "trim"
    OUTPUT_NODE = True
    CATEGORY = "Yuan Tool/MiniMax"
    DESCRIPTION = ("把 H3 采样器的 AV 潜空间解码为视频图像与音频，按"
                   "「H3 运动上下文」输出的裁剪帧数字符串\"状态:长度\"从"
                   "头部裁掉被固定窗口覆盖的帧后输出（已解码，无需再接 VAE "
                   "解码），并把交付部分尾段（uint8 帧 + 音频波形）保存到"
                   "本地，供下一片段衔接。像素域裁切无相位约束，任意帧数"
                   "均可干净裁掉。")

    def trim(self, 潜空间, 裁剪帧数, VAE, audio_vae, 片段序号=1,
             保存到本地=True, 存储位置="H3-Mubu"):
        # 解析裁剪帧数字符串"状态:长度"："1:22"→裁头22帧且尾段存22帧；"0:22"→不裁头、
        # 尾段仍存22帧供衔接；兼容纯数字手填输入（按启用语义：n=tail=该值）。
        s = str(裁剪帧数).strip()
        if ":" in s:
            state_str, _, len_str = s.partition(":")
            try:
                state = int(state_str or "1")
            except ValueError:
                state = 1
            try:
                maxlen = int(len_str or "0")
            except ValueError:
                maxlen = 0
            if state:
                n, tail = maxlen, maxlen
            else:
                n, tail = 0, maxlen
        else:
            try:
                n = int(float(s or 0))
            except ValueError:
                n = 0
            tail = n
        n = max(0, n)
        tail = max(0, tail)
        # 像素域裁切：任意帧数均可，无需吸附整组（无 VRF 相位约束）
        parts = _streams_from_latent(潜空间)
        if len(parts) < 2:
            raise ValueError(
                "h3_motion_context: 裁剪需要含视频和音频两流的 AV 潜空间，"
                "得到 %d 个流。请连接 H3 采样器的输出。"
                % len(parts))
        video, audio = parts[0], parts[1]
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if audio.ndim == 3:
            audio = audio.unsqueeze(0)
        # 解码：视频流 → [T,H,W,3] 像素帧；音频流 → [1,2,L] 波形
        # （VAE 包装器 decode 返回 [B,L,2]，movedim 回 ComfyUI AUDIO
        # 约定的 [B,声道,样本]）
        pixels = VAE.decode(video)
        if pixels.ndim == 5:
            pixels = pixels[0]
        total = int(pixels.shape[0])
        if n >= total:
            raise ValueError(
                "h3_motion_context: asked to trim %d frames from a %d frame "
                "clip" % (n, total))
        sample_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
        wave = audio_vae.decode(audio)
        if wave.ndim == 3:
            wave = wave.movedim(1, -1)  # [1,L,2] → [1,2,L]
        elif wave.ndim == 2:
            wave = wave[None].movedim(1, -1)  # [L,2] → [1,2,L]
        audio_cut = int(round(n / float(FPS) * sample_rate))
        audio_cut = min(audio_cut, int(wave.shape[-1]))
        delivered_pixels = pixels[n:]
        delivered_wave = wave[..., audio_cut:]
        # 第二次裁切（受「保存到本地」开关控制）：在交付部分尾部按解析长度再切后段存盘，
        # 供下一片段「H3 运动上下文」取尾部窗口；交付帧数不足时保存整个交付部分。
        if 保存到本地 and tail > 0:
            t_frames = min(tail, total - n)
            tail_pixels = delivered_pixels[delivered_pixels.shape[0] - t_frames:]
            tail_samples = int(round(t_frames / float(FPS) * sample_rate))
            tail_samples = min(tail_samples, int(delivered_wave.shape[-1]))
            tail_wave = (delivered_wave[..., delivered_wave.shape[-1] - tail_samples:]
                         if tail_samples > 0 else delivered_wave[..., :0])
            _save_av_media(tail_pixels, tail_wave, sample_rate,
                           存储位置, 片段序号)
        return (delivered_pixels,
                {"waveform": delivered_wave, "sample_rate": sample_rate})


def _save_av_media(pixels, waveform, sample_rate, 存储位置, 片段序号):
    """把尾段媒体（像素帧+音频波形）混流存为 {输出}/{存储位置}/clip_%05d.mp4。

    视频用 libx264 无损模式（yuv444p + crf 0）作为引导中间格式：画面是下一片段
    VAE 重编码的引导源，任何有损伪影都会随链条逐段累积（画面污染）；无损保存把
    该累积归零。音频仍 aac。重生成同一片段覆盖自身、不堆叠；文件名与「H3 运动
    上下文」自动索引加载约定一致（读取端对旧 yuv420 文件同样可解码）。
    """
    import av
    # pixels 约定 [T,H,W,3]；仅当仍带 [B,...] 批维（5D）时才塌缩首帧轴
    if pixels.ndim == 5:
        pixels = pixels[0]
    if waveform is not None and waveform.ndim == 2:
        waveform = waveform[None]
    has_audio = waveform is not None and int(waveform.shape[-1]) > 0
    loc = (存储位置 or "").strip().strip('"').strip("'") or "H3-Mubu"
    clip_dir = os.path.join(folder_paths.get_output_directory(), loc)
    os.makedirs(clip_dir, exist_ok=True)
    path = os.path.join(clip_dir, "%s_%05d%s" % (CLIP_FILE_PREFIX,
                                                 int(片段序号),
                                                 CLIP_FILE_EXT))
    # 原子写：先写同目录临时文件，编码成功后再 os.replace 覆盖目标——
    # 读取方永远不会读到半截文件；失败时旧文件保留
    tmp_path = "%s.tmp%d" % (path, os.getpid())
    try:
        os.remove(tmp_path)  # 清理上一次异常遗留的临时文件
    except OSError:
        pass
    h, w = int(pixels.shape[1]), int(pixels.shape[2])
    video_u8 = (pixels.clamp(0.0, 1.0).mul(255.0).round_()
                .to(torch.uint8).cpu().numpy())
    # 写入模式靠文件扩展名推断封装格式；tmp 后缀 .tmp<pid> 无法识别，
    # 必须显式指定 format="mp4"
    container = av.open(tmp_path, mode="w", format="mp4")
    try:
        # 两个流须先于任何写包建好（av 18 在已有时间戳后再加流会报
        # "Cannot rebase to zero time"）；先建后按序编码并无冲突
        vstream = container.add_stream("libx264", rate=FPS)
        vstream.width = w
        vstream.height = h
        # 无损引导中间格式：h264 无损仅支持 4:4:4（yuv444p）+ qp 0。
        # crf 16/yuv420p 的量化与色度抽样伪影会随多段再生逐段累积成画面污染，
        # 故引导源用 crf 0 无损（文件较大，但仅尾段短窗、且不对外分发）
        vstream.pix_fmt = "yuv444p"
        vstream.options = {"crf": "0", "preset": "slow"}
        astream = None
        if has_audio:
            astream = container.add_stream("aac", rate=int(sample_rate))
            astream.layout = "stereo"
        for i in range(video_u8.shape[0]):
            frame = av.VideoFrame.from_ndarray(
                np.ascontiguousarray(video_u8[i]), format="rgb24")
            for pkt in vstream.encode(frame):
                container.mux(pkt)
        for pkt in vstream.encode():
            container.mux(pkt)
        if astream is not None:
            wave = waveform[0]
            if wave.shape[0] == 1:
                wave = wave.repeat(2, 1)
            elif wave.shape[0] > 2:
                wave = wave[:2]
            wave = wave.to(torch.float32).cpu()
            total = int(wave.shape[-1])
            chunk = int(sample_rate) // 10  # 0.1s 一块，控内存
            n = 0
            while n < total:
                seg = wave[:, n:n + chunk]
                aframe = av.AudioFrame(format="fltp", layout="stereo",
                                       samples=int(seg.shape[-1]))
                aframe.sample_rate = int(sample_rate)
                for ch in range(2):
                    aframe.planes[ch].update(
                        np.ascontiguousarray(seg[ch].numpy()))
                for pkt in astream.encode(aframe):
                    container.mux(pkt)
                n += chunk
            for pkt in astream.encode():
                container.mux(pkt)
    finally:
        container.close()
    # 封装全部成功且容器已关闭落盘后，才将临时文件原子替换为正式文件；
    # 中途任何异常都会在上方上抛（跳过此行），旧文件保持完整可用
    os.replace(tmp_path, path)


# ============================================================================
# 存储位置/文件名解析（约定：存储位置目录下 clip_%05d.mp4）
# ============================================================================

def _clip_file_path(存储位置, idx):
    """返回 clip_%05d.mp4 绝对路径（不存在抛 FileNotFoundError，由调用方兜底扫描 output 目录）。"""
    loc = (存储位置 or "").strip().strip('"').strip("'") or "H3-Mubu"
    path = os.path.join(folder_paths.get_output_directory(), loc,
                        "%s_%05d%s" % (CLIP_FILE_PREFIX, int(idx),
                                       CLIP_FILE_EXT))
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "h3_motion_context: no clip media for index %d (%s)."
            % (int(idx), path))
    return path


def _find_clip_in_output(idx):
    """兜底搜索：主「存储位置」解析失败时（工作流参数可能错位），扫描 output 目录
    全部子目录找 clip_%05d.mp4，命中多个取 mtime 最新的。"""
    try:
        idx_i = int(idx)
    except (TypeError, ValueError):
        return None
    target = "%s_%05d%s" % (CLIP_FILE_PREFIX, idx_i, CLIP_FILE_EXT)
    out = folder_paths.get_output_directory()
    if not out or not os.path.isdir(out):
        return None
    try:
        subs = sorted(os.listdir(out))
    except OSError:
        return None
    for sub in subs:
        subp = os.path.join(out, sub)
        if not os.path.isdir(subp):
            continue
        try:
            files = [os.path.join(subp, f) for f in os.listdir(subp)
                     if f == target]
        except OSError:
            continue
        if files:
            return max(files, key=os.path.getmtime)
    return None


def _build_load_path(存储位置):
    """存储位置参数 → 目录级指纹前缀（用户只设目录名，clip 前缀内部固定、不可改）。"""
    loc = (存储位置 or "").strip().strip('"').strip("'") or "H3-Mubu"
    return os.path.join(loc, CLIP_FILE_PREFIX)


def _dir_fingerprint(prefix_path):
    """目录级综合指纹：对 prefix_path 所在目录下所有 clip_*.mp4/.safetensors 按
    「文件名+mtime+size」哈希（仅读元数据），任一文件增/删/改都改变指纹。

    IS_CHANGED 专用：链接输入拿不到真实片段序号、无法定位单文件，故对整目录做指纹
    ——保存节点每次覆盖写同一路径（mtime 必变）→ 指纹变 → 下游重跑；同内容重试→命中。
    目录不存在返回确定性 "missing"（可缓存），首次保存出现文件后自然触发重跑。
    """
    p = (prefix_path or "").strip().strip('"').strip("'")
    if not p:
        p = "H3-Mubu/clip"
    h = hashlib.sha256()
    h.update(p.encode("utf-8"))
    # 候选目录顺序与 _clip_file_path 一致：先 output 目录下的原路径
    for c in (os.path.join(folder_paths.get_output_directory(), p), p):
        dir_part = os.path.dirname(c)
        prefix = os.path.basename(c)
        if dir_part and prefix and os.path.isdir(dir_part):
            files = sorted(f for f in os.listdir(dir_part)
                           if f.startswith(prefix)
                           and (f.endswith(CLIP_FILE_EXT)
                                or f.endswith(".safetensors")))
            for fname in files:
                h.update(fname.encode("utf-8"))
                try:
                    st = os.stat(os.path.join(dir_part, fname))
                    h.update(("%d:%d;" % (st.st_mtime_ns, st.st_size))
                             .encode("utf-8"))
                except OSError:
                    pass
            return "%s:%s" % (p, h.hexdigest())
    return "missing:%s" % p


# ============================================================================
# 手动上传上下文媒体：分块上传 .mp4（兼容旧版 .safetensors）到
# input/h3_motion_latent/
# ============================================================================

_MANUAL_UPLOAD_SUBDIR = "h3_motion_latent"


@PromptServer.instance.routes.post("/yuan_h3_motion_upload_latent")
async def _yuan_h3_motion_upload_latent(request):
    """接收「H3 运动上下文」节点手动上传的媒体文件（分块追加写入）。"""

    def _normalize(name):
        return os.path.basename(name)

    def _validate(file_path):
        upload_dir = os.path.join(folder_paths.get_input_directory(),
                                  _MANUAL_UPLOAD_SUBDIR)
        if not os.path.basename(file_path).lower().endswith(
                (".mp4", ".safetensors")):
            return web.json_response(
                {"error": "仅支持 .mp4 上下文媒体文件（兼容旧版 .safetensors）"},
                status=400)
        if not os.path.realpath(file_path).startswith(os.path.realpath(upload_dir)):
            return web.json_response({"error": "无效的文件名"}, status=400)
        return None

    def _response_name(stored):
        return "%s/%s" % (_MANUAL_UPLOAD_SUBDIR, stored)

    upload_dir = os.path.join(folder_paths.get_input_directory(),
                              _MANUAL_UPLOAD_SUBDIR)
    return await handle_chunk_upload(request, upload_dir,
                                     normalize_name=_normalize,
                                     validate=_validate,
                                     response_name=_response_name)


def _resolve_manual_media_path(手动上传):
    """解析「手动上传」媒体路径为绝对路径（空输入返回 None）。

    支持 input:/output:/temp: 前缀与绝对路径；无前缀时依次在 input、output 目录查找。
    """
    p = (手动上传 or "").strip().strip('"').strip("'")
    if not p:
        return None
    candidates = []
    matched = False
    for prefix, base in (
        ("input:", folder_paths.get_input_directory()),
        ("output:", folder_paths.get_output_directory()),
        ("temp:", folder_paths.get_temp_directory()),
    ):
        if p.startswith(prefix):
            candidates.append(os.path.join(base, p[len(prefix):].lstrip("/\\")))
            matched = True
            break
    if not matched:
        if os.path.isabs(p):
            candidates.append(p)
        else:
            candidates.append(os.path.join(
                folder_paths.get_input_directory(), p))
            candidates.append(os.path.join(
                folder_paths.get_output_directory(), p))
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "h3_motion_context: 手动上传的媒体文件未找到：%r"
        "（支持 input:/output:/temp: 前缀、绝对路径；无前缀时在 input 与"
        " output 目录下查找）。" % p)


NODE_CLASS_MAPPINGS = {
    "Yuan_H3MotionContext": Yuan_H3MotionContext,
    "Yuan_H3MotionContextTrim": Yuan_H3MotionContextTrim,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Yuan_H3MotionContext": "H3 运动上下文",
    "Yuan_H3MotionContextTrim": "H3 运动裁剪",
}

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""extract_ocr.py -- LAB-2125 / module A of bili-story-pipe.

Video -> bottom-dialog OCR -> de-duplicated finalized dialogue lines.

Pipeline stages (all runnable without a GPU / model via the ``mock`` backend):

  1. ffmpeg decode with CUDA auto-detection and CPU fallback (fps=1 default).
  2. Automatic bottom-dialog ROI detection from a lower-half text-edge heatmap
     (the centre machine-translation overlay is ignored by construction).
  3. Typewriter "finalized frame" detection: a run of >=2 identical ROI
     fingerprints after a content change is the settled frame; transition and
     black frames are dropped and the precise ``ts_sec`` is recorded.
  4. Batch OCR (PaddleOCR-GPU / RapidOCR-ONNX / deterministic mock), speaker
     and text separation, near-duplicate de-duplication, low-confidence
     review spilling.

Outputs under ``<work_dir>/output/``: ``script.jsonl`` and ``extract_stats.json``,
matching the contract shared with module B (LAB-2127).
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image

LOW_CONF_THRESHOLD = 0.85
DEFAULT_FPS = 1.0
DEFAULT_ROI_SAMPLES = 30
# LAB-2129: jm 实测 0.65 会蹭到中部 LunaTranslator 机翻悬浮窗，收紧到仅扫底部 28%。
DEFAULT_BOTTOM_FRACTION = 0.72
DEFAULT_BATCH_SIZE = 32
DEFAULT_PART_WORKERS = 1

# LAB-2133: jm(aarch64 + 2×A100) 全 9P 实测固化的 3 项优化参数。
# 单帧 RapidOCR 调用仅 2.1 fps（Python/det 固定开销），16 帧竖拼后单卡实测 69.7 fps。
STRIP16_SLOTS = 16
STRIP16_SEPARATOR_PX = 12
# detect_roi 在 bottom_fraction=0.72 时曾框出 h=50 截掉下方 2 行正文，强制延伸到画面底部。
ROI_BOTTOM_MARGIN_FRACTION = 0.02
# 抽帧直接输出轻量 JPEG（替代整帧 640×360 无压缩 PNG），解码+落盘 490s -> 目标 <60s。
DEFAULT_FRAME_EXT = "jpg"
JPEG_QUALITY_Q = 3

# 机翻插件状态栏残留特征串（含常见 OCR 误识变体与半角冒号写法）。
OVERLAY_NOISE_TOKENS = (
    "翻译结束",
    "翻译结果",
    "耗时：",
    "耗时:",
    "利时：",
    "利时:",
)


def is_overlay_noise(text: str) -> bool:
    """True when ``text`` is a machine-translation status-bar leftover.

    jm 实测中，LunaTranslator 的「翻译结束 / 翻译结果 / 耗时：xx ms」状态栏
    偶尔会掉进底部 ROI 的 OCR 结果，必须作为后过滤丢弃，避免混入台词。
    """
    if not text:
        return False
    s = str(text)
    return any(tok in s for tok in OVERLAY_NOISE_TOKENS)


# --------------------------------------------------------------------------- #
# ffmpeg / hardware acceleration
# --------------------------------------------------------------------------- #
def find_ffmpeg() -> str:
    """Locate an ffmpeg binary: PATH, then the imageio-ffmpeg static build."""
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("未找到 ffmpeg：安装系统 ffmpeg 或 pip install imageio-ffmpeg") from exc


def _gpu_present() -> bool:
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and "GPU" in r.stdout:
            return True
    except Exception:
        pass
    return os.path.exists("/dev/nvidia0")


def probe_hwaccel(ffmpeg: str | None = None, force_cpu: bool = False) -> list[str]:
    """Return ffmpeg hwaccel flags, empty list when CPU-only.

    CUDA is used only when an actual NVIDIA device is visible AND this ffmpeg
    build was compiled with the cuda hwaccel; otherwise degrade to CPU so the
    same code runs in the jm 2xA100 box and in the sandbox unit tests.
    """
    if force_cpu or os.environ.get("BILI_STORY_FORCE_CPU"):
        return []
    if not _gpu_present():
        return []
    ffmpeg = ffmpeg or find_ffmpeg()
    try:
        out = subprocess.run(
            [ffmpeg, "-hide_banner", "-hwaccels"], capture_output=True, text=True, timeout=10
        )
        if re.search(r"^\s*cuda\s*$", out.stdout, re.MULTILINE):
            return ["-hwaccel", "cuda"]
    except Exception:
        pass
    return []


@dataclass
class Frame:
    index: int
    ts_sec: float
    path: str
    image: Image.Image


def build_decode_cmd(
    ffmpeg: str,
    video_path: str | Path,
    fps: float,
    out_dir: str | Path,
    roi: tuple[int, int, int, int] | None = None,
    ext: str = DEFAULT_FRAME_EXT,
    hwaccel: list[str] | None = None,
    max_frames: int | None = None,
) -> list[str]:
    """Build the ffmpeg command for resampled frame extraction.

    LAB-2133: when ``roi`` is known the crop filter is pushed into ffmpeg
    (``-vf fps=X,crop=w:h:x:y``) so only the small dialogue strip is decoded
    and written. Frames are emitted as lightweight JPEGs with ``-q:v 3``
    instead of uncompressed full-frame PNGs -- this cut decode+disk time from
    490s to <60s on the full 12.2h / 9P video.
    """
    vf = f"fps={fps}"
    if roi:
        x, y, w, h = (int(v) for v in roi)
        if w > 0 and h > 0:
            vf += f",crop={w}:{h}:{x}:{y}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = (
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
        + list(hwaccel or [])
        + ["-i", str(video_path), "-vf", vf]
    )
    if max_frames and max_frames > 0:
        cmd += ["-frames:v", str(int(max_frames))]
    if ext in ("jpg", "jpeg"):
        cmd += ["-q:v", str(JPEG_QUALITY_Q)]
    cmd += [str(out_dir / f"f_%06d.{ext}")]
    return cmd


def decode_frames(
    video_path: str | Path,
    fps: float = DEFAULT_FPS,
    hwaccel: list[str] | None = None,
    cache_dir: str | Path | None = None,
    roi: tuple[int, int, int, int] | None = None,
    ext: str = DEFAULT_FRAME_EXT,
    max_frames: int | None = None,
) -> list[Frame]:
    """Extract resampled frames paired with their timestamps.

    LAB-2133: outputs JPEG by default (``-q:v 3``), and when ``roi`` is known
    the ``crop`` filter is pushed into ffmpeg so only the dialogue strip is
    written. Backward compatible: omitting ``roi`` yields full frames.
    """
    ffmpeg = find_ffmpeg()
    tmp = Path(cache_dir) if cache_dir else Path(tempfile.mkdtemp(prefix="bili_decode_"))
    tmp.mkdir(parents=True, exist_ok=True)
    cmd = build_decode_cmd(ffmpeg, video_path, fps, tmp, roi=roi, ext=ext, hwaccel=hwaccel,
                           max_frames=max_frames)
    subprocess.run(cmd, check=True, capture_output=True, timeout=3600)
    frames: list[Frame] = []
    for i, path in enumerate(sorted(tmp.glob(f"f_*.{ext}"))):
        frames.append(
            Frame(
                index=i,
                ts_sec=round(i / fps, 3),
                path=str(path),
                image=Image.open(path).convert("RGB"),
            )
        )
    return frames


def decode_frames_with_roi(
    video_path: str | Path,
    fps: float,
    hwaccel: list[str] | None,
    cache_dir: str | Path,
    roi: tuple[int, int, int, int] | None = None,
    sample_count: int = DEFAULT_ROI_SAMPLES,
) -> tuple[list[Frame], tuple[int, int, int, int]]:
    """Decode frames using the fastest path for the given ROI situation.

    - ROI known (``--roi`` or auto-detected earlier): decode once with the
      ``crop`` filter baked into ffmpeg, emitting JPEGs.
    - ROI unknown: first extract only ``sample_count`` (default 30) JPEG probe
      frames, detect the bottom-dialog ROI from them, then re-decode the whole
      video with ``crop``. Both passes use JPEG, never PNG.
    """
    if roi:
        return decode_frames(video_path, fps, hwaccel, cache_dir, roi=roi), tuple(int(v) for v in roi)
    probe_dir = Path(cache_dir) / "roi_probe"
    probe = decode_frames(video_path, fps, hwaccel, probe_dir, max_frames=sample_count)
    detected = detect_roi([f.image for f in probe])
    frames = decode_frames(video_path, fps, hwaccel, cache_dir, roi=detected)
    return frames, detected


# --------------------------------------------------------------------------- #
# automatic bottom-dialog ROI detection
# --------------------------------------------------------------------------- #
def extend_roi_to_bottom(
    roi: tuple[int, int, int, int],
    height: int,
    margin_fraction: float = ROI_BOTTOM_MARGIN_FRACTION,
) -> tuple[int, int, int, int]:
    """Force a detected ROI to cover the bottom text area.

    LAB-2133: jm 实测 ``bottom_fraction=0.72`` 时边缘热力图只框到角色名栏
    (``h=50``)，截掉下方 2 行正文。给定检测到的对话框顶部 ``y``，把下沿强制
    延伸到画面底部（预留 2% 边距），避免漏字；结果始终被裁剪到画面内。
    """
    x, y, w, h = (int(v) for v in roi)
    target = int(height) - y - int(int(height) * margin_fraction)
    new_h = max(h, target)
    new_h = min(new_h, max(0, int(height) - y))
    return (x, y, w, new_h)


def _largest_run(mask: np.ndarray) -> tuple[int, int] | None:
    idx = np.where(mask)[0]
    if idx.size == 0:
        return None
    runs: list[tuple[int, int]] = []
    start = prev = int(idx[0])
    for v in idx[1:]:
        v = int(v)
        if v == prev + 1:
            prev = v
        else:
            runs.append((start, prev))
            start = prev = v
    runs.append((start, prev))
    return max(runs, key=lambda r: r[1] - r[0])


def _sample(images: Iterable[Image.Image], n: int) -> list[Image.Image]:
    images = list(images)
    if not images:
        return []
    if len(images) <= n:
        return images
    idx = np.linspace(0, len(images) - 1, n).astype(int)
    return [images[i] for i in idx]


def detect_roi(
    images: Iterable[Image.Image],
    bottom_fraction: float = DEFAULT_BOTTOM_FRACTION,
    sample_count: int = DEFAULT_ROI_SAMPLES,
) -> tuple[int, int, int, int]:
    """Return (x, y, w, h) of the bottom dialogue box.

    Only the lower part of the frame (y >= bottom_fraction*H) is examined, so
    a centred LunaTranslator machine-translation overlay can never be picked
    up. Text edges accumulated over ~30 frames form a heatmap; the densest
    contiguous row/column band bounds the box.
    """
    sampled = _sample(images, sample_count)
    if not sampled:
        return (0, 0, 0, 0)
    heat: np.ndarray | None = None
    height = width = 0
    y0 = 0
    for im in sampled:
        arr = np.asarray(im.convert("L"), dtype=np.float32)
        height, width = arr.shape
        y0 = int(height * bottom_fraction)
        lower = arr[y0:, :]
        gx = np.abs(np.diff(lower, axis=1, prepend=lower[:, :1]))
        gy = np.abs(np.diff(lower, axis=0, prepend=lower[:1, :]))
        edges = np.minimum(gx + gy, 255.0)
        heat = edges if heat is None else heat + edges
    assert heat is not None

    row = heat.mean(axis=1)
    col = heat.mean(axis=0)
    rband = _largest_run(row >= max(row.max() * 0.18, 1.0))
    cband = _largest_run(col >= max(col.max() * 0.18, 1.0))
    if rband is None or cband is None:
        # conservative fallback: bottom centre, extended to the bottom edge (LAB-2133)
        return extend_roi_to_bottom(
            (int(width * 0.10), int(height * 0.72), int(width * 0.80), int(height * 0.24)), height
        )

    ry0, ry1 = rband
    cx0, cx1 = cband
    mx, my = int(width * 0.02), int(height * 0.02)
    x = max(0, cx0 - mx)
    y = max(0, y0 + ry0 - my)
    w = min(width - x, (cx1 - cx0 + 1) + 2 * mx)
    h = min(height - y, (ry1 - ry0 + 1) + 2 * my)
    # guard against a too-small / degenerate box
    if w < width * 0.15 or h < height * 0.05:
        return extend_roi_to_bottom(
            (int(width * 0.10), int(height * 0.72), int(width * 0.80), int(height * 0.24)), height
        )
    # LAB-2133: 下沿强制覆盖底部正文，杜绝 h=50 截断。
    return extend_roi_to_bottom((x, y, w, h), height)


# --------------------------------------------------------------------------- #
# typewriter finalized-frame detection
# --------------------------------------------------------------------------- #
def roi_fingerprint(image: Image.Image, roi: tuple[int, int, int, int]) -> np.ndarray | None:
    """Grayscale perceptual fingerprint of the ROI; None means black/blank.

    Kept near quarter resolution so that a few added typewriter strokes still
    move the fingerprint well above the change threshold, while staying cheap.
    """
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        return None
    fw = max(32, min(160, w // 4))
    fh = max(8, min(48, h // 4))
    crop = image.crop((x, y, x + w, y + h)).convert("L").resize((fw, fh))
    arr = np.asarray(crop, dtype=np.float32)
    if arr.mean() < 14.0 or arr.std() < 6.0:  # black screen or flat filler
        return None
    return arr


def _fp_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())


@dataclass
class FinalizedFrame:
    frame_index: int
    ts_sec: float
    crop: Image.Image
    path: str


def detect_finalized_frames(
    frames: list[Frame],
    roi: tuple[int, int, int, int],
    diff_threshold: float = 0.5,
    min_gap_sec: float = 0.8,
) -> list[FinalizedFrame]:
    """Capture a line once it stops changing (typewriter settled).

    A frame qualifies when (a) its ROI fingerprint differs from the previous
    frame -- a content change just happened -- and (b) the next frame has the
    same fingerprint, i.e. the content is stable for >=1 extra frame. Black
    frames are ignored on both sides.
    """
    fps_estimate = frames[1].ts_sec - frames[0].ts_sec if len(frames) > 1 else 1.0
    fps_estimate = fps_estimate or 1.0
    fps = 1.0 / fps_estimate
    fingerprints = [roi_fingerprint(f.image, roi) for f in frames]
    out: list[FinalizedFrame] = []
    n = len(frames)
    for i, frame in enumerate(frames):
        cur = fingerprints[i]
        if cur is None:
            continue
        prev = fingerprints[i - 1] if i > 0 else None
        changed = i == 0 or prev is None or _fp_dist(cur, prev) > diff_threshold
        if not changed:
            continue
        nxt = fingerprints[i + 1] if i + 1 < n else None
        stable = i == n - 1 or nxt is None or _fp_dist(cur, nxt) <= diff_threshold
        if not stable:
            continue
        if out and (frame.ts_sec - out[-1].ts_sec) < max(min_gap_sec, 0.9 / fps):
            continue
        x, y, w, h = roi
        out.append(
            FinalizedFrame(
                frame_index=frame.index,
                ts_sec=frame.ts_sec,
                crop=frame.image.crop((x, y, x + w, y + h)).convert("RGB"),
                path=frame.path,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# OCR backends
# --------------------------------------------------------------------------- #
class BaseOCRBackend:
    name = "base"

    def recognize_batch(self, images: list[Image.Image], ts_secs: list[float], batch_size: int) -> list[list[dict]]:
        results: list[list[dict]] = []
        for image, ts in zip(images, ts_secs):
            results.append(self.recognize(image, ts))
        return results

    def recognize(self, image: Image.Image, ts_sec: float) -> list[dict]:  # pragma: no cover
        raise NotImplementedError


class MockOCRBackend(BaseOCRBackend):
    """Deterministic backend for GPU-free unit tests.

    Wire protocol: ``mock_lines`` is a JSON list of
    ``{"ts_start", "ts_end", "speaker", "text_ja", "conf"}``.  A finalized
    frame is "read" as the line whose timestamp window contains ``ts_sec``.
    Empty result -> the frame's timestamp fell outside every window, which is
    exactly how the tests catch finalized-frame drift. The CLI also accepts
    plain ``--mock-lines <path>`` pointing at this JSON.
    """

    name = "mock"

    def __init__(self, mock_lines: list[dict] | None = None):
        self.lines = mock_lines or []

    def recognize(self, image: Image.Image, ts_sec: float) -> list[dict]:
        for entry in self.lines:
            if float(entry["ts_start"]) <= ts_sec <= float(entry["ts_end"]):
                return [
                    {
                        "text": str(entry.get("text_ja", "")),
                        "conf": float(entry.get("conf", 0.95)),
                        "speaker": str(entry.get("speaker", "")),
                    }
                ]
        return []


def _paddleocr_major_version(module) -> int:
    """Read the installed PaddleOCR major version (0 when unavailable)."""
    raw = getattr(module, "__version__", "0")
    m = re.match(r"\s*(\d+)", str(raw))
    return int(m.group(1)) if m else 0


def _parse_paddle_v3_result(result) -> list[dict]:
    """Parse a PaddleOCR 3.x ``predict`` result.

    3.x returns ``OCRResult`` objects exposing ``rec_texts`` / ``rec_scores`` /
    ``dt_polys`` (either via ``.json`` mapping or as direct attributes), not the
    nested ``[[box, (text, conf)], ...]`` shape of 2.x ``ocr()``.
    """
    data = None
    if isinstance(result, dict):
        data = result
    elif hasattr(result, "json"):
        try:
            data = result.json
        except Exception:
            data = None
    if not isinstance(data, dict):
        data = {
            key: getattr(result, key, None)
            for key in ("rec_texts", "rec_scores", "dt_polys", "rec_polys")
        }

    texts = data.get("rec_texts") or []
    scores = data.get("rec_scores") or []
    polys = data.get("dt_polys") or data.get("rec_polys") or []
    lines: list[dict] = []
    for i, text in enumerate(texts):
        if not str(text).strip():
            continue
        try:
            conf = float(scores[i]) if i < len(scores) else 1.0
        except (TypeError, ValueError):
            conf = 1.0
        box = polys[i] if i < len(polys) else None
        lines.append({"text": str(text), "conf": conf, "box": box})
    return lines


class PaddleOCRBackend(BaseOCRBackend):
    name = "paddle"

    def __init__(self, use_gpu: bool = True):
        import paddleocr  # type: ignore

        self.major = _paddleocr_major_version(paddleocr)
        PaddleOCR = paddleocr.PaddleOCR
        if self.major >= 3:
            # LAB-2129: 3.x 用 predict() + device；关闭文档方向/畸变/文本行方向
            # 三个额外模型，避免 jm 上的多余下载与误检。
            self.engine = PaddleOCR(
                lang="japan",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device="gpu:0" if use_gpu else "cpu",
            )
        else:
            # 2.x 保留原有契约。
            self.engine = PaddleOCR(lang="japan", use_gpu=use_gpu)

    def _to_bgr_list(self, images) -> list[np.ndarray]:
        # PaddleOCR 3.x predict 接受 BGR 图像列表；PIL 读入的是 RGB。
        return [np.asarray(im.convert("RGB"))[:, :, ::-1] for im in images]

    def recognize_batch(self, images, ts_secs, batch_size):
        if self.major >= 3:
            results = self.engine.predict(self._to_bgr_list(images))
            # predict 可能返回单个结果对象（单图）或结果列表。
            if not isinstance(results, (list, tuple)):
                results = [results]
            return [_parse_paddle_v3_result(r) for r in results]
        raw = self.engine.ocr([np.asarray(im) for im in images], cls=True, batch_size=batch_size)
        return [_parse_paddle_page(page) for page in raw]

    def recognize(self, image, ts_sec):
        if self.major >= 3:
            results = self.engine.predict(self._to_bgr_list([image]))
            if isinstance(results, (list, tuple)):
                result = results[0] if results else None
            else:
                result = results
            return _parse_paddle_v3_result(result)
        return _parse_paddle_page(self.engine.ocr(np.asarray(image), cls=True))


# --------------------------------------------------------------------------- #
# LAB-2133: RapidOCR PyTorch-CUDA + Strip16 vertical-stitch batching
# --------------------------------------------------------------------------- #
def _box_ycenter(box) -> float | None:
    """Center-y of a polygon or ``[x1,y1,x2,y2]`` bbox; None when absent."""
    if box is None:
        return None
    try:
        pts = np.asarray(box, dtype=float).reshape(-1, 2)
        if pts.size == 0:
            return None
        return float(pts[:, 1].mean())
    except Exception:
        return None


def _shift_box_y(box, dy: float):
    """Shift a bbox vertically by ``dy`` (keeps list/np.ndarray container type)."""
    if box is None:
        return None
    try:
        arr = np.asarray(box)
        pts = arr.astype(float).reshape(-1, 2)
    except Exception:
        return box
    pts[:, 1] += dy
    if isinstance(box, np.ndarray):
        return pts.reshape(arr.shape)
    return pts.tolist()


def build_strip16(
    images: list[Image.Image], sep: int = STRIP16_SEPARATOR_PX
) -> tuple[Image.Image | None, dict]:
    """Vertically stack frames with a black separator for one OCR call.

    Returns ``(strip, meta)``; ``meta['y_offsets'][i]`` is the top y of frame i
    inside the strip, used by :func:`map_strip16_results` to reverse-map boxes.
    """
    if not images:
        return None, {"slot_h": 0, "sep": sep, "y_offsets": [], "width": 0}
    width = max(int(im.width) for im in images)
    slot_h = max(int(im.height) for im in images)
    y_offsets = [i * (slot_h + sep) for i in range(len(images))]
    strip_h = len(images) * slot_h + (len(images) - 1) * sep
    strip = Image.new("RGB", (width, strip_h), (0, 0, 0))
    for im, y in zip(images, y_offsets):
        strip.paste(im.convert("RGB"), (0, y))
    meta = {"slot_h": slot_h, "sep": sep, "y_offsets": y_offsets, "width": width}
    return strip, meta


def strip_slot_index(y_center: float, slot_h: int, sep: int = STRIP16_SEPARATOR_PX) -> int:
    """Reverse-map a strip y-coordinate to its source frame index."""
    return int(y_center // (slot_h + sep))


def map_strip16_results(raw_lines: list[dict], meta: dict) -> dict[int, list[dict]]:
    """Split strip OCR lines back into per-frame lists, sorted top-to-bottom.

    Uses ``slot_idx = int(yc // (h + 12))`` for attribution and shifts each box
    back into its frame-local coordinate system (LAB-2133 measured 69.7 fps).
    """
    y_offsets = meta.get("y_offsets", [])
    slots: dict[int, list[dict]] = {i: [] for i in range(len(y_offsets))}
    slot_h = int(meta.get("slot_h", 0) or 0)
    sep = int(meta.get("sep", STRIP16_SEPARATOR_PX))
    for line in raw_lines or []:
        yc = _box_ycenter(line.get("box"))
        if yc is None or slot_h <= 0:
            continue
        idx = strip_slot_index(yc, slot_h, sep)
        if idx not in slots:
            continue
        item = dict(line)
        item["box"] = _shift_box_y(line.get("box"), -y_offsets[idx])
        item["_local_y"] = yc - y_offsets[idx]
        slots[idx].append(item)
    for lines in slots.values():
        lines.sort(key=lambda it: it.get("_local_y", 0.0))
        for it in lines:
            it.pop("_local_y", None)
    return slots


class RapidOCRBackend(BaseOCRBackend):
    """LAB-2133: RapidOCR v3 (PyTorch-CUDA) with Strip16 batching.

    Prefers ``rapidocr`` v3 configured with ``EngineType.TORCH`` + PP-OCRv4 +
    ``LangRec.JAPAN`` and ``EngineConfig.torch.use_cuda = use_gpu``; falls back
    to ``rapidocr_onnxruntime`` when the v3 package is unavailable.
    """

    name = "rapid"

    def __init__(self, use_gpu: bool = True, strip_size: int = STRIP16_SLOTS):
        self.strip_size = max(1, min(int(strip_size), STRIP16_SLOTS))
        self.v3 = False
        self._parse = _parse_rapid_page
        try:
            from rapidocr import EngineConfig, EngineType, LangRec, RapidOCR  # type: ignore

            cfg = EngineConfig()
            try:
                cfg.torch.use_cuda = bool(use_gpu)
            except Exception:  # pragma: no cover - variant config object
                try:
                    cfg.use_cuda = bool(use_gpu)
                except Exception:
                    pass
            kwargs: dict = {"engine_config": cfg}
            torch = getattr(EngineType, "TORCH", None)
            if torch is not None:
                kwargs["det_engine_type"] = torch
                kwargs["rec_engine_type"] = torch
            kwargs["det_model_name"] = "ch_PP-OCRv4_det"
            kwargs["rec_model_name"] = getattr(LangRec, "JAPAN", "japan")
            self.engine = RapidOCR(**kwargs)
            self.v3 = True
            self._parse = _parse_rapid_v3_result
        except Exception:
            from rapidocr_onnxruntime import RapidOCR as _RapidOCR  # type: ignore

            self.engine = _RapidOCR()
            self.v3 = False
            self._parse = _parse_rapid_page

    def _call_engine(self, image: Image.Image) -> list[dict]:
        raw = self.engine(np.asarray(image.convert("RGB")))
        if isinstance(raw, tuple):  # onnxruntime backend returns (result, elapse)
            raw = raw[0]
        return self._parse(raw)

    def recognize_batch(self, images, ts_secs, batch_size):
        if not images:
            return []
        results: list[list[dict]] = []
        for start in range(0, len(images), self.strip_size):
            chunk = images[start : start + self.strip_size]
            strip, meta = build_strip16(chunk)
            slots = map_strip16_results(self._call_engine(strip), meta)
            for i in range(len(chunk)):
                results.append(slots.get(i, []))
        return results

    def recognize(self, image, ts_sec):
        return self._call_engine(image)


def _parse_paddle_page(page) -> list[dict]:
    lines: list[dict] = []
    if not page:
        return lines
    try:
        items = page[0] if isinstance(page, (list, tuple)) and page and isinstance(page[0], list) else page
        for item in items or []:
            try:
                text, conf = item[1]
            except Exception:
                continue
            lines.append({"text": str(text), "conf": float(conf), "box": item[0] if len(item) > 0 else None})
    except Exception:
        pass
    return lines


def _parse_rapid_page(result) -> list[dict]:
    lines: list[dict] = []
    for item in result or []:
        try:
            lines.append({"text": str(item[1]), "conf": float(item[2]), "box": item[0]})
        except Exception:
            continue
    return lines


def _parse_rapid_v3_result(result) -> list[dict]:
    """Parse a rapidocr v3 result object (``.boxes/.txts/.scores``)."""
    if result is None:
        return []
    boxes = getattr(result, "boxes", None)
    txts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)
    if txts is None and isinstance(result, dict):
        boxes = result.get("boxes")
        txts = result.get("txts")
        scores = result.get("scores")
    if txts is None:
        # tolerate the v2-style (list, elapse) tuple shape
        if isinstance(result, (list, tuple)) and result and isinstance(result[0], list):
            return _parse_rapid_page(result[0])
        return []
    lines: list[dict] = []
    for i, text in enumerate(txts or []):
        box = boxes[i] if boxes is not None and i < len(boxes) else None
        try:
            conf = float(scores[i]) if scores is not None and i < len(scores) else 1.0
        except (TypeError, ValueError):
            conf = 1.0
        lines.append({"text": str(text), "conf": conf, "box": box})
    return lines


def build_backend(name: str, use_gpu: bool, mock_lines: list[dict] | None) -> BaseOCRBackend:
    if name == "mock":
        return MockOCRBackend(mock_lines)
    if name == "paddle":
        return PaddleOCRBackend(use_gpu=use_gpu)
    if name == "rapid":
        return RapidOCRBackend(use_gpu=use_gpu)
    raise ValueError(f"未知 --ocr-backend: {name} (可选 paddle|rapid|mock)")


# --------------------------------------------------------------------------- #
# parsing / de-duplication
# --------------------------------------------------------------------------- #
def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def parse_ocr_lines(raw: list[dict], conf_threshold: float = LOW_CONF_THRESHOLD) -> dict | None:
    """Collapse raw OCR rows into one dialogue line.

    The topmost short row is the speaker name bar; remaining rows are joined
    into the utterance. A direct ``speaker`` field (mock / VLM) wins.
    """
    # LAB-2129: OCR 后过滤，丢弃机翻插件状态栏残留行。
    rows = [
        r
        for r in raw
        if r.get("text", "").strip() and not is_overlay_noise(r.get("text", ""))
    ]
    if not rows:
        return None
    speaker = next((r.get("speaker") for r in rows if r.get("speaker")), "")
    body = list(rows)
    if not speaker:
        for r in body:
            if len(r["text"].strip()) <= 8:
                speaker = r["text"].strip()
                body.remove(r)
                break
    text_ja = "".join(r["text"].strip() for r in body).strip()
    if not text_ja:
        return None
    conf = min((float(r.get("conf", 1.0)) for r in rows), default=1.0)
    return {
        "speaker": speaker,
        "text_ja": text_ja,
        "conf": round(conf, 4),
        "needs_review": conf < conf_threshold,
    }


def dedup_line(previous: dict | None, current: dict, ratio: float = 0.86) -> bool:
    """True when ``current`` is a near-copy / residual half-line of previous."""
    if previous is None:
        return False
    a, b = previous["text_ja"], current["text_ja"]
    if not a or not b:
        return False
    if a == b:
        return True
    if a in b or b in a:
        short, long = (a, b) if len(a) < len(b) else (b, a)
        if len(short) / max(len(long), 1) >= 0.6:
            return True
    return _similarity(a, b) >= ratio


def vlm_review(url: str, crop_path: str, original: dict) -> dict:
    """Optional low-confidence VLM fallback (best-effort, never blocks pipeline)."""
    try:
        import base64
        import urllib.request

        with open(crop_path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        payload = json.dumps({"image": b64, "original": original}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if data.get("text_ja"):
            original["text_ja"] = data["text_ja"]
        if "conf" in data:
            original["conf"] = float(data["conf"])
        original["needs_review"] = original["conf"] < LOW_CONF_THRESHOLD
    except Exception as exc:  # network/model failure -> keep machine output
        print(f"[warn] VLM fallback failed: {exc}", file=sys.stderr)
    return original


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def _video_list(videos_dir: str | Path) -> list[tuple[int, Path]]:
    videos_dir = Path(videos_dir)
    vids = [p for p in videos_dir.glob("*.mp4")]
    vids = [p for p in vids if re.match(r"^P\d+", p.name)] or vids

    def num(p: Path) -> int:
        m = re.search(r"(\d+)", p.stem)
        return int(m.group(1)) if m else 9999

    return [(i, p) for i, p in enumerate(sorted(vids, key=num), start=1)]


def _load_mock_lines(path: str | None) -> list[dict] | None:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def run_extract(
    bvid: str,
    videos_dir: str | Path,
    work_dir: str | Path,
    fps: float = DEFAULT_FPS,
    roi: tuple[int, int, int, int] | None = None,
    ocr_backend: str = "paddle",
    batch_size: int = DEFAULT_BATCH_SIZE,
    mock_lines_path: str | None = None,
    force_cpu: bool = False,
    vlm_fallback_url: str | None = None,
    use_gpu: bool = True,
    part_workers: int = DEFAULT_PART_WORKERS,
    max_parts: int | None = None,
) -> dict:
    """Run extraction for all videos in ``videos_dir``; write the two contract files."""
    t_start = time.time()
    work = Path(work_dir)
    out_dir = work / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    low_conf_dir = out_dir / "low_conf"
    low_conf_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg = find_ffmpeg()
    hwaccel = probe_hwaccel(ffmpeg, force_cpu=force_cpu)
    backend = build_backend(ocr_backend, use_gpu and bool(hwaccel), _load_mock_lines(mock_lines_path))
    videos = _video_list(videos_dir)
    if max_parts is not None and max_parts > 0:
        videos = videos[:max_parts]
    if not videos:
        raise FileNotFoundError(f"未在 {videos_dir} 找到 .mp4 视频")

    def _decode_part(item: tuple[int, Path]):
        part, video = item
        t0 = time.time()
        decode_cache = work / "cache" / f"P{part}"
        # LAB-2133: jpg 抽帧 + 把 crop 下推进 ffmpeg（已知 ROI 时零冗余像素落盘）。
        frames, used_roi = decode_frames_with_roi(video, fps, hwaccel, decode_cache, roi=roi)
        return item, frames, used_roi, time.time() - t0

    # LAB-2129: --part-workers 并发多 P 抽帧（解码是 IO/CPU 密集，线程并发安全）；
    # OCR 仍在主线程按 P 顺序执行，避免多线程共享 PaddleOCR 引擎。
    if part_workers and part_workers > 1:
        with ThreadPoolExecutor(max_workers=part_workers) as pool:
            decoded = list(pool.map(_decode_part, videos))
    else:
        decoded = [_decode_part(item) for item in videos]

    roi_used = roi
    t_decode = 0.0
    prepared: list[tuple[int, list[FinalizedFrame]]] = []
    for (part, _video), frames, used_roi, dt in decoded:
        t_decode += dt
        if not frames:
            continue
        roi_used = roi_used or used_roi
        # frames 已是按 used_roi 裁好的小图，指纹检测用相对 ROI 即可。
        crop_roi = (0, 0, used_roi[2], used_roi[3])
        prepared.append((part, detect_finalized_frames(frames, crop_roi)))

    all_lines: list[dict] = []
    t_ocr = 0.0
    for part, finals in prepared:
        t1 = time.time()
        images = [f.crop for f in finals]
        ts_secs = [f.ts_sec for f in finals]
        batches = backend.recognize_batch(images, ts_secs, batch_size)
        previous: dict | None = None
        for final, raw in zip(finals, batches):
            parsed = parse_ocr_lines(raw)
            if not parsed:
                continue
            if dedup_line(previous, parsed):
                continue
            previous = parsed
            ts = round(final.ts_sec, 3)
            crop_name = f"P{part}_{ts:07.2f}.png"
            crop_path = low_conf_dir / crop_name
            final.crop.save(crop_path)
            line = {
                "part": part,
                "ts_sec": ts,
                "speaker": parsed["speaker"],
                "text_ja": parsed["text_ja"],
                "conf": parsed["conf"],
                "needs_review": parsed["needs_review"],
                "_crop": str(crop_path),
            }
            if line["needs_review"] and vlm_fallback_url:
                line.update(vlm_review(vlm_fallback_url, str(crop_path), dict(parsed)))
            all_lines.append(line)
        t_ocr += time.time() - t1

    low_conf_count = sum(1 for ln in all_lines if ln["needs_review"])
    for idx, line in enumerate(all_lines, start=1):
        line.pop("_crop", None)
        out = {
            "line_id": idx,
            "part": line["part"],
            "ts_sec": line["ts_sec"],
            "speaker": line["speaker"],
            "text_ja": line["text_ja"],
            "conf": line["conf"],
            "needs_review": line["needs_review"],
        }
        all_lines[idx - 1] = out

    script_path = out_dir / "script.jsonl"
    with script_path.open("w", encoding="utf-8") as fh:
        for line in all_lines:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")

    total = round(time.time() - t_start, 3)
    stats = {
        "bvid": bvid,
        "roi": list(roi_used or (0, 0, 0, 0)),
        "total_parts": len(videos),
        "total_lines": len(all_lines),
        "avg_conf": round(float(np.mean([l["conf"] for l in all_lines])), 4) if all_lines else 0.0,
        "low_conf_count": low_conf_count,
        "timings_sec": {
            "decode_roi": round(t_decode, 3),
            "ocr": round(t_ocr, 3),
            "total": total,
        },
    }
    with (out_dir / "extract_stats.json").open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    print(
        f"[extract_ocr] hwaccel={hwaccel or 'cpu'} backend={backend.name} "
        f"parts={len(videos)} lines={len(all_lines)} low_conf={low_conf_count} total={total}s"
    )
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LAB-2125 视频抽取与本地 GPU OCR 引擎")
    p.add_argument("--bvid", default="", help="BV 号（写入统计）")
    p.add_argument("--videos-dir", required=True, help="P1.mp4..P9.mp4 所在目录")
    p.add_argument("--work-dir", required=True, help="工作目录，输出位于 <work-dir>/output/")
    p.add_argument("--fps", type=float, default=DEFAULT_FPS)
    p.add_argument("--roi", default=None, help="手动覆写 ROI: x,y,w,h")
    p.add_argument("--ocr-backend", choices=["paddle", "rapid", "mock"], default="paddle")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--mock-lines", default=None, help="mock 后端：ts 窗口 -> 台词 JSON 路径")
    p.add_argument("--force-cpu", action="store_true", help="强制 CPU 解码与 OCR")
    p.add_argument("--vlm-fallback-url", default=None, help="低置信度帧的可选 VLM 复核 HTTP 接口")
    p.add_argument(
        "--part-workers", type=int, default=DEFAULT_PART_WORKERS,
        help="并发抽帧的分 P 数（默认 1，多卡或 CPU 富余时可调大）",
    )
    p.add_argument("--max-parts", type=int, default=None, help="仅处理前 N 个分 P（调试用）")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    roi = tuple(int(v) for v in args.roi.split(",")) if args.roi else None
    if roi and len(roi) != 4:
        print("--roi 必须是 x,y,w,h 四个整数", file=sys.stderr)
        return 2
    run_extract(
        bvid=args.bvid,
        videos_dir=args.videos_dir,
        work_dir=args.work_dir,
        fps=args.fps,
        roi=roi,
        ocr_backend=args.ocr_backend,
        batch_size=args.batch_size,
        mock_lines_path=args.mock_lines,
        force_cpu=args.force_cpu,
        vlm_fallback_url=args.vlm_fallback_url,
        part_workers=args.part_workers,
        max_parts=args.max_parts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

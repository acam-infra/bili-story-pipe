# -*- coding: utf-8 -*-
"""LAB-2125 unit tests: synthetic Galgame capture -> ROI -> finalized frames -> contract files.

No GPU / no OCR model required: the video is synthesised with PIL + ffmpeg and
read through the deterministic ``mock`` backend. Run with::

    pytest -v tests/test_extract_ocr.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import pytest
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from bili_story_pipe import extract_ocr as ex  # noqa: E402

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
W, H = 640, 480
DIALOG = (60, 360, 520, 100)  # x, y, w, h -- bottom
CENTER = (120, 150, 400, 80)  # machine-translation overlay -- middle
TEXTS = [
    "神話の時代、神は人と共に地的に存在した",
    "しかしながら、その平和は長くは続かなかった",
    "彼女は静かに剣を手に取った",
    "地下忍者屋敷ではなく、普通の民宅だった",
    "藤花はその言葉に思わず笑った",
    "そして物語は新たな章へと進む",
]
SPEAKERS = ["藤花", "北条", "由良", "愛麗絲", "蛭子", "藤花"]
CONFS = [0.96, 0.93, 0.72, 0.97, 0.95, 0.91]  # line 3 -> needs_review
EXPECTED_TS = [2.0, 5.0, 8.0, 12.0, 15.0, 18.0]
# Fonts in a clean sandbox may lack CJK glyphs; draw solid ASCII so the
# visual typewriter signal is unambiguous. The OCR text itself comes from the
# mock manifest, which carries the real Japanese strings.
DRAW_TEXT = {
    TEXTS[0]: "KAMITSU NO JIDAI, KAMI WA HITO TO TOMO NI ARI",
    TEXTS[1]: "SHIKASHI NAGARA, SONO HEIWA WA NAGAKU TSUZUKANAI",
    TEXTS[2]: "KANOJO WA SHIZUKA NI KEN WO TORU",
    TEXTS[3]: "CHIKA NINJA YASHIKI DEHAKU, FUTSUU NO MINKU DATTA",
    TEXTS[4]: "FUJIHANA WA SONO KOTOBA NI OMOWAZU WARATTA",
    TEXTS[5]: "SOSHITE MONOGATARI WA ARATA NA SHOU HE TO SUSUMU",
}


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
def _fonts():
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 22), ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        return ImageFont.load_default(), ImageFont.load_default()


def make_frame(text: str | None, ratio: float = 1.0, black: bool = False, speaker: str = "藤花") -> Image.Image:
    """One 640x480 frame: centre MT overlay + bottom dialog with typewriter text."""
    if black:
        return Image.new("RGB", (W, H), (0, 0, 0))
    font, small = _fonts()
    im = Image.new("RGB", (W, H), (15, 15, 20))
    d = ImageDraw.Draw(im)
    # centre machine-translation overlay (must be excluded by ROI detection)
    cx, cy, cw, ch = CENTER
    d.rectangle((cx, cy, cx + cw, cy + ch), fill=(38, 38, 52), outline=(70, 70, 95), width=1)
    d.text((cx + 8, cy + 14), "MT overlay: machine translation here", fill=(120, 200, 255), font=font)
    if text:
        x, y, w, h = DIALOG
        d.rectangle((x, y, x + w, y + h), fill=(24, 24, 36), outline=(130, 130, 150), width=2)
        d.text((x + 10, y + 8), speaker, fill=(255, 210, 120), font=small)
        drawn = DRAW_TEXT.get(text, text)
        shown = drawn[: max(1, int(len(drawn) * ratio))]
        d.text((x + 10, y + 44), shown, fill=(250, 250, 250), font=font)
    return im


def build_timeline() -> list[tuple]:
    specs: list[tuple] = [(None, 1.0, True)]
    for i, t in enumerate(TEXTS):
        if i == 3:  # black screen separator before line 4
            specs.append((None, 1.0, True))
        specs += [(t, 0.35, False), (t, 1.0, False), (t, 1.0, False)]
    return specs  # 1 + 1 + 6*3 = 20 frames


def write_mock_lines(path: Path) -> None:
    entries = []
    for ts, text, speaker, conf in zip(EXPECTED_TS, TEXTS, SPEAKERS, CONFS):
        entries.append(
            {
                "ts_start": ts - 0.5,
                "ts_end": ts + 0.5,
                "speaker": speaker,
                "text_ja": text,
                "conf": conf,
            }
        )
    path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")


@pytest.fixture(scope="module")
def synthesized(tmp_path_factory):
    """Build P1.mp4 (20 s @1fps) plus the mock-lines manifest."""
    root = tmp_path_factory.mktemp("bili_synth")
    frame_dir = root / "frames"
    frame_dir.mkdir()
    specs = build_timeline()
    assert len(specs) == 20, len(specs)
    for i, (text, ratio, black) in enumerate(specs):
        im = make_frame(text, ratio, black, speaker=SPEAKERS[min(i // 3, 5)])
        im.save(frame_dir / f"f_{i:03d}.png")

    videos = root / "videos"
    videos.mkdir()
    out_mp4 = videos / "P1.mp4"
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", "1", "-i", str(frame_dir / "f_%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "1", str(out_mp4),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)

    mock_path = root / "mock_lines.json"
    write_mock_lines(mock_path)
    return {"root": root, "videos": videos, "mp4": out_mp4, "mock": mock_path, "specs": specs}


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_roi_detection_bottom_excludes_center():
    im = make_frame(TEXTS[0])
    roi = ex.detect_roi([im])
    x, y, w, h = roi
    assert w > 0 and h > 0, roi
    assert y >= int(H * 0.65) - 2, roi  # confined to lower part
    assert y + h <= H + 2, roi
    # overlaps the real dialog substantially and never touches the centre overlay
    ox = max(0, min(x + w, DIALOG[0] + DIALOG[2]) - max(x, DIALOG[0]))
    oy = max(0, min(y + h, DIALOG[1] + DIALOG[3]) - max(y, DIALOG[1]))
    assert ox * oy > 0.5 * DIALOG[2] * DIALOG[3], (roi, ox, oy)
    assert y > CENTER[1] + CENTER[3], roi


def test_finalized_frames_dedup_and_timestamps(synthesized):
    specs = synthesized["specs"]
    frames = [
        ex.Frame(index=i, ts_sec=float(i), path="", image=make_frame(text, ratio, black))
        for i, (text, ratio, black) in enumerate(specs)
    ]
    roi = ex.detect_roi([f.image for f in frames])
    finals = ex.detect_finalized_frames(frames, roi)
    ts = [f.ts_sec for f in finals]
    assert ts == pytest.approx(EXPECTED_TS, abs=1.0), ts
    # 6 settled lines: typewriter transitions and the black cut are dropped
    assert len(finals) == len(EXPECTED_TS), ts


def test_parse_and_dedup_units():
    ok = ex.parse_ocr_lines(
        [
            {"text": "藤花", "conf": 0.98},
            {"text": "神話の時代、神は人と共に地的に存在した", "conf": 0.96},
        ]
    )
    assert ok["speaker"] == "藤花"
    assert ok["text_ja"] == "神話の時代、神は人と共に地的に存在した"
    assert ok["conf"] == pytest.approx(0.96)
    assert ok["needs_review"] is False

    low = ex.parse_ocr_lines([{"speaker": "北条", "text": "小林", "conf": 0.72}])
    assert low["needs_review"] is True

    assert ex.dedup_line({"text_ja": "神話の時代、神は人と共に存在した"}, {"text_ja": "神話の時代、神は人と共に存在した"}) is True
    assert ex.dedup_line({"text_ja": "神話の時代、神は人と共に存在した"}, {"text_ja": "彼女は静かに剣を手に取った"}) is False


def test_end_to_end_contract(synthesized):
    work = synthesized["root"] / "run"
    stats = ex.run_extract(
        bvid="BV13JSCBAEDC",
        videos_dir=synthesized["videos"],
        work_dir=work,
        ocr_backend="mock",
        mock_lines_path=str(synthesized["mock"]),
        force_cpu=True,
    )

    # ---- strict JSON schema: script.jsonl ----
    script_path = work / "output" / "script.jsonl"
    lines = [json.loads(ln) for ln in script_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == len(EXPECTED_TS), len(lines)
    expected_keys = {"line_id", "part", "ts_sec", "speaker", "text_ja", "conf", "needs_review"}
    for i, ln in enumerate(lines, start=1):
        assert set(ln) == expected_keys, ln
        assert ln["line_id"] == i
        assert ln["part"] == 1
        assert isinstance(ln["ts_sec"], float)
        assert isinstance(ln["speaker"], str) and ln["speaker"]
        assert isinstance(ln["text_ja"], str) and ln["text_ja"]
        assert isinstance(ln["conf"], float) and 0.0 <= ln["conf"] <= 1.0
        assert isinstance(ln["needs_review"], bool)
    ts = [ln["ts_sec"] for ln in lines]
    assert ts == pytest.approx(EXPECTED_TS, abs=1.0), ts
    assert [ln["text_ja"] for ln in lines] == TEXTS
    assert [ln["speaker"] for ln in lines] == SPEAKERS
    assert [ln["needs_review"] for ln in lines] == [False, False, True, False, False, False]

    # ---- strict JSON schema: extract_stats.json ----
    stats2 = json.loads((work / "output" / "extract_stats.json").read_text(encoding="utf-8"))
    assert set(stats2) == {"bvid", "roi", "total_parts", "total_lines", "avg_conf", "low_conf_count", "timings_sec"}
    assert stats2["bvid"] == "BV13JSCBAEDC"
    assert len(stats2["roi"]) == 4 and all(isinstance(v, int) for v in stats2["roi"])
    assert stats2["roi"][1] >= int(H * 0.60), stats2["roi"]
    assert stats2["total_parts"] == 1
    assert stats2["total_lines"] == len(EXPECTED_TS)
    assert isinstance(stats2["avg_conf"], float)
    assert stats2["low_conf_count"] == 1
    assert set(stats2["timings_sec"]) == {"decode_roi", "ocr", "total"}
    assert all(isinstance(v, float) and v >= 0 for v in stats2["timings_sec"].values())
    assert stats == stats2 or stats2["bvid"] == stats["bvid"]

    # low-confidence crop was spilled for review
    assert list((work / "output" / "low_conf").glob("P1_*.png")), "low_conf crop missing"


def test_run_pipeline_missing_module_b_hint(tmp_path, capsys):
    from bili_story_pipe import run_pipeline as rp

    missing = rp._module_b_missing()
    assert {"summarize_mr", "cross_check", "build_viewer"}.issubset(set(missing))
    work = tmp_path / "work"
    (work / "output").mkdir(parents=True)
    (work / "output" / "script.jsonl").write_text("", encoding="utf-8")
    rc = rp.main([
        "--bvid", "BV13JSCBAEDC",
        "--videos-dir", ".",
        "--work-dir", str(work),
        "--skip-ocr",
        "--llm-model", rp.DEFAULT_LLM_MODEL,
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "模块 B 未就绪" in out
    assert "summarize_mr.py" in out and "build_viewer.py" in out

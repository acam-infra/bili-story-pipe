# -*- coding: utf-8 -*-
"""LAB-2129 unit tests: PaddleOCR 3.x + overlay-noise filter + compare report.

No GPU / no OCR model required: PaddleOCR is monkeypatched in-process with a
fake 3.x ``predict()`` module. Run with::

    pytest -v tests/
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from bili_story_pipe import compare_transcripts as ct  # noqa: E402
from bili_story_pipe import extract_ocr as ex  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. constants / overlay noise
# --------------------------------------------------------------------------- #
def test_default_bottom_fraction_is_072():
    assert ex.DEFAULT_BOTTOM_FRACTION == 0.72


@pytest.mark.parametrize(
    "text,expected",
    [
        ("翻译结束", True),
        ("翻译结果：已完成", True),
        ("耗时：1234ms", True),
        ("耗时:999ms", True),
        ("利时：0.5s", True),
        ("神話の時代、神は人と共に存在した", False),
        ("藤花", False),
        ("", False),
    ],
)
def test_is_overlay_noise(text, expected):
    assert ex.is_overlay_noise(text) is expected


def test_parse_ocr_lines_filters_overlay_leftover():
    """A status-bar row dropped into the ROI must not become dialogue."""
    parsed = ex.parse_ocr_lines(
        [
            {"text": "藤花", "conf": 0.98},
            {"text": "神話の時代、神は人と共に存在した", "conf": 0.96},
            {"text": "翻译结束 耗时：12ms", "conf": 0.99},
            {"text": "翻译结果：已完成", "conf": 0.99},
        ]
    )
    assert parsed is not None
    assert parsed["speaker"] == "藤花"
    assert parsed["text_ja"] == "神話の時代、神は人と共に存在した"

    # all-noise input is discarded entirely
    assert ex.parse_ocr_lines([{"text": "利时：3ms", "conf": 0.99}]) is None


# --------------------------------------------------------------------------- #
# 2. PaddleOCR version detection / v3 predict parsing
# --------------------------------------------------------------------------- #
def _install_fake_paddleocr(monkeypatch, version: str, engine_cls):
    fake = types.ModuleType("paddleocr")
    fake.__version__ = version
    fake.PaddleOCR = engine_cls
    monkeypatch.setitem(sys.modules, "paddleocr", fake)


class _FakeResultV3:
    def __init__(self, data: dict):
        self.json = data


def test_paddle_v3_predict_parsing(monkeypatch):
    captured: dict = {}

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def predict(self, bgr_list):
            # 3.x predict takes a list of BGR ndarrays and returns OCRResult list.
            assert isinstance(bgr_list, list) and len(bgr_list) == 2
            data = {
                "rec_texts": ["藤花", "神話の時代、神は人と共に存在した"],
                "rec_scores": [0.99, 0.96],
                "dt_polys": [
                    [[0, 0], [10, 0], [10, 10], [0, 10]],
                    [[2, 2], [20, 2], [20, 20], [2, 20]],
                ],
            }
            return [_FakeResultV3(data) for _ in bgr_list]

    _install_fake_paddleocr(monkeypatch, "3.7.0", FakePaddleOCR)
    backend = ex.PaddleOCRBackend(use_gpu=True)

    assert backend.major == 3
    assert captured == {
        "lang": "japan",
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "device": "gpu:0",
    }

    images = [Image.new("RGB", (32, 32), (30, 30, 30)) for _ in range(2)]
    pages = backend.recognize_batch(images, [0.0, 1.0], batch_size=32)
    assert len(pages) == 2
    assert pages[0][0]["text"] == "藤花"
    assert pages[0][0]["conf"] == pytest.approx(0.99)
    assert pages[0][1]["text"] == "神話の時代、神は人と共に存在した"
    assert pages[0][1]["box"] == [[2, 2], [20, 2], [20, 20], [2, 20]]

    # CPU path must map device=cpu
    _install_fake_paddleocr(monkeypatch, "3.7.0", FakePaddleOCR)
    ex.PaddleOCRBackend(use_gpu=False)
    assert captured["device"] == "cpu"


def test_paddle_v3_parse_plain_dict(monkeypatch):
    """Results may also be plain dicts (no .json attribute)."""
    parsed = ex._parse_paddle_v3_result(
        {"rec_texts": ["菊丸", "いいぞ"], "rec_scores": [0.91, 0.88], "dt_polys": []}
    )
    assert [row["text"] for row in parsed] == ["菊丸", "いいぞ"]
    assert [row["conf"] for row in parsed] == pytest.approx([0.91, 0.88])
    assert parsed[0]["box"] is None


def test_paddle_v2_legacy_ocr(monkeypatch):
    """<3 versions keep the original ocr(..., cls=True) contract."""
    captured: dict = {}

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def ocr(self, images, cls=True, batch_size=None):
            captured["cls"] = cls
            captured["batch_size"] = batch_size
            box = [[0, 0], [9, 0], [9, 9], [0, 9]]
            # Legacy _parse_paddle_page consumes a doubly-wrapped page level.
            return [[[[box, ("北条", 0.93)]]]]

    _install_fake_paddleocr(monkeypatch, "2.7.3", FakePaddleOCR)
    backend = ex.PaddleOCRBackend(use_gpu=True)
    assert backend.major == 2
    assert captured["use_gpu"] is True

    pages = backend.recognize_batch([Image.new("RGB", (16, 16))], [0.0], 32)
    assert pages[0][0]["text"] == "北条"
    assert captured["cls"] is True and captured["batch_size"] == 32


# --------------------------------------------------------------------------- #
# 3. CLI options
# --------------------------------------------------------------------------- #
def test_new_cli_flags():
    args = ex.build_arg_parser().parse_args(
        ["--videos-dir", "v", "--work-dir", "w", "--part-workers", "3", "--max-parts", "2"]
    )
    assert args.part_workers == 3
    assert args.max_parts == 2
    assert ex.build_arg_parser().parse_args(["--videos-dir", "v", "--work-dir", "w"]).part_workers == 1


# --------------------------------------------------------------------------- #
# 4. compare_transcripts end-to-end
# --------------------------------------------------------------------------- #
def test_compare_transcripts_end_to_end(tmp_path):
    new_rows = [
        {
            "line_id": 1,
            "part": 1,
            "ts_sec": 2.0,
            "speaker": "藤花",
            "text_ja": "神話の時代、神は人と共に存在した",
            "conf": 0.96,
            "needs_review": False,
        },
        {
            "line_id": 2,
            "part": 1,
            "ts_sec": 8.0,
            "speaker": "藤花",
            "text_ja": "ちょっと待って",
            "conf": 0.95,
            "needs_review": False,
        },
        {
            "line_id": 3,
            "part": 2,
            "ts_sec": 65.0,
            "speaker": "ナナコ",
            "text_ja": "彼女は静かに剣を手に取った",
            "conf": 0.72,
            "needs_review": True,
        },
    ]
    new_jsonl = tmp_path / "script.jsonl"
    new_jsonl.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in new_rows), encoding="utf-8"
    )

    old_dir = tmp_path / "old"
    old_dir.mkdir()
    (old_dir / "P1.txt").write_text(
        "[藤花] 神話の時代、神は人と共に存在した\n"
        "[藤花] 不明\n"
        "[菊丸] 枚目はここから\n"
        "[北条] 翻译结束 耗时：12ms\n",
        encoding="utf-8",
    )
    (old_dir / "P2.txt").write_text(
        "[ナナコ] 彼女は静かに剣を手に取った\n", encoding="utf-8"
    )

    out = tmp_path / "diff_report.md"
    rc = ct.main(
        [
            "--new-jsonl", str(new_jsonl),
            "--old-dir", str(old_dir),
            "--output", str(out),
        ]
    )
    assert rc == 0
    md = out.read_text(encoding="utf-8")

    # section headers
    assert "## 1. 分 P" in md
    assert "## 2. 噪音率对比" in md
    assert "## 3. 核心角色名出现频次对比" in md
    assert "## 4. 新版新捕获样本" in md

    # per-part counts: P1 new=2 old=4, P2 new=1 old=1
    assert "| P1 | 2 | 4 |" in md
    assert "| P2 | 1 | 1 |" in md
    assert "00:02–00:08" in md

    # noise detection: old has 2 VLM annotations + 1 overlay leftover
    assert "| 旧版（fps=1/5） | 2 | 1 | 5 |" in md
    assert "| 新版（script.jsonl） | 0 | 0 | 3 |" in md

    # all 10 core character names present
    for name in (
        "藤花", "菊丸", "ナナコ", "チカ", "伊緒",
        "北条/春希", "ゆら", "たま", "アイリス", "蛭子",
    ):
        assert f"| {name} |" in md

    # newly-captured short line with [P{part} mm:ss] anchor
    assert "`[P1 00:08]`" in md
    assert "ちょっと待って" in md

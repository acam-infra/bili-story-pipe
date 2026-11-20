# bili-story-pipe

End-to-end automated pipeline for extracting visual-novel / Galgame dialogue from long gameplay videos (e.g. Bilibili multi-part walkthroughs), structuring full plot arcs via Map-Reduce LLM summarization, cross-checking against external VNDB metadata, and compiling a zero-dependency offline single-file interactive HTML story viewer.

## Pipeline Architecture

1. **Fast Video Extraction & Local GPU OCR (`bili_story_pipe/extract_ocr.py`)**
   - `ffmpeg` CUDA hardware decoding (`-hwaccel cuda`, auto-fallback to CPU) with direct ROI cropping to lightweight `.jpg` frames.
   - Automatic bottom dialogue-box ROI detection (`bottom_fraction=0.72` with bottom-edge extension) to exclude centered machine-translation overlays, plus `is_overlay_noise()` post-filtering.
   - Typewriter settled-frame deduplication with frame-accurate `[mm:ss]` timestamps.
   - Local GPU batch OCR backends:
     - **`rapid` (recommended for x86_64 & aarch64 CUDA)**: `RapidOCR` PyTorch-CUDA (`PP-OCRv4-JAPAN`) with 16-frame vertical strip tiling (`Strip16`), reaching ~70 frames/s on A100 (~5 min pure GPU OCR for 12.2h of video).
     - **`paddle`**: PaddleOCR 2.x / 3.x (`predict()`).
     - **`mock`**: Deterministic backend for unit testing.
   - Outputs standardized `output/script.jsonl` and `output/extract_stats.json`.

2. **Map-Reduce Plot Structuring & Coverage Audit (`bili_story_pipe/summarize_mr.py`)**
   - Splits `script.jsonl` by episode/chunk (300–500 lines) and runs concurrent OpenAI-compatible LLM calls with compact JSON output (`strict=False`) to prevent long-context truncation and "lost in the middle" plot omissions.
   - Runs automated coverage auditing across Top-15 speakers and 500-line sliding windows (excluding narration meta-tags) with targeted backfill.
   - Outputs `output/plot_structured.json` and `output/script_bilingual.jsonl`.

3. **External Cross-Check (`bili_story_pipe/cross_check.py`)**
   - Queries the public **VNDB Kana API** (`/kana/vn` + `/kana/character` with Japanese `original` names & `aliases`) and supports optional external search hooks (`--search-cmd`).
   - Aligns character roster, multi-part cumulative runtime, content tags, and synopsis highlights into `output/cross_check.json` (`verified` / `uncovered` / `conflict`).

4. **Offline Single-File Story Viewer (`bili_story_pipe/build_viewer.py`)**
   - Compiles all structured plot cards, cross-check diffs, and full searchable transcripts into one self-contained `.html` file.
   - **Zero blocking external CDN resources** (instant offline open with built-in SVG relation graph; Mermaid loads asynchronously on demand).
   - `safeStorage` `try/catch` wrapper (safe under `file://` and sandboxed contexts).
   - Lazy-rendered, paginated transcript table (100 rows/page) with speaker filters, keyword search, and one-click Bilibili `?p=<part>&t=<sec>` timestamp links.

5. **Transcript Diff Benchmark (`bili_story_pipe/compare_transcripts.py`)**
   - Compares newly extracted `script.jsonl` against legacy transcript folders (per-part line counts, noise rate, character frequency, and newly captured short lines).

## Quickstart

```bash
pip install -r requirements-extract.txt -r requirements-analyze.txt

# One-click pipeline (existing multi-part P1.mp4..PN.mp4 directory)
python3 bili_story_pipe/run_pipeline.py \
  --bvid BV13JSCBAEDC \
  --videos-dir ./videos \
  --work-dir ./work_run \
  --ocr-backend rapid \
  --fps 1 \
  --batch-size 16 \
  --llm-base-url https://your-openai-compatible-endpoint/v1 \
  --llm-api-key "$API_KEY" \
  --llm-model gemini-3.6-flash
```

## Tests

```bash
pytest -v tests/
```

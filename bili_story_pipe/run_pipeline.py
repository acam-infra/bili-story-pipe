#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_pipeline.py -- one-click CLI for bili-story-pipe (LAB-2123 phase 1 MVP).

  python run_pipeline.py --bvid BV13JSCBAEDC --videos-dir /data/bili_work/videos \\
      --work-dir /data/bili_work/run1 [--skip-ocr] [--skip-cross-check]

Stage 1 (this module / LAB-2125) is always available: extract_ocr.
Stages 2-4 belong to module B (LAB-2127): summarize_mr, cross_check,
build_viewer. When their scripts are not yet present, a clear hint is printed
and the pipeline exits cleanly so module B can be dropped in with no rewiring.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DEFAULT_LLM_MODEL = "geminiweb2api/gemini-3.6-flash"
MODULE_B = {
    "summarize_mr": "LAB-2127 模块 B 尚未合入：缺少 summarize_mr.py（Map-Reduce 剧情归约）",
    "cross_check": "LAB-2127 模块 B 尚未合入：缺少 cross_check.py（VNDB / grok-search 交叉校验）",
    "build_viewer": "LAB-2127 模块 B 尚未合入：缺少 build_viewer.py（离线单页剧情浏览器）",
}


def _module_b_missing() -> list[str]:
    return [name for name in MODULE_B if not (HERE / f"{name}.py").exists()]


def _run_script(name: str, args: list[str]) -> int:
    cmd = [sys.executable, str(HERE / f"{name}.py")] + args
    print(f"[run_pipeline] -> {' '.join(cmd)}")
    return subprocess.call(cmd)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="bili-story-pipe 一键流水线")
    p.add_argument("--bvid", default="BV13JSCBAEDC")
    p.add_argument("--videos-dir", required=True, help="P1.mp4..P9.mp4 目录")
    p.add_argument("--work-dir", required=True, help="输出目录，契约文件在 <work-dir>/output/")
    p.add_argument("--llm-base-url", default="", help="OpenAI 兼容接口 base URL（模块 B 使用）")
    p.add_argument("--llm-api-key", default="", help="API Key（模块 B 使用）")
    p.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    p.add_argument("--ocr-backend", choices=["paddle", "rapid", "mock"], default="paddle")
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--roi", default=None, help="手动 ROI x,y,w,h（默认自动检测）")
    p.add_argument("--mock-lines", default=None)
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--vlm-fallback-url", default=None)
    p.add_argument("--search-cmd", default="", help="模块 B cross_check 外挂检索命令")
    p.add_argument("--skip-ocr", action="store_true", help="跳过 OCR，复用已有 output/script.jsonl")
    p.add_argument("--skip-cross-check", action="store_true", help="跳过 cross_check 阶段")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1: extract (LAB-2125, always available) ----
    if not args.skip_ocr:
        from extract_ocr import run_extract

        roi = tuple(int(v) for v in args.roi.split(",")) if args.roi else None
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
        )
    else:
        script = work / "output" / "script.jsonl"
        if not script.exists():
            print(f"[run_pipeline] --skip-ocr 但缺少 {script}，无法继续", file=sys.stderr)
            return 2
        print(f"[run_pipeline] 跳过 OCR，复用 {script}")

    # ---- Stages 2-4: module B (may not be merged yet) ----
    missing = _module_b_missing()
    if missing:
        print("\n[run_pipeline] 模块 B 未就绪，前半段产出已完成：")
        for name in missing:
            print(f"  - {MODULE_B[name]}")
        print("  将 summarize_mr.py / cross_check.py / build_viewer.py 放入本目录后重跑即可无缝续接。")
        return 0

    llm_args = [
        "--work-dir", str(work),
        "--base-url", args.llm_base_url,
        "--api-key", args.llm_api_key,
        "--model", args.llm_model,
    ]
    rc = _run_script("summarize_mr", llm_args)
    if rc != 0:
        return rc

    if not args.skip_cross_check:
        cross_args = ["--work-dir", str(work)]
        if args.search_cmd:
            cross_args += ["--search-cmd", args.search_cmd]
        rc = _run_script("cross_check", cross_args)
        if rc != 0:
            return rc

    out_dir = work / "output"
    viewer_args = [
        "--plot", str(out_dir / "plot_structured.json"),
        "--script", str(out_dir / "script_bilingual.jsonl"),
        "--output", str(out_dir / "story_viewer.html"),
        "--bvid", args.bvid,
    ]
    cross_path = out_dir / "cross_check.json"
    if not args.skip_cross_check and cross_path.exists():
        viewer_args += ["--cross", str(cross_path)]
    rc = _run_script("build_viewer", viewer_args)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

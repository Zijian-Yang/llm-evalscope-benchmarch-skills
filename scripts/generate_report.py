#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility wrapper for detailed model-benchmark Markdown reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model_benchmark import DEFAULT_CONFIG, generate_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a detailed model benchmark Markdown report")
    parser.add_argument("benchmark_dir", help="Directory containing EvalScope benchmark_summary.json files")
    parser.add_argument("--output", "-o", help="Output Markdown path")
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    output = Path(args.output) if args.output else benchmark_dir / "model_benchmark_report.md"
    generate_report(DEFAULT_CONFIG, benchmark_dir, output)
    print(f"报告已生成: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

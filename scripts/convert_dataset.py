#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility wrapper for converting prompt datasets to EvalScope openqa JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model_benchmark import convert_to_openqa  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert JSONL/TXT data to EvalScope openqa format")
    parser.add_argument("input", help="Input JSONL/TXT path")
    parser.add_argument("output", help="Output JSONL path")
    parser.add_argument("--format", "-f", default="auto", choices=["auto", "messages", "openqa", "text"])
    args = parser.parse_args()

    total, success, failed = convert_to_openqa(Path(args.input), Path(args.output), args.format)
    print(json.dumps({"total": total, "success": success, "failed": failed, "output": args.output}, ensure_ascii=False, indent=2))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

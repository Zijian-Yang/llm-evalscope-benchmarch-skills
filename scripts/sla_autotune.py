#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility wrapper for SLA auto-tuning through model_benchmark.py."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    script = Path(__file__).resolve().parent / "model_benchmark.py"
    import subprocess

    return subprocess.call([sys.executable, str(script), "legacy-sla", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())

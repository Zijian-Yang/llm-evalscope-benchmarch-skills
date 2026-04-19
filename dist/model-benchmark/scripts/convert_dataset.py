#!/usr/bin/env python3
import json
import sys
from pathlib import Path

source, target = Path(sys.argv[1]), Path(sys.argv[2])
with source.open(encoding="utf-8") as src, target.open("w", encoding="utf-8") as dst:
    for line in src:
        item = json.loads(line)
        text = item.get("question") or item.get("prompt") or item.get("text") or ""
        dst.write(json.dumps({"question": text}, ensure_ascii=False) + "\n")

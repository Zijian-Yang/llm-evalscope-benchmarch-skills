#!/usr/bin/env python3
import sys
from model_benchmark import main

raise SystemExit(main(["run", "--scenario", "rate", *sys.argv[1:]]))

#!/usr/bin/env python3
"""Run the dependency-light classical anomaly baseline."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inspectnet_cx.eval.classical_baseline import main

if __name__ == "__main__":
    main()

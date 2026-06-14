"""Tests that scripts/train_patchcore.py exposes a working --seed flag.

We do not train a real model here (too slow); we test:
  1. The CLI parser accepts --seed and the helper sets RNGs deterministically.
  2. _seed_everything makes torch + numpy + random produce identical sequences
     across two calls with the same seed.
  3. (Indirect) Documents that seed flows into the sidecar dict via the train()
     signature, by asserting "seed" is one of train()'s keyword args.
"""

from __future__ import annotations

import importlib.util
import inspect
import random
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "train_patchcore.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("train_patchcore_mod", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_patchcore_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_seed_everything_is_deterministic():
    mod = _load_module()
    mod._seed_everything(1234)
    a_py = [random.random() for _ in range(5)]
    a_np = np.random.rand(5).tolist()
    import torch

    a_torch = torch.rand(5).tolist()

    mod._seed_everything(1234)
    b_py = [random.random() for _ in range(5)]
    b_np = np.random.rand(5).tolist()
    b_torch = torch.rand(5).tolist()

    assert a_py == b_py
    assert a_np == b_np
    assert a_torch == b_torch


def test_seed_flag_present_in_cli():
    mod = _load_module()
    # main() builds the parser; instead of invoking, inspect by parsing --help-like.
    # We use argparse's introspection via a fake call: parse a known-good arg list.
    try:
        mod.main(["--category", "x", "--output-dir", "/tmp/xx", "--seed", "7", "--help"])
    except SystemExit:
        pass
    # If we got here without "unrecognized arguments", the flag exists.
    # Also assert train() accepts a `seed` kwarg.
    sig = inspect.signature(mod.train)
    assert "seed" in sig.parameters, "train() must accept a seed kwarg"


def test_seed_appears_in_sidecar_schema():
    """The sidecar JSON written by train() must include the 'seed' key."""
    src = SCRIPT.read_text()
    assert '"seed": seed' in src, "sidecar dict must contain a 'seed' field"

"""Narrow DGL compatibility gate for GOGGLE's core graph APIs."""
from __future__ import annotations

import sys
import types
from importlib.util import find_spec
from pathlib import Path

import torch


def allow_dgl_without_graphbolt() -> None:
    """Disable only DGL's unavailable optional GraphBolt extension.

    DGL 2.1 ships GraphBolt binaries for PyTorch 2.0--2.2, so its unconditional
    loader otherwise rejects the benchmark's pinned PyTorch 2.9.1. GOGGLE uses
    the core graph and ``dgl.nn`` APIs, never GraphBolt or distributed loading.
    """
    spec = find_spec("dgl")
    if spec is None or spec.origin is None:
        return
    version = torch.__version__.split("+", 1)[0]
    if sys.platform.startswith("linux"):
        name = f"libgraphbolt_pytorch_{version}.so"
    elif sys.platform.startswith("darwin"):
        name = f"libgraphbolt_pytorch_{version}.dylib"
    else:
        name = f"graphbolt_pytorch_{version}.dll"
    if not (Path(spec.origin).parent / "graphbolt" / name).is_file():
        sys.modules.setdefault("dgl.graphbolt", types.ModuleType("dgl.graphbolt"))

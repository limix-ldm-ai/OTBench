"""Workspace paths shared by the data layer, methods, and command line."""
from __future__ import annotations

import os
from pathlib import Path


def workspace_root(value: str | Path | None = None) -> Path:
    """Resolve an explicit workspace, environment default, or current directory."""
    root = value if value is not None else os.environ.get("OTBENCH_WORKSPACE", Path.cwd())
    return Path(root).expanduser().resolve()

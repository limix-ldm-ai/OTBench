# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Shared task-runner orchestration: the train-or-reuse checkpoint policy."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from typing import Optional

from . import checkpoints as ckpt_utils
from .base import TabularMethod, TrainContext


def ensure_trained(
    method_obj: TabularMethod,
    bundle,
    ckpt_dir: Path,
    train_seed: int,
    device: Optional[str] = None,
    force_retrain: bool = False,
) -> bool:
    """Train when needed ("no ckpt -> train, valid ckpt -> reuse").

    Returns True if a training run was executed.
    """
    if not method_obj.trainable:
        return False
    ckpt_dir = Path(ckpt_dir)
    tctx = TrainContext(
        bundle=bundle, config=method_obj.config, checkpoint_dir=Path(ckpt_dir),
        train_seed=train_seed, device=device, force_retrain=force_retrain,
    )
    if not force_retrain and method_obj.is_trained(tctx):
        print(f"[OTBench] reuse checkpoint: {ckpt_dir}")
        return False

    # Different task ranks may route to the same checkpoint_group.  Take the
    # group/dataset/seed lock and re-check after waiting: normally exactly one
    # process trains and every peer reuses its completed checkpoint.
    with ckpt_utils.checkpoint_lock(ckpt_dir):
        if not force_retrain and method_obj.is_trained(tctx):
            print(f"[OTBench] reuse checkpoint after wait: {ckpt_dir}")
            return False
        if ckpt_dir.exists() and any(ckpt_dir.iterdir()):
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            history = ckpt_dir.parent / ".history"
            history.mkdir(parents=True, exist_ok=True)
            archive = history / f"{ckpt_dir.name}-{stamp}-{uuid4().hex[:8]}"
            # Rename the complete old directory, including its metadata, before
            # a new writer can replace files. No old artifact is deleted.
            ckpt_dir.rename(archive)
            print(f"[OTBench] archived previous checkpoint: {archive}")
        print(f"[OTBench] training {method_obj.name} -> {ckpt_dir}")
        method_obj.train(tctx)
        return True

# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Bootstrap resampling of training rows: the other diagnostic bound.

Where the independent-marginal baseline is a fidelity floor, this one is a
fidelity ceiling and a privacy floor. It returns training rows verbatim, so:

* every fidelity, utility and distinguishability metric is at its best possible
  value — any generator that "beats" it is measuring noise, and any metric where
  a real generator beats it is suspect;
* every memorisation and privacy metric is at its worst — exact-copy rate 1.0,
  DCR 0, membership inference maximally successful.

A metric that cannot separate this baseline from a real generator is not
measuring privacy. It is emphatically not a generator: ``status="diagnostic"``
in the registry, and the resampling never pretends to protect anything.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from otbench.methods.common import checkpoints as ckpt_utils
from otbench.methods.common.base import GenerateContext, TabularMethod, Task, TrainContext
from otbench.methods.common.seeding import seed_everything

_TRAIN_FILE = "train_rows.parquet"


class TrainBootstrapMethod(TabularMethod):
    name = "train_bootstrap"
    tasks = frozenset({Task.GENERATION})
    trainable = True

    def train(self, ctx: TrainContext) -> None:
        ckpt_dir = Path(ctx.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        frame = ctx.bundle.train.reset_index(drop=True)[ctx.bundle.schema.all_columns]
        try:
            frame.to_parquet(ckpt_dir / _TRAIN_FILE, index=False)
            artifact = _TRAIN_FILE
        except Exception:  # noqa: BLE001 — no parquet engine available
            artifact = "train_rows.csv"
            frame.to_csv(ckpt_dir / artifact, index=False)
        with (ckpt_dir / "bootstrap.json").open("w", encoding="utf-8") as f:
            json.dump({"artifact": artifact, "n_train_rows": len(frame)}, f)
        ckpt_utils.write_metadata(
            ckpt_dir, self.name, ctx.bundle, self.config, ctx.train_seed,
            artifacts={"train_rows": artifact, "bootstrap": "bootstrap.json"},
        )

    def generate(self, ctx: GenerateContext) -> pd.DataFrame:
        seed_everything(ctx.seed)
        rng = np.random.default_rng(ctx.seed)
        ckpt_dir = Path(ctx.checkpoint_dir)
        with (ckpt_dir / "bootstrap.json").open("r", encoding="utf-8") as f:
            payload = json.load(f)
        path = ckpt_dir / payload["artifact"]
        frame = (pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path))

        n = int(ctx.n_samples or len(frame))
        replace = bool(self.cfg("model", "with_replacement", True))
        index = rng.choice(len(frame), size=n, replace=replace or n > len(frame))
        return frame.iloc[index].reset_index(drop=True)

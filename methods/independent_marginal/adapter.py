# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Independent per-column marginal sampler: a diagnostic, not a generator.

Each column is sampled independently from its own training marginal, so every
univariate marginal is (asymptotically) perfect while all dependence between
columns is destroyed.

That is exactly its purpose. Any metric this baseline scores well on is a metric
that only looks at marginals, and cannot distinguish a model of the joint
distribution from a product of marginals. Running it alongside the real
generators is the cheapest way to find out which metrics in a benchmark are
doing no work — which is the question this benchmark exists to answer.

Expected behaviour: near-perfect Shape/KS/TVD, poor Trend, near-zero
association similarity, C2ST AUROC approaching 1 on any dataset with real
dependence, and poor downstream utility.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from otbench.methods.common import checkpoints as ckpt_utils
from otbench.methods.common.base import GenerateContext, TabularMethod, Task, TrainContext
from otbench.methods.common.seeding import seed_everything

_MARGINALS_FILE = "marginals.json"


class IndependentMarginalMethod(TabularMethod):
    name = "independent_marginal"
    tasks = frozenset({Task.GENERATION})
    trainable = True

    def train(self, ctx: TrainContext) -> None:
        """"Training" is storing each column's empirical marginal."""
        schema = ctx.bundle.schema
        train = ctx.bundle.train.reset_index(drop=True)
        marginals: dict[str, dict] = {}
        for column in schema.all_columns:
            values = train[column]
            if column in set(schema.numerical_cols):
                numeric = pd.to_numeric(values, errors="coerce").dropna()
                marginals[column] = {
                    "kind": "numerical",
                    # The empirical marginal is the observed values themselves;
                    # resampling them keeps the marginal exact instead of imposing
                    # a parametric shape the data may not have.
                    "values": numeric.tolist(),
                }
            else:
                counts = values.astype(str).value_counts(normalize=True)
                marginals[column] = {
                    "kind": "categorical",
                    "categories": counts.index.tolist(),
                    "probabilities": counts.to_numpy().tolist(),
                }

        ckpt_dir = Path(ctx.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with (ckpt_dir / _MARGINALS_FILE).open("w", encoding="utf-8") as f:
            json.dump({"columns": schema.all_columns, "marginals": marginals}, f)
        ckpt_utils.write_metadata(
            ckpt_dir, self.name, ctx.bundle, self.config, ctx.train_seed,
            artifacts={"marginals": _MARGINALS_FILE},
        )

    def generate(self, ctx: GenerateContext) -> pd.DataFrame:
        seed_everything(ctx.seed)
        rng = np.random.default_rng(ctx.seed)
        with (Path(ctx.checkpoint_dir) / _MARGINALS_FILE).open("r", encoding="utf-8") as f:
            payload = json.load(f)

        n = int(ctx.n_samples or len(ctx.bundle.train))
        out = {}
        for column in payload["columns"]:
            record = payload["marginals"][column]
            if record["kind"] == "numerical":
                pool = np.asarray(record["values"], dtype=float)
                out[column] = (rng.choice(pool, size=n, replace=True) if len(pool)
                               else np.full(n, np.nan))
            else:
                out[column] = rng.choice(
                    record["categories"], size=n, p=np.asarray(record["probabilities"]))
        return pd.DataFrame(out)[payload["columns"]]

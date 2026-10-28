# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Gaussian copula across all three tasks, from one joint checkpoint.

The same fitted joint model answers all three questions, which is the point of
including it: generation samples the joint, imputation conditions on the
observed cells of a row, and prediction conditions on the features. Nothing is
retrained per task, so a difference between the three results is a difference in
the question, not in the model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from otbench.methods.common import checkpoints as ckpt_utils
from otbench.methods.common.base import (
    GenerateContext,
    ImputationOutput,
    ImputeContext,
    PredictContext,
    PredictionOutput,
    TabularMethod,
    Task,
    TrainContext,
)
from otbench.methods.common.seeding import seed_everything

from .model import GaussianCopula

_MODEL_FILE = "copula.json"


class GaussianCopulaMethod(TabularMethod):
    name = "gaussian_copula"
    tasks = frozenset({Task.GENERATION, Task.IMPUTATION, Task.PREDICTION})
    trainable = True

    # ---------------------------------------------------------------- train
    def train(self, ctx: TrainContext) -> None:
        seed_everything(ctx.train_seed)
        schema = ctx.bundle.schema
        frame = ctx.bundle.train.reset_index(drop=True)[schema.all_columns]
        model = GaussianCopula.fit(
            frame,
            numerical=list(schema.numerical_cols),
            categorical=list(schema.categorical_cols),
            seed=int(ctx.train_seed),
            max_quantiles=int(self.cfg("model", "max_quantiles", 1000)),
        )
        ckpt_dir = Path(ctx.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save(ckpt_dir / _MODEL_FILE)
        ckpt_utils.write_metadata(
            ckpt_dir, self.name, ctx.bundle, self.config, ctx.train_seed,
            artifacts={"model": _MODEL_FILE},
        )

    def _load(self, ctx) -> GaussianCopula:
        return GaussianCopula.load(Path(ctx.checkpoint_dir) / _MODEL_FILE)

    # ------------------------------------------------------------- generate
    def generate(self, ctx: GenerateContext) -> pd.DataFrame:
        seed_everything(ctx.seed)
        model = self._load(ctx)
        rng = np.random.default_rng(ctx.seed)
        n = int(ctx.n_samples or len(ctx.bundle.train))
        return model.sample(n, rng)[ctx.bundle.schema.all_columns]

    # --------------------------------------------------------------- impute
    def impute_draws(self, ctx: ImputeContext) -> ImputationOutput:
        """K draws from the conditional, each conditioned on the same observed cells.

        The draws come from the model's own conditional distribution rather than
        from restarting a deterministic routine, so they are a genuine posterior
        sample and the K-draw calibration metrics apply.
        """
        model = self._load(ctx)
        schema = ctx.bundle.schema
        masked = ctx.masked_test.reset_index(drop=True)
        mask = ctx.mask.reset_index(drop=True)
        k = max(1, int(ctx.trials))

        draws = []
        for draw_index in range(k):
            seed_everything(ctx.seed + draw_index)
            rng = np.random.default_rng(ctx.seed + draw_index)
            completed = masked.copy()
            # Rows are grouped by their missingness pattern so each group gets one
            # exact conditional (a different pattern is a different conditional).
            patterns = mask[schema.all_columns].apply(
                lambda row: tuple(row.astype(bool)), axis=1)
            for pattern, rows in patterns.groupby(patterns).groups.items():
                targets = [c for c, missing in zip(schema.all_columns, pattern, strict=True)
                           if missing]
                if not targets:
                    continue
                index = np.asarray(rows)
                sampled = model.sample_conditional(
                    masked.loc[index, schema.all_columns], targets, rng)
                for column in targets:
                    completed.loc[index, column] = sampled[column].to_numpy()
            draws.append(completed[masked.columns].reset_index(drop=True))

        return ImputationOutput.from_frames(draws, metadata={
            "source": "native",
            "draw_strategy": "closed-form Gaussian conditional, one exact conditional per "
                             "missingness pattern",
            "n_draws": k,
        })

    def impute(self, ctx: ImputeContext) -> pd.DataFrame:
        return self.impute_draws(ctx).draws[0]

    # -------------------------------------------------------------- predict
    def predict(self, ctx: PredictContext) -> PredictionOutput:
        seed_everything(ctx.seed)
        model = self._load(ctx)
        schema = ctx.bundle.schema
        rng = np.random.default_rng(ctx.seed)
        target = schema.target_col
        test = ctx.bundle.test.reset_index(drop=True)[schema.all_columns].copy()
        test[target] = np.nan  # the target is what we are conditioning *for*

        is_regression = (
            schema.task_type == "regression" or target in set(schema.numerical_cols)
        )
        if is_regression:
            n_samples = int(self.cfg("predict", "n_samples", ctx.n_samples))
            samples = np.column_stack([
                pd.to_numeric(
                    model.sample_conditional(test, [target], rng)[target], errors="coerce"
                ).to_numpy(float)
                for _ in range(n_samples)
            ])
            return PredictionOutput(
                y_pred=np.median(samples, axis=1),
                y_samples=samples,
                metadata={
                    "method": self.name,
                    "n_samples": n_samples,
                    "quantiles": "not emitted: empirical quantiles of Monte Carlo samples "
                                 "are not presented as native quantile predictions",
                    "density": "none: the copula has a density, but the marginals are "
                               "empirical step functions, so a normalised conditional "
                               "log-density on the original scale is not available",
                },
            )

        proba, categories = model.conditional_category_probabilities(
            test, target, rng, n_samples=int(self.cfg("predict", "n_class_samples", 200)),
        )
        codes = proba.argmax(axis=1)
        return PredictionOutput(
            y_pred=np.asarray([categories[int(c)] for c in codes], dtype=object),
            y_proba=proba,
            y_score=proba[:, 1] if len(categories) == 2 else None,
            labels=list(categories),
            threshold=0.5 if len(categories) == 2 else None,
            threshold_source="argmax",
            metadata={
                "method": self.name,
                "probability_source": "Monte Carlo over the closed-form conditional latent",
                "n_class_samples": int(self.cfg("predict", "n_class_samples", 200)),
            },
        )

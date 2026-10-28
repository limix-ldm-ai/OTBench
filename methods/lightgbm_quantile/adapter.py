# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Quantile-regression LightGBM: one model per quantile level.

This is a regression-only method that emits *native* predictive quantiles, so
it can be scored on pinball loss, interval coverage and width. It is the honest
version of "a GBDT with uncertainty": nothing here pretends to be a density, and
the fixed quantile grid is part of the method's configuration rather than being
read off the metric that happens to be requested.

The known weakness is stated rather than hidden. Each level is fitted by an
independent model, so the predicted quantiles are not monotone by construction.
The adapter reports the crossing rate and does **not** sort the quantiles into
order: sorting would silently produce a well-behaved interval from a model that
did not predict one, and the interval metrics would then describe the sort.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from otbench.methods.common import checkpoints as ckpt_utils
from otbench.methods.common.base import (
    PredictContext,
    PredictionOutput,
    Task,
    TrainContext,
)
from otbench.methods.common.data import split_features_target
from otbench.methods.common.seeding import seed_everything
from otbench.methods.common.supervised import SupervisedMethod, _split_indices

from otbench.methods.lightgbm.adapter import _eval_kwargs, import_lightgbm

_META_FILE = "quantile.json"

# Fixed grid: contains the exact tails the 90% and 95% interval metrics need, so
# those metrics never have to interpolate a quantile the model did not predict.
DEFAULT_LEVELS = (0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975)


class LightGBMQuantileMethod(SupervisedMethod):
    name = "lightgbm_quantile"
    tasks = frozenset({Task.PREDICTION})
    emits_proba = False

    @property
    def levels(self) -> list[float]:
        return [float(q) for q in self.cfg("model", "quantile_levels", list(DEFAULT_LEVELS))]

    def build_estimator(self, task_type, random_state, n_classes):
        raise NotImplementedError("lightgbm_quantile trains one model per quantile level")

    # ---------------------------------------------------------------- train
    def train(self, ctx: TrainContext) -> None:
        import joblib

        lgb = import_lightgbm()
        seed_everything(ctx.train_seed)
        schema = ctx.bundle.schema
        if (schema.task_type or "") != "regression":
            raise ValueError(
                "lightgbm_quantile is a regression method; quantiles of a categorical "
                "target are not defined"
            )

        split = split_features_target(ctx.bundle.train.reset_index(drop=True), schema)
        y = pd.to_numeric(split.y, errors="coerce")
        keep = y.notna()
        x, y = split.X.loc[keep], y[keep]
        train_idx, valid_idx = _split_indices(len(y), ctx.train_seed)

        preprocessor = self._preprocessor(schema)
        x_train = preprocessor.fit_transform(x.iloc[train_idx])
        y_train = y.iloc[train_idx].to_numpy(float)
        x_valid = preprocessor.transform(x.iloc[valid_idx]) if len(valid_idx) else None
        y_valid = y.iloc[valid_idx].to_numpy(float) if len(valid_idx) else None

        models = {}
        for level in self.levels:
            model = lgb.LGBMRegressor(
                objective="quantile", alpha=level,
                n_estimators=int(self.cfg("model", "n_estimators", 500)),
                learning_rate=float(self.cfg("model", "learning_rate", 0.05)),
                num_leaves=int(self.cfg("model", "num_leaves", 31)),
                random_state=int(ctx.train_seed), n_jobs=int(self.cfg("model", "n_jobs", -1)),
                verbose=-1,
            )
            if x_valid is not None:
                model.fit(x_train, y_train, **_eval_kwargs(model, x_valid, y_valid),
                          callbacks=[lgb.early_stopping(
                              int(self.cfg("train", "early_stopping_rounds", 50)),
                              verbose=False), lgb.log_evaluation(0)])
            else:
                model.fit(x_train, y_train)
            models[level] = model

        ckpt_dir = Path(ctx.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"preprocessor": preprocessor, "models": models}, ckpt_dir / "model.joblib")
        with (ckpt_dir / _META_FILE).open("w", encoding="utf-8") as f:
            json.dump({
                "levels": self.levels,
                # One model per level: the cost metrics must attribute all of them.
                "n_auxiliary_models": len(models),
                "n_train_rows": int(len(train_idx)),
                "n_validation_rows": int(len(valid_idx)),
            }, f, indent=2)
        ckpt_utils.write_metadata(
            ckpt_dir, self.name, ctx.bundle, self.config, ctx.train_seed,
            artifacts={"model": "model.joblib", "quantile": _META_FILE},
        )

    # -------------------------------------------------------------- predict
    def predict(self, ctx: PredictContext) -> PredictionOutput:
        import joblib

        ckpt_dir = Path(ctx.checkpoint_dir)
        payload = joblib.load(ckpt_dir / "model.joblib")
        with (ckpt_dir / _META_FILE).open("r", encoding="utf-8") as f:
            meta = json.load(f)

        schema = ctx.bundle.schema
        x_test = split_features_target(ctx.bundle.test.reset_index(drop=True), schema).X
        encoded = payload["preprocessor"].transform(x_test)

        levels = sorted(payload["models"])
        quantiles = np.column_stack([
            payload["models"][level].predict(encoded) for level in levels
        ])
        crossing = float(np.mean(np.diff(quantiles, axis=1) < 0)) if len(levels) > 1 else 0.0

        median_index = int(np.argmin(np.abs(np.asarray(levels) - 0.5)))
        return PredictionOutput(
            y_pred=quantiles[:, median_index],
            quantiles=quantiles,
            quantile_levels=list(levels),
            metadata={
                "method": self.name,
                "point_prediction": f"the q={levels[median_index]} model, not a mean",
                "quantile_crossing_rate": crossing,
                "monotonicity": "not enforced: the levels are fitted independently and the "
                                "predictions are reported unsorted, so the crossing rate "
                                "stays visible to the interval metrics",
                "n_auxiliary_models": meta["n_auxiliary_models"],
            },
        )

"""Shared foundation-model workflow (TabPFN / TabICL / LimiX).

Pretrained, training-free methods supporting:
- prediction: fit the pretrained estimator on the train split (sklearn-like
  in-context interface), predict the test target;
- imputation: column-wise protocol — for each masked column, fit an estimator
  on clean train rows (other features -> column) and predict the masked cells.

Subclasses implement ``make_regressor`` / ``make_classifier`` (and optionally
``_prepare_features`` for models needing fully numeric inputs).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from . import data as data_utils
from .base import (
    ImputeContext,
    PredictContext,
    PredictionOutput,
    TabularMethod,
    Task,
)
from .capabilities import UnsupportedRun
from .seeding import seed_everything


class FoundationMethod(TabularMethod):
    tasks = frozenset({Task.IMPUTATION, Task.PREDICTION})
    trainable = False
    # Foundation backends generally train their classification head for a
    # bounded number of labels. Subclasses set the limits they actually ship.
    max_imputation_classes: int | None = None
    max_prediction_classes: int | None = None

    # --- subclass hooks -----------------------------------------------------
    def make_regressor(self, ctx):
        raise NotImplementedError

    def make_classifier(self, ctx):
        raise NotImplementedError

    def _prepare_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """Optionally transform features (e.g. numeric encoding for LimiX)."""
        return X_train, X_test

    # --- config helpers -----------------------------------------------------
    def _inference_cfg(self, key: str, default):
        return self.cfg("impute", key, default)

    def estimator_kwargs(self, ctx=None) -> dict:
        """Extra kwargs forwarded to the underlying estimator constructors."""
        kwargs = dict(self.config.get("model", {}).get("estimator_kwargs", {}))
        if ctx is not None:
            kwargs.setdefault("device", self.resolve_device(ctx))
            kwargs.setdefault("random_state", int(ctx.seed))
        return kwargs

    # --- imputation (column-wise) --------------------------------------------
    def impute(self, ctx: ImputeContext) -> pd.DataFrame:
        seed_everything(ctx.seed)
        bundle, schema = ctx.bundle, ctx.bundle.schema
        X_train_full = data_utils.feature_only(bundle.train.reset_index(drop=True), schema)
        X_test_masked = data_utils.feature_only(ctx.masked_test.reset_index(drop=True), schema)

        train_idx = self._train_indices(len(X_train_full), ctx.seed)
        X_train = X_train_full.iloc[train_idx].reset_index(drop=True)
        current = data_utils.fill_feature_nans(X_train_full, X_test_masked, schema).reset_index(drop=True)

        mask = ctx.mask.reset_index(drop=True)
        missing_cols = [
            col for col in mask.columns
            if col in current.columns and mask[col].fillna(False).any()
        ]

        iterations = int(self._inference_cfg("iterations", 1))
        for _iteration in range(max(1, iterations)):
            for col in missing_cols:
                rows = mask[col].fillna(False).to_numpy(dtype=bool)
                if not rows.any():
                    continue
                other_cols = [c for c in current.columns if c != col]
                X_fit_raw = data_utils.fill_feature_nans(X_train_full, X_train[other_cols], schema)
                X_pred_raw = data_utils.fill_feature_nans(X_train_full, current.loc[rows, other_cols], schema)
                y_fit = X_train[col].reset_index(drop=True)

                X_fit, X_pred = self._prepare_features(X_fit_raw, X_pred_raw)
                if schema.column_type(col) == "categorical":
                    modes = y_fit.mode(dropna=True)
                    values = y_fit.fillna(modes.iloc[0] if len(modes) else "__MISSING__")
                    encoder = LabelEncoder()
                    y_encoded = encoder.fit_transform(values.astype(str))
                    self._check_class_limit(
                        len(encoder.classes_), self.max_imputation_classes,
                        task="imputation", column=col,
                    )
                    estimator = self.make_classifier(ctx)
                    estimator.fit(X_fit, y_encoded)
                    preds = self._batched_predict(estimator, X_pred)
                    labels = encoder.inverse_transform(np.asarray(preds, dtype=int))
                    current.loc[rows, col] = labels
                else:
                    y_num = pd.to_numeric(y_fit, errors="coerce")
                    fill = y_num.mean()
                    y_num = y_num.fillna(0.0 if pd.isna(fill) else fill)
                    estimator = self.make_regressor(ctx)
                    estimator.fit(X_fit, y_num)
                    preds = self._batched_predict(estimator, X_pred)
                    current.loc[rows, col] = np.asarray(preds, dtype=float)

        return data_utils.append_target(current, bundle.test, schema)

    # --- prediction (native) --------------------------------------------------
    def predict(self, ctx: PredictContext) -> PredictionOutput:
        seed_everything(ctx.seed)
        bundle, schema = ctx.bundle, ctx.bundle.schema
        train = bundle.train.reset_index(drop=True)
        test = bundle.test.reset_index(drop=True)

        X_train_full = data_utils.fill_feature_nans(
            data_utils.feature_only(train, schema),
            data_utils.feature_only(train, schema), schema,
        )
        X_test_full = data_utils.fill_feature_nans(
            data_utils.feature_only(train, schema),
            data_utils.feature_only(test, schema), schema,
        )
        train_idx = self._train_indices(len(X_train_full), ctx.seed)
        X_train_raw = X_train_full.iloc[train_idx].reset_index(drop=True)
        X_train, X_test = self._prepare_features(X_train_raw, X_test_full)

        is_regression = schema.task_type == "regression"
        if is_regression:
            y_train = pd.to_numeric(train[schema.target_col].iloc[train_idx], errors="coerce")
            fill = y_train.mean()
            y_train = y_train.fillna(0.0 if pd.isna(fill) else fill)
            estimator = self.make_regressor(ctx)
            estimator.fit(X_train, y_train.to_numpy(dtype=np.float32))
            y_pred = np.asarray(self._batched_predict(estimator, X_test), dtype=np.float64).reshape(-1)
            return PredictionOutput(y_pred=y_pred, metadata={"adapter": self.name})

        encoder = LabelEncoder()
        y_all = train[schema.target_col].astype(str)
        encoder.fit(y_all)
        self._check_class_limit(
            len(encoder.classes_), self.max_prediction_classes,
            task="prediction", column=schema.target_col,
        )
        y_train = encoder.transform(y_all.iloc[train_idx])
        estimator = self.make_classifier(ctx)
        estimator.fit(X_train, y_train)
        y_pred_codes = np.asarray(self._batched_predict(estimator, X_test), dtype=np.int64)
        y_proba = self._batched_predict_proba(estimator, X_test)

        classes_str = list(encoder.classes_)
        estimator_classes = np.asarray(
            getattr(estimator, "classes_", np.arange(len(classes_str))), dtype=np.int64
        )
        y_pred = encoder.inverse_transform(np.clip(y_pred_codes, 0, len(classes_str) - 1))
        labels = [classes_str[int(c)] for c in estimator_classes if 0 <= int(c) < len(classes_str)]
        return PredictionOutput(
            y_pred=np.asarray(y_pred, dtype=object),
            y_proba=y_proba,
            labels=labels if y_proba is not None else None,
            metadata={"adapter": self.name},
        )

    # --- shared plumbing ------------------------------------------------------
    def _batched_predict(self, estimator, X: pd.DataFrame):
        batch = int(self._inference_cfg("predict_batch_size", 4096))
        if len(X) <= batch:
            return estimator.predict(X)
        outputs = []
        for start in range(0, len(X), batch):
            outputs.append(estimator.predict(X.iloc[start:start + batch]))
        return np.concatenate(outputs)

    def _batched_predict_proba(self, estimator, X: pd.DataFrame) -> Optional[np.ndarray]:
        if not hasattr(estimator, "predict_proba"):
            return None
        batch = int(self._inference_cfg("predict_batch_size", 4096))
        if len(X) <= batch:
            return np.asarray(estimator.predict_proba(X))
        outputs = []
        for start in range(0, len(X), batch):
            outputs.append(np.asarray(estimator.predict_proba(X.iloc[start:start + batch])))
        return np.concatenate(outputs, axis=0)

    def _train_indices(self, n_rows: int, seed: int) -> np.ndarray:
        max_rows = self._inference_cfg("max_train_rows", 50000)
        if max_rows is None or n_rows <= int(max_rows):
            return np.arange(n_rows)
        rng = np.random.default_rng(int(self._inference_cfg("train_seed", 0)) + seed)
        return np.sort(rng.choice(n_rows, size=int(max_rows), replace=False))

    def _check_class_limit(
        self, n_classes: int, limit: int | None, *, task: str, column: str
    ) -> None:
        if limit is not None and int(n_classes) > int(limit):
            raise UnsupportedRun(
                f"{self.name} {task} supports at most {limit} classes, but "
                f"column {column!r} has {n_classes}."
            )


def numeric_frame(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Train-fitted numeric encoding (LimiX needs fully numeric matrices)."""
    train = X_train.copy()
    test = X_test.copy()
    for col in train.columns:
        if pd.api.types.is_numeric_dtype(train[col]):
            fill = pd.to_numeric(train[col], errors="coerce").mean()
            fill = 0.0 if pd.isna(fill) else fill
            train[col] = pd.to_numeric(train[col], errors="coerce").fillna(fill)
            test[col] = pd.to_numeric(test[col], errors="coerce").fillna(fill)
        else:
            mode = train[col].mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else "__MISSING__"
            categories = {
                value: idx
                for idx, value in enumerate(pd.Series(train[col].fillna(fill).unique()).astype(str))
            }
            fallback = categories[str(fill)]
            train[col] = train[col].fillna(fill).astype(str).map(categories).fillna(fallback)
            test[col] = test[col].fillna(fill).astype(str).map(categories).fillna(fallback)
    return train.astype(np.float32), test.astype(np.float32)


__all__ = ["FoundationMethod", "numeric_frame"]

# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Shared adapter for supervised predictors (linear, RF, GBDT, MLP, ...).

Every traditional predictor differs only in which estimator it constructs. The
parts that are easy to get subtly wrong — preprocessing fitted on train only, a
label vocabulary that ignores the test split, a decision threshold that is never
tuned on test, probability columns labelled with the classes they belong to —
are identical across all of them, so they live here and are written once.

Each predictor still gets its own registry name, config file and checkpoint
directory. Sharing the adapter is an implementation detail; it must not make two
methods share a checkpoint, because then a bug in one would silently move the
other's numbers.

What the adapter deliberately does *not* do:

* It does not fit a KDE over anything to manufacture a density. A method emits
  ``log_prob_fn`` only if its estimator has a real conditional density.
* It does not tune the classification threshold. The rule is argmax for
  multiclass and a fixed 0.5 for binary unless the subclass fits one on the
  train/validation split and says so through ``threshold_source``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import checkpoints as ckpt_utils
from .base import (
    ImputeContext,
    MethodContext,
    PredictContext,
    PredictionOutput,
    TabularMethod,
    Task,
    TrainContext,
)
from .data import categorical_feature_columns, numerical_feature_columns, split_features_target
from .seeding import seed_everything

_MODEL_FILE = "model.joblib"
_META_FILE = "supervised.json"

# Fraction of train held out for methods that need an unbiased selection signal
# (early stopping, threshold fitting). It is carved out of train, never test.
VALIDATION_FRACTION = 0.2


class SupervisedMethod(TabularMethod):
    """Base for ``fit(X, y) / predict(X)`` estimators from a standard library.

    Subclasses implement :meth:`build_estimator` and set ``task_types``. The
    rest — encoding, checkpointing, probability alignment, output packaging — is
    handled here.
    """

    tasks = frozenset({Task.PREDICTION})
    trainable = True

    #: Whether the estimator natively exposes ``predict_proba``.
    emits_proba: bool = True
    #: Whether the estimator handles raw categorical columns itself (CatBoost).
    native_categorical: bool = False
    #: Whether this adapter consumes an explicit outer validation split.
    uses_validation: bool = False

    # ---------------------------------------------------------------- build
    def build_estimator(self, task_type: str, random_state: int, n_classes: int | None):
        raise NotImplementedError(f"{self.name} must implement build_estimator()")

    def fit_estimator(self, estimator, x_train, y_train, x_valid, y_valid, task_type: str):
        """Hook for estimators with an early-stopping or eval-set protocol."""
        estimator.fit(x_train, y_train)
        return {}

    # ---------------------------------------------------------------- train
    def train(self, ctx: TrainContext) -> None:
        import joblib

        seed_everything(ctx.train_seed)
        schema = ctx.bundle.schema
        task_type = _task_type(schema)
        split = split_features_target(ctx.bundle.train.reset_index(drop=True), schema)

        classes: list | None = None
        y = split.y
        if task_type != "regression":
            classes = sorted(pd.Series(y.dropna().unique()).astype(str).tolist())
            y = y.astype(str).map({c: i for i, c in enumerate(classes)})
            keep = y.notna()
        else:
            y = pd.to_numeric(y, errors="coerce")
            keep = y.notna()
        x, y = split.X.loc[keep], y[keep]
        if len(y) == 0:
            raise ValueError(f"{self.name}: no training row has a usable target value.")

        if self.uses_validation:
            train_idx, valid_idx = _split_indices(len(y), ctx.train_seed)
        else:
            train_idx = np.arange(len(y), dtype=int)
            valid_idx = np.asarray([], dtype=int)
        preprocessor = self._preprocessor(schema)
        estimator = self.build_estimator(
            task_type, int(ctx.train_seed), len(classes) if classes else None
        )
        pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])

        fit_info = self.fit_estimator(
            pipeline,
            x.iloc[train_idx], y.iloc[train_idx].to_numpy(),
            x.iloc[valid_idx] if len(valid_idx) else None,
            y.iloc[valid_idx].to_numpy() if len(valid_idx) else None,
            task_type,
        )

        ckpt_dir = Path(ctx.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, ckpt_dir / _MODEL_FILE)
        meta = {
            "task_type": task_type,
            "classes": classes,
            "label_vocabulary_source": "train",
            "n_train_rows": int(len(train_idx)),
            "n_validation_rows": int(len(valid_idx)),
            "validation_fraction": VALIDATION_FRACTION if self.uses_validation else 0.0,
            "train_target_scale": _train_target_scale(y) if task_type == "regression" else None,
            "fit_info": fit_info,
        }
        with (ckpt_dir / _META_FILE).open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
        ckpt_utils.write_metadata(
            ckpt_dir, self.name, ctx.bundle, self.config, ctx.train_seed,
            artifacts={"model": _MODEL_FILE, "supervised": _META_FILE},
        )

    # -------------------------------------------------------------- predict
    def predict(self, ctx: PredictContext) -> PredictionOutput:
        pipeline, meta = self._load(ctx)
        schema = ctx.bundle.schema
        x_test = split_features_target(ctx.bundle.test.reset_index(drop=True), schema).X

        if meta["task_type"] == "regression":
            return self._regression_output(pipeline, x_test, meta, ctx)
        return self._classification_output(pipeline, x_test, meta)

    def _classification_output(self, pipeline, x_test, meta) -> PredictionOutput:
        classes = meta["classes"]
        model, encoded = _staged_transform(pipeline, x_test)
        proba = None
        if self.emits_proba and hasattr(model, "predict_proba"):
            raw = np.asarray(model.predict_proba(encoded), dtype=float)
            # sklearn orders columns by the codes it saw; map them back onto the
            # full train class list so a class absent from the fit still has a
            # (zero) column and column k always means classes[k].
            model_classes = np.asarray(model.classes_, dtype=int)
            proba = np.zeros((len(x_test), len(classes)), dtype=float)
            for column, code in enumerate(model_classes):
                if 0 <= int(code) < len(classes):
                    proba[:, int(code)] = raw[:, column]

        if proba is not None:
            codes = proba.argmax(axis=1)
            threshold_source = "argmax" if len(classes) > 2 else "fixed_0.5"
        else:
            codes = np.asarray(model.predict(encoded), dtype=int).ravel()
            threshold_source = "argmax"

        return PredictionOutput(
            y_pred=np.asarray([classes[int(c)] for c in codes], dtype=object),
            y_proba=proba,
            y_score=None if proba is None or len(classes) != 2 else proba[:, 1],
            labels=list(classes),
            threshold=0.5 if len(classes) == 2 else None,
            threshold_source=threshold_source,
            metadata={
                "method": self.name,
                "label_vocabulary_source": "train",
                "n_classes": len(classes),
                **{k: meta[k] for k in ("n_train_rows", "n_validation_rows")},
            },
        )

    def _regression_output(self, pipeline, x_test, meta, ctx) -> PredictionOutput:
        model, encoded = _staged_transform(pipeline, x_test)
        y_pred = np.asarray(model.predict(encoded), dtype=float).ravel()
        return PredictionOutput(
            y_pred=y_pred,
            metadata={
                "method": self.name,
                "predictive_distribution": "none: this estimator emits a point prediction "
                                           "only, so CRPS/NLL/interval metrics are skipped "
                                           "rather than approximated",
                **{k: meta[k] for k in ("n_train_rows", "n_validation_rows")},
            },
        )

    # ------------------------------------------------------------ internals
    def _load(self, ctx: MethodContext):
        import joblib

        ckpt_dir = Path(ctx.checkpoint_dir)
        pipeline = joblib.load(ckpt_dir / _MODEL_FILE)
        with (ckpt_dir / _META_FILE).open("r", encoding="utf-8") as f:
            meta = json.load(f)
        return pipeline, meta

    def _preprocessor(self, schema) -> ColumnTransformer:
        """Train-fitted encoding. Unknown test categories map to the zero row."""
        numerical = numerical_feature_columns(schema)
        categorical = categorical_feature_columns(schema)
        if self.native_categorical:
            # The estimator consumes raw categories (CatBoost's ordered target
            # statistics are the reason to use it at all). Stringifying inside the
            # pipeline keeps fit and predict on the same representation; doing it
            # only at fit time is how the two silently drift apart.
            return ColumnTransformer(
                transformers=[
                    ("num", SimpleImputer(strategy="median"), numerical),
                    ("cat", Pipeline([
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("stringify", FunctionTransformer(_as_strings)),
                    ]), categorical),
                ],
                remainder="drop",
            )
        try:
            one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:  # scikit-learn < 1.2
            one_hot = OneHotEncoder(handle_unknown="ignore", sparse=False)
        return ColumnTransformer(
            transformers=[
                ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                  ("scale", StandardScaler())]), numerical),
                ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                  ("encode", one_hot)]), categorical),
            ],
            remainder="drop",
        )


def _as_strings(values):
    """Cast every cell to ``str`` (used for estimators with native categories)."""
    return np.asarray(values, dtype=object).astype(str)


def _staged_transform(pipeline, x):
    """Encode with the fitted preprocessor and return it alongside the estimator.

    Inference deliberately does not go through ``Pipeline.predict``. Third-party
    estimators that only partially implement the sklearn estimator API (CatBoost
    has no ``__sklearn_tags__``) crash inside Pipeline's fitted-state check even
    though the estimator itself predicts fine. Calling the two stages directly
    is the same computation without depending on that machinery.
    """
    preprocessor = pipeline.named_steps["preprocess"]
    return pipeline.named_steps["model"], preprocessor.transform(x)


def _task_type(schema) -> str:
    if schema.task_type:
        return schema.task_type
    return "regression" if schema.target_col in schema.numerical_cols else "classification"


def _split_indices(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic train/validation indices, carved out of the TRAIN rows only."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_valid = int(round(n * VALIDATION_FRACTION))
    if n_valid < 1 or n - n_valid < 2:
        return order, np.array([], dtype=int)
    return order[n_valid:], order[:n_valid]


def _train_target_scale(y: pd.Series) -> float | None:
    from otbench.metrics.stats import robust_scale

    scale, _ = robust_scale(pd.to_numeric(y, errors="coerce").dropna().to_numpy(float))
    return float(scale) if scale > 0 else None

"""Shared scaffolding for imputers whose state is fitted on the train split.

Every imputer here follows the same contract: fit whatever statistics or models
it needs on the *clean training* frame, checkpoint them, and then apply them to
a masked test frame without ever refitting on the test data. That is what makes
these methods inductive, and it is the difference between an imputer and a
transductive completion of the evaluation matrix.

The base class also enforces the invariant every imputation metric depends on:
**observed cells are never modified**. A method may produce a full completed
frame internally, but only the masked cells are taken from it. Without that
rule, a method could improve its downstream utility score by "cleaning" the
cells it was not asked to touch.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import checkpoints as ckpt_utils
from .base import ImputeContext, TabularMethod, Task, TrainContext
from .seeding import seed_everything

_STATE_FILE = "imputer.joblib"
_META_FILE = "imputer.json"


class TrainFittedImputer(TabularMethod):
    """Base for imputers with a train-fitted state and a deterministic apply step."""

    tasks = frozenset({Task.IMPUTATION})
    trainable = True

    # ------------------------------------------------------------- subclass
    def fit_state(self, train: pd.DataFrame, schema, seed: int) -> Any:
        """Return the picklable object that :meth:`apply_state` needs."""
        raise NotImplementedError

    def apply_state(self, state: Any, masked: pd.DataFrame, schema, seed: int) -> pd.DataFrame:
        """Return a fully completed frame; only masked cells will be kept."""
        raise NotImplementedError

    def state_metadata(self, state: Any) -> dict:
        return {}

    # ---------------------------------------------------------------- train
    def train(self, ctx: TrainContext) -> None:
        import joblib

        seed_everything(ctx.train_seed)
        schema = ctx.bundle.schema
        train = ctx.bundle.train.reset_index(drop=True)[schema.all_columns]
        state = self.fit_state(train, schema, int(ctx.train_seed))

        ckpt_dir = Path(ctx.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(state, ckpt_dir / _STATE_FILE)
        with (ckpt_dir / _META_FILE).open("w", encoding="utf-8") as f:
            json.dump({
                "columns": list(schema.all_columns),
                "fitted_on": "train",
                **self.state_metadata(state),
            }, f, indent=2, default=str)
        ckpt_utils.write_metadata(
            ckpt_dir, self.name, ctx.bundle, self.config, ctx.train_seed,
            artifacts={"state": _STATE_FILE, "meta": _META_FILE},
            runtime_packages=self.checkpoint_runtime_packages,
        )

    # --------------------------------------------------------------- impute
    def impute(self, ctx: ImputeContext) -> pd.DataFrame:
        import joblib

        seed_everything(ctx.seed)
        state = joblib.load(Path(ctx.checkpoint_dir) / _STATE_FILE)
        schema = ctx.bundle.schema
        masked = ctx.masked_test.reset_index(drop=True)
        completed = self.apply_state(state, masked[schema.all_columns], schema, int(ctx.seed))
        return merge_masked_only(masked, completed, ctx.mask.reset_index(drop=True))


def merge_masked_only(masked: pd.DataFrame, completed: pd.DataFrame,
                      mask: pd.DataFrame) -> pd.DataFrame:
    """Take imputed values at masked cells and the original values everywhere else.

    This is the invariant the whole imputation task rests on. It is enforced here
    rather than trusted to each method, because a method that quietly rewrites an
    observed cell would score better on every downstream metric while solving a
    different problem.
    """
    out = masked.copy()
    for column in out.columns:
        if column not in completed.columns or column not in mask.columns:
            continue
        rows = mask[column].fillna(False).to_numpy(dtype=bool)
        if rows.any():
            values = completed[column].to_numpy()
            out.loc[rows, column] = values[rows]
    return out


def encode_for_neighbors(train: pd.DataFrame, schema) -> tuple[Any, list[str]]:
    """A train-fitted numeric encoding usable for distance-based imputation."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numerical = [c for c in schema.numerical_cols if c in train.columns]
    categorical = [c for c in schema.categorical_cols if c in train.columns]
    try:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse=False)
    transformer = ColumnTransformer(
        [("num", StandardScaler(), numerical), ("cat", one_hot, categorical)],
        remainder="drop",
    )
    transformer.fit(train)
    return transformer, numerical + categorical


def column_types(schema, columns) -> tuple[list[str], list[str]]:
    numerical = [c for c in columns if c in set(schema.numerical_cols)]
    categorical = [c for c in columns if c in set(schema.categorical_cols)]
    return numerical, categorical


def initial_fill(frame: pd.DataFrame, statistics: dict[str, Any]) -> pd.DataFrame:
    """Seed an iterative imputer with train-fitted per-column statistics."""
    out = frame.copy()
    for column, value in statistics.items():
        if column in out.columns:
            out[column] = out[column].fillna(value)
    return out


def train_statistics(train: pd.DataFrame, schema) -> dict[str, Any]:
    """Median for numerical columns, mode for categorical. Train only."""
    statistics: dict[str, Any] = {}
    for column in train.columns:
        if column in set(schema.numerical_cols):
            values = pd.to_numeric(train[column], errors="coerce").dropna()
            statistics[column] = float(values.median()) if len(values) else 0.0
        else:
            modes = train[column].astype(str).mode()
            statistics[column] = str(modes.iloc[0]) if len(modes) else ""
    return statistics

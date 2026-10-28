# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Shared dataframe utilities used across methods and task runners."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from otbench.tabdata import EvalSchema


@dataclass(frozen=True)
class FeatureTargetSplit:
    """Feature/target split with row order preserved."""

    X: pd.DataFrame
    y: pd.Series


def feature_columns(schema: EvalSchema) -> list[str]:
    """Return modelled feature columns, excluding target and id columns."""
    return list(schema.feature_cols)


def numerical_feature_columns(schema: EvalSchema) -> list[str]:
    features = set(feature_columns(schema))
    return [col for col in schema.numerical_cols if col in features]


def categorical_feature_columns(schema: EvalSchema) -> list[str]:
    features = set(feature_columns(schema))
    return [col for col in schema.categorical_cols if col in features]


def split_features_target(df: pd.DataFrame, schema: EvalSchema) -> FeatureTargetSplit:
    target = schema.target_col
    if target not in df.columns:
        raise KeyError(f"Target column {target!r} is missing from dataframe.")
    return FeatureTargetSplit(X=df[feature_columns(schema)].copy(), y=df[target].copy())


def feature_only(df: pd.DataFrame, schema: EvalSchema) -> pd.DataFrame:
    return df[feature_columns(schema)].copy()


def reset_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize row identity to 0-based positions used by mask metadata."""
    return df.reset_index(drop=True)


def append_target(
    imputed_features: pd.DataFrame,
    original_test: pd.DataFrame,
    schema: EvalSchema,
) -> pd.DataFrame:
    """Return a full test-shaped dataframe with the true target copied back."""
    columns = list(original_test.columns)
    target = schema.target_col
    base = original_test.reset_index(drop=True)
    frame = pd.DataFrame(index=base.index)
    for col in columns:
        if col == target:
            frame[col] = base[col]
        elif col in imputed_features.columns:
            frame[col] = imputed_features.reset_index(drop=True)[col]
        else:
            frame[col] = base[col]
    return frame[columns]


def fill_feature_nans(
    train_features: pd.DataFrame,
    features: pd.DataFrame,
    schema: EvalSchema,
) -> pd.DataFrame:
    """Fill feature NaNs using train means/modes, without touching the target."""
    out = features.copy()
    for col in numerical_feature_columns(schema):
        if col not in out.columns:
            continue
        values = pd.to_numeric(train_features[col], errors="coerce")
        fill = values.mean()
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0 if pd.isna(fill) else fill)
    for col in categorical_feature_columns(schema):
        if col not in out.columns:
            continue
        mode = train_features[col].mode(dropna=True)
        fill = mode.iloc[0] if len(mode) else "__MISSING__"
        out[col] = out[col].fillna(fill)
    return out


def target_placeholder(train: pd.DataFrame, schema: EvalSchema) -> Any:
    """Train mean/mode placeholder for the target column."""
    target = schema.target_col
    if target in schema.numerical_cols:
        value = pd.to_numeric(train[target], errors="coerce").mean()
        return 0.0 if pd.isna(value) else float(value)
    mode = train[target].mode(dropna=True)
    return _to_python_scalar(mode.iloc[0]) if len(mode) else "__MISSING__"


def fill_values_from_train(train: pd.DataFrame, schema: EvalSchema) -> dict:
    """Per-column back-fill stats (mode/mean from train) for schema alignment."""
    cat = set(schema.categorical_cols)
    fill: dict = {}
    for col in schema.all_columns:
        if col not in train.columns:
            fill[col] = "__MISSING__" if col in cat else 0.0
            continue
        if col in cat:
            mode = train[col].astype(str).mode()
            fill[col] = mode.iloc[0] if len(mode) else "__MISSING__"
        else:
            series = pd.to_numeric(train[col], errors="coerce")
            mean = series.mean()
            fill[col] = 0.0 if pd.isna(mean) else float(mean)
    return fill


def align_generated_frame(
    gen: pd.DataFrame,
    schema: EvalSchema,
    fill_values: dict,
) -> pd.DataFrame:
    """Align a generated frame to the schema: column set/order + basic dtypes."""
    out = pd.DataFrame(index=gen.reset_index(drop=True).index)
    gen = gen.reset_index(drop=True)
    num = set(schema.numerical_cols)
    for col in schema.all_columns:
        if col in gen.columns:
            values = gen[col]
        else:
            values = pd.Series([fill_values.get(col)] * len(gen))
        if col in num:
            values = pd.to_numeric(values, errors="coerce")
            fallback = fill_values.get(col, 0.0)
            values = values.fillna(fallback if isinstance(fallback, (int, float)) else 0.0)
        out[col] = values
    return out[schema.all_columns]


# --- hashing (checkpoint reuse contracts) ----------------------------------
def dataframe_hash(df: pd.DataFrame) -> str:
    normalized = df.reset_index(drop=True)
    dtypes = json.dumps([(str(c), str(t)) for c, t in normalized.dtypes.items()]).encode("utf-8")
    payload = dtypes + b"\0" + normalized.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def schema_hash(schema: EvalSchema) -> str:
    payload = {
        "target_col": schema.target_col,
        "numerical_cols": list(schema.numerical_cols),
        "categorical_cols": list(schema.categorical_cols),
        "id_cols": list(schema.id_cols),
        "task_type": schema.task_type,
        "ci_test": schema.ci_test,
        "constraints": list(schema.constraints),
        "category_domains": dict(schema.category_domains),
        "sensitive_cols": list(schema.sensitive_cols),
        "quasi_identifier_cols": list(schema.quasi_identifier_cols),
        "group_cols": list(schema.group_cols),
        "immutable_cols": list(getattr(schema, "immutable_cols", []) or []),
        "ood_split": schema.ood_split,
        "csv_format": dict(getattr(schema, "csv_format", {}) or {}),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def column_order_hash(columns: Iterable[str]) -> str:
    raw = json.dumps(list(columns), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


class CategoryCodec:
    """Train-fitted categorical value <-> ordinal id mapping."""

    def __init__(self, mappings: dict[str, list[Any]], modes: dict[str, Any]) -> None:
        self.mappings = mappings
        self.modes = modes
        self._to_id = {
            col: {value: idx for idx, value in enumerate(values)}
            for col, values in mappings.items()
        }

    @classmethod
    def fit(cls, train: pd.DataFrame, categorical_cols: Iterable[str]) -> "CategoryCodec":
        mappings: dict[str, list[Any]] = {}
        modes: dict[str, Any] = {}
        for col in categorical_cols:
            values = pd.Series(train[col].dropna().unique()).sort_values(kind="stable").tolist()
            if not values:
                values = ["__MISSING__"]
            mappings[col] = values
            mode = train[col].mode(dropna=True)
            modes[col] = mode.iloc[0] if len(mode) else values[0]
        return cls(mappings, modes)

    def encode(self, df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
        out = df.copy()
        cols = list(columns) if columns is not None else list(self.mappings)
        for col in cols:
            mapping = self._to_id[col]
            fallback = mapping[self.modes[col]]
            out[col] = out[col].map(mapping).fillna(fallback).astype(int)
        return out

    def decode(self, df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
        out = df.copy()
        cols = list(columns) if columns is not None else list(self.mappings)
        for col in cols:
            values = self.mappings[col]
            ids = pd.to_numeric(out[col], errors="coerce").fillna(0).round().astype(int)
            ids = ids.clip(0, len(values) - 1)
            out[col] = ids.map(lambda idx: values[int(idx)])
        return out


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value

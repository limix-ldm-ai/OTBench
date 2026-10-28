"""Train-fitted ordinal frame codec shared by native conditional adapters."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, Sequence
import numpy as np
import pandas as pd

@dataclass
class _FrameCodec:
    """Train-fitted dataframe <-> float-matrix codec used by the extension."""

    columns: list[str]
    categorical: frozenset[str]
    category_values: dict[str, list[Any]]
    category_to_code: dict[str, dict[Any, int]]
    fills: dict[str, Any]
    constant_values: dict[str, Any]

    @classmethod
    def fit(
        cls,
        train: pd.DataFrame,
        columns: Iterable[str],
        categorical: Iterable[str],
    ) -> "_FrameCodec":
        columns = list(columns)
        categorical = frozenset(c for c in categorical if c in columns)
        category_values: dict[str, list[Any]] = {}
        category_to_code: dict[str, dict[Any, int]] = {}
        fills: dict[str, Any] = {}

        for column in columns:
            series = train[column]
            if column in categorical:
                values = series.dropna().drop_duplicates().tolist()
                if not values:
                    values = ["__MISSING__"]
                try:
                    mode = series.mode(dropna=True)
                    fill = mode.iloc[0] if len(mode) else values[0]
                except TypeError:
                    # Mixed incomparable Python scalar types can make pandas'
                    # mode sorting fail. The first train value is deterministic.
                    fill = values[0]
                category_values[column] = values
                category_to_code[column] = {
                    value: index for index, value in enumerate(values)
                }
                fills[column] = fill
            else:
                numeric = pd.to_numeric(series, errors="coerce").replace(
                    [np.inf, -np.inf], np.nan
                )
                mean = numeric.mean()
                fills[column] = 0.0 if pd.isna(mean) else float(mean)

        provisional = cls(
            columns=columns,
            categorical=categorical,
            category_values=category_values,
            category_to_code=category_to_code,
            fills=fills,
            constant_values={},
        )
        encoded = provisional.encode(train[columns], preserve_missing=False)
        constant_values: dict[str, Any] = {}
        for index, column in enumerate(columns):
            unique = np.unique(encoded[:, index])
            if len(unique) == 1:
                if column in categorical:
                    values = category_values[column]
                    code = int(np.clip(round(float(unique[0])), 0, len(values) - 1))
                    constant_values[column] = values[code]
                else:
                    constant_values[column] = float(unique[0])
        provisional.constant_values = constant_values
        return provisional

    @property
    def variable_columns(self) -> list[str]:
        return [c for c in self.columns if c not in self.constant_values]

    def categorical_indices(self, columns: Iterable[str]) -> list[int]:
        return [i for i, c in enumerate(columns) if c in self.categorical]

    def cardinality(self, column: str) -> int:
        return len(self.category_values[column])

    def encode(
        self,
        frame: pd.DataFrame,
        columns: Iterable[str] | None = None,
        *,
        preserve_missing: bool,
    ) -> np.ndarray:
        columns = list(columns) if columns is not None else self.columns
        out = np.empty((len(frame), len(columns)), dtype=np.float32)
        for index, column in enumerate(columns):
            series = frame[column]
            original_missing = series.isna().to_numpy()
            if column in self.categorical:
                mapping = self.category_to_code[column]
                fallback = mapping[self.fills[column]]
                encoded = series.map(mapping).fillna(fallback).to_numpy(dtype=np.float32)
            else:
                encoded_series = pd.to_numeric(series, errors="coerce").replace(
                    [np.inf, -np.inf], np.nan
                )
                encoded = encoded_series.fillna(self.fills[column]).to_numpy(dtype=np.float32)
            if preserve_missing:
                encoded[original_missing] = np.nan
            out[:, index] = encoded
        return out

    def decode(self, values: Any, columns: Iterable[str]) -> pd.DataFrame:
        columns = list(columns)
        array = _as_numpy(values)
        if array.ndim != 2 or array.shape[1] != len(columns):
            raise ValueError(
                f"decoded matrix has shape {array.shape}; expected (*, {len(columns)})"
            )
        out = pd.DataFrame(index=range(len(array)))
        for index, column in enumerate(columns):
            vector = np.asarray(array[:, index], dtype=float)
            if column in self.categorical:
                categories = self.category_values[column]
                fallback = self.category_to_code[column][self.fills[column]]
                vector = np.nan_to_num(
                    vector,
                    nan=float(fallback),
                    posinf=float(fallback),
                    neginf=float(fallback),
                )
                codes = np.clip(np.rint(vector).astype(int), 0, len(categories) - 1)
                out[column] = [categories[int(code)] for code in codes]
            else:
                out[column] = vector
        return out[columns]


def _as_numpy(values: Any) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        values = values.numpy()
    return np.asarray(values)


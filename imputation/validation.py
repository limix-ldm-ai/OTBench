"""The completion contract enforced before a benchmark result is committed."""
from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_completion(clean: pd.DataFrame, completed: pd.DataFrame, schema) -> pd.DataFrame:
    """Restore declared nonmodelled IDs and normalize column order only."""
    if not isinstance(completed, pd.DataFrame):
        raise ValueError("imputation output must be a pandas DataFrame")
    expected = set(clean.columns)
    missing = expected - set(completed.columns)
    if (len(completed) != len(clean) or completed.columns.duplicated().any()
            or missing - set(schema.id_cols) or set(completed.columns) - expected):
        raise ValueError(
            "imputation output shape/columns differ from the requested table: "
            f"expected {clean.shape} {list(clean.columns)}, "
            f"got {completed.shape} {list(completed.columns)}"
        )
    out = completed.reset_index(drop=True).copy()
    for column in missing:
        out[column] = clean[column].reset_index(drop=True)
    return out[list(clean.columns)]


def validate_completion(clean: pd.DataFrame, completed: pd.DataFrame,
                        mask: pd.DataFrame, schema, *, check_observed: bool = True) -> None:
    completed = normalize_completion(clean, completed, schema)
    if mask.shape != clean.shape or list(mask.columns) != list(clean.columns):
        raise ValueError("imputation mask is not aligned with the input table")
    truth, prediction = clean.reset_index(drop=True), completed.reset_index(drop=True)
    hidden = mask.reset_index(drop=True).fillna(False).astype(bool)
    if schema.target_col in hidden and hidden[schema.target_col].any():
        raise ValueError("imputation mask must not hide the supervised target")
    for column in clean.columns:
        required = hidden[column] & truth[column].notna()
        values = prediction.loc[required, column]
        if column in schema.numerical_cols:
            parsed = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
            invalid = int((~np.isfinite(parsed)).sum())
        else:
            invalid = int(values.isna().sum())
        if invalid:
            raise ValueError(f"imputation output {column!r} has {invalid} invalid masked predictions")
        if check_observed:
            observed = ~hidden[column] & truth[column].notna()
            try:
                pd.testing.assert_series_equal(
                    prediction.loc[observed, column].reset_index(drop=True),
                    truth.loc[observed, column].reset_index(drop=True),
                    check_dtype=False, check_names=False, check_exact=False,
                    rtol=1e-12, atol=1e-12,
                )
            except AssertionError as exc:
                raise ValueError(f"imputation output changed observed values in {column!r}") from exc

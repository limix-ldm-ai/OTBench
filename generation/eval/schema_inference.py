# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Infer a generation-evaluation schema from a training frame."""
from __future__ import annotations

import numpy as np
import pandas as pd

from otbench.tabdata.schema import EvalSchema

__all__ = ["infer_schema_from_frames", "infer_task_type"]


def infer_schema_from_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    gen_df: pd.DataFrame,
) -> EvalSchema:
    """Infer column roles and a target task from the training frame.

    The last column is treated as target. Non-target columns are treated as numerical
    by default; for classification tasks the target column is categorical.
    """
    del test_df, gen_df

    cols = list(train_df.columns)
    if not cols:
        raise ValueError("Cannot infer schema from an empty-column CSV.")

    target_col = cols[-1]
    task_type = infer_task_type(train_df, target_col)

    categorical_cols: list[str] = []
    numerical_cols = list(cols)
    if task_type == "regression":
        if target_col not in numerical_cols:
            numerical_cols.append(target_col)
    else:
        if target_col in numerical_cols:
            numerical_cols.remove(target_col)
        categorical_cols.append(target_col)

    return EvalSchema(
        target_col=target_col,
        numerical_cols=[col for col in cols if col in set(numerical_cols)],
        categorical_cols=[col for col in cols if col in set(categorical_cols)],
        task_type=task_type,
    )


def infer_task_type(train_df: pd.DataFrame, target_col: str) -> str:
    y = train_df[target_col]
    if (
        pd.api.types.is_object_dtype(y)
        or pd.api.types.is_bool_dtype(y)
        or pd.api.types.is_categorical_dtype(y)
    ):
        return "binclass"

    y_num = pd.to_numeric(y, errors="coerce").dropna()
    if len(y_num) == 0:
        return "binclass"

    integer_like = np.isclose(y_num.values, np.floor(y_num.values), atol=1e-6).mean() > 0.999
    if integer_like and y_num.nunique() <= 20:
        return "binclass"
    return "regression"


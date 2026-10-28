# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Raw vs. aligned generated artifacts, and the repair log between them.

A generator's output is written twice:

``<name>_gen_raw.csv``
    Byte-for-byte what the method produced. Never repaired, never re-written.
    Validity, missingness, parse-failure, exact-copy and privacy metrics read
    *only* this file — repairing before measuring would silently turn a broken
    generator into a good one.

``<name>_gen.csv``
    The raw frame coerced onto the schema (column set/order, numeric parsing,
    train-value fill). Downstream-model metrics (C2ST, TSTR, utility) read this,
    because a scikit-learn pipeline cannot consume a frame with missing columns.

:func:`align_generated` returns the aligned frame together with a structured
report of every repair it applied, so the number of coercions is itself a
reported quantity rather than an invisible side effect.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from otbench.tabdata import EvalSchema


@dataclass
class AlignmentReport:
    """Everything that had to change to make a raw frame schema-shaped."""

    n_rows_raw: int = 0
    n_rows_aligned: int = 0
    missing_columns: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)
    duplicated_columns: list[str] = field(default_factory=list)
    # column -> number of cells that failed numeric parsing in the raw frame
    parse_failures: dict[str, int] = field(default_factory=dict)
    # column -> number of NaN cells filled with a train statistic
    filled_cells: dict[str, int] = field(default_factory=dict)
    # column -> number of values outside the train category domain
    unknown_categories: dict[str, int] = field(default_factory=dict)

    @property
    def n_repairs(self) -> int:
        return (
            len(self.missing_columns)
            + sum(self.parse_failures.values())
            + sum(self.filled_cells.values())
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows_raw": self.n_rows_raw,
            "n_rows_aligned": self.n_rows_aligned,
            "missing_columns": self.missing_columns,
            "extra_columns": self.extra_columns,
            "duplicated_columns": self.duplicated_columns,
            "parse_failures": self.parse_failures,
            "filled_cells": self.filled_cells,
            "unknown_categories": self.unknown_categories,
            "n_repairs": self.n_repairs,
        }


def align_generated(
    raw: pd.DataFrame,
    schema: EvalSchema,
    fill_values: dict,
) -> tuple[pd.DataFrame, AlignmentReport]:
    """Coerce ``raw`` onto ``schema`` and report every repair performed."""
    raw = raw.reset_index(drop=True)
    report = AlignmentReport(n_rows_raw=len(raw))
    report.duplicated_columns = sorted(raw.columns[raw.columns.duplicated()].unique().tolist())
    if report.duplicated_columns:
        raw = raw.loc[:, ~raw.columns.duplicated()]

    schema_cols = schema.all_columns
    report.missing_columns = [c for c in schema_cols if c not in raw.columns]
    report.extra_columns = [c for c in raw.columns if c not in schema_cols]

    numerical = set(schema.numerical_cols)
    out = pd.DataFrame(index=raw.index)
    for col in schema_cols:
        if col in raw.columns:
            values = raw[col]
        else:
            values = pd.Series([fill_values.get(col)] * len(raw), index=raw.index)

        if col in numerical:
            parsed = pd.to_numeric(values, errors="coerce")
            # A cell that was non-null but did not parse is a parse failure;
            # a cell that was already null is plain missingness.
            report.parse_failures[col] = int((values.notna() & parsed.isna()).sum())
            n_missing = int(parsed.isna().sum())
            if n_missing and col in raw.columns:
                report.filled_cells[col] = n_missing
            fallback = fill_values.get(col, 0.0)
            out[col] = parsed.fillna(fallback if isinstance(fallback, (int, float)) else 0.0)
        else:
            domain = schema.category_domain(col)
            if domain is not None:
                allowed = {str(v) for v in domain}
                as_str = values.astype("string")
                unknown = as_str.notna() & ~as_str.isin(allowed)
                report.unknown_categories[col] = int(unknown.sum())
            n_missing = int(values.isna().sum())
            if n_missing and col in raw.columns:
                report.filled_cells[col] = n_missing
            out[col] = values.fillna(fill_values.get(col, "__MISSING__"))

    report.parse_failures = {k: v for k, v in report.parse_failures.items() if v}
    report.unknown_categories = {k: v for k, v in report.unknown_categories.items() if v}
    report.n_rows_aligned = len(out)
    return out[schema_cols], report

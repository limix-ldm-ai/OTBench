# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""G01: schema validity and constraint violations, measured on the RAW output.

Everything here reads :attr:`EvalFrames.gen_raw`. The aligned artifact has had
its unparseable cells coerced, its missing columns filled and its nulls
back-filled with train statistics, so measuring validity on it would report the
alignment step's quality rather than the generator's.

Declared and empirical constraints are reported separately on purpose. An
empirical range is a description of the training sample: a generated value just
outside it may be perfectly legitimate. A declared range comes from the data
definition, and a value outside it is genuinely invalid.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from otbench.generation.eval.data import EvalFrames
from otbench.metrics.catalog import MetricResult


def evaluate_validity(frames: EvalFrames) -> list[MetricResult]:
    names = [
        "generation.validity.schema_type_violation_rate",
        "generation.validity.unknown_category_rate",
        "generation.validity.empirical_range_violation_rate",
        "generation.validity.declared_constraint_violation_rate",
        "generation.validity.missing_cell_rate",
        "generation.validity.missing_column_count",
        "generation.validity.row_validity_rate",
    ]
    if not frames.has_raw:
        reason = (
            "no raw generator artifact was supplied; validity measured on the aligned "
            "frame would be vacuously perfect"
        )
        return [MetricResult.skipped(name, reason) for name in names]

    schema = frames.schema
    raw = frames.gen_raw.reset_index(drop=True)
    present = [c for c in schema.all_columns if c in raw.columns]
    missing_columns = [c for c in schema.all_columns if c not in raw.columns]
    n_rows = len(raw)

    # A row is valid only if it passes every applicable check.
    row_ok = np.ones(n_rows, dtype=bool) if n_rows else np.zeros(0, dtype=bool)
    if missing_columns:
        row_ok[:] = False

    # Every rate is divided by the cells it could apply to, not by the whole
    # table: an unknown-category rate over numerical cells too would shrink
    # simply because the dataset has many numerical columns.
    type_violations = 0
    unknown_categories = 0
    range_violations = 0
    missing_cells = 0
    checked_cells = 0
    numerical_cells = 0
    categorical_cells = 0
    ranged_cells = 0
    per_column: dict[str, dict[str, Any]] = {}

    empirical_ranges = {
        c["column"]: c
        for c in schema.constraints_of("empirical")
        if c["kind"] == "range"
    }

    for col in present:
        values = raw[col]
        null_mask = values.isna()
        column_record: dict[str, Any] = {"n_null": int(null_mask.sum())}
        missing_cells += int(null_mask.sum())
        checked_cells += n_rows
        row_ok &= ~null_mask.to_numpy()

        if col in set(schema.numerical_cols):
            numerical_cells += n_rows
            parsed = pd.to_numeric(values, errors="coerce")
            bad_type = (~null_mask) & parsed.isna()
            type_violations += int(bad_type.sum())
            row_ok &= ~bad_type.to_numpy()
            column_record["n_type_violations"] = int(bad_type.sum())

            bounds = empirical_ranges.get(col)
            if bounds is not None:
                ranged_cells += n_rows
                outside = parsed.notna() & (
                    (parsed < bounds["min"]) | (parsed > bounds["max"])
                )
                range_violations += int(outside.sum())
                column_record["n_outside_train_range"] = int(outside.sum())
        else:
            categorical_cells += n_rows
            domain = schema.category_domain(col)
            if domain is not None:
                allowed = {str(v) for v in domain}
                unknown = (~null_mask) & ~values.astype("string").isin(allowed)
                unknown_categories += int(unknown.sum())
                row_ok &= ~unknown.to_numpy()
                column_record["n_unknown_categories"] = int(unknown.sum())
        per_column[col] = column_record

    declared = schema.constraints_of("declared")
    declared_result = _evaluate_declared_constraints(raw, declared, row_ok)

    results = [
        _rate(
            "generation.validity.schema_type_violation_rate",
            type_violations, numerical_cells,
            "the schema declares no numerical column to type-check",
            details={"per_column": per_column},
        ),
        _rate(
            "generation.validity.unknown_category_rate",
            unknown_categories, categorical_cells,
            "the schema declares no categorical column",
        ),
        _rate(
            "generation.validity.empirical_range_violation_rate",
            range_violations, ranged_cells,
            "no empirical range was derived from the train split",
            extra={"note": "descriptive: the train range is a sample, not a declared bound"},
        ),
        _rate(
            "generation.validity.missing_cell_rate",
            missing_cells, checked_cells,
            "the raw output has no cells to check",
        ),
        MetricResult.ok(
            "generation.validity.missing_column_count",
            float(len(missing_columns)), n=len(frames.schema.all_columns),
            details={"missing_columns": missing_columns},
        ),
        (MetricResult.ok(
            "generation.validity.row_validity_rate", float(row_ok.mean()), n=n_rows,
        ) if n_rows else MetricResult.skipped(
            "generation.validity.row_validity_rate", "the raw output contains no row"
        )),
        declared_result,
    ]
    return results


def _rate(name: str, failures: int, denominator: int, empty_reason: str, **kwargs) -> MetricResult:
    """A violation rate that skips rather than dividing by an empty denominator."""
    if denominator == 0:
        return MetricResult.skipped(name, empty_reason)
    return MetricResult.ok(name, failures / denominator, n=denominator, **kwargs)


def _evaluate_declared_constraints(
    raw: pd.DataFrame,
    constraints: list[dict],
    row_ok: np.ndarray,
) -> MetricResult:
    name = "generation.validity.declared_constraint_violation_rate"
    if not constraints:
        return MetricResult.skipped(
            name,
            "the schema declares no constraints; empirical ranges are reported separately "
            "and must not be presented as business rules",
        )
    checks = 0
    failures = 0
    per_constraint = []
    for constraint in constraints:
        mask = _constraint_failures(raw, constraint)
        if mask is None:
            per_constraint.append({"constraint": constraint, "status": "not_applicable"})
            continue
        checks += len(mask)
        failures += int(mask.sum())
        row_ok &= ~mask
        per_constraint.append({
            "constraint": constraint,
            "n_checked": int(len(mask)),
            "n_failed": int(mask.sum()),
        })
    if checks == 0:
        return MetricResult.skipped(name, "no declared constraint applied to the raw output")
    return MetricResult.ok(
        name, failures / checks, n=checks, details={"per_constraint": per_constraint}
    )


def _constraint_failures(raw: pd.DataFrame, constraint: dict) -> np.ndarray | None:
    """Boolean per-row failure mask for one constraint, or ``None`` if inapplicable."""
    kind = constraint["kind"]
    columns = [constraint[key] for key in ("column", "left", "right") if key in constraint]
    columns += list(constraint.get("columns", []))
    if any(col not in raw.columns for col in columns):
        return None

    if kind == "non_null":
        return raw[constraint["column"]].isna().to_numpy()

    if kind == "range":
        values = pd.to_numeric(raw[constraint["column"]], errors="coerce")
        failed = values.isna()
        if constraint.get("min") is not None:
            failed |= values < constraint["min"]
        if constraint.get("max") is not None:
            failed |= values > constraint["max"]
        return failed.to_numpy()

    if kind == "category_set":
        allowed = {str(v) for v in constraint["values"]}
        return (~raw[constraint["column"]].astype("string").isin(allowed)).to_numpy()

    if kind == "dtype":
        values = raw[constraint["column"]]
        if constraint["dtype"] == "categorical":
            return values.isna().to_numpy()
        parsed = pd.to_numeric(values, errors="coerce")
        failed = parsed.isna()
        if constraint["dtype"] == "integer":
            failed |= parsed.notna() & (parsed % 1 != 0)
        return failed.to_numpy()

    if kind == "unique":
        return raw.duplicated(subset=list(constraint["columns"]), keep=False).to_numpy()

    if kind == "inequality":
        left = pd.to_numeric(raw[constraint["left"]], errors="coerce")
        right = pd.to_numeric(raw[constraint["right"]], errors="coerce")
        op = constraint.get("op", "<=")
        satisfied = {
            "<": left < right, "<=": left <= right,
            ">": left > right, ">=": left >= right,
        }[op]
        return (~satisfied.fillna(False)).to_numpy()

    if kind == "sum_to":
        total = sum(pd.to_numeric(raw[c], errors="coerce") for c in constraint["columns"])
        tolerance = float(constraint.get("tol", 1e-6))
        return (~((total - constraint["total"]).abs() <= tolerance).fillna(False)).to_numpy()

    return None

"""Cross-layer tabular schema contract (the JSON glue between all layers).

The materialized ``schema.json`` contains dataset declarations. Runtime-only
empirical fields (training ranges and categorical domains) are derived from the
reference split by :func:`enrich_schema`; they must never use held-out data.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EvalSchema:
    """Declarative column and evaluation contract for one dataset."""

    target_col: str
    numerical_cols: list[str]
    categorical_cols: list[str]
    id_cols: list[str] = field(default_factory=list)
    task_type: str | None = None
    ci_test: str = "chi_square"
    constraints: list[dict[str, Any]] = field(default_factory=list)
    category_domains: dict[str, list[Any]] = field(default_factory=dict)
    sensitive_cols: list[str] = field(default_factory=list)
    quasi_identifier_cols: list[str] = field(default_factory=list)
    group_cols: list[str] = field(default_factory=list)
    immutable_cols: list[str] = field(default_factory=list)
    ood_split: dict[str, Any] | None = None
    csv_format: dict[str, Any] = field(default_factory=dict)

    @property
    def feature_cols(self) -> list[str]:
        excluded = set(self.id_cols) | {self.target_col}
        return [col for col in self.all_columns if col not in excluded]

    @property
    def all_columns(self) -> list[str]:
        cols: list[str] = []
        for col in self.numerical_cols + self.categorical_cols:
            if col not in cols:
                cols.append(col)
        if self.target_col not in cols:
            cols.append(self.target_col)
        return [col for col in cols if col not in set(self.id_cols)]

    def column_type(self, col: str) -> str:
        if col in self.categorical_cols:
            return "categorical"
        if col in self.numerical_cols:
            return "numerical"
        raise KeyError(f"Column {col!r} is not declared in schema.")

    def category_domain(self, col: str) -> list[Any] | None:
        domain = self.category_domains.get(col)
        return list(domain) if domain is not None else None

    def constraints_of(self, source: str) -> list[dict[str, Any]]:
        return [constraint for constraint in self.constraints if constraint.get("source") == source]

    def with_constraints(self, constraints: list[dict[str, Any]]) -> "EvalSchema":
        updated = replace(self, constraints=[dict(value) for value in constraints])
        _validate_optional_fields(updated)
        return updated

    def has_privacy_declaration(self) -> bool:
        return bool(self.sensitive_cols and self.quasi_identifier_cols)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "columns": {
                col: {"sdtype": "numerical"}
                for col in self.numerical_cols
                if col not in set(self.id_cols)
            }
            | {
                col: {"sdtype": "categorical"}
                for col in self.categorical_cols
                if col not in set(self.id_cols)
            }
        }


def schema_from_dict(raw: dict[str, Any]) -> EvalSchema:
    """Parse and validate the dataset schema."""
    unknown = raw.keys() - {item.name for item in fields(EvalSchema)}
    if unknown:
        raise ValueError(f"Unknown schema fields: {sorted(unknown)}")
    target_col = raw.get("target_col")
    if not target_col:
        raise ValueError("schema.json must define `target_col`.")

    numerical_cols = list(raw.get("numerical_cols", []))
    categorical_cols = list(raw.get("categorical_cols", []))
    id_cols = list(raw.get("id_cols", []))
    if target_col not in numerical_cols and target_col not in categorical_cols:
        if raw.get("task_type") == "regression":
            numerical_cols.append(target_col)
        else:
            categorical_cols.append(target_col)

    overlap = set(numerical_cols) & set(categorical_cols)
    if overlap:
        raise ValueError(f"Columns cannot be both numerical and categorical: {sorted(overlap)}")

    schema = EvalSchema(
        target_col=target_col,
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        id_cols=id_cols,
        task_type=raw.get("task_type"),
        ci_test=raw.get("ci_test", "chi_square"),
        constraints=[dict(value) for value in raw.get("constraints", [])],
        category_domains={key: list(value) for key, value in raw.get("category_domains", {}).items()},
        sensitive_cols=list(raw.get("sensitive_cols", [])),
        quasi_identifier_cols=list(raw.get("quasi_identifier_cols", [])),
        group_cols=list(raw.get("group_cols", [])),
        immutable_cols=list(raw.get("immutable_cols", [])),
        ood_split=raw.get("ood_split"),
        csv_format=dict(raw.get("csv_format", {})),
    )
    _validate_optional_fields(schema)
    return schema


def load_schema(path: str | Path) -> EvalSchema:
    """Load a ``schema.json`` without looking at any data split."""
    with Path(path).open("r", encoding="utf-8") as file:
        return schema_from_dict(json.load(file))


def derive_empirical_constraints(train: pd.DataFrame, schema: EvalSchema) -> list[dict[str, Any]]:
    """Describe numerical train support without turning it into a hard rule."""
    constraints: list[dict[str, Any]] = []
    for column in schema.numerical_cols:
        if column not in train.columns:
            continue
        values = pd.to_numeric(train[column], errors="coerce").dropna()
        if values.empty:
            continue
        finite = values[np.isfinite(values.to_numpy(dtype=float))]
        if finite.empty:
            continue
        dtype = "integer" if np.allclose(finite, np.round(finite)) else "float"
        constraints.extend([
            {
                "kind": "range",
                "column": column,
                "min": _python_scalar(finite.min()),
                "max": _python_scalar(finite.max()),
                "source": "empirical",
            },
            {"kind": "dtype", "column": column, "dtype": dtype, "source": "empirical"},
        ])
    return constraints


def enrich_schema(train: pd.DataFrame, schema: EvalSchema) -> EvalSchema:
    """Attach train-derived domains and constraints to a declared schema."""
    missing = [column for column in schema.all_columns if column not in train.columns]
    if missing:
        raise ValueError(f"Training frame is missing schema columns: {missing}")

    # Recompute empirical state for the exact reference split. Declared fields
    # survive unchanged; stale empirical fields must not cross split boundaries.
    constraints = schema.constraints_of("declared") + derive_empirical_constraints(train, schema)
    domains = {key: list(value) for key, value in schema.category_domains.items()}
    for column in schema.categorical_cols:
        if column not in domains:
            values = train[column].dropna().unique().tolist()
            domains[column] = [_python_scalar(value) for value in values]
    return replace(schema, constraints=constraints, category_domains=domains)


def _validate_optional_fields(schema: EvalSchema) -> None:
    known = set(schema.numerical_cols) | set(schema.categorical_cols) | set(schema.id_cols)
    references = {
        "sensitive_cols": schema.sensitive_cols,
        "quasi_identifier_cols": schema.quasi_identifier_cols,
        "group_cols": schema.group_cols,
        "immutable_cols": schema.immutable_cols,
        "category_domains": schema.category_domains,
    }
    for field_name, columns in references.items():
        unknown = set(columns) - known
        if unknown:
            raise ValueError(f"{field_name} references unknown columns: {sorted(unknown)}")
    invalid_domains = set(schema.category_domains) - set(schema.categorical_cols)
    if invalid_domains:
        raise ValueError(f"category_domains must reference categorical columns: {sorted(invalid_domains)}")
    for constraint in schema.constraints:
        column = constraint.get("column")
        if column is not None and column not in known:
            raise ValueError(f"Constraint references unknown column {column!r}")


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value

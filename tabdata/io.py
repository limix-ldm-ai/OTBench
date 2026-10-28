"""Schema-aware CSV storage with an explicit missing-value representation.

An explicit missing token and escape prefix preserve empty and ``NA`` labels.
Plain CSV files use the missing-value semantics declared by their schema.
"""
from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CSV_FORMAT_VERSION = 2
MISSING_TOKEN = "__OTBENCH_MISSING_V2__"
ESCAPE_PREFIX = "__OTBENCH_ESC_V2__"


def csv_format_for(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "version": CSV_FORMAT_VERSION,
        "missing_token": MISSING_TOKEN,
        "escape_prefix": ESCAPE_PREFIX,
        "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
    }


def _format(schema) -> dict:
    value = getattr(schema, "csv_format", {}) if schema is not None else {}
    if value and value.get("version") != CSV_FORMAT_VERSION:
        raise ValueError(f"Unsupported CSV format version: {value.get('version')!r}")
    return value


def csv_text(frame: pd.DataFrame, schema=None) -> str:
    """Serialize values without repairing or coercing model output."""
    fmt = _format(schema)
    if not fmt:
        return frame.to_csv(index=False)
    missing, escape = fmt["missing_token"], fmt["escape_prefix"]
    out = frame.copy()
    for column in out:
        def escaped(value):
            return isinstance(value, str) and (value == missing or value.startswith(escape))
        # Mapping an unchanged nullable integer Series can infer float64 and
        # round large integers. Leave unaffected columns in their native dtype.
        if any(escaped(value) for value in out[column].array):
            out[column] = pd.Series(
                [escape + value if escaped(value) else value for value in out[column].array],
                index=out.index, dtype=object,
            )
    return out.to_csv(index=False, na_rep=missing)


def write_frame_csv(frame: pd.DataFrame, path: str | Path, schema=None) -> None:
    Path(path).write_text(csv_text(frame, schema), encoding="utf-8")


def _integer_series(values: pd.Series, dtype: str) -> pd.Series:
    """Parse integer tokens before any float conversion, including nullable values."""
    parsed = []
    for value in values:
        if pd.isna(value):
            parsed.append(pd.NA)
            continue
        try:
            decimal = Decimal(str(value))
        except InvalidOperation as exc:
            raise ValueError("not an integer token") from exc
        if not decimal.is_finite() or decimal != decimal.to_integral_value():
            raise ValueError("not an exact integer")
        parsed.append(int(decimal))
    numpy_dtype = dtype.lower()
    bounds = np.iinfo(np.dtype(numpy_dtype))
    if any(value is not pd.NA and not bounds.min <= value <= bounds.max for value in parsed):
        raise ValueError(f"integer outside {dtype} range")
    nullable_dtype = ("UInt" if numpy_dtype.startswith("uint") else "Int") + str(bounds.bits)
    target_dtype = nullable_dtype if any(value is pd.NA for value in parsed) else dtype
    return pd.Series(pd.array(parsed, dtype=target_dtype), index=values.index)


def _numeric_values(values: pd.Series, dtype: str, *, strict: bool,
                    path: Path, column: str) -> pd.Series:
    if dtype.lower().startswith(("int", "uint")):
        try:
            return _integer_series(values, dtype)
        except (ValueError, TypeError, OverflowError):
            if strict:
                raise ValueError(f"{path}: {column!r} contains values incompatible with {dtype}") from None
            # Preserve fractional/invalid raw output and exact large integers.
            # Coercing the whole column to float64 would modify valid cells too.
            parsed = []
            for value in values:
                try:
                    decimal = Decimal(str(value)) if pd.notna(value) else None
                    if decimal is None:
                        parsed.append(np.nan)
                    elif decimal.is_finite() and decimal == decimal.to_integral_value():
                        parsed.append(int(decimal))
                    else:
                        parsed.append(value)
                except InvalidOperation:
                    parsed.append(value)
            return pd.Series(parsed, index=values.index, dtype=object)
    parsed = pd.to_numeric(values, errors="coerce")
    invalid = values.notna() & parsed.isna()
    if strict and invalid.any():
        raise ValueError(f"{path}: {column!r} has {int(invalid.sum())} invalid numeric values")
    if invalid.any():
        return values
    if strict and dtype.lower().startswith("float"):
        return pd.Series(pd.array(parsed, dtype=dtype), index=values.index)
    return parsed


def _category_values(values: pd.Series, domain: list, dtype: str, *, strict: bool,
                     path: Path, column: str) -> pd.Series:
    present_domain = [value for value in domain if not pd.isna(value)]
    mapping = {str(value): value for value in present_domain}
    if len(mapping) != len(present_domain):
        raise ValueError(f"{path}: category domain of {column!r} has ambiguous CSV spellings")
    all_known = bool(mapping) and all(str(value) in mapping for value in values if pd.notna(value))
    if mapping:
        # dtype=object prevents [large_int, missing] from becoming float64.
        values = pd.Series(
            [mapping.get(str(value), value) if pd.notna(value) else value for value in values],
            index=values.index, dtype=object,
        )
    if not (strict or all_known):
        return values
    if dtype.lower().startswith(("int", "uint")):
        try:
            return _integer_series(values, dtype)
        except (ValueError, TypeError, OverflowError):
            if strict:
                raise ValueError(f"{path}: {column!r} contains values incompatible with {dtype}") from None
            return values
    if dtype.lower().startswith("float"):
        parsed = _numeric_values(values, dtype, strict=strict, path=path, column=column)
        if strict:
            return parsed
        # Known category-domain values already identify the original labels;
        # only restore a dtype when the cast leaves those values unchanged.
        try:
            cast = pd.Series(pd.array(parsed, dtype=dtype), index=values.index)
            if all(pd.isna(a) and pd.isna(b) or a == b for a, b in zip(cast, values)):
                return cast
        except (ValueError, TypeError, OverflowError):
            pass
        return values
    if dtype in {"bool", "boolean"}:
        parsed = [{"True": True, "False": False}.get(str(value), value)
                  if pd.notna(value) else pd.NA for value in values]
        if all(value is pd.NA or isinstance(value, (bool, np.bool_)) for value in parsed):
            target = "boolean" if any(value is pd.NA for value in parsed) else dtype
            return pd.Series(pd.array(parsed, dtype=target), index=values.index)
        if strict:
            raise ValueError(f"{path}: {column!r} contains invalid boolean labels")
    if dtype.startswith("string"):
        return values.astype(dtype)
    return values


def read_frame_csv(path: str | Path, schema=None, *, strict: bool = False,
                   usecols=None) -> pd.DataFrame:
    """Read declared values; ``usecols`` retains pandas' target-only fast path.

    Strict reference reads restore declared numeric dtypes and reject malformed
    values. Raw artifacts retain out-of-domain categorical values and numeric
    output that cannot be parsed without loss. With usecols, strict validation
    checks only selected schema columns.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), [])
    if len(header) != len(set(header)):
        raise ValueError(f"CSV has duplicated column names: {path}")
    fmt = _format(schema)
    if schema is None:
        return pd.read_csv(path, keep_default_na=False, na_values=[""], usecols=usecols)
    frame = pd.read_csv(path, dtype=object, keep_default_na=False, usecols=usecols)
    missing = fmt.get("missing_token", "")
    escape = fmt.get("escape_prefix")
    for column in frame:
        values = frame[column]
        empty_is_category = not fmt and "" in (schema.category_domain(column) or [])
        if not empty_is_category:
            values = values.mask(values == missing, np.nan)
        if escape:
            values = pd.Series(
                [value[len(escape):] if isinstance(value, str) and value.startswith(escape)
                 else value for value in values], index=values.index, dtype=object,
            )
        dtype = fmt.get("dtypes", {}).get(column, "")
        if column in schema.numerical_cols:
            frame[column] = _numeric_values(values, dtype, strict=strict, path=path, column=column)
        else:
            frame[column] = _category_values(values, schema.category_domain(column) or [], dtype,
                                             strict=strict, path=path, column=column)
    if strict:
        expected = set(schema.all_columns)
        if usecols is not None:
            if callable(usecols):
                expected &= {column for column in expected if usecols(column)}
            else:
                selected = {header[column] if isinstance(column, int) else column for column in usecols}
                expected &= selected
        absent = expected - set(frame.columns)
        if absent:
            raise ValueError(f"{path}: missing schema columns: {sorted(absent)}")
    return frame

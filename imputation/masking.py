# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Column selection and cell-level missingness generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from otbench.methods.common.data import feature_columns, reset_rows, dataframe_hash
from otbench.methods.common.locking import atomic_output_path, exclusive_directory_lock
from otbench.tabdata.io import csv_text, read_frame_csv
from otbench.tabdata.schema import schema_from_dict


SCENARIOS = {"MCAR", "MAR", "MNAR", "MIXED_MCAR_MAR_MNAR"}


def _json_scalar(value):
    return value.item() if isinstance(value, np.generic) else str(value)


@dataclass
class ConditionSpec:
    column: str | None
    operator: str
    value: Any
    fallback_reason: str | None = None


@dataclass
class ColumnSelection:
    target_col: str
    missing_columns: list[str]
    complete_columns: list[str]
    column_types: dict[str, str]
    rank_scores: dict[str, dict[str, float | None]]
    conditions: dict[str, dict[str, ConditionSpec]]
    selection_seed: int


@dataclass
class MaskBundle:
    dataset: str
    scenario: str
    rate: float
    selection_seed: int
    mask_seed: int
    selection: ColumnSelection
    mask: pd.DataFrame
    masked: pd.DataFrame
    output_dir: Path | None = None
    source_data_hash: str = ""
    schema: Any = None


def select_columns(
    train: pd.DataFrame,
    schema,
    seed: int = 0,
    max_missing_columns: int = 5,
    missing_fraction: float = 0.3,
    min_condition_size: int = 20,
    min_condition_fraction: float = 0.05,
) -> ColumnSelection:
    """Select missingness columns and mechanism conditions from train only."""

    rng = np.random.default_rng(seed)
    features = feature_columns(schema)
    candidates = [
        col for col in features
        if col in train.columns and train[col].nunique(dropna=True) > 1
    ]
    if not candidates:
        raise ValueError("No non-constant feature columns available for missingness selection.")

    target = schema.target_col
    encoded = _encode_for_selection(train[candidates + [target]])
    target_values = encoded[target]

    spearman_scores = _spearman_scores(encoded, candidates, target)
    importance_scores = _importance_scores(encoded, candidates, target_values, schema.task_type, seed)
    combined = _combined_rank(candidates, spearman_scores, importance_scores, rng)

    k = int(np.ceil(len(candidates) * missing_fraction))
    k = max(1, min(max_missing_columns, k, len(candidates)))
    missing_columns = [col for col, _score in sorted(combined.items(), key=lambda item: (-item[1], item[0]))[:k]]
    complete_columns = [col for col in features if col not in missing_columns]

    column_types = {col: schema.column_type(col) for col in features if col in train.columns}
    rank_scores = {
        col: {
            "spearman_abs": None if pd.isna(spearman_scores.get(col, np.nan)) else float(spearman_scores[col]),
            "importance": None if pd.isna(importance_scores.get(col, np.nan)) else float(importance_scores[col]),
            "combined": float(combined[col]),
        }
        for col in candidates
    }

    mar_conditions: dict[str, ConditionSpec] = {}
    for col in missing_columns:
        mar_conditions[col] = _choose_mar_condition(
            train,
            col,
            complete_columns,
            encoded,
            min_condition_size,
            min_condition_fraction,
        )

    mnar_conditions = {
        col: _self_condition(train, col, min_condition_size, min_condition_fraction)
        for col in missing_columns
    }

    return ColumnSelection(
        target_col=target,
        missing_columns=missing_columns,
        complete_columns=complete_columns,
        column_types=column_types,
        rank_scores=rank_scores,
        conditions={"MAR": mar_conditions, "MNAR": mnar_conditions},
        selection_seed=seed,
    )


def generate_mask(
    test: pd.DataFrame,
    selection: ColumnSelection,
    scenario: str,
    rate: float,
    mask_seed: int = 0,
) -> pd.DataFrame:
    """Generate a cell-level boolean mask for the selected feature columns."""

    scenario = scenario.upper()
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}. Expected one of {sorted(SCENARIOS)}.")
    if not 0 <= rate <= 1:
        raise ValueError("rate must be in [0, 1].")

    test = reset_rows(test)
    rng = np.random.default_rng(mask_seed)
    feature_cols = selection.missing_columns
    mask = pd.DataFrame(False, index=test.index, columns=test.columns)

    if scenario == "MCAR":
        _apply_mcar(mask, test, feature_cols, rate, rng)
    elif scenario == "MAR":
        _apply_conditional(mask, test, feature_cols, selection.conditions["MAR"], rate, rng)
    elif scenario == "MNAR":
        _apply_conditional(mask, test, feature_cols, selection.conditions["MNAR"], rate, rng)
    else:
        for offset, sub_scenario in enumerate(("MCAR", "MAR", "MNAR"), start=1):
            sub_rng = np.random.default_rng(mask_seed + offset * 1009)
            sub = pd.DataFrame(False, index=test.index, columns=test.columns)
            if sub_scenario == "MCAR":
                _apply_mcar(sub, test, feature_cols, rate / 3.0, sub_rng)
            else:
                _apply_conditional(sub, test, feature_cols, selection.conditions[sub_scenario], rate / 3.0, sub_rng)
            mask |= sub
        _calibrate_total(mask, test, feature_cols, rate, rng)

    mask[selection.target_col] = False
    return mask.astype(bool)


def apply_mask(test: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Apply a cell mask to a reset-index test dataframe."""

    test = reset_rows(test)
    out = test.copy(deep=True)
    out[mask.astype(bool)] = np.nan
    return out


def make_mask_bundle(
    dataset: str,
    test: pd.DataFrame,
    selection: ColumnSelection,
    scenario: str,
    rate: float,
    mask_seed: int,
    schema=None,
) -> MaskBundle:
    mask = generate_mask(test, selection, scenario, rate, mask_seed)
    masked = apply_mask(test, mask)
    return MaskBundle(
        dataset=dataset,
        scenario=scenario,
        rate=float(rate),
        selection_seed=selection.selection_seed,
        mask_seed=int(mask_seed),
        selection=selection,
        mask=mask,
        masked=masked,
        source_data_hash=dataframe_hash(test),
        schema=schema,
    )


def mask_bundle_dir(root: str | Path, bundle: MaskBundle) -> Path:
    return (
        Path(root)
        / "masks"
        / bundle.dataset
        / f"{bundle.scenario}_rate{_rate_str(bundle.rate)}_selection_seed{bundle.selection_seed}_mask_seed{bundle.mask_seed}"
        / f"data_{_mask_identity(bundle)[:16]}"
    )


def _mask_identity(bundle: MaskBundle) -> str:
    payload = {
        "version": 2, "data": bundle.source_data_hash or dataframe_hash(bundle.masked),
        "selection": _selection_to_json(bundle.selection), "mask": _mask_to_json(bundle),
        "schema": asdict(bundle.schema) if bundle.schema is not None else None,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=_json_scalar).encode()).hexdigest()


def save_mask_bundle(root: str | Path, bundle: MaskBundle) -> Path:
    out_dir = mask_bundle_dir(root, bundle)
    # Masks are method-independent and immutable for a fixed protocol key.
    # Concurrent ranks therefore publish once and subsequently reuse the same
    # bundle.  mkdir is atomic on the cluster's shared filesystem; unlike flock
    # it does not intermittently raise EAGAIN on this NFS mount.
    if _mask_bundle_matches(out_dir, bundle):
        bundle.output_dir = out_dir
        return out_dir
    with exclusive_directory_lock(Path(f"{out_dir}.lockdir")):
        if _mask_bundle_matches(out_dir, bundle):
            bundle.output_dir = out_dir
            return out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        with atomic_output_path(out_dir / "selection.json") as tmp:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(_selection_to_json(bundle.selection), f, indent=2, ensure_ascii=False, default=_json_scalar)
        with atomic_output_path(out_dir / "mask.json") as tmp:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(_mask_to_json(bundle), f, indent=2, ensure_ascii=False, default=_json_scalar)
        with atomic_output_path(out_dir / "masked.csv") as tmp:
            tmp.write_text(csv_text(bundle.masked, bundle.schema), encoding="utf-8")
        with atomic_output_path(out_dir / "bundle.json") as tmp:
            manifest = {
                "version": 2, "identity": _mask_identity(bundle),
                "source_data_hash": bundle.source_data_hash,
                "schema": asdict(bundle.schema) if bundle.schema is not None else None,
                "artifacts": {name: hashlib.sha256((out_dir / name).read_bytes()).hexdigest()
                              for name in ("selection.json", "mask.json", "masked.csv")},
            }
            tmp.write_text(json.dumps(manifest, indent=2, default=_json_scalar), encoding="utf-8")
    bundle.output_dir = out_dir
    return out_dir


def _mask_bundle_matches(out_dir: Path, bundle: MaskBundle) -> bool:
    """Whether an atomically published immutable bundle can be reused."""
    selection_path = out_dir / "selection.json"
    mask_path = out_dir / "mask.json"
    masked_path = out_dir / "masked.csv"
    manifest_path = out_dir / "bundle.json"
    if not all(path.is_file() for path in (selection_path, mask_path, masked_path, manifest_path)):
        return False
    if masked_path.stat().st_size <= 0:
        return False
    try:
        with selection_path.open("r", encoding="utf-8") as handle:
            saved_selection = json.load(handle)
        with mask_path.open("r", encoding="utf-8") as handle:
            saved_mask = json.load(handle)
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(manifest, dict):
        return False
    return (
        saved_selection == _selection_to_json(bundle.selection)
        and saved_mask == _mask_to_json(bundle)
        and manifest.get("version") == 2
        and manifest.get("identity") == _mask_identity(bundle)
        and manifest.get("artifacts") == {
            name: hashlib.sha256((out_dir / name).read_bytes()).hexdigest()
            for name in ("selection.json", "mask.json", "masked.csv")
        }
        and masked_path.read_text(encoding="utf-8") == csv_text(bundle.masked, bundle.schema)
    )


def load_mask_bundle(path: str | Path, test: pd.DataFrame, schema=None) -> MaskBundle:
    path = Path(path)
    with (path / "selection.json").open("r", encoding="utf-8") as f:
        selection = _selection_from_json(json.load(f))
    with (path / "mask.json").open("r", encoding="utf-8") as f:
        raw_mask = json.load(f)
    test = reset_rows(test)
    mask = pd.DataFrame(False, index=test.index, columns=test.columns)
    for cell in raw_mask["cells"]:
        mask.loc[int(cell["row"]), cell["column"]] = True
    manifest_path = path / "bundle.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if manifest_path.is_file() and (
        not isinstance(manifest, dict) or manifest.get("version") != 2
        or set(manifest.get("artifacts", {})) != {"selection.json", "mask.json", "masked.csv"}
    ):
        raise ValueError("mask bundle manifest is malformed")
    if manifest:
        if manifest.get("source_data_hash") and manifest["source_data_hash"] != dataframe_hash(test):
            raise ValueError("mask bundle belongs to a different source dataset")
        for name, expected in manifest.get("artifacts", {}).items():
            if hashlib.sha256((path / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"mask bundle artifact differs: {name}")
    if schema is None and manifest.get("schema"):
        schema = schema_from_dict(manifest["schema"])
    masked = read_frame_csv(path / "masked.csv", schema)
    bundle = MaskBundle(
        dataset=raw_mask["dataset"],
        scenario=raw_mask["scenario"],
        rate=float(raw_mask["rate"]),
        selection_seed=int(raw_mask["selection_seed"]),
        mask_seed=int(raw_mask["mask_seed"]),
        selection=selection,
        mask=mask,
        masked=masked,
        output_dir=path,
        source_data_hash=manifest.get("source_data_hash", ""),
        schema=schema,
    )
    return bundle


def _encode_for_selection(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            out[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            out[col] = pd.factorize(df[col], sort=True)[0].astype(float)
            out.loc[df[col].isna(), col] = np.nan
        out[col] = out[col].fillna(out[col].median() if out[col].notna().any() else 0.0)
    return out


def _spearman_scores(encoded: pd.DataFrame, candidates: list[str], target: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    for col in candidates:
        corr = encoded[[col, target]].corr(method="spearman").iloc[0, 1]
        scores[col] = abs(float(corr)) if pd.notna(corr) else np.nan
    return scores


def _importance_scores(
    encoded: pd.DataFrame,
    candidates: list[str],
    target_values: pd.Series,
    task_type: str | None,
    seed: int,
) -> dict[str, float]:
    try:
        X = encoded[candidates]
        if task_type == "regression":
            y = pd.to_numeric(target_values, errors="coerce").fillna(0.0)
            model = RandomForestRegressor(n_estimators=64, random_state=seed, n_jobs=1)
        else:
            y = pd.factorize(target_values, sort=True)[0]
            if len(np.unique(y)) < 2:
                return {col: np.nan for col in candidates}
            model = RandomForestClassifier(n_estimators=64, random_state=seed, n_jobs=1)
        model.fit(X, y)
        return {col: float(value) for col, value in zip(candidates, model.feature_importances_)}
    except Exception:
        return {col: np.nan for col in candidates}


def _combined_rank(
    candidates: list[str],
    spearman_scores: dict[str, float],
    importance_scores: dict[str, float],
    rng: np.random.Generator,
) -> dict[str, float]:
    spearman = pd.Series(spearman_scores).replace([np.inf, -np.inf], np.nan)
    importance = pd.Series(importance_scores).replace([np.inf, -np.inf], np.nan)
    if spearman.notna().any():
        spearman_rank = spearman.rank(pct=True, method="average").fillna(0.0)
    else:
        spearman_rank = pd.Series(0.0, index=candidates)
    if importance.notna().any():
        importance_rank = importance.rank(pct=True, method="average").fillna(0.0)
    else:
        importance_rank = pd.Series(0.0, index=candidates)
    combined = (spearman_rank + importance_rank) / 2.0
    if float(combined.max()) <= 0:
        jitter = pd.Series(rng.random(len(candidates)) * 1e-9, index=candidates)
        combined = pd.Series({col: 1.0 / (idx + 1) for idx, col in enumerate(sorted(candidates))}) + jitter
    return {col: float(combined[col]) for col in candidates}


def _choose_mar_condition(
    train: pd.DataFrame,
    missing_col: str,
    complete_columns: list[str],
    encoded: pd.DataFrame,
    min_size: int,
    min_fraction: float,
) -> ConditionSpec:
    candidates = [col for col in complete_columns if col in train.columns and col in encoded.columns and col != missing_col]
    if not candidates:
        return ConditionSpec(None, "all", None, "no_complete_condition_columns")
    correlations: list[tuple[str, float]] = []
    for col in candidates:
        corr = encoded[[col, missing_col]].corr(method="spearman").iloc[0, 1]
        correlations.append((col, abs(float(corr)) if pd.notna(corr) else 0.0))
    for col, _score in sorted(correlations, key=lambda item: (-item[1], item[0])):
        spec = _condition_for_column(train, col)
        if _condition_ok(train, spec, min_size, min_fraction):
            return spec
    return ConditionSpec(None, "all", None, "condition_groups_too_small")


def _self_condition(train: pd.DataFrame, col: str, min_size: int, min_fraction: float) -> ConditionSpec:
    for quantile in (0.5, 0.25, 0.75, 0.1, 0.9):
        spec = _condition_for_column(train, col, quantile=quantile)
        if _condition_ok(train, spec, min_size, min_fraction):
            return spec
    return ConditionSpec(None, "all", None, "self_condition_too_small")


def _condition_for_column(train: pd.DataFrame, col: str, quantile: float = 0.5) -> ConditionSpec:
    series = train[col]
    # pandas and NumPy consider bool numeric, but quantiles over boolean
    # arrays are undefined (and recent NumPy raises on boolean subtraction).
    # Boolean features are categorical in the tabular protocol, so construct
    # their condition from the observed mode just like other categories.
    if pd.api.types.is_bool_dtype(series.dtype):
        mode = series.mode(dropna=True)
        if not len(mode):
            return ConditionSpec(None, "all", None, "no_mode")
        return ConditionSpec(col, "eq", mode.iloc[0])
    if pd.api.types.is_numeric_dtype(series):
        value = pd.to_numeric(series, errors="coerce").quantile(quantile)
        if pd.isna(value):
            return ConditionSpec(None, "all", None, "nan_threshold")
        return ConditionSpec(col, "ge", float(value))
    mode = series.mode(dropna=True)
    if not len(mode):
        return ConditionSpec(None, "all", None, "no_mode")
    return ConditionSpec(col, "eq", mode.iloc[0])


def _condition_rows(df: pd.DataFrame, spec: ConditionSpec) -> np.ndarray:
    if spec.column is None or spec.operator == "all":
        return np.arange(len(df))
    if spec.operator == "ge":
        cond = pd.to_numeric(df[spec.column], errors="coerce") >= float(spec.value)
    elif spec.operator == "eq":
        cond = df[spec.column] == spec.value
    else:
        raise ValueError(f"Unsupported operator: {spec.operator}")
    return np.flatnonzero(cond.fillna(False).to_numpy())


def _condition_ok(df: pd.DataFrame, spec: ConditionSpec, min_size: int, min_fraction: float) -> bool:
    rows = _condition_rows(df, spec)
    return len(rows) >= min_size and len(rows) >= int(np.ceil(len(df) * min_fraction))


def _apply_mcar(mask: pd.DataFrame, df: pd.DataFrame, cols: list[str], rate: float, rng: np.random.Generator) -> None:
    _sample_cells(mask, df, cols, np.arange(len(df)), rate, rng)


def _apply_conditional(
    mask: pd.DataFrame,
    df: pd.DataFrame,
    cols: list[str],
    conditions: dict[str, ConditionSpec],
    rate: float,
    rng: np.random.Generator,
) -> None:
    for col in cols:
        spec = conditions.get(col, ConditionSpec(None, "all", None, "missing_condition"))
        rows = _condition_rows(df, spec)
        if len(rows) == 0:
            rows = np.arange(len(df))
        target_count = int(round(len(df) * rate))
        eligible = [(row, col) for row in rows if pd.notna(df.at[row, col])]
        if not eligible:
            continue
        chosen = rng.choice(len(eligible), size=min(target_count, len(eligible)), replace=False)
        for idx in np.atleast_1d(chosen):
            row, col_name = eligible[int(idx)]
            mask.at[row, col_name] = True


def _sample_cells(
    mask: pd.DataFrame,
    df: pd.DataFrame,
    cols: list[str],
    rows: np.ndarray,
    rate: float,
    rng: np.random.Generator,
) -> None:
    eligible = [(row, col) for row in rows for col in cols if pd.notna(df.at[row, col])]
    if not eligible:
        return
    target_count = int(round(len(rows) * len(cols) * rate))
    chosen = rng.choice(len(eligible), size=min(target_count, len(eligible)), replace=False)
    for idx in np.atleast_1d(chosen):
        row, col = eligible[int(idx)]
        mask.at[row, col] = True


def _calibrate_total(mask: pd.DataFrame, df: pd.DataFrame, cols: list[str], rate: float, rng: np.random.Generator) -> None:
    target = int(round(len(df) * len(cols) * rate))
    current_cells = [(row, col) for row in df.index for col in cols if bool(mask.at[row, col])]
    if len(current_cells) > target:
        drop = rng.choice(len(current_cells), size=len(current_cells) - target, replace=False)
        for idx in np.atleast_1d(drop):
            row, col = current_cells[int(idx)]
            mask.at[row, col] = False
    elif len(current_cells) < target:
        eligible = [
            (row, col)
            for row in df.index
            for col in cols
            if pd.notna(df.at[row, col]) and not bool(mask.at[row, col])
        ]
        add = rng.choice(len(eligible), size=min(target - len(current_cells), len(eligible)), replace=False)
        for idx in np.atleast_1d(add):
            row, col = eligible[int(idx)]
            mask.at[row, col] = True


def _selection_to_json(selection: ColumnSelection) -> dict[str, Any]:
    data = asdict(selection)
    for scenario, mapping in selection.conditions.items():
        data["conditions"][scenario] = {col: asdict(spec) for col, spec in mapping.items()}
    return data


def _selection_from_json(data: dict[str, Any]) -> ColumnSelection:
    conditions = {
        scenario: {col: ConditionSpec(**spec) for col, spec in mapping.items()}
        for scenario, mapping in data["conditions"].items()
    }
    return ColumnSelection(
        target_col=data["target_col"],
        missing_columns=list(data["missing_columns"]),
        complete_columns=list(data["complete_columns"]),
        column_types=dict(data["column_types"]),
        rank_scores=dict(data["rank_scores"]),
        conditions=conditions,
        selection_seed=int(data["selection_seed"]),
    )


def _mask_to_json(bundle: MaskBundle) -> dict[str, Any]:
    cells = []
    for row in bundle.mask.index:
        for col in bundle.selection.missing_columns:
            if bool(bundle.mask.at[row, col]):
                spec = _cell_condition(bundle.selection, bundle.scenario, col)
                cells.append({
                    "row": int(row),
                    "column": col,
                    "mechanism": bundle.scenario,
                    "condition_col": spec.column,
                    "condition_value": spec.value,
                    "rate": bundle.rate,
                })
    return {
        "dataset": bundle.dataset,
        "scenario": bundle.scenario,
        "rate": bundle.rate,
        "selection_seed": bundle.selection_seed,
        "mask_seed": bundle.mask_seed,
        "target_col": bundle.selection.target_col,
        "missing_columns": bundle.selection.missing_columns,
        "cells": cells,
    }


def _cell_condition(selection: ColumnSelection, scenario: str, col: str) -> ConditionSpec:
    if scenario == "MAR":
        return selection.conditions["MAR"].get(col, ConditionSpec(None, "all", None))
    if scenario == "MNAR":
        return selection.conditions["MNAR"].get(col, ConditionSpec(None, "all", None))
    return ConditionSpec(None, "all", None)


def _rate_str(rate: float) -> str:
    return ("%g" % float(rate)).replace(".", "p")

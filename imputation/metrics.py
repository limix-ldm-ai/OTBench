# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""I01-I07: point accuracy and structure preservation on masked cells.

Everything is scored on the masked cells only, against the clean test values
that were hidden. Two conventions run through the whole module:

* Normalisers (SD, IQR, category domains) come from the **clean train** split.
  Using the test split's own spread would make an imputer look better simply
  because the test sample happens to be tight, and it would make the number
  incomparable across mask rates.
* A column with no masked cell is skipped, not scored as zero error. The
  per-column detail records the count so the macro average is auditable.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from otbench.methods.common.data import categorical_feature_columns, numerical_feature_columns
from otbench.metrics.catalog import MetricResult
from otbench.metrics.stats import (
    association_matrix,
    jensen_shannon_divergence,
    matrix_difference,
    relative_frobenius_distance,
    robust_scale,
    wasserstein1,
)

from .masking import MaskBundle
from .validation import validate_completion

EPS = 1e-12

_NAMES = {
    "nrmse": "imputation.point.macro_nrmse",
    "nmae": "imputation.point.macro_nmae",
    "f1": "imputation.point.categorical_macro_f1",
    "bacc": "imputation.point.categorical_balanced_accuracy",
    "w1": "imputation.structure.numeric_wasserstein1",
    "jsd": "imputation.structure.categorical_jsd",
    "assoc": "imputation.structure.association_matrix_error",
}


def evaluate_point_metrics(
    clean_test: pd.DataFrame,
    imputed: pd.DataFrame,
    mask_bundle: MaskBundle,
    schema,
    train: pd.DataFrame,
    selected: set[str] | None = None,
) -> list[MetricResult]:
    """I01-I07 for one completed dataset."""
    # Direct evaluator callers must not be able to improve a score by omitting
    # predictions. Observation preservation is enforced by the task boundary.
    validate_completion(clean_test, imputed, mask_bundle.mask, schema, check_observed=False)
    clean = clean_test.reset_index(drop=True)
    pred = imputed.reset_index(drop=True)
    mask = mask_bundle.mask.reset_index(drop=True)

    numeric = _numeric_columns(clean, pred, mask, schema, train)
    categorical = _categorical_columns(clean, pred, mask, schema)

    results = [
        _macro(_NAMES["nrmse"], numeric, "nrmse", "no numerical column had a masked cell"),
        _macro(_NAMES["nmae"], numeric, "nmae", "no numerical column had a masked cell"),
        _macro(_NAMES["f1"], categorical, "macro_f1", "no categorical column had a masked cell"),
        _macro(_NAMES["bacc"], categorical, "balanced_accuracy",
               "no categorical column had a masked cell"),
        _macro(_NAMES["w1"], numeric, "wasserstein1", "no numerical column had a masked cell"),
        _macro(_NAMES["jsd"], categorical, "jsd", "no categorical column had a masked cell"),
    ]
    if selected is None or _NAMES["assoc"] in selected:
        results.append(_association(clean, pred, schema))
    return results if selected is None else [r for r in results if r.name in selected]


def _numeric_columns(clean, pred, mask, schema, train) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for col in numerical_feature_columns(schema):
        rows = _masked_rows(mask, col, len(clean))
        if not rows.any() or col not in pred.columns:
            continue
        y_true = pd.to_numeric(clean.loc[rows, col], errors="coerce").to_numpy(float)
        y_pred = pd.to_numeric(pred.loc[rows, col], errors="coerce").to_numpy(float)
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        n_unparseable = int((~np.isfinite(y_pred) & np.isfinite(y_true)).sum())
        y_true, y_pred = y_true[valid], y_pred[valid]
        if len(y_true) == 0:
            continue

        scale, scale_kind = _train_scale(col, train)
        diff = y_pred - y_true
        rmse = float(np.sqrt(np.mean(diff**2)))
        mae = float(np.mean(np.abs(diff)))
        record: dict[str, Any] = {
            "count": len(y_true),
            "n_unparseable_predictions": n_unparseable,
            "rmse": rmse,
            "mae": mae,
            "scale": scale,
            "scale_kind": scale_kind,
        }
        if scale > EPS:
            record["nrmse"] = rmse / scale
            record["nmae"] = mae / scale
            record["wasserstein1"] = wasserstein1(y_true / scale, y_pred / scale)
        out[col] = record
    return out


def _categorical_columns(clean, pred, mask, schema) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for col in categorical_feature_columns(schema):
        rows = _masked_rows(mask, col, len(clean))
        if not rows.any() or col not in pred.columns:
            continue
        y_true = clean.loc[rows, col].astype(str)
        y_pred = pred.loc[rows, col].astype(str)
        out[col] = {
            "count": int(rows.sum()),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "accuracy": float((y_true.to_numpy() == y_pred.to_numpy()).mean()),
            "jsd": jensen_shannon_divergence(y_true, y_pred),
            "n_true_categories": int(y_true.nunique()),
            "n_predicted_categories": int(y_pred.nunique()),
        }
    return out


def _association(clean: pd.DataFrame, pred: pd.DataFrame, schema) -> MetricResult:
    """I07: does the completed table keep the clean table's dependence structure?"""
    numerical = [c for c in schema.numerical_cols if c in clean.columns and c in pred.columns]
    categorical = [c for c in schema.categorical_cols if c in clean.columns and c in pred.columns]
    if len(numerical) + len(categorical) < 2:
        return MetricResult.skipped(
            _NAMES["assoc"], "an association matrix needs at least two shared columns"
        )
    clean_matrix = association_matrix(clean, numerical, categorical)
    pred_matrix = association_matrix(pred, numerical, categorical)
    frobenius, mean_abs, n_pairs = matrix_difference(clean_matrix, pred_matrix)
    relative_frobenius = relative_frobenius_distance(clean_matrix, pred_matrix)
    if relative_frobenius is None:
        return MetricResult.skipped(
            _NAMES["assoc"],
            "every association was undefined or the clean association norm was zero"
        )
    return MetricResult.ok(
        _NAMES["assoc"], relative_frobenius, n=n_pairs,
        details={"frobenius_norm": frobenius, "mean_absolute_difference": mean_abs,
                 "clean_matrix": clean_matrix.round(6).to_dict(),
                 "imputed_matrix": pred_matrix.round(6).to_dict()},
        extra={"encoding": "|Pearson r| numeric-numeric, correlation ratio mixed, "
                           "Cramer's V categorical-categorical",
               "note": "scalar is ||A_clean-A_imputed||_F / ||A_clean||_F over comparable "
                       "off-diagonal pairs"},
    )


def _macro(name: str, per_column: dict[str, dict], field: str, empty_reason: str) -> MetricResult:
    values = {col: rec[field] for col, rec in per_column.items() if rec.get(field) is not None}
    if not values:
        return MetricResult.skipped(name, empty_reason)
    counts = {col: per_column[col]["count"] for col in values}
    return MetricResult.ok(
        name, float(np.mean(list(values.values()))), n=sum(counts.values()),
        details={"per_column": {c: {"value": v, "n_masked": counts[c]} for c, v in values.items()}},
        extra={"aggregation": "unweighted mean over columns with at least one masked cell"},
    )






def _masked_rows(mask: pd.DataFrame, col: str, n: int) -> np.ndarray:
    if col not in mask.columns:
        return np.zeros(n, dtype=bool)
    return mask[col].fillna(False).to_numpy(dtype=bool)


def _train_scale(col: str, train: pd.DataFrame) -> tuple[float, str]:
    """Train SD, falling back to the IQR when the SD degenerates.

    Routed through the shared :func:`robust_scale` so an imputation NRMSE and a
    prediction NRMSE use the same estimator.
    """
    if col not in train.columns:
        return 0.0, "unavailable"
    values = pd.to_numeric(train[col], errors="coerce").dropna().to_numpy(float)
    scale, kind = robust_scale(values)
    return (scale, f"train_{kind}") if scale > EPS else (0.0, "constant_column")


def aggregate_results(per_trial: list[list[MetricResult]]) -> dict[str, Any]:
    """Mean/std across trials, keeping the ok-count so partial success is visible."""
    by_name: dict[str, list[float]] = {}
    statuses: dict[str, list[str]] = {}
    for results in per_trial:
        for result in results:
            statuses.setdefault(result.name, []).append(result.status)
            if result.status == "ok" and result.value is not None:
                by_name.setdefault(result.name, []).append(result.value)

    aggregate = {}
    for name, statuses_seen in statuses.items():
        values = by_name.get(name, [])
        aggregate[name] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values)) if values else None,
            "n_ok": len(values),
            "n_trials": len(per_trial),
            "statuses": sorted(set(statuses_seen)),
        }
    return aggregate

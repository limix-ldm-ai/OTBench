# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""I10-I17: the metrics that need K joint draws of the same mask.

These exist because a single completed table cannot answer the question multiple
imputation is for: not "what value?" but "how uncertain is the value, and are
the uncertain cells of one row consistent with each other?".

Two rules make the answers meaningful:

* The K draws must come from **one** ``impute_draws`` call over the **same**
  mask, and each draw must stay a whole row. Averaging the draws per cell first
  destroys the joint structure the Energy Score measures.
* The draws must be a native posterior sample. Re-running a deterministic
  imputer under different seeds produces K identical (or arbitrarily jittered)
  tables, and calibration computed from those is meaningless. The caller passes
  ``native_draws=False`` in that case and every metric here skips.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from otbench.methods.common.base import ImputationOutput
from otbench.methods.common.data import categorical_feature_columns, numerical_feature_columns
from otbench.metrics.catalog import MetricResult
from otbench.metrics.stats import crps_ensemble, energy_score, interval_coverage, interval_width

from .masking import MaskBundle

EPS = 1e-12

# Laplace smoothing for the empirical category distribution over K draws. With
# K draws a category that never appears has empirical probability 0, and log
# loss would be infinite; the constant is reported with the metric.
LAPLACE_ALPHA = 1.0

NOMINAL_LEVELS = (0.90, 0.95)

_NAMES = {
    "crps": "imputation.draws.numeric_crps",
    "logloss": "imputation.draws.categorical_log_loss",
    "brier": "imputation.draws.categorical_brier",
    "cov90": "imputation.draws.interval_coverage_error_90",
    "cov95": "imputation.draws.interval_coverage_error_95",
    "width90": "imputation.draws.normalized_interval_width_90",
    "width95": "imputation.draws.normalized_interval_width_95",
    "energy": "imputation.draws.energy_score",
    "mad": "imputation.draws.mad_diversity",
}
ALL_NAMES = tuple(_NAMES.values())


def evaluate_draw_metrics(
    clean_test: pd.DataFrame,
    draws_output: ImputationOutput,
    mask_bundle: MaskBundle,
    schema,
    train: pd.DataFrame,
    native_draws: bool = True,
) -> list[MetricResult]:
    """I10-I17 over the K jointly-sampled completions in ``draws_output``."""
    if not native_draws:
        reason = (
            "the method has no native draws capability; the repeated runs are not a "
            "posterior sample, so calibration and sharpness are not defined for them"
        )
        return [MetricResult.skipped(name, reason) for name in ALL_NAMES]

    k = draws_output.n_draws
    if k < 2:
        return [MetricResult.skipped(name, f"K={k}: probabilistic metrics need at least 2 draws")
                for name in ALL_NAMES]

    clean = clean_test.reset_index(drop=True)
    mask = mask_bundle.mask.reset_index(drop=True)
    draws = [d.reset_index(drop=True) for d in draws_output.draws]
    shared = {"n_draws": k, "draw_source": draws_output.metadata.get("source", "native")}

    numeric = _numeric_draw_records(clean, draws, mask, schema, train)
    categorical = _categorical_draw_records(clean, draws, mask, schema)

    results = [
        _macro(_NAMES["crps"], numeric, "crps", "no numerical column had a masked cell", shared),
        _macro(_NAMES["logloss"], categorical, "log_loss",
               "no categorical column had a masked cell",
               {**shared, "laplace_alpha": LAPLACE_ALPHA}),
        _macro(_NAMES["brier"], categorical, "brier",
               "no categorical column had a masked cell", shared),
        _macro(_NAMES["cov90"], numeric, "coverage_error_90",
               "no numerical column had a masked cell",
               {**shared, "nominal": 0.90, "interval": "empirical 5th-95th percentile of the draws"}),
        _macro(_NAMES["cov95"], numeric, "coverage_error_95",
               "no numerical column had a masked cell",
               {**shared, "nominal": 0.95, "interval": "empirical 2.5th-97.5th percentile"}),
        _macro(_NAMES["width90"], numeric, "width_90",
               "no numerical column had a masked cell",
               {**shared, "read_with": _NAMES["cov90"]}),
        _macro(_NAMES["width95"], numeric, "width_95",
               "no numerical column had a masked cell",
               {**shared, "read_with": _NAMES["cov95"]}),
        _energy(clean, draws, mask, schema, train, shared),
        _mad_diversity(draws, mask, schema, train, numeric, categorical, shared),
    ]
    return results


def _numeric_draw_records(clean, draws, mask, schema, train) -> dict[str, dict[str, Any]]:
    from .metrics import _masked_rows, _train_scale

    out: dict[str, dict[str, Any]] = {}
    for col in numerical_feature_columns(schema):
        rows = _masked_rows(mask, col, len(clean))
        if not rows.any() or any(col not in d.columns for d in draws):
            continue
        y_true = pd.to_numeric(clean.loc[rows, col], errors="coerce").to_numpy(float)
        samples = np.column_stack([
            pd.to_numeric(d.loc[rows, col], errors="coerce").to_numpy(float) for d in draws
        ])  # (n_masked, K)
        valid = np.isfinite(y_true) & np.isfinite(samples).all(axis=1)
        y_true, samples = y_true[valid], samples[valid]
        if len(y_true) == 0:
            continue

        scale, scale_kind = _train_scale(col, train)
        record: dict[str, Any] = {"count": len(y_true), "scale_kind": scale_kind}
        crps = float(np.mean(crps_ensemble(y_true, samples)))
        record["crps"] = crps / scale if scale > EPS else crps
        record["crps_unscaled"] = crps

        for level, key in zip(NOMINAL_LEVELS, ("90", "95"), strict=True):
            tail = (1.0 - level) / 2.0
            lower = np.quantile(samples, tail, axis=1)
            upper = np.quantile(samples, 1.0 - tail, axis=1)
            record[f"coverage_{key}"] = interval_coverage(y_true, lower, upper)
            record[f"coverage_error_{key}"] = record[f"coverage_{key}"] - level
            if scale > EPS:
                record[f"width_{key}"] = interval_width(lower, upper, scale=scale)

        median = np.median(samples, axis=1, keepdims=True)
        mad = float(np.mean(np.abs(samples - median)))
        record["mad"] = mad / scale if scale > EPS else mad
        out[col] = record
    return out


def _categorical_draw_records(clean, draws, mask, schema) -> dict[str, dict[str, Any]]:
    from .metrics import _masked_rows

    out: dict[str, dict[str, Any]] = {}
    for col in categorical_feature_columns(schema):
        rows = _masked_rows(mask, col, len(clean))
        if not rows.any() or any(col not in d.columns for d in draws):
            continue
        y_true = clean.loc[rows, col].astype(str).to_numpy()
        drawn = np.column_stack([d.loc[rows, col].astype(str).to_numpy() for d in draws])

        # The support is the truth's categories plus anything the draws invented,
        # so an invented category costs probability mass instead of vanishing.
        categories = sorted(set(y_true) | set(drawn.ravel()))
        index = {c: i for i, c in enumerate(categories)}
        k = drawn.shape[1]
        counts = np.zeros((len(y_true), len(categories)), dtype=float)
        for j in range(k):
            counts[np.arange(len(y_true)), [index[v] for v in drawn[:, j]]] += 1.0
        proba = (counts + LAPLACE_ALPHA) / (k + LAPLACE_ALPHA * len(categories))

        true_idx = np.array([index[v] for v in y_true])
        picked = proba[np.arange(len(y_true)), true_idx]
        onehot = np.zeros_like(proba)
        onehot[np.arange(len(y_true)), true_idx] = 1.0

        mode = counts.argmax(axis=1)
        out[col] = {
            "count": len(y_true),
            "log_loss": float(-np.mean(np.log(picked))),
            "brier": float(np.mean(np.sum((proba - onehot) ** 2, axis=1))),
            "mad": float(np.mean(counts[np.arange(len(y_true)), mode] < k)),
            "n_categories": len(categories),
        }
    return out


def _energy(clean, draws, mask, schema, train, shared) -> MetricResult:
    """I16: the joint score. Every masked numeric cell of a row at once."""
    from .metrics import _train_scale

    name = _NAMES["energy"]
    numeric_cols = [c for c in numerical_feature_columns(schema)
                    if c in mask.columns and all(c in d.columns for d in draws)]
    if not numeric_cols:
        return MetricResult.skipped(name, "no numerical column is available in every draw")

    scales = {}
    for col in numeric_cols:
        scale, _ = _train_scale(col, train)
        scales[col] = scale if scale > EPS else 1.0

    row_mask = mask[numeric_cols].fillna(False).to_numpy(dtype=bool)
    rows_with_missing = np.flatnonzero(row_mask.any(axis=1))
    if len(rows_with_missing) == 0:
        return MetricResult.skipped(name, "no row has a masked numerical cell")

    truth = np.column_stack([
        pd.to_numeric(clean[c], errors="coerce").to_numpy(float) / scales[c] for c in numeric_cols
    ])
    drawn = np.stack([
        np.column_stack([
            pd.to_numeric(d[c], errors="coerce").to_numpy(float) / scales[c] for c in numeric_cols
        ])
        for d in draws
    ], axis=1)  # (n_rows, K, d)

    scores = []
    for row in rows_with_missing:
        cells = row_mask[row]
        y = truth[row, cells]
        samples = drawn[row][:, cells]
        if not np.isfinite(y).all() or not np.isfinite(samples).all():
            continue
        scores.append(energy_score(y[None, :], samples[None, :, :])[0])
    if not scores:
        return MetricResult.skipped(name, "every row with a masked cell had a non-finite value")
    return MetricResult.ok(
        name, float(np.mean(scores)), n=len(scores),
        details={"columns": numeric_cols, "n_rows_scored": len(scores)},
        extra={**shared, "units": "train-standardised", "beta": 1.0},
    )


def _mad_diversity(draws, mask, schema, train, numeric, categorical, shared) -> MetricResult:
    """I17: how much the draws disagree. Spread only, no notion of correctness."""
    name = _NAMES["mad"]
    values = [rec["mad"] for rec in list(numeric.values()) + list(categorical.values())
              if rec.get("mad") is not None]
    if not values:
        return MetricResult.skipped(name, "no column had a masked cell in every draw")
    return MetricResult.ok(
        name, float(np.mean(values)), n=len(values),
        details={"numeric_mad": {c: r["mad"] for c, r in numeric.items()},
                 "categorical_disagreement": {c: r["mad"] for c, r in categorical.items()}},
        extra={**shared,
               "interpretation": "mean absolute deviation around the per-cell median "
                                 "(numeric, train-standardised) and the share of draws off "
                                 "the modal category (categorical). Neither direction is "
                                 "good on its own; read with the CRPS and coverage metrics."},
    )


def _macro(name: str, per_column: dict, field: str, empty_reason: str, extra: dict) -> MetricResult:
    values = {c: r[field] for c, r in per_column.items() if r.get(field) is not None}
    if not values:
        return MetricResult.skipped(name, empty_reason)
    counts = {c: per_column[c]["count"] for c in values}
    return MetricResult.ok(
        name, float(np.mean(list(values.values()))), n=sum(counts.values()),
        details={"per_column": {c: {"value": v, "n_masked": counts[c]} for c, v in values.items()}},
        extra=extra,
    )

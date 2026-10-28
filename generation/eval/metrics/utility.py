# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""G12 global conditional utility and G13 target TSTR gap.

Global utility is an unclipped synthetic-trained/real-trained ratio over each
column used as a target. TSTR reports the signed gap on the declared supervised
target, using the same real test split for both training sources.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from otbench.generation.eval.data import EvalFrames
from otbench.metrics.catalog import MetricResult

_GLOBAL = "generation.utility.global_utility_ratio"
_TSTR = "generation.utility.tstr_gap"


def evaluate_utility(
    frames: EvalFrames,
    include_global: bool = True,
    include_tstr: bool = True,
    seed: int = 0,
) -> list[MetricResult]:
    train_ord, test_ord, gen_ord = frames.ordinal_all()
    columns = frames.schema.all_columns
    target = frames.schema.target_col
    scored = columns if include_global else [target]

    real_scores = _score_all_targets(train_ord[columns], test_ord[columns], frames, scored, seed)
    syn_scores = _score_all_targets(gen_ord[columns], test_ord[columns], frames, scored, seed)

    per_column: dict[str, Any] = {}
    ratios: list[float] = []
    for col in scored:
        col_type = frames.schema.column_type(col)
        real = real_scores.get(col)
        syn = syn_scores.get(col)
        ratio = _utility_ratio(real, syn, col_type)
        per_column[col] = {
            "type": col_type, "real": real, "synthetic": syn, "ratio": ratio,
        }
        if ratio is not None and math.isfinite(ratio):
            ratios.append(float(ratio))

    results: list[MetricResult] = []
    if include_global:
        if ratios:
            value = float(np.mean(ratios))
            results.append(MetricResult.ok(
                _GLOBAL, value, n=len(ratios), seed=seed,
                details={"per_column": per_column},
                extra={"note": "not clipped: ratios above 1 mean synthetic-trained models "
                               "beat real-trained ones on the real test split"},
            ))
        else:
            for name in (_GLOBAL,):
                results.append(MetricResult.skipped(name, "no column produced a usable score"))

    if include_tstr:
        results.append(_tstr_gap(frames, real_scores.get(target), syn_scores.get(target), seed))
    return results


def _tstr_gap(frames: EvalFrames, real, syn, seed: int) -> MetricResult:
    if real is None or syn is None:
        return MetricResult.skipped(_TSTR, "the target column produced no usable score")
    col_type = frames.schema.column_type(frames.schema.target_col)
    if col_type == "categorical":
        # Both are balanced accuracies: a plain difference is already oriented.
        gap = float(syn - real)
        detail = {"metric": "balanced_accuracy", "tstr": float(syn), "trtr": float(real)}
    else:
        # Stored as negative RMSE; convert back and express the gap relative to
        # the TRTR denominator so datasets with different target scales compare.
        real_rmse, syn_rmse = abs(float(real)), abs(float(syn))
        if real_rmse <= np.finfo(float).eps:
            return MetricResult.skipped(_TSTR, "TRTR RMSE is zero, so a relative gap is undefined")
        gap = float((real_rmse - syn_rmse) / real_rmse)
        detail = {"metric": "relative_rmse", "tstr_rmse": syn_rmse, "trtr_rmse": real_rmse}
    return MetricResult.ok(
        _TSTR, gap, seed=seed, details=detail,
        extra={"orientation": "positive means the synthetic-trained model scored better"},
    )


def _score_all_targets(train_df, eval_df, frames: EvalFrames, targets: list[str], seed: int):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import balanced_accuracy_score, root_mean_squared_error
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

    scores: dict[str, float | None] = {}
    for col in targets:
        x_train = train_df.drop(columns=[col])
        x_eval = eval_df.drop(columns=[col])
        y_train = train_df[col]
        y_eval = eval_df[col]

        if frames.schema.column_type(col) == "categorical":
            if y_train.nunique() <= 1:
                # A constant training target teaches nothing; recording 1.0 would
                # award a perfect score to a degenerate generator.
                scores[col] = None
                continue
            models = [
                # Dataset/seed jobs are already parallelized by the scheduler;
                # nested all-core forests caused host OOM on high-cardinality data.
                RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1),
                KNeighborsClassifier(n_neighbors=min(5, max(1, len(train_df)))),
            ]
            values = []
            for model in models:
                model.fit(x_train, y_train)
                values.append(balanced_accuracy_score(y_eval, model.predict(x_eval)))
            scores[col] = float(np.mean(values))
        else:
            models = [
                RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1),
                KNeighborsRegressor(n_neighbors=min(5, max(1, len(train_df)))),
            ]
            values = []
            for model in models:
                model.fit(x_train, y_train)
                values.append(-root_mean_squared_error(y_eval, model.predict(x_eval)))
            scores[col] = float(np.mean(values))
    return scores


def _utility_ratio(real, syn, col_type: str) -> float | None:
    if real is None or syn is None:
        return None
    if col_type == "categorical":
        return float(syn / real) if real > 0 else None
    real_rmse, syn_rmse = abs(float(real)), abs(float(syn))
    if syn_rmse <= np.finfo(float).eps:
        # A zero-RMSE synthetic model is a red flag (usually leakage or a
        # constant column), not an infinitely good one.
        return None
    return float(real_rmse / syn_rmse)

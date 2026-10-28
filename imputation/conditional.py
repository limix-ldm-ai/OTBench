# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""I18, I20-I22: imputation metrics that need something beyond a single run.

Each of these is a real metric with a real definition, and each is undefined
without a precondition the benchmark cannot manufacture:

* **I18 group gap** needs the schema to say which columns define a subgroup.
  Picking one from the data would be the benchmark deciding what "protected"
  means.
* **I20 label stability** needs native draws, because it asks whether the
  downstream label flips across posterior samples.
* **I21 missingness-shift degradation** needs a *pair* of matched runs at
  different mechanisms or rates. It is a property of two runs and cannot be
  read off one.
* **I22 Rubin validity** needs a pre-registered scientific model. Rubin's rules
  pool an estimand; without one declared in advance there is nothing to pool,
  and choosing an estimand after seeing the completions invalidates the
  inference the metric is meant to check.

When the precondition is absent these return a ``skipped`` result naming it,
never a number.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from otbench.metrics.catalog import MetricResult

GROUP_GAP = "imputation.fairness.group_gap"
LABEL_STABILITY = "imputation.stability.label_stability"
SHIFT_DEGRADATION = "imputation.robustness.missingness_shift_degradation"
RUBIN_VALIDITY = "imputation.inference.rubin_validity"

# The point/structure metrics whose worst-group gap I18 reports.
GROUP_SOURCE_METRICS = (
    "imputation.point.macro_nrmse",
    "imputation.point.macro_nmae",
    "imputation.point.categorical_macro_f1",
    "imputation.point.categorical_balanced_accuracy",
    "imputation.structure.numeric_wasserstein1",
    "imputation.structure.categorical_jsd",
)

MIN_GROUP_ROWS = 20


@dataclass
class PreparedLabelStability:
    """Train-fitted nuisance classifier shared across missingness settings."""

    pipeline: Any | None = None
    reason: str | None = None
    error: Exception | None = None


def evaluate_group_gap(
    clean_test: pd.DataFrame,
    imputed: pd.DataFrame,
    mask_bundle,
    schema,
    train: pd.DataFrame,
    overall: list[MetricResult],
) -> MetricResult:
    """I18: worst-group minus overall value of the point/structure metrics."""
    from .metrics import evaluate_point_metrics

    if not schema.group_cols:
        return MetricResult.skipped(
            GROUP_GAP,
            "the schema declares no group columns; a fairness slice requires the dataset "
            "definition to say which attribute defines the groups",
        )
    available = [c for c in schema.group_cols if c in clean_test.columns]
    if not available:
        return MetricResult.skipped(
            GROUP_GAP, f"declared group columns {schema.group_cols} are absent from the test frame"
        )

    overall_values = {r.name: r.value for r in overall if r.status == "ok"}
    per_column: dict[str, Any] = {}
    worst_gaps: list[float] = []

    for col in available:
        groups = clean_test.groupby(clean_test[col].astype(str)).groups
        per_group: dict[str, dict[str, float]] = {}
        for value, index in groups.items():
            rows = np.asarray(index)
            if len(rows) < MIN_GROUP_ROWS:
                continue
            sliced = evaluate_point_metrics(
                clean_test.iloc[rows], imputed.iloc[rows],
                _slice_mask(mask_bundle, rows), schema, train, selected=set(GROUP_SOURCE_METRICS),
            )
            per_group[str(value)] = {
                r.name: r.value for r in sliced
                if r.status == "ok" and r.name in GROUP_SOURCE_METRICS
            }
        if len(per_group) < 2:
            continue

        gaps = {}
        for metric in GROUP_SOURCE_METRICS:
            values = [g[metric] for g in per_group.values() if metric in g]
            if len(values) < 2 or metric not in overall_values:
                continue
            # Worst group is the one furthest from the overall value, in the
            # direction that means "worse", so the sign is metric-agnostic.
            gaps[metric] = float(max(abs(v - overall_values[metric]) for v in values))
        if gaps:
            per_column[col] = {"per_group": per_group, "gaps": gaps}
            worst_gaps.append(max(gaps.values()))

    if not worst_gaps:
        return MetricResult.skipped(
            GROUP_GAP,
            f"no declared group column had two subgroups with at least {MIN_GROUP_ROWS} rows "
            f"and a shared computable metric",
        )
    return MetricResult.ok(
        GROUP_GAP, float(max(worst_gaps)), n=len(per_column),
        details={"per_column": per_column},
        extra={"aggregation": "worst absolute deviation of any subgroup from the overall "
                              "value, maximised over declared group columns and metrics",
               "min_group_rows": MIN_GROUP_ROWS},
    )


def evaluate_label_stability(
    train: pd.DataFrame,
    draws: list[pd.DataFrame],
    schema,
    random_state: int = 0,
    native_draws: bool = True,
    protocol: dict | None = None,
    prepared: PreparedLabelStability | None = None,
) -> MetricResult:
    """I20: how often the downstream predicted label survives a different draw."""
    if not native_draws:
        return MetricResult.skipped(
            LABEL_STABILITY,
            "the method has no native draws capability; label agreement across repeated "
            "deterministic runs is trivially 1 and says nothing about stability",
        )
    if len(draws) < 2:
        return MetricResult.skipped(
            LABEL_STABILITY, f"K={len(draws)}: stability needs at least two draws"
        )
    if not protocol or not protocol.get("pre_registered"):
        return MetricResult.skipped(
            LABEL_STABILITY,
            "no pre-registered stability protocol; the downstream model and draw count "
            "must be fixed before inspecting the completions",
        )
    task = schema.task_type or (
        "regression" if schema.target_col in schema.numerical_cols else "classification"
    )
    if task != "classification":
        return MetricResult.skipped(
            LABEL_STABILITY, "label stability is defined for classification targets only"
        )

    prepared = prepared or prepare_label_stability(train, schema, random_state)
    if prepared.reason is not None:
        return MetricResult.skipped(LABEL_STABILITY, prepared.reason)
    if prepared.error is not None:
        return MetricResult.failed(LABEL_STABILITY, prepared.error)
    assert prepared.pipeline is not None

    from otbench.methods.common.data import split_features_target

    try:
        predictions = np.column_stack([
            prepared.pipeline.predict(
                split_features_target(d.reset_index(drop=True), schema).X
            )
            for d in draws
        ])
    except Exception as exc:  # noqa: BLE001
        return MetricResult.failed(LABEL_STABILITY, exc)

    identical = (predictions == predictions[:, [0]]).all(axis=1)
    return MetricResult.ok(
        LABEL_STABILITY, float(identical.mean()), n=len(identical),
        extra={"n_draws": len(draws), "model": "logistic_regression",
               "definition": "share of rows whose predicted label is the same under every draw",
               "random_state": random_state, "protocol": protocol},
    )


def prepare_label_stability(
    train: pd.DataFrame,
    schema,
    random_state: int = 0,
) -> PreparedLabelStability:
    """Fit the invariant stability classifier without scoring any draw."""

    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder

    from otbench.methods.common.data import split_features_target

    from .downstream import _preprocessor

    train_split = split_features_target(train.reset_index(drop=True), schema)
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_split.y.astype(str))
    if len(encoder.classes_) < 2:
        return PreparedLabelStability(reason="the training target has a single class")

    pipe = Pipeline([
        ("preprocess", _preprocessor(schema)),
        # Eight of the datasets ship native missing values, and columns the run
        # did not mask keep them, so the nuisance classifier would refuse to fit.
        # Imputing the transformed matrix keeps the metric about label agreement
        # across draws instead of about the training frame's own gaps.
        ("impute", SimpleImputer(strategy="median")),
        ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
    ])
    try:
        pipe.fit(train_split.X, y_train)
    except Exception as exc:  # noqa: BLE001
        return PreparedLabelStability(error=exc)
    return PreparedLabelStability(pipeline=pipe)


def evaluate_shift_degradation(matched_runs: list[dict] | None) -> MetricResult:
    """I21: paired degradation between matched runs at different missingness.

    ``matched_runs`` is a list of ``{"scenario", "rate", "metrics": {name: value}}``
    records from runs that differ *only* in the missingness mechanism or rate.
    """
    if not matched_runs or len(matched_runs) < 2:
        return MetricResult.skipped(
            SHIFT_DEGRADATION,
            "missingness-shift degradation is a property of two matched runs at different "
            "mechanisms or rates; it cannot be computed from a single run",
        )
    matched_set_ids = {run.get("matched_set_id") for run in matched_runs}
    if None in matched_set_ids or len(matched_set_ids) != 1:
        return MetricResult.skipped(
            SHIFT_DEGRADATION,
            "matched runs must carry one shared matched_set_id proving that method, dataset "
            "and seeds were held fixed",
        )
    conditions = {(run.get("scenario"), float(run.get("rate", 0.0))) for run in matched_runs}
    if len(conditions) < 2:
        return MetricResult.skipped(
            SHIFT_DEGRADATION,
            "matched runs do not differ in missingness mechanism or rate",
        )
    baseline, *rest = sorted(matched_runs, key=lambda r: (float(r.get("rate", 0.0)),
                                                          str(r.get("scenario", ""))))
    tracked = ("imputation.point.macro_nrmse", "imputation.point.categorical_macro_f1")
    per_run = []
    degradations = []
    for run in rest:
        record = {"scenario": run.get("scenario"), "rate": run.get("rate"), "deltas": {}}
        for metric in tracked:
            base = baseline.get("metrics", {}).get(metric)
            value = run.get("metrics", {}).get(metric)
            if base is None or value is None:
                continue
            # NRMSE up and F1 down both mean "worse"; sign so positive = worse.
            delta = (value - base) if metric.endswith("nrmse") else (base - value)
            record["deltas"][metric] = delta
            degradations.append(delta)
        per_run.append(record)

    if not degradations:
        return MetricResult.skipped(
            SHIFT_DEGRADATION, "the matched runs share no computable point metric"
        )
    return MetricResult.ok(
        SHIFT_DEGRADATION, float(max(degradations)), n=len(per_run),
        details={"baseline": {"scenario": baseline.get("scenario"), "rate": baseline.get("rate")},
                 "per_run": per_run},
        extra={"aggregation": "worst paired degradation against the lowest-rate run; "
                              "positive means the harder missingness hurt",
               "matched_set_id": next(iter(matched_set_ids))},
    )


def evaluate_rubin_validity(
    draws: list[pd.DataFrame],
    scientific_model: dict | None,
    truth: pd.DataFrame | None = None,
    native_draws: bool = True,
) -> MetricResult:
    """I22: Rubin's rules (Rubin 1987) applied to a pre-registered estimand.

    ``scientific_model`` must declare the estimand in advance, e.g.
    ``{"kind": "ols", "outcome": "y", "predictors": ["x1", "x2"], "coefficient": "x1"}``
    or ``{"kind": "mean", "column": "x1"}``. The result reports the pooled
    estimate, its Rubin standard error, and — when the clean data is available —
    the bias and whether the 95% interval covers the complete-data value.
    """
    if not native_draws:
        return MetricResult.skipped(
            RUBIN_VALIDITY,
            "Rubin's rules pool the between-draw variance of a posterior sample; repeated "
            "deterministic runs have none and would report a spuriously tiny standard error",
        )
    if not scientific_model:
        return MetricResult.skipped(
            RUBIN_VALIDITY,
            "no pre-registered scientific model; Rubin's rules need an estimand declared "
            "before the completions are seen, otherwise the inference being validated is "
            "chosen after the fact",
        )
    if len(draws) < 2:
        return MetricResult.skipped(
            RUBIN_VALIDITY, f"K={len(draws)}: pooling needs at least two completions"
        )

    try:
        estimates, variances = zip(*(_estimand(d, scientific_model) for d in draws), strict=True)
    except _EstimandError as exc:
        return MetricResult.skipped(RUBIN_VALIDITY, str(exc))

    k = len(estimates)
    q_bar = float(np.mean(estimates))               # pooled estimate
    u_bar = float(np.mean(variances))               # within-imputation variance
    b = float(np.var(estimates, ddof=1))            # between-imputation variance
    total = u_bar + (1.0 + 1.0 / k) * b             # Rubin's total variance
    se = float(np.sqrt(total))

    # Barnard-Rubin degrees of freedom would need the complete-data df; the
    # classical Rubin (1987) value is used and recorded as such.
    if b > 0:
        df = (k - 1) * (1.0 + u_bar / ((1.0 + 1.0 / k) * b)) ** 2
    else:
        df = float("inf")
    t_crit = float(stats.t.ppf(0.975, df)) if np.isfinite(df) else 1.959963985
    lower, upper = q_bar - t_crit * se, q_bar + t_crit * se
    fraction_missing = ((1.0 + 1.0 / k) * b) / total if total > 0 else 0.0

    details: dict[str, Any] = {
        "pooled_estimate": q_bar,
        "within_variance": u_bar,
        "between_variance": b,
        "total_variance": total,
        "standard_error": se,
        "degrees_of_freedom": df if np.isfinite(df) else None,
        "interval_95": [lower, upper],
        "fraction_missing_information": fraction_missing,
        "per_draw_estimate": [float(e) for e in estimates],
        "estimand": scientific_model,
    }

    if truth is None:
        return MetricResult.ok(
            RUBIN_VALIDITY, float(fraction_missing), n=k, details=details,
            extra={"reported": "fraction of missing information; estimate bias and interval "
                               "coverage need the clean data, which was not supplied",
                   "source": "Rubin (1987), Multiple Imputation for Nonresponse in Surveys"},
        )

    try:
        truth_estimate, _ = _estimand(truth, scientific_model)
    except _EstimandError as exc:
        return MetricResult.skipped(RUBIN_VALIDITY, f"the estimand failed on the clean data: {exc}")

    bias = q_bar - truth_estimate
    details.update({
        "complete_data_estimate": float(truth_estimate),
        "bias": float(bias),
        "standardized_bias": float(bias / se) if se > 0 else None,
        "interval_covers_truth": bool(lower <= truth_estimate <= upper),
    })
    if se <= 0:
        return MetricResult.skipped(
            RUBIN_VALIDITY,
            "Rubin total variance is zero, so standardised bias is undefined",
            n=k, details=details,
        )
    return MetricResult.ok(
        RUBIN_VALIDITY, float(abs(bias / se)), n=k, details=details,
        extra={"reported": "absolute standardised bias of the pooled estimate; 0 is ideal and "
                           "values above ~2 mean the interval systematically misses",
               "source": "Rubin (1987), Multiple Imputation for Nonresponse in Surveys"},
    )


class _EstimandError(RuntimeError):
    pass


def _estimand(frame: pd.DataFrame, model: dict) -> tuple[float, float]:
    """(point estimate, its sampling variance) for one completed dataset."""
    kind = model.get("kind", "mean")

    if kind == "mean":
        column = model["column"]
        if column not in frame.columns:
            raise _EstimandError(f"the estimand column {column!r} is not in the completed data")
        values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(float)
        if len(values) < 2:
            raise _EstimandError(f"column {column!r} has fewer than two numeric values")
        return float(values.mean()), float(values.var(ddof=1) / len(values))

    if kind == "ols":
        outcome, predictors = model["outcome"], list(model["predictors"])
        target = model.get("coefficient", predictors[0])
        missing = [c for c in [outcome, *predictors] if c not in frame.columns]
        if missing:
            raise _EstimandError(f"the estimand needs columns {missing}, absent from the data")
        y = pd.to_numeric(frame[outcome], errors="coerce").to_numpy(float)
        x = pd.get_dummies(frame[predictors], drop_first=True).astype(float)
        if target not in x.columns:
            raise _EstimandError(f"coefficient {target!r} is not in the design matrix {list(x.columns)}")
        design = np.column_stack([np.ones(len(x)), x.to_numpy(float)])
        finite = np.isfinite(y) & np.isfinite(design).all(axis=1)
        y, design = y[finite], design[finite]
        if len(y) <= design.shape[1]:
            raise _EstimandError("too few complete rows for the declared regression")
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ beta
        sigma2 = float(residual @ residual) / (len(y) - design.shape[1])
        covariance = sigma2 * np.linalg.pinv(design.T @ design)
        position = list(x.columns).index(target) + 1  # +1 for the intercept
        return float(beta[position]), float(covariance[position, position])

    raise _EstimandError(f"unsupported estimand kind {kind!r}; use 'mean' or 'ols'")


def _slice_mask(mask_bundle, rows: np.ndarray):
    """A MaskBundle view restricted to ``rows`` (used for subgroup slices)."""
    from dataclasses import replace

    return replace(
        mask_bundle,
        mask=mask_bundle.mask.iloc[rows].reset_index(drop=True),
        masked=mask_bundle.masked.iloc[rows].reset_index(drop=True),
    )

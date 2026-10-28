# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""P01-P20: prediction metrics, gated on what the method actually emits.

The gating is the substance of this module, not a technicality. A method that
returns only point predictions cannot be scored on log loss; a method that
returns only samples has no normalised density, so it gets CRPS but not NLL.
The alternative — fitting a KDE to the samples and calling the result the
model's likelihood — would report a bandwidth choice as if it were the model's
own calibration, and would rank a sample-only method against a density model on
a quantity only one of them produced.

Two other conventions:

* Test labels outside the train label domain make the classification task
  unsupported on that split and are counted separately. Treating them as an
  extra class would silently change every macro denominator; scoring only the
  known-label subset would silently change the test population.
* The decision threshold is never tuned on test. It comes from the method (which
  must say where it got it) or defaults to 0.5 / argmax, and the source is
  recorded next to the metrics.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    roc_auc_score,
)

from otbench.metrics.catalog import MetricResult
from otbench.metrics.stats import (
    area_under_risk_coverage,
    brier_score,
    crps_ensemble,
    expected_calibration_error,
    interval_coverage,
    interval_width,
    pinball_loss,
    quantile_crossing_rate,
    robust_scale,
)

ECE_BINS = 15
NOMINAL_LEVELS = (0.90, 0.95)

_CLASSIFICATION = (
    "prediction.classification.accuracy",
    "prediction.classification.balanced_accuracy",
    "prediction.classification.macro_f1",
    "prediction.classification.mcc",
)
_PROBA = (
    "prediction.classification.roc_auc",
    "prediction.classification.average_precision",
    "prediction.calibration.log_loss",
    "prediction.calibration.brier_score",
    "prediction.calibration.ece",
    "prediction.calibration.aurc",
)
_REGRESSION = (
    "prediction.regression.rmse",
    "prediction.regression.mae",
    "prediction.regression.r2",
    "prediction.regression.nrmse",
    "prediction.regression.median_absolute_error",
)
_SAMPLES = ("prediction.probabilistic.ensemble_crps",)
_DENSITY = ("prediction.probabilistic.nll",)
_QUANTILES = (
    "prediction.probabilistic.pinball_loss",
    "prediction.probabilistic.coverage_error_90",
    "prediction.probabilistic.coverage_error_95",
    "prediction.probabilistic.nmpiw_90",
    "prediction.probabilistic.nmpiw_95",
    "prediction.probabilistic.quantile_crossing_rate",
)
UNKNOWN_LABEL_RATE = "prediction.classification.unknown_test_label_rate"


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def evaluate_classification(
    y_true_enc: np.ndarray,
    y_pred_enc: np.ndarray,
    n_classes: int,
    y_proba: np.ndarray | None = None,
    n_unknown: int = 0,
    n_total: int | None = None,
    threshold_source: str | None = None,
) -> list[MetricResult]:
    """P01-P10 for one set of predictions, already encoded to train label codes."""
    n_total = len(y_true_enc) + n_unknown if n_total is None else n_total
    shared = {"n_classes": n_classes, "label_vocabulary_source": "train"}
    if threshold_source:
        shared["threshold_source"] = threshold_source

    results = [
        MetricResult.ok(
            UNKNOWN_LABEL_RATE, n_unknown / max(1, n_total), n=n_total,
            extra={"n_unknown": n_unknown,
                   "note": "P01-P10 are unsupported when this is nonzero; unknown labels "
                           "are neither added as a class nor removed from the test population"},
        )
    ]

    if n_unknown:
        reason = (
            f"{n_unknown}/{n_total} test labels are outside the train label domain; the "
            "classification task is unsupported on this split and those labels are not "
            "silently added to the model vocabulary"
        )
        return results + [
            MetricResult.unsupported(name, reason, extra=shared)
            for name in _CLASSIFICATION + _PROBA
        ]

    if len(y_true_enc) == 0:
        reason = "no test row carries a label from the train label domain"
        return results + [MetricResult.skipped(name, reason)
                          for name in _CLASSIFICATION + _PROBA]

    results.extend([
        MetricResult.ok("prediction.classification.accuracy",
                        float(accuracy_score(y_true_enc, y_pred_enc)),
                        n=len(y_true_enc), extra=shared),
        MetricResult.ok("prediction.classification.balanced_accuracy",
                        float(balanced_accuracy_score(y_true_enc, y_pred_enc)),
                        n=len(y_true_enc), extra=shared),
        MetricResult.ok("prediction.classification.macro_f1",
                        float(f1_score(y_true_enc, y_pred_enc, average="macro",
                                       zero_division=0)),
                        n=len(y_true_enc), extra=shared),
        MetricResult.ok("prediction.classification.mcc",
                        float(matthews_corrcoef(y_true_enc, y_pred_enc)),
                        n=len(y_true_enc), extra=shared),
    ])

    if y_proba is None:
        reason = ("the method emits no class probabilities; ranking and calibration "
                  "metrics are undefined for a hard-label output")
        return results + [MetricResult.skipped(name, reason) for name in _PROBA]
    results.extend(_probability_metrics(y_true_enc, np.asarray(y_proba, float),
                                        n_classes, shared))
    return results


def _probability_metrics(y_true, proba, n_classes, shared) -> list[MetricResult]:
    # Probability validity is part of the output protocol.  Clipping or
    # renormalising here would hide an adapter failure and make calibration look
    # better than the distribution the method actually emitted.
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])
    if proba.shape[1] != n_classes:
        reason = (f"the probability matrix has {proba.shape[1]} columns but the train label "
                  f"domain has {n_classes} classes")
        return [MetricResult.skipped(name, reason) for name in _PROBA]
    finite = np.isfinite(proba).all()
    in_range = bool(((proba >= 0.0) & (proba <= 1.0)).all()) if finite else False
    normalized = bool(np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)) if finite else False
    if not (finite and in_range and normalized):
        error = ValueError(
            "invalid class probabilities: values must be finite, lie in [0, 1], "
            "and every row must sum to one"
        )
        return [MetricResult.failed(name, error) for name in _PROBA]

    present = np.unique(y_true)
    results: list[MetricResult] = []

    if len(present) < 2:
        reason = "only one class is present among the scored test rows"
        results.extend(MetricResult.skipped(name, reason) for name in
                       ("prediction.classification.roc_auc",
                        "prediction.classification.average_precision"))
    else:
        results.append(_ranking("prediction.classification.roc_auc", y_true, proba,
                                n_classes, present, roc_auc_score, shared))
        results.append(_ranking("prediction.classification.average_precision", y_true, proba,
                                n_classes, present, average_precision_score, shared))

    results.append(MetricResult.ok(
        "prediction.calibration.log_loss",
        float(log_loss(y_true, proba, labels=list(range(n_classes)))),
        n=len(y_true), extra=shared,
    ))
    results.append(MetricResult.ok(
        "prediction.calibration.brier_score", brier_score(y_true, proba),
        n=len(y_true),
        extra={**shared, "form": "multiclass: mean over rows of sum_k (p_k - [y=k])^2; on "
                                 "binary data this is twice the classic binary Brier"},
    ))

    ece, bins = expected_calibration_error(y_true, proba, n_bins=ECE_BINS)
    results.append(MetricResult.ok(
        "prediction.calibration.ece", ece, n=len(y_true),
        details={"reliability_bins": bins},
        extra={**shared, "n_bins": ECE_BINS, "binning": "equal_width_top_label",
               "note": "diagnostic only; ECE is not a proper scoring rule and cannot "
                       "replace log loss or Brier"},
    ))

    confidence = proba.max(axis=1)
    losses = (proba.argmax(axis=1) != y_true).astype(float)
    results.append(MetricResult.ok(
        "prediction.calibration.aurc", area_under_risk_coverage(losses, confidence),
        n=len(y_true), extra={**shared, "risk": "0/1 loss", "confidence": "max probability"},
    ))
    return results


def _ranking(name, y_true, proba, n_classes, present, scorer, shared) -> MetricResult:
    try:
        if n_classes == 2:
            value = float(scorer(y_true, proba[:, 1]))
        elif scorer is roc_auc_score:
            value = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro",
                                        labels=list(range(n_classes))))
        else:
            onehot = np.eye(n_classes)[y_true]
            # Classes absent from the scored rows have no defined AP; dropping
            # them keeps the macro average over classes that exist here.
            columns = [c for c in range(n_classes) if c in set(present.tolist())]
            value = float(average_precision_score(onehot[:, columns], proba[:, columns],
                                                  average="macro"))
    except ValueError as exc:
        return MetricResult.skipped(name, f"undefined for this split: {exc}")
    return MetricResult.ok(name, value, n=len(y_true),
                           extra={**shared, "averaging": "binary" if n_classes == 2
                                  else "macro one-vs-rest"})


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_target: np.ndarray | None = None,
) -> list[MetricResult]:
    """P11-P15. The NRMSE normaliser is fitted on the train target only."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    n_dropped = int((~finite).sum())
    y_true, y_pred = y_true[finite], y_pred[finite]

    if len(y_true) == 0:
        reason = "no test row has both a finite target and a finite prediction"
        return [MetricResult.skipped(name, reason) for name in _REGRESSION]

    shared: dict[str, Any] = {"n_non_finite_dropped": n_dropped}
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    results = [
        MetricResult.ok("prediction.regression.rmse", rmse, n=len(y_true), extra=shared),
        MetricResult.ok("prediction.regression.mae",
                        float(mean_absolute_error(y_true, y_pred)), n=len(y_true), extra=shared),
        MetricResult.ok("prediction.regression.r2",
                        float(r2_score(y_true, y_pred)), n=len(y_true),
                        extra={**shared, "note": "negative values are informative, not errors"}),
        MetricResult.ok("prediction.regression.median_absolute_error",
                        float(median_absolute_error(y_true, y_pred)), n=len(y_true),
                        extra=shared),
    ]

    scale, kind = _target_scale(train_target)
    if scale is None:
        results.append(MetricResult.skipped(
            "prediction.regression.nrmse",
            "the train target is constant or unavailable, so no train-fitted normaliser "
            "exists; the test target's own spread must not be substituted",
        ))
    else:
        results.append(MetricResult.ok(
            "prediction.regression.nrmse", rmse / scale, n=len(y_true),
            extra={**shared, "normalizer": scale, "normalizer_source": kind},
        ))
    return results


def evaluate_probabilistic_regression(
    y_true: np.ndarray,
    y_samples: np.ndarray | None = None,
    log_prob: np.ndarray | None = None,
    log_prob_error: BaseException | None = None,
    quantiles: np.ndarray | None = None,
    quantile_levels: list[float] | None = None,
    train_target: np.ndarray | None = None,
) -> list[MetricResult]:
    """P16-P20, each gated on the corresponding native output being present."""
    y_true = np.asarray(y_true, float)
    results: list[MetricResult] = []

    # --- P16: CRPS, from native samples of p(y|x) --------------------------
    if y_samples is None:
        results.append(MetricResult.skipped(
            _SAMPLES[0],
            "the method emits no samples from the predictive distribution; CRPS of a point "
            "prediction would just be the MAE under a different name",
        ))
    else:
        samples = np.asarray(y_samples, float)
        if samples.ndim == 1:
            samples = samples[:, None]
        rows = np.isfinite(y_true) & np.isfinite(samples).all(axis=1)
        if rows.sum() == 0 or samples.shape[1] < 2:
            results.append(MetricResult.skipped(
                _SAMPLES[0], f"only {samples.shape[1]} finite sample(s) per row"))
        else:
            results.append(MetricResult.ok(
                _SAMPLES[0], float(np.mean(crps_ensemble(y_true[rows], samples[rows]))),
                n=int(rows.sum()),
                extra={"n_samples": samples.shape[1],
                       "source": "Gneiting & Raftery (2007), eq. 21"},
            ))

    # --- P17: NLL, only from a normalised density --------------------------
    if log_prob_error is not None:
        results.append(MetricResult.failed(_DENSITY[0], log_prob_error))
    elif log_prob is None:
        results.append(MetricResult.skipped(
            _DENSITY[0],
            "the method exposes no normalised log-density; no KDE is fitted to its samples, "
            "because that would report a bandwidth choice as the model's own likelihood",
        ))
    else:
        values = np.asarray(log_prob, float)
        finite = np.isfinite(values)
        if finite.sum() == 0:
            results.append(MetricResult.skipped(
                _DENSITY[0], "every reported log-density was non-finite"))
        else:
            results.append(MetricResult.ok(
                _DENSITY[0], float(-np.mean(values[finite])), n=int(finite.sum()),
                extra={"n_non_finite_dropped": int((~finite).sum()),
                       "source": "the method's own log p(y|x) at the test target"},
            ))

    # --- P18-P20: quantile-based -------------------------------------------
    if quantiles is None or not quantile_levels:
        reason = ("the method emits no predictive quantiles; interval coverage and width "
                  "cannot be derived from a point prediction")
        results.extend(MetricResult.skipped(name, reason) for name in _QUANTILES)
        return results

    q = np.asarray(quantiles, float)
    levels = list(quantile_levels)
    order = np.argsort(levels)
    levels = [levels[i] for i in order]
    q = q[:, order]

    results.append(MetricResult.ok(
        "prediction.probabilistic.pinball_loss",
        float(np.mean(pinball_loss(y_true, q, levels))), n=len(y_true),
        details={"per_level": dict(zip([str(l) for l in levels],
                                       [float(v) for v in pinball_loss(y_true, q, levels)],
                                       strict=True))},
        extra={"levels": levels},
    ))
    crossing = quantile_crossing_rate(q)
    results.append(MetricResult.ok(
        "prediction.probabilistic.quantile_crossing_rate", crossing, n=len(y_true),
        extra={"note": "independently fitted quantile models are not monotone by "
                       "construction; a non-zero rate invalidates the interval metrics "
                       "below and is reported rather than silently sorted away"},
    ))

    scale, scale_kind = _target_scale(train_target)
    for level, key in zip(NOMINAL_LEVELS, ("90", "95"), strict=True):
        if crossing > 0:
            reason = (
                f"quantile_crossing_rate={crossing:.6g}; crossed quantiles do not define "
                "valid predictive intervals and are not silently sorted"
            )
            results.append(MetricResult.skipped(
                f"prediction.probabilistic.coverage_error_{key}", reason))
            results.append(MetricResult.skipped(
                f"prediction.probabilistic.nmpiw_{key}", reason))
            continue
        bounds = _interval_from_quantiles(q, levels, level)
        if bounds is None:
            reason = (f"the quantile grid {levels} does not contain the "
                      f"{(1 - level) / 2:.3f} and {1 - (1 - level) / 2:.3f} levels needed "
                      f"for a {level:.0%} interval")
            results.append(MetricResult.skipped(
                f"prediction.probabilistic.coverage_error_{key}", reason))
            results.append(MetricResult.skipped(
                f"prediction.probabilistic.nmpiw_{key}", reason))
            continue
        lower, upper = bounds
        coverage = interval_coverage(y_true, lower, upper)
        results.append(MetricResult.ok(
            f"prediction.probabilistic.coverage_error_{key}", coverage - level,
            n=len(y_true),
            extra={"nominal": level, "empirical_coverage": coverage,
                   "quantile_crossing_rate": crossing},
        ))
        if scale is None:
            results.append(MetricResult.skipped(
                f"prediction.probabilistic.nmpiw_{key}",
                "no train-fitted normaliser for the target scale",
            ))
        else:
            results.append(MetricResult.ok(
                f"prediction.probabilistic.nmpiw_{key}",
                interval_width(lower, upper, scale=scale), n=len(y_true),
                extra={"normalizer": scale, "normalizer_source": scale_kind,
                       "read_with": f"prediction.probabilistic.coverage_error_{key}"},
            ))
    return results


def _interval_from_quantiles(q, levels, nominal) -> tuple[np.ndarray, np.ndarray] | None:
    """Exact lower/upper columns for a nominal interval, or None if absent.

    Interpolating between the available levels would invent a quantile the model
    never predicted, so a grid that does not contain the needed levels is a skip.
    """
    tail = (1.0 - nominal) / 2.0
    lower_idx = _closest(levels, tail)
    upper_idx = _closest(levels, 1.0 - tail)
    if lower_idx is None or upper_idx is None or lower_idx == upper_idx:
        return None
    return q[:, lower_idx], q[:, upper_idx]


def _closest(levels, target, tolerance: float = 1e-6) -> int | None:
    for index, level in enumerate(levels):
        if abs(level - target) <= tolerance:
            return index
    return None


def _target_scale(train_target) -> tuple[float | None, str]:
    if train_target is None:
        return None, "unavailable"
    values = pd.to_numeric(pd.Series(np.asarray(train_target).reshape(-1)),
                           errors="coerce").dropna().to_numpy(float)
    if len(values) < 2:
        return None, "unavailable"
    scale, kind = robust_scale(values)
    if scale is None or scale <= 0:
        return None, "constant_target"
    return float(scale), f"train_{kind}"

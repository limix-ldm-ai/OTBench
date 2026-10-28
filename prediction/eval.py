# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Prediction evaluation dispatcher.

Takes one :class:`PredictionOutput` and produces exactly one result per catalog
metric for the task, so a report never has to distinguish "the metric was not
in this run's output" from "the metric was not applicable".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from otbench.methods.common.base import PredictionOutput
from otbench.metrics import load_all_specs
from otbench.metrics.catalog import CATALOG, MetricResult
from otbench.metrics.cost import evaluate_cost

from .metrics import (
    evaluate_classification,
    evaluate_probabilistic_regression,
    evaluate_regression,
)
from .robustness import evaluate_adversarial, evaluate_ood

TASK = "prediction"


@dataclass(frozen=True)
class PredictionEvalOptions:
    metric: str = "all"
    profile: dict | None = None
    ood: dict | None = None
    attack_artifact: dict | None = None


def resolve_metrics(metric_arg: str) -> set[str]:
    load_all_specs()
    available = set(CATALOG.names(TASK))
    selected: set[str] = set()
    unknown: list[str] = []
    for token in (part.strip() for part in metric_arg.split(",")):
        if not token:
            continue
        if token == "all":
            selected |= available
            continue
        try:
            selected |= CATALOG.select(token, TASK)
        except ValueError:
            unknown.append(token)
    if unknown:
        raise ValueError(
            f"Unknown metric selector(s) {unknown}. Families: {CATALOG.families(TASK)}"
        )
    return selected or available


def evaluate_prediction(
    y_true: pd.Series,
    output: PredictionOutput,
    schema,
    train: pd.DataFrame,
    options: PredictionEvalOptions = PredictionEvalOptions(),
) -> list[MetricResult]:
    load_all_specs()
    wanted = resolve_metrics(options.metric)
    task = schema.task_type or (
        "regression" if schema.target_col in schema.numerical_cols else "classification"
    )
    results: list[MetricResult] = []

    def run(families: tuple[str, ...], compute) -> None:
        names = {n for n in wanted if CATALOG.require(n).family in families}
        if not names:
            return
        try:
            produced = compute()
        except Exception as exc:  # noqa: BLE001
            produced = [MetricResult.failed(name, exc) for name in sorted(names)]
        results.extend(r for r in produced if r.name in wanted)

    train_target = train[schema.target_col].to_numpy() if schema.target_col in train else None

    if task == "regression":
        y = pd.to_numeric(y_true, errors="coerce").to_numpy(float)
        log_prob, log_prob_error = _log_prob(output, y)
        run(("regression",), lambda: evaluate_regression(
            y, np.asarray(output.y_pred, float), train_target))
        run(("probabilistic",), lambda: evaluate_probabilistic_regression(
            y,
            y_samples=output.y_samples,
            log_prob=log_prob,
            log_prob_error=log_prob_error,
            quantiles=output.quantiles,
            quantile_levels=output.quantile_levels,
            train_target=train_target,
        ))
        reason = "the dataset target is numerical; classification metrics do not apply"
        run(("classification", "calibration"), lambda: [
            MetricResult.skipped(n, reason) for n in sorted(
                {n for n in wanted
                 if CATALOG.require(n).family in ("classification", "calibration")})
        ])
    else:
        encoded = encode_labels(y_true, output, train, schema)
        run(("classification", "calibration"), lambda: evaluate_classification(
            encoded["y_true"], encoded["y_pred"], encoded["n_classes"],
            y_proba=encoded["y_proba"], n_unknown=encoded["n_unknown"],
            n_total=len(y_true), threshold_source=output.threshold_source,
        ))
        reason = "the dataset target is categorical; regression metrics do not apply"
        run(("regression", "probabilistic"), lambda: [
            MetricResult.skipped(n, reason) for n in sorted(
                {n for n in wanted
                 if CATALOG.require(n).family in ("regression", "probabilistic")})
        ])

    run(("robustness",), lambda: (
        evaluate_ood(options.ood) + evaluate_adversarial(options.attack_artifact, schema)
    ))
    run(("cost",), lambda: evaluate_cost(TASK, options.profile))

    produced = {r.name for r in results}
    for name in sorted(wanted - produced):
        results.append(MetricResult.skipped(name, "no evaluator produced this metric"))
    return sorted(results, key=lambda r: r.name)


def encode_labels(y_true: pd.Series, output: PredictionOutput, train: pd.DataFrame,
                  schema) -> dict[str, Any]:
    """Encode truth, predictions and probabilities to the TRAIN label domain.

    Unknown test labels encode to -1 and are counted; the metric layer marks
    that split unsupported rather than reporting a partial score. The probability
    matrix is reordered to the train class order so column ``k`` always means
    the same class regardless of how the method ordered its own outputs.
    """
    classes = (
        pd.Series(train[schema.target_col].dropna().unique())
        .sort_values(kind="stable").tolist()
    )
    to_code = {value: index for index, value in enumerate(classes)}
    to_code_str = {str(value): index for index, value in enumerate(classes)}

    def encode(values) -> np.ndarray:
        codes = []
        for value in np.asarray(values, dtype=object).reshape(-1):
            codes.append(to_code.get(value, to_code_str.get(str(value), -1)))
        return np.asarray(codes, dtype=np.int64)

    y_true_enc = encode(y_true)
    y_pred_enc = encode(output.y_pred)
    known = y_true_enc >= 0

    proba = None
    probability_output = output.y_proba
    probability_labels = output.labels
    if probability_output is None and output.y_score is not None and len(classes) == 2:
        score = np.asarray(output.y_score, dtype=float).reshape(-1)
        probability_output = np.column_stack([1.0 - score, score])
        probability_labels = list(classes)
    if probability_output is not None:
        raw = np.asarray(probability_output, dtype=float)
        if raw.ndim == 1:
            raw = np.column_stack([1.0 - raw, raw])
        if probability_labels is not None:
            columns = encode(probability_labels)
            aligned = np.zeros((len(raw), len(classes)), dtype=float)
            for source, target in enumerate(columns):
                if target >= 0 and source < raw.shape[1]:
                    aligned[:, target] = raw[:, source]
            proba = aligned
        elif raw.shape[1] == len(classes):
            proba = raw
        # A probability matrix with neither labels nor a matching width cannot be
        # aligned; leaving it None makes the calibration metrics skip with a reason
        # rather than score the wrong columns.
        if proba is not None:
            proba = proba[known]

    return {
        "y_true": y_true_enc[known],
        "y_pred": y_pred_enc[known],
        "y_proba": proba,
        "n_classes": len(classes),
        "n_unknown": int((~known).sum()),
        "known": known,
        "classes": classes,
    }


def _log_prob(output: PredictionOutput, y_true: np.ndarray):
    """The method's own log p(y|x) at the test targets, if it exposes one."""
    if output.log_prob_fn is None:
        return None, None
    try:
        return np.asarray(output.log_prob_fn(y_true), dtype=float), None
    except Exception as exc:  # noqa: BLE001 — surfaced as failed NLL, not unsupported
        return None, exc


def summarize(results: list[MetricResult]) -> dict[str, Any]:
    return {
        "metrics": {r.name: r.value for r in results if r.status == "ok"},
        "results": [r.to_dict() for r in results],
        "skipped": {r.name: r.reason for r in results if r.status != "ok"},
    }

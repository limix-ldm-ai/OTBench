# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""P21-P24: OOD and adversarial robustness, both strictly precondition-gated.

Neither of these can be manufactured from a standard train/test split:

* An **OOD split** is a claim about where the distribution shifts. A random
  resplit is not one, and the degradation measured against a random split says
  nothing about robustness. The split must come from the dataset definition
  (``schema.ood_split``) or a declared time column.
* An **attack artifact** is the output of an actual attack run. This module
  scores an artifact someone else produced; it does not run attacks. Its main
  job is to keep "the attack was invalid" separate from "the attack failed",
  because they point at opposite conclusions: the first says the evaluation is
  broken, the second says the model is robust.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from otbench.metrics.catalog import (
    CATALOG,
    HIGHER,
    LOWER,
    TARGET_HALF,
    TARGET_ZERO,
    MetricResult,
)

OOD_DEGRADATION = "prediction.robustness.ood_degradation"
OOD_DETECTION = "prediction.robustness.ood_detection_auroc"
ROBUST_ACCURACY = "prediction.robustness.adversarial_robust_accuracy"
ATTACK_VALIDITY = "prediction.robustness.attack_validity_rate"


def evaluate_ood(ood: dict | None) -> list[MetricResult]:
    """P21-P22 from a completed OOD run.

    ``ood`` is ``{"iid": {metric: value}, "ood": {metric: value},
    "uncertainty": {"iid": [...], "ood": [...]}, "split_source": "..."}``.
    """
    if not ood:
        reason = (
            "no OOD split was declared; a random resplit is not a distribution shift, and "
            "degradation measured against one would not be a robustness result"
        )
        return [MetricResult.skipped(OOD_DEGRADATION, reason),
                MetricResult.skipped(OOD_DETECTION, reason)]
    if not ood.get("split_source"):
        reason = (
            "the OOD payload has no declared split_source; robustness cannot be inferred "
            "from two arbitrary result dictionaries"
        )
        return [MetricResult.skipped(OOD_DEGRADATION, reason),
                MetricResult.skipped(OOD_DETECTION, reason)]

    results = [_degradation(ood), _detection(ood)]
    return results


def _degradation(ood: dict) -> MetricResult:
    iid, shifted = ood.get("iid") or {}, ood.get("ood") or {}
    shared = [k for k in iid if k in shifted]
    if not shared:
        return MetricResult.skipped(
            OOD_DEGRADATION, "the IID and OOD runs share no comparable metric"
        )
    directions = ood.get("metric_directions") or {}
    deltas = {}
    used_directions = {}
    for key in shared:
        if iid[key] is None or shifted[key] is None:
            continue
        direction = directions.get(key) or _metric_direction(key)
        if direction is None:
            continue
        left, right = float(iid[key]), float(shifted[key])
        if direction == HIGHER:
            delta = left - right
        elif direction == LOWER:
            delta = right - left
        elif direction == TARGET_ZERO:
            delta = abs(right) - abs(left)
        elif direction == TARGET_HALF:
            delta = abs(right - 0.5) - abs(left - 0.5)
        else:
            continue
        deltas[key] = float(delta)
        used_directions[key] = direction
    if not deltas:
        return MetricResult.skipped(
            OOD_DEGRADATION, "the shared metrics have no value on one of the two splits"
        )
    headline = max(deltas, key=lambda k: abs(deltas[k]))
    return MetricResult.ok(
        OOD_DEGRADATION, deltas[headline], n=len(deltas),
        details={"per_metric": deltas, "directions": used_directions,
                 "iid": iid, "ood": shifted},
        extra={"headline_metric": headline,
               "split_source": ood.get("split_source", "declared"),
               "definition": "direction-normalized degradation; positive means the shift hurt"},
    )


def _detection(ood: dict) -> MetricResult:
    uncertainty = ood.get("uncertainty") or {}
    iid_scores = np.asarray(uncertainty.get("iid", []), dtype=float)
    ood_scores = np.asarray(uncertainty.get("ood", []), dtype=float)
    if len(iid_scores) == 0 or len(ood_scores) == 0:
        return MetricResult.skipped(
            OOD_DETECTION,
            "the run stored no per-row uncertainty for both splits; detection AUROC needs "
            "the model's own uncertainty, not a post-hoc proxy",
        )
    from sklearn.metrics import roc_auc_score

    labels = np.concatenate([np.zeros(len(iid_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([iid_scores, ood_scores])
    finite = np.isfinite(scores)
    if finite.sum() < 2 or len(np.unique(labels[finite])) < 2:
        return MetricResult.skipped(OOD_DETECTION, "too few finite uncertainty values")
    return MetricResult.ok(
        OOD_DETECTION, float(roc_auc_score(labels[finite], scores[finite])),
        n=int(finite.sum()),
        extra={"uncertainty_source": uncertainty.get("source", "unspecified"),
               "note": "an attack AUROC would be named privacy.*; this one is a detection "
                       "AUROC and cannot be confused with the task AUROC"},
    )


def evaluate_adversarial(artifact: dict | None, schema) -> list[MetricResult]:
    """P23-P24 from an attack artifact produced by an actual attack run.

    ``artifact`` is ``{"x_adv": DataFrame, "y_true": [...], "y_pred_adv": [...],
    "x_clean": DataFrame, "attack": {...}}``.
    """
    if not artifact:
        reason = (
            "no attack artifact was supplied; robustness is a property of an executed "
            "attack and is not inferable from clean predictions"
        )
        return [MetricResult.skipped(ROBUST_ACCURACY, reason),
                MetricResult.skipped(ATTACK_VALIDITY, reason)]

    attack = artifact.get("attack") or {}
    if attack.get("status") in {"failed", "timeout", "oom"}:
        reason = f"the attack run ended with status={attack['status']}; no robustness claim follows"
        return [MetricResult.skipped(ROBUST_ACCURACY, reason),
                MetricResult.skipped(ATTACK_VALIDITY, reason)]

    required = ("x_adv", "y_true", "y_pred_adv")
    missing = [key for key in required if artifact.get(key) is None]
    if missing:
        reason = f"the attack artifact is missing {missing}"
        return [MetricResult.skipped(ROBUST_ACCURACY, reason),
                MetricResult.skipped(ATTACK_VALIDITY, reason)]

    x_adv = pd.DataFrame(artifact["x_adv"]).reset_index(drop=True)
    y_true = np.asarray(artifact["y_true"], dtype=object).reshape(-1)
    y_pred = np.asarray(artifact["y_pred_adv"], dtype=object).reshape(-1)
    if not (len(x_adv) == len(y_true) == len(y_pred)):
        reason = (f"the artifact is internally inconsistent: {len(x_adv)} adversarial rows, "
                  f"{len(y_true)} labels, {len(y_pred)} predictions")
        return [MetricResult.skipped(ROBUST_ACCURACY, reason),
                MetricResult.skipped(ATTACK_VALIDITY, reason)]

    valid, per_check = _feasible(x_adv, schema, artifact.get("x_clean"))
    if valid is None:
        reason = (
            "the schema declares no enforceable constraints, category domains or immutable "
            "columns; attack feasibility cannot be verified"
        )
        return [MetricResult.skipped(ROBUST_ACCURACY, reason),
                MetricResult.skipped(ATTACK_VALIDITY, reason, details={"per_check": per_check})]
    validity = MetricResult.ok(
        ATTACK_VALIDITY, float(valid.mean()), n=len(valid),
        details={"per_check": per_check,
                 "attack": artifact.get("attack", {})},
        extra={"note": "an example that violates the declared constraints is an invalid "
                       "attack, which is a broken evaluation rather than evidence of "
                       "robustness"},
    )

    if valid.sum() == 0:
        return [MetricResult.skipped(
            ROBUST_ACCURACY,
            "every attack example violated the declared feasibility constraints; the "
            "attack is invalid, which is not the same as the attack having failed",
        ), validity]

    # Only feasible examples count: an infeasible one is not an attack the model
    # has to survive, and including it would inflate robust accuracy.
    correct = (y_true[valid] == y_pred[valid])
    return [
        MetricResult.ok(
            ROBUST_ACCURACY, float(correct.mean()), n=int(valid.sum()),
            details={"n_valid": int(valid.sum()), "n_total": len(valid),
                     "attack": artifact.get("attack", {})},
            extra={"scored_on": "feasible attack examples only",
                   "invalid_excluded": int((~valid).sum())},
        ),
        validity,
    ]


def _feasible(
    x_adv: pd.DataFrame, schema, x_clean
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Which adversarial rows respect the schema's declared feasibility rules."""
    valid = np.ones(len(x_adv), dtype=bool)
    per_check: dict[str, Any] = {}

    for constraint in schema.constraints_of("declared"):
        from otbench.generation.eval.metrics.validity import _constraint_failures

        failures = _constraint_failures(x_adv, constraint)
        if failures is None:
            continue
        valid &= ~failures
        per_check[f"constraint:{constraint.get('kind')}:{constraint.get('column', '')}"] = \
            int(failures.sum())

    for column in schema.categorical_cols:
        domain = schema.category_domain(column)
        if domain is None or column not in x_adv.columns:
            continue
        allowed = {str(v) for v in domain}
        outside = ~x_adv[column].astype("string").isin(allowed).to_numpy()
        valid &= ~outside
        per_check[f"category_domain:{column}"] = int(outside.sum())

    # Immutable columns are the classic tabular feasibility rule: an "attack"
    # that edits a person's age by ten years is not a perturbation.
    immutable = list(getattr(schema, "immutable_cols", []) or [])
    if immutable and x_clean is not None:
        clean = pd.DataFrame(x_clean).reset_index(drop=True)
        for column in immutable:
            if column in x_adv.columns and column in clean.columns:
                changed = (x_adv[column].astype(str) != clean[column].astype(str)).to_numpy()
                valid &= ~changed
                per_check[f"immutable:{column}"] = int(changed.sum())

    if not per_check:
        per_check["note"] = ("the schema declares no constraints, category domains or "
                             "immutable columns, so feasibility could not be checked")
        return None, per_check
    return valid, per_check


def _metric_direction(name: str) -> str | None:
    """Resolve a catalog direction, with explicit aliases for compact OOD payloads."""
    spec = CATALOG.get(name)
    if spec is not None:
        return spec.direction
    leaf = name.rsplit(".", 1)[-1]
    if leaf in {"accuracy", "balanced_accuracy", "macro_f1", "mcc", "roc_auc",
                "average_precision", "r2"}:
        return HIGHER
    if leaf in {"rmse", "mae", "nrmse", "median_absolute_error", "nll", "log_loss",
                "brier_score", "ece", "aurc", "pinball_loss"}:
        return LOWER
    return None

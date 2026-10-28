# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""G23-G24: metrics that only run when the dataset or the method declares more.

Attribute Disclosure Risk needs the schema to say which columns are sensitive
and which are quasi-identifiers -- that is a modelling decision about the
dataset, not something to infer from column names. Differential privacy needs
the *method* to supply an accountant; the benchmark verifies and reports the
claim, it never estimates an epsilon on the method's behalf.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from otbench.generation.eval.data import EvalFrames
from otbench.metrics.catalog import MetricResult

_ADR = "generation.disclosure.attribute_disclosure_risk"
_DP = "generation.privacy.dp_epsilon"

# A DP claim without all four of these is not checkable.
REQUIRED_DP_FIELDS = ("accountant", "epsilon", "delta", "noise_multiplier", "clipping_norm", "steps")


def evaluate_attribute_disclosure(
    frames: EvalFrames,
    seed: int = 0,
) -> list[MetricResult]:
    """Attack gain of inferring a sensitive column from quasi-identifiers.

    The attacker trains on the *generated* table (all they are given) and is
    scored on real holdout rows. The score is reported as a gain over a
    marginal baseline that ignores the quasi-identifiers entirely, because the
    absolute accuracy of predicting an imbalanced sensitive column says more
    about its marginal than about disclosure.
    """
    schema = frames.schema
    sensitive = [c for c in schema.sensitive_cols if c in frames.gen.columns]
    quasi = [c for c in schema.quasi_identifier_cols if c in frames.gen.columns]
    if not sensitive or not quasi:
        return [MetricResult.skipped(
            _ADR,
            "the schema declares no sensitive columns and/or no quasi-identifiers; "
            "attribute disclosure is undefined without both",
        )]

    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import balanced_accuracy_score, r2_score

    train_ord, test_ord, gen_ord = frames.ordinal_all()
    per_column: dict[str, Any] = {}
    gains: list[float] = []
    for column in sensitive:
        predictors = [c for c in quasi if c != column]
        if not predictors:
            per_column[column] = {"status": "skipped", "reason": "no quasi-identifier left"}
            continue
        y_attack = gen_ord[column]
        y_true = test_ord[column]
        is_categorical = schema.column_type(column) == "categorical"
        try:
            if is_categorical:
                if y_attack.nunique() < 2:
                    per_column[column] = {"status": "skipped",
                                          "reason": "generated sensitive column is constant"}
                    continue
                model = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
                model.fit(gen_ord[predictors], y_attack)
                attack = float(balanced_accuracy_score(y_true, model.predict(test_ord[predictors])))
                # Baseline: always predict the generated table's majority value.
                majority = y_attack.mode().iloc[0]
                baseline = float(balanced_accuracy_score(
                    y_true, np.full(len(y_true), majority)
                ))
            else:
                model = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1)
                model.fit(gen_ord[predictors], y_attack)
                attack = float(r2_score(y_true, model.predict(test_ord[predictors])))
                baseline = float(r2_score(y_true, np.full(len(y_true), y_attack.mean())))
        except Exception as exc:  # noqa: BLE001 — one column failing must not hide the rest
            per_column[column] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
            continue
        gain = attack - baseline
        per_column[column] = {"status": "ok", "attack": attack, "baseline": baseline,
                              "gain": gain,
                              "metric": "balanced_accuracy" if is_categorical else "r2"}
        gains.append(gain)

    if not gains:
        return [MetricResult.skipped(_ADR, "no sensitive column could be attacked",
                                     details={"per_column": per_column})]
    return [MetricResult.ok(
        _ADR, float(np.mean(gains)), n=len(gains), seed=seed,
        details={"per_column": per_column, "quasi_identifiers": quasi},
        extra={"interpretation": "gain over a marginal baseline; 0 means the generated "
                                 "table leaks nothing about the sensitive column beyond "
                                 "its marginal"},
    )]


def evaluate_dp_claim(method_metadata: dict | None) -> list[MetricResult]:
    """Verify and report a method's own DP accounting; never estimate it."""
    claim = (method_metadata or {}).get("differential_privacy")
    if not claim:
        return [MetricResult.skipped(
            _DP, "the method makes no differential-privacy claim"
        )]
    missing = [field for field in REQUIRED_DP_FIELDS if claim.get(field) is None]
    if missing:
        return [MetricResult.skipped(
            _DP,
            f"the method claims DP but did not report {missing}; an unverifiable claim is "
            f"not recorded as a privacy guarantee",
            details={"claim": claim},
        )]
    return [MetricResult.ok(
        _DP, float(claim["epsilon"]), details={"claim": claim},
        extra={"note": "reported by the method's own accountant and verified for "
                       "completeness only; epsilon values are comparable across methods "
                       "only at equal utility"},
    )]

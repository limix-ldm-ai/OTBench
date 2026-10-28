# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Catalog entries for the prediction metrics (P01-P25).

IDs refer to ``documents/implementation_matrix.md``. The probabilistic
regression packages (P16-P20) are gated on what the adapter actually emits:
a method that only produces samples gets CRPS but not NLL, because fitting a
kernel density to its samples would report a bandwidth choice as if it were the
model's likelihood.
"""
from __future__ import annotations

from otbench.metrics.catalog import (
    CATALOG,
    DIAGNOSTIC,
    HIGHER,
    LOWER,
    NEUTRAL,
    P0_CONDITIONAL,
    REQ_CLASSIFICATION,
    REQ_DENSITY,
    REQ_OOD,
    REQ_PROBA,
    REQ_QUANTILES,
    REQ_REGRESSION,
    REQ_SAMPLES,
    TARGET_ZERO,
)

TASK = "prediction"
_CLS = frozenset({REQ_CLASSIFICATION})
_REG = frozenset({REQ_REGRESSION})


def _add(**kwargs):
    kwargs.setdefault("task", TASK)
    return CATALOG.add(**kwargs)


# --- P01-P06: classification point predictions -----------------------------
_add(
    name="prediction.classification.accuracy", family="classification", direction=HIGHER,
    requires=_CLS, aggregation="rows", source="sklearn.metrics.accuracy_score",
    description="P01: always reported, never sufficient on its own",
)
_add(
    name="prediction.classification.balanced_accuracy", family="classification", direction=HIGHER,
    requires=_CLS, aggregation="macro_over_classes",
    source="sklearn.metrics.balanced_accuracy_score",
    description="P02: mean per-class recall",
)
_add(
    name="prediction.classification.macro_f1", family="classification", direction=HIGHER,
    requires=_CLS, aggregation="macro_over_classes",
    source="sklearn.metrics.f1_score(average='macro')",
    description="P03: macro for both binary and multiclass, so the number means the "
                "same thing across datasets",
)
_add(
    name="prediction.classification.mcc", family="classification", direction=HIGHER,
    requires=_CLS, aggregation="global", source="sklearn.metrics.matthews_corrcoef",
    description="P04: stays informative under severe imbalance",
)
_add(
    name="prediction.classification.roc_auc", family="classification", direction=HIGHER,
    requires=_CLS | {REQ_PROBA}, aggregation="global",
    source="sklearn.metrics.roc_auc_score; binary, or macro one-vs-rest for multiclass",
    description="P05: the task AUROC. Attack AUROCs live under privacy.* names so the "
                "two can never be confused.",
)
_add(
    name="prediction.classification.average_precision", family="classification", direction=HIGHER,
    requires=_CLS | {REQ_PROBA}, aggregation="global",
    source="sklearn.metrics.average_precision_score; macro one-vs-rest for multiclass",
    description="P06: PR-AUC, the informative summary when positives are rare",
)

# --- P07-P10: classification probability quality ---------------------------
_add(
    name="prediction.calibration.log_loss", family="calibration", direction=LOWER,
    requires=_CLS | {REQ_PROBA}, aggregation="rows", source="sklearn.metrics.log_loss",
    description="P07: probabilities are aligned to the train label domain before scoring",
)
_add(
    name="prediction.calibration.brier_score", family="calibration", direction=LOWER,
    requires=_CLS | {REQ_PROBA}, aggregation="rows",
    source="multiclass Brier: mean over rows of sum_k (p_k - [y=k])^2",
    description="P08: the multiclass form; on binary data it is twice the classic "
                "binary Brier, which is why the formula is stated here",
)
_add(
    name="prediction.calibration.ece", family="calibration", direction=LOWER,
    requires=_CLS | {REQ_PROBA}, aggregation="bins",
    source="top-label ECE with 15 equal-width bins (Naeini et al. 2015; Guo et al. 2017)",
    description="P09: a calibration diagnostic only -- it is not a proper scoring rule "
                "and cannot replace P07/P08. Reliability bins are stored in the details.",
)
_add(
    name="prediction.calibration.aurc", family="calibration", direction=LOWER,
    requires=_CLS | {REQ_PROBA}, aggregation="rows",
    source="area under the risk-coverage curve (Geifman, Uziel & El-Yaniv 2019)",
    description="P10: whether the model's confidence ranks its own errors last",
)

# --- P11-P15: regression point predictions ---------------------------------
_add(
    name="prediction.regression.rmse", family="regression", direction=LOWER,
    requires=_REG, aggregation="rows", source="root mean squared error on the original target scale",
    description="P11",
)
_add(
    name="prediction.regression.mae", family="regression", direction=LOWER,
    requires=_REG, aggregation="rows", source="mean absolute error on the original target scale",
    description="P12",
)
_add(
    name="prediction.regression.r2", family="regression", direction=HIGHER,
    requires=_REG, aggregation="rows", source="sklearn.metrics.r2_score",
    description="P13: can be negative; that is informative, not an error",
)
_add(
    name="prediction.regression.nrmse", family="regression", direction=LOWER,
    requires=_REG, aggregation="rows",
    source="RMSE divided by the TRAIN target SD (IQR when the SD degenerates)",
    description="P14: the normaliser never comes from the test target",
)
_add(
    name="prediction.regression.median_absolute_error", family="regression", direction=LOWER,
    requires=_REG, aggregation="rows", source="sklearn.metrics.median_absolute_error",
    description="P15: heavy-tail diagnostic",
)

# --- P16-P20: probabilistic regression -------------------------------------
_add(
    name="prediction.probabilistic.ensemble_crps", family="probabilistic", direction=LOWER,
    requires=_REG | {REQ_SAMPLES}, aggregation="rows",
    source="ensemble CRPS (Gneiting & Raftery 2007, eq. 21)",
    description="P16: requires native samples from p(y|x)",
)
_add(
    name="prediction.probabilistic.nll", family="probabilistic", direction=LOWER,
    requires=_REG | {REQ_DENSITY}, aggregation="rows",
    source="the method's own normalised log p(y|x), evaluated at the test target",
    description="P17: sample-only methods report skipped(no_normalized_density). No KDE "
                "is fitted to samples to manufacture a likelihood.",
)
_add(
    name="prediction.probabilistic.pinball_loss", family="probabilistic", direction=LOWER,
    requires=_REG | {REQ_QUANTILES}, aggregation="macro_over_quantiles",
    source="mean pinball loss over a fixed quantile grid",
    description="P18",
)
_add(
    name="prediction.probabilistic.coverage_error_90", family="probabilistic",
    direction=TARGET_ZERO, requires=_REG | {REQ_QUANTILES}, aggregation="rows",
    source="PICP of the 90% predictive interval minus 0.90",
    description="P19",
)
_add(
    name="prediction.probabilistic.coverage_error_95", family="probabilistic",
    direction=TARGET_ZERO, requires=_REG | {REQ_QUANTILES}, aggregation="rows",
    source="PICP of the 95% predictive interval minus 0.95",
    description="P19 at the 95% level",
)
_add(
    name="prediction.probabilistic.nmpiw_90", family="probabilistic", direction=LOWER,
    requires=_REG | {REQ_QUANTILES}, aggregation="rows",
    source="mean 90% interval width divided by the train target SD/IQR",
    description="P20: must be read together with P19",
)
_add(
    name="prediction.probabilistic.nmpiw_95", family="probabilistic", direction=LOWER,
    requires=_REG | {REQ_QUANTILES}, aggregation="rows",
    source="mean 95% interval width divided by the train target SD/IQR",
    description="P20 at the 95% level",
)
_add(
    name="prediction.probabilistic.quantile_crossing_rate", family="probabilistic",
    direction=LOWER, tier=DIAGNOSTIC, requires=_REG | {REQ_QUANTILES}, aggregation="rows",
    source="share of adjacent quantile pairs that are out of order",
    description="independently fitted quantile models are not monotone by construction; "
                "a non-zero rate invalidates P19/P20 and is reported rather than repaired",
)

# --- P21-P24: conditional metrics -----------------------------------------
_add(
    name="prediction.robustness.ood_degradation", family="robustness", direction=LOWER,
    tier=P0_CONDITIONAL, requires=frozenset({REQ_OOD}), aggregation="global",
    source="IID minus OOD score on a declared distribution-shift split",
    description="P21: requires a formal OOD split or a declared time column",
)
_add(
    name="prediction.robustness.ood_detection_auroc", family="robustness", direction=HIGHER,
    tier=P0_CONDITIONAL, requires=frozenset({REQ_OOD, REQ_PROBA}), aggregation="global",
    source="AUROC of the model's own uncertainty separating IID from OOD rows",
    description="P22: needs both an OOD split and an interpretable uncertainty output",
)
_add(
    name="prediction.robustness.adversarial_robust_accuracy", family="robustness",
    direction=HIGHER, tier=P0_CONDITIONAL, requires=frozenset({"attack_artifact"}),
    aggregation="rows",
    source="accuracy under a constraint-respecting attack artifact",
    description="P23: an attack that violates the declared constraints is reported as "
                "'invalid attack', which is not the same as 'attack failed'",
)
_add(
    name="prediction.robustness.attack_validity_rate", family="robustness", direction=HIGHER,
    tier=P0_CONDITIONAL, requires=frozenset({"attack_artifact"}), aggregation="rows",
    source="share of attack examples that satisfy the declared feasibility constraints",
    description="P24: reported alongside perturbation norms, runtime and query counts",
)

# --- P25: compute ----------------------------------------------------------
for _name, _direction, _desc in (
    ("fit_seconds", LOWER, "wall-clock training time (0 when a checkpoint was reused)"),
    ("inference_seconds_per_1k_rows", LOWER, "prediction throughput"),
    ("eval_seconds", LOWER, "metric computation time"),
    ("peak_ram_mb", LOWER, "peak host memory"),
    ("peak_vram_mb", LOWER, "peak device memory"),
    ("n_parameters", NEUTRAL, "trainable parameter count where available"),
    ("n_auxiliary_models", NEUTRAL, "models trained per logical run (quantile regressors "
                                    "train one per quantile)"),
):
    _add(
        name=f"prediction.cost.{_name}", family="cost", direction=_direction,
        aggregation="global", source="unified phase profiler", description=f"P25: {_desc}",
    )

_add(
    name="prediction.classification.unknown_test_label_rate", family="classification",
    direction=LOWER, tier=DIAGNOSTIC, requires=_CLS, aggregation="rows",
    source="share of test rows whose label is outside the train label domain",
    description="such rows are excluded from P01-P10 rather than treated as an extra class; "
                "this field keeps the exclusion visible",
)

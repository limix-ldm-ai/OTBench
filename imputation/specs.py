# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Catalog entries for the imputation metrics (I01-I22).

IDs refer to ``documents/implementation_matrix.md``. Point metrics (I01-I09)
apply to every imputer; the K-draw metrics (I10-I17) apply only to methods that
declare a native ``draws`` output, because re-running a deterministic imputer
with a different seed does not produce a posterior sample.
"""
from __future__ import annotations

from otbench.metrics.catalog import (
    CATALOG,
    DIAGNOSTIC,
    HIGHER,
    LOWER,
    NEUTRAL,
    P0_CONDITIONAL,
    REQ_CATEGORICAL,
    REQ_DRAWS,
    REQ_GROUPS,
    REQ_MATCHED_RUNS,
    REQ_NUMERICAL,
    REQ_SCIENTIFIC_MODEL,
    TARGET_ZERO,
)

TASK = "imputation"


def _add(**kwargs):
    kwargs.setdefault("task", TASK)
    return CATALOG.add(**kwargs)


# --- I01-I07: point accuracy and structure ---------------------------------
_add(
    name="imputation.point.macro_nrmse", family="point", direction=LOWER,
    requires=frozenset({REQ_NUMERICAL}), aggregation="macro_over_columns",
    source="per-column RMSE over masked cells divided by the clean TRAIN standard "
           "deviation (IQR when the SD degenerates), then averaged over columns",
    description="I01",
)
_add(
    name="imputation.point.macro_nmae", family="point", direction=LOWER,
    requires=frozenset({REQ_NUMERICAL}), aggregation="macro_over_columns",
    source="same normaliser as I01, applied to MAE",
    description="I02",
)
_add(
    name="imputation.point.categorical_macro_f1", family="point", direction=HIGHER,
    requires=frozenset({REQ_CATEGORICAL}), aggregation="macro_over_columns",
    source="per-column macro-F1 over masked cells, then averaged over columns",
    description="I03",
)
_add(
    name="imputation.point.categorical_balanced_accuracy", family="point", direction=HIGHER,
    requires=frozenset({REQ_CATEGORICAL}), aggregation="macro_over_columns",
    source="per-column balanced accuracy over masked cells",
    description="I04: rare-category behaviour that plain accuracy hides",
)
_add(
    name="imputation.structure.numeric_wasserstein1", family="structure", direction=LOWER,
    requires=frozenset({REQ_NUMERICAL}), aggregation="macro_over_columns",
    source="1-Wasserstein distance between the true and imputed value distributions "
           "on masked cells, in train-standardised units",
    description="I05: distributional agreement, which a point error cannot see",
)
_add(
    name="imputation.structure.categorical_jsd", family="structure", direction=LOWER,
    requires=frozenset({REQ_CATEGORICAL}), aggregation="macro_over_columns",
    source="base-2 Jensen-Shannon divergence over the union of categories",
    description="I06: symmetric divergence between categorical value distributions",
)
_add(
    name="imputation.structure.association_matrix_error", family="structure", direction=LOWER,
    aggregation="matrix_frobenius",
    source="relative Frobenius distance between the mixed association matrices of the "
           "clean and imputed frames (|Pearson r| / Cramer's V / correlation ratio)",
    description="I07: whether imputation preserved the dependence structure",
)

# --- I08: downstream utility ----------------------------------------------
_add(
    name="imputation.downstream.utility_gap", family="downstream", direction=TARGET_ZERO,
    aggregation="macro_over_models", seeded=True,
    source="score(model trained on imputed data) - score(same model on the clean oracle)",
    description="I08: the gap against a clean-data oracle, not an absolute score. "
                "Classification covers a linear, a tree and a nonlinear model with "
                "AUROC / AUPRC / macro-F1 / Brier; regression covers NRMSE / MAE / R2.",
)

# --- I09: compute ----------------------------------------------------------
for _name, _direction, _desc in (
    ("fit_seconds", LOWER, "wall-clock fit time (0 when a checkpoint was reused)"),
    ("seconds_per_1k_missing_cells", LOWER, "imputation throughput"),
    ("eval_seconds", LOWER, "metric computation time"),
    ("peak_ram_mb", LOWER, "peak host memory"),
    ("peak_vram_mb", LOWER, "peak device memory"),
    ("n_parameters", NEUTRAL, "trainable parameter count where available"),
    ("n_auxiliary_models", NEUTRAL, "models trained per logical run (per-column imputers "
                                    "train one per column)"),
):
    _add(
        name=f"imputation.cost.{_name}", family="cost", direction=_direction,
        aggregation="global", source="unified phase profiler", description=f"I09: {_desc}",
    )

# --- I10-I17: K-draw probabilistic metrics ---------------------------------
_DRAWS = frozenset({REQ_DRAWS})

_add(
    name="imputation.draws.numeric_crps", family="draws", direction=LOWER,
    requires=_DRAWS | {REQ_NUMERICAL}, aggregation="macro_over_columns",
    source="ensemble CRPS (Gneiting & Raftery 2007, eq. 21) per masked cell",
    description="I10",
)
_add(
    name="imputation.draws.categorical_log_loss", family="draws", direction=LOWER,
    requires=_DRAWS | {REQ_CATEGORICAL}, aggregation="macro_over_columns",
    source="log loss of the empirical category distribution over the K draws, with a "
           "fixed Laplace smoothing constant recorded in the result",
    description="I11",
)
_add(
    name="imputation.draws.categorical_brier", family="draws", direction=LOWER,
    requires=_DRAWS | {REQ_CATEGORICAL}, aggregation="macro_over_columns",
    source="multiclass Brier score of the same empirical distribution as I11",
    description="I12",
)
_add(
    name="imputation.draws.interval_coverage_error_90", family="draws", direction=TARGET_ZERO,
    requires=_DRAWS | {REQ_NUMERICAL}, aggregation="macro_over_columns",
    source="empirical 5th/95th percentile interval of the K draws minus the 0.90 nominal level",
    description="I13",
)
_add(
    name="imputation.draws.interval_coverage_error_95", family="draws", direction=TARGET_ZERO,
    requires=_DRAWS | {REQ_NUMERICAL}, aggregation="macro_over_columns",
    source="empirical 2.5th/97.5th percentile interval minus the 0.95 nominal level",
    description="I14",
)
_add(
    name="imputation.draws.normalized_interval_width_90", family="draws", direction=LOWER,
    requires=_DRAWS | {REQ_NUMERICAL}, aggregation="macro_over_columns",
    source="mean 90% interval width divided by the clean train SD/IQR",
    description="I15: meaningless without I13 next to it -- an infinitely wide interval "
                "has perfect coverage",
)
_add(
    name="imputation.draws.normalized_interval_width_95", family="draws", direction=LOWER,
    requires=_DRAWS | {REQ_NUMERICAL}, aggregation="macro_over_columns",
    source="mean 95% interval width divided by the clean train SD/IQR",
    description="I15 at the 95% level",
)
_add(
    name="imputation.draws.energy_score", family="draws", direction=LOWER,
    requires=_DRAWS | {REQ_NUMERICAL}, aggregation="rows",
    source="row-wise multivariate energy score (Gneiting & Raftery 2007, section 4.3) "
           "over the standardised masked numeric cells of each row",
    description="I16: the only metric here that scores the joint completion; it is why "
                "draws must not be averaged per cell",
)
_add(
    name="imputation.draws.mad_diversity", family="draws", direction=NEUTRAL,
    tier=DIAGNOSTIC, requires=_DRAWS, aggregation="macro_over_columns",
    source="mean absolute deviation of the draws around their per-cell median/mode",
    description="I17: spread only. High diversity is neither good nor bad on its own; "
                "read it together with I10 and I13.",
)

# --- I18-I22: conditional metrics -----------------------------------------
_add(
    name="imputation.fairness.group_gap", family="fairness", direction=TARGET_ZERO,
    tier=P0_CONDITIONAL, requires=frozenset({REQ_GROUPS}), aggregation="macro_over_groups",
    source="worst-group minus overall value of the I01-I06 metrics",
    description="I18: requires schema-declared groups",
)
_add(
    name="imputation.fairness.downstream_gap", family="fairness", direction=TARGET_ZERO,
    tier=P0_CONDITIONAL, requires=frozenset({REQ_GROUPS}), aggregation="macro_over_groups",
    source="between-group spread of the downstream classification scores",
    description="I19: requires groups plus stored per-row downstream predictions",
)
_add(
    name="imputation.stability.label_stability", family="stability", direction=HIGHER,
    tier=P0_CONDITIONAL, requires=frozenset({REQ_DRAWS}), aggregation="rows", seeded=True,
    source="share of rows whose downstream predicted label is identical across draws",
    description="I20: needs a pre-registered bootstrap/draw configuration and stored "
                "per-row predictions",
)
_add(
    name="imputation.robustness.missingness_shift_degradation", family="robustness",
    direction=LOWER, tier=P0_CONDITIONAL, requires=frozenset({REQ_MATCHED_RUNS}),
    aggregation="global",
    source="paired difference of I01/I03 between matched runs at different mechanisms or rates",
    description="I21: requires the matched runs to exist; it cannot be read off a single run",
)
_add(
    name="imputation.inference.rubin_validity", family="inference", direction=TARGET_ZERO,
    tier=P0_CONDITIONAL, requires=frozenset({REQ_DRAWS, REQ_SCIENTIFIC_MODEL}),
    aggregation="global",
    source="Rubin's rules (Rubin 1987) applied to a pre-registered estimand over the K "
           "completed datasets",
    description="I22: reports estimate bias, standard-error bias and interval coverage. "
                "Without a declared scientific model there is no estimand to pool.",
)

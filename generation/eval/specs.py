# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Catalog entries for the generation metrics (G01-G27).

Pure metadata: importing this module registers the specs but pulls in nothing
heavier than the catalog itself, so documentation and the redundancy study can
read the full metric list without touching a model backend.

IDs in the comments refer to ``documents/implementation_matrix.md``.
"""
from __future__ import annotations

from otbench.metrics.catalog import (
    CATALOG,
    DIAGNOSTIC,
    HIGHER,
    LOWER,
    NEUTRAL,
    P0,
    P0_CONDITIONAL,
    REQ_DP_ACCOUNTANT,
    REQ_QUASI_IDENTIFIERS,
    REQ_RAW_GEN,
    REQ_SCM_GRAPH,
    REQ_SENSITIVE,
    REQ_SHADOW,
    TARGET_HALF,
    TARGET_ZERO,
)

TASK = "generation"
SDV = "SDMetrics KSComplement/TVComplement definition"
ALAA = "Alaa, van Breugel, Saveliev & van der Schaar (2022), ICML: 'How Faithful is your Synthetic Data?'"


def _add(**kwargs):
    kwargs.setdefault("task", TASK)
    return CATALOG.add(**kwargs)


# --- G01: validity (raw artifact only) -------------------------------------
_VALIDITY_NOTE = (
    "computed on the raw generator output; the aligned artifact has these "
    "violations repaired out of existence by construction"
)

_add(
    name="generation.validity.schema_type_violation_rate", family="validity",
    direction=LOWER, requires=frozenset({REQ_RAW_GEN}), aggregation="cells",
    source="schema.json declared column types",
    description=f"share of cells whose value cannot be read as its declared type; {_VALIDITY_NOTE}",
)
_add(
    name="generation.validity.unknown_category_rate", family="validity",
    direction=LOWER, requires=frozenset({REQ_RAW_GEN}), aggregation="cells",
    source="schema.json category domains (declared, else train-derived)",
    description="share of categorical cells outside the declared value domain",
)
_add(
    name="generation.validity.empirical_range_violation_rate", family="validity",
    direction=LOWER, requires=frozenset({REQ_RAW_GEN}), aggregation="cells",
    source="train-derived (empirical) numeric ranges",
    description="share of numeric cells outside the observed train range; descriptive, "
                "not a business-rule violation",
)
_add(
    name="generation.validity.declared_constraint_violation_rate", family="validity",
    direction=LOWER, tier=P0_CONDITIONAL, requires=frozenset({REQ_RAW_GEN}),
    aggregation="constraint_checks",
    source="schema.json `constraints` with source='declared'",
    description="share of failed declared constraint checks (ranges, inequalities, "
                "uniqueness, sums); skipped when no constraint is declared",
)
_add(
    name="generation.validity.missing_cell_rate", family="validity",
    direction=LOWER, requires=frozenset({REQ_RAW_GEN}), aggregation="cells",
    source="raw artifact null count",
    description="share of null cells the generator emitted",
)
_add(
    name="generation.validity.missing_column_count", family="validity",
    direction=LOWER, requires=frozenset({REQ_RAW_GEN}), aggregation="columns",
    source="schema.json column set",
    description="number of schema columns absent from the raw output",
)
_add(
    name="generation.validity.row_validity_rate", family="validity",
    direction=HIGHER, requires=frozenset({REQ_RAW_GEN}), aggregation="rows",
    source="conjunction of the type / domain / declared-constraint checks",
    description="share of rows that pass every applicable validity check",
)

# --- G02-G05: marginal and dependence fidelity -----------------------------
_add(
    name="generation.fidelity.numerical_ks_error", family="fidelity",
    direction=LOWER, aggregation="macro_over_columns",
    source="two-sample Kolmogorov-Smirnov statistic (scipy.stats.ks_2samp)",
    description="G02: mean per-column KS distance between real and generated marginals",
)
_add(
    name="generation.fidelity.categorical_tvd_error", family="fidelity",
    direction=LOWER, aggregation="macro_over_columns",
    source="total variation distance over the union of observed categories",
    description="G03: mean per-column TVD; generated-only categories keep their mass",
)
_add(
    name="generation.fidelity.shape_error", family="fidelity",
    direction=LOWER, aggregation="macro_over_columns",
    source=SDV,
    description="G02+G03 combined: mean of the numerical KS and categorical TVD errors",
)
_add(
    name="generation.fidelity.pairwise_trend_error", family="fidelity",
    direction=LOWER, aggregation="macro_over_column_pairs",
    source="|corr_real - corr_syn| / 2 for numerical pairs (SDMetrics CorrelationSimilarity) "
           "and contingency TVD for pairs involving a categorical column",
    description="G04: mean absolute deviation of pairwise dependence",
)
_add(
    name="generation.fidelity.mixed_association_error", family="fidelity",
    direction=LOWER, aggregation="mean_abs_over_pairs",
    source="mixed association matrix: |Pearson r| (num-num), Cramer's V (cat-cat), "
           "correlation ratio eta (num-cat)",
    description="G05: mean absolute difference over comparable off-diagonal "
                "association pairs; the full matrices and Frobenius norm are in the details",
)

# --- G06-G09: support, diversity, memorisation -----------------------------
_add(
    name="generation.support.alpha_precision", family="support",
    direction=HIGHER, aggregation="global", source=ALAA,
    description="G06: share of generated samples inside the alpha-support of the real data",
)
_add(
    name="generation.support.beta_recall_coverage", family="support",
    direction=HIGHER, aggregation="global", source=ALAA,
    description="G07: share of real samples covered by the beta-support of the generated "
                "data. 'Coverage' here is beta-recall, not Naeem coverage and not a "
                "prediction interval.",
)
_add(
    name="generation.support.authenticity", family="support",
    direction=HIGHER, aggregation="global", source=ALAA,
    description="G08: share of generated samples closer to another generated sample than "
                "to their nearest real neighbour; a novelty proxy, not a privacy guarantee",
)
_add(
    name="generation.memorization.exact_copy_rate", family="memorization",
    direction=LOWER, requires=frozenset({REQ_RAW_GEN}), aggregation="rows",
    source="canonical row hash equality against the training split",
    description="G09: share of generated rows byte-identical (after canonical formatting) "
                "to a training row. Tolerance matching is a different metric and is not "
                "reported under this name.",
)

# --- G10-G11: distinguishability -------------------------------------------
_add(
    name="generation.distinguishability.logistic_c2st_auroc", family="distinguishability",
    direction=TARGET_HALF, aggregation="global", seeded=True,
    source="Classifier Two-Sample Test (Lopez-Paz & Oquab 2017) with a logistic classifier",
    description="G10: AUROC of a linear real-vs-generated discriminator on a held-out split; "
                "0.5 means indistinguishable",
)
_add(
    name="generation.distinguishability.gbdt_c2st_auroc", family="distinguishability",
    direction=TARGET_HALF, aggregation="global", seeded=True,
    source="Classifier Two-Sample Test (Lopez-Paz & Oquab 2017) with a gradient-boosted tree",
    description="G11: same split as G10, nonlinear discriminator",
)

# --- G12-G13: utility ------------------------------------------------------
_add(
    name="generation.utility.global_utility_ratio", family="utility",
    direction=HIGHER, aggregation="macro_over_columns", seeded=True,
    source="each column in turn as the target; balanced accuracy ratio (classification) "
           "or RMSE ratio (regression) of a synthetic-trained vs real-trained model",
    description="G12: unclipped ratio. Values above 1 (synthetic beating real) are kept "
                "visible rather than truncated.",
)
_add(
    name="generation.utility.tstr_gap", family="utility",
    direction=TARGET_ZERO, aggregation="global", seeded=True,
    source="Train-on-Synthetic-Test-on-Real (Esteban, Hyland & Rätsch 2017), "
           "differenced against the Train-on-Real denominator",
    description="G13: TSTR minus TRTR on the schema target, evaluated on the same real "
                "test split. Sign is oriented so that positive means synthetic is better.",
)

# --- G14-G16: query error --------------------------------------------------
for _way, _id in ((1, "G14"), (2, "G15"), (3, "G16")):
    _add(
        name=f"generation.query.{['one', 'two', 'three'][_way - 1]}_way_error",
        family="query", direction=LOWER, aggregation="mean_over_queries", seeded=True,
        source="marginal/range query workload error as used by the DP synthetic-data "
               "literature (NIST 2018 challenge; McKenna et al. 2019) and reported per "
               "k-way order by SynMeter",
        description=f"{_id}: mean absolute difference between real and generated answers to "
                    f"seeded {_way}-way point and range count queries, as a fraction of n",
    )

# --- G17-G22: privacy / memorisation diagnostics ---------------------------
_add(
    name="generation.privacy.distance_mia_auroc", family="privacy",
    direction=TARGET_HALF, aggregation="global", seeded=True,
    source="post-hoc distance-to-closest-record membership attack over a mixed "
           "(Gower) distance, members = train rows, non-members = holdout rows",
    description="G17: cheap post-hoc attack; not a substitute for shadow-model auditing",
)
_add(
    name="generation.privacy.distance_mia_tpr_at_1pct_fpr", family="privacy",
    direction=LOWER, aggregation="global", seeded=True,
    source="TPR at 1% FPR of the same attack (Carlini et al. 2022 reporting convention)",
    description="G17: low-FPR operating point, which average AUROC hides",
)
_add(
    name="generation.privacy.shadow_mia_auroc", family="privacy",
    direction=TARGET_HALF, tier=P0_CONDITIONAL, requires=frozenset({REQ_SHADOW}),
    aggregation="global", seeded=True,
    source="shadow-model membership inference (Shokri et al. 2017) against the generator",
    description="G18: requires retraining the generator on member-in / member-out splits",
)
_add(
    name="generation.privacy.shadow_mia_tpr_at_1pct_fpr", family="privacy",
    direction=LOWER, tier=P0_CONDITIONAL, requires=frozenset({REQ_SHADOW}),
    aggregation="global", seeded=True,
    source="TPR at 1% FPR of the shadow attack",
    description="G18: low-FPR operating point of the shadow audit",
)
_add(
    name="generation.privacy.membership_disclosure_score", family="privacy",
    direction=LOWER, tier=P0_CONDITIONAL, requires=frozenset({REQ_SHADOW}),
    aggregation="global", seeded=True,
    source="membership disclosure over include/exclude retraining pairs",
    description="G19: cannot be approximated from a single train/generated CSV pair",
)
_add(
    name="generation.privacy.mean_min_euclidean_dcr", family="privacy",
    direction=HIGHER, tier=DIAGNOSTIC, aggregation="rows",
    source="mean over generated rows of the Euclidean distance to the closest training row, "
           "in the train-standardised numeric + one-hot space",
    description="G20: one specific DCR definition. The bare name 'DCR' is not reported.",
)
_add(
    name="generation.privacy.normalized_median_gower_dcr", family="privacy",
    direction=NEUTRAL, aggregation="rows",
    source="median Gower DCR of generated->train divided by median Gower DCR of "
           "holdout->train, using one shared Gower definition for both",
    description="G21: about 1 means generated rows sit no closer to train than real "
                "holdout rows do; below 1 indicates memorisation",
)
_add(
    name="generation.privacy.train_holdout_nn_preference", family="privacy",
    direction=NEUTRAL, aggregation="rows",
    source="share of generated rows whose nearest real neighbour is a train row rather "
           "than a holdout row",
    description="G22: 0.5 is the ideal; higher means the generator prefers its training rows",
)

# --- G23-G26: conditional metrics -----------------------------------------
_add(
    name="generation.disclosure.attribute_disclosure_risk", family="disclosure",
    direction=LOWER, tier=P0_CONDITIONAL,
    requires=frozenset({REQ_SENSITIVE, REQ_QUASI_IDENTIFIERS}), aggregation="global", seeded=True,
    source="attribute inference from quasi-identifiers, scored against a marginal baseline",
    description="G23: runs only when the schema declares sensitive columns and "
                "quasi-identifiers; reported as attack gain over the baseline",
)
_add(
    name="generation.privacy.dp_epsilon", family="privacy",
    direction=LOWER, tier=P0_CONDITIONAL, requires=frozenset({REQ_DP_ACCOUNTANT}),
    aggregation="global",
    source="the method's own DP accountant output",
    description="G24: verified and reported, never estimated by the benchmark; a method "
                "that claims DP without an accountant, clipping bound, noise multiplier "
                "and step count is reported as an unverified claim",
)
_add(
    name="generation.structural.local_conditional_independence", family="structural",
    direction=HIGHER, tier=P0_CONDITIONAL, requires=frozenset({REQ_SCM_GRAPH}),
    aggregation="macro_over_tests",
    source="conditional-independence tests against the ground-truth SCM graph",
    description="G25: 'CI' here means conditional independence, never confidence interval",
)
_add(
    name="generation.structural.global_conditional_independence", family="structural",
    direction=HIGHER, tier=P0_CONDITIONAL, requires=frozenset({REQ_SCM_GRAPH}),
    aggregation="macro_over_tests",
    source="conditional-independence tests against the ground-truth SCM graph",
    description="G26: global pass rate over the graph's implied independences",
)

# --- G27: compute ----------------------------------------------------------
for _name, _direction, _desc in (
    ("fit_seconds", LOWER, "wall-clock training time (0 when a checkpoint was reused)"),
    ("sampling_seconds_per_1k_rows", LOWER, "sampling throughput"),
    ("eval_seconds", LOWER, "metric computation time"),
    ("peak_ram_mb", LOWER, "peak host memory during the phase"),
    ("peak_vram_mb", LOWER, "peak device memory during the phase"),
    ("n_parameters", NEUTRAL, "trainable parameter count where the backend exposes one"),
    ("n_auxiliary_models", NEUTRAL, "number of models one logical run trains"),
):
    _add(
        name=f"generation.cost.{_name}", family="cost", direction=_direction,
        aggregation="global", source="unified phase profiler",
        description=f"G27: {_desc}",
    )

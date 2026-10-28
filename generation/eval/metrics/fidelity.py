# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""G02-G05: marginal and dependence fidelity, implemented in NumPy/SciPy.

These reproduce the SDMetrics definitions rather than calling SDMetrics, so the
formula is visible in one place and the benchmark does not inherit a definition
change from an upstream release:

- ``KSComplement``          = 1 - KS statistic            -> G02 error = KS
- ``TVComplement``          = 1 - total variation distance -> G03 error = TVD
- ``CorrelationSimilarity`` = 1 - |r_real - r_syn| / 2     -> G04 numeric pairs
- ``ContingencySimilarity`` = 1 - 0.5 * sum |p_real - p_syn| -> G04 other pairs

Continuous columns entering a contingency comparison are discretised with
**train-fitted** quantile edges, so the binning does not shift with the frames
being compared.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from otbench.generation.eval.data import EvalFrames
from otbench.metrics.catalog import MetricResult
from otbench.metrics.stats import (
    association_matrix,
    ks_statistic,
    matrix_difference,
    total_variation_distance,
)

N_CONTINGENCY_BINS = 10


def evaluate_fidelity(frames: EvalFrames, reference_split: str = "test") -> list[MetricResult]:
    schema = frames.schema
    real = frames.original(reference_split)
    gen = frames.original("gen")
    numerical = [c for c in schema.numerical_cols if c in real.columns]
    categorical = [c for c in schema.categorical_cols if c in real.columns]

    results: list[MetricResult] = []
    shape_errors: list[float] = []

    # --- G02 numeric marginals --------------------------------------------
    ks_per_column = {
        col: ks_statistic(real[col].to_numpy(), gen[col].to_numpy()) for col in numerical
    }
    ks_values = [v for v in ks_per_column.values() if v is not None]
    if ks_values:
        shape_errors.extend(ks_values)
        results.append(MetricResult.ok(
            "generation.fidelity.numerical_ks_error", float(np.mean(ks_values)),
            n=len(ks_values), details={"per_column": ks_per_column},
        ))
    else:
        results.append(MetricResult.skipped(
            "generation.fidelity.numerical_ks_error", "no numerical columns to compare"
        ))

    # --- G03 categorical marginals ----------------------------------------
    tvd_per_column = {
        col: total_variation_distance(
            real[col], gen[col], domain=schema.category_domain(col)
        )
        for col in categorical
    }
    tvd_values = [v for v in tvd_per_column.values() if v is not None]
    if tvd_values:
        shape_errors.extend(tvd_values)
        results.append(MetricResult.ok(
            "generation.fidelity.categorical_tvd_error", float(np.mean(tvd_values)),
            n=len(tvd_values), details={"per_column": tvd_per_column},
        ))
    else:
        results.append(MetricResult.skipped(
            "generation.fidelity.categorical_tvd_error", "no categorical columns to compare"
        ))

    if shape_errors:
        shape_error = float(np.mean(shape_errors))
        results.append(MetricResult.ok(
            "generation.fidelity.shape_error", shape_error, n=len(shape_errors)
        ))
    else:
        results.append(MetricResult.skipped("generation.fidelity.shape_error", "no comparable columns"))

    # --- G04 pairwise trends ----------------------------------------------
    results.extend(_pairwise_trend(frames, real, gen, numerical, categorical))

    # --- G05 mixed association matrix -------------------------------------
    results.extend(_association(real, gen, numerical, categorical))
    return results


def _pairwise_trend(
    frames: EvalFrames,
    real: pd.DataFrame,
    gen: pd.DataFrame,
    numerical: list[str],
    categorical: list[str],
) -> list[MetricResult]:
    columns = numerical + categorical
    if len(columns) < 2:
        return [
            MetricResult.skipped(name, "fewer than two comparable columns")
            for name in ("generation.fidelity.pairwise_trend_error",)
        ]

    edges = {col: _train_bin_edges(frames.train[col]) for col in numerical}
    num_set = set(numerical)
    per_pair: dict[str, Any] = {}
    errors: list[float] = []
    for i, left in enumerate(columns):
        for right in columns[i + 1:]:
            if left in num_set and right in num_set:
                error = _correlation_error(real, gen, left, right)
                kind = "correlation"
            else:
                error = _contingency_error(real, gen, left, right, edges)
                kind = "contingency"
            if error is None:
                continue
            per_pair[f"{left}|{right}"] = {"error": error, "kind": kind}
            errors.append(error)

    if not errors:
        return [
            MetricResult.skipped(name, "no comparable column pair")
            for name in ("generation.fidelity.pairwise_trend_error",)
        ]
    value = float(np.mean(errors))
    return [
        MetricResult.ok("generation.fidelity.pairwise_trend_error", value,
                        n=len(errors), details={"per_pair": per_pair}),
    ]


def _association(
    real: pd.DataFrame,
    gen: pd.DataFrame,
    numerical: list[str],
    categorical: list[str],
) -> list[MetricResult]:
    names = ("generation.fidelity.mixed_association_error",)
    if len(numerical) + len(categorical) < 2:
        return [MetricResult.skipped(name, "fewer than two comparable columns") for name in names]

    real_matrix = association_matrix(real, numerical, categorical)
    gen_matrix = association_matrix(gen, numerical, categorical)
    frobenius, mean_abs, n_pairs = matrix_difference(real_matrix, gen_matrix)
    if mean_abs is None:
        return [MetricResult.skipped(name, "no comparable association pair") for name in names]

    details = {
        "frobenius_norm": frobenius,
        "n_pairs": n_pairs,
        "real_matrix": real_matrix.round(6).to_dict(),
        "generated_matrix": gen_matrix.round(6).to_dict(),
    }
    return [
        MetricResult.ok(names[0], mean_abs, n=n_pairs, details=details,
                        extra={"note": "scalar is the mean absolute difference over pairs; "
                                       "the Frobenius norm is in the details"}),
    ]


def _correlation_error(real, gen, left: str, right: str) -> float | None:
    r_real = _pearson(real[left], real[right])
    r_gen = _pearson(gen[left], gen[right])
    if r_real is None or r_gen is None:
        return None
    return float(abs(r_real - r_gen) / 2.0)


def _contingency_error(real, gen, left: str, right: str, edges: dict) -> float | None:
    real_pair = pd.DataFrame({
        "l": _discretize(real[left], edges.get(left)),
        "r": _discretize(real[right], edges.get(right)),
    }).dropna()
    gen_pair = pd.DataFrame({
        "l": _discretize(gen[left], edges.get(left)),
        "r": _discretize(gen[right], edges.get(right)),
    }).dropna()
    if len(real_pair) == 0 or len(gen_pair) == 0:
        return None
    real_joint = real_pair.value_counts(normalize=True)
    gen_joint = gen_pair.value_counts(normalize=True)
    keys = set(real_joint.index) | set(gen_joint.index)
    return float(
        0.5 * sum(abs(real_joint.get(k, 0.0) - gen_joint.get(k, 0.0)) for k in keys)
    )


def _pearson(a: pd.Series, b: pd.Series) -> float | None:
    frame = pd.DataFrame({
        "a": pd.to_numeric(a, errors="coerce"),
        "b": pd.to_numeric(b, errors="coerce"),
    }).dropna()
    if len(frame) < 2 or frame["a"].std() == 0 or frame["b"].std() == 0:
        return None
    return float(np.corrcoef(frame["a"], frame["b"])[0, 1])


def _train_bin_edges(train_column: pd.Series) -> np.ndarray | None:
    """Quantile edges fitted on train, so binning never depends on the compared frames."""
    values = pd.to_numeric(train_column, errors="coerce").dropna()
    if len(values) < 2:
        return None
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, N_CONTINGENCY_BINS + 1)))
    return edges if len(edges) >= 2 else None


def _discretize(column: pd.Series, edges: np.ndarray | None) -> pd.Series:
    if edges is None:
        return column.astype("string")
    values = pd.to_numeric(column, errors="coerce")
    # Out-of-range values land in the first/last bin rather than becoming NaN,
    # so an out-of-range generator is penalised by the marginal metric (G02)
    # instead of quietly dropping out of the pairwise one.
    binned = np.digitize(values.to_numpy(dtype=float), edges[1:-1], right=False).astype(float)
    binned[~np.isfinite(values.to_numpy(dtype=float))] = np.nan
    return pd.Series(binned, index=column.index).astype("string")

# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""G14-G16: k-way marginal / range query error.

The workload measure used throughout the DP synthetic-data literature (the NIST
2018 differential privacy synthetic data challenge; McKenna, Sheldon & Miklau,
ICML 2019) and reported per k-way order by SynMeter: build a fixed set of
conjunctive queries, answer each one on the real and the generated table, and
average the absolute difference in the answers.

The answer of a query is a *fraction of rows*, so the error is comparable
across tables of different sizes. The workload is generated from an explicit
seed and the queries themselves are stored in the details, so two runs of the
benchmark score the same generator against the same questions.

Predicates by column type:
    categorical -- point:  ``col == v`` for a value drawn from the train domain
    numerical   -- range:  ``lo <= col <= hi`` for two train quantiles
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from otbench.generation.eval.data import EvalFrames
from otbench.metrics.catalog import MetricResult

N_QUERIES_PER_WAY = 300
MAX_STORED_QUERIES = 20
_NAMES = {1: "generation.query.one_way_error",
          2: "generation.query.two_way_error",
          3: "generation.query.three_way_error"}


def evaluate_query_error(
    frames: EvalFrames,
    reference_split: str = "test",
    seed: int = 0,
    n_queries: int = N_QUERIES_PER_WAY,
    ways: tuple[int, ...] = (1, 2, 3),
) -> list[MetricResult]:
    schema = frames.schema
    real = frames.original(reference_split)
    gen = frames.original("gen")
    columns = [c for c in schema.all_columns if c in real.columns]

    results: list[MetricResult] = []
    for way in ways:
        name = _NAMES[way]
        if len(columns) < way:
            results.append(MetricResult.skipped(
                name, f"the dataset has {len(columns)} usable columns, fewer than {way}"
            ))
            continue
        # A distinct seed per order keeps the 1-way workload from being a prefix
        # of the 2-way one, which would correlate the three metrics artificially.
        rng = np.random.default_rng([seed, way])
        queries = [
            _sample_query(rng, columns, way, frames.train, schema) for _ in range(n_queries)
        ]
        queries = [q for q in queries if q is not None]
        if not queries:
            results.append(MetricResult.skipped(name, "no valid predicate could be built"))
            continue

        errors, point_errors, range_errors = [], [], []
        for query in queries:
            real_answer = _answer(real, query)
            gen_answer = _answer(gen, query)
            error = abs(real_answer - gen_answer)
            errors.append(error)
            if all(p["kind"] == "point" for p in query):
                point_errors.append(error)
            elif all(p["kind"] == "range" for p in query):
                range_errors.append(error)

        results.append(MetricResult.ok(
            name, float(np.mean(errors)), n=len(errors), seed=seed,
            details={
                "n_queries": len(errors),
                "reference_split": reference_split,
                "mean_error_point_only": _mean_or_none(point_errors),
                "mean_error_range_only": _mean_or_none(range_errors),
                "max_error": float(np.max(errors)),
                "example_queries": queries[:MAX_STORED_QUERIES],
            },
        ))
    return results


def _sample_query(rng, columns: list[str], way: int, train: pd.DataFrame, schema) -> list[dict] | None:
    chosen = rng.choice(len(columns), size=way, replace=False)
    predicates = []
    for index in chosen:
        column = columns[int(index)]
        predicate = _sample_predicate(rng, column, train, schema)
        if predicate is None:
            return None
        predicates.append(predicate)
    return predicates


def _sample_predicate(rng, column: str, train: pd.DataFrame, schema) -> dict | None:
    if column in set(schema.numerical_cols):
        values = pd.to_numeric(train[column], errors="coerce").dropna()
        if len(values) < 2:
            return None
        levels = np.sort(rng.uniform(0.0, 1.0, size=2))
        lo, hi = np.quantile(values, levels)
        return {"kind": "range", "column": column, "lo": float(lo), "hi": float(hi)}

    domain = schema.category_domain(column)
    if not domain:
        domain = train[column].dropna().unique().tolist()
    if not domain:
        return None
    value = domain[int(rng.integers(0, len(domain)))]
    return {"kind": "point", "column": column, "value": str(value)}


def _answer(frame: pd.DataFrame, query: list[dict]) -> float:
    """Fraction of rows satisfying the conjunction of the predicates."""
    if len(frame) == 0:
        return 0.0
    mask = np.ones(len(frame), dtype=bool)
    for predicate in query:
        column = predicate["column"]
        if column not in frame.columns:
            return 0.0
        if predicate["kind"] == "range":
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            with np.errstate(invalid="ignore"):
                mask &= (values >= predicate["lo"]) & (values <= predicate["hi"])
        else:
            mask &= (frame[column].astype("string") == predicate["value"]).to_numpy()
    return float(np.mean(mask))


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None

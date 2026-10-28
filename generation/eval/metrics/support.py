# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""G06-G09: support coverage, novelty, and exact memorisation.

G06-G08 come from one pass of the Alaa et al. (2022) support estimator, so
alpha-precision, beta-recall and authenticity are guaranteed to describe the
same embedding. Authenticity was already produced by the backend but never
surfaced; it is a nearest-neighbour novelty score and is **not** a privacy
guarantee, which is why it lives here and not under ``privacy``.

G09 is deliberately separate and deliberately exact. A tolerance-based "near
duplicate" count is a different quantity with a different threshold parameter;
reporting it as an exact-copy rate would understate or overstate memorisation
depending on an arbitrary radius.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from otbench.generation.eval.data import EvalFrames
from otbench.generation.eval.metrics._backends import alpha_precision_beta_recall
from otbench.metrics.catalog import MetricResult


def evaluate_support(frames: EvalFrames, reference_split: str = "test") -> list[MetricResult]:
    names = ("generation.support.alpha_precision",
             "generation.support.beta_recall_coverage",
             "generation.support.authenticity")
    real_onehot, syn_onehot = frames.high_order_pair(reference_split)
    if real_onehot.shape[1] == 0 or len(real_onehot) == 0 or len(syn_onehot) == 0:
        return [MetricResult.skipped(n, "empty encoded frame") for n in names]

    scores = alpha_precision_beta_recall(real_onehot, syn_onehot)
    details = {"reference_split": reference_split, "n_real": len(real_onehot),
               "n_generated": len(syn_onehot), "n_features": int(real_onehot.shape[1])}
    results = []
    for name, key in zip(names, ("alpha_precision", "beta_recall", "authenticity"), strict=True):
        value = _finite(scores[key])
        results.append(
            MetricResult.ok(name, value, details=details)
            if value is not None
            else MetricResult.skipped(name, "the support estimator returned a non-finite value")
        )
    return results


def evaluate_memorization(frames: EvalFrames) -> list[MetricResult]:
    name = "generation.memorization.exact_copy_rate"
    if not frames.has_raw:
        return [MetricResult.skipped(
            name,
            "no raw generator artifact; the aligned frame has been coerced, so an exact "
            "match against it would measure the alignment step",
        )]
    schema = frames.schema
    raw = frames.gen_raw
    missing = [c for c in schema.all_columns if c not in raw.columns]
    if missing:
        return [MetricResult.skipped(
            name,
            f"raw output is missing schema columns {missing}; partial-row matching would "
            "overstate exact memorisation",
        )]
    shared = list(schema.all_columns)
    if len(raw) == 0:
        return [MetricResult.skipped(name, "raw output contains no row to compare")]

    train_hashes = _row_hashes(frames.train, shared, schema)
    gen_hashes = _row_hashes(raw, shared, schema)
    known = set(train_hashes)
    copied = np.array([h in known for h in gen_hashes], dtype=bool)

    n_unique_copied = len({h for h, is_copy in zip(gen_hashes, copied, strict=True) if is_copy})
    return [MetricResult.ok(
        name, float(copied.mean()), n=len(copied),
        details={
            "n_copied_rows": int(copied.sum()),
            "n_distinct_copied_train_rows": n_unique_copied,
            "columns_compared": shared,
            "matching": "exact, after canonical formatting (float round-trip / string cast)",
        },
    )]


def _row_hashes(frame: pd.DataFrame, columns: list[str], schema) -> list[str]:
    """Canonical per-row hashes.

    Canonicalisation only removes representation differences that carry no
    information -- ``1`` vs ``1.0``, trailing whitespace -- via an exact float
    round-trip. It never merges two distinct values.
    """
    numerical = set(schema.numerical_cols)
    canonical = pd.DataFrame(index=frame.index)
    for col in columns:
        if col in numerical:
            values = pd.to_numeric(frame[col], errors="coerce")
            canonical[col] = values.map(lambda v: "NA" if pd.isna(v) else repr(float(v)))
        else:
            canonical[col] = frame[col].astype("string").fillna("NA").str.strip()
    joined = canonical.agg("\x1f".join, axis=1)
    return [hashlib.sha1(value.encode("utf-8")).hexdigest() for value in joined]


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None

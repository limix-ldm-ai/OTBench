# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""G17, G20-G22: post-hoc memorisation and membership diagnostics.

"DCR" is not a metric. It is a family of at least three incompatible
quantities -- a raw mean Euclidean distance, a Gower ratio against a real
baseline, and a nearest-neighbour preference rate -- that different papers all
abbreviate the same way and that cannot be compared to each other. Each one is
computed and reported under its own explicit name.

All three, plus the distance-based membership attack, share:

* one train-fitted :class:`GowerSpace`, so a ratio's numerator and denominator
  really are the same metric;
* the *train* split as members and the *test* split as the non-member holdout;
* equal-size subsamples of the two real sides, because an unbalanced comparison
  biases both the ratio and the attack AUROC.

None of these is a privacy guarantee. They are cheap post-hoc diagnostics; the
formal audit is the shadow-model track in
:mod:`otbench.generation.privacy_audit`.
"""
from __future__ import annotations


import numpy as np
import pandas as pd

from otbench.generation.eval.data import EvalFrames, sparse_onehot_pair
from otbench.generation.eval.metrics._distance import (
    GowerSpace,
    euclidean_min_distances,
    tpr_at_fpr,
)
from otbench.metrics.catalog import MetricResult

MAX_REFERENCE_ROWS = 4000
_MIA_AUROC = "generation.privacy.distance_mia_auroc"
_MIA_TPR = "generation.privacy.distance_mia_tpr_at_1pct_fpr"
_EUCLID = "generation.privacy.mean_min_euclidean_dcr"
_GOWER_RATIO = "generation.privacy.normalized_median_gower_dcr"
_NN_PREF = "generation.privacy.train_holdout_nn_preference"
_POSTHOC = (_MIA_AUROC, _MIA_TPR, _EUCLID, _GOWER_RATIO, _NN_PREF)


def evaluate_privacy(
    frames: EvalFrames,
    seed: int = 0,
    max_reference_rows: int = MAX_REFERENCE_ROWS,
) -> list[MetricResult]:
    schema = frames.schema
    rng = np.random.default_rng(seed)

    train = frames.original("train")
    holdout = frames.original("test")
    gen, raw_reason = frames.raw_schema_frame()
    if gen is None:
        reason = f"privacy metrics require the immutable raw output: {raw_reason}"
        return [MetricResult.skipped(name, reason) for name in _POSTHOC]

    # Members and non-members must be the same size, or the attack AUROC and the
    # Gower ratio both pick up the imbalance rather than the leak.
    n_reference = min(len(train), len(holdout), max_reference_rows)
    if n_reference < 10 or len(gen) == 0:
        reason = (
            f"needs at least 10 rows in each of train/holdout/generated "
            f"(train={len(train)}, holdout={len(holdout)}, generated={len(gen)})"
        )
        return [MetricResult.skipped(name, reason) for name in _POSTHOC]

    members = _subsample(train, n_reference, rng)
    non_members = _subsample(holdout, n_reference, rng)
    generated = _subsample(gen, min(len(gen), max_reference_rows), rng)

    space = GowerSpace.fit(frames.train, schema)
    results: list[MetricResult] = []
    shared = {"n_reference_rows": int(n_reference), "n_generated_rows": int(len(generated))}

    # --- G20: mean minimum Euclidean distance in the standardised space ----
    results.append(_mean_min_euclidean(frames, generated, members, seed, shared))

    # --- G21 / G22: Gower-based comparisons against the real baseline ------
    gen_to_train = space.min_distances(generated, members)
    holdout_to_train = space.min_distances(non_members, members)
    results.append(_gower_ratio(gen_to_train, holdout_to_train, seed, shared))
    results.append(_nn_preference(space, generated, members, non_members, seed, shared))

    # --- G17: distance-based membership attack ----------------------------
    results.extend(_distance_mia(space, generated, members, non_members, seed, shared))

    return results


def _mean_min_euclidean(frames, generated, members, seed, shared) -> MetricResult:
    real_enc, gen_enc = sparse_onehot_pair(
        members, generated, frames.schema, train_df=frames.train
    )
    if real_enc.shape[1] == 0:
        return MetricResult.skipped(_EUCLID, "the train-fitted encoding is empty")
    distances = euclidean_min_distances(gen_enc, real_enc)
    finite = distances[np.isfinite(distances)]
    if len(finite) == 0:
        return MetricResult.skipped(_EUCLID, "no finite distance could be computed")
    return MetricResult.ok(
        _EUCLID, float(finite.mean()), n=len(finite), seed=seed,
        details={**shared, "median": float(np.median(finite)),
                 "p05": float(np.quantile(finite, 0.05)),
                 "n_zero_distance": int((finite == 0).sum()),
                 "space": "train-standardised numerical + train-fitted one-hot"},
        extra={"note": "scale depends on the encoding, so it is comparable across "
                       "generators on one dataset but not across datasets"},
    )


def _gower_ratio(gen_to_train, holdout_to_train, seed, shared) -> MetricResult:
    gen_median = _median(gen_to_train)
    holdout_median = _median(holdout_to_train)
    if gen_median is None or holdout_median is None:
        return MetricResult.skipped(_GOWER_RATIO, "no finite Gower distance available")
    if holdout_median <= 0:
        return MetricResult.skipped(
            _GOWER_RATIO,
            "the holdout baseline median is zero (train and holdout overlap), so the "
            "ratio has no denominator",
        )
    return MetricResult.ok(
        _GOWER_RATIO, float(gen_median / holdout_median), n=shared["n_generated_rows"], seed=seed,
        details={**shared,
                 "median_gower_generated_to_train": gen_median,
                 "median_gower_holdout_to_train": holdout_median},
        extra={"interpretation": "1.0 means generated rows are no closer to the training "
                                 "set than genuine holdout rows are; below 1 indicates "
                                 "memorisation"},
    )


def _nn_preference(space, generated, members, non_members, seed, shared) -> MetricResult:
    to_train = space.min_distances(generated, members)
    to_holdout = space.min_distances(generated, non_members)
    valid = np.isfinite(to_train) & np.isfinite(to_holdout)
    if not valid.any():
        return MetricResult.skipped(_NN_PREF, "no finite Gower distance available")
    prefers_train = to_train[valid] < to_holdout[valid]
    ties = int((to_train[valid] == to_holdout[valid]).sum())
    return MetricResult.ok(
        _NN_PREF, float(prefers_train.mean()), n=int(valid.sum()), seed=seed,
        details={**shared, "n_ties": ties,
                 "reference_train": "benchmark train split (members)",
                 "reference_holdout": "benchmark test split (non-members)"},
        extra={"interpretation": "0.5 is the ideal; above 0.5 means generated rows sit "
                                 "closer to the training rows than to equally real holdout rows"},
    )


def _distance_mia(space, generated, members, non_members, seed, shared) -> list[MetricResult]:
    from sklearn.metrics import roc_auc_score

    member_distance = space.min_distances(members, generated)
    non_member_distance = space.min_distances(non_members, generated)
    valid = np.isfinite(member_distance).all() and np.isfinite(non_member_distance).all()
    if not valid:
        reason = "the Gower distance to the generated set is undefined for some rows"
        return [MetricResult.skipped(name, reason) for name in (_MIA_AUROC, _MIA_TPR)]

    # The attacker's guess: a record close to the generated data was probably a
    # member, so the score is the negated distance.
    scores = -np.concatenate([member_distance, non_member_distance])
    labels = np.concatenate([np.ones(len(member_distance), dtype=int),
                             np.zeros(len(non_member_distance), dtype=int)])
    details = {
        **shared,
        "threat_model": "attacker holds the generated table and one candidate record, and "
                        "guesses membership from the Gower distance to the closest "
                        "generated row",
        "members": "benchmark train split",
        "non_members": "benchmark test split",
    }
    auroc = MetricResult.ok(
        _MIA_AUROC, float(roc_auc_score(labels, scores)), n=len(labels), seed=seed, details=details
    )
    tpr = tpr_at_fpr(scores, labels, target_fpr=0.01)
    tpr_result = (
        MetricResult.ok(_MIA_TPR, tpr, n=len(labels), seed=seed, details=details)
        if tpr is not None
        else MetricResult.skipped(_MIA_TPR, "the ROC curve has no point at or below 1% FPR")
    )
    return [auroc, tpr_result]


def _subsample(frame: pd.DataFrame, size: int, rng) -> pd.DataFrame:
    if len(frame) <= size:
        return frame.reset_index(drop=True)
    index = np.sort(rng.choice(len(frame), size=size, replace=False))
    return frame.iloc[index].reset_index(drop=True)


def _median(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if len(finite) else None

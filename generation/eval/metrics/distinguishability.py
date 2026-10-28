# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""G10-G11: classifier two-sample tests (Lopez-Paz & Oquab, ICLR 2017).

A discriminator is trained to tell real rows from generated ones. If the two
distributions match, no classifier can beat chance and the held-out AUROC sits
at 0.5; the further above 0.5, the more separable the generated data is.

Two design points matter for the number to be interpretable:

* **Equal class sizes.** With unequal sizes a classifier can score above chance
  by exploiting the prior, so both sides are subsampled to the same count.
* **A held-out split.** The AUROC is measured on rows neither the logistic nor
  the tree discriminator was fitted on. Both discriminators share the same
  split, so G10 and G11 differ only in model class.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from otbench.generation.eval.data import EvalFrames
from otbench.metrics.catalog import MetricResult

TEST_FRACTION = 0.3
_NAMES = {
    "logistic": "generation.distinguishability.logistic_c2st_auroc",
    "gbdt": "generation.distinguishability.gbdt_c2st_auroc",
}


def evaluate_distinguishability(
    frames: EvalFrames,
    reference_split: str = "test",
    seed: int = 0,
) -> list[MetricResult]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    real_enc, gen_enc = frames.onehot_pair(reference_split)
    n = min(len(real_enc), len(gen_enc))
    if real_enc.shape[1] == 0 or n < 20:
        reason = f"needs at least 20 rows per side in a non-empty encoding (got {n})"
        return [MetricResult.skipped(name, reason) for name in _NAMES.values()]

    rng = np.random.default_rng(seed)
    real_sample = real_enc.iloc[np.sort(rng.choice(len(real_enc), n, replace=False))]
    gen_sample = gen_enc.iloc[np.sort(rng.choice(len(gen_enc), n, replace=False))]

    X = pd.concat([real_sample, gen_sample], ignore_index=True).to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])

    X_fit, X_eval, y_fit, y_eval = train_test_split(
        X, y, test_size=TEST_FRACTION, random_state=seed, stratify=y
    )
    if len(np.unique(y_eval)) < 2:
        reason = "the held-out split degenerated to a single class"
        return [MetricResult.skipped(name, reason) for name in _NAMES.values()]

    models = {
        "logistic": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)
        ),
        "gbdt": HistGradientBoostingClassifier(random_state=seed),
    }
    details = {
        "reference_split": reference_split,
        "n_per_class": int(n),
        "n_eval_rows": int(len(y_eval)),
        "test_fraction": TEST_FRACTION,
    }

    results = []
    for key, model in models.items():
        try:
            model.fit(X_fit, y_fit)
            scores = model.predict_proba(X_eval)[:, 1]
            results.append(MetricResult.ok(
                _NAMES[key], float(roc_auc_score(y_eval, scores)),
                n=int(len(y_eval)), seed=seed, details=details,
            ))
        except Exception as exc:  # noqa: BLE001 — recorded, not hidden
            results.append(MetricResult.failed(_NAMES[key], exc, seed=seed))
    return results

# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""SMOTE adapter (imblearn route — no synth plugin, fit at sample time).

Faithful port of the original imblearn wrapper: interpolation happens in the
one-hot processed feature space; classification oversamples per class to the
stratified totals; regression uses the dummy-class trick (target appended as a
feature, all real rows form class 0, random dummy rows form class 1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from otbench.methods.common.base import GenerateContext, TabularMethod, Task
from otbench.methods.common.capabilities import UnsupportedRun
from otbench.methods.common.seeding import seed_everything
from otbench.methods.common.synth_method import OneHotSpace


class SMOTEMethod(TabularMethod):
    name = "smote"
    tasks = frozenset({Task.GENERATION})
    trainable = False

    def generate(self, ctx: GenerateContext) -> pd.DataFrame:
        from imblearn.over_sampling import SMOTE as ImblearnSMOTE

        seed_everything(ctx.seed)
        bundle = ctx.bundle
        schema = bundle.schema
        target = schema.target_col
        n_samples = int(ctx.n_samples or len(bundle.train))
        is_classification = schema.task_type != "regression"
        k_neighbors = int(self.cfg("model", "k_neighbors", 5))

        space = OneHotSpace(bundle)
        train = bundle.train.reset_index(drop=True)
        X = space.encode(train[schema.feature_cols]).to_numpy(dtype=np.float64)
        y, classes = space.encode_target(train[target])

        if is_classification:
            counts = np.bincount(y, minlength=len(classes))
            min_class_count = int(counts.min())
            if min_class_count < 2:
                raise UnsupportedRun(
                    "smote needs at least two training rows in every class; "
                    f"the smallest class has {min_class_count}."
                )
            effective_k = min(k_neighbors, min_class_count - 1)
            class2synthetic = self._class2samples(counts, n_samples)
            class2total = {
                code: int(counts[code]) + syn for code, syn in class2synthetic.items()
            }
            model = ImblearnSMOTE(
                sampling_strategy=class2total,
                k_neighbors=effective_k,
                random_state=int(ctx.seed),
            )
            X_full, y_full = model.fit_resample(X, y)
            X_syn = np.asarray(X_full)[len(X):]
            y_syn = np.asarray(y_full)[len(X):]
        else:
            # dummy-class trick: [features, y] rows are class 0, random rows class 1
            X_aug = np.concatenate([X, np.asarray(
                pd.to_numeric(train[target], errors="coerce").fillna(0.0)
            ).reshape(-1, 1)], axis=1)
            X_dummy = np.random.rand(*X_aug.shape)
            X_fit = np.concatenate([X_aug, X_dummy], axis=0)
            y_fit = np.concatenate([
                np.zeros(len(X_aug), dtype=np.int64), np.ones(len(X_dummy), dtype=np.int64)
            ])
            class2total = {0: len(X_aug) + n_samples, 1: len(X_dummy)}
            model = ImblearnSMOTE(
                sampling_strategy=class2total,
                k_neighbors=k_neighbors,
                random_state=int(ctx.seed),
            )
            X_full, _ = model.fit_resample(X_fit, y_fit)
            gen = np.asarray(X_full)[len(X_fit):]
            X_syn, y_syn = gen[:, :-1], gen[:, -1]

        out = space.decode(np.asarray(X_syn, dtype=np.float64))
        if is_classification:
            codes = np.clip(np.asarray(y_syn, dtype=np.int64), 0, len(classes) - 1)
            out[target] = [classes[int(c)] for c in codes]
        else:
            out[target] = np.asarray(y_syn, dtype=np.float64)
        return out[schema.all_columns]

    def _class2samples(self, counts: np.ndarray, n_total: int) -> dict[int, int]:
        shares = counts.astype(float) / max(counts.sum(), 1.0)
        class2samples = {code: int(n_total * shares[code]) for code in range(len(counts))}
        tol = 0
        while sum(class2samples.values()) != n_total and tol < 100:
            code = int(np.random.choice(list(class2samples.keys())))
            class2samples[code] -= int(np.sign(sum(class2samples.values()) - n_total))
            class2samples[code] = max(class2samples[code], 1)
            tol += 1
        return class2samples

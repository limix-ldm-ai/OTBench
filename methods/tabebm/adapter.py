# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""TabEBM adapter (energy-based, training-free, TabPFN backbone).

Works in the one-hot processed feature space like the original wrapper:
classification generates ``count`` rows per class in one plugin call and
slices exact per-class counts; regression appends the target as an extra
feature and generates from a single pseudo-class (SMOTE-style).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from otbench.methods.common.base import GenerateContext, TabularMethod, Task
from otbench.methods.common.capabilities import UnsupportedRun
from otbench.methods.common.seeding import seed_everything
from otbench.methods.common.synth_method import OneHotSpace


class TabEBMMethod(TabularMethod):
    name = "tabebm"
    tasks = frozenset({Task.GENERATION})
    trainable = False

    def generate(self, ctx: GenerateContext) -> pd.DataFrame:
        from .plugin import TabEBMPlugin

        seed_everything(ctx.seed)
        bundle = ctx.bundle
        schema = bundle.schema
        target = schema.target_col
        n_samples = int(ctx.n_samples or len(bundle.train))
        is_classification = schema.task_type != "regression"

        space = OneHotSpace(bundle)
        width = len(space.columns)
        max_width = int(self.cfg("model", "max_onehot_features", 20_000))
        _validate_onehot_width(width, max_width)
        train = bundle.train.reset_index(drop=True)
        X = space.encode(train[schema.feature_cols])
        y, classes = space.encode_target(train[target])

        data_df = X.copy()
        data_df[target] = y

        kwargs = dict(self.config.get("model", {}))
        kwargs.update(self.config.get("train", {}))
        kwargs["classifier_ckpt"] = self.pretrained_path("classifier_ckpt")
        kwargs["model_device"] = self.resolve_device(ctx)
        plugin = TabEBMPlugin(seed=ctx.seed, **kwargs)
        plugin.fit(
            data_df,
            target_col=target,
            task_type="classification" if is_classification else "regression",
        )

        if is_classification:
            class2samples = self._class2samples(y, len(classes), n_samples)
            # Generate exactly the requested number for each class.  The old
            # route requested the majority count for every class and discarded
            # the excess, which was catastrophic on imbalanced datasets.
            loader = plugin.generate(count=n_samples, class_counts=class2samples)
            X_all, y_all = loader.unpack()
            X_all = pd.DataFrame(X_all).reset_index(drop=True)
            y_all = pd.Series(y_all).reset_index(drop=True)

            X_parts, y_parts = [], []
            for code, count in class2samples.items():
                pool = X_all.loc[y_all == code]
                samples = pool.iloc[:count].to_numpy()
                if samples.shape[0] < count:
                    if samples.shape[0] == 0:
                        raise RuntimeError(f"TabEBM generated no samples for class {code}.")
                    idx = np.random.choice(samples.shape[0], count)
                    samples = samples[idx]
                X_parts.append(samples)
                y_parts.append(np.full(count, code, dtype=np.int64))
            X_syn = np.concatenate(X_parts, axis=0)
            y_syn = np.concatenate(y_parts, axis=0)
        else:
            loader = plugin.generate(count=n_samples)
            X_all, y_all = loader.unpack()
            X_syn = pd.DataFrame(X_all).reset_index(drop=True).to_numpy()
            y_syn = pd.Series(y_all).reset_index(drop=True).to_numpy()
            if X_syn.shape[0] < n_samples:
                if X_syn.shape[0] == 0:
                    raise RuntimeError("TabEBM generated no samples for regression.")
                idx = np.random.choice(X_syn.shape[0], n_samples)
                X_syn, y_syn = X_syn[idx], y_syn[idx]
            X_syn, y_syn = X_syn[:n_samples], y_syn[:n_samples]

        out = space.decode(np.asarray(X_syn, dtype=np.float64))
        if is_classification:
            codes = np.clip(np.asarray(y_syn, dtype=np.int64), 0, len(classes) - 1)
            out[target] = [classes[int(c)] for c in codes]
        else:
            out[target] = np.asarray(y_syn, dtype=np.float64)
        return out[schema.all_columns]

    def _class2samples(self, y_codes: np.ndarray, n_classes: int, n_total: int) -> dict[int, int]:
        counts = np.bincount(y_codes, minlength=n_classes).astype(float)
        shares = counts / max(counts.sum(), 1.0)
        class2samples = {code: int(n_total * shares[code]) for code in range(n_classes)}
        tol = 0
        while sum(class2samples.values()) != n_total and tol < 100:
            code = int(np.random.choice(list(class2samples.keys())))
            class2samples[code] -= int(np.sign(sum(class2samples.values()) - n_total))
            class2samples[code] = max(class2samples[code], 1)
            tol += 1
        return class2samples


def _validate_onehot_width(width: int, max_width: int) -> None:
    if int(max_width) <= 0:
        raise ValueError(f"max_onehot_features must be positive, got {max_width}")
    if int(width) > int(max_width):
        raise UnsupportedRun(
            f"TabEBM one-hot width {int(width):,} exceeds the configured validated "
            f"limit {int(max_width):,}; increase max_onehot_features only after a "
            "width-matched SGLD memory validation."
        )

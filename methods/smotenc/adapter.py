# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""SMOTE-NC for mixed numerical/categorical classification data.

SMOTE-NC is the variant Chawla et al. (2002, section 6.1) defined for mixed
data: numerical features are interpolated as in plain SMOTE, while a categorical
feature takes the *majority value among the neighbours*, never an interpolated
one. The distance metric adds a penalty (the median standard deviation of the
numerical features) for each categorical mismatch.

"""
from __future__ import annotations

import numpy as np
import pandas as pd

from otbench.methods.common.base import GenerateContext, TabularMethod, Task
from otbench.methods.common.data import categorical_feature_columns, numerical_feature_columns
from otbench.methods.common.seeding import seed_everything


class SMOTENCMethod(TabularMethod):
    name = "smotenc"
    tasks = frozenset({Task.GENERATION})
    trainable = False

    def generate(self, ctx: GenerateContext) -> pd.DataFrame:
        try:
            from imblearn.over_sampling import SMOTENC
        except Exception as exc:  # noqa: BLE001 — optional dependency
            raise ImportError(
                "imbalanced-learn is not installed. Install the 'smote' extra "
                "(pip install imbalanced-learn)."
            ) from exc

        seed_everything(ctx.seed)
        schema = ctx.bundle.schema
        if (schema.task_type or "classification") == "regression":
            raise ValueError(
                "smotenc oversamples classes; regression targets have none. Use smoter."
            )

        train = ctx.bundle.train.reset_index(drop=True)
        target = schema.target_col
        numerical = numerical_feature_columns(schema)
        categorical = categorical_feature_columns(schema)
        features = numerical + categorical
        if not categorical:
            raise ValueError(
                "smotenc needs at least one categorical feature; an all-numerical dataset "
                "is plain SMOTE's case and should use the 'smote' method"
            )

        x = train[features].copy()
        for column in numerical:
            x[column] = pd.to_numeric(x[column], errors="coerce")
        x[numerical] = x[numerical].fillna(x[numerical].median())
        for column in categorical:
            x[column] = x[column].astype(str)

        classes = sorted(train[target].astype(str).unique())
        codes = train[target].astype(str).map({c: i for i, c in enumerate(classes)}).to_numpy()
        counts = np.bincount(codes, minlength=len(classes))

        k_neighbors = int(self.cfg("model", "k_neighbors", 5))
        if counts.min() <= k_neighbors:
            raise ValueError(
                f"smotenc needs more than k_neighbors={k_neighbors} rows in every class; "
                f"the smallest class has {counts.min()}"
            )

        n_samples = int(ctx.n_samples or len(train))
        # Oversample every class to its stratified share of the requested total, so
        # the synthetic table keeps the training class balance rather than being
        # silently rebalanced by the oversampler's default strategy.
        shares = counts / counts.sum()
        targets = {
            code: int(counts[code]) + max(0, int(round(n_samples * shares[code])))
            for code in range(len(classes))
        }

        cat_indices = list(range(len(numerical), len(features)))
        model = SMOTENC(
            categorical_features=cat_indices,
            sampling_strategy=targets,
            k_neighbors=k_neighbors,
            random_state=int(ctx.seed),
        )
        x_resampled, y_resampled = model.fit_resample(x, codes)

        synthetic = pd.DataFrame(x_resampled, columns=features).iloc[len(x):]
        synthetic_y = np.asarray(y_resampled)[len(x):]
        out = synthetic.reset_index(drop=True)
        for column in numerical:
            out[column] = pd.to_numeric(out[column], errors="coerce")
        out[target] = [classes[int(c)] for c in synthetic_y]
        return out[schema.all_columns].head(n_samples).reset_index(drop=True)

# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""SMOTER: SMOTE for regression (Torgo, Ribeiro, Pfahringer & Branco, 2013).

The defining rule of SMOTER, and the reason it is a separate method rather than
a configuration of SMOTE, is that **the target is interpolated together with the
features by the same convex weight**:

    x_new = x_i + w * (x_j - x_i)
    y_new = y_i + w * (y_j - y_i)     with the same w

Categorical features follow the SMOTE-NC convention: the majority value among
the two parents' neighbourhood, never an interpolated one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from otbench.methods.common.base import GenerateContext, TabularMethod, Task
from otbench.methods.common.data import categorical_feature_columns, numerical_feature_columns
from otbench.methods.common.seeding import seed_everything


class SMOTERMethod(TabularMethod):
    name = "smoter"
    tasks = frozenset({Task.GENERATION})
    trainable = False

    def generate(self, ctx: GenerateContext) -> pd.DataFrame:
        seed_everything(ctx.seed)
        rng = np.random.default_rng(ctx.seed)
        schema = ctx.bundle.schema
        if (schema.task_type or "") != "regression":
            raise ValueError(
                "smoter interpolates a numerical target; use smotenc for classification"
            )

        train = ctx.bundle.train.reset_index(drop=True)
        target = schema.target_col
        numerical = [c for c in numerical_feature_columns(schema)]
        categorical = categorical_feature_columns(schema)
        n_samples = int(ctx.n_samples or len(train))
        k = int(self.cfg("model", "k_neighbors", 5))

        y = pd.to_numeric(train[target], errors="coerce")
        keep = y.notna()
        train, y = train.loc[keep].reset_index(drop=True), y[keep].to_numpy(float)
        if len(train) <= k:
            raise ValueError(f"smoter needs more than k_neighbors={k} training rows")

        # Neighbours are found in the standardised numerical space, with a
        # categorical-mismatch penalty of the median numerical SD (SMOTE-NC's
        # distance convention), so a category flip costs about one SD.
        num_values = train[numerical].apply(pd.to_numeric, errors="coerce")
        num_values = num_values.fillna(num_values.median())
        scaled = (StandardScaler().fit_transform(num_values) if numerical
                  else np.zeros((len(train), 0)))
        penalty = float(np.median(np.std(scaled, axis=0))) if scaled.shape[1] else 1.0
        # One-hot at scale `penalty`: any mismatch contributes penalty^2 * 2.
        cat_space = np.zeros((len(train), 0))
        if categorical:
            dummies = pd.get_dummies(train[categorical].astype(str)).to_numpy(float)
            cat_space = dummies * (penalty / np.sqrt(2.0))
        space = np.hstack([scaled, cat_space])

        neighbors = NearestNeighbors(n_neighbors=k + 1).fit(space)
        _, indices = neighbors.kneighbors(space)

        seeds = rng.integers(0, len(train), size=n_samples)
        picks = indices[seeds, 1 + rng.integers(0, k, size=n_samples)]
        weights = rng.random(n_samples)

        out = pd.DataFrame(index=range(n_samples))
        for column in numerical:
            a = num_values[column].to_numpy()[seeds]
            b = num_values[column].to_numpy()[picks]
            out[column] = a + weights * (b - a)
        for column in categorical:
            # Majority over the seed's neighbourhood, per SMOTE-NC.
            values = train[column].astype(str).to_numpy()
            neighbourhood = values[indices[seeds]]
            out[column] = [_majority(row) for row in neighbourhood]
        # The target moves with the same weight as the features. This is the rule.
        out[target] = y[seeds] + weights * (y[picks] - y[seeds])

        return out[schema.all_columns].reset_index(drop=True)


def _majority(values: np.ndarray) -> str:
    unique, counts = np.unique(values, return_counts=True)
    return str(unique[int(np.argmax(counts))])

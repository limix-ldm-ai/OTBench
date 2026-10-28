# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""k-nearest-neighbour imputation against a fixed train reference set.

The neighbours always come from the *training* rows, never from other test rows.
scikit-learn's ``KNNImputer`` fits and transforms the same matrix, which on a
test frame would let test rows impute each other — transductive behaviour that
would make the result incomparable with the inductive methods it is benchmarked
against. This adapter therefore keeps the train frame as the reference set and
searches it for each incomplete test row.

Distances are computed on a train-fitted encoding (standardised numerical
columns, one-hot categorical) over the observed cells of the query row only,
which is the nan-aware distance ``KNNImputer`` uses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from otbench.methods.common.base import Task
from otbench.methods.common.imputer_base import TrainFittedImputer, train_statistics


class KNNImputerMethod(TrainFittedImputer):
    name = "knn_imputer"
    tasks = frozenset({Task.IMPUTATION})

    def fit_state(self, train, schema, seed):
        from sklearn.preprocessing import StandardScaler

        numerical = [c for c in schema.numerical_cols if c in train.columns]
        categorical = [c for c in schema.categorical_cols if c in train.columns]
        statistics = train_statistics(train, schema)

        numeric_frame = train[numerical].apply(pd.to_numeric, errors="coerce")
        numeric_frame = numeric_frame.fillna(pd.Series(
            {c: statistics[c] for c in numerical}, dtype=float))
        scaler = StandardScaler().fit(numeric_frame) if numerical else None

        return {
            "train": train.reset_index(drop=True),
            "numerical": numerical,
            "categorical": categorical,
            "scaler": scaler,
            "statistics": statistics,
            "k": int(self.cfg("model", "n_neighbors", 5)),
        }

    def apply_state(self, state, masked, schema, seed):
        train = state["train"]
        numerical, categorical = state["numerical"], state["categorical"]
        k = min(state["k"], len(train))

        reference_num = self._numeric_matrix(train, numerical, state)
        reference_cat = train[categorical].astype(str).to_numpy() if categorical else None
        query_num = self._numeric_matrix(masked, numerical, state)
        query_cat = masked[categorical].astype(str).to_numpy() if categorical else None

        out = masked.copy()
        incomplete = masked.isna().any(axis=1).to_numpy()
        for row in np.flatnonzero(incomplete):
            observed_num = ~np.isnan(query_num[row]) if numerical else np.zeros(0, bool)
            distance = np.zeros(len(train), dtype=float)
            n_terms = int(observed_num.sum())
            if n_terms:
                diff = reference_num[:, observed_num] - query_num[row, observed_num]
                distance += np.nansum(diff**2, axis=1)
            if categorical:
                observed_cat = pd.notna(masked[categorical].iloc[row]).to_numpy()
                if observed_cat.any():
                    mismatch = (reference_cat[:, observed_cat]
                                != query_cat[row, observed_cat])
                    # A category mismatch costs one squared unit, matching the
                    # standardised scale of the numerical terms.
                    distance += mismatch.sum(axis=1).astype(float)
                    n_terms += int(observed_cat.sum())
            if n_terms == 0:
                # Nothing observed in this row: fall back to the train statistic,
                # and say so rather than silently picking arbitrary neighbours.
                for column in masked.columns:
                    if pd.isna(out.iloc[row][column]):
                        out.iloc[row, out.columns.get_loc(column)] = \
                            state["statistics"].get(column)
                continue

            nearest = np.argpartition(distance, k - 1)[:k] if k < len(distance) \
                else np.arange(len(distance))
            for column in masked.columns:
                if not pd.isna(masked.iloc[row][column]):
                    continue
                values = train[column].iloc[nearest]
                if column in set(numerical):
                    filled = pd.to_numeric(values, errors="coerce").mean()
                else:
                    modes = values.astype(str).mode()
                    filled = modes.iloc[0] if len(modes) else state["statistics"].get(column)
                out.iloc[row, out.columns.get_loc(column)] = filled
        return out

    @staticmethod
    def _numeric_matrix(frame, numerical, state):
        if not numerical:
            return np.zeros((len(frame), 0))
        values = frame[numerical].apply(pd.to_numeric, errors="coerce")
        scaled = state["scaler"].transform(values.fillna(0.0))
        # Re-introduce NaN so the distance skips unobserved cells rather than
        # treating a filled zero as an observed value.
        return np.where(values.isna().to_numpy(), np.nan, scaled)

    def state_metadata(self, state):
        return {
            "n_neighbors": state["k"],
            "reference_set": "train",
            "inductive": True,
            "note": "neighbours are searched in the train frame only; test rows never "
                    "impute one another",
        }

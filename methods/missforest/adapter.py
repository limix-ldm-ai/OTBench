# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""MissForest: iterative random-forest imputation (Stekhoven & Buhlmann 2012).

The algorithm as published: initialise the missing cells with a simple guess,
then repeatedly, in order of increasing missingness, fit a random forest for
each column on the currently-completed data and replace that column's missing
cells with its predictions. Iterate until the change criterion stops decreasing.

Regression columns use ``RandomForestRegressor`` and categorical columns use
``RandomForestClassifier`` — the paper's own split, and the reason MissForest
handles mixed data natively rather than through an encoding.

MissForest predicts the conditional mean/mode, so it is deterministic. It
declares ``point`` output only, and the K-draw metrics correctly skip for it
rather than being computed over K identical tables.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from otbench.methods.common.base import Task
from otbench.methods.common.imputer_base import TrainFittedImputer, train_statistics


class MissForestMethod(TrainFittedImputer):
    name = "missforest"
    tasks = frozenset({Task.IMPUTATION})

    def fit_state(self, train, schema, seed):
        return {
            "train": train.reset_index(drop=True),
            "statistics": train_statistics(train, schema),
            "numerical": [c for c in schema.numerical_cols if c in train.columns],
            "categorical": [c for c in schema.categorical_cols if c in train.columns],
            "max_iterations": int(self.cfg("model", "max_iterations", 10)),
            "n_estimators": int(self.cfg("model", "n_estimators", 100)),
            "tolerance": float(self.cfg("model", "tolerance", 1e-3)),
            "seed": int(seed),
        }

    def state_metadata(self, state):
        return {
            "max_iterations": state["max_iterations"],
            "n_estimators": state["n_estimators"],
            "stopping_criterion": "the paper's gamma: stop when the change in the imputed "
                                  "values stops decreasing",
            "n_auxiliary_models": len(state["numerical"]) + len(state["categorical"]),
            "deterministic": True,
        }

    def apply_state(self, state, masked, schema, seed):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        numerical = set(state["numerical"])
        train = state["train"]
        domains = {c: sorted(train[c].astype(str).unique()) for c in state["categorical"]}

        current = masked.copy()
        missing = {c: current[c].isna().to_numpy() for c in current.columns}
        for column, value in state["statistics"].items():
            if column in current.columns:
                current[column] = current[column].fillna(value)

        # The paper's ordering: impute the least-missing column first, so later
        # columns are predicted from the best-completed data available.
        order = sorted(
            [c for c in current.columns if missing[c].any()],
            key=lambda c: int(missing[c].sum()),
        )
        if not order:
            return current

        previous = None
        previous_gamma = np.inf
        for _ in range(state["max_iterations"]):
            snapshot = current.copy()
            for column in order:
                rows = missing[column]
                predictors = [c for c in current.columns if c != column]
                x_train = _design(train[predictors], predictors, numerical, domains)
                x_query = _design(current.loc[rows, predictors], predictors, numerical, domains)

                if column in numerical:
                    y = pd.to_numeric(train[column], errors="coerce")
                    keep = y.notna().to_numpy()
                    if keep.sum() < 2:
                        continue
                    model = RandomForestRegressor(
                        n_estimators=state["n_estimators"], random_state=state["seed"], n_jobs=-1)
                    model.fit(x_train[keep], y[keep])
                    current.loc[rows, column] = model.predict(x_query)
                else:
                    y = train[column].astype(str)
                    if y.nunique() < 2:
                        continue
                    model = RandomForestClassifier(
                        n_estimators=state["n_estimators"], random_state=state["seed"], n_jobs=-1)
                    model.fit(x_train, y)
                    current.loc[rows, column] = model.predict(x_query)

            gamma = _change(snapshot, current, missing, numerical)
            if previous is not None and gamma >= previous_gamma:
                # The published stopping rule: stop as soon as the difference
                # stops decreasing, and return the PREVIOUS iterate.
                return previous
            previous, previous_gamma = current.copy(), gamma
            if gamma < state["tolerance"]:
                break
        return current


def _change(before: pd.DataFrame, after: pd.DataFrame, missing, numerical) -> float:
    """The paper's gamma: normalised change in the imputed cells between sweeps."""
    numeric_num, numeric_den, categorical_changes, categorical_total = 0.0, 0.0, 0, 0
    for column, rows in missing.items():
        if not rows.any():
            continue
        if column in numerical:
            a = pd.to_numeric(before.loc[rows, column], errors="coerce").to_numpy(float)
            b = pd.to_numeric(after.loc[rows, column], errors="coerce").to_numpy(float)
            numeric_num += float(np.nansum((b - a) ** 2))
            numeric_den += float(np.nansum(b**2))
        else:
            a = before.loc[rows, column].astype(str).to_numpy()
            b = after.loc[rows, column].astype(str).to_numpy()
            categorical_changes += int((a != b).sum())
            categorical_total += len(b)

    numeric_gamma = numeric_num / numeric_den if numeric_den > 0 else 0.0
    categorical_gamma = (categorical_changes / categorical_total
                         if categorical_total > 0 else 0.0)
    return float(numeric_gamma + categorical_gamma)


def _design(frame: pd.DataFrame, columns, numerical, domains) -> np.ndarray:
    """Design matrix on a fixed train-derived category domain."""
    parts = []
    for column in columns:
        if column in numerical:
            values = pd.to_numeric(frame[column], errors="coerce")
            fill = values.median() if values.notna().any() else 0.0
            parts.append(values.fillna(fill).to_numpy(float).reshape(-1, 1))
        else:
            categories = domains.get(column, [])
            index = {c: i for i, c in enumerate(categories)}
            block = np.zeros((len(frame), len(categories)), dtype=float)
            for row, value in enumerate(frame[column].astype(str).to_numpy()):
                position = index.get(value)
                if position is not None:
                    block[row, position] = 1.0
            parts.append(block)
    return np.hstack(parts) if parts else np.zeros((len(frame), 0))

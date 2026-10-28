# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Unconditional column-statistic imputation: the floor every imputer must clear.

Mean/median for numerical columns, mode for categorical, all computed on the
training split. These methods ignore every other column in the row, so they
establish exactly how much of an imputer's score comes from modelling
dependence and how much from just knowing the marginal.

They also make a specific failure visible: mean imputation shrinks a column's
variance and attenuates its correlations, so it can score well on point error
(I01/I02) while doing badly on the distributional and association metrics
(I05-I07). A benchmark where these baselines are competitive on every metric is
a benchmark whose metrics only measure central tendency.
"""
from __future__ import annotations

import pandas as pd

from otbench.methods.common.base import Task
from otbench.methods.common.imputer_base import TrainFittedImputer


class _StatisticImputer(TrainFittedImputer):
    #: "mean" or "median"; categorical columns always use the mode.
    numerical_statistic = "median"

    def fit_state(self, train, schema, seed):
        statistics: dict[str, object] = {}
        for column in train.columns:
            if column in set(schema.numerical_cols):
                values = pd.to_numeric(train[column], errors="coerce").dropna()
                if len(values) == 0:
                    statistics[column] = 0.0
                elif self.numerical_statistic == "mean":
                    statistics[column] = float(values.mean())
                else:
                    statistics[column] = float(values.median())
            else:
                modes = train[column].astype(str).mode()
                statistics[column] = str(modes.iloc[0]) if len(modes) else ""
        return {"statistics": statistics, "numerical_statistic": self.numerical_statistic}

    def apply_state(self, state, masked, schema, seed):
        out = masked.copy()
        for column, value in state["statistics"].items():
            if column in out.columns:
                out[column] = out[column].fillna(value)
        return out

    def state_metadata(self, state):
        return {
            "numerical_statistic": state["numerical_statistic"],
            "categorical_statistic": "mode",
            "conditional": False,
        }


class MeanModeMethod(_StatisticImputer):
    name = "simple_mean_mode"
    tasks = frozenset({Task.IMPUTATION})
    numerical_statistic = "mean"


class MedianModeMethod(_StatisticImputer):
    name = "simple_median_mode"
    tasks = frozenset({Task.IMPUTATION})
    numerical_statistic = "median"

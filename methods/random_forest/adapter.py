# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Random forest baseline (scikit-learn, Breiman 2001)."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from otbench.methods.common.base import Task
from otbench.methods.common.supervised import SupervisedMethod


class RandomForestMethod(SupervisedMethod):
    name = "random_forest"
    tasks = frozenset({Task.PREDICTION})

    def build_estimator(self, task_type, random_state, n_classes):
        kwargs = {
            "n_estimators": int(self.cfg("model", "n_estimators", 300)),
            "max_depth": self.cfg("model", "max_depth", None),
            "min_samples_leaf": int(self.cfg("model", "min_samples_leaf", 1)),
            "random_state": random_state,
            "n_jobs": int(self.cfg("model", "n_jobs", -1)),
        }
        if task_type == "regression":
            return RandomForestRegressor(**kwargs)
        return RandomForestClassifier(**kwargs)

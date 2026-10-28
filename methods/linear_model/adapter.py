# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Linear and logistic regression baselines (scikit-learn).

Two separate registry names sharing one module, because they are two different
models: ``linear_regression`` is ridge-regularised least squares for regression
targets, ``logistic_regression`` is a multinomial logistic classifier. Each has
its own config and checkpoint directory.

These exist to answer "does the fancy model beat a line?", which is only a
meaningful question if the line is fitted properly: standardised features (done
by the shared preprocessor), a regularisation strength that is not zero, and no
peeking at the test split.
"""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression, Ridge

from otbench.methods.common.base import Task
from otbench.methods.common.supervised import SupervisedMethod


class LinearRegressionMethod(SupervisedMethod):
    name = "linear_regression"
    tasks = frozenset({Task.PREDICTION})
    emits_proba = False

    def build_estimator(self, task_type, random_state, n_classes):
        if task_type != "regression":
            raise ValueError(
                "linear_regression is a regression model; use logistic_regression for "
                "classification targets"
            )
        return Ridge(
            alpha=float(self.cfg("model", "alpha", 1.0)),
            random_state=random_state,
        )


class LogisticRegressionMethod(SupervisedMethod):
    name = "logistic_regression"
    tasks = frozenset({Task.PREDICTION})

    def build_estimator(self, task_type, random_state, n_classes):
        if task_type == "regression":
            raise ValueError(
                "logistic_regression is a classifier; use linear_regression for "
                "numerical targets"
            )
        return LogisticRegression(
            C=float(self.cfg("model", "C", 1.0)),
            max_iter=int(self.cfg("model", "max_iter", 1000)),
            random_state=random_state,
            n_jobs=None,
        )

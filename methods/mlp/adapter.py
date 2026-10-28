# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Multi-layer perceptron baseline (scikit-learn).

Early stopping uses scikit-learn's own internal validation fraction, taken from
the rows this adapter passes it — which are train rows only.
"""
from __future__ import annotations

from sklearn.neural_network import MLPClassifier, MLPRegressor

from otbench.methods.common.base import Task
from otbench.methods.common.supervised import SupervisedMethod


class MLPMethod(SupervisedMethod):
    name = "mlp"
    tasks = frozenset({Task.PREDICTION})

    def build_estimator(self, task_type, random_state, n_classes):
        hidden = tuple(self.cfg("model", "hidden_layer_sizes", [256, 128]))
        kwargs = {
            "hidden_layer_sizes": hidden,
            "activation": self.cfg("model", "activation", "relu"),
            "alpha": float(self.cfg("model", "alpha", 1e-4)),
            "learning_rate_init": float(self.cfg("model", "learning_rate_init", 1e-3)),
            "batch_size": self.cfg("model", "batch_size", "auto"),
            "max_iter": int(self.cfg("train", "max_iter", 500)),
            "early_stopping": bool(self.cfg("train", "early_stopping", True)),
            "n_iter_no_change": int(self.cfg("train", "n_iter_no_change", 20)),
            "random_state": random_state,
        }
        if task_type == "regression":
            return MLPRegressor(**kwargs)
        return MLPClassifier(**kwargs)

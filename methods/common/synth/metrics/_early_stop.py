# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Generator early stopping with a stratified MLP real-versus-synthetic detector.

The objective is mean ROC-AUC across folds; lower values mean the generated
rows are harder to distinguish from reference rows.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


class DetectionMLP:
    """real-vs-synthetic detection ROC-AUC (sklearn MLP). Smaller is better."""

    def __init__(
        self,
        task_type: str = "classification",
        random_state: int = 0,
        workspace: Path = Path("logs/otbench_workspace"),
        n_folds: int = 3,
        reduction: str = "mean",
        **kwargs: Any,
    ) -> None:
        self._task_type = task_type
        self._random_state = random_state
        self._workspace = workspace
        self._n_folds = n_folds
        self._reduction = reduction

    @staticmethod
    def type() -> str:
        return "detection"

    @staticmethod
    def name() -> str:
        return "detection_mlp"

    @staticmethod
    def direction() -> str:
        return "minimize"

    def evaluate_default(self, X_gt: Any, X_syn: Any) -> float:
        arr_gt = np.asarray(X_gt.numpy()).reshape(len(X_gt), -1)
        arr_syn = np.asarray(X_syn.numpy()).reshape(len(X_syn), -1)
        data = np.concatenate([arr_gt, arr_syn]).astype(float)
        labels = np.concatenate([np.zeros(len(arr_gt)), np.ones(len(arr_syn))]).astype(int)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        per_class = int(min((labels == 0).sum(), (labels == 1).sum()))
        n_folds = max(2, min(self._n_folds, per_class))
        if per_class < 2:
            return 0.5  # not enough data to discriminate

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self._random_state)
        scores = []
        for train_idx, test_idx in skf.split(data, labels):
            scaler = StandardScaler().fit(data[train_idx])
            x_tr = scaler.transform(data[train_idx])
            x_te = scaler.transform(data[test_idx])
            clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, random_state=self._random_state)
            clf.fit(x_tr, labels[train_idx])
            proba = clf.predict_proba(x_te)[:, 1]
            try:
                scores.append(roc_auc_score(labels[test_idx], proba))
            except ValueError:
                scores.append(0.5)
        return float(np.mean(scores))


# (type, name) -> self-contained metric class. Extend here if a generator needs
# another early-stopping metric.
EARLY_STOP_METRICS = {
    ("detection", "detection_mlp"): DetectionMLP,
}

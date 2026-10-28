# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""LimiX adapter: pretrained tabular foundation model (prediction + imputation).

The vendored LimiX inference code (methods/limix/model) uses absolute
top-level imports (``inference``, ``model``, ``utils``), so its root is put on
``sys.path`` when the estimator loads the backend. Conflicting implementations
require separate processes. Features are numerically encoded (train-fitted mapping) because
LimiX consumes float matrices.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from otbench.methods.common.foundation import FoundationMethod, numeric_frame

_LIMIX_ROOT = Path(__file__).resolve().parent / "model"


def _ensure_limix_on_path() -> None:
    for name in ("inference", "model", "utils", "config"):
        module = sys.modules.get(name)
        filename = getattr(module, "__file__", None) if module is not None else None
        if module is not None and (not filename or not Path(filename).resolve().is_relative_to(_LIMIX_ROOT)):
            raise ImportError(f"LimiX conflicts with loaded {name!r}; run it in a separate process")
    if str(_LIMIX_ROOT) not in sys.path:
        sys.path.insert(0, str(_LIMIX_ROOT))


class LimiXMethod(FoundationMethod):
    name = "limix"
    max_imputation_classes = 10
    max_prediction_classes = 10

    def _prepare_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        return numeric_frame(X_train, X_test)

    def _estimator(self, ctx, task_type: str) -> "_LimiXEstimator":
        model_path = self.pretrained_path("model_ckpt")
        inference_config = self.cfg("model", "inference_config", None)
        if inference_config is None:
            name = (
                "cls_default_noretrieval.json"
                if task_type == "Classification"
                else "reg_default_noretrieval_impute.json"
            )
            inference_config = str(_LIMIX_ROOT / "config" / name)
        device = ctx.device
        return _LimiXEstimator(model_path, inference_config, device, task_type)

    def additional_resource_identities(self) -> list[dict]:
        from otbench.methods.common.resources import resource_identity
        configured = self.cfg("model", "inference_config", None)
        paths = [Path(configured).expanduser()] if configured else [
            _LIMIX_ROOT / "config" / "cls_default_noretrieval.json",
            _LIMIX_ROOT / "config" / "reg_default_noretrieval_impute.json",
        ]
        return [{"key": "inference_config", **resource_identity(path)} for path in paths]

    def make_regressor(self, ctx):
        return self._estimator(ctx, "Regression")

    def make_classifier(self, ctx):
        return self._estimator(ctx, "Classification")


class _LimiXEstimator:
    def __init__(self, model_path: str, inference_config: str, device, task_type: str) -> None:
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"LimiX checkpoint not found: {model_path}")
        self.model_path = model_path
        self.inference_config = inference_config
        self.device = device
        self.task_type = task_type
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=np.float32)
        self.y_train = np.asarray(
            y, dtype=np.float32 if self.task_type == "Regression" else np.int64
        )
        if self.task_type == "Classification":
            self.classes_ = np.unique(self.y_train)
        return self

    def _native_output(self, X):
        import torch

        _ensure_limix_on_path()
        LimiXPredictor = importlib.import_module("inference.predictor").LimiXPredictor
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Estimator must be fitted before predict().")
        device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        predictor = LimiXPredictor(
            device=device,
            model_path=self.model_path,
            inference_config=self.inference_config,
        )
        output = predictor.predict(
            self.X_train, self.y_train, np.asarray(X, dtype=np.float32), task_type=self.task_type
        )
        if isinstance(output, tuple):
            output = output[0]
        if hasattr(output, "detach"):
            output = output.detach().cpu().numpy()
        return np.asarray(output)

    def predict_proba(self, X):
        if self.task_type != "Classification":
            raise AttributeError("Regression has no class probabilities")
        proba = self._native_output(X)
        if proba.ndim != 2 or proba.shape[1] != len(self.classes_):
            raise ValueError("LimiX classification output does not match fitted classes")
        return proba

    def predict(self, X):
        if self.task_type == "Classification":
            return self.classes_[self.predict_proba(X).argmax(axis=1)]
        return self._native_output(X).reshape(-1)

# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""LightGBM baseline (Ke et al. 2017), via the scikit-learn API."""
from __future__ import annotations

from otbench.methods.common.base import Task
from otbench.methods.common.supervised import SupervisedMethod


def import_lightgbm():
    try:
        import lightgbm as lgb
    except Exception as exc:  # noqa: BLE001 — optional dependency
        raise ImportError(
            "lightgbm is not installed in this environment. Install the 'gbdt' extra "
            "(pip install lightgbm) or run this method in an environment that has it."
        ) from exc
    return lgb


def _eval_kwargs(model, x_valid, y_valid) -> dict:
    """LightGBM 4.7 renamed ``eval_set`` to ``eval_X``/``eval_y``.

    Both spellings exist across the versions likely to be installed, so pick
    whichever this one accepts instead of pinning the benchmark to one release.
    """
    import inspect

    if "eval_X" in inspect.signature(model.fit).parameters:
        return {"eval_X": x_valid, "eval_y": y_valid}
    return {"eval_set": [(x_valid, y_valid)]}


class LightGBMMethod(SupervisedMethod):
    name = "lightgbm"
    tasks = frozenset({Task.PREDICTION})
    uses_validation = True

    def build_estimator(self, task_type, random_state, n_classes):
        lgb = import_lightgbm()
        kwargs = {
            "n_estimators": int(self.cfg("model", "n_estimators", 500)),
            "learning_rate": float(self.cfg("model", "learning_rate", 0.05)),
            "num_leaves": int(self.cfg("model", "num_leaves", 31)),
            "min_child_samples": int(self.cfg("model", "min_child_samples", 20)),
            "subsample": float(self.cfg("model", "subsample", 1.0)),
            "colsample_bytree": float(self.cfg("model", "colsample_bytree", 1.0)),
            "random_state": random_state,
            "n_jobs": int(self.cfg("model", "n_jobs", -1)),
            "verbose": -1,
        }
        if task_type == "regression":
            return lgb.LGBMRegressor(objective="regression", **kwargs)
        return lgb.LGBMClassifier(**kwargs)

    def fit_estimator(self, pipeline, x_train, y_train, x_valid, y_valid, task_type):
        if x_valid is None or len(x_valid) == 0:
            pipeline.fit(x_train, y_train)
            return {"early_stopping": "disabled: no validation rows available"}

        lgb = import_lightgbm()
        preprocessor = pipeline.named_steps["preprocess"]
        x_train_enc = preprocessor.fit_transform(x_train, y_train)
        x_valid_enc = preprocessor.transform(x_valid)
        model = pipeline.named_steps["model"]
        rounds = int(self.cfg("train", "early_stopping_rounds", 50))
        model.fit(
            x_train_enc, y_train,
            **_eval_kwargs(model, x_valid_enc, y_valid),
            callbacks=[lgb.early_stopping(rounds, verbose=False), lgb.log_evaluation(0)],
        )
        return {
            "early_stopping": "validation split carved out of train",
            "best_iteration": int(getattr(model, "best_iteration_", -1) or -1),
        }

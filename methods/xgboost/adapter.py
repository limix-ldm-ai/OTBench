# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""XGBoost baseline (Chen & Guestrin 2016), via the scikit-learn API.

Early stopping uses the validation rows carved out of *train* by the shared
adapter. Using the test split as an eval set would tune the stopping round on
the data the method is scored against, which is the same leak the benchmark
fixes elsewhere.
"""
from __future__ import annotations

from otbench.methods.common.base import Task
from otbench.methods.common.supervised import SupervisedMethod


class XGBoostMethod(SupervisedMethod):
    name = "xgboost"
    tasks = frozenset({Task.PREDICTION})
    uses_validation = True

    def build_estimator(self, task_type, random_state, n_classes):
        try:
            import xgboost as xgb
        except Exception as exc:  # noqa: BLE001 — optional dependency
            raise ImportError(
                "xgboost is not installed in this environment. Install the 'gbdt' extra "
                "(pip install xgboost) or run this method in an environment that has it."
            ) from exc

        kwargs = {
            "n_estimators": int(self.cfg("model", "n_estimators", 500)),
            "learning_rate": float(self.cfg("model", "learning_rate", 0.1)),
            "max_depth": int(self.cfg("model", "max_depth", 6)),
            "subsample": float(self.cfg("model", "subsample", 1.0)),
            "colsample_bytree": float(self.cfg("model", "colsample_bytree", 1.0)),
            "random_state": random_state,
            "n_jobs": int(self.cfg("model", "n_jobs", -1)),
            "early_stopping_rounds": int(self.cfg("train", "early_stopping_rounds", 50)),
        }
        if task_type == "regression":
            return xgb.XGBRegressor(objective="reg:squarederror", **kwargs)
        objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
        return xgb.XGBClassifier(objective=objective, **kwargs)

    def fit_estimator(self, pipeline, x_train, y_train, x_valid, y_valid, task_type):
        if x_valid is None or len(x_valid) == 0:
            # Without a validation split there is nothing to early-stop on, so the
            # full round budget is used and that fact is recorded.
            model = pipeline.named_steps["model"]
            model.set_params(early_stopping_rounds=None)
            pipeline.fit(x_train, y_train)
            return {"early_stopping": "disabled: no validation rows available"}

        preprocessor = pipeline.named_steps["preprocess"]
        x_train_enc = preprocessor.fit_transform(x_train, y_train)
        x_valid_enc = preprocessor.transform(x_valid)
        model = pipeline.named_steps["model"]
        model.fit(x_train_enc, y_train, eval_set=[(x_valid_enc, y_valid)], verbose=False)
        return {
            "early_stopping": "validation split carved out of train",
            "best_iteration": int(getattr(model, "best_iteration", -1)),
        }

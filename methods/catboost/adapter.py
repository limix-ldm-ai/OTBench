# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""CatBoost baseline (Prokhorenkova et al. 2018).

CatBoost consumes raw categorical columns through its own ordered target
statistics, which is the whole point of the method, so this adapter sets
``native_categorical`` and passes the categories through instead of one-hot
encoding them first.
"""
from __future__ import annotations

from otbench.methods.common.base import Task
from otbench.methods.common.data import categorical_feature_columns, numerical_feature_columns
from otbench.methods.common.supervised import SupervisedMethod


class CatBoostMethod(SupervisedMethod):
    name = "catboost"
    tasks = frozenset({Task.PREDICTION})
    native_categorical = True
    uses_validation = True

    def build_estimator(self, task_type, random_state, n_classes):
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
        except Exception as exc:  # noqa: BLE001 — optional dependency
            raise ImportError(
                "catboost is not installed in this environment. Install the 'gbdt' extra "
                "(pip install catboost)."
            ) from exc

        kwargs = {
            "iterations": int(self.cfg("model", "iterations", 500)),
            "learning_rate": float(self.cfg("model", "learning_rate", 0.05)),
            "depth": int(self.cfg("model", "depth", 6)),
            "l2_leaf_reg": float(self.cfg("model", "l2_leaf_reg", 3.0)),
            "random_seed": random_state,
            "verbose": False,
            "allow_writing_files": False,
        }
        if task_type == "regression":
            return CatBoostRegressor(loss_function="RMSE", **kwargs)
        loss = "Logloss" if n_classes == 2 else "MultiClass"
        return CatBoostClassifier(loss_function=loss, **kwargs)

    def fit_estimator(self, pipeline, x_train, y_train, x_valid, y_valid, task_type):
        # The ColumnTransformer emits numerical columns first, then categorical,
        # so the categorical feature indices are the tail of the encoded matrix.
        n_numerical = len(numerical_feature_columns(self._schema))
        n_categorical = len(categorical_feature_columns(self._schema))
        cat_indices = list(range(n_numerical, n_numerical + n_categorical))

        preprocessor = pipeline.named_steps["preprocess"]
        x_train_enc = preprocessor.fit_transform(x_train, y_train)
        model = pipeline.named_steps["model"]

        fit_kwargs = {"cat_features": cat_indices}
        info: dict = {"cat_features": cat_indices}
        if x_valid is not None and len(x_valid):
            fit_kwargs["eval_set"] = (preprocessor.transform(x_valid), y_valid)
            fit_kwargs["early_stopping_rounds"] = int(
                self.cfg("train", "early_stopping_rounds", 50))
            info["early_stopping"] = "validation split carved out of train"
        else:
            info["early_stopping"] = "disabled: no validation rows available"

        model.fit(x_train_enc, y_train, **fit_kwargs)
        return info

    # The fit hook needs the schema; the adapter stashes it during train().
    _schema = None

    def train(self, ctx):
        self._schema = ctx.bundle.schema
        return super().train(ctx)

    def predict(self, ctx):
        self._schema = ctx.bundle.schema
        return super().predict(ctx)

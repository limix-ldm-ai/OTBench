# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""I08: downstream utility, measured as a gap against a clean-data oracle.

An absolute downstream score conflates two things: how hard the prediction task
is, and how much the imputation hurt. This module reports

    gap = score(model on imputed data) - score(same model on clean data)

so the task's own difficulty cancels. The direction is normalised: the gap is
always signed so that 0 means "imputation cost nothing" and a negative value
means the imputed data is worse, including for error metrics where the raw
difference has the opposite sign.

Three model families are used rather than one, because imputation error
interacts differently with a linear model (which sees a biased coefficient), a
tree ensemble (which can route around a corrupted split) and a nonlinear
kernel/network model. Every preprocessing statistic is fitted on the clean
train split only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from otbench.methods.common.data import (
    categorical_feature_columns,
    feature_columns,
    numerical_feature_columns,
    split_features_target,
)
from otbench.metrics.catalog import MetricResult

UTILITY_GAP = "imputation.downstream.utility_gap"
FAIRNESS_GAP = "imputation.fairness.downstream_gap"

# "higher is better" per metric, used to sign the gap consistently.
_HIGHER_IS_BETTER = {
    "roc_auc": True, "average_precision": True, "macro_f1": True, "brier": False,
    "nrmse": False, "mae": False, "r2": True,
}


def _classification_models(random_state: int) -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=random_state, n_jobs=1),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                             random_state=random_state),
    }


def _regression_models(random_state: int) -> dict:
    return {
        "ridge": Ridge(random_state=random_state),
        "random_forest": RandomForestRegressor(
            n_estimators=200, random_state=random_state, n_jobs=1),
        "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
        "mlp": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500,
                            random_state=random_state),
    }


@dataclass
class PreparedDownstreamUtility:
    """Train-fitted downstream models and their clean-test reference scores."""

    scorer: "_PreparedScorer | None" = None
    clean_scores: dict[str, dict[str, Any]] | None = None
    reason: str | None = None
    error: Exception | None = None


@dataclass
class _PreparedScorer:
    task: str
    schema: Any
    model_names: tuple[str, ...]
    models: dict[str, Pipeline]
    model_errors: dict[str, str]
    encoder: LabelEncoder | None = None
    scale: float | None = None

    def score(self, test: pd.DataFrame):
        if self.task == "classification":
            return self._score_classification(test)
        return self._score_regression(test)

    def _score_classification(self, test: pd.DataFrame):
        test_split = split_features_target(test.reset_index(drop=True), self.schema)
        assert self.encoder is not None
        known = test_split.y.astype(str).isin(set(self.encoder.classes_)).to_numpy()
        if known.sum() == 0:
            raise _NotEvaluable("no test row carries a label seen during training")
        y_test = self.encoder.transform(test_split.y.astype(str)[known])
        x_test = test_split.X.loc[known]

        scores: dict[str, dict[str, float | None]] = {}
        predictions = None
        for name in self.model_names:
            if name in self.model_errors:
                scores[name] = {
                    key: None
                    for key in ("macro_f1", "roc_auc", "average_precision", "brier")
                }
                scores[name]["error"] = self.model_errors[name]
                continue
            try:
                pipe = self.models[name]
                pred = pipe.predict(x_test)
                proba = pipe.predict_proba(x_test)
                scores[name] = {
                    "macro_f1": float(
                        f1_score(y_test, pred, average="macro", zero_division=0)
                    ),
                    "roc_auc": _roc_auc(y_test, proba, len(self.encoder.classes_)),
                    "average_precision": _average_precision(
                        y_test, proba, len(self.encoder.classes_)
                    ),
                    "brier": _brier(y_test, proba),
                }
                if predictions is None:
                    predictions = {
                        "y_true": np.asarray(y_test), "y_pred": np.asarray(pred)
                    }
            except Exception as exc:  # noqa: BLE001 — one model may fail at predict time
                scores[name] = {
                    key: None
                    for key in ("macro_f1", "roc_auc", "average_precision", "brier")
                }
                scores[name]["error"] = f"{type(exc).__name__}: {exc}"
        return scores, predictions

    def _score_regression(self, test: pd.DataFrame):
        test_split = split_features_target(test.reset_index(drop=True), self.schema)
        y_test = pd.to_numeric(test_split.y, errors="coerce")
        valid = y_test.notna().to_numpy()
        if valid.sum() == 0:
            raise _NotEvaluable("no test row has a parseable numerical target")
        x_test, y_test = test_split.X.loc[valid], y_test[valid].to_numpy(float)
        scale = float(self.scale)

        scores: dict[str, dict[str, float | None]] = {}
        predictions = None
        for name in self.model_names:
            if name in self.model_errors:
                scores[name] = {key: None for key in ("nrmse", "mae", "r2")}
                scores[name]["error"] = self.model_errors[name]
                continue
            try:
                pred = np.asarray(self.models[name].predict(x_test), dtype=float)
                scores[name] = {
                    "nrmse": float(np.sqrt(mean_squared_error(y_test, pred))) / scale,
                    "mae": float(mean_absolute_error(y_test, pred)) / scale,
                    "r2": float(r2_score(y_test, pred)),
                }
                if predictions is None:
                    predictions = {"y_true": y_test, "y_pred": pred}
            except Exception as exc:  # noqa: BLE001
                scores[name] = {key: None for key in ("nrmse", "mae", "r2")}
                scores[name]["error"] = f"{type(exc).__name__}: {exc}"
        return scores, predictions


def prepare_downstream_utility(
    train: pd.DataFrame,
    clean_test: pd.DataFrame,
    schema,
    random_state: int = 0,
) -> PreparedDownstreamUtility:
    """Fit downstream models once and score the invariant clean test once."""

    try:
        scorer = _prepare_downstream_scorer(train, schema, random_state)
        clean_scores, _ = scorer.score(clean_test)
        return PreparedDownstreamUtility(scorer=scorer, clean_scores=clean_scores)
    except _NotEvaluable as exc:
        return PreparedDownstreamUtility(reason=str(exc))
    except Exception as exc:  # surfaced by the metric-family error boundary
        return PreparedDownstreamUtility(error=exc)


def evaluate_downstream_utility(
    train: pd.DataFrame,
    imputed_test: pd.DataFrame,
    clean_test: pd.DataFrame,
    schema,
    random_state: int = 0,
    group_cols: list[str] | None = None,
    prepared: PreparedDownstreamUtility | None = None,
) -> list[MetricResult]:
    """I08 (and I19 when groups are declared) for one completed dataset."""
    task = schema.task_type or (
        "regression" if schema.target_col in schema.numerical_cols else "classification"
    )
    prepared = prepared or prepare_downstream_utility(
        train, clean_test, schema, random_state
    )
    if prepared.reason is not None:
        return [MetricResult.skipped(UTILITY_GAP, prepared.reason),
                MetricResult.skipped(FAIRNESS_GAP, prepared.reason)]
    if prepared.error is not None:
        raise prepared.error
    assert prepared.scorer is not None and prepared.clean_scores is not None
    try:
        imputed_scores, predictions = prepared.scorer.score(imputed_test)
    except _NotEvaluable as exc:
        return [MetricResult.skipped(UTILITY_GAP, str(exc)),
                MetricResult.skipped(FAIRNESS_GAP, str(exc))]
    clean_scores = prepared.clean_scores

    per_model: dict[str, dict[str, Any]] = {}
    gaps: list[float] = []
    for model_name, imputed_metrics in imputed_scores.items():
        clean_metrics = clean_scores.get(model_name, {})
        record: dict[str, Any] = {}
        for metric_name, imputed_value in imputed_metrics.items():
            if metric_name not in _HIGHER_IS_BETTER:  # e.g. the recorded failure reason
                record[metric_name] = imputed_value
                continue
            clean_value = clean_metrics.get(metric_name)
            if imputed_value is None or clean_value is None:
                record[metric_name] = {"imputed": imputed_value, "clean": clean_value,
                                       "gap": None}
                continue
            raw = imputed_value - clean_value
            gap = raw if _HIGHER_IS_BETTER[metric_name] else -raw
            record[metric_name] = {"imputed": imputed_value, "clean": clean_value, "gap": gap}
            gaps.append(gap)
        per_model[model_name] = record

    if not gaps:
        reason = "every downstream model failed on both the imputed and the clean data"
        return [MetricResult.skipped(UTILITY_GAP, reason),
                MetricResult.skipped(FAIRNESS_GAP, reason)]

    utility = MetricResult.ok(
        UTILITY_GAP, float(np.mean(gaps)), n=len(gaps),
        details={"per_model": per_model, "task": task},
        extra={"definition": "imputed minus clean-oracle score, signed so that 0 means "
                             "the imputation cost nothing and negative means it hurt",
               "models": sorted(per_model), "random_state": random_state},
    )
    return [utility, _fairness_gap(clean_test, predictions, group_cols, task)]


def _fairness_gap(clean_test, predictions, group_cols, task) -> MetricResult:
    """I19: between-group spread of the downstream score on the imputed data."""
    if not group_cols:
        return MetricResult.skipped(
            FAIRNESS_GAP,
            "the schema declares no group columns; subgroup gaps cannot be invented from "
            "the data without deciding which attribute is protected",
        )
    available = [c for c in group_cols if c in clean_test.columns]
    if not available or predictions is None:
        return MetricResult.skipped(
            FAIRNESS_GAP, f"declared group columns {group_cols} are absent from the test frame"
        )

    per_group: dict[str, dict[str, float]] = {}
    for col in available:
        scores: dict[str, float] = {}
        for value, index in clean_test.groupby(clean_test[col].astype(str)).groups.items():
            rows = np.asarray(index)
            if len(rows) < 10:
                continue
            y_true, y_pred = predictions["y_true"][rows], predictions["y_pred"][rows]
            if task == "classification":
                scores[str(value)] = float(f1_score(y_true, y_pred, average="macro",
                                                    zero_division=0))
            else:
                scores[str(value)] = -float(mean_absolute_error(y_true, y_pred))
        if len(scores) >= 2:
            per_group[col] = scores

    if not per_group:
        return MetricResult.skipped(
            FAIRNESS_GAP, "no declared group column had at least two subgroups of >=10 rows"
        )
    spreads = {col: max(s.values()) - min(s.values()) for col, s in per_group.items()}
    return MetricResult.ok(
        FAIRNESS_GAP, float(max(spreads.values())), n=len(per_group),
        details={"per_group_score": per_group, "per_column_spread": spreads},
        extra={"aggregation": "worst spread over declared group columns",
               "score": "macro-F1" if task == "classification" else "negative MAE"},
    )


class _NotEvaluable(RuntimeError):
    """The downstream comparison cannot be set up at all (not a model failure)."""


def _fit_models(train_x, train_y, schema, models: dict) -> tuple[
    tuple[str, ...], dict[str, Pipeline], dict[str, str]
]:
    names = tuple(models)
    fitted: dict[str, Pipeline] = {}
    errors: dict[str, str] = {}
    for name, model in models.items():
        pipe = Pipeline([("preprocess", _preprocessor(schema)), ("model", model)])
        try:
            pipe.fit(train_x, train_y)
            fitted[name] = pipe
        except Exception as exc:  # noqa: BLE001 — one model failure is recorded
            errors[name] = f"{type(exc).__name__}: {exc}"
    return names, fitted, errors


def _prepare_downstream_scorer(train, schema, random_state) -> _PreparedScorer:
    task = schema.task_type or (
        "regression" if schema.target_col in schema.numerical_cols else "classification"
    )
    if task == "classification":
        return _prepare_classification_scorer(train, schema, random_state)
    return _prepare_regression_scorer(train, schema, random_state)


def _prepare_classification_scorer(train, schema, random_state) -> _PreparedScorer:
    train_split = split_features_target(train.reset_index(drop=True), schema)

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_split.y.astype(str))
    if len(encoder.classes_) < 2:
        raise _NotEvaluable("the training target has a single class")
    names, models, errors = _fit_models(
        train_split.X, y_train, schema, _classification_models(random_state)
    )
    return _PreparedScorer(
        task="classification",
        schema=schema,
        model_names=names,
        models=models,
        model_errors=errors,
        encoder=encoder,
    )


def _prepare_regression_scorer(train, schema, random_state) -> _PreparedScorer:
    train_split = split_features_target(train.reset_index(drop=True), schema)
    y_train = pd.to_numeric(train_split.y, errors="coerce")
    finite = y_train.notna().to_numpy()
    x_train, y_train = train_split.X.loc[finite], y_train[finite]

    # A train-fitted normaliser keeps NRMSE comparable across imputers.
    scale = float(y_train.std()) or 1.0
    names, models, errors = _fit_models(
        x_train, y_train, schema, _regression_models(random_state)
    )
    return _PreparedScorer(
        task="regression",
        schema=schema,
        model_names=names,
        models=models,
        model_errors=errors,
        scale=scale,
    )


def _score_classification(train, test, schema, random_state):
    return _prepare_classification_scorer(train, schema, random_state).score(test)


def _score_regression(train, test, schema, random_state):
    return _prepare_regression_scorer(train, schema, random_state).score(test)


def _roc_auc(y_true, proba, n_classes) -> float | None:
    try:
        if n_classes == 2:
            return float(roc_auc_score(y_true, proba[:, 1]))
        return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
    except ValueError:
        return None  # a split with a single class present in the test rows


def _average_precision(y_true, proba, n_classes) -> float | None:
    try:
        if n_classes == 2:
            return float(average_precision_score(y_true, proba[:, 1]))
        onehot = np.eye(n_classes)[y_true]
        return float(average_precision_score(onehot, proba, average="macro"))
    except ValueError:
        return None


def _brier(y_true, proba) -> float | None:
    try:
        onehot = np.zeros_like(proba)
        onehot[np.arange(len(y_true)), y_true] = 1.0
        return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))
    except (ValueError, IndexError):
        return None


def _preprocessor(schema) -> ColumnTransformer:
    try:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_feature_columns(schema)),
            ("cat", one_hot, categorical_feature_columns(schema)),
        ],
        remainder="drop",
    )

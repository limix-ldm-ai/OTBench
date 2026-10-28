# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""HyperImpute (Jarrett et al., ICML 2022) — AutoML iterative imputation.

The official plugin is transductive: ``fit()`` is intentionally a no-op and
``fit_transform()`` selects and fits per-column models on the incomplete matrix
being completed. Consequently there is no honest train-only checkpoint to
reuse across MCAR/MAR/MNAR settings; doing so would be the paper's fixed-model
ablation, not HyperImpute. This adapter records that cost as inference and does
not create a misleading empty checkpoint.

Categorical columns are ordinal-coded against the clean train vocabulary only
for a stable numeric interface and decoded by nearest valid code. Upstream makes
no multiple-imputation claim, so the registry declares point output only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from otbench.methods.common.base import ImputationOutput, ImputeContext, TabularMethod, Task
from otbench.methods.common.imputer_base import merge_masked_only
from otbench.methods.common.seeding import seed_everything
from otbench.methods.common.upstream import require_package

UPSTREAM_URL = "https://github.com/vanderschaarlab/hyperimpute"

# Upstream's own defaults for the inner search.
DEFAULT_CLASSIFIERS = ["random_forest", "logistic_regression", "xgboost", "catboost",
                       "neural_nets"]
DEFAULT_REGRESSORS = ["random_forest_regressor", "linear_regression", "xgboost_regressor",
                      "catboost_regressor", "neural_nets_regression"]


def _usable_zoo(classifiers, regressors):
    """Drop search-space entries that cannot be constructed here, and say which.

    hyperimpute 0.1.17 builds some of its plugins with arguments that newer
    scikit-learn removed (``LogisticRegression(multi_class=...)``, dropped in
    1.7). Those plugins raise on construction and would abort the whole fit.

    Shrinking the search space changes the method, so the exclusions are
    recorded in the result metadata rather than applied quietly: a HyperImpute run
    that never considered logistic regression is not the published HyperImpute,
    and the results have to say so.
    """
    from hyperimpute.plugins.prediction import Predictions

    usable, excluded = {}, {}
    for category, names in (("classifier", classifiers), ("regression", regressors)):
        registry = Predictions(category=category)
        keep = []
        for name in names:
            try:
                registry.get(name)
                keep.append(name)
            except Exception as exc:  # noqa: BLE001 — upstream/sklearn drift
                excluded[name] = f"{type(exc).__name__}: {exc}"
        usable[category] = keep
    return usable["classifier"], usable["regression"], excluded


class HyperImputeMethod(TabularMethod):
    name = "hyperimpute"
    tasks = frozenset({Task.IMPUTATION})
    trainable = False

    def _new_plugin(self, seed, classifiers, regressors):
        require_package("hyperimpute", method="hyperimpute", url=UPSTREAM_URL,
                        extra="imputation")
        from hyperimpute.plugins.imputers import Imputers

        return Imputers().get(
            "hyperimpute",
            classifier_seed=classifiers,
            regression_seed=regressors,
            random_state=int(seed),
            n_inner_iter=int(self.cfg("model", "n_inner_iter", 10)),
            select_model_by_column=bool(self.cfg("model", "select_model_by_column", True)),
            select_model_by_iteration=bool(self.cfg("model", "select_model_by_iteration", True)),
            select_lazy=bool(self.cfg("model", "select_lazy", True)),
            optimize_thresh=int(self.cfg("model", "optimize_thresh", 1000)),
            baseline_imputer=int(self.cfg("model", "baseline_imputer", 0)),
        )

    def impute(self, ctx: ImputeContext) -> pd.DataFrame:
        return self.impute_draws(ctx).point_estimate(ctx.bundle.schema)

    def impute_draws(self, ctx: ImputeContext) -> ImputationOutput:
        require_package("hyperimpute", method=self.name, url=UPSTREAM_URL,
                        extra="imputation")
        schema = ctx.bundle.schema
        train = ctx.bundle.train.reset_index(drop=True)[schema.all_columns]
        masked = ctx.masked_test.reset_index(drop=True)[schema.all_columns]
        codec = _Codec.fit(train, schema)
        encoded = codec.encode(masked)
        classifiers, regressors, excluded = _usable_zoo(
            list(self.cfg("model", "classifier_seed", DEFAULT_CLASSIFIERS)),
            list(self.cfg("model", "regression_seed", DEFAULT_REGRESSORS)),
        )

        draws, traces, reset_counts = [], [], []
        for draw_index in range(max(1, int(ctx.trials))):
            draw_seed = int(ctx.seed) + draw_index
            seed_everything(draw_seed)
            plugin = self._new_plugin(draw_seed, classifiers, regressors)
            compat_stats = _patch_lazy_xgboost_reuse(plugin)
            # This is the official execution route. fit() alone does not learn
            # anything; fit_transform() fits on this setting's incomplete rows.
            imputed = plugin.fit_transform(encoded.copy())
            completed = codec.decode(pd.DataFrame(
                np.asarray(imputed, dtype=float), columns=codec.columns,
            ))
            draws.append(merge_masked_only(
                masked, completed, ctx.mask.reset_index(drop=True),
            ))
            trace = plugin.trace() if callable(getattr(plugin, "trace", None)) else {}
            traces.append(_selected(trace))
            reset_counts.append(int(compat_stats["xgboost_objective_resets"]))

        return ImputationOutput.from_frames(draws, metadata={
            "source": "official hyperimpute fit_transform",
            "n_draws": len(draws),
            "learning_mode": "transductive",
            "fitted_on": "masked_test_setting",
            "inference_includes_fit": True,
            "cross_setting_checkpoint_reuse": False,
            "cross_setting_reuse_reason": (
                "upstream resets and adaptively selects per-column models for each "
                "incomplete matrix; fixed reuse is a different ablation"
            ),
            "search_space": {"classifiers": classifiers, "regressors": regressors},
            "excluded_models": excluded,
            "selected_models_per_draw": traces,
            "select_lazy": bool(self.cfg("model", "select_lazy", True)),
            "lazy_xgboost_objective_resets": int(sum(reset_counts)),
            "multiple_imputation": (
                "not claimed upstream; only point metrics are declared"
            ),
        })


def _patch_lazy_xgboost_reuse(plugin) -> dict[str, int]:
    """Reset stale XGBClassifier objective state on upstream lazy reuse.

    HyperImpute deep-copies an already fitted same-type estimator. XGBoost keeps
    ``multi:softprob`` after a multiclass fit; refitting that copy on a binary
    column then fails with ``num_class=0``. Resetting the sklearn estimator's
    objective before the required refit preserves HyperImpute's lazy architecture
    reuse while allowing XGBoost to infer the new target cardinality.
    """
    core = getattr(plugin, "model", None)
    original = getattr(core, "_check_similar", None)
    stats = {"xgboost_objective_resets": 0}
    if not callable(original):
        return stats

    def checked(frame, column):
        candidate = original(frame, column)
        categorical = column in set(getattr(core, "categorical_cols", []))
        name = candidate.name() if candidate is not None and callable(
            getattr(candidate, "name", None)
        ) else ""
        if categorical and name == "xgboost":
            estimator = getattr(candidate, "model", None)
            if estimator is not None and callable(getattr(estimator, "set_params", None)):
                estimator.set_params(objective="binary:logistic", num_class=None)
                stats["xgboost_objective_resets"] += 1
        return candidate

    core._check_similar = checked
    return stats


def _selected(trace) -> dict:
    if not isinstance(trace, dict):
        return {}
    models = trace.get("models") if "models" in trace else trace
    if not isinstance(models, dict):
        return {}
    return {
        str(key): [str(item) for item in value] if isinstance(value, list) else str(value)
        for key, value in list(models.items())[:64]
    }


class _Codec:
    """Train-fitted ordinal coding, because upstream takes a numeric frame."""

    def __init__(self, columns, categories):
        self.columns = list(columns)
        self.categories = categories

    @classmethod
    def fit(cls, train: pd.DataFrame, schema) -> "_Codec":
        categories = {
            c: sorted(train[c].dropna().astype(str).unique())
            for c in schema.categorical_cols if c in train.columns
        }
        return cls(schema.all_columns, categories)

    def encode(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=range(len(frame)))
        for column in self.columns:
            values = frame[column].reset_index(drop=True)
            if column in self.categories:
                index = {v: i for i, v in enumerate(self.categories[column])}
                # An unseen test category has no code; leaving it NaN would make
                # upstream impute an observed cell, so it maps to the nearest
                # thing available - the code is absent, and it stays missing
                # only when the cell really is missing.
                out[column] = [np.nan if pd.isna(v) else float(index.get(str(v), 0))
                               for v in values]
            else:
                out[column] = pd.to_numeric(values, errors="coerce").astype(float)
        return out

    def decode(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=range(len(frame)))
        for column in self.columns:
            values = frame[column].to_numpy(dtype=float)
            if column in self.categories:
                levels = self.categories[column]
                codes = np.clip(np.rint(np.nan_to_num(values)), 0, len(levels) - 1).astype(int)
                out[column] = [levels[c] for c in codes]
            else:
                out[column] = values
        return out

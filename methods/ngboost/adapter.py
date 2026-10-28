# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""NGBoost (Duan et al., ICML 2020) — natural-gradient boosting.

NGBoost fits the *parameters* of a chosen distribution by gradient boosting, so
unlike every other GBDT here it has a genuine conditional density. That is the
reason it is in the benchmark: it is the reference point for the probabilistic
regression metrics, and the only tree method that can be scored on NLL without
anyone inventing a density for it.

Everything probabilistic is read off the upstream ``pred_dist`` object:

* ``samples`` from ``dist.sample(S)``,
* ``density`` from ``dist.logpdf(y)``,
* ``quantiles`` from ``dist.ppf(level)``.

No KDE, no Gaussian fitted to the samples, no quantiles interpolated from an
empirical CDF. If a distribution upstream cannot answer one of these, that
output is simply absent and the dependent metrics skip.

Regression only. ``NGBClassifier`` exists upstream but adds nothing over the
GBDT baselines already registered, and the registry declares regression only.
"""
from __future__ import annotations

import numpy as np

from otbench.methods.common.base import PredictionOutput, Task
from otbench.methods.common.supervised import SupervisedMethod
from otbench.methods.common.upstream import require_package

UPSTREAM_URL = "https://github.com/stanfordmlgroup/ngboost"


class NGBoostMethod(SupervisedMethod):
    name = "ngboost"
    tasks = frozenset({Task.PREDICTION})
    # NGBoost's own output is the distribution; there is no predict_proba here
    # because the method is registered for regression only.
    emits_proba = False
    uses_validation = True

    def build_estimator(self, task_type, random_state, n_classes):
        if task_type != "regression":
            raise ValueError(
                "ngboost is registered for regression only. Use one of the GBDT "
                "baselines (xgboost / lightgbm / catboost) for classification."
            )
        ngboost = require_package("ngboost", method="ngboost", url=UPSTREAM_URL,
                                  extra="prediction")
        from ngboost.distns import Normal
        from ngboost.learners import default_tree_learner

        distribution = {"normal": Normal}.get(str(self.cfg("model", "distribution", "normal")))
        if distribution is None:
            raise ValueError(
                f"ngboost: unsupported distribution "
                f"{self.cfg('model', 'distribution', 'normal')!r}. Only 'normal' is wired up; "
                f"adding another means checking that it implements sample/logpdf/ppf."
            )

        return ngboost.NGBRegressor(
            Dist=distribution,
            Base=default_tree_learner,
            n_estimators=int(self.cfg("model", "n_estimators", 500)),
            learning_rate=float(self.cfg("model", "learning_rate", 0.01)),
            minibatch_frac=float(self.cfg("model", "minibatch_frac", 1.0)),
            natural_gradient=True,
            random_state=random_state,
            verbose=False,
        )

    def fit_estimator(self, pipeline, x_train, y_train, x_valid, y_valid, task_type):
        preprocessor = pipeline.named_steps["preprocess"]
        model = pipeline.named_steps["model"]
        encoded = preprocessor.fit_transform(x_train, y_train)

        info: dict = {}
        if x_valid is not None and len(x_valid):
            # Upstream selects the best boosting iteration on this split; it is
            # carved out of train, so test never influences the model size.
            model.fit(encoded, y_train,
                      X_val=preprocessor.transform(x_valid), Y_val=y_valid,
                      early_stopping_rounds=int(self.cfg("train", "early_stopping_rounds", 50)))
            info["early_stopping"] = "validation split carved out of train"
            best = getattr(model, "best_val_loss_itr", None)
            if best is not None:
                info["best_iteration"] = int(best)
        else:
            model.fit(encoded, y_train)
            info["early_stopping"] = "disabled: no validation rows available"
        return info

    def _regression_output(self, pipeline, x_test, meta, ctx) -> PredictionOutput:
        from otbench.methods.common.supervised import _staged_transform

        model, encoded = _staged_transform(pipeline, x_test)
        distribution = model.pred_dist(encoded)

        n_samples = max(1, int(getattr(ctx, "n_samples", 100)))
        rng = np.random.default_rng(int(ctx.seed))
        # ngboost samples with the global numpy state; seed it so the draws are
        # reproducible for a given inference seed.
        np.random.seed(int(rng.integers(0, 2**31 - 1)))
        samples = np.asarray(distribution.sample(n_samples), dtype=float)
        if samples.shape[0] == n_samples:  # upstream returns (S, n)
            samples = samples.T

        levels = list(ctx.quantile_levels)
        quantiles = np.column_stack([np.asarray(distribution.ppf(q), dtype=float)
                                     for q in levels])

        def log_prob_fn(y, _distribution=distribution):
            return np.asarray(_distribution.logpdf(np.asarray(y, dtype=float)), dtype=float)

        return PredictionOutput(
            y_pred=np.asarray(model.predict(encoded), dtype=float).ravel(),
            y_samples=samples,
            quantiles=quantiles,
            quantile_levels=levels,
            log_prob_fn=log_prob_fn,
            metadata={
                "method": self.name,
                "predictive_distribution": type(distribution).__name__,
                "density_source": "ngboost pred_dist.logpdf (a real conditional density, "
                                  "not a KDE over samples)",
                "quantile_source": "ngboost pred_dist.ppf",
                "n_samples": n_samples,
                **{k: meta[k] for k in ("n_train_rows", "n_validation_rows")},
            },
        )

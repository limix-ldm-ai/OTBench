# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""MICE: multiple imputation by chained equations (van Buuren & Groothuis-Oudshoorn 2011).

Two things distinguish this from "run an iterative imputer K times", and both
are the reason the method exists:

**1. Each chain draws from an approximate posterior predictive distribution,
not its mean.** A Bayesian-bootstrap weight draw refits every conditional model;
then a regression chain adds residual noise and a classification chain samples
from the fitted class probabilities. An
imputer that fills in the conditional mean produces K nearly identical tables;
its between-draw variance is ~0, Rubin's total variance collapses to the
within-imputation term, and every interval it produces is too narrow. Reporting
that as multiple imputation would make an overconfident method look calibrated.

**2. Categorical columns are modelled by a classifier.** Encoding a category as
an integer, fitting a regression and rounding the prediction back is not a model
of a categorical variable: it imposes an order that does not exist, and the
rounding is not sampling from anything. Each categorical column here gets a
logistic regression over the other columns and is sampled from its predicted
probabilities.

The K chains are independent and differ only by seed.  The Bayesian bootstrap
captures conditional-model uncertainty without pretending the fixed coefficients
of one Ridge/Logistic fit are posterior draws.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from otbench.methods.common.base import ImputationOutput, ImputeContext, Task
from otbench.methods.common.imputer_base import (
    TrainFittedImputer,
    merge_masked_only,
    train_statistics,
)
from otbench.methods.common.seeding import seed_everything


class MICEMethod(TrainFittedImputer):
    name = "mice"
    tasks = frozenset({Task.IMPUTATION})

    def fit_state(self, train, schema, seed):
        # The chains are refitted on each completed frame during the sweeps, so
        # what is checkpointed is the training frame that seeds them plus the
        # train-fitted starting statistics.
        return {
            "train": train.reset_index(drop=True),
            "statistics": train_statistics(train, schema),
            "numerical": [c for c in schema.numerical_cols if c in train.columns],
            "categorical": [c for c in schema.categorical_cols if c in train.columns],
            # Category domains are fixed here, from train, so every chain's design
            # matrix has the same columns in the same order.
            "domains": {c: sorted(train[c].astype(str).unique())
                        for c in schema.categorical_cols if c in train.columns},
            "n_iterations": int(self.cfg("model", "n_iterations", 10)),
        }

    def state_metadata(self, state):
        return {
            "n_iterations": state["n_iterations"],
            "categorical_model": "Bayesian-bootstrap logistic regression plus class draw",
            "numerical_model": "Bayesian-bootstrap ridge plus residual noise draw",
            "n_auxiliary_models": len(state["numerical"]) + len(state["categorical"]),
        }

    # A single draw is one chain; the multi-draw path is the real interface.
    def apply_state(self, state, masked, schema, seed):
        return _run_chain(state, masked, schema, seed, design_cache=_new_design_cache(state))

    def impute_draws(self, ctx: ImputeContext) -> ImputationOutput:
        import joblib

        state = joblib.load(Path(ctx.checkpoint_dir) / "imputer.joblib")
        schema = ctx.bundle.schema
        masked = ctx.masked_test.reset_index(drop=True)
        mask = ctx.mask.reset_index(drop=True)
        k = max(1, int(ctx.trials))

        # Train-side encodings and scalers depend only on the checkpointed train
        # split. Share them across independent chains; posterior weights and
        # conditional models remain chain- and sweep-specific below.
        design_cache = _new_design_cache(state)
        draws = []
        for index in range(k):
            seed_everything(ctx.seed + index)
            completed = _run_chain(
                state,
                masked[schema.all_columns],
                schema,
                ctx.seed + index,
                design_cache=design_cache,
            )
            draws.append(merge_masked_only(masked, completed, mask))

        return ImputationOutput.from_frames(draws, metadata={
            "source": "native",
            "draw_strategy": "K independent Bayesian-bootstrap MICE chains sampling the "
                             "approximate posterior predictive distribution",
            "n_draws": k,
            "n_iterations": state["n_iterations"],
        })


@dataclass(frozen=True)
class _TrainingProblem:
    """Immutable train-side inputs reused by every sweep and MICE chain."""

    predictors: tuple[str, ...]
    x_scaled: Any
    scaler: Any | None
    y: np.ndarray
    keep: np.ndarray | None
    classes: tuple[str, ...]


def _new_design_cache(state) -> dict[str, Any]:
    """Return a cache local to one inference call, never persisted in a checkpoint."""
    return {
        "domain_indexes": {
            column: {value: index for index, value in enumerate(categories)}
            for column, categories in state["domains"].items()
        },
        "training_problems": {},
    }


def _training_problem(state, column: str, predictors, design_cache) -> _TrainingProblem:
    """Build a sparse train design and unweighted scaler once per target column."""
    from sklearn.preprocessing import StandardScaler

    key = (column, tuple(predictors))
    cached = design_cache["training_problems"].get(key)
    if cached is not None:
        return cached

    train = state["train"]
    numerical = set(state["numerical"])
    x_train = _design(
        train[list(predictors)],
        predictors,
        numerical,
        state["domains"],
        domain_indexes=design_cache["domain_indexes"],
    )

    if column in numerical:
        y_series = pd.to_numeric(train[column], errors="coerce")
        keep = y_series.notna().to_numpy()
        y = y_series.to_numpy(float)
        classes: tuple[str, ...] = ()
    else:
        y_series = train[column].astype(str)
        classes = tuple(sorted(y_series.unique()))
        class_index = {value: index for index, value in enumerate(classes)}
        y = y_series.map(class_index).to_numpy()
        keep = None

    enough_rows = keep is None or int(keep.sum()) >= 2
    enough_classes = column in numerical or len(classes) >= 2
    if enough_rows and enough_classes:
        x_fit = x_train if keep is None else x_train[keep]
        # Sparse matrices cannot be explicitly centred without becoming dense.
        # For Ridge/LogisticRegression with their unpenalized fitted intercept,
        # omitting the fixed feature translation is prediction-equivalent: the
        # intercept absorbs -mean / scale while scale_ stays identical to the
        # former dense StandardScaler.
        scaler = StandardScaler(with_mean=False).fit(x_fit)
        x_scaled = scaler.transform(x_fit).tocsr()
    else:
        scaler = None
        x_scaled = None

    problem = _TrainingProblem(
        predictors=tuple(predictors),
        x_scaled=x_scaled,
        scaler=scaler,
        y=y,
        keep=keep,
        classes=classes,
    )
    design_cache["training_problems"][key] = problem
    return problem


def _run_chain(
    state,
    masked: pd.DataFrame,
    schema,
    seed: int,
    *,
    design_cache: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """One MICE chain: seed with train statistics, then sweep column by column."""
    from sklearn.linear_model import LogisticRegression, Ridge

    rng = np.random.default_rng(seed)
    numerical = state["numerical"]
    numerical_set = set(numerical)
    train = state["train"]
    domains = state["domains"]
    if design_cache is None:
        design_cache = _new_design_cache(state)

    current = masked.copy()
    missing = {c: current[c].isna().to_numpy() for c in current.columns}
    for column, value in state["statistics"].items():
        if column in current.columns:
            current[column] = current[column].fillna(value)

    columns_to_impute = [c for c in current.columns if missing.get(c, np.zeros(0)).any()]
    if not columns_to_impute:
        return current

    for _ in range(state["n_iterations"]):
        for column in columns_to_impute:
            rows = missing[column]
            predictors = [c for c in current.columns if c != column]
            if not predictors:
                continue

            # The model is fitted on the TRAIN frame (complete by construction),
            # with the current completion supplying the test-side predictors.
            problem = _training_problem(state, column, predictors, design_cache)
            # Bayesian-bootstrap posterior over the empirical training
            # distribution.  Refitting with an exponential weight draw makes
            # conditional-model uncertainty vary across chains and sweeps.
            posterior_weights = rng.exponential(1.0, size=len(train))

            if column in numerical_set:
                keep = problem.keep
                if keep.sum() < 2:
                    continue
                x_query = _design(
                    current.loc[rows, predictors],
                    predictors,
                    numerical_set,
                    domains,
                    domain_indexes=design_cache["domain_indexes"],
                )
                x_query_scaled = problem.scaler.transform(x_query).tocsr()
                # ``lsqr`` is sparse-native. A tight tolerance keeps its solution
                # aligned with the former dense ``auto``/Cholesky Ridge path.
                model = Ridge(alpha=1.0, solver="lsqr", tol=1e-10).fit(
                    problem.x_scaled, problem.y[keep],
                    sample_weight=posterior_weights[keep],
                )
                fitted = model.predict(problem.x_scaled)
                residual_scale = float(np.std(problem.y[keep] - fitted))
                mean = model.predict(x_query_scaled)
                # The noise draw is what makes this a posterior sample.
                current.loc[rows, column] = mean + rng.normal(0.0, max(residual_scale, 1e-9),
                                                              size=len(mean))
            else:
                classes = problem.classes
                if len(classes) < 2:
                    current.loc[rows, column] = classes[0] if classes else ""
                    continue
                x_query = _design(
                    current.loc[rows, predictors],
                    predictors,
                    numerical_set,
                    domains,
                    domain_indexes=design_cache["domain_indexes"],
                )
                x_query_scaled = problem.scaler.transform(x_query).tocsr()
                model = LogisticRegression(max_iter=500).fit(
                    problem.x_scaled, problem.y, sample_weight=posterior_weights
                )
                proba = model.predict_proba(x_query_scaled)
                # Sample the category, never take the argmax: the argmax is the
                # conditional mode, and K copies of a mode carry no uncertainty.
                code_to_class = {int(c): classes[int(c)] for c in model.classes_}
                order = [int(c) for c in model.classes_]
                current.loc[rows, column] = [
                    code_to_class[order[int(rng.choice(len(order), p=row))]] for row in proba
                ]
    return current


def _design(frame: pd.DataFrame, columns, numerical, domains, *, domain_indexes=None):
    """CSR design matrix on a FIXED train-derived category domain.

    The domain is passed in rather than inferred per frame, so the train and
    query matrices always have the same columns in the same order. Inferring it
    from each frame would silently shift the meaning of a coefficient whenever a
    category happened to be absent from the rows being imputed.
    """
    from scipy import sparse

    numerical = set(numerical)
    if domain_indexes is None:
        domain_indexes = {
            column: {value: index for index, value in enumerate(categories)}
            for column, categories in domains.items()
        }

    parts = []
    for column in columns:
        if column in numerical:
            values = pd.to_numeric(frame[column], errors="coerce")
            fill = values.median() if values.notna().any() else 0.0
            dense_column = values.fillna(fill).to_numpy(float).reshape(-1, 1)
            parts.append(sparse.csr_matrix(dense_column))
        else:
            categories = domains.get(column, [])
            values = frame[column].astype(str).to_numpy()
            index = domain_indexes.get(column, {})
            positions = np.fromiter(
                (index.get(value, -1) for value in values),
                dtype=np.int64,
                count=len(values),
            )
            valid = positions >= 0
            rows = np.flatnonzero(valid)
            block = sparse.csr_matrix(
                (np.ones(len(rows), dtype=float), (rows, positions[valid])),
                shape=(len(frame), len(categories)),
                dtype=float,
            )
            parts.append(block)
    return (
        sparse.hstack(parts, format="csr", dtype=float)
        if parts
        else sparse.csr_matrix((len(frame), 0), dtype=float)
    )

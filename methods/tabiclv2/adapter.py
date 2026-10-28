# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""TabICL v2 plus its native unsupervised generation/imputation extension.

The extension models a train-fitted floating matrix.  This adapter keeps the
benchmark schema authoritative through the same train-only ordinal codec used
by the TabPFN extension, loads the existing local TabICL v2 checkpoints, and
never allows an evaluation value to define a category code.
"""
from __future__ import annotations

from dataclasses import replace
from importlib import metadata as importlib_metadata
from typing import Any, Sequence

import numpy as np
import pandas as pd

from otbench.methods.common.base import (
    GenerateContext,
    ImputationOutput,
    ImputeContext,
    PredictContext,
    PredictionOutput,
    Task,
)
from otbench.methods.common.foundation import FoundationMethod
from otbench.methods.common.seeding import seed_everything
from otbench.methods.common.frame_codec import _FrameCodec, _as_numpy

class TabICLv2Method(FoundationMethod):
    """Native TabICLUnsupervised G/I plus the ordinary TabICL v2 predictor."""

    name = "tabiclv2"
    tasks = frozenset({Task.GENERATION, Task.IMPUTATION, Task.PREDICTION})
    trainable = False

    def _classifier_path(self) -> str:
        return self.pretrained_path("classifier_ckpt")

    def _regressor_path(self) -> str:
        return self.pretrained_path("regressor_ckpt")

    def _estimator_kwargs(self, ctx) -> dict[str, Any]:
        kwargs = self.estimator_kwargs()
        kwargs.setdefault("n_estimators", int(self.cfg("model", "n_estimators", 8)))
        kwargs.setdefault("batch_size", self.cfg("model", "batch_size", 8))
        kwargs.setdefault("device", self.resolve_device(ctx))
        kwargs.setdefault("random_state", int(ctx.seed))
        kwargs.setdefault("verbose", False)
        return kwargs

    def make_classifier(self, ctx):
        from tabicl import TabICLClassifier

        kwargs = self._estimator_kwargs(ctx)
        kwargs.setdefault("support_many_classes", True)
        kwargs.update(
            model_path=self._classifier_path(),
            allow_auto_download=False,
        )
        return TabICLClassifier(**kwargs)

    def make_regressor(self, ctx):
        from tabicl import TabICLRegressor

        kwargs = self._estimator_kwargs(ctx)
        # Classifier-only parameter; the shared config is intentionally usable
        # for both supervised heads.
        kwargs.pop("support_many_classes", None)
        kwargs.update(
            model_path=self._regressor_path(),
            allow_auto_download=False,
        )
        return TabICLRegressor(**kwargs)

    def _new_unsupervised(self, ctx, categorical_indices: list[int]):
        from tabicl import (
            TabICLClassifier,
            TabICLRegressor,
            TabICLUnsupervised,
        )

        classifier_path = self._classifier_path()
        regressor_path = self._regressor_path()

        class _LocalCheckpointUnsupervised(TabICLUnsupervised):
            def _load_shared_model(inner_self, estimator_cls):
                kwargs = dict(inner_self._estimator_kwargs)
                if estimator_cls is TabICLClassifier:
                    kwargs["model_path"] = classifier_path
                    kwargs.setdefault("support_many_classes", True)
                elif estimator_cls is TabICLRegressor:
                    kwargs["model_path"] = regressor_path
                    kwargs.pop("support_many_classes", None)
                else:  # pragma: no cover - protects against an upstream API change
                    raise TypeError(f"Unexpected TabICL estimator: {estimator_cls}")
                kwargs["allow_auto_download"] = False
                estimator = estimator_cls(**kwargs)
                estimator._resolve_device()
                estimator._load_model()
                estimator.model_.to(estimator.device_)
                return estimator.model_

        estimator_params = self.estimator_kwargs()
        # Paths differ between the two heads and are injected by the override.
        estimator_params.pop("model_path", None)
        estimator_params.pop("allow_auto_download", None)
        # Upstream passes estimator_params to both head constructors.
        estimator_params.pop("support_many_classes", None)
        return _LocalCheckpointUnsupervised(
            n_estimators=int(self.cfg("model", "n_estimators", 8)),
            categorical_features=list(categorical_indices),
            max_categories=int(self.cfg("model", "max_categories", 10)),
            batch_size=self.cfg("model", "batch_size", 8),
            random_state=int(ctx.seed),
            device=self.resolve_device(ctx),
            estimator_params=estimator_params,
        )

    def _train_indices(self, n_rows: int, seed: int) -> np.ndarray:
        max_rows = self.cfg("model", "max_train_rows", 50000)
        if max_rows is None or n_rows <= int(max_rows):
            return np.arange(n_rows)
        rng = np.random.default_rng(int(seed))
        return np.sort(rng.choice(n_rows, size=int(max_rows), replace=False))

    def _target_access(self) -> str:
        access = str(self.cfg("impute", "target_access", "target_observed"))
        access = access.strip().lower().replace("-", "_")
        if access not in {"target_observed", "feature_only"}:
            raise ValueError(
                "impute.target_access must be 'target_observed' or 'feature_only', "
                f"got {access!r}"
            )
        return access

    # ------------------------------------------------------------------ G
    def generate(self, ctx: GenerateContext) -> pd.DataFrame:
        seed_everything(ctx.seed)
        schema = ctx.bundle.schema
        columns = list(schema.all_columns)
        train = ctx.bundle.train.reset_index(drop=True)[columns]
        codec = _FrameCodec.fit(train, columns, schema.categorical_cols)
        variable = codec.variable_columns
        n_samples = int(ctx.n_samples if ctx.n_samples is not None else len(train))
        if n_samples < 1:
            raise ValueError("generation n_samples must be >= 1")

        generated = pd.DataFrame(index=range(n_samples))
        if variable:
            rows = self._train_indices(len(train), ctx.train_seed)
            reference = codec.encode(
                train.iloc[rows], variable, preserve_missing=False
            )
            model = self._new_unsupervised(
                ctx, codec.categorical_indices(variable)
            )
            model.fit(reference)
            sampled = model.generate(
                n_samples=n_samples,
                temperature=float(self.cfg("sample", "temperature", 1.0)),
            )
            sampled = _as_numpy(sampled)
            if sampled.shape != (n_samples, len(variable)):
                raise RuntimeError(
                    "TabICLUnsupervised returned shape "
                    f"{sampled.shape}; expected {(n_samples, len(variable))}"
                )
            decoded = codec.decode(sampled, variable)
            for column in variable:
                generated[column] = decoded[column]
        for column, value in codec.constant_values.items():
            generated[column] = value
        return generated[columns]

    # ------------------------------------------------------------------ I
    def impute_draws_many(
        self, contexts: Sequence[ImputeContext]
    ) -> list[ImputationOutput]:
        contexts = list(contexts)
        if not contexts:
            return []
        first = contexts[0]
        schema = first.bundle.schema
        all_columns = list(schema.all_columns)
        target = schema.target_col
        target_access = self._target_access()
        model_columns = (
            all_columns if target_access == "target_observed"
            else list(schema.feature_cols)
        )
        train = first.bundle.train.reset_index(drop=True)[model_columns]
        codec = _FrameCodec.fit(train, model_columns, schema.categorical_cols)
        variable = codec.variable_columns
        rows = self._train_indices(len(train), first.train_seed)

        model = None
        if variable:
            reference = codec.encode(
                train.iloc[rows], variable, preserve_missing=False
            )
            model = self._new_unsupervised(
                first, codec.categorical_indices(variable)
            )
            model.fit(reference)

        outputs: list[ImputationOutput] = []
        for context_index, ctx in enumerate(contexts):
            if ctx.bundle.name != first.bundle.name:
                raise ValueError("impute_draws_many accepts one dataset at a time")
            if list(ctx.bundle.schema.all_columns) != all_columns:
                raise ValueError(f"imputation context {context_index} has a different schema")
            if ctx.train_seed != first.train_seed or ctx.device != first.device:
                raise ValueError("all imputation contexts must share train_seed and device")

            masked = ctx.masked_test.reset_index(drop=True)[all_columns].copy()
            mask = (
                ctx.mask.reset_index(drop=True)
                .reindex(columns=all_columns, fill_value=False)
                .fillna(False)
                .astype(bool)
            )
            if len(masked) != len(ctx.bundle.test) or len(mask) != len(masked):
                raise ValueError(f"imputation context {context_index} has misaligned rows")
            if bool(mask[target].any()):
                raise ValueError("TabICL feature imputation must not mask the target")
            masked_cells = mask.to_numpy(dtype=bool)
            if not pd.isna(masked.to_numpy())[masked_cells].all():
                raise ValueError("mask=True cells must be NaN in masked_test")

            model_mask = mask.reindex(columns=model_columns, fill_value=False)
            variable_mask = model_mask.reindex(
                columns=variable, fill_value=False
            ).to_numpy(dtype=bool)
            base = codec.encode(
                masked[model_columns], variable, preserve_missing=False
            )
            base[variable_mask] = np.nan
            n_draws = max(1, int(ctx.trials))
            draws: list[pd.DataFrame] = []
            for draw_index in range(n_draws):
                completed = masked.copy()
                for column, value in codec.constant_values.items():
                    rows_to_fill = model_mask[column].to_numpy(dtype=bool)
                    if rows_to_fill.any():
                        completed.loc[rows_to_fill, column] = value
                if model is not None and variable_mask.any():
                    draw_seed = int(
                        np.random.SeedSequence(
                            [int(ctx.seed), int(draw_index)]
                        ).generate_state(1, dtype=np.uint32)[0]
                    )
                    model.random_state = draw_seed
                    sampled = model.impute(
                        base.copy(),
                        temperature=float(
                            self.cfg("impute", "temperature", 1.0)
                        ),
                        n_iterations=max(
                            1, int(self.cfg("impute", "iterations", 2))
                        ),
                    )
                    decoded = codec.decode(sampled, variable)
                    for column in variable:
                        rows_to_fill = model_mask[column].to_numpy(dtype=bool)
                        if rows_to_fill.any():
                            completed.loc[rows_to_fill, column] = decoded.loc[
                                rows_to_fill, column
                            ]
                if pd.isna(completed.to_numpy())[masked_cells].any():
                    raise RuntimeError(
                        "TabICLUnsupervised left an artificially masked cell empty"
                    )
                draws.append(completed[all_columns])

            native_missing = int(
                (
                    masked[model_columns].isna().to_numpy()
                    & ~model_mask.to_numpy(dtype=bool)
                ).sum()
            )
            outputs.append(
                ImputationOutput.from_frames(
                    draws,
                    metadata={
                        "source": "native",
                        "backend": "tabicl.TabICLUnsupervised.impute",
                        "draw_strategy": "native iterative conditional sampling",
                        "n_draws": len(draws),
                        "target_access": target_access,
                        "imputation_scope": "artificial_mask_only",
                        "native_missing_output_policy": "preserve",
                        "native_missing_conditioning_policy": "train_fitted_fill",
                        "native_missing_conditioning_cells": native_missing,
                        "model_fit_shared_across_settings": len(contexts),
                    },
                )
            )
        return outputs

    def impute_draws(self, ctx: ImputeContext) -> ImputationOutput:
        return self.impute_draws_many([ctx])[0]

    def impute(self, ctx: ImputeContext) -> pd.DataFrame:
        return self.impute_draws(replace(ctx, trials=1)).draws[0]

    # ------------------------------------------------------------------ P
    def predict(self, ctx: PredictContext) -> PredictionOutput:
        output = super().predict(ctx)
        output.metadata.update(
            {
                "adapter": self.name,
                "prediction_route": "native TabICL v2 classifier/regressor",
            }
        )
        return output

    def evaluation_metadata(self, ctx) -> dict[str, Any]:
        try:
            version = importlib_metadata.version("tabicl")
        except importlib_metadata.PackageNotFoundError:
            version = "missing"
        return {
            "variant": "TabICL v2 + TabICLUnsupervised",
            "tabicl_version": version,
            "generation_route": "TabICLUnsupervised.generate",
            "imputation_route": "TabICLUnsupervised.impute",
            "imputation_target_access": self._target_access(),
            "prediction_route": "native TabICLClassifier/TabICLRegressor",
        }


__all__ = ["TabICLv2Method"]

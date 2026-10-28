# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""TabPFN-3 G/I/P adapter using ``tabpfn-extensions`` for G and I.

Generation uses the upstream experimental sequential conditional sampler.
Imputation uses the same upstream conditional-density primitive, but schedules a
whole mask grid column-major: each artificial missing column is fit exactly once
and that fitted conditional is reused for every setting and draw. Prediction uses
the native TabPFN classifier/regressor on the same V3 weights; this preserves
class probabilities and the standard supervised TabPFN semantics instead of
reducing prediction to one stochastic target draw.

The upstream unsupervised implementation accepts only a floating tensor. This
adapter therefore fits an ordinal codec on *train only*, restores categorical
values after sampling, and never lets an evaluation/test value create a code.
"""
from __future__ import annotations

from dataclasses import replace
from importlib import metadata as importlib_metadata
from typing import Any, Iterable, Sequence

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
from otbench.methods.common.capabilities import UnsupportedRun
from otbench.methods.common.foundation import FoundationMethod
from otbench.methods.common.seeding import seed_everything
from otbench.methods.common.frame_codec import _FrameCodec, _as_numpy

class TabPFNUnsupervisedMethod(FoundationMethod):
    """TabPFN-3 plus the upstream unsupervised G/I extension."""

    name = "tabpfn_unsupervised"
    tasks = frozenset({Task.GENERATION, Task.IMPUTATION, Task.PREDICTION})
    trainable = False
    max_imputation_classes = 160
    max_prediction_classes = 160

    def _estimator_kwargs(self, ctx) -> dict[str, Any]:
        kwargs = self.estimator_kwargs()
        kwargs.setdefault("device", self.resolve_device(ctx))
        kwargs.setdefault("random_state", int(ctx.seed))
        kwargs.setdefault("show_progress_bar", False)
        return kwargs

    def make_regressor(self, ctx):
        from tabpfn import TabPFNRegressor

        path = self.pretrained_path("regressor_ckpt")
        return TabPFNRegressor(model_path=path, **self._estimator_kwargs(ctx))

    def make_classifier(self, ctx):
        from tabpfn import TabPFNClassifier

        path = self.pretrained_path("classifier_ckpt")
        return TabPFNClassifier(model_path=path, **self._estimator_kwargs(ctx))

    def _new_unsupervised(self, ctx):
        from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel

        return TabPFNUnsupervisedModel(
            tabpfn_clf=self.make_classifier(ctx),
            tabpfn_reg=self.make_regressor(ctx),
        )

    def _fit_unsupervised(
        self,
        ctx,
        train_matrix: np.ndarray,
        categorical_indices: list[int],
    ):
        model = self._new_unsupervised(ctx)
        model.categorical_features = list(categorical_indices)
        model.fit(train_matrix)
        # Upstream also applies a low-cardinality heuristic. Restore the schema
        # declaration after fit so a low-cardinality numerical column is not
        # silently changed into a classification target.
        model.categorical_features = list(categorical_indices)
        return model

    def _train_indices(self, n_rows: int, seed: int) -> np.ndarray:
        max_rows = self.cfg("model", "max_train_rows", 50000)
        if max_rows is None or n_rows <= int(max_rows):
            return np.arange(n_rows)
        rng = np.random.default_rng(int(seed))
        return np.sort(rng.choice(n_rows, size=int(max_rows), replace=False))

    def _check_categorical_targets(
        self,
        codec: _FrameCodec,
        columns: Iterable[str],
        *,
        task: str,
    ) -> None:
        limit = int(self.cfg("model", "max_classes", self.max_imputation_classes))
        for column in columns:
            if column not in codec.categorical or column in codec.constant_values:
                continue
            n_classes = codec.cardinality(column)
            if n_classes > limit:
                raise UnsupportedRun(
                    f"{self.name} {task} needs to model categorical column {column!r} "
                    f"as a target, but TabPFN-3 supports at most {limit} classes "
                    f"and the train split has {n_classes}. The upstream extension "
                    "would route this column through a regressor, which this adapter "
                    "does not treat as valid categorical generation/imputation."
                )

    # ------------------------------------------------------------------ G
    def generate(self, ctx: GenerateContext) -> pd.DataFrame:
        seed_everything(ctx.seed)
        schema = ctx.bundle.schema
        columns = list(schema.all_columns)
        full_train = ctx.bundle.train.reset_index(drop=True)[columns]
        codec = _FrameCodec.fit(full_train, columns, schema.categorical_cols)
        variable = codec.variable_columns
        self._check_categorical_targets(codec, variable, task="generation")

        n_samples = int(
            ctx.n_samples if ctx.n_samples is not None else len(full_train)
        )
        if n_samples < 1:
            raise ValueError("generation n_samples must be >= 1")
        generated = pd.DataFrame(index=range(n_samples))
        if variable:
            rows = self._train_indices(len(full_train), ctx.train_seed)
            reference = codec.encode(
                full_train.iloc[rows], variable, preserve_missing=False
            )
            model = self._fit_unsupervised(
                ctx, reference, codec.categorical_indices(variable)
            )
            sampled = model.generate_synthetic_data(
                n_samples=n_samples,
                t=float(self.cfg("sample", "temperature", 1.0)),
                n_permutations=max(
                    1, int(self.cfg("sample", "n_permutations", 1))
                ),
            )
            sampled_array = _as_numpy(sampled)
            if sampled_array.shape[0] != n_samples:
                raise RuntimeError(
                    f"upstream extension returned {sampled_array.shape[0]} rows; "
                    f"requested {n_samples} (ensure FAST_TEST_MODE is unset)"
                )
            decoded = codec.decode(sampled_array, variable)
            for column in variable:
                generated[column] = decoded[column]
        for column, value in codec.constant_values.items():
            generated[column] = value
        return generated[columns]

    # ------------------------------------------------------------------ I
    def _imputation_target_access(self) -> str:
        access = str(
            self.cfg("impute", "target_access", "target_observed")
        ).strip().lower().replace("-", "_")
        if access not in {"target_observed", "feature_only"}:
            raise ValueError(
                "impute.target_access must be 'target_observed' or 'feature_only', "
                f"got {access!r}"
            )
        return access

    def _configure_imputation_estimators(self, model: Any) -> None:
        """Cache each train context during its one conditional fit.

        The extension's ``density_`` calls sklearn-style ``fit`` immediately
        before prediction.  ``fit_with_cache`` makes subsequent prediction
        batches reuse that fitted train representation; the cache is replaced
        when the same classifier/regressor object is fit for the next column.
        """

        fit_mode = str(self.cfg("impute", "fit_mode", "fit_with_cache"))
        keep_on_device = bool(
            self.cfg("impute", "keep_cache_on_device", False)
        )
        for name in ("tabpfn_clf", "tabpfn_reg"):
            estimator = getattr(model, name, None)
            if estimator is None:
                continue
            updates = {
                "fit_mode": fit_mode,
                "keep_cache_on_device": keep_on_device,
            }
            setter = getattr(estimator, "set_params", None)
            if callable(setter):
                try:
                    setter(**updates)
                    continue
                except (TypeError, ValueError):
                    pass
            for key, value in updates.items():
                if hasattr(estimator, key):
                    setattr(estimator, key, value)

    def _predict_conditional_batches(
        self,
        fitted_model: Any,
        X_predict: Any,
        *,
        use_classifier: bool,
    ) -> tuple[Any, Any | None]:
        """Predict one fitted conditional, batching queries but never refitting."""

        import torch

        array = _as_numpy(X_predict)
        configured = int(self.cfg("impute", "predict_batch_size", 4096))
        batch_size = len(array) if configured <= 0 else configured
        batch_size = max(1, batch_size)
        payloads: list[Any] = []
        criterion = None
        for start in range(0, len(array), batch_size):
            batch = array[start : start + batch_size]
            if use_classifier:
                payloads.append(
                    torch.as_tensor(
                        fitted_model.predict_proba(batch), dtype=torch.float32
                    )
                )
            else:
                prediction = fitted_model.predict(batch, output_type="full")
                logits = prediction["logits"]
                payloads.append(
                    logits.detach().cpu()
                    if torch.is_tensor(logits)
                    else torch.as_tensor(logits)
                )
                if criterion is None:
                    criterion = prediction["criterion"]
        return torch.cat(payloads, dim=0), criterion

    @staticmethod
    def _column_draw_seed(seed: int, draw_index: int, column_index: int) -> int:
        return int(
            np.random.SeedSequence(
                [int(seed), int(draw_index), int(column_index)]
            ).generate_state(1, dtype=np.uint32)[0]
        )

    def impute_draws_many(
        self, contexts: Sequence[ImputeContext]
    ) -> list[ImputationOutput]:
        """Impute a dataset's setting grid with one fit per masked column.

        Only cells marked ``True`` in each context's explicit mask are targets.
        Native NaNs are filled with train-fitted codec values *only inside the
        model input* and are left untouched in every returned table.
        """

        import torch

        contexts = list(contexts)
        if not contexts:
            return []
        first = contexts[0]
        schema = first.bundle.schema
        all_columns = list(schema.all_columns)
        target = schema.target_col
        target_access = self._imputation_target_access()
        model_columns = (
            all_columns if target_access == "target_observed"
            else list(schema.feature_cols)
        )
        train = first.bundle.train.reset_index(drop=True)[model_columns]
        codec = _FrameCodec.fit(train, model_columns, schema.categorical_cols)
        variable = codec.variable_columns
        variable_index = {column: index for index, column in enumerate(variable)}

        if int(self.cfg("impute", "n_permutations", 1)) != 1:
            raise ValueError(
                "grid-cached TabPFN imputation requires impute.n_permutations=1; "
                "multiple permutations would change the sequential conditioning "
                "state and require repeated conditional fits"
            )

        full_masks: list[pd.DataFrame] = []
        model_masks: list[pd.DataFrame] = []
        masked_frames: list[pd.DataFrame] = []
        for index, ctx in enumerate(contexts):
            if list(ctx.bundle.schema.all_columns) != all_columns:
                raise ValueError(
                    f"imputation context {index} has a different schema"
                )
            if ctx.bundle.name != first.bundle.name:
                raise ValueError(
                    "impute_draws_many only accepts settings from one dataset"
                )
            if ctx.train_seed != first.train_seed or ctx.device != first.device:
                raise ValueError(
                    "all imputation contexts must share train_seed and device"
                )
            if len(ctx.masked_test) != len(ctx.bundle.test) or len(ctx.mask) != len(
                ctx.bundle.test
            ):
                raise ValueError(f"imputation context {index} has misaligned rows")
            if not set(all_columns).issubset(ctx.masked_test.columns):
                raise ValueError(
                    f"imputation context {index} is missing table columns"
                )

            masked = ctx.masked_test.reset_index(drop=True)[all_columns].copy()
            mask = (
                ctx.mask.reset_index(drop=True)
                .reindex(columns=all_columns, fill_value=False)
                .fillna(False)
                .astype(bool)
            )
            if bool(mask[target].any()):
                raise ValueError(
                    "TabPFN feature imputation must not mask the supervised target"
                )
            masked_cells = mask.to_numpy(dtype=bool)
            if not pd.isna(masked.to_numpy())[masked_cells].all():
                raise ValueError(
                    f"imputation context {index} has mask=True cells that are not NaN"
                )
            full_masks.append(mask)
            model_masks.append(mask.reindex(columns=model_columns, fill_value=False))
            masked_frames.append(masked)

        if target_access == "target_observed" and target in codec.categorical:
            known = codec.category_to_code[target]
            for index, frame in enumerate(masked_frames):
                unknown = [
                    value
                    for value in frame[target].dropna().drop_duplicates().tolist()
                    if value not in known
                ]
                if unknown:
                    raise UnsupportedRun(
                        f"{self.name} target-observed imputation cannot encode "
                        f"{len(unknown)} unseen target level(s) in context {index}"
                    )

        missing_columns = [
            column
            for column in model_columns
            if any(bool(mask[column].any()) for mask in model_masks)
        ]
        self._check_categorical_targets(codec, missing_columns, task="imputation")
        variable_missing = [c for c in variable if c in missing_columns]

        # Encode native NaNs with train-only fills for conditioning, then restore
        # NaN exclusively at artificial mask cells. Thus native missingness never
        # creates an extra imputation target or conditional fit.
        base_working: list[np.ndarray] = []
        native_missing_counts: list[int] = []
        for frame, mask in zip(masked_frames, model_masks):
            encoded = codec.encode(
                frame[model_columns], variable, preserve_missing=False
            )
            variable_mask = mask.reindex(columns=variable, fill_value=False).to_numpy(
                dtype=bool
            )
            encoded[variable_mask] = np.nan
            base_working.append(encoded)
            native_missing_counts.append(
                int(
                    (
                        frame[model_columns].isna().to_numpy()
                        & ~mask.to_numpy(dtype=bool)
                    ).sum()
                )
            )

        working = [
            [base.copy() for _ in range(max(1, int(ctx.trials)))]
            for base, ctx in zip(base_working, contexts)
        ]
        completions = [
            [frame.copy() for _ in range(max(1, int(ctx.trials)))]
            for frame, ctx in zip(masked_frames, contexts)
        ]
        for context_index, mask in enumerate(model_masks):
            for column, value in codec.constant_values.items():
                rows_to_fill = mask[column].to_numpy(dtype=bool)
                if not rows_to_fill.any():
                    continue
                for completed in completions[context_index]:
                    completed.loc[rows_to_fill, column] = value

        conditional_fit_columns: list[str] = []
        if variable_missing:
            rows = self._train_indices(len(train), first.train_seed)
            reference_array = codec.encode(
                train.iloc[rows], variable, preserve_missing=False
            )
            reference = torch.as_tensor(reference_array, dtype=torch.float32)
            unsupervised = self._fit_unsupervised(
                first, reference_array, codec.categorical_indices(variable)
            )
            self._configure_imputation_estimators(unsupervised)
            temperature = float(self.cfg("impute", "temperature", 1e-9))

            # A common schema order is the single declared imputation order for
            # every setting and draw. All rows are concatenated for prediction,
            # while sampling is reseeded per setting/draw for stable results.
            for column in variable_missing:
                column_index = variable_index[column]
                query_blocks: list[np.ndarray] = []
                segments: list[tuple[int, int, np.ndarray, int, int]] = []
                cursor = 0
                for context_index, (ctx, mask) in enumerate(
                    zip(contexts, model_masks)
                ):
                    masked_rows = np.flatnonzero(
                        mask[column].to_numpy(dtype=bool)
                    )
                    if len(masked_rows) == 0:
                        continue
                    for draw_index in range(max(1, int(ctx.trials))):
                        block = working[context_index][draw_index][masked_rows]
                        query_blocks.append(block)
                        end = cursor + len(block)
                        segments.append(
                            (
                                context_index,
                                draw_index,
                                masked_rows,
                                cursor,
                                end,
                            )
                        )
                        cursor = end
                if not query_blocks:
                    continue

                query = torch.as_tensor(
                    np.concatenate(query_blocks, axis=0), dtype=torch.float32
                )
                conditional_indices = [
                    index for index in range(len(variable)) if index != column_index
                ]
                fitted_model, X_predict, _ = unsupervised.density_(
                    query, reference, conditional_indices, column_index
                )
                use_classifier = unsupervised.use_classifier_(
                    column_index, reference[:, column_index]
                )
                prediction, criterion = self._predict_conditional_batches(
                    fitted_model,
                    X_predict,
                    use_classifier=use_classifier,
                )
                conditional_fit_columns.append(column)

                for context_index, draw_index, masked_rows, start, end in segments:
                    seed_everything(
                        self._column_draw_seed(
                            contexts[context_index].seed,
                            draw_index,
                            column_index,
                        )
                    )
                    if use_classifier:
                        sampled = torch.distributions.Categorical(
                            probs=prediction[start:end]
                        ).sample().float()
                    else:
                        if criterion is None:
                            raise RuntimeError(
                                "TabPFN regressor returned no sampling criterion"
                            )
                        sampled = criterion.sample(
                            prediction[start:end], t=temperature
                        )
                    sampled_array = _as_numpy(sampled).reshape(-1)
                    if len(sampled_array) != len(masked_rows):
                        raise RuntimeError(
                            f"TabPFN returned {len(sampled_array)} values for "
                            f"{len(masked_rows)} masked cells in {column!r}"
                        )
                    working[context_index][draw_index][
                        masked_rows, column_index
                    ] = sampled_array

        outputs: list[ImputationOutput] = []
        for context_index, ctx in enumerate(contexts):
            draws: list[pd.DataFrame] = []
            for draw_index, completed in enumerate(completions[context_index]):
                if variable:
                    decoded = codec.decode(
                        working[context_index][draw_index], variable
                    )
                    for column in variable_missing:
                        rows_to_fill = model_masks[context_index][column].to_numpy(
                            dtype=bool
                        )
                        if rows_to_fill.any():
                            completed.loc[rows_to_fill, column] = decoded.loc[
                                rows_to_fill, column
                            ]
                artificially_missing = full_masks[context_index].to_numpy(
                    dtype=bool
                )
                if pd.isna(completed.to_numpy())[artificially_missing].any():
                    raise RuntimeError(
                        "TabPFN unsupervised left an artificially masked cell empty"
                    )
                draws.append(completed[all_columns])

            outputs.append(
                ImputationOutput.from_frames(
                    draws,
                    metadata={
                        "source": "native",
                        "backend": (
                            "tabpfn_extensions.unsupervised."
                            "TabPFNUnsupervisedModel.density_"
                        ),
                        "draw_strategy": (
                            "column-major conditional sampling; one conditional "
                            "fit per artificially masked variable column"
                        ),
                        "n_draws": len(draws),
                        "target_access": target_access,
                        "imputation_scope": "artificial_mask_only",
                        "native_missing_output_policy": "preserve",
                        "native_missing_conditioning_policy": "train_fitted_fill",
                        "conditional_fit_columns": conditional_fit_columns,
                        "n_conditional_fits": len(conditional_fit_columns),
                        "fits_shared_across_settings": len(contexts),
                        "conditional_order": variable_missing,
                        "native_missing_conditioning_cells": (
                            native_missing_counts[context_index]
                        ),
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
        from otbench.methods.common.resources import pretrained_resources, resolve_resource
        output.metadata.update({
            "adapter": self.name,
            "prediction_route": "native TabPFN-3 classifier/regressor",
            "checkpoints": {resource.key: str(resolve_resource(
                self.name, self.config, resource, self.checkpoints_root, require_exists=False,
            )) for resource in pretrained_resources(self.name)},
        })
        return output

    def evaluation_metadata(self, ctx) -> dict[str, Any]:
        def version(package: str) -> str:
            try:
                return importlib_metadata.version(package)
            except importlib_metadata.PackageNotFoundError:
                return "missing"

        return {
            "variant": "TabPFN-3 + tabpfn-extensions unsupervised",
            "tabpfn_version": version("tabpfn"),
            "tabpfn_extensions_version": version("tabpfn-extensions"),
            "generation_route": "TabPFNUnsupervisedModel.generate_synthetic_data",
            "imputation_route": (
                "column-major TabPFNUnsupervisedModel.density_; one fit per "
                "artificially masked column"
            ),
            "imputation_target_access": self._imputation_target_access(),
            "prediction_route": "native TabPFNClassifier/TabPFNRegressor",
            "categorical_target_limit": int(
                self.cfg("model", "max_classes", self.max_prediction_classes)
            ),
        }




__all__ = ["TabPFNUnsupervisedMethod"]

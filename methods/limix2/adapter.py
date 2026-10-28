# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""LimiX-2 generation, native feature-imputation, and prediction adapter.

Generation follows the same chain-rule construction as the unsupervised
TabPFN/TabICL adapters.  A categorical target with 2--10 train levels uses the
public classification interface; a target with more than ten levels is encoded
ordinally and sampled from LimiX-2's native bucket-regression distribution.
That high-cardinality route is deliberate and auditable: it is never converted
to a structured skip.

Imputation uses the public ``Feature_imputation`` task and copies back only the
explicitly masked cells.  Native NaNs therefore remain native NaNs in outputs.

Prediction uses the public classification/regression entry points.  Following
the benchmark policy for this method, classification targets with more than
ten train levels are ordinally represented and sent through native regression
instead of being skipped; that route intentionally exposes point labels only.
"""
from __future__ import annotations

from dataclasses import replace
from importlib import import_module, invalidate_caches, metadata as importlib_metadata
import os
from pathlib import Path
import sys
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

_ENV_SOURCE = "LIMIX2_SOURCE_ROOT"
_CONFIG_DEFAULTS = {
    "classification": "config/cls_default_noretrieval_v2.json",
    "regression": "config/reg_default_noretrieval_v2.json",
    "imputation": "config/reg_default_noretrieval_MVI_v2.json",
}
_HALF_NORMAL_MEDIAN = 0.6744897501960817


class LimiX2Method(FoundationMethod):
    """Experimental chain-rule generator plus native LimiX-2 MVI/prediction."""

    name = "limix2"
    tasks = frozenset({Task.GENERATION, Task.IMPUTATION, Task.PREDICTION})
    trainable = False

    def _source_root(self) -> Path:
        configured = os.environ.get(_ENV_SOURCE) or self.cfg(
            "model", "source_root", "model"
        )
        path = Path(str(configured)).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        return path.resolve()

    def _model_path(self) -> str:
        return self.pretrained_path("model_ckpt")

    def _config_path(self, kind: str) -> str:
        if kind not in _CONFIG_DEFAULTS:
            raise ValueError(f"unknown LimiX-2 inference config kind: {kind}")
        configured = self.cfg(
            "model", f"{kind}_config", _CONFIG_DEFAULTS[kind]
        )
        path = Path(str(configured)).expanduser()
        if not path.is_absolute():
            path = self._source_root() / path
        if not path.is_file():
            raise FileNotFoundError(
                f"LimiX-2 {kind} config not found: {path}. Extract the current "
                "official LimiX repository under methods/limix2/model/."
            )
        return str(path)

    @staticmethod
    def _module_is_under(module: Any, root: Path) -> bool:
        filename = getattr(module, "__file__", None)
        if not filename:
            return False
        try:
            Path(filename).resolve().relative_to(root)
            return True
        except (OSError, ValueError):
            return False

    def _predictor_class(self):
        """Import the official v2 router without reusing a vendored v1 module."""

        root = self._source_root()
        local_source = (root / "inference" / "predictor.py").is_file()
        if local_source:
            conflicts = []
            for top_level in ("limix", "inference", "model", "utils", "config"):
                for name, loaded in tuple(sys.modules.items()):
                    if (name == top_level or name.startswith(f"{top_level}.")) and loaded is not None and not self._module_is_under(loaded, root):
                        conflicts.append(name)
            if conflicts:
                raise ImportError(
                    "LimiX-2's upstream top-level imports conflict with modules already "
                    f"loaded in this interpreter: {', '.join(sorted(conflicts))}. "
                    "Run LimiX-2 in a separate process/environment; existing modules "
                    "have been left unchanged."
                )
            if str(root) in sys.path:
                sys.path.remove(str(root))
            sys.path.insert(0, str(root))
            invalidate_caches()
        try:
            return import_module("limix").LimiXPredictor
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "The current official LimiX-2 source is required. Extract it to "
                f"{root} and run `python -m pip install -e {root}` in the isolated "
                "environment."
            ) from exc

    def additional_resource_identities(self) -> list[dict]:
        from otbench.methods.common.resources import resource_identity, source_tree_identity
        root = self._source_root()
        return [{"key": "inference_source", "path": str(root), "files": source_tree_identity(root)}] + [
            {"key": f"{kind}_config", **resource_identity(self._config_path(kind))}
            for kind in _CONFIG_DEFAULTS
        ]

    def _predictor_for(
        self,
        ctx,
        kind: str,
        seed: int,
        *,
        temperature: float | None = None,
    ):
        config_path = self._config_path(kind)
        temperature_key = (
            "classification_temperature"
            if kind == "classification"
            else "regression_temperature"
        )
        if temperature is None:
            temperature = float(self.cfg("sample", temperature_key, 1.0))
        else:
            temperature = float(temperature)
        if temperature <= 0:
            raise ValueError(f"{temperature_key} must be > 0")

        predictor = getattr(self, "_predictor_cache", None)
        if predictor is None:
            import torch

            kwargs = dict(
                self.config.get("model", {}).get("predictor_kwargs", {})
            )
            for reserved in (
                "device", "model_path", "inference_config",
                "softmax_temperature", "seed",
            ):
                kwargs.pop(reserved, None)
            predictor = self._predictor_class()(
                device=torch.device(self.resolve_device(ctx)),
                model_path=self._model_path(),
                inference_config=config_path,
                softmax_temperature=temperature,
                seed=int(seed),
                **kwargs,
            )
            self._predictor_cache = predictor
        else:
            predictor.set_inference_config(
                config_path,
                softmax_temperature=temperature,
                seed=int(seed),
            )
        self._predictor_kind = kind
        return predictor

    def _train_indices(self, n_rows: int, seed: int) -> np.ndarray:
        max_rows = self.cfg("model", "max_train_rows", 50000)
        if max_rows is None or n_rows <= int(max_rows):
            return np.arange(n_rows)
        rng = np.random.default_rng(int(seed))
        return np.sort(rng.choice(n_rows, size=int(max_rows), replace=False))

    @staticmethod
    def _draw_discrete(probabilities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.ndim != 2:
            raise RuntimeError(
                f"expected a probability matrix, got shape {probabilities.shape}"
            )
        probabilities = np.nan_to_num(
            probabilities, nan=0.0, posinf=0.0, neginf=0.0
        )
        probabilities = np.clip(probabilities, 0.0, None)
        totals = probabilities.sum(axis=1, keepdims=True)
        if np.any(totals <= 0):
            raise RuntimeError("LimiX-2 returned a zero-mass probability row")
        probabilities = probabilities / totals
        cumulative = np.cumsum(probabilities, axis=1)
        cumulative[:, -1] = 1.0
        uniforms = rng.random(len(probabilities))
        return (uniforms[:, None] > cumulative).sum(axis=1).astype(np.int64)

    def _sample_classification(
        self,
        predictor,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_query: np.ndarray,
        rng: np.random.Generator,
        *,
        dataset_name: str,
    ) -> np.ndarray:
        output = predictor.predict(
            x_train,
            np.asarray(y_train, dtype=np.int64),
            x_query,
            task_type="Classification",
            unique_dataset_name=dataset_name,
        )
        if isinstance(output, tuple):
            output = output[0]
        probabilities = _as_numpy(output)
        n_classes = len(np.unique(y_train))
        if probabilities.shape != (len(x_query), n_classes):
            raise RuntimeError(
                "LimiX-2 classification returned shape "
                f"{probabilities.shape}; expected {(len(x_query), n_classes)}"
            )
        return self._draw_discrete(probabilities, rng).astype(np.float32)

    def _sample_regression(
        self,
        predictor,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_query: np.ndarray,
        rng: np.random.Generator,
        *,
        dataset_name: str,
    ) -> np.ndarray:
        """Sample the current LimiX-2 bucket decoder instead of taking its mean.

        The public regression call exposes only the bucket expectation.  The v2
        predictor itself computes that expectation from per-member bucket mass;
        this method uses the same current private collection/rebinning path, then
        samples one bucket and its piecewise-uniform/half-normal density.
        """

        import torch

        required = ("_collect_reg_member_outputs", "model")
        missing = [name for name in required if not hasattr(predictor, name)]
        if missing:
            raise RuntimeError(
                "Installed LimiX-2 no longer exposes the bucket sampler inputs: "
                + ", ".join(missing)
            )
        ensure_loaded = getattr(predictor, "_ensure_model_loaded", None)
        if callable(ensure_loaded):
            ensure_loaded()
        if getattr(predictor, "model", None) is None:
            raise RuntimeError("LimiX-2 did not load its regression model")
        collected = predictor._collect_reg_member_outputs(
            x_train,
            np.asarray(y_train, dtype=np.float64),
            x_query,
            "Regression",
            unique_dataset_name=dataset_name,
        )
        members = [
            output
            for _, output in sorted(
                collected["members"], key=lambda item: item[0]
            )
        ]
        if not members:
            raise RuntimeError("LimiX-2 returned no regression ensemble members")
        borders = predictor.model._reg_borders
        translate = import_module(
            "inference.v2_0.inference_utils"
        ).translate_probs_across_borders
        member_probabilities = []
        temperature = float(getattr(predictor, "softmax_temperature", 1.0))
        if temperature <= 0:
            raise RuntimeError("LimiX-2 softmax_temperature must be > 0")
        for member in members:
            try:
                logits = member[0].squeeze(0).float() / temperature
            except (IndexError, AttributeError, TypeError) as exc:
                raise RuntimeError(
                    "Unexpected LimiX-2 regression member output; the private "
                    "bucket API may have changed"
                ) from exc
            probabilities = translate(
                logits.squeeze(0),
                frm=borders.to(logits.device),
                to=borders.to(logits.device),
            )
            member_probabilities.append(probabilities.float().cpu())
        probabilities = torch.stack(member_probabilities, dim=0).mean(dim=0)
        probabilities = _as_numpy(probabilities)
        if probabilities.shape[0] != len(x_query):
            raise RuntimeError(
                "LimiX-2 bucket probabilities have shape "
                f"{probabilities.shape}; expected first dimension {len(x_query)}"
            )
        bucket_index = self._draw_discrete(probabilities, rng)
        border_values = _as_numpy(borders).astype(np.float64).reshape(-1)
        n_buckets = len(border_values) - 1
        if probabilities.shape[1] != n_buckets or n_buckets < 2:
            raise RuntimeError("LimiX-2 checkpoint has an invalid regression border grid")

        sampled = np.empty(len(bucket_index), dtype=np.float64)
        for index, bucket in enumerate(bucket_index):
            if bucket == 0:
                width = float(border_values[1] - border_values[0])
                sampled[index] = border_values[1] - abs(
                    rng.normal(0.0, max(width, 1e-8) / _HALF_NORMAL_MEDIAN)
                )
            elif bucket == n_buckets - 1:
                width = float(border_values[-1] - border_values[-2])
                sampled[index] = border_values[-2] + abs(
                    rng.normal(0.0, max(width, 1e-8) / _HALF_NORMAL_MEDIAN)
                )
            else:
                sampled[index] = rng.uniform(
                    border_values[bucket], border_values[bucket + 1]
                )
        y_mean = float(getattr(predictor, "y_mean"))
        y_std = float(getattr(predictor, "y_std"))
        return (sampled * y_std + y_mean).astype(np.float32)

    def _generation_order(
        self, columns: list[str], seed: int
    ) -> list[str]:
        policy = str(self.cfg("sample", "column_order", "seeded")).lower()
        if policy == "schema":
            return list(columns)
        if policy != "seeded":
            raise ValueError("sample.column_order must be 'seeded' or 'schema'")
        rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), 0x4C494D49])
        )
        return [columns[index] for index in rng.permutation(len(columns))]

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
        routes: dict[str, str] = {}
        order = self._generation_order(variable, ctx.seed)
        if variable:
            rows = self._train_indices(len(train), ctx.train_seed)
            reference = codec.encode(
                train.iloc[rows], variable, preserve_missing=False
            )
            variable_index = {column: index for index, column in enumerate(variable)}
            generated_matrix = np.empty(
                (n_samples, len(variable)), dtype=np.float32
            )
            rng = np.random.default_rng(
                np.random.SeedSequence([int(ctx.seed), 0x53414D50])
            )
            preceding: list[str] = []
            for step, column in enumerate(order):
                target_index = variable_index[column]
                if preceding:
                    conditional_indices = [variable_index[c] for c in preceding]
                    x_train = reference[:, conditional_indices]
                    x_query = generated_matrix[:, conditional_indices]
                else:
                    # LimiX requires at least one input feature. Independent
                    # Gaussian noise makes the first conditional a learned
                    # marginal without leaking another table column.
                    x_train = rng.standard_normal((len(reference), 1)).astype(
                        np.float32
                    )
                    x_query = rng.standard_normal((n_samples, 1)).astype(
                        np.float32
                    )
                y_train = reference[:, target_index]
                dataset_name = f"{ctx.bundle.name}_generation_{step}_{column}"
                if column in codec.categorical and codec.cardinality(column) <= 10:
                    predictor = self._predictor_for(
                        ctx, "classification", ctx.seed + step
                    )
                    sampled = self._sample_classification(
                        predictor,
                        x_train,
                        y_train,
                        x_query,
                        rng,
                        dataset_name=dataset_name,
                    )
                    routes[column] = "categorical_classification_2_to_10"
                else:
                    predictor = self._predictor_for(
                        ctx, "regression", ctx.seed + step
                    )
                    sampled = self._sample_regression(
                        predictor,
                        x_train,
                        y_train,
                        x_query,
                        rng,
                        dataset_name=dataset_name,
                    )
                    routes[column] = (
                        "categorical_as_numerical_regression"
                        if column in codec.categorical
                        else "numerical_regression"
                    )
                generated_matrix[:, target_index] = np.asarray(
                    sampled, dtype=np.float32
                ).reshape(-1)
                preceding.append(column)
            decoded = codec.decode(generated_matrix, variable)
            for column in variable:
                generated[column] = decoded[column]
        for column, value in codec.constant_values.items():
            generated[column] = value
            routes[column] = "constant"
        self._last_generation_audit = {
            "conditional_order": order,
            "column_routes": routes,
            "categorical_classification_limit": 10,
            "high_cardinality_policy": "numerical_regression_then_round_clip_decode",
        }
        return generated[columns]

    # ------------------------------------------------------------------ I
    def _target_access(self) -> str:
        access = str(self.cfg("impute", "target_access", "feature_only"))
        access = access.strip().lower().replace("-", "_")
        if access not in {"feature_only", "target_observed"}:
            raise ValueError(
                "impute.target_access must be 'feature_only' or 'target_observed', "
                f"got {access!r}"
            )
        return access

    @staticmethod
    def _query_imputation_rows(
        prediction: Any,
        *,
        n_train: int,
        n_query: int,
        n_columns: int,
    ) -> np.ndarray:
        values = _as_numpy(prediction)
        if values.ndim == 3 and values.shape[0] == 1:
            values = values[0]
        if values.ndim != 2 or values.shape[1] != n_columns:
            raise RuntimeError(
                "LimiX-2 Feature_imputation returned shape "
                f"{values.shape}; expected (*, {n_columns})"
            )
        if values.shape[0] == n_query:
            return np.asarray(values)
        if values.shape[0] == n_train + n_query:
            return np.asarray(values[-n_query:])
        raise RuntimeError(
            "LimiX-2 Feature_imputation returned an unexpected row count: "
            f"{values.shape[0]} (train={n_train}, query={n_query})"
        )

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
            list(schema.feature_cols)
            if target_access == "feature_only"
            else all_columns
        )
        train = first.bundle.train.reset_index(drop=True)[model_columns]
        codec = _FrameCodec.fit(train, model_columns, schema.categorical_cols)
        variable = codec.variable_columns
        rows = self._train_indices(len(train), first.train_seed)
        reference = codec.encode(
            train.iloc[rows], variable, preserve_missing=False
        )
        dummy_target = np.zeros(len(reference), dtype=np.int64)

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
                raise ValueError("LimiX-2 feature imputation must not mask the target")
            masked_cells = mask.to_numpy(dtype=bool)
            if not pd.isna(masked.to_numpy())[masked_cells].all():
                raise ValueError("mask=True cells must be NaN in masked_test")

            model_mask = mask.reindex(columns=model_columns, fill_value=False)
            variable_mask = model_mask.reindex(
                columns=variable, fill_value=False
            ).to_numpy(dtype=bool)
            completed = masked.copy()
            for column, value in codec.constant_values.items():
                rows_to_fill = model_mask[column].to_numpy(dtype=bool)
                if rows_to_fill.any():
                    completed.loc[rows_to_fill, column] = value

            if variable and variable_mask.any():
                query = codec.encode(
                    masked[model_columns], variable, preserve_missing=False
                )
                query[variable_mask] = np.nan
                predictor = self._predictor_for(ctx, "imputation", ctx.seed)
                prediction = predictor.predict(
                    reference,
                    dummy_target,
                    query,
                    task_type="Feature_imputation",
                    unique_dataset_name=(
                        f"{ctx.bundle.name}_imputation_{context_index}_{ctx.seed}"
                    ),
                )
                if prediction is None:
                    raise RuntimeError("LimiX-2 Feature_imputation returned None")
                query_prediction = self._query_imputation_rows(
                    prediction,
                    n_train=len(reference),
                    n_query=len(query),
                    n_columns=len(variable),
                )
                decoded = codec.decode(query_prediction, variable)
                for column in variable:
                    rows_to_fill = model_mask[column].to_numpy(dtype=bool)
                    if rows_to_fill.any():
                        completed.loc[rows_to_fill, column] = decoded.loc[
                            rows_to_fill, column
                        ]

            if pd.isna(completed.to_numpy())[masked_cells].any():
                raise RuntimeError(
                    "LimiX-2 left an artificially masked cell empty"
                )
            n_draws = max(1, int(ctx.trials))
            native_missing = int(
                (
                    masked[model_columns].isna().to_numpy()
                    & ~model_mask.to_numpy(dtype=bool)
                ).sum()
            )
            outputs.append(
                ImputationOutput.from_frames(
                    [completed[all_columns].copy() for _ in range(n_draws)],
                    metadata={
                        "source": "native_point",
                        "backend": "LimiXPredictor Feature_imputation",
                        "draw_strategy": "point reconstruction",
                        "n_draws": n_draws,
                        "target_access": target_access,
                        "auxiliary_context_target": "constant_zero",
                        "imputation_scope": "artificial_mask_only",
                        "native_missing_output_policy": "preserve",
                        "native_missing_conditioning_policy": "train_fitted_fill",
                        "native_missing_conditioning_cells": native_missing,
                        "predictor_weights_shared_across_settings": len(contexts),
                    },
                )
            )
        return outputs

    def impute_draws(self, ctx: ImputeContext) -> ImputationOutput:
        return self.impute_draws_many([ctx])[0]

    def impute(self, ctx: ImputeContext) -> pd.DataFrame:
        return self.impute_draws(replace(ctx, trials=1)).draws[0]

    # ------------------------------------------------------------------ P
    def _prediction_features(
        self, ctx: PredictContext
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Build a train-fitted numeric feature representation for prediction."""

        schema = ctx.bundle.schema
        feature_columns = list(schema.feature_cols)
        if not feature_columns:
            raise ValueError("LimiX-2 prediction requires at least one feature column")
        train = ctx.bundle.train.reset_index(drop=True)[feature_columns]
        test = ctx.bundle.test.reset_index(drop=True)[feature_columns]
        codec = _FrameCodec.fit(train, feature_columns, schema.categorical_cols)
        variable = codec.variable_columns
        rows = self._train_indices(len(train), ctx.train_seed)
        if variable:
            x_train = codec.encode(
                train.iloc[rows], variable, preserve_missing=False
            )
            x_test = codec.encode(test, variable, preserve_missing=False)
        else:
            # The public predictor rejects zero input columns.  A synthetic
            # constant carries no information and therefore represents the
            # correct intercept-only problem when all real features are fixed.
            x_train = np.zeros((len(rows), 1), dtype=np.float32)
            x_test = np.zeros((len(test), 1), dtype=np.float32)
        return x_train, x_test, rows, [
            column for column in feature_columns if column not in variable
        ]

    @staticmethod
    def _point_regression_prediction(
        predictor,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        *,
        dataset_name: str,
    ) -> np.ndarray:
        output = predictor.predict(
            x_train,
            np.asarray(y_train, dtype=np.float64),
            x_test,
            task_type="Regression",
            unique_dataset_name=dataset_name,
        )
        if isinstance(output, tuple):
            output = output[0]
        values = _as_numpy(output).astype(np.float64, copy=False).reshape(-1)
        if len(values) != len(x_test):
            raise RuntimeError(
                "LimiX-2 regression returned "
                f"{len(values)} rows; expected {len(x_test)}"
            )
        if not np.isfinite(values).all():
            raise RuntimeError("LimiX-2 regression returned a non-finite prediction")
        return values

    def predict(self, ctx: PredictContext) -> PredictionOutput:
        """Run native LimiX-2 P, including the requested >10-class route."""

        seed_everything(ctx.seed)
        schema = ctx.bundle.schema
        train = ctx.bundle.train.reset_index(drop=True)
        target = schema.target_col
        x_train, x_test, rows, constant_features = self._prediction_features(ctx)
        base_metadata = {
            "adapter": self.name,
            "feature_encoding": "train_fitted_numeric",
            "constant_features_removed": constant_features,
            "train_rows": int(len(rows)),
            "categorical_classification_limit": 10,
            "high_cardinality_policy": (
                "numerical_regression_then_round_clip_decode"
            ),
        }

        if schema.task_type == "regression":
            y_all = pd.to_numeric(train[target], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            fill = y_all.mean()
            y_all = y_all.fillna(0.0 if pd.isna(fill) else float(fill))
            temperature = float(
                self.cfg("predict", "regression_temperature", 1.0)
            )
            predictor = self._predictor_for(
                ctx, "regression", ctx.seed, temperature=temperature
            )
            y_pred = self._point_regression_prediction(
                predictor,
                x_train,
                y_all.iloc[rows].to_numpy(dtype=np.float64),
                x_test,
                dataset_name=f"{ctx.bundle.name}_prediction_{ctx.seed}",
            )
            return PredictionOutput(
                y_pred=y_pred,
                metadata={
                    **base_metadata,
                    "prediction_route": "public LimiXPredictor Regression",
                    "native_output": "point",
                },
            )

        target_frame = train[[target]]
        target_codec = _FrameCodec.fit(target_frame, [target], [target])
        labels = list(target_codec.category_values[target])
        n_classes = len(labels)
        y_all = target_codec.encode(
            target_frame, [target], preserve_missing=False
        )[:, 0]

        if n_classes == 1:
            y_pred = np.asarray([labels[0]] * len(x_test), dtype=object)
            return PredictionOutput(
                y_pred=y_pred,
                y_proba=np.ones((len(x_test), 1), dtype=np.float64),
                labels=labels,
                metadata={
                    **base_metadata,
                    "prediction_route": "constant_classification_target",
                    "target_class_count": 1,
                    "native_output": "constant_point_and_probability",
                },
            )

        if n_classes <= 10:
            observed_codes = np.unique(
                np.asarray(y_all[rows], dtype=np.int64)
            )
            if len(observed_codes) == 1:
                code = int(observed_codes[0])
                probabilities = np.zeros(
                    (len(x_test), n_classes), dtype=np.float64
                )
                probabilities[:, code] = 1.0
                y_pred = np.asarray([labels[code]] * len(x_test), dtype=object)
                return PredictionOutput(
                    y_pred=y_pred,
                    y_proba=probabilities,
                    labels=labels,
                    metadata={
                        **base_metadata,
                        "prediction_route": "constant_subsample_classification_target",
                        "target_class_count": n_classes,
                        "context_class_count": 1,
                        "native_output": "constant_point_and_probability",
                    },
                )
            temperature = float(
                self.cfg("predict", "classification_temperature", 0.9)
            )
            predictor = self._predictor_for(
                ctx, "classification", ctx.seed, temperature=temperature
            )
            output = predictor.predict(
                x_train,
                np.asarray(y_all[rows], dtype=np.int64),
                x_test,
                task_type="Classification",
                unique_dataset_name=f"{ctx.bundle.name}_prediction_{ctx.seed}",
            )
            if isinstance(output, tuple):
                output = output[0]
            observed_probabilities = _as_numpy(output).astype(
                np.float64, copy=False
            )
            expected_shape = (len(x_test), len(observed_codes))
            if observed_probabilities.shape != expected_shape:
                raise RuntimeError(
                    "LimiX-2 classification returned shape "
                    f"{observed_probabilities.shape}; expected {expected_shape}"
                )
            observed_probabilities = np.nan_to_num(
                observed_probabilities, nan=0.0, posinf=0.0, neginf=0.0
            )
            observed_probabilities = np.clip(
                observed_probabilities, 0.0, None
            )
            totals = observed_probabilities.sum(axis=1, keepdims=True)
            if np.any(totals <= 0):
                raise RuntimeError("LimiX-2 returned a zero-mass probability row")
            observed_probabilities = observed_probabilities / totals
            probabilities = np.zeros(
                (len(x_test), n_classes), dtype=np.float64
            )
            probabilities[:, observed_codes] = observed_probabilities
            codes = probabilities.argmax(axis=1)
            y_pred = target_codec.decode(
                codes.reshape(-1, 1), [target]
            )[target].to_numpy(dtype=object)
            return PredictionOutput(
                y_pred=y_pred,
                y_proba=probabilities,
                labels=labels,
                metadata={
                    **base_metadata,
                    "prediction_route": "public LimiXPredictor Classification",
                    "target_class_count": n_classes,
                    "context_class_count": int(len(observed_codes)),
                    "native_output": "point_and_proba",
                },
            )

        temperature = float(
            self.cfg("predict", "regression_temperature", 1.0)
        )
        predictor = self._predictor_for(
            ctx, "regression", ctx.seed, temperature=temperature
        )
        continuous = self._point_regression_prediction(
            predictor,
            x_train,
            np.asarray(y_all[rows], dtype=np.float64),
            x_test,
            dataset_name=f"{ctx.bundle.name}_prediction_{ctx.seed}",
        )
        y_pred = target_codec.decode(
            continuous.reshape(-1, 1), [target]
        )[target].to_numpy(dtype=object)
        return PredictionOutput(
            y_pred=y_pred,
            metadata={
                **base_metadata,
                "prediction_route": "categorical_as_numerical_regression",
                "target_class_count": n_classes,
                "native_output": "point",
                "probabilities_available": False,
            },
        )

    def evaluation_metadata(self, ctx) -> dict[str, Any]:
        try:
            version = importlib_metadata.version("LimiX")
        except importlib_metadata.PackageNotFoundError:
            version = "source-tree"
        metadata = {
            "variant": "LimiX-2 architecture v2.0",
            "limix_version": version,
            "generation_route": "sequential conditional LimiX-2 inference",
            "categorical_classification_limit": 10,
            "high_cardinality_policy": "numerical_regression_then_round_clip_decode",
            "imputation_route": "public LimiXPredictor Feature_imputation",
            "imputation_target_access": self._target_access(),
            "prediction_route": "public LimiXPredictor classification/regression",
            "prediction_high_cardinality_policy": (
                "categorical_as_numerical_regression_point_only"
            ),
        }
        audit = getattr(self, "_last_generation_audit", None)
        if audit is not None:
            metadata["generation_audit"] = audit
        return metadata


__all__ = ["LimiX2Method"]

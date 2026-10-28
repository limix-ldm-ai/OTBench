# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""GReaT generation, arbitrary-mask imputation and target-mask prediction.

All three tasks share one fine-tuned ``be_great.GReaT`` checkpoint. Missing
cells are passed to the package's public ``model.impute(...)`` API; prediction
is that same operation with the target column masked. The adapter keeps the
benchmark offline: the base language model must already exist locally.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from otbench.methods.common import checkpoints as ckpt_utils
from otbench.methods.common.base import (
    GenerateContext,
    ImputationOutput,
    ImputeContext,
    MethodContext,
    PredictContext,
    PredictionOutput,
    Task,
    TrainContext,
)
from otbench.methods.common.capabilities import UnsupportedRun
from otbench.methods.common.imputer_base import merge_masked_only
from otbench.methods.common.profiling import count_parameters
from otbench.methods.common.seeding import seed_everything
from otbench.methods.common.synth_method import SynthPluginMethod

_MODEL_FILE = "model.pkl"
_CODEC_FILE = "codec.json"
_MAX_TRAIN_ROWS = 3000  # mirrors GReaTPlugin._fit


class GReaTMethod(SynthPluginMethod):
    name = "great"
    tasks = frozenset({Task.GENERATION, Task.IMPUTATION, Task.PREDICTION})
    conditional = True
    fit_cond_target_name = True
    regression_cond = True
    misc_plugin_kwargs = {"sampling_patience": 1, "strict": True}

    def plugin_class(self):
        from .plugin import GReaTPlugin

        return GReaTPlugin

    def plugin_kwargs(self, ctx) -> dict:
        kwargs = super().plugin_kwargs(ctx)
        kwargs["llm"] = self.pretrained_path("llm")
        return kwargs

    def checkpoint_profile(self, plugin) -> dict:
        # plugin.model -> TabularGReaT -> be_great.GReaT -> HF torch module.
        transformer = getattr(getattr(plugin, "model", None), "model", None)
        transformer = getattr(transformer, "model", None)
        n_parameters = count_parameters(transformer)
        profile = {"n_parameters": n_parameters} if n_parameters is not None else {}
        report = getattr(self, "_training_context_report", None)
        if report is not None:
            profile["training_context_preflight"] = report
        return profile

    def train(self, ctx: TrainContext) -> None:
        # Column count is not an upstream limit. The actual constraint is the
        # causal LM context after GReaTDataset serializes a full row as
        # "column is value, ..." in a random column order. Check that exact
        # representation before spending hours in Trainer.
        frame, _ = self._fit_frame(ctx.bundle)
        if len(frame) > _MAX_TRAIN_ROWS:
            frame = frame.sample(_MAX_TRAIN_ROWS, random_state=int(ctx.train_seed))
        llm = self.plugin_kwargs(ctx)["llm"]
        self._training_context_report = _training_context_preflight(frame, llm)
        super().train(ctx)

    def generate(self, ctx: GenerateContext) -> pd.DataFrame:
        try:
            return super().generate(ctx)
        except (RuntimeError, ValueError) as exc:
            message = str(exc)
            if "generated no samples" in message or "No objects to concatenate" in message:
                raise UnsupportedRun(
                    "GReaT produced no valid rows after its bounded constrained-sampling retries."
                ) from exc
            raise

    def impute(self, ctx: ImputeContext) -> pd.DataFrame:
        return self.impute_draws(ctx).point_estimate(ctx.bundle.schema)

    def impute_draws(self, ctx: ImputeContext) -> ImputationOutput:
        plugin, codec = self._load_plugin(ctx)
        schema = ctx.bundle.schema
        masked = ctx.masked_test.reset_index(drop=True)[schema.all_columns]
        encoded = _encode_masked(masked, schema, codec)

        draws, draw_stats = [], []
        for draw_index in range(max(1, int(ctx.trials))):
            seed_everything(int(ctx.seed) + draw_index)
            try:
                completed, stats = plugin.impute(
                    encoded.copy(),
                    temperature=float(self.cfg("impute", "temperature", 0.7)),
                    k=int(self.cfg("impute", "batch_size", 100)),
                    max_length=int(self.cfg("impute", "max_length", 1000)),
                    max_retries=int(self.cfg("impute", "max_retries", 15)),
                    device=self.resolve_device(ctx),
                    return_stats=True,
                )
            except ValueError as exc:
                if "No objects to concatenate" in str(exc):
                    raise UnsupportedRun(
                        "GReaT imputation produced no valid rows after its bounded retries."
                    ) from exc
                raise
            if len(completed) != len(encoded) or set(completed.index) != set(encoded.index):
                raise UnsupportedRun(
                    "great official impute() exhausted max_retries and dropped "
                    f"{len(encoded) - len(completed)} row(s); stats={stats}"
                )
            completed = completed.reindex(encoded.index)
            repairs = _decode_repair_counts(completed, schema, codec)
            decoded = self._decode_frame(completed, ctx.bundle, codec)
            draws.append(merge_masked_only(masked, decoded, ctx.mask))
            draw_stats.append({**stats, **repairs, "seed": int(ctx.seed) + draw_index})

        return ImputationOutput.from_frames(draws, metadata={
            "source": "native",
            "draw_strategy": "independent calls to be_great.GReaT.impute on the same mask",
            "n_draws": len(draws),
            "impute_accounting": _sum_stats(draw_stats),
            "per_draw_accounting": draw_stats,
        })

    def predict(self, ctx: PredictContext) -> PredictionOutput:
        schema = ctx.bundle.schema
        test = ctx.bundle.test.reset_index(drop=True)[schema.all_columns]
        mask = pd.DataFrame(False, index=test.index, columns=test.columns)
        mask[schema.target_col] = True
        result = self.impute_draws(ImputeContext(
            bundle=ctx.bundle,
            config=ctx.config,
            checkpoint_dir=ctx.checkpoint_dir,
            train_seed=ctx.train_seed,
            seed=ctx.seed,
            device=ctx.device,
            masked_test=test.mask(mask),
            mask=mask,
            trials=max(1, int(ctx.n_samples)),
        ))

        target = schema.target_col
        shared = {
            "source": "native",
            "prediction_route": "be_great.GReaT.impute with the target column masked",
            "n_samples": result.n_draws,
            "impute_accounting": result.metadata["impute_accounting"],
        }
        if schema.task_type == "regression" or target in set(schema.numerical_cols):
            samples = np.column_stack([
                pd.to_numeric(draw[target], errors="coerce").to_numpy(float)
                for draw in result.draws
            ])
            return PredictionOutput(
                y_pred=samples.mean(axis=1), y_samples=samples,
                metadata={**shared, "point_rule": "mean of native target-mask draws"},
            )

        labels = pd.Series(
            ctx.bundle.train[target].dropna().unique()
        ).sort_values(kind="stable").tolist()
        sampled = np.column_stack([draw[target].to_numpy() for draw in result.draws])
        proba = np.column_stack([(sampled == label).mean(axis=1) for label in labels])
        codes = proba.argmax(axis=1)
        return PredictionOutput(
            y_pred=np.asarray([labels[int(code)] for code in codes], dtype=object),
            y_proba=proba,
            y_score=proba[:, 1] if len(labels) == 2 else None,
            labels=labels,
            threshold=0.5 if len(labels) == 2 else None,
            threshold_source="fixed_0.5" if len(labels) == 2 else "argmax",
            metadata={**shared, "probability_rule": "class frequency over native draws"},
        )

    def _load_plugin(self, ctx: MethodContext):
        from otbench.methods.common.synth.plugins.core.plugin import Plugin

        ckpt_dir = Path(ctx.checkpoint_dir)
        plugin = Plugin.load((ckpt_dir / _MODEL_FILE).read_bytes())
        with (ckpt_dir / _CODEC_FILE).open("r", encoding="utf-8") as handle:
            codec = json.load(handle)
        return plugin, codec


def _training_context_preflight(frame: pd.DataFrame, llm: str) -> dict:
    """Reject only rows that cannot fit the configured local LLM context."""
    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(llm, local_files_only=True)
    model_config = AutoConfig.from_pretrained(llm, local_files_only=True)
    context_window = _context_window(tokenizer, model_config)
    report = _serialized_token_report(frame, tokenizer)
    row_bounds = report.pop("_row_bounds")
    report.update({
        "context_window": context_window,
        "rows_over_context": int(sum(length > context_window for length in row_bounds)),
        "llm": str(Path(llm).resolve()),
        "modeled_columns": int(len(frame.columns)),
        "checked_rows": int(len(frame)),
        "serialization": "comma-joined '<column> is <value>'; random column order",
        "bound": "conservative all-permutation upper bound",
    })
    if report["rows_over_context"]:
        raise UnsupportedRun(
            "GReaT serialized training rows exceed the configured local LLM context: "
            f"{report['rows_over_context']}/{len(frame)} checked row(s) exceed "
            f"{context_window} tokens (maximum conservative bound "
            f"{report['max_tokens']}; {len(frame.columns)} modeled columns). "
            "Use a longer-context local checkpoint; the former fixed 50-feature "
            "limit has been removed."
        )
    return report


def _context_window(tokenizer, model_config) -> int:
    """Return the smallest finite context limit declared by tokenizer/model."""
    candidates = []
    for value in (
        getattr(tokenizer, "model_max_length", None),
        getattr(model_config, "max_position_embeddings", None),
        getattr(model_config, "n_positions", None),
        getattr(model_config, "n_ctx", None),
    ):
        try:
            parsed = int(value)
        except (TypeError, ValueError, OverflowError):
            continue
        # HuggingFace uses enormous sentinel values when no tokenizer limit is
        # known; those are not real context windows.
        if 0 < parsed < 10_000_000:
            candidates.append(parsed)
    if not candidates:
        raise UnsupportedRun(
            "GReaT could not determine the context window of the configured local LLM."
        )
    return min(candidates)


def _serialized_token_report(frame: pd.DataFrame, tokenizer) -> dict[str, int | str]:
    """Upper-bound GReaTDataset token lengths over every column permutation.

    Every non-first field has the same ``, `` prefix. Tokenizing each segment
    independently can only miss cross-segment BPE merges, so summing those
    lengths is conservative. Trying each field as the unprefixed first segment
    covers the only position-dependent difference without enumerating N! orders.
    """
    special = 0
    if callable(getattr(tokenizer, "num_special_tokens_to_add", None)):
        special = int(tokenizer.num_special_tokens_to_add(pair=False))

    max_tokens = 0
    max_row = ""
    row_bounds = []
    columns = [str(column) for column in frame.columns]
    n_columns = len(columns)
    records = list(frame.itertuples(index=True, name=None))
    for start in range(0, len(records), 64):
        block = records[start:start + 64]
        bare_texts = []
        for record in block:
            bare_texts.extend(
                f"{column} is {str(value).strip()}"
                for column, value in zip(columns, record[1:])
            )
        encoded = tokenizer(
            bare_texts + [f", {field}" for field in bare_texts],
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )["input_ids"]
        split = len(bare_texts)
        bare_lengths = [len(ids) for ids in encoded[:split]]
        prefixed_lengths = [len(ids) for ids in encoded[split:]]
        for offset, record in enumerate(block):
            left = offset * n_columns
            right = left + n_columns
            first = bare_lengths[left:right]
            prefixed = prefixed_lengths[left:right]
            bound = sum(prefixed) + max(
                (bare - continuation for bare, continuation in zip(first, prefixed)),
                default=0,
            ) + special
            row_bounds.append(int(bound))
            if bound > max_tokens:
                max_tokens = int(bound)
                max_row = str(record[0])

    return {
        "max_tokens": max_tokens,
        "max_token_row": max_row,
        "_row_bounds": row_bounds,
    }


def _encode_masked(frame: pd.DataFrame, schema, codec: dict) -> pd.DataFrame:
    """Apply the train-fitted synth codec while preserving missing cells."""
    out = pd.DataFrame(index=frame.index)
    for column in codec["numerical_columns"]:
        out[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in codec["categorical_columns"]:
        mapping = {str(value): code for code, value in enumerate(codec["category_mappings"][column])}
        out[column] = [np.nan if pd.isna(value) else mapping.get(str(value), np.nan)
                       for value in frame[column]]

    target = schema.target_col
    if schema.task_type == "regression":
        out[target] = pd.to_numeric(frame[target], errors="coerce")
    else:
        mapping = {str(value): code for code, value in enumerate(codec["y_classes"])}
        out[target] = [np.nan if pd.isna(value) else mapping.get(str(value), np.nan)
                       for value in frame[target]]
    return out[schema.all_columns]


def _decode_repair_counts(frame: pd.DataFrame, schema, codec: dict) -> dict[str, int]:
    """Count adapter-side numeric parsing, rounding and clipping before decode."""
    numeric_parse_failures = 0
    categorical_rounding_repairs = 0
    categorical_clipping_repairs = 0
    domains = {
        column: len(codec["category_mappings"][column])
        for column in codec["categorical_columns"]
    }
    if schema.task_type != "regression":
        domains[schema.target_col] = len(codec["y_classes"])

    for column in schema.all_columns:
        raw = frame[column]
        parsed = pd.to_numeric(raw, errors="coerce")
        numeric_parse_failures += int((raw.notna() & parsed.isna()).sum())
        if column in domains:
            finite = parsed[np.isfinite(parsed)]
            categorical_rounding_repairs += int((np.abs(finite - np.rint(finite)) > 1e-9).sum())
            categorical_clipping_repairs += int(
                ((finite < 0) | (finite > max(0, domains[column] - 1))).sum()
            )
    return {
        "decode_numeric_parse_failures": numeric_parse_failures,
        "decode_rounding_repairs": categorical_rounding_repairs,
        "decode_clipping_repairs": categorical_clipping_repairs,
    }


def _sum_stats(items: list[dict]) -> dict:
    summed: dict[str, float | int] = {}
    for item in items:
        for key, value in item.items():
            if key == "seed" or not isinstance(value, (int, float)):
                continue
            summed[key] = summed.get(key, 0) + value
    return summed

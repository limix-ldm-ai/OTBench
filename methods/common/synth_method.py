"""Shared workflow for plugin-backed generators.

Train-fitted category encoding, conditional/joint sampling, bounded batches,
method-specific row limits, and plugin/codec checkpoint persistence live here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from . import checkpoints as ckpt_utils
from . import data as data_utils
from .base import GenerateContext, TabularMethod, Task, TrainContext
from .seeding import seed_everything

_MODEL_FILE = "model.pkl"
_CODEC_FILE = "codec.json"
_BATCH = 1000


class SynthPluginMethod(TabularMethod):
    tasks = frozenset({Task.GENERATION})
    trainable = True
    checkpoint_runtime_packages = ("cloudpickle", "numpy", "pandas", "pydantic")
    require_checkpoint_runtime_fingerprint = True

    conditional = False            # class-conditional plugin (ctgan/tvae/great)
    fit_task_type_kwarg = False    # scm: fit(df, task_type=...)
    fit_cond_target_name = False   # great: fit cond = target column NAME
    regression_cond = False        # conditional+regression: cond=y values (great)
    row_cap: Optional[int] = None            # nflow: 2000, bn: 10000
    discretize_numeric_bins: Optional[int] = None  # bn: 100
    restore_numeric_units: bool = False  # separate BN identity; bn emits bin indices
    misc_plugin_kwargs: dict = {}

    # ------------------------------------------------------------- subclass
    def plugin_class(self):
        raise NotImplementedError

    def plugin_kwargs(self, ctx) -> dict:
        kwargs = dict(self.config.get("model", {}))
        kwargs.update(self.config.get("train", {}))
        kwargs.update(self.misc_plugin_kwargs)
        kwargs.setdefault("random_state", int(ctx.train_seed))
        return kwargs

    def checkpoint_profile(self, plugin) -> dict:
        """Optional model-size facts written into generic checkpoint metadata."""
        return {}

    # ---------------------------------------------------------------- train
    def train(self, ctx: TrainContext) -> None:
        seed_everything(ctx.train_seed)
        frame, codec_payload = self._fit_frame(ctx.bundle)
        plugin = self.plugin_class()(**self.plugin_kwargs(ctx))

        fit_kwargs: dict[str, Any] = {}
        if self.fit_task_type_kwarg:
            fit_kwargs["task_type"] = (
                "regression" if ctx.bundle.schema.task_type == "regression" else "classification"
            )
        cond = self._fit_cond(ctx.bundle, frame)
        if cond is not None:
            plugin.fit(frame, cond=cond, **fit_kwargs)
        else:
            plugin.fit(frame, **fit_kwargs)

        ckpt_dir = Path(ctx.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / _MODEL_FILE).write_bytes(plugin.save())
        with (ckpt_dir / _CODEC_FILE).open("w", encoding="utf-8") as f:
            json.dump(codec_payload, f, indent=2, ensure_ascii=False, default=str)
        ckpt_utils.write_metadata(
            ckpt_dir, self.name, ctx.bundle, self.config, ctx.train_seed,
            artifacts={"model": _MODEL_FILE, "codec": _CODEC_FILE},
            extra=self.checkpoint_profile(plugin),
            runtime_packages=self.checkpoint_runtime_packages,
        )

    # ------------------------------------------------------------- generate
    def generate(self, ctx: GenerateContext) -> pd.DataFrame:
        seed_everything(ctx.seed)
        if self.trainable:
            from .synth.plugins.core.plugin import Plugin

            ckpt_dir = Path(ctx.checkpoint_dir)
            plugin = Plugin.load((ckpt_dir / _MODEL_FILE).read_bytes())
            with (ckpt_dir / _CODEC_FILE).open("r", encoding="utf-8") as f:
                codec_payload = json.load(f)
        else:
            # fit-at-generate methods (non-serializable backends): train in-memory
            frame, codec_payload = self._fit_frame(ctx.bundle)
            plugin = self.plugin_class()(**self.plugin_kwargs(ctx))
            fit_kwargs: dict[str, Any] = {}
            if self.fit_task_type_kwarg:
                fit_kwargs["task_type"] = (
                    "regression" if ctx.bundle.schema.task_type == "regression" else "classification"
                )
            cond = self._fit_cond(ctx.bundle, frame)
            if cond is not None:
                plugin.fit(frame, cond=cond, **fit_kwargs)
            else:
                plugin.fit(frame, **fit_kwargs)

        bundle = ctx.bundle
        schema = bundle.schema
        n_samples = int(ctx.n_samples or len(bundle.train))
        target = schema.target_col
        is_classification = schema.task_type != "regression"

        if is_classification:
            class_codes = list(range(len(codec_payload["y_classes"])))
            class2samples = self._class2samples(bundle, codec_payload, n_samples)
            parts = []
            for code in class_codes:
                count = class2samples.get(code, 0)
                if count <= 0:
                    continue
                parts.append(self._sample_class(plugin, target, code, count))
            frame = pd.concat(parts, axis=0).reset_index(drop=True) if parts else pd.DataFrame()
        else:
            cond_pool = None
            if self.conditional and self.regression_cond:
                cond_pool = pd.to_numeric(
                    bundle.train[target], errors="coerce"
                ).fillna(0.0).to_numpy()
            frame = self._sample_joint(plugin, n_samples, cond_pool)

        return self._decode_frame(frame, bundle, codec_payload)

    # ------------------------------------------------------------- fit data
    def _fit_frame(self, bundle) -> tuple[pd.DataFrame, dict]:
        schema = bundle.schema
        train = bundle.train.reset_index(drop=True)
        num_cols = data_utils.numerical_feature_columns(schema)
        cat_cols = data_utils.categorical_feature_columns(schema)

        features = data_utils.fill_feature_nans(
            data_utils.feature_only(train, schema),
            data_utils.feature_only(train, schema),
            schema,
        )
        codec = data_utils.CategoryCodec.fit(features, cat_cols)
        frame = features.copy()
        for col in num_cols:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = codec.encode(frame, cat_cols)

        y_classes: list[Any] = []
        if schema.task_type == "regression":
            frame[schema.target_col] = pd.to_numeric(train[schema.target_col], errors="coerce").fillna(0.0)
        else:
            y_classes = pd.Series(train[schema.target_col].dropna().unique()).sort_values(kind="stable").tolist()
            to_code = {v: i for i, v in enumerate(y_classes)}
            frame[schema.target_col] = train[schema.target_col].map(to_code).fillna(0).astype(int)

        # corner cases (original BaseJointPluginGenerator.process_with_corner_case)
        bin_edges: dict[str, list[float]] = {}
        if self.discretize_numeric_bins:
            for col in num_cols:
                frame[col], edges = pd.cut(
                    frame[col], bins=self.discretize_numeric_bins, labels=False,
                    duplicates="drop", retbins=True,
                )
                if self.restore_numeric_units:
                    bin_edges[col] = edges.tolist()
        if self.row_cap is not None:
            frame = frame.loc[: self.row_cap, :]

        payload = {
            "feature_columns": list(features.columns),
            "numerical_columns": num_cols,
            "categorical_columns": cat_cols,
            "category_mappings": {k: [str(v) for v in vals] for k, vals in codec.mappings.items()},
            "category_modes": {k: str(v) for k, v in codec.modes.items()},
            "y_classes": y_classes,
            "train_class_counts": (
                frame[schema.target_col].value_counts().sort_index().to_dict()
                if schema.task_type != "regression" else {}
            ),
            "discretized_bins": self.discretize_numeric_bins,
            "bin_edges": bin_edges,
        }
        return frame, payload

    def _fit_cond(self, bundle, frame: pd.DataFrame):
        if not self.conditional:
            return None
        target = bundle.schema.target_col
        if self.fit_cond_target_name:
            return target
        if bundle.schema.task_type == "regression":
            return frame[target].to_numpy() if self.regression_cond else None
        return frame[target].to_numpy()

    # ------------------------------------------------------------- sampling
    def _class2samples(self, bundle, codec_payload: dict, n_total: int) -> dict[int, int]:
        counts = {int(k): int(v) for k, v in codec_payload["train_class_counts"].items()}
        total = sum(counts.values()) or 1
        class2samples = {
            code: int(n_total * (count / total)) for code, count in counts.items()
        }
        tol = 0
        while sum(class2samples.values()) != n_total and tol < 100:
            code = int(np.random.choice(list(class2samples.keys())))
            class2samples[code] -= int(np.sign(sum(class2samples.values()) - n_total))
            class2samples[code] = max(class2samples[code], 1)
            tol += 1
        if sum(class2samples.values()) != n_total:
            raise ValueError("Could not align per-class sample counts to the total.")
        return class2samples

    def _class_constraints(self, target: str, code: int):
        from .synth.constraints import Constraints

        return Constraints(rules=[(target, "==", code), (target, "dtype", "int")])

    def _sample_class(self, plugin, target: str, code: int, count: int) -> pd.DataFrame:
        constraints = self._class_constraints(target, code)
        collected: list[pd.DataFrame] = []
        n_current, patience = 0, 0
        max_patience = count // _BATCH + 10
        while n_current < count and patience < max_patience:
            ask = min(count - n_current, _BATCH)
            kwargs: dict[str, Any] = {"constraints": constraints}
            if self.conditional:
                kwargs["cond"] = [code] * ask
            loader = plugin.generate(count=ask, **kwargs)
            got = loader.dataframe().reset_index(drop=True)
            collected.append(got)
            n_current += len(got)
            patience += 1
        frame = pd.concat(collected, axis=0).reset_index(drop=True) if collected else pd.DataFrame()
        if len(frame) < count:
            if len(frame) == 0:
                raise RuntimeError(f"{self.name} generated no samples for class {code}.")
            idx = np.random.choice(len(frame), count)
            frame = frame.iloc[idx].reset_index(drop=True)
        return frame.iloc[:count]

    def _sample_joint(self, plugin, count: int, cond_pool: Optional[np.ndarray]) -> pd.DataFrame:
        collected: list[pd.DataFrame] = []
        n_current, patience = 0, 0
        max_patience = count // _BATCH + 10
        while n_current < count and patience < max_patience:
            ask = min(count - n_current, _BATCH)
            kwargs: dict[str, Any] = {}
            if self.conditional and cond_pool is not None:
                kwargs["cond"] = np.resize(cond_pool, ask)
            loader = plugin.generate(count=ask, **kwargs)
            got = loader.dataframe().reset_index(drop=True)
            collected.append(got)
            n_current += len(got)
            patience += 1
        frame = pd.concat(collected, axis=0).reset_index(drop=True) if collected else pd.DataFrame()
        if len(frame) < count:
            if len(frame) == 0:
                raise RuntimeError(f"{self.name} generated no samples.")
            idx = np.random.choice(len(frame), count)
            frame = frame.iloc[idx].reset_index(drop=True)
        return frame.iloc[:count]

    # --------------------------------------------------------------- decode
    def _decode_frame(self, frame: pd.DataFrame, bundle, codec_payload: dict) -> pd.DataFrame:
        schema = bundle.schema
        target = schema.target_col
        out = pd.DataFrame(index=frame.index)
        for col in codec_payload["numerical_columns"]:
            values = pd.to_numeric(frame.get(col), errors="coerce")
            if self.restore_numeric_units:
                edges = np.asarray(codec_payload.get("bin_edges", {}).get(col, []), dtype=float)
                if len(edges) < 2:
                    raise ValueError(f"Missing fitted bin edges for {self.name} column {col!r}")
                codes = values.round().clip(0, len(edges) - 2)
                midpoints = (edges[:-1] + edges[1:]) / 2.0
                values = codes.map(lambda code: midpoints[int(code)] if pd.notna(code) else np.nan)
            out[col] = values
        for col in codec_payload["categorical_columns"]:
            values = codec_payload["category_mappings"].get(col, ["__MISSING__"])
            ids = pd.to_numeric(frame.get(col), errors="coerce").fillna(0).round().astype(int)
            ids = ids.clip(0, len(values) - 1)
            out[col] = ids.map(lambda i: values[int(i)])
        if schema.task_type == "regression":
            out[target] = pd.to_numeric(frame.get(target), errors="coerce")
        else:
            classes = codec_payload["y_classes"]
            codes = pd.to_numeric(frame.get(target), errors="coerce").fillna(0).round().astype(int)
            codes = codes.clip(0, len(classes) - 1)
            out[target] = codes.map(lambda i: classes[int(i)])
        return out[schema.all_columns]


# --- one-hot helpers shared by smote / tabebm (processed-space methods) ------
class OneHotSpace:
    """Train-fitted one-hot feature space with argmax decoding."""

    def __init__(self, bundle) -> None:
        schema = bundle.schema
        self.schema = schema
        self.num_cols = data_utils.numerical_feature_columns(schema)
        self.cat_cols = data_utils.categorical_feature_columns(schema)
        train_features = data_utils.fill_feature_nans(
            data_utils.feature_only(bundle.train.reset_index(drop=True), schema),
            data_utils.feature_only(bundle.train.reset_index(drop=True), schema),
            schema,
        )
        self.categories = {
            col: pd.Series(train_features[col].astype(str).unique()).sort_values(kind="stable").tolist()
            for col in self.cat_cols
        }
        self.columns: list[str] = list(self.num_cols) + [
            f"{col}::{value}" for col in self.cat_cols for value in self.categories[col]
        ]
        self._train_features = train_features

    def encode(self, features: pd.DataFrame) -> pd.DataFrame:
        features = data_utils.fill_feature_nans(self._train_features, features, self.schema)
        out = pd.DataFrame(index=features.reset_index(drop=True).index)
        for col in self.num_cols:
            out[col] = pd.to_numeric(features[col], errors="coerce").fillna(0.0).to_numpy()
        for col in self.cat_cols:
            values = features[col].astype(str).to_numpy()
            for value in self.categories[col]:
                out[f"{col}::{value}"] = (values == value).astype(np.float64)
        return out[self.columns]

    def decode(self, matrix: np.ndarray) -> pd.DataFrame:
        frame = pd.DataFrame(matrix, columns=self.columns)
        out = pd.DataFrame(index=frame.index)
        for col in self.num_cols:
            out[col] = pd.to_numeric(frame[col], errors="coerce")
        for col in self.cat_cols:
            block = frame[[f"{col}::{v}" for v in self.categories[col]]].to_numpy()
            idx = block.argmax(axis=1)
            out[col] = [self.categories[col][int(i)] for i in idx]
        return out

    def encode_target(self, y: pd.Series):
        if self.schema.task_type == "regression":
            return pd.to_numeric(y, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64), []
        classes = pd.Series(y.dropna().unique()).sort_values(kind="stable").tolist()
        to_code = {v: i for i, v in enumerate(classes)}
        return y.map(to_code).fillna(0).astype(np.int64).to_numpy(), classes

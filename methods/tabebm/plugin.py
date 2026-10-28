# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""TabEBM as a vendored ``otbench.methods.common.synth`` generic plugin.

TabEBM is training-free but still follows the ``Plugin.fit`` lifecycle so the
base class initializes schema/data_info before generation.
"""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pandas as pd

from otbench.methods.common.synth.plugins.core.dataloader import DataLoader, GenericDataLoader
from otbench.methods.common.synth.plugins.core.distribution import Distribution
from otbench.methods.common.synth.plugins.core.plugin import Plugin
from otbench.methods.common.synth.plugins.core.schema import Schema


_CORE_KWARGS = {
    "sampling_patience",
    "strict",
    "device",
    "random_state",
    "workspace",
    "compress_dataset",
    "sampling_strategy",
}


def _split_kwargs(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    core: dict[str, Any] = {}
    hp: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in _CORE_KWARGS:
            core[key] = value
        else:
            hp[key] = value
    return core, hp


class TabEBMPlugin(Plugin):
    def __init__(self, seed: Optional[int] = None, **kwargs: Any) -> None:
        core, hp = _split_kwargs(kwargs)
        if seed is not None:
            core["random_state"] = int(seed)
        self.seed = int(core.get("random_state", 0))
        self.hp = dict(hp)
        super().__init__(**core)

    @staticmethod
    def name() -> str:
        return "tabebm"

    @staticmethod
    def type() -> str:
        return "generic"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return []

    def _fit(
        self,
        X: DataLoader,
        *args: Any,
        target_col: str = "target",
        task_type: str = "classification",
        **_: Any,
    ) -> "TabEBMPlugin":
        if args:
            raise ValueError("Only keyword arguments are allowed")
        if task_type not in ("classification", "regression"):
            raise ValueError(f"TabEBMPlugin does not support task type {task_type!r}.")

        data_df = X.dataframe().copy()
        data_df.columns = data_df.columns.astype(str)
        self.task_type = task_type
        self.target_col = str(target_col)
        self.feature_order = [col for col in data_df.columns if col != self.target_col]
        if task_type == "regression":
            # SMOTE-style regression handling (same as TabStruct): append the target as
            # an extra feature and treat all real rows as a single pseudo-class 0
            # ("real"); the target is recovered from the last column after generation.
            self.X_train = data_df[self.feature_order + [self.target_col]].to_numpy()
            self.y_train = np.zeros(len(data_df), dtype=np.int64)
        else:
            self.X_train = data_df[self.feature_order].to_numpy()
            self.y_train = np.asarray(data_df[self.target_col]).reshape(-1)
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> GenericDataLoader:
        from .model.tabebm_model import TabEBM as _TabEBM

        count = int(count)
        columns = list(self.feature_order) + [self.target_col]
        if count <= 0:
            return GenericDataLoader(pd.DataFrame(columns=columns), target_column=self.target_col)

        max_data_size = int(self.hp.get("max_data_size", 10000))
        feature_width = int(self.X_train.shape[1])
        max_width = int(self.hp.get("max_onehot_features", 20_000))
        if feature_width > max_width:
            raise ValueError(
                f"TabEBM input width {feature_width:,} exceeds max_onehot_features={max_width:,}"
            )
        max_context_cells = int(self.hp.get("max_context_cells", 1_000_000))
        max_sgld_batch_cells = int(self.hp.get("max_sgld_batch_cells", 1_000_000))
        min_chunk = int(self.hp.get("min_chunk_size", 1))
        if min(max_data_size, max_context_cells, max_sgld_batch_cells, min_chunk) <= 0:
            raise ValueError("TabEBM memory limits and min_chunk_size must be positive")
        context_rows = min(
            max_data_size,
            max(8, max_context_cells // max(1, feature_width)),
        )
        initial_chunk_size = max(
            min_chunk,
            max_sgld_batch_cells // max(1, feature_width),
        )
        model = _TabEBM(max_data_size=context_rows,
                        model_path=self.hp.get("classifier_ckpt"),
                        device=self.hp.get("model_device"))

        # Both the fitted context and query batch are capped by encoded cells,
        # not rows alone. This is essential for Amazon's 14,440-column one-hot
        # representation. Each class is completed before the next one replaces
        # TabPFN's active fit, and a CUDA OOM can still halve the query chunk all
        # the way down to the configured minimum.
        import torch

        class_counts = kwargs.get("class_counts")
        classes = [int(cls) for cls in sorted(np.unique(self.y_train))]
        if self.task_type == "regression":
            requested = {0: count}
        elif class_counts is not None:
            requested = {int(cls): int(n) for cls, n in dict(class_counts).items()}
            unknown = set(requested) - set(classes)
            if unknown:
                raise ValueError(f"TabEBM class_counts contains unknown classes: {sorted(unknown)}")
            if any(n < 0 for n in requested.values()) or sum(requested.values()) != count:
                raise ValueError("TabEBM class_counts must be non-negative and sum to count")
        else:
            observed = np.asarray([(self.y_train == cls).sum() for cls in classes], dtype=float)
            shares = observed / observed.sum()
            raw = shares * count
            base = np.floor(raw).astype(int)
            for index in np.argsort(-(raw - base))[: count - int(base.sum())]:
                base[index] += 1
            requested = dict(zip(classes, base.tolist()))

        generated: dict[str, list[np.ndarray]] = {}
        chunk_index = 0
        for class_id in classes:
            remaining = int(requested.get(class_id, 0))
            chunk_size = min(max(1, remaining), initial_chunk_size)
            while remaining > 0:
                chunk = min(remaining, chunk_size)
                try:
                    chunk_generated = model.generate(
                        self.X_train,
                        self.y_train,
                        num_samples=chunk,
                        target_classes=[class_id],
                        starting_point_noise_std=float(self.hp.get("starting_point_noise_std", 0.01)),
                        sgld_step_size=float(self.hp.get("sgld_step_size", 0.1)),
                        sgld_noise_std=float(self.hp.get("sgld_noise_std", 0.01)),
                        sgld_steps=int(self.hp.get("sgld_steps", 200)),
                        distance_negative_class=float(self.hp.get("distance_negative_class", 5)),
                        seed=self.seed + chunk_index,
                        debug=bool(self.hp.get("debug", False)),
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if chunk <= min_chunk:
                        raise
                    chunk_size = max(min_chunk, chunk // 2)
                    continue
                class_key = f"class_{class_id}"
                generated.setdefault(class_key, []).append(
                    np.asarray(chunk_generated[class_key])
                )
                remaining -= chunk
                chunk_index += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        generated = {key: np.concatenate(chunks, axis=0) for key, chunks in generated.items()}

        frames = []
        if self.task_type == "regression":
            # All rows belong to the pseudo-class 0 ("real"); the target column is the
            # last generated feature.
            samples = np.asarray(generated["class_0"])
            frames.append(pd.DataFrame(samples, columns=self.feature_order + [self.target_col]))
        else:
            for class_id in classes:
                if int(requested.get(class_id, 0)) <= 0:
                    continue
                class_key = f"class_{class_id}"
                samples = np.asarray(generated[class_key])
                frame = pd.DataFrame(samples, columns=self.feature_order)
                frame[self.target_col] = class_id
                frames.append(frame)

        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
        constraints = syn_schema.as_constraints()
        if len(getattr(constraints, "rules", [])) > 0:
            out = constraints.match(out)

        return GenericDataLoader(out.reset_index(drop=True), target_column=self.target_col)


plugin = TabEBMPlugin

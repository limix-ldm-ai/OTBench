# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Common method interface: every method implements :class:`TabularMethod`.

A method declares which tasks it supports (``tasks``) and whether it has a
training phase (``trainable``). Training is task-agnostic: one checkpoint per
(method, dataset, train_seed) is reused by every task the method supports.
"""
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from otbench.tabdata import DatasetBundle


class Task(str, Enum):
    GENERATION = "generation"
    IMPUTATION = "imputation"
    PREDICTION = "prediction"


@dataclass
class MethodContext:
    """Everything a method needs to run one phase on one dataset."""

    bundle: DatasetBundle
    config: dict                    # merged method config (config.toml + overrides)
    checkpoint_dir: Path            # checkpoints/<method>/<dataset>/train_seed<train_seed>
    train_seed: int = 0
    seed: int = 0                   # inference seed (sampling / imputing / predicting)
    device: Optional[str] = None    # e.g. "cuda:0"; None = auto


@dataclass
class TrainContext(MethodContext):
    force_retrain: bool = False


@dataclass
class GenerateContext(MethodContext):
    n_samples: Optional[int] = None  # None -> len(bundle.train)


@dataclass
class ImputeContext(MethodContext):
    # test frame with NaNs at masked cells; mask is a bool frame aligned with
    # bundle.test (True = cell was masked and must be imputed).
    masked_test: pd.DataFrame = None
    mask: pd.DataFrame = None
    trials: int = 1


@dataclass
class PredictContext(MethodContext):
    trials: int = 1
    # Regression sample budget for methods that expose a native p(y|x).
    n_samples: int = 100
    # Quantile grid requested from methods that expose native quantiles.
    quantile_levels: tuple = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


@dataclass
class ImputationOutput:
    """One or more *jointly sampled* completions of the same masked frame.

    ``draws`` holds ``K`` full test-shaped frames. Keeping the frames whole (as
    opposed to averaging them into one) is what makes the row-wise multiple
    imputation metrics — Energy Score, joint coverage, Rubin's rules — well
    defined: they need the dependence structure *within* a completed row, which
    a per-cell average destroys.

    ``K == 1`` is the ordinary deterministic-imputer case.
    """

    draws: list  # list[pd.DataFrame], all with identical columns and row order
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.draws:
            raise ValueError("ImputationOutput needs at least one draw.")
        reference = list(self.draws[0].columns)
        for i, frame in enumerate(self.draws):
            if list(frame.columns) != reference:
                raise ValueError(f"Draw {i} has different columns than draw 0.")
            if len(frame) != len(self.draws[0]):
                raise ValueError(f"Draw {i} has {len(frame)} rows, draw 0 has {len(self.draws[0])}.")

    @property
    def n_draws(self) -> int:
        return len(self.draws)

    @property
    def columns(self) -> list[str]:
        return list(self.draws[0].columns)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, metadata: dict | None = None) -> "ImputationOutput":
        return cls(draws=[frame.reset_index(drop=True)], metadata=dict(metadata or {}))

    @classmethod
    def from_frames(cls, frames, metadata: dict | None = None) -> "ImputationOutput":
        return cls(
            draws=[f.reset_index(drop=True) for f in frames],
            metadata=dict(metadata or {}),
        )

    def point_estimate(self, schema) -> pd.DataFrame:
        """Per-cell mean (numerical) / plurality vote (categorical) across draws.

        Used only by the *point* metrics (NRMSE / F1 / ...). Distributional
        metrics must consume :attr:`draws` directly.
        """
        if self.n_draws == 1:
            return self.draws[0].copy()
        out = self.draws[0].copy()
        numerical = set(schema.numerical_cols)
        for col in out.columns:
            stacked = pd.concat([d[col] for d in self.draws], axis=1)
            if col in numerical:
                out[col] = stacked.apply(pd.to_numeric, errors="coerce").mean(axis=1)
            else:
                out[col] = stacked.mode(axis=1, dropna=True).iloc[:, 0]
        return out


@dataclass
class PredictionOutput:
    """Prediction result in original label space.

    Beyond the point prediction, a method may expose whatever parts of ``p(y|x)``
    it natively represents. Fields left as ``None`` mean "this method has no such
    output"; the runner then reports the dependent metrics as ``skipped`` rather
    than approximating them (no KDE over samples for NLL, no empirical quantiles
    presented as native quantiles).
    """

    y_pred: np.ndarray
    y_score: Optional[np.ndarray] = None      # binary: positive-class probability
    y_proba: Optional[np.ndarray] = None      # (n, k) class probabilities
    labels: Optional[list] = None              # class labels for y_proba columns

    # --- probabilistic regression -----------------------------------------
    y_samples: Optional[np.ndarray] = None    # (n, S) draws from p(y|x)
    quantiles: Optional[np.ndarray] = None    # (n, Q) native quantiles of p(y|x)
    quantile_levels: Optional[list] = None    # length-Q levels matching `quantiles`
    # Native log-density evaluator: log_prob_fn(y) -> (n,) log p(y_i | x_i).
    # Only set by methods with a closed-form conditional density.
    log_prob_fn: Optional[Callable] = None

    # --- classification decision rule -------------------------------------
    threshold: Optional[float] = None
    # "argmax" | "fixed_0.5" | "train_fitted" | "validation_fitted"
    threshold_source: str = "argmax"

    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject malformed distribution outputs at the adapter boundary."""
        n_rows = len(np.asarray(self.y_pred).reshape(-1))
        if self.y_score is not None and len(np.asarray(self.y_score).reshape(-1)) != n_rows:
            raise ValueError("y_score must contain one positive-class probability per row")
        if self.y_proba is not None:
            proba = np.asarray(self.y_proba)
            if proba.ndim not in (1, 2) or proba.shape[0] != n_rows:
                raise ValueError("y_proba must have shape (n_rows,) or (n_rows, n_classes)")
            if proba.ndim == 1:
                if self.labels is not None and len(self.labels) != 2:
                    raise ValueError("one-dimensional y_proba requires exactly two labels")
                proba = np.column_stack((1.0 - proba, proba))
                if self.y_score is None:
                    self.y_score = proba[:, 1]
            if self.labels is not None and proba.shape[1] != len(self.labels):
                raise ValueError("labels must match y_proba columns")
            self.y_proba = proba
        if self.y_samples is not None:
            samples = np.asarray(self.y_samples)
            if samples.ndim != 2 or samples.shape[0] != n_rows:
                raise ValueError("y_samples must have shape (n_rows, n_samples)")
        if self.quantiles is not None:
            quantiles = np.asarray(self.quantiles)
            if quantiles.ndim != 2 or quantiles.shape[0] != n_rows:
                raise ValueError("quantiles must have shape (n_rows, n_levels)")
            if self.quantile_levels is None or quantiles.shape[1] != len(self.quantile_levels):
                raise ValueError("quantile_levels must match the quantiles columns")

    def available_outputs(self) -> set[str]:
        out = {"point"}
        if self.y_proba is not None or self.y_score is not None:
            out.add("proba")
        if self.y_samples is not None:
            out.add("samples")
        if self.quantiles is not None and self.quantile_levels is not None:
            out.add("quantiles")
        if self.log_prob_fn is not None:
            out.add("density")
        return out


class TabularMethod(ABC):
    """Base class for all methods.

    Subclasses set ``name`` / ``tasks`` / ``trainable`` and implement the task
    methods they support. Unsupported tasks raise by default so the registry
    metadata and the implementation cannot silently diverge.
    """

    name: str = ""
    tasks: frozenset = frozenset()
    trainable: bool = True
    # Pickle/joblib checkpoints may embed Python bytecode and package-specific
    # classes. Such methods opt into an ABI fingerprint so an incompatible
    # artifact is rejected before deserialization.
    checkpoint_runtime_packages: tuple[str, ...] = ()
    require_checkpoint_runtime_fingerprint: bool = False

    def __init__(self, config: dict | None = None) -> None:
        self.config = dict(config or {})
        self.checkpoints_root: Path | None = None

    # --- training ---------------------------------------------------------
    def train(self, ctx: TrainContext) -> None:
        if not self.trainable:
            return  # training-free methods: no-op
        raise NotImplementedError(f"{self.name}.train() is not implemented")

    def is_trained(self, ctx: MethodContext) -> bool:
        """Whether a valid, reusable checkpoint exists for this context."""
        if not self.trainable:
            return True
        from . import checkpoints

        return checkpoints.is_checkpoint_valid(
            ctx.checkpoint_dir,
            method=self.name,
            bundle=ctx.bundle,
            config=self.config,
            runtime_packages=self.checkpoint_runtime_packages,
            require_runtime_fingerprint=self.require_checkpoint_runtime_fingerprint,
        )

    # --- tasks ------------------------------------------------------------
    def generate(self, ctx: GenerateContext) -> pd.DataFrame:
        raise NotImplementedError(f"{self.name} does not support generation")

    def impute(self, ctx: ImputeContext) -> pd.DataFrame:
        raise NotImplementedError(f"{self.name} does not support imputation")

    def impute_draws(self, ctx: ImputeContext) -> ImputationOutput:
        """Return ``ctx.trials`` jointly-sampled completions of the same mask.

        The default repeats :meth:`impute` with a per-draw inference seed, which
        is correct for any method whose ``impute`` samples a full frame in one
        pass. Methods with a native multiple-imputation routine (MICE posterior
        draws, diffusion repaint, ...) override this so the joint row structure
        comes from the model rather than from independent restarts.
        """
        draws = []
        for k in range(max(1, int(ctx.trials))):
            draw_ctx = replace(ctx, seed=ctx.seed + k, trials=1)
            draws.append(self.impute(draw_ctx).reset_index(drop=True))
        return ImputationOutput.from_frames(
            draws, metadata={"draw_strategy": "repeated_impute", "n_draws": len(draws)}
        )

    def impute_draws_many(
        self, contexts: list[ImputeContext]
    ) -> list[ImputationOutput]:
        """Complete several mask settings, preserving their input order.

        The default is deliberately conservative and executes the ordinary
        single-setting contract. Methods with reusable conditional state may
        override this hook to share work across a dataset-level setting grid.
        """

        return [self.impute_draws(ctx) for ctx in contexts]

    def predict(self, ctx: PredictContext) -> PredictionOutput:
        raise NotImplementedError(f"{self.name} does not support prediction")

    def evaluation_metadata(self, ctx: MethodContext) -> dict:
        """Auditable method claims consumed by conditional metrics.

        Adapters may override this to expose artifacts such as a differential-
        privacy accountant.  An empty mapping means no claim; runners never
        infer formal guarantees from a method name or config label.
        """
        return {}

    # --- helpers ----------------------------------------------------------
    def supports(self, task: Task | str) -> bool:
        task = Task(task)
        return task in self.tasks

    @property
    def capabilities(self):
        """The registry-declared capabilities of this method."""
        from .registry import get_spec

        return get_spec(self.name).capabilities

    def cfg(self, section: str, key: str, default: Any = None) -> Any:
        return self.config.get(section, {}).get(key, default)

    def pretrained_path(self, filename_key: str) -> str:
        """Resolve a declared offline resource through the shared checkpoint root."""
        from .resources import pretrained_resources, resolve_resource

        resource = next((r for r in pretrained_resources(self.name) if r.key == filename_key), None)
        if resource is None:
            raise KeyError(f"{self.name} has no pretrained resource {filename_key!r}")
        return str(resolve_resource(self.name, self.config, resource, self.checkpoints_root))

    def additional_resource_identities(self) -> list[dict]:
        """External inference code/configs not represented by pretrained descriptors."""
        return []

    def resolve_device(self, ctx: MethodContext) -> str:
        if ctx.device:
            return ctx.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001 — torch-free methods
            return "cpu"

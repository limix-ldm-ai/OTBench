"""Shared training loop for the deep tabular predictors (FT-Transformer, TabM).

Both upstreams ship a ``torch.nn.Module`` and nothing else: the optimiser,
batching, early stopping and encoding are the caller's job. Writing that twice
would mean two chances to differ on the things that decide the numbers — how
features are standardised, where the validation split comes from, when training
stops — so it is written once here and the modules plug into it.

What each subclass supplies is only what genuinely differs: the module, and how
its output tensor becomes logits or a prediction.

Conventions that are deliberate:

* Standardisation and category vocabularies are fitted on the training rows
  only. Categories unseen at fit time map to a dedicated last bucket rather
  than to an arbitrary known category.
* Early stopping uses a split carved out of train. Test is never looked at.
* Classification emits probabilities from the model's own softmax. Regression
  emits a point prediction and nothing else — neither model defines a
  conditional distribution, so no interval or sample output is fabricated.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import checkpoints as ckpt_utils
from .base import (
    MethodContext,
    PredictContext,
    PredictionOutput,
    TabularMethod,
    Task,
    TrainContext,
)
from .data import (
    categorical_feature_columns,
    numerical_feature_columns,
    split_features_target,
)
from .seeding import seed_everything
from .supervised import VALIDATION_FRACTION, _split_indices, _task_type, _train_target_scale
from .upstream import require_torch

_WEIGHTS_FILE = "model.pt"
_META_FILE = "deep_tabular.json"


class DeepTabularMethod(TabularMethod):
    tasks = frozenset({Task.PREDICTION})
    trainable = True

    # ------------------------------------------------------- subclass hooks
    def build_module(self, n_numerical: int, cardinalities: list[int], d_out: int):
        """Return the upstream ``nn.Module``."""
        raise NotImplementedError(f"{self.name} must implement build_module()")

    def reduce_output(self, output):
        """Collapse the module output to ``(batch, d_out)`` at *inference* time.

        Single-prediction models return their output unchanged. Ensemble models
        override this; how the members are combined is a property of the method,
        not of the training loop.
        """
        return output

    def compute_loss(self, output, target, loss_fn, task_type: str):
        """The *training* objective, which is not always the reduced output.

        The default reduces and then scores, which is right for a model with one
        prediction head. It is a separate hook from :meth:`reduce_output`
        because for a deep ensemble the two genuinely differ: reducing first
        would train the members to agree instead of training each of them.
        """
        return loss_fn(_shape(self.reduce_output(output), task_type), target)

    def auxiliary_model_count(self) -> int:
        """Number of logical prediction members represented by the module."""
        return 1

    # ---------------------------------------------------------------- train
    def train(self, ctx: TrainContext) -> None:
        torch = require_torch(self.name)

        seed_everything(ctx.train_seed)
        schema = ctx.bundle.schema
        task_type = _task_type(schema)
        frame = ctx.bundle.train.reset_index(drop=True)
        split = split_features_target(frame, schema)

        y_raw = split.y
        if task_type == "classification":
            classes = sorted(y_raw.dropna().astype(str).unique())
            codes = pd.Series(y_raw.astype(str)).map({c: i for i, c in enumerate(classes)})
            keep = codes.notna()
            targets = codes[keep].to_numpy(np.int64)
        else:
            classes = None
            values = pd.to_numeric(y_raw, errors="coerce")
            keep = values.notna()
            targets = values[keep].to_numpy(np.float32)

        x = split.X.loc[keep.to_numpy()]
        if len(targets) == 0:
            raise ValueError(f"{self.name}: no training row has a usable target value.")

        codec = _Codec.fit(x, schema)
        numerical, categorical = codec.encode(x)

        train_idx, valid_idx = _split_indices(len(targets), ctx.train_seed)
        d_out = len(classes) if task_type == "classification" else 1
        device = torch.device(self.resolve_device(ctx))
        module = self.build_module(codec.n_numerical, codec.cardinalities, d_out).to(device)

        history = _fit(
            self, torch, module, device, task_type,
            numerical, categorical, targets, train_idx, valid_idx,
        )

        ckpt_dir = Path(ctx.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(module.state_dict(), ckpt_dir / _WEIGHTS_FILE)
        meta = {
            "task_type": task_type,
            "classes": classes,
            "label_vocabulary_source": "train",
            "codec": codec.to_dict(),
            "d_out": d_out,
            "n_train_rows": int(len(train_idx)),
            "n_validation_rows": int(len(valid_idx)),
            "validation_fraction": VALIDATION_FRACTION,
            "train_target_scale": (_train_target_scale(pd.Series(targets))
                                   if task_type == "regression" else None),
            "n_parameters": int(sum(p.numel() for p in module.parameters())),
            "n_auxiliary_models": int(self.auxiliary_model_count()),
            **history,
        }
        with (ckpt_dir / _META_FILE).open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
        ckpt_utils.write_metadata(
            ckpt_dir, self.name, ctx.bundle, self.config, ctx.train_seed,
            artifacts={"weights": _WEIGHTS_FILE, "meta": _META_FILE},
        )

    # -------------------------------------------------------------- predict
    def predict(self, ctx: PredictContext) -> PredictionOutput:
        torch = require_torch(self.name)

        module, meta, codec, device = self._load(ctx, torch)
        schema = ctx.bundle.schema
        x_test = split_features_target(ctx.bundle.test.reset_index(drop=True), schema).X
        numerical, categorical = codec.encode(x_test)

        outputs = []
        batch_size = int(self.cfg("predict", "batch_size", 1024))
        with torch.no_grad():
            for start in range(0, len(x_test), batch_size):
                stop = start + batch_size
                outputs.append(self.reduce_output(module(
                    _tensor(torch, numerical[start:stop], device, torch.float32),
                    _tensor(torch, categorical[start:stop], device, torch.long),
                )).cpu().numpy())
        raw = np.concatenate(outputs, axis=0) if outputs else np.zeros((0, meta["d_out"]))

        if meta["task_type"] == "regression":
            return PredictionOutput(
                y_pred=np.asarray(raw, dtype=float).ravel(),
                metadata={
                    "method": self.name,
                    "predictive_distribution": "none: this model is trained with a "
                                               "point loss and defines no conditional "
                                               "distribution, so CRPS/NLL/interval "
                                               "metrics are skipped, not approximated",
                    "n_parameters": meta["n_parameters"],
                },
            )

        classes = meta["classes"]
        proba = _softmax(raw)
        codes = proba.argmax(axis=1)
        return PredictionOutput(
            y_pred=np.asarray([classes[int(c)] for c in codes], dtype=object),
            y_proba=proba,
            y_score=proba[:, 1] if len(classes) == 2 else None,
            labels=list(classes),
            threshold=0.5 if len(classes) == 2 else None,
            threshold_source="argmax" if len(classes) > 2 else "fixed_0.5",
            metadata={
                "method": self.name,
                "label_vocabulary_source": "train",
                "n_classes": len(classes),
                "n_parameters": meta["n_parameters"],
            },
        )

    # ------------------------------------------------------------ internals
    def _load(self, ctx: MethodContext, torch):
        ckpt_dir = Path(ctx.checkpoint_dir)
        with (ckpt_dir / _META_FILE).open("r", encoding="utf-8") as f:
            meta = json.load(f)
        codec = _Codec.from_dict(meta["codec"])
        device = torch.device(self.resolve_device(ctx))
        module = self.build_module(codec.n_numerical, codec.cardinalities, meta["d_out"])
        module.load_state_dict(torch.load(ckpt_dir / _WEIGHTS_FILE, map_location=device))
        module.to(device).eval()
        return module, meta, codec, device


def _fit(method, torch, module, device, task_type, numerical, categorical, targets,
         train_idx, valid_idx) -> dict:
    """AdamW with early stopping on a validation split carved out of train."""
    from torch import nn

    epochs = int(method.cfg("train", "epochs", 100))
    patience = int(method.cfg("train", "patience", 16))
    batch_size = int(method.cfg("train", "batch_size", 256))
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=float(method.cfg("train", "learning_rate", 1e-3)),
        weight_decay=float(method.cfg("train", "weight_decay", 0.0)),
    )
    loss_fn = nn.CrossEntropyLoss() if task_type == "classification" else nn.MSELoss()

    y = torch.as_tensor(targets, device=device)
    if task_type == "regression":
        y = y.float()

    def batches(indices, shuffle):
        order = np.random.permutation(indices) if shuffle else indices
        for start in range(0, len(order), batch_size):
            yield order[start:start + batch_size]

    def evaluate(indices) -> float:
        if len(indices) == 0:
            return float("nan")
        module.eval()
        total = 0.0
        with torch.no_grad():
            for chunk in batches(indices, shuffle=False):
                out = module(
                    _tensor(torch, numerical[chunk], device, torch.float32),
                    _tensor(torch, categorical[chunk], device, torch.long),
                )
                total += float(method.compute_loss(out, y[chunk], loss_fn, task_type)) \
                    * len(chunk)
        return total / len(indices)

    best_loss, best_state, best_epoch, waited = float("inf"), None, -1, 0
    for epoch in range(epochs):
        module.train()
        for chunk in batches(train_idx, shuffle=True):
            optimizer.zero_grad()
            out = module(
                _tensor(torch, numerical[chunk], device, torch.float32),
                _tensor(torch, categorical[chunk], device, torch.long),
            )
            method.compute_loss(out, y[chunk], loss_fn, task_type).backward()
            optimizer.step()

        validation = evaluate(valid_idx)
        if not np.isfinite(validation):
            best_epoch = epoch
            continue
        if validation < best_loss - 1e-6:
            best_loss, best_epoch, waited = validation, epoch, 0
            best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break

    if best_state is not None:
        module.load_state_dict(best_state)
    return {
        "epochs_run": best_epoch + 1,
        "best_validation_loss": None if not np.isfinite(best_loss) else float(best_loss),
        "early_stopping": ("validation split carved out of train" if len(valid_idx)
                           else "disabled: no validation rows available"),
    }


def _shape(output, task_type):
    return output if task_type == "classification" else output.squeeze(-1)


def _tensor(torch, array, device, dtype):
    if array is None or array.shape[1] == 0:
        return None
    return torch.as_tensor(array, device=device, dtype=dtype)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.asarray(logits, dtype=float) - np.max(logits, axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


class _Codec:
    """Train-fitted feature encoding: standardised numericals, coded categoricals."""

    def __init__(self, numerical_columns, categorical_columns, mean, scale, vocabularies):
        self.numerical_columns = list(numerical_columns)
        self.categorical_columns = list(categorical_columns)
        self.mean = np.asarray(mean, dtype=float)
        self.scale = np.asarray(scale, dtype=float)
        self.vocabularies = vocabularies

    @classmethod
    def fit(cls, x: pd.DataFrame, schema) -> "_Codec":
        numerical = [c for c in numerical_feature_columns(schema) if c in x.columns]
        categorical = [c for c in categorical_feature_columns(schema) if c in x.columns]

        values = x[numerical].apply(pd.to_numeric, errors="coerce") if numerical \
            else pd.DataFrame(index=x.index)
        mean = values.mean().to_numpy(float) if numerical else np.zeros(0)
        scale = values.std(ddof=1).to_numpy(float) if numerical else np.zeros(0)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)

        vocabularies = {c: sorted(x[c].dropna().astype(str).unique()) for c in categorical}
        return cls(numerical, categorical, np.nan_to_num(mean), scale, vocabularies)

    @property
    def n_numerical(self) -> int:
        return len(self.numerical_columns)

    @property
    def cardinalities(self) -> list[int]:
        # One extra bucket per column for values not seen during training.
        return [len(self.vocabularies[c]) + 1 for c in self.categorical_columns]

    def encode(self, x: pd.DataFrame):
        if self.numerical_columns:
            values = x[self.numerical_columns].apply(pd.to_numeric, errors="coerce")
            numerical = ((values.to_numpy(float) - self.mean) / self.scale)
            numerical = np.nan_to_num(numerical, nan=0.0).astype(np.float32)
        else:
            numerical = np.zeros((len(x), 0), dtype=np.float32)

        categorical = np.zeros((len(x), len(self.categorical_columns)), dtype=np.int64)
        for j, column in enumerate(self.categorical_columns):
            levels = self.vocabularies[column]
            index = {v: i for i, v in enumerate(levels)}
            unknown = len(levels)
            categorical[:, j] = [index.get(str(v), unknown) for v in x[column]]
        return numerical, categorical

    def to_dict(self) -> dict:
        return {
            "numerical_columns": self.numerical_columns,
            "categorical_columns": self.categorical_columns,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "vocabularies": self.vocabularies,
            "fitted_on": "train",
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "_Codec":
        return cls(payload["numerical_columns"], payload["categorical_columns"],
                   payload["mean"], payload["scale"], payload["vocabularies"])

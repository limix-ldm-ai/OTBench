# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""DiffPuter: EM diffusion imputation (Zhang et al., ICLR 2025).

The denoiser (``MLPDiffusion``/``Precond``), the EDM loss and the RePaint-style
``impute_mask`` sampler are the vendored upstream files and are called as-is.
What this adapter supplies is the EM driver from ``main.py``, which upstream
ships as a script around ``argparse``, ``np.load`` of pre-generated masks and
hard-coded checkpoint paths.

**This method is transductive, and that is intrinsic to it, not a shortcut.**
DiffPuter's contribution is the EM loop: the M-step fits a density on the
*current completion of the data being imputed*, and the E-step re-imputes from
that density. Running it on a dataset whose missing cells all live in the test
block means the test rows are part of the M-step from the second round onward.
The registry keeps it on a separate transductive track for this reason, and its
scores must not be pooled with the inductive imputers.

``em_iterations = 1`` degenerates to upstream's *out-of-sample* branch: the
model fitted on the training block only, applied to held-out rows. That variant
is inductive, and the metadata says which of the two was run.

Encoding follows upstream's ``dataset.py``: numerical columns as they are,
categorical columns as ``(n_levels - 1).bit_length()`` analog bits in
``{0, 1}``, then the whole matrix standardised by ``(x - mean) / std / 2`` with
mean and std over observed cells. One deliberate correction: upstream's
``main.py`` inverts the standardisation on the categorical block only, because
its metric compares numerical columns in standardised space. Producing a frame
means inverting all of it, so this adapter does.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from otbench.methods.common import checkpoints as ckpt_utils
from otbench.methods.common.base import (
    ImputationOutput,
    ImputeContext,
    MethodContext,
    TabularMethod,
    Task,
    TrainContext,
)
from otbench.methods.common.imputer_base import merge_masked_only
from otbench.methods.common.matrix_codec import BlockLayout
from otbench.methods.common.seeding import seed_everything
from otbench.methods.common.upstream import require_torch

UPSTREAM_URL = "https://github.com/hengruizhang98/DiffPuter"

_WEIGHTS_FILE = "diffputer.pt"
_META_FILE = "diffputer.json"


class DiffPuterMethod(TabularMethod):
    name = "diffputer"
    tasks = frozenset({Task.IMPUTATION})
    trainable = True

    # ---------------------------------------------------------------- train
    def train(self, ctx: TrainContext) -> None:
        torch = require_torch(self.name)

        seed_everything(ctx.train_seed)
        device = self.resolve_device(ctx)
        schema = ctx.bundle.schema
        train = ctx.bundle.train.reset_index(drop=True)[schema.all_columns]

        layout = BlockLayout.fit(train, schema, encoding="analog_bits",
                                 bit_values=(0.0, 1.0))
        matrix, observed = layout.encode(train)
        scaler = _Standardizer.fit(matrix, observed)

        model, history = _m_step(self, scaler.apply(matrix), device, torch, warm_start=None)

        ckpt_dir = Path(ctx.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "layout": layout,
            "scaler": scaler,
            "train_matrix": scaler.apply(matrix),
            "hid_dim": int(self.cfg("model", "hid_dim", 1024)),
        }, ckpt_dir / _WEIGHTS_FILE)
        with (ckpt_dir / _META_FILE).open("w", encoding="utf-8") as f:
            json.dump({
                "upstream": UPSTREAM_URL,
                "categorical_encoding": "analog_bits",
                "n_matrix_columns": layout.width,
                "n_parameters": int(sum(p.numel() for p in model.parameters())),
                "n_auxiliary_models": 1,
                "em_round": 0,
                "em_round_scope": "training block only",
                "epochs_run": history["epochs"],
                "final_train_loss": history["loss"],
                "device": device,
            }, f, indent=2, ensure_ascii=False)
        ckpt_utils.write_metadata(
            ckpt_dir, self.name, ctx.bundle, self.config, ctx.train_seed,
            artifacts={"weights": _WEIGHTS_FILE, "meta": _META_FILE},
        )

    # --------------------------------------------------------------- impute
    def impute(self, ctx: ImputeContext) -> pd.DataFrame:
        """Upstream's point estimate: the mean of ``num_trials`` E-step samples."""
        frame, _ = self._run(ctx, mode="mean")
        return frame

    def impute_draws(self, ctx: ImputeContext) -> ImputationOutput:
        draws, info = self._run(ctx, mode="draws")
        return ImputationOutput.from_frames(draws, metadata={
            "source": "native",
            "draw_strategy": "one upstream impute_mask() trajectory per draw; upstream's "
                             "own point estimate averages num_trials of these, which "
                             "would erase the within-row dependence the draw metrics need",
            "n_draws": len(draws),
            **info,
        })

    def is_trained(self, ctx: MethodContext) -> bool:
        return super().is_trained(ctx) and (Path(ctx.checkpoint_dir) / _WEIGHTS_FILE).exists()

    # ------------------------------------------------------------ internals
    def _run(self, ctx: ImputeContext, mode: str):
        torch = require_torch(self.name)

        seed_everything(ctx.seed)
        device = self.resolve_device(ctx)
        payload = torch.load(Path(ctx.checkpoint_dir) / _WEIGHTS_FILE,
                             map_location=device, weights_only=False)
        layout: BlockLayout = payload["layout"]
        scaler: _Standardizer = payload["scaler"]
        train_matrix = np.asarray(payload["train_matrix"], dtype=np.float32)

        model = _build(self, layout.width, device)
        model.load_state_dict(payload["state_dict"])

        masked = ctx.masked_test.reset_index(drop=True)
        matrix, observed = layout.encode(masked[layout.columns])
        data = scaler.apply(matrix)
        missing = ~observed  # upstream's mask convention: 1 = impute this cell

        iterations = max(1, int(self.cfg("em", "em_iterations", 10)))
        num_trials = max(1, int(self.cfg("em", "num_trials", 20)))
        num_steps = int(self.cfg("em", "num_steps", 50))
        rounds: list[dict] = []

        # Rounds 1..n-1 are EM refinements: E-step on the test block, then an
        # M-step over train + the current completion. The final round's E-step
        # is what gets returned, so it is not followed by another M-step.
        for round_index in range(1, iterations):
            completion = _e_step(model, data, missing, num_trials, num_steps, device, torch)
            joined = np.concatenate([train_matrix, completion], axis=0)
            model, history = _m_step(self, joined, device, torch, warm_start=model)
            rounds.append({"round": round_index, "m_step_rows": int(joined.shape[0]),
                           "epochs": history["epochs"], "loss": history["loss"]})

        info = {
            "em_iterations": iterations,
            "learning_mode": "inductive" if iterations == 1 else "transductive",
            "learning_mode_reason":
                "with one iteration the train-fitted density is applied to held-out rows "
                "(upstream's out-of-sample branch); with more, the M-step refits on the "
                "test block's own completion",
            "num_trials": num_trials,
            "num_steps": num_steps,
            "em_rounds": rounds,
        }

        if mode == "mean":
            completion = _e_step(model, data, missing, num_trials, num_steps, device, torch)
            frame = layout.decode(scaler.invert(completion), masked)
            return merge_masked_only(masked, frame, ctx.mask), info

        draws = []
        for _ in range(max(1, int(ctx.trials))):
            sample = _e_step(model, data, missing, 1, num_steps, device, torch)
            frame = layout.decode(scaler.invert(sample), masked)
            draws.append(merge_masked_only(masked, frame, ctx.mask))
        return draws, info


# ------------------------------------------------------------------ upstream
def _build(method: DiffPuterMethod, width: int, device: str):
    from .model.model import MLPDiffusion, Model

    hid_dim = int(method.cfg("model", "hid_dim", 1024))
    denoise_fn = MLPDiffusion(width, hid_dim).to(device)
    # Upstream passes `hid_dim = in_dim` to Model: the argument only reaches
    # Precond.hid_dim and EDMLoss.hid_dim, neither of which uses it.
    return Model(denoise_fn=denoise_fn, hid_dim=width).to(device)


def _m_step(method: DiffPuterMethod, data: np.ndarray, device: str, torch,
            warm_start) -> tuple:
    """Upstream's density-estimation step: Adam + ReduceLROnPlateau + patience.

    Upstream restarts from a fresh network every round and reloads the
    best-loss checkpoint from disk. Warm-starting from the previous round's
    weights is the same objective reached faster; ``em.warm_start = false``
    restores the cold restart.
    """
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    width = data.shape[1]
    if warm_start is not None and bool(method.cfg("em", "warm_start", True)):
        model = warm_start
    else:
        model = _build(method, width, device)

    max_epochs = int(method.cfg("train", "max_epochs", 10000))
    patience_limit = int(method.cfg("train", "patience", 500))
    batch_size = int(method.cfg("train", "batch_size", 4096))
    lr = float(method.cfg("train", "lr", 1e-4))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=50)

    tensor = torch.as_tensor(data, dtype=torch.float32, device=device)
    n = tensor.shape[0]
    generator = np.random.default_rng(0)

    best_loss, best_state, patience, epoch = float("inf"), None, 0, 0
    model.train()
    for epoch in range(max_epochs):
        order = generator.permutation(n)
        running, counted = 0.0, 0
        for start in range(0, n, batch_size):
            rows = order[start:start + batch_size]
            inputs = tensor[rows]
            loss = model(inputs).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * len(rows)
            counted += len(rows)
        current = running / max(counted, 1)
        scheduler.step(current)
        if current < best_loss:
            best_loss = current
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience == patience_limit:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"epochs": epoch + 1, "loss": float(best_loss)}


def _e_step(model, data: np.ndarray, missing: np.ndarray, num_trials: int,
            num_steps: int, device: str, torch) -> np.ndarray:
    """Upstream's imputation step, in standardised space.

    ``num_trials == 1`` returns a single RePaint trajectory; larger values
    return upstream's average of that many, which is the point estimate the
    paper reports.
    """
    from .model.diffusion_utils import impute_mask

    model.eval()
    net = model.denoise_fn_D
    n, width = data.shape

    observed = torch.as_tensor(data, dtype=torch.float32, device=device)
    mask = torch.as_tensor(missing.astype(np.float32), dtype=torch.float32, device=device)
    known = observed * (1.0 - mask)  # upstream zeroes the cells it will impute

    total = torch.zeros_like(known)
    for _ in range(num_trials):
        sample = impute_mask(net, known, mask, n, width, num_steps, device)
        total += sample * mask + known * (1.0 - mask)
    return (total / num_trials).detach().cpu().numpy()


class _Standardizer:
    """Upstream's ``(x - mean) / std / 2``, with mean/std over observed cells.

    The halving is upstream's: it puts the data roughly inside the ``sigma_data
    = 0.5`` that the EDM preconditioning in ``Precond`` assumes.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, matrix: np.ndarray, observed: np.ndarray) -> "_Standardizer":
        weight = observed.astype(np.float64)
        counts = np.maximum(weight.sum(axis=0), 1.0)
        mean = (matrix * weight).sum(axis=0) / counts
        var = (((matrix - mean) ** 2) * weight).sum(axis=0) / counts
        std = np.sqrt(var)
        # A constant column has zero spread; dividing by it would give inf, and
        # upstream never hits this because its datasets have none.
        std[std < 1e-8] = 1.0
        return cls(mean, std)

    def apply(self, matrix: np.ndarray) -> np.ndarray:
        return ((np.asarray(matrix, dtype=np.float64) - self.mean) / self.std / 2.0
                ).astype(np.float32)

    def invert(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray(matrix, dtype=np.float64) * 2.0 * self.std + self.mean

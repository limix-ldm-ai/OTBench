# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""TabCSDI: conditional score-based diffusion imputation (Zheng & Charoenphakdee,
NeurIPS 2022 Table Representation Learning workshop).

The model, the diffusion schedule and the sampler are the vendored upstream
``src/`` and are called, not reimplemented. Upstream ships the *encoding* as a
per-dataset script (``dataset_census_analog.py`` and friends) rather than as a
reusable function, so that part is rebuilt here, following the analog-bits
script the paper reports as its best mixed-type variant:

* a categorical column becomes ``n_levels.bit_length()`` matrix columns holding
  the binary expansion of its category index *starting at 1*, in ``{-1, +1}``,
  which leaves the all-zero pattern free to mean "missing";
* a numerical column is min-max scaled as ``(x - (min - 1)) / (max - min + 1)``,
  with min and max taken over observed **training** cells only. Upstream
  computes them on train indices too, and the ``-1``/``+1`` offsets are theirs:
  they keep the scaled value strictly positive so that a genuine zero and a
  masked-out zero stay distinguishable.

Where the benchmark's protocol meets upstream's tensor layout:

``observed_mask``
    Upstream means "this cell has a ground-truth value". At inference every test
    cell does — the masked ones are exactly what we are scoring — so it is all
    ones and ``target_mask = observed_mask - gt_mask`` picks out the cells to
    impute.
``gt_mask``
    The conditioning set: one where the masked test frame has a value, zero
    where it is NaN.

Training is self-supervised on the complete training frame via upstream's
``get_randmask``, and inference conditions row by row, so the method is
inductive: no test row is seen before ``impute``.
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

UPSTREAM_URL = "https://github.com/pfnet-research/TabCSDI"

_WEIGHTS_FILE = "tabcsdi.pt"
_META_FILE = "tabcsdi.json"


class TabCSDIMethod(TabularMethod):
    name = "tabcsdi"
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
                                 bit_values=(-1.0, 1.0), reserve_missing_code=True)
        matrix, observed = layout.encode(train)
        scaler = _MinMaxScaler.fit(matrix, observed, layout.numerical_slices())

        config = _upstream_config(self)
        model = _build(config, device)
        history = _fit(self, model, scaler.apply(matrix), observed, device, torch)

        ckpt_dir = Path(ctx.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "layout": layout,
            "scaler": scaler,
            "config": config,
        }, ckpt_dir / _WEIGHTS_FILE)
        with (ckpt_dir / _META_FILE).open("w", encoding="utf-8") as f:
            json.dump({
                "upstream": UPSTREAM_URL,
                "categorical_encoding": "analog_bits",
                "categorical_encoding_reason":
                    "the variant the paper reports as best for mixed-type tables "
                    "(exe_census_analog.py)",
                "n_matrix_columns": layout.width,
                "n_parameters": int(sum(p.numel() for p in model.parameters())),
                "n_auxiliary_models": 1,
                "epochs_run": history["epochs"],
                "final_train_loss": history["loss"],
                "diffusion_steps": config["diffusion"]["num_steps"],
                "device": device,
            }, f, indent=2, ensure_ascii=False)
        ckpt_utils.write_metadata(
            ckpt_dir, self.name, ctx.bundle, self.config, ctx.train_seed,
            artifacts={"weights": _WEIGHTS_FILE, "meta": _META_FILE},
        )

    # --------------------------------------------------------------- impute
    def impute(self, ctx: ImputeContext) -> pd.DataFrame:
        return self.impute_draws(ctx).point_estimate(ctx.bundle.schema)

    def impute_draws(self, ctx: ImputeContext) -> ImputationOutput:
        torch = require_torch(self.name)

        seed_everything(ctx.seed)
        device = self.resolve_device(ctx)
        payload = torch.load(Path(ctx.checkpoint_dir) / _WEIGHTS_FILE,
                             map_location=device, weights_only=False)
        layout: BlockLayout = payload["layout"]
        scaler: _MinMaxScaler = payload["scaler"]

        model = _build(payload["config"], device)
        model.load_state_dict(payload["state_dict"])
        model.eval()

        masked = ctx.masked_test.reset_index(drop=True)
        matrix, observed = layout.encode(masked[layout.columns])
        data = scaler.apply(matrix)
        # A cell is conditioned on when the masked frame still holds its value.
        # `observed` is per-matrix-column, which is what the model indexes.
        k = max(1, int(ctx.trials))

        samples = _sample(model, data, observed, k, int(self.cfg("impute", "batch_size", 64)),
                          device, torch)  # (n_rows, k, width)

        draws = []
        for i in range(k):
            completed = scaler.invert(samples[:, i, :])
            draws.append(merge_masked_only(masked, layout.decode(completed, masked), ctx.mask))
        return ImputationOutput.from_frames(draws, metadata={
            "source": "native",
            "draw_strategy": "upstream CSDI_base.impute(n_samples=K); each draw is one "
                             "full reverse-diffusion trajectory, so a row's cells are "
                             "sampled jointly",
            "n_draws": len(draws),
        })

    def is_trained(self, ctx: MethodContext) -> bool:
        return super().is_trained(ctx) and (Path(ctx.checkpoint_dir) / _WEIGHTS_FILE).exists()


# ------------------------------------------------------------------ upstream
def _upstream_config(method: TabCSDIMethod) -> dict:
    """The nested dict upstream's ``CSDI_base`` reads, from config.toml."""
    return {
        "train": {
            "epochs": int(method.cfg("train", "epochs", 250)),
            "batch_size": int(method.cfg("train", "batch_size", 64)),
            "lr": float(method.cfg("train", "lr", 5e-4)),
        },
        "diffusion": {
            "layers": int(method.cfg("diffusion", "layers", 4)),
            "channels": int(method.cfg("diffusion", "channels", 64)),
            "nheads": int(method.cfg("diffusion", "nheads", 4)),
            "diffusion_embedding_dim": int(method.cfg("diffusion", "embedding_dim", 128)),
            "beta_start": float(method.cfg("diffusion", "beta_start", 1e-4)),
            "beta_end": float(method.cfg("diffusion", "beta_end", 0.5)),
            "num_steps": int(method.cfg("diffusion", "num_steps", 50)),
            "schedule": str(method.cfg("diffusion", "schedule", "quad")),
            "is_unconditional": 0,
            # "mixed" selects upstream's feature-tokenization variant, which
            # lives in main_model_table_ft.py and needs a token embedding per
            # column. The analog-bits variant used here is the non-mixed path.
            "mixed": False,
        },
        "model": {
            "is_unconditional": 0,
            "timeemb": int(method.cfg("model", "time_embedding_dim", 128)),
            "featureemb": int(method.cfg("model", "feature_embedding_dim", 8)),
            "target_strategy": "random",
            "mixed": False,
        },
    }


def _build(config: dict, device: str):
    from .model.main_model_table import TabCSDI

    # CSDI_base mutates config["diffusion"] (it writes side_dim into it), so it
    # gets a copy; the checkpointed dict must stay the one we can rebuild from.
    spec = {k: dict(v) for k, v in config.items()}
    return TabCSDI(spec, device).to(device)


def _batch(data: np.ndarray, observed: np.ndarray, gt: np.ndarray, torch, device):
    """The dict upstream's ``process_data`` destructures.

    ``timepoints`` indexes matrix columns: upstream folds the feature axis into
    CSDI's time axis (``K = 1``) and gets positional information from the
    sinusoidal time embedding, which is why feature count is not baked into the
    architecture.
    """
    n, width = data.shape
    tensor = lambda a, dtype: torch.as_tensor(a, dtype=dtype, device=device)  # noqa: E731
    return {
        "observed_data": tensor(data, torch.float32),
        "observed_mask": tensor(observed.astype(np.float32), torch.float32),
        "gt_mask": tensor(gt.astype(np.float32), torch.float32),
        "timepoints": tensor(np.tile(np.arange(width), (n, 1)), torch.float32),
    }


def _fit(method: TabCSDIMethod, model, data: np.ndarray, observed: np.ndarray,
         device: str, torch) -> dict:
    """Upstream ``utils_table.train``, without its file I/O and tqdm.

    Upstream's optimiser, weight decay and the four-milestone decay schedule are
    reproduced exactly; the validation branch is dropped because it only prints.
    """
    from torch.optim import Adam

    epochs = int(method.cfg("train", "epochs", 250))
    batch_size = int(method.cfg("train", "batch_size", 64))
    lr = float(method.cfg("train", "lr", 5e-4))

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    milestones = [int(f * epochs) for f in (0.25, 0.5, 0.75, 0.9)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    n = data.shape[0]
    generator = np.random.default_rng(0)
    average = float("nan")
    for _ in range(epochs):
        model.train()
        order = generator.permutation(n)
        running, steps = 0.0, 0
        for start in range(0, n, batch_size):
            rows = order[start:start + batch_size]
            if len(rows) < 2:
                continue  # the model has BatchNorm-free but LayerNorm-ful blocks
            optimizer.zero_grad()
            loss = model(_batch(data[rows], observed[rows], observed[rows], torch, device))
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            steps += 1
        scheduler.step()
        average = running / max(steps, 1)
    return {"epochs": epochs, "loss": float(average)}


def _sample(model, data: np.ndarray, observed: np.ndarray, k: int, batch_size: int,
            device: str, torch) -> np.ndarray:
    """Run upstream ``evaluate`` and return ``(n_rows, k, width)`` in scaled space."""
    n, width = data.shape
    ones = np.ones_like(observed, dtype=bool)
    out = np.empty((n, k, width), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            batch = _batch(data[start:stop], ones[start:stop], observed[start:stop],
                           torch, device)
            samples, _, _, _, _ = model.evaluate(batch, k)  # (B, k, K=1, width)
            out[start:stop] = samples.squeeze(2).cpu().numpy()
    return out


class _MinMaxScaler:
    """Upstream's ``(x - (min - 1)) / (max - min + 1)`` over numerical columns only.

    Analog-bit columns are already in ``{-1, +1}`` and upstream leaves them
    alone; scaling them would destroy the codebook the decoder matches against.
    """

    def __init__(self, minimum: np.ndarray, maximum: np.ndarray, columns: list[int]) -> None:
        self.minimum = minimum
        self.maximum = maximum
        self.columns = list(columns)

    @classmethod
    def fit(cls, matrix: np.ndarray, observed: np.ndarray, columns: list[int]) -> "_MinMaxScaler":
        values = np.where(observed, matrix, np.nan)[:, columns]
        with np.errstate(invalid="ignore"):
            minimum = np.nan_to_num(np.nanmin(values, axis=0), nan=0.0)
            maximum = np.nan_to_num(np.nanmax(values, axis=0), nan=0.0)
        return cls(minimum.astype(float), maximum.astype(float), columns)

    @property
    def _span(self) -> np.ndarray:
        return self.maximum - self.minimum + 1.0

    def apply(self, matrix: np.ndarray) -> np.ndarray:
        out = np.asarray(matrix, dtype=np.float32).copy()
        out[:, self.columns] = (out[:, self.columns] - (self.minimum - 1.0)) / self._span
        return out

    def invert(self, matrix: np.ndarray) -> np.ndarray:
        out = np.asarray(matrix, dtype=np.float64).copy()
        out[:, self.columns] = out[:, self.columns] * self._span + (self.minimum - 1.0)
        return out

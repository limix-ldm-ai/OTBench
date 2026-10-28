# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""TabSyn adapter: original VAE + latent EDM diffusion behind the common interface.

- train: original two-stage pipeline (VAE, then latent diffusion) on the
  train_view layout (its "test" split is the val subset, so VAE best-epoch
  selection never sees the benchmark test).
- generation: original EDM sampler + decoder + inverse transforms.
- imputation: original RePaint-style latent ``sample_impute`` (the vendored
  version already takes a row x token cell-level mask matrix). Masked cells
  are blanked before encoding (train-mean / frequency-sampled category), as in
  the original ``impute.py``.
- prediction: mask the target token for every row (original predict-y
  protocol), aggregate over trials.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from otbench.methods.common import checkpoints as ckpt_utils
from otbench.methods.common import data as data_utils
from otbench.methods.common.base import (
    GenerateContext,
    ImputeContext,
    MethodContext,
    PredictContext,
    PredictionOutput,
    TabularMethod,
    Task,
    TrainContext,
)
from otbench.methods.common.seeding import seed_everything
from otbench.methods.common.vendor_data import ensure_vendor_data, load_info, train_view_dir

_ARTIFACTS = {
    "vae_model": "vae/model.pt",
    "vae_encoder": "vae/encoder.pt",
    "vae_decoder": "vae/decoder.pt",
    "train_z": "vae/train_z.npy",
    "diffusion": "model.pt",
}


class TabSynMethod(TabularMethod):
    name = "tabsyn"
    tasks = frozenset({Task.GENERATION, Task.IMPUTATION, Task.PREDICTION})
    trainable = True

    # ------------------------------------------------------------------ train
    def train(self, ctx: TrainContext) -> None:
        from .model.pipeline import train_diffusion, train_vae

        data_dir = ensure_vendor_data(ctx.bundle)
        info = load_info(data_dir)
        device = self.resolve_device(ctx)
        ckpt_dir = Path(ctx.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        seed_everything(ctx.train_seed)
        train_vae(
            data_dir=str(train_view_dir(data_dir)),
            task_type=info["task_type"],
            ckpt_dir=str(ckpt_dir / "vae"),
            device=device,
            num_epochs=int(self.cfg("train", "vae_epochs", 4000)),
            lr=float(self.cfg("train", "vae_lr", 1e-3)),
            wd=float(self.cfg("train", "vae_wd", 0.0)),
            batch_size=int(self.cfg("train", "batch_size", 4096)),
            max_beta=float(self.cfg("train", "max_beta", 1e-2)),
            min_beta=float(self.cfg("train", "min_beta", 1e-5)),
            lambd=float(self.cfg("train", "lambd", 0.7)),
            num_layers=int(self.cfg("model", "num_layers", 2)),
            d_token=int(self.cfg("model", "d_token", 4)),
            n_head=int(self.cfg("model", "n_head", 1)),
            factor=int(self.cfg("model", "factor", 32)),
        )
        seed_everything(ctx.train_seed)
        train_diffusion(
            vae_ckpt_dir=str(ckpt_dir / "vae"),
            ckpt_dir=str(ckpt_dir),
            device=device,
            num_epochs=int(self.cfg("train", "diff_epochs", 10001)),
            lr=float(self.cfg("train", "diff_lr", 1e-3)),
            wd=float(self.cfg("train", "diff_wd", 0.0)),
            batch_size=int(self.cfg("train", "batch_size", 4096)),
            dim_t=int(self.cfg("model", "dim_t", 1024)),
            early_stop_patience=int(self.cfg("train", "early_stop_patience", 500)),
        )
        ckpt_utils.write_metadata(
            ckpt_dir, self.name, ctx.bundle, self.config, ctx.train_seed, artifacts=_ARTIFACTS
        )

    # --------------------------------------------------------------- generate
    def generate(self, ctx: GenerateContext) -> pd.DataFrame:
        from .model.diffusion_utils import sample as edm_sample

        state = self._load_state(ctx)
        seed_everything(ctx.seed)
        n_samples = int(ctx.n_samples or len(ctx.bundle.train))

        x_next = edm_sample(
            state["net"],
            n_samples,
            state["in_dim"],
            num_steps=int(self.cfg("sample", "num_steps", 50)),
            device=state["device"],
        )
        x_next = x_next * 2 + state["mean"].to(state["device"])
        syn_data = x_next.float().cpu().numpy()

        syn_num, syn_cat, syn_target, _ = self._decode(syn_data, state)
        return self._recover_frame(syn_num, syn_cat, syn_target, state["info"])

    # ----------------------------------------------------------------- impute
    def impute(self, ctx: ImputeContext) -> pd.DataFrame:
        from .model.diffusion_utils import sample_impute

        state = self._load_state(ctx)
        seed_everything(ctx.seed)
        info = state["info"]

        x_test_num, x_test_cat = self._blank_masked_inputs(state, ctx.mask, mask_target=False)
        x_known = self._encode_test(state, x_test_num, x_test_cat)
        mask_matrix = self._token_mask(ctx.mask, info, state, mask_target=False)

        x_t = sample_impute(
            state["net"],
            x_known,
            mask_matrix,
            state["token_dim"],
            num_steps=int(self.cfg("impute", "num_steps", 50)),
            resample_rounds=int(self.cfg("impute", "resample_rounds", 20)),
            device=state["device"],
        )
        x_t = x_t * 2 + state["mean"].to(state["device"])
        syn_data = x_t.float().cpu().numpy()
        syn_num, syn_cat, _, _ = self._decode(syn_data, state)

        num_cols = [info["column_names"][i] for i in info["num_col_idx"]]
        cat_cols = [info["column_names"][i] for i in info["cat_col_idx"]]
        result = ctx.masked_test.reset_index(drop=True).copy()
        mask = ctx.mask.reset_index(drop=True)
        for i, col in enumerate(num_cols):
            if col in mask.columns and mask[col].any():
                rows = mask[col].to_numpy(dtype=bool)
                result.loc[rows, col] = np.asarray(syn_num[:, i], dtype=np.float64)[rows]
        for j, col in enumerate(cat_cols):
            if col in mask.columns and mask[col].any():
                rows = mask[col].to_numpy(dtype=bool)
                result.loc[rows, col] = np.asarray(syn_cat[:, j]).astype(str)[rows]
        return data_utils.append_target(result, ctx.bundle.test, ctx.bundle.schema)

    # ---------------------------------------------------------------- predict
    def predict(self, ctx: PredictContext) -> PredictionOutput:
        from .model.diffusion_utils import sample_impute

        state = self._load_state(ctx)
        info = state["info"]
        is_regression = info["task_type"] == "regression"
        trials = int(ctx.trials or self.cfg("predict", "trials", 5))
        empty_mask = pd.DataFrame(
            False, index=range(len(ctx.bundle.test)), columns=list(ctx.bundle.test.columns)
        )

        y_preds, y_probas = [], []
        for trial in range(trials):
            seed_everything(ctx.seed + trial)
            x_test_num, x_test_cat = self._blank_masked_inputs(state, empty_mask, mask_target=True)
            x_known = self._encode_test(state, x_test_num, x_test_cat)
            mask_matrix = self._token_mask(empty_mask, info, state, mask_target=True)
            x_t = sample_impute(
                state["net"],
                x_known,
                mask_matrix,
                state["token_dim"],
                num_steps=int(self.cfg("impute", "num_steps", 50)),
                resample_rounds=int(self.cfg("impute", "resample_rounds", 20)),
                device=state["device"],
            )
            x_t = x_t * 2 + state["mean"].to(state["device"])
            syn_data = x_t.float().cpu().numpy()
            _, _, syn_target, y_proba = self._decode(syn_data, state, want_probs=not is_regression)
            if is_regression:
                y_preds.append(np.asarray(syn_target, dtype=np.float64).reshape(-1))
            else:
                codes = pd.to_numeric(pd.Series(np.asarray(syn_target).reshape(-1)), errors="coerce")
                y_preds.append(codes.fillna(0).round().astype(int).to_numpy())
                if y_proba is not None:
                    y_probas.append(y_proba)

        if is_regression:
            samples = np.stack(y_preds, axis=1)
            return PredictionOutput(
                y_pred=samples.mean(axis=1), y_samples=samples,
                metadata={"trials": trials, "sample_source": "native diffusion draws"},
            )

        classes = list(info["y_classes"])
        if y_probas:
            proba = np.mean(np.stack(y_probas, axis=0), axis=0)
            proba = proba / np.clip(proba.sum(axis=1, keepdims=True), 1e-12, None)
            codes = proba.argmax(axis=1)
        else:
            stacked = np.stack(y_preds, axis=0)
            codes = np.apply_along_axis(
                lambda v: np.bincount(np.clip(v, 0, len(classes) - 1), minlength=len(classes)).argmax(),
                0,
                stacked,
            )
            proba = None
        codes = np.clip(codes, 0, len(classes) - 1)
        y_pred = np.asarray([classes[c] for c in codes], dtype=object)
        y_score = proba[:, 1] if proba is not None and proba.shape[1] == 2 else None
        return PredictionOutput(
            y_pred=y_pred, y_score=y_score, y_proba=proba, labels=classes,
            metadata={"trials": trials},
        )

    # ---------------------------------------------------------------- helpers
    def _load_state(self, ctx: MethodContext) -> dict:
        import torch

        from .model.model import MLPDiffusion, Model
        from .model.utils_train import preprocess
        from .model.vae.model import Decoder_model, Encoder_model

        data_dir = ensure_vendor_data(ctx.bundle)
        info = load_info(data_dir)
        device = self.resolve_device(ctx)
        ckpt_dir = Path(ctx.checkpoint_dir)

        X_num, X_cat, categories, d_numerical, num_inverse, cat_inverse = preprocess(
            str(data_dir), task_type=info["task_type"], inverse=True
        )
        X_train_num, X_test_num = X_num
        X_train_cat, X_test_cat = X_cat
        X_test_cat, oov_by_index = _sanitize_categorical_codes(
            X_test_cat, X_train_cat, categories,
        )
        category_names = self._category_block_names(info)
        oov_counts = {
            category_names[index] if index < len(category_names) else f"cat_block_{index}": count
            for index, count in oov_by_index.items()
        }
        if oov_counts:
            print(
                "[tabsyn] mapped test-only categorical codes to train-mode fallbacks "
                f"before encoding: {oov_counts}"
            )

        num_layers = int(self.cfg("model", "num_layers", 2))
        d_token = int(self.cfg("model", "d_token", 4))
        n_head = int(self.cfg("model", "n_head", 1))
        factor = int(self.cfg("model", "factor", 32))

        pre_encoder = Encoder_model(num_layers, d_numerical, categories, d_token, n_head=n_head, factor=factor).to(device)
        pre_decoder = Decoder_model(num_layers, d_numerical, categories, d_token, n_head=n_head, factor=factor)
        pre_encoder.load_state_dict(torch.load(ckpt_dir / _ARTIFACTS["vae_encoder"], map_location=device))
        pre_decoder.load_state_dict(torch.load(ckpt_dir / _ARTIFACTS["vae_decoder"], map_location="cpu"))
        pre_encoder.eval()
        pre_decoder.eval()

        train_z = torch.tensor(np.load(ckpt_dir / _ARTIFACTS["train_z"])).float()
        train_z = train_z[:, 1:, :]
        B, num_tokens, token_dim = train_z.size()
        in_dim = num_tokens * token_dim
        train_z = train_z.view(B, in_dim)
        mean = train_z.mean(0)

        denoise_fn = MLPDiffusion(in_dim, int(self.cfg("model", "dim_t", 1024))).to(device)
        model = Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)
        model.load_state_dict(torch.load(ckpt_dir / _ARTIFACTS["diffusion"], map_location=device))
        model.eval()

        return {
            "info": info,
            "device": device,
            "categories": list(categories),
            "d_numerical": int(d_numerical),
            "num_inverse": num_inverse,
            "cat_inverse": cat_inverse,
            "X_train_num": X_train_num,
            "X_train_cat": X_train_cat,
            "X_test_num": X_test_num,
            "X_test_cat": X_test_cat,
            "test_category_oov_counts": oov_counts,
            "pre_encoder": pre_encoder,
            "pre_decoder": pre_decoder,
            "mean": mean,
            "in_dim": in_dim,
            "num_tokens": int(num_tokens),
            "token_dim": int(token_dim),
            "net": model.denoise_fn_D,
        }

    def _block_layout(self, info: dict, state: dict):
        """Column -> block position maps for the [y,x] concat layout."""
        num_cols = [info["column_names"][i] for i in info["num_col_idx"]]
        cat_cols = [info["column_names"][i] for i in info["cat_col_idx"]]
        is_regression = info["task_type"] == "regression"
        num_offset = 1 if is_regression else 0   # y first in the num block
        cat_offset = 0 if is_regression else 1   # y first in the cat block
        return num_cols, cat_cols, num_offset, cat_offset, is_regression

    @staticmethod
    def _category_block_names(info: dict) -> list[str]:
        cat_cols = [info["column_names"][i] for i in info["cat_col_idx"]]
        if info["task_type"] == "regression":
            return cat_cols
        return [info["target_col"], *cat_cols]

    def _blank_masked_inputs(self, state: dict, mask: pd.DataFrame, mask_target: bool):
        """Copy transformed test blocks; blank masked cells before encoding.

        Numerical placeholder: transformed-train column mean; categorical
        placeholder: one frequency-sampled train category per column — the
        original impute.py protocol, applied per masked cell.
        """
        import torch

        info = state["info"]
        num_cols, cat_cols, num_offset, cat_offset, is_regression = self._block_layout(info, state)
        mask = mask.reset_index(drop=True)

        x_num = torch.tensor(np.asarray(state["X_test_num"], dtype=np.float32)).float().clone()
        x_cat = torch.tensor(np.asarray(state["X_test_cat"], dtype=np.int64)).clone()
        X_train_num = np.asarray(state["X_train_num"], dtype=np.float32)
        X_train_cat = np.asarray(state["X_train_cat"], dtype=np.int64)

        def blank_num(col_idx: int, rows: np.ndarray) -> None:
            avg = float(X_train_num[:, col_idx].mean(0))
            x_num[torch.as_tensor(rows), col_idx] = avg

        def blank_cat(col_idx: int, rows: np.ndarray) -> None:
            unique_values, counts = np.unique(X_train_cat[:, col_idx], return_counts=True)
            sampled = np.random.choice(unique_values, size=1, p=counts / counts.sum())
            x_cat[torch.as_tensor(rows), col_idx] = int(sampled[0])

        all_rows = np.ones(len(mask), dtype=bool)
        if mask_target:
            if is_regression:
                blank_num(0, all_rows)
            else:
                blank_cat(0, all_rows)
        for i, col in enumerate(num_cols):
            if col in mask.columns and mask[col].any():
                blank_num(num_offset + i, mask[col].to_numpy(dtype=bool))
        for j, col in enumerate(cat_cols):
            if col in mask.columns and mask[col].any():
                blank_cat(cat_offset + j, mask[col].to_numpy(dtype=bool))
        return x_num, x_cat

    def _encode_test(self, state: dict, x_num, x_cat):
        import torch

        device = state["device"]
        with torch.no_grad():
            z = state["pre_encoder"](x_num.to(device), x_cat.to(device)).detach().cpu().numpy()
        x = torch.tensor(z[:, 1:, :]).view(-1, state["in_dim"])
        return ((x - state["mean"]) / 2).to(device)

    def _token_mask(self, mask: pd.DataFrame, info: dict, state: dict, mask_target: bool) -> np.ndarray:
        num_cols, cat_cols, num_offset, cat_offset, is_regression = self._block_layout(info, state)
        d_num = state["d_numerical"]
        mask = mask.reset_index(drop=True)
        out = np.zeros((len(mask), state["num_tokens"]), dtype=bool)
        if mask_target:
            out[:, 0 if is_regression else d_num] = True
        for i, col in enumerate(num_cols):
            if col in mask.columns:
                out[:, num_offset + i] = mask[col].to_numpy(dtype=bool)
        for j, col in enumerate(cat_cols):
            if col in mask.columns:
                out[:, d_num + cat_offset + j] = mask[col].to_numpy(dtype=bool)
        return out

    def _decode(self, syn_data: np.ndarray, state: dict, want_probs: bool = False):
        """Original split_num_cat_target logic + optional y class probabilities."""
        import torch

        info = state["info"]
        token_dim = state["token_dim"]
        is_regression = info["task_type"] == "regression"

        syn = syn_data.reshape(syn_data.shape[0], -1, token_dim)
        with torch.no_grad():
            x_hat_num, x_hat_cat = state["pre_decoder"](torch.tensor(syn, dtype=torch.float32))

        syn_cat_codes = []
        for pred in x_hat_cat:
            syn_cat_codes.append(pred.argmax(dim=-1))
        syn_num = x_hat_num.cpu().numpy()
        syn_cat = (
            torch.stack(syn_cat_codes).t().cpu().numpy()
            if syn_cat_codes
            else np.zeros((len(syn_num), 0), dtype=np.int64)
        )

        syn_num = state["num_inverse"](syn_num) if syn_num.shape[1] > 0 else syn_num
        syn_cat = state["cat_inverse"](syn_cat) if syn_cat.shape[1] > 0 else syn_cat

        y_proba = None
        if is_regression:
            syn_target = syn_num[:, :1]
            syn_num = syn_num[:, 1:]
        else:
            syn_target = syn_cat[:, :1]
            syn_cat = syn_cat[:, 1:]
            if want_probs and x_hat_cat:
                y_proba = torch.softmax(x_hat_cat[0], dim=-1).cpu().numpy()
        return syn_num, syn_cat, syn_target, y_proba

    def _recover_frame(self, syn_num, syn_cat, syn_target, info: dict) -> pd.DataFrame:
        num_cols = [info["column_names"][i] for i in info["num_col_idx"]]
        cat_cols = [info["column_names"][i] for i in info["cat_col_idx"]]
        frame = pd.DataFrame(index=np.arange(len(syn_target)))
        for i, col in enumerate(num_cols):
            frame[col] = np.asarray(syn_num[:, i], dtype=np.float64)
        for j, col in enumerate(cat_cols):
            frame[col] = np.asarray(syn_cat[:, j]).astype(str)
        frame[info["target_col"]] = self._decode_target(np.asarray(syn_target).reshape(-1), info)
        return frame[info["column_names"]]

    def _decode_target(self, y_values: np.ndarray, info: dict) -> np.ndarray:
        if info["task_type"] == "regression":
            return np.asarray(y_values, dtype=np.float64)
        classes = list(info["y_classes"])
        codes = pd.to_numeric(pd.Series(np.asarray(y_values).reshape(-1)), errors="coerce")
        codes = codes.fillna(0).round().astype(int).clip(0, len(classes) - 1)
        return np.asarray([classes[c] for c in codes], dtype=object)


def _sanitize_categorical_codes(
    test_codes: np.ndarray,
    train_codes: np.ndarray,
    category_sizes,
) -> tuple[np.ndarray, dict[int, int]]:
    """Map test-only ordinal codes to deterministic train-only fallbacks.

    The vendored ordinal transform encodes an unseen test category as exactly
    the number of train categories.  That value is an invalid embedding index.
    Fitting the vocabulary on test would leak evaluation data, so each invalid
    value is replaced only for the encoder input by the train-mode code.  The
    adapter still merges predictions under the mask, leaving observed raw test
    values untouched in the returned frame.
    """
    test = np.asarray(test_codes, dtype=np.int64).copy()
    train = np.asarray(train_codes, dtype=np.int64)
    sizes = [int(size) for size in category_sizes]
    if test.ndim != 2 or train.ndim != 2:
        raise ValueError("TabSyn categorical blocks must be two-dimensional")
    if test.shape[1] != len(sizes) or train.shape[1] != len(sizes):
        raise ValueError(
            "TabSyn categorical block width does not match the fitted embedding sizes"
        )

    counts: dict[int, int] = {}
    for index, size in enumerate(sizes):
        if size <= 0:
            raise ValueError(f"TabSyn categorical block {index} has no train categories")
        valid_train = train[:, index]
        valid_train = valid_train[(valid_train >= 0) & (valid_train < size)]
        fallback = int(np.bincount(valid_train, minlength=size).argmax()) if len(valid_train) else 0
        invalid = (test[:, index] < 0) | (test[:, index] >= size)
        count = int(invalid.sum())
        if count:
            test[invalid, index] = fallback
            counts[index] = count
    return test, counts

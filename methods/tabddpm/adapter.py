# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""TabDDPM adapter: original tab-ddpm implementation behind the common interface.

Protocol notes:
- Trained with ``is_y_cond=False`` (y concatenated into the modelled vector as
  ``[y, x]``) — hard requirement for reusing one checkpoint across generation,
  imputation, and prediction (the original ``scripts/impute.py`` demands it).
- generation: ``GaussianMultinomialDiffusion.sample_all`` + the original
  inverse-transform post-processing from ``scripts/sample.py``.
- imputation: original RePaint-style ``sample_impute`` with the mask
  generalized from batch-shared column indices to per-cell mask matrices.
- prediction: mask the target column for every row (the original predict-y
  protocol) and aggregate over stochastic trials.
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

_ARTIFACTS = {"model": "model.pt", "model_ema": "model_ema.pt", "loss": "loss.csv"}


class TabDDPMMethod(TabularMethod):
    name = "tabddpm"
    tasks = frozenset({Task.GENERATION, Task.IMPUTATION, Task.PREDICTION})
    trainable = True

    # ------------------------------------------------------------------ setup
    def _prepare(self, ctx: MethodContext):
        data_dir = ensure_vendor_data(ctx.bundle)
        info = load_info(data_dir)
        return data_dir, info

    def _num_feature_count(self, info: dict) -> int:
        return int(info["n_num_features"])

    def _transformations(self, data_dir: Path, info: dict):
        """Build lib.Transformations matching the exported raw data.

        tab-ddpm's ``num_process_nans`` asserts policy is None when the data has
        no NaN and non-None otherwise, so the nan policy is data-dependent.
        Normalization is disabled when there is no numerical column to fit on
        (classification datasets without numerical features).
        """
        from .model import lib

        has_num_dim = self._num_feature_count(info) > 0 or info["task_type"] == "regression"
        normalization = self.cfg("train", "normalization", "quantile") if has_num_dim else None
        return lib.Transformations(
            seed=0,
            normalization=normalization,
            # Fixed by the schema, not selected after inspecting val/test.  The
            # backend treats mean-imputation as a no-op when no NaN is present.
            num_nan_policy="mean" if has_num_dim else None,
            cat_nan_policy=None,
            cat_min_frequency=None,
            cat_encoding=None,
            y_policy="default",
        )

    def _model_params(self, info: dict) -> dict:
        num_classes = int(info["n_classes"]) if info["task_type"] != "regression" else 0
        return {
            "num_classes": num_classes,
            "is_y_cond": False,  # hard-coded: shared-checkpoint contract
            "rtdl_params": {
                "d_layers": list(self.cfg("model", "d_layers", [1024, 2048, 2048, 1024])),
                "dropout": float(self.cfg("model", "dropout", 0.0)),
            },
            "dim_t": int(self.cfg("model", "dim_t", 128)),
        }

    def _make_dataset(self, data_dir: Path, info: dict, T):
        from .model.pipeline.utils_train import make_dataset

        num_classes = int(info["n_classes"]) if info["task_type"] != "regression" else 0
        return make_dataset(
            str(data_dir), T, num_classes=num_classes, is_y_cond=False, change_val=False
        )

    def _build_diffusion(self, D, info: dict, device: str, model_file: str, ctx: MethodContext):
        import torch

        from .model.pipeline.utils_train import get_model
        from .model.tab_ddpm import GaussianMultinomialDiffusion

        model_params = self._model_params(info)
        K = np.array(D.get_category_sizes("train"))
        if len(K) == 0:
            K = np.array([0])
        num_numerical_features = D.X_num["train"].shape[1] if D.X_num is not None else 0
        model_params["d_in"] = int(np.sum(K) + num_numerical_features)

        model = get_model(
            self.cfg("model", "model_type", "mlp"),
            model_params,
            num_numerical_features,
            category_sizes=D.get_category_sizes("train"),
        )
        model.load_state_dict(
            torch.load(Path(ctx.checkpoint_dir) / model_file, map_location="cpu")
        )
        diffusion = GaussianMultinomialDiffusion(
            K,
            num_numerical_features=num_numerical_features,
            denoise_fn=model,
            num_timesteps=int(self.cfg("model", "num_timesteps", 1000)),
            gaussian_loss_type=self.cfg("model", "gaussian_loss_type", "mse"),
            scheduler=self.cfg("model", "scheduler", "cosine"),
            device=torch.device(device),
        )
        diffusion.to(device)
        diffusion.eval()
        return diffusion, num_numerical_features, K

    # ------------------------------------------------------------------ train
    def train(self, ctx: TrainContext) -> None:
        import torch

        from .model.pipeline.train import train as tabddpm_train

        data_dir, _ = self._prepare(ctx)
        training_dir = train_view_dir(data_dir)
        info = load_info(training_dir)
        T = self._transformations(training_dir, info)
        device = self.resolve_device(ctx)
        seed_everything(ctx.train_seed)

        ckpt_dir = Path(ctx.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        tabddpm_train(
            parent_dir=str(ckpt_dir),
            real_data_path=str(training_dir),
            steps=int(self.cfg("train", "steps", 100000)),
            lr=float(self.cfg("train", "lr", 1e-3)),
            weight_decay=float(self.cfg("train", "weight_decay", 5e-4)),
            batch_size=int(self.cfg("train", "batch_size", 4096)),
            model_type=self.cfg("model", "model_type", "mlp"),
            model_params=self._model_params(info),
            num_timesteps=int(self.cfg("model", "num_timesteps", 1000)),
            gaussian_loss_type=self.cfg("model", "gaussian_loss_type", "mse"),
            scheduler=self.cfg("model", "scheduler", "cosine"),
            T_dict=dict(
                seed=T.seed,
                normalization=T.normalization,
                num_nan_policy=T.num_nan_policy,
                cat_nan_policy=T.cat_nan_policy,
                cat_min_frequency=T.cat_min_frequency,
                cat_encoding=T.cat_encoding,
                y_policy=T.y_policy,
            ),
            device=torch.device(device),
            seed=ctx.train_seed,
            change_val=False,
        )
        ckpt_utils.write_metadata(
            ckpt_dir, self.name, ctx.bundle, self.config, ctx.train_seed, artifacts=_ARTIFACTS
        )

    # --------------------------------------------------------------- generate
    def generate(self, ctx: GenerateContext) -> pd.DataFrame:
        import torch

        from .model.lib import round_columns

        data_dir, info = self._prepare(ctx)
        T = self._transformations(data_dir, info)
        D = self._make_dataset(data_dir, info, T)
        device = self.resolve_device(ctx)
        seed_everything(ctx.seed)

        diffusion, d_num, K = self._build_diffusion(D, info, device, _ARTIFACTS["model"], ctx)
        n_samples = int(ctx.n_samples or len(ctx.bundle.train))
        batch_size = int(self.cfg("sample", "batch_size", 10000))

        _, empirical_class_dist = torch.unique(
            torch.from_numpy(D.y["train"]), return_counts=True
        )
        x_gen, y_gen = diffusion.sample_all(
            n_samples, batch_size, empirical_class_dist.float(), ddim=False
        )
        X_gen, y_gen = x_gen.numpy(), y_gen.numpy()

        is_regression = info["task_type"] == "regression"
        num_cols = [info["column_names"][i] for i in info["num_col_idx"]]
        cat_cols = [info["column_names"][i] for i in info["cat_col_idx"]]

        # --- categorical part (ordinal indices -> raw values) ---
        X_cat_feats = None
        if d_num < X_gen.shape[1]:
            X_cat_raw = D.cat_transform.inverse_transform(X_gen[:, d_num:])
            if not is_regression:
                # is_y_cond=False: y is the FIRST categorical column
                y_gen = X_cat_raw[:, 0]
                X_cat_feats = X_cat_raw[:, 1:]
            else:
                X_cat_feats = X_cat_raw

        # --- numerical part (inverse normalization + discrete rounding) ---
        X_num_feats = None
        if d_num > 0:
            X_num_raw = X_gen[:, :d_num]
            if D.num_transform is not None:
                X_num_raw = D.num_transform.inverse_transform(X_num_raw)
            if is_regression:
                # is_y_cond=False: y is the FIRST numerical column
                y_gen = X_num_raw[:, 0]
                X_num_feats = X_num_raw[:, 1:]
            else:
                X_num_feats = X_num_raw
            if X_num_feats.shape[1] > 0:
                X_num_real = np.load(Path(data_dir) / "X_num_train.npy", allow_pickle=True)
                X_num_real = X_num_real.astype(np.float64)
                disc_cols = []
                for col in range(X_num_real.shape[1]):
                    uniq_vals = np.unique(X_num_real[~np.isnan(X_num_real[:, col]), col])
                    if len(uniq_vals) <= 32 and ((uniq_vals - np.round(uniq_vals)) == 0).all():
                        disc_cols.append(col)
                if disc_cols:
                    X_num_feats = round_columns(
                        np.nan_to_num(X_num_real), X_num_feats, disc_cols
                    )

        # --- assemble original-schema frame ---
        frame = pd.DataFrame(index=np.arange(n_samples))
        for i, col in enumerate(num_cols):
            frame[col] = X_num_feats[:, i].astype(np.float64)
        for i, col in enumerate(cat_cols):
            frame[col] = np.asarray(X_cat_feats[:, i]).astype(str)
        frame[info["target_col"]] = self._decode_target(y_gen, info)
        return frame[info["column_names"]]

    # ----------------------------------------------------------------- impute
    def impute(self, ctx: ImputeContext) -> pd.DataFrame:
        data_dir, info = self._prepare(ctx)
        seed_everything(ctx.seed)
        state = self._load_inference_state(ctx, data_dir, info)

        num_mask, cat_mask = self._feature_mask_matrices(ctx.mask, info, mask_target=False)
        sample = self._run_sample_impute(state, num_mask, cat_mask, return_cat_probs=False)
        num_frame, cat_frame, _ = self._split_sample(sample, state, info)

        # copy imputed values only into benchmark-masked cells
        result = ctx.masked_test.reset_index(drop=True).copy()
        mask = ctx.mask.reset_index(drop=True)
        num_cols = [info["column_names"][i] for i in info["num_col_idx"]]
        cat_cols = [info["column_names"][i] for i in info["cat_col_idx"]]
        for col in num_cols:
            if col in mask.columns and mask[col].any():
                rows = mask[col].to_numpy(dtype=bool)
                result.loc[rows, col] = num_frame.loc[rows, col].to_numpy()
        for col in cat_cols:
            if col in mask.columns and mask[col].any():
                rows = mask[col].to_numpy(dtype=bool)
                result.loc[rows, col] = cat_frame.loc[rows, col].astype(str).to_numpy()
        return data_utils.append_target(result, ctx.bundle.test, ctx.bundle.schema)

    # ---------------------------------------------------------------- predict
    def predict(self, ctx: PredictContext) -> PredictionOutput:
        data_dir, info = self._prepare(ctx)
        seed_everything(ctx.seed)
        state = self._load_inference_state(ctx, data_dir, info)

        is_regression = info["task_type"] == "regression"
        trials = int(ctx.trials or self.cfg("predict", "trials", 5))
        n_rows = len(ctx.bundle.test)

        # target column mask: y sits at position 0 of the num (regression) or
        # cat (classification) block — exactly the original predict-y protocol.
        y_preds, y_probas = [], []
        for trial in range(trials):
            seed_everything(ctx.seed + trial)
            if is_regression:
                num_mask = np.zeros((n_rows, state["d_num"]), dtype=bool)
                num_mask[:, 0] = True
                cat_mask = np.zeros((n_rows, len(state["K"])), dtype=bool) if state["has_cat"] else []
                sample = self._run_sample_impute(state, num_mask, cat_mask, return_cat_probs=False)
                _, _, y_raw = self._split_sample(sample, state, info, want_target=True)
                y_preds.append(np.asarray(y_raw, dtype=np.float64))
            else:
                num_mask = (
                    np.zeros((n_rows, state["d_num"]), dtype=bool) if state["d_num"] > 0 else []
                )
                cat_mask = np.zeros((n_rows, len(state["K"])), dtype=bool)
                cat_mask[:, 0] = True
                sample, extra = self._run_sample_impute(
                    state, num_mask, cat_mask, return_cat_probs=True
                )
                _, _, y_codes = self._split_sample(sample, state, info, want_target=True)
                y_preds.append(np.asarray(y_codes, dtype=np.int64))
                cat_probs = extra.get("cat_probs")
                if cat_probs is not None:
                    k0 = int(state["K"][0])
                    y_probas.append(cat_probs.numpy()[:, :k0])

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
                lambda v: np.bincount(v, minlength=len(classes)).argmax(), 0, stacked
            )
            proba = None
        y_pred = np.asarray([classes[c] for c in codes], dtype=object)
        y_score = None
        if proba is not None and proba.shape[1] == 2:
            y_score = proba[:, 1]
        return PredictionOutput(
            y_pred=y_pred, y_score=y_score, y_proba=proba, labels=classes,
            metadata={"trials": trials},
        )

    # ---------------------------------------------------------------- helpers
    def _load_inference_state(self, ctx: MethodContext, data_dir: Path, info: dict) -> dict:
        import torch

        T = self._transformations(data_dir, info)
        D = self._make_dataset(data_dir, info, T)
        device = self.resolve_device(ctx)
        diffusion, d_num, K = self._build_diffusion(D, info, device, _ARTIFACTS["model"], ctx)

        x_num_test = D.X_num["test"] if D.X_num is not None else None
        x_cat_test = D.X_cat["test"] if D.X_cat is not None else None
        return {
            "D": D,
            "diffusion": diffusion,
            "device": device,
            "d_num": d_num,
            "K": K if len(K) and K[0] != 0 else np.array([], dtype=np.int64),
            "has_cat": x_cat_test is not None,
            "x_num_test": (
                torch.from_numpy(x_num_test.astype(np.float32)).to(device)
                if x_num_test is not None
                else None
            ),
            "x_cat_test": (
                torch.from_numpy(x_cat_test.astype(np.int64)).to(device)
                if x_cat_test is not None
                else None
            ),
        }

    def _feature_mask_matrices(self, mask: pd.DataFrame, info: dict, mask_target: bool):
        """Map the benchmark cell mask (original columns) to [y,x]-block matrices."""
        mask = mask.reset_index(drop=True)
        n = len(mask)
        num_cols = [info["column_names"][i] for i in info["num_col_idx"]]
        cat_cols = [info["column_names"][i] for i in info["cat_col_idx"]]
        is_regression = info["task_type"] == "regression"

        def block(cols: list[str], target_first: bool) -> np.ndarray:
            width = len(cols) + (1 if target_first else 0)
            out = np.zeros((n, width), dtype=bool)
            if target_first and mask_target:
                out[:, 0] = True
            offset = 1 if target_first else 0
            for i, col in enumerate(cols):
                if col in mask.columns:
                    out[:, offset + i] = mask[col].to_numpy(dtype=bool)
            return out

        num_mask = block(num_cols, target_first=is_regression)
        cat_mask = block(cat_cols, target_first=not is_regression)
        return num_mask, cat_mask

    def _run_sample_impute(self, state: dict, num_mask, cat_mask, return_cat_probs: bool):
        x_num = state["x_num_test"].clone() if state["x_num_test"] is not None else None
        x_cat = state["x_cat_test"].clone() if state["x_cat_test"] is not None else None
        return state["diffusion"].sample_impute(
            x_num,
            x_cat,
            num_mask if np.size(num_mask) else [],
            cat_mask if np.size(cat_mask) else [],
            resample_rounds=int(self.cfg("impute", "resample_rounds", 1)),
            impute_condition=self.cfg("impute", "impute_condition", "x_t"),
            w_num=float(self.cfg("impute", "w_num", 0.6)),
            w_cat=float(self.cfg("impute", "w_cat", 0.6)),
            return_cat_probs=return_cat_probs,
        )

    def _split_sample(self, sample, state: dict, info: dict, want_target: bool = False):
        """Inverse-transform a sample back to raw feature frames (+ target)."""
        D = state["D"]
        d_num = state["d_num"]
        is_regression = info["task_type"] == "regression"
        num_cols = [info["column_names"][i] for i in info["num_col_idx"]]
        cat_cols = [info["column_names"][i] for i in info["cat_col_idx"]]

        arr = sample.numpy() if hasattr(sample, "numpy") else np.asarray(sample)
        X_num_part = arr[:, :d_num]
        X_cat_part = arr[:, d_num:]

        y_raw = None
        num_frame = pd.DataFrame(index=np.arange(len(arr)))
        cat_frame = pd.DataFrame(index=np.arange(len(arr)))

        if d_num > 0:
            X_num_raw = X_num_part
            if D.num_transform is not None:
                X_num_raw = D.num_transform.inverse_transform(X_num_part)
            if is_regression:
                y_raw = X_num_raw[:, 0]
                X_num_raw = X_num_raw[:, 1:]
            for i, col in enumerate(num_cols):
                num_frame[col] = X_num_raw[:, i].astype(np.float64)

        if X_cat_part.shape[1] > 0:
            X_cat_raw = D.cat_transform.inverse_transform(X_cat_part.astype(np.int64))
            if not is_regression:
                y_codes = X_cat_raw[:, 0]
                X_cat_raw = X_cat_raw[:, 1:]
                if want_target:
                    y_raw = pd.to_numeric(pd.Series(y_codes), errors="coerce").fillna(0).astype(int).to_numpy()
            for i, col in enumerate(cat_cols):
                cat_frame[col] = np.asarray(X_cat_raw[:, i]).astype(str)

        return num_frame, cat_frame, y_raw

    def _decode_target(self, y_values: np.ndarray, info: dict) -> np.ndarray:
        if info["task_type"] == "regression":
            return np.asarray(y_values, dtype=np.float64)
        classes = list(info["y_classes"])
        codes = pd.to_numeric(pd.Series(np.asarray(y_values).reshape(-1)), errors="coerce")
        codes = codes.fillna(0).round().astype(int).clip(0, len(classes) - 1)
        return np.asarray([classes[c] for c in codes], dtype=object)

# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""TabDiff adapter: original unified continuous-time diffusion behind the
common interface.

- train: original pipeline on the train_view layout; the main model first,
  then (config ``train.train_y_only``) the y_only guidance model — strictly
  serial, because y_only training reads the main checkpoint's config.pkl to
  infer the learnable-schedule noise parameters of the target column.
- generation: ``sample_all`` + the original split/recover post-processing.
- imputation: original RePaint-style ``sample_impute`` with the mask
  generalized to per-cell matrices; y_only guidance stays OFF here (it is
  target-column-specific by construction).
- prediction: original predict-y protocol (mask the target column for every
  row, with y_only classifier-free guidance when available), aggregated over
  trials. The original only wired binclass; multiclass is handled the same way
  (target is a categorical column).
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


class TabDiffMethod(TabularMethod):
    name = "tabdiff"
    tasks = frozenset({Task.GENERATION, Task.IMPUTATION, Task.PREDICTION})
    trainable = True

    # ---------------------------------------------------------------- config
    def _raw_config(self) -> dict:
        """Map the unified config sections back to the original toml structure."""
        m = self.config.get("model", {})
        t = self.config.get("train", {})
        s = self.config.get("sample", {})

        def g(section, key, default):
            return section.get(key, default)

        return {
            "data": {
                "dequant_dist": g(m, "dequant_dist", "none"),
                "int_dequant_factor": g(m, "int_dequant_factor", 0),
            },
            "unimodmlp_params": {
                "num_layers": int(g(m, "num_layers", 2)),
                "d_token": int(g(m, "d_token", 4)),
                "n_head": int(g(m, "n_head", 1)),
                "factor": int(g(m, "factor", 32)),
                "bias": bool(g(m, "bias", True)),
                "dim_t": int(g(m, "dim_t", 1024)),
                "use_mlp": bool(g(m, "use_mlp", True)),
            },
            "diffusion_params": {
                "num_timesteps": int(g(m, "num_timesteps", 50)),
                "scheduler": g(m, "scheduler", "power_mean"),
                "cat_scheduler": g(m, "cat_scheduler", "log_linear"),
                "noise_dist": g(m, "noise_dist", "uniform_t"),
                "sampler_params": {
                    "stochastic_sampler": bool(g(m, "stochastic_sampler", True)),
                    "second_order_correction": bool(g(m, "second_order_correction", True)),
                },
                "edm_params": {
                    "precond": bool(g(m, "precond", True)),
                    "sigma_data": float(g(m, "sigma_data", 1.0)),
                    "net_conditioning": g(m, "net_conditioning", "sigma"),
                },
                "noise_dist_params": {
                    "P_mean": float(g(m, "P_mean", -1.2)),
                    "P_std": float(g(m, "P_std", 1.2)),
                },
                "noise_schedule_params": {
                    "sigma_min": float(g(m, "sigma_min", 0.002)),
                    "sigma_max": float(g(m, "sigma_max", 80)),
                    "rho": float(g(m, "rho", 7)),
                    "eps_max": float(g(m, "eps_max", 1e-3)),
                    "eps_min": float(g(m, "eps_min", 1e-5)),
                    "rho_init": float(g(m, "rho_init", 7.0)),
                    "rho_offset": float(g(m, "rho_offset", 5.0)),
                    "k_init": float(g(m, "k_init", -6.0)),
                    "k_offset": float(g(m, "k_offset", 1.0)),
                },
            },
            "train": {
                "main": {
                    "steps": int(g(t, "steps", 8000)),
                    "lr": float(g(t, "lr", 1e-3)),
                    "weight_decay": float(g(t, "weight_decay", 0)),
                    "ema_decay": float(g(t, "ema_decay", 0.997)),
                    "batch_size": int(g(t, "batch_size", 4096)),
                    "check_val_every": int(g(t, "check_val_every", 2000)),
                    "lr_scheduler": g(t, "lr_scheduler", "reduce_lr_on_plateau"),
                    "factor": float(g(t, "factor", 0.90)),
                    "reduce_lr_patience": int(g(t, "reduce_lr_patience", 50)),
                    "closs_weight_schedule": g(t, "closs_weight_schedule", "anneal"),
                    "c_lambda": float(g(t, "c_lambda", 1.0)),
                    "d_lambda": float(g(t, "d_lambda", 1.0)),
                },
            },
            "sample": {"batch_size": int(g(s, "batch_size", 10000))},
        }

    def _main_dir(self, ctx: MethodContext) -> Path:
        return Path(ctx.checkpoint_dir) / "main"

    def _y_only_dir(self, ctx: MethodContext) -> Path:
        return Path(ctx.checkpoint_dir) / "y_only"

    # ------------------------------------------------------------------ train
    def train(self, ctx: TrainContext) -> None:
        from .model.pipeline import train_tabdiff

        data_dir = ensure_vendor_data(ctx.bundle)
        view_dir = train_view_dir(data_dir)
        view_info = load_info(view_dir)
        device = self.resolve_device(ctx)
        raw_config = self._raw_config()
        non_learnable = bool(self.cfg("model", "non_learnable_schedule", False))

        ckpt_dir = Path(ctx.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        seed_everything(ctx.train_seed)
        train_tabdiff(
            data_dir=str(view_dir),
            info=view_info,
            ckpt_dir=str(self._main_dir(ctx)),
            result_dir=str(ckpt_dir / "train_eval" / "main"),
            device=device,
            raw_config=raw_config,
            y_only=False,
            non_learnable_schedule=non_learnable,
        )
        artifacts = {"main_config": "main/config.pkl", "main_ema": self._rel_best(ctx, "main")}

        if bool(self.cfg("train", "train_y_only", True)):
            # strictly AFTER the main model: y_only reads main/config.pkl.
            seed_everything(ctx.train_seed)
            train_tabdiff(
                data_dir=str(view_dir),
                info=view_info,
                ckpt_dir=str(self._y_only_dir(ctx)),
                result_dir=str(ckpt_dir / "train_eval" / "y_only"),
                device=device,
                raw_config=raw_config,
                y_only=True,
                non_learnable_schedule=non_learnable,
                main_ckpt_dir=str(self._main_dir(ctx)),
            )
            artifacts["y_only_config"] = "y_only/config.pkl"
            artifacts["y_only_ema"] = self._rel_best(ctx, "y_only")

        ckpt_utils.write_metadata(
            ckpt_dir, self.name, ctx.bundle, self.config, ctx.train_seed, artifacts=artifacts
        )

    def _rel_best(self, ctx: MethodContext, sub: str) -> str:
        from .model.pipeline import find_best_ema_ckpt

        best = find_best_ema_ckpt(str(Path(ctx.checkpoint_dir) / sub))
        return str(Path(best).relative_to(Path(ctx.checkpoint_dir)))

    # --------------------------------------------------------------- generate
    def generate(self, ctx: GenerateContext) -> pd.DataFrame:
        import torch

        from .model.pipeline import build_diffusion
        from .model.tabdiff.trainer import recover_data, split_num_cat_target

        data_dir = ensure_vendor_data(ctx.bundle)
        info = load_info(data_dir)
        device = self.resolve_device(ctx)
        seed_everything(ctx.seed)

        diffusion, train_data, _, raw_config = build_diffusion(
            str(data_dir), info, str(self._main_dir(ctx)), device
        )
        n_samples = int(ctx.n_samples or len(ctx.bundle.train))
        with torch.no_grad():
            syn_data = diffusion.sample_all(
                n_samples, raw_config["sample"]["batch_size"], keep_nan_samples=True
            )

        syn_num, syn_cat, syn_target = split_num_cat_target(
            syn_data, info, train_data.num_inverse, train_data.int_inverse, train_data.cat_inverse
        )
        syn_df = recover_data(syn_num, syn_cat, syn_target, info)
        idx_name_mapping = {int(key): value for key, value in info["idx_name_mapping"].items()}
        syn_df.rename(columns=idx_name_mapping, inplace=True)

        return self._typed_frame(syn_df, info)

    # ----------------------------------------------------------------- impute
    def impute(self, ctx: ImputeContext) -> pd.DataFrame:
        state = self._inference_state(ctx, with_guidance=False)
        seed_everything(ctx.seed)
        info = state["info"]

        num_mask, cat_mask = self._block_masks(ctx.mask, info, mask_target=False)
        syn_df = self._run_impute(state, num_mask, cat_mask)

        num_cols = [info["column_names"][i] for i in info["num_col_idx"]]
        cat_cols = [info["column_names"][i] for i in info["cat_col_idx"]]
        result = ctx.masked_test.reset_index(drop=True).copy()
        mask = ctx.mask.reset_index(drop=True)
        for col in num_cols:
            if col in mask.columns and mask[col].any():
                rows = mask[col].to_numpy(dtype=bool)
                result.loc[rows, col] = pd.to_numeric(syn_df[col], errors="coerce").to_numpy()[rows]
        for col in cat_cols:
            if col in mask.columns and mask[col].any():
                rows = mask[col].to_numpy(dtype=bool)
                result.loc[rows, col] = syn_df[col].astype(str).to_numpy()[rows]
        return data_utils.append_target(result, ctx.bundle.test, ctx.bundle.schema)

    # ---------------------------------------------------------------- predict
    def predict(self, ctx: PredictContext) -> PredictionOutput:
        state = self._inference_state(ctx, with_guidance=True)
        info = state["info"]
        is_regression = info["task_type"] == "regression"
        trials = int(ctx.trials or self.cfg("predict", "trials", 5))
        target_col = info["target_col"]

        y_preds = []
        for trial in range(trials):
            seed_everything(ctx.seed + trial)
            # original predict-y protocol: batch-shared column mask on target.
            num_mask_idx = [0] if is_regression else []
            cat_mask_idx = [] if is_regression else [0]
            syn_df = self._run_impute(state, num_mask_idx, cat_mask_idx)
            if is_regression:
                y_preds.append(pd.to_numeric(syn_df[target_col], errors="coerce").to_numpy(dtype=np.float64))
            else:
                codes = pd.to_numeric(syn_df[target_col], errors="coerce").fillna(0)
                y_preds.append(codes.round().astype(int).to_numpy())

        if is_regression:
            samples = np.stack(y_preds, axis=1)
            return PredictionOutput(
                y_pred=samples.mean(axis=1), y_samples=samples,
                metadata={"trials": trials, "guidance": state["guided"],
                          "sample_source": "native diffusion draws"},
            )

        classes = list(info["y_classes"])
        stacked = np.stack(y_preds, axis=0)
        codes = np.apply_along_axis(
            lambda v: np.bincount(np.clip(v, 0, len(classes) - 1), minlength=len(classes)).argmax(),
            0,
            stacked,
        )
        y_pred = np.asarray([classes[c] for c in codes], dtype=object)
        return PredictionOutput(
            y_pred=y_pred, labels=classes,
            metadata={"trials": trials, "guidance": state["guided"]},
        )

    # ---------------------------------------------------------------- helpers
    def _inference_state(self, ctx: MethodContext, with_guidance: bool) -> dict:
        from .model.pipeline import build_diffusion

        data_dir = ensure_vendor_data(ctx.bundle)
        info = load_info(data_dir)
        device = self.resolve_device(ctx)

        y_only_dir = self._y_only_dir(ctx)
        use_guidance = with_guidance and (y_only_dir / "config.pkl").exists()
        diffusion, train_data, test_data, raw_config = build_diffusion(
            str(data_dir), info, str(self._main_dir(ctx)), device,
            y_only_ckpt_dir=str(y_only_dir) if use_guidance else None,
        )
        return {
            "info": info,
            "device": device,
            "diffusion": diffusion,
            "train_data": train_data,
            "test_data": test_data,
            "raw_config": raw_config,
            "guided": use_guidance,
        }

    def _block_masks(self, mask: pd.DataFrame, info: dict, mask_target: bool):
        """Benchmark cell mask -> per-cell [y,x]-block matrices."""
        mask = mask.reset_index(drop=True)
        n = len(mask)
        num_cols = [info["column_names"][i] for i in info["num_col_idx"]]
        cat_cols = [info["column_names"][i] for i in info["cat_col_idx"]]
        is_regression = info["task_type"] == "regression"

        def block(cols, target_first: bool) -> np.ndarray:
            width = len(cols) + (1 if target_first else 0)
            out = np.zeros((n, width), dtype=bool)
            if target_first and mask_target:
                out[:, 0] = True
            offset = 1 if target_first else 0
            for i, col in enumerate(cols):
                if col in mask.columns:
                    out[:, offset + i] = mask[col].to_numpy(dtype=bool)
            return out

        return block(num_cols, is_regression), block(cat_cols, not is_regression)

    def _run_impute(self, state: dict, num_mask, cat_mask) -> pd.DataFrame:
        """Original test_impute core: placeholders + sample_impute + recover."""
        import torch

        from .model.tabdiff.trainer import recover_data, split_num_cat_target

        info = state["info"]
        diffusion = state["diffusion"]
        train_data, test_data = state["train_data"], state["test_data"]
        device = state["device"]
        d_numerical, categories = train_data.d_numerical, train_data.categories

        X_train = train_data.X
        x_num_train = X_train[:, :d_numerical]

        X_test = test_data.X.clone().to(device)
        x_num_test = X_test[:, :d_numerical]
        x_cat_test = X_test[:, d_numerical:].long()

        cat_tokens = torch.tensor(categories, dtype=x_cat_test.dtype, device=device) \
            if len(categories) else None

        # Apply placeholders to x_0 at masked positions (original protocol:
        # numerical -> train column mean; categorical -> [MASK] token).
        def apply_placeholders(mask_spec, block: str) -> None:
            if isinstance(mask_spec, np.ndarray) and mask_spec.ndim == 2:
                rows_cols = mask_spec
                for col in range(rows_cols.shape[1]):
                    rows = torch.as_tensor(rows_cols[:, col], device=device)
                    if not bool(rows.any()):
                        continue
                    if block == "num":
                        avg = x_num_train[:, [col]].mean(0).to(device)
                        x_num_test[rows, col] = avg[0]
                    else:
                        x_cat_test[rows, col] = cat_tokens[col]
            elif len(mask_spec):
                if block == "num":
                    avg = x_num_train[:, list(mask_spec)].mean(0).to(device)
                    x_num_test[:, list(mask_spec)] = avg
                else:
                    x_cat_test[:, list(mask_spec)] = cat_tokens[list(mask_spec)]

        apply_placeholders(num_mask, "num")
        apply_placeholders(cat_mask, "cat")

        with torch.no_grad():
            syn_data = diffusion.sample_impute(
                x_num_test,
                x_cat_test,
                num_mask if np.size(num_mask) else [],
                cat_mask if np.size(cat_mask) else [],
                resample_rounds=int(self.cfg("impute", "resample_rounds", 1)),
                impute_condition=self.cfg("impute", "impute_condition", "x_t"),
                w_num=float(self.cfg("impute", "w_num", 0.6)),
                w_cat=float(self.cfg("impute", "w_cat", 0.6)),
            )

        # Clamp generated categorical codes into the train vocabulary before the
        # ordinal inverse (the model vocabulary includes the [MASK] token K).
        if len(categories):
            cat_part = syn_data[:, d_numerical:]
            limits = torch.tensor(categories, dtype=cat_part.dtype) - 1
            syn_data[:, d_numerical:] = torch.minimum(
                torch.clamp(cat_part, min=0), limits.unsqueeze(0)
            )

        syn_num, syn_cat, syn_target = split_num_cat_target(
            syn_data, info, train_data.num_inverse, train_data.int_inverse, train_data.cat_inverse
        )
        syn_df = recover_data(syn_num, syn_cat, syn_target, info)
        idx_name_mapping = {int(key): value for key, value in info["idx_name_mapping"].items()}
        syn_df.rename(columns=idx_name_mapping, inplace=True)
        return syn_df

    def _typed_frame(self, syn_df: pd.DataFrame, info: dict) -> pd.DataFrame:
        """Cast a recovered frame to original dtypes and decode the target."""
        num_cols = [info["column_names"][i] for i in info["num_col_idx"]]
        cat_cols = [info["column_names"][i] for i in info["cat_col_idx"]]
        frame = pd.DataFrame(index=syn_df.index)
        for col in num_cols:
            frame[col] = pd.to_numeric(syn_df[col], errors="coerce").astype(np.float64)
        for col in cat_cols:
            frame[col] = syn_df[col].astype(str)
        target = info["target_col"]
        if info["task_type"] == "regression":
            frame[target] = pd.to_numeric(syn_df[target], errors="coerce").astype(np.float64)
        else:
            classes = list(info["y_classes"])
            codes = pd.to_numeric(syn_df[target], errors="coerce").fillna(0)
            codes = codes.round().astype(int).clip(0, len(classes) - 1)
            frame[target] = np.asarray([classes[c] for c in codes], dtype=object)
        return frame[info["column_names"]]

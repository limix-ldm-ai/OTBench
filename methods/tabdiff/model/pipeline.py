"""TabDiff pipeline — faithful port of the original ``tabdiff/main.py``.

``train_tabdiff`` reproduces the original train path (dataset assembly, model
construction, learnable-schedule override, config.pkl caching, Trainer loop)
with argparse/hardcoded paths replaced by arguments. ``build_diffusion`` is the
original test-mode assembly (cached config.pkl + best_ema checkpoint), used by
the adapter for generation/imputation/prediction.

Deviations (documented):
- wandb is replaced by a no-op logger (equivalent to the original
  ``wandb.init(mode='disabled')`` without the wandb dependency).
- y_only training skips the periodic generation eval (its single-column sample
  cannot be scored against the full-table density report).
- if the run is shorter than the original's hard-coded 4000-epoch warmup for
  best-checkpoint tracking, the final EMA weights are saved as the best-ema
  checkpoint so short runs still produce a usable artifact.
"""
from __future__ import annotations

import glob
import os
import pickle
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader

from .tabdiff.metrics import TabMetrics
from .tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
from .tabdiff.modules.main_modules import Model, UniModMLP
from .tabdiff.trainer import Trainer
from .utils_train import TabDiffDataset


class NullLogger:
    """Drop-in for the disabled wandb logger used by the original Trainer."""

    def log(self, *args, **kwargs):
        pass

    def define_metric(self, *args, **kwargs):
        pass


def find_best_ema_ckpt(ckpt_dir: str) -> str:
    paths = glob.glob(os.path.join(ckpt_dir, "best_ema_model*"))
    assert paths, (
        f"Cannot infer ckpt_path from {ckpt_dir}, please make sure that you "
        f"first train a model before testing!"
    )
    return paths[0]


def _build_metrics(data_dir: str, info: dict, device, metric_list) -> TabMetrics:
    dataname = info["name"]
    real_data_path = os.path.join(data_dir, "synthetic", dataname, "real.csv")
    test_data_path = os.path.join(data_dir, "synthetic", dataname, "test.csv")
    val_data_path = os.path.join(data_dir, "synthetic", dataname, "val.csv")
    if not os.path.exists(val_data_path):
        val_data_path = None
    return TabMetrics(real_data_path, test_data_path, val_data_path, info, device, metric_list)


def train_tabdiff(
    data_dir: str,
    info: dict,
    ckpt_dir: str,
    result_dir: str,
    device: str,
    raw_config: dict,
    y_only: bool = False,
    non_learnable_schedule: bool = False,
    main_ckpt_dir: str | None = None,
) -> None:
    """Original main.py train mode with paths/config as arguments."""
    raw_config = deepcopy(raw_config)
    dataname = info["name"]
    batch_size = raw_config["train"]["main"]["batch_size"]

    train_data = TabDiffDataset(
        dataname, data_dir, info, y_only=y_only, isTrain=True,
        dequant_dist=raw_config["data"]["dequant_dist"],
        int_dequant_factor=raw_config["data"]["int_dequant_factor"],
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    d_numerical, categories = train_data.d_numerical, train_data.categories

    val_data = TabDiffDataset(
        dataname, data_dir, info, y_only=y_only, isTrain=False,
        dequant_dist=raw_config["data"]["dequant_dist"],
        int_dequant_factor=raw_config["data"]["int_dequant_factor"],
    )

    metrics = _build_metrics(data_dir, info, device, metric_list=["density"])

    raw_config["unimodmlp_params"]["d_numerical"] = d_numerical
    raw_config["unimodmlp_params"]["categories"] = (categories + 1).tolist()  # add one for the mask category
    if y_only:
        raw_config["unimodmlp_params"]["use_mlp"] = False   # drop the mlp when training the unconditional model
        raw_config["unimodmlp_params"]["dim_t"] = 128       # reduce the size of the mlp
        assert main_ckpt_dir is not None, "y_only training requires the main model checkpoint dir"
        main_model_path = find_best_ema_ckpt(main_ckpt_dir)
        main_model_configs = pickle.load(
            open(os.path.join(os.path.dirname(main_model_path), "config.pkl"), "rb")
        )
        if main_model_configs["diffusion_params"]["scheduler"] == "power_mean_per_column":
            # infer noise params of the target column from the main model config
            from .tabdiff.models.noise_schedule import (
                LogLinearNoise_PerColumn,
                PowerMeanNoise_PerColumn,
            )

            if info["task_type"] == "regression":
                noise_schedule = PowerMeanNoise_PerColumn(
                    num_numerical=main_model_configs["unimodmlp_params"]["d_numerical"],
                    **main_model_configs["diffusion_params"]["noise_schedule_params"],
                )
                raw_config["diffusion_params"]["noise_schedule_params"]["rho"] = noise_schedule.rho()[0].item()
            else:
                noise_schedule = LogLinearNoise_PerColumn(
                    num_categories=len(main_model_configs["unimodmlp_params"]["categories"]),
                    **main_model_configs["diffusion_params"]["noise_schedule_params"],
                )
                raw_config["diffusion_params"]["noise_schedule_params"]["k"] = noise_schedule.k()[0].item()

    backbone = UniModMLP(**raw_config["unimodmlp_params"])
    model = Model(backbone, **raw_config["diffusion_params"]["edm_params"])
    model.to(device)

    if not y_only and not non_learnable_schedule:
        raw_config["diffusion_params"]["scheduler"] = "power_mean_per_column"
        raw_config["diffusion_params"]["cat_scheduler"] = "log_linear_per_column"
    diffusion = UnifiedCtimeDiffusion(
        num_classes=categories,
        num_numerical_features=d_numerical,
        denoise_fn=model,
        y_only_model=None,
        **raw_config["diffusion_params"],
        device=device,
    )
    num_params = sum(p.numel() for p in diffusion.parameters())
    print("The number of parameters = ", num_params)
    diffusion.to(device)
    diffusion.train()

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    raw_config["model_save_path"] = ckpt_dir
    raw_config["result_save_path"] = result_dir

    train_main = dict(raw_config["train"]["main"])
    if y_only:
        # single-column samples cannot be scored by the full-table density
        # report; skip the periodic generation eval for the guidance model.
        train_main["check_val_every"] = train_main["steps"] + 1

    trainer = Trainer(
        diffusion,
        train_loader,
        train_data,
        val_data,
        metrics,
        NullLogger(),
        **train_main,
        sample_batch_size=raw_config["sample"]["batch_size"],
        num_samples_to_generate=None,
        model_save_path=ckpt_dir,
        result_save_path=result_dir,
        device=device,
        ckpt_path=None,
        y_only=y_only,
    )

    with open(os.path.join(ckpt_dir, "config.pkl"), "wb") as f:
        pickle.dump(raw_config, f)
    trainer.run_loop()

    # Short-run fallback: the original tracks best checkpoints only after a
    # hard-coded 4000-epoch warmup; persist the final EMA weights otherwise.
    if not glob.glob(os.path.join(ckpt_dir, "best_ema_model*")):
        state_dicts = {
            "denoise_fn": trainer.ema_model.state_dict(),
            "num_schedule": trainer.ema_num_schedule.state_dict(),
            "cat_schedule": trainer.ema_cat_schedule.state_dict(),
        }
        steps = train_main["steps"]
        torch.save(state_dicts, os.path.join(ckpt_dir, f"best_ema_model_final_{steps}.pt"))


def build_diffusion(
    data_dir: str,
    info: dict,
    ckpt_dir: str,
    device: str,
    y_only_ckpt_dir: str | None = None,
) -> tuple:
    """Original main.py test-mode assembly: cached config.pkl + best_ema ckpt.

    Returns (diffusion, train_data, test_data, raw_config).
    """
    ckpt_path = find_best_ema_ckpt(ckpt_dir)
    with open(os.path.join(os.path.dirname(ckpt_path), "config.pkl"), "rb") as f:
        raw_config = pickle.load(f)

    dataname = info["name"]
    train_data = TabDiffDataset(
        dataname, data_dir, info, y_only=False, isTrain=True,
        dequant_dist=raw_config["data"]["dequant_dist"],
        int_dequant_factor=raw_config["data"]["int_dequant_factor"],
    )
    test_data = TabDiffDataset(
        dataname, data_dir, info, y_only=False, isTrain=False,
        dequant_dist=raw_config["data"]["dequant_dist"],
        int_dequant_factor=raw_config["data"]["int_dequant_factor"],
    )
    d_numerical, categories = train_data.d_numerical, train_data.categories

    backbone = UniModMLP(**raw_config["unimodmlp_params"])
    model = Model(backbone, **raw_config["diffusion_params"]["edm_params"])
    model.to(device)

    # y_only guidance model (imputation/prediction only)
    y_only_model = None
    if y_only_ckpt_dir is not None:
        y_only_model_path = find_best_ema_ckpt(y_only_ckpt_dir)
        with open(os.path.join(os.path.dirname(y_only_model_path), "config.pkl"), "rb") as f:
            y_only_model_config = pickle.load(f)
        y_only_model = UniModMLP(**y_only_model_config["unimodmlp_params"])
        y_only_model = Model(y_only_model, **y_only_model_config["diffusion_params"]["edm_params"])
        y_only_model.to(device)
        state_dicts = torch.load(y_only_model_path, map_location=device)
        y_only_model.load_state_dict(state_dicts["denoise_fn"])

    diffusion = UnifiedCtimeDiffusion(
        num_classes=categories,
        num_numerical_features=d_numerical,
        denoise_fn=model,
        y_only_model=y_only_model,
        **raw_config["diffusion_params"],
        device=device,
    )
    diffusion.to(device)

    state_dicts = torch.load(ckpt_path, map_location=device)
    diffusion._denoise_fn.load_state_dict(state_dicts["denoise_fn"])
    diffusion.num_schedule.load_state_dict(state_dicts["num_schedule"])
    diffusion.cat_schedule.load_state_dict(state_dicts["cat_schedule"])
    diffusion.eval()

    return diffusion, train_data, test_data, raw_config

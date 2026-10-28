# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""TabSyn training pipeline — faithful ports of the original entry scripts.

``train_vae``       <- tabsyn/vae/main.py  (paths/epochs parameterized)
``train_diffusion`` <- tabsyn/main.py      (paths/epochs parameterized)

The training loops, losses, schedulers, beta annealing, and best-model
selection are kept line-for-line; only argparse/hardcoded paths were replaced
by function arguments. VAE model selection uses the layout's "test" split
(callers pass the train_view layout whose "test" is the val subset).
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import MLPDiffusion, Model
from .utils_train import TabularDataset, preprocess
from .vae.model import Decoder_model, Encoder_model, Model_VAE


def _drop_unencodable_rows(X_num, X_cat, categories):
    """Drop validation rows carrying a category code the tokenizer cannot embed.

    Column ``i`` owns embedding rows ``[offset_i, offset_i + categories[i])``, so
    any code at or above ``categories[i]`` reads another column's block.
    """
    if X_cat is None or not categories or X_cat.shape[1] == 0:
        return X_num, X_cat
    keep = (np.asarray(X_cat) < np.asarray(categories)[None, :]).all(axis=1)
    if keep.all():
        return X_num, X_cat
    if not keep.any():
        raise RuntimeError(
            "Every validation row carries a category unseen during training, so "
            "none of them can be embedded. Re-split the dataset so the training "
            "part covers each category, or lower the validation fraction."
        )
    print(
        f"tabsyn: dropping {int((~keep).sum())}/{len(keep)} validation rows with "
        "categories unseen during training (early-stopping signal only)"
    )
    return X_num[keep], X_cat[keep]


def compute_loss(X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z):
    ce_loss_fn = nn.CrossEntropyLoss()
    if X_num.shape[1] > 0:
        mse_loss = (X_num - Recon_X_num).pow(2).mean()
    else:
        # OTBench: datasets without numerical columns — an empty-tensor mean
        # is NaN, which poisons the loss and the best-model selection.
        mse_loss = torch.tensor(0.0, device=X_num.device)
    ce_loss = 0
    acc = 0
    total_num = 0

    for idx, x_cat in enumerate(Recon_X_cat):
        if x_cat is not None:
            ce_loss += ce_loss_fn(x_cat, X_cat[:, idx])
            x_hat = x_cat.argmax(dim=-1)
        acc += (x_hat == X_cat[:, idx]).float().sum()
        total_num += x_hat.shape[0]

    if Recon_X_cat:
        ce_loss /= (idx + 1)
        acc /= total_num
    else:
        # OTBench: datasets without categorical columns — CE/acc are void.
        ce_loss = torch.tensor(0.0, device=X_num.device)
        acc = torch.tensor(1.0, device=X_num.device)

    temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()

    loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
    return mse_loss, ce_loss, loss_kld, acc


def train_vae(
    data_dir: str,
    task_type: str,
    ckpt_dir: str,
    device: str,
    num_epochs: int = 4000,
    lr: float = 1e-3,
    wd: float = 0.0,
    batch_size: int = 4096,
    max_beta: float = 1e-2,
    min_beta: float = 1e-5,
    lambd: float = 0.7,
    num_layers: int = 2,
    d_token: int = 4,
    n_head: int = 1,
    factor: int = 32,
):
    """Original tabsyn/vae/main.py main() with paths/hparams as arguments."""
    os.makedirs(ckpt_dir, exist_ok=True)
    model_save_path = f"{ckpt_dir}/model.pt"
    encoder_save_path = f"{ckpt_dir}/encoder.pt"
    decoder_save_path = f"{ckpt_dir}/decoder.pt"

    X_num, X_cat, categories, d_numerical = preprocess(data_dir, task_type=task_type)

    X_train_num, X_test_num = X_num
    X_train_cat, X_test_cat = X_cat

    # OTBench: a category the training split never showed is encoded as
    # `max_train_code + 1` (src/data.py), but the tokenizer sizes column i's
    # embedding block to exactly the number of codes train did show. That value
    # therefore addresses the *next* column's block — out of range for the last
    # column (device-side assert) and a silent mis-embedding everywhere else.
    # This split only drives early stopping, so drop the affected rows instead
    # of scoring them against the wrong embedding.
    X_test_num, X_test_cat = _drop_unencodable_rows(X_test_num, X_test_cat, categories)

    X_train_num, X_test_num = torch.tensor(X_train_num).float(), torch.tensor(X_test_num).float()
    X_train_cat, X_test_cat = torch.tensor(X_train_cat), torch.tensor(X_test_cat)

    train_data = TabularDataset(X_train_num.float(), X_train_cat)

    X_test_num = X_test_num.float().to(device)
    X_test_cat = X_test_cat.to(device)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)

    model = Model_VAE(num_layers, d_numerical, categories, d_token, n_head=n_head, factor=factor, bias=True)
    model = model.to(device)

    pre_encoder = Encoder_model(num_layers, d_numerical, categories, d_token, n_head=n_head, factor=factor).to(device)
    pre_decoder = Decoder_model(num_layers, d_numerical, categories, d_token, n_head=n_head, factor=factor).to(device)

    pre_encoder.eval()
    pre_decoder.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.95, patience=10)

    best_train_loss = float("inf")
    current_lr = optimizer.param_groups[0]["lr"]
    patience = 0
    beta = max_beta
    start_time = time.time()
    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0
        curr_loss_kl = 0.0
        curr_count = 0

        for batch_num, batch_cat in pbar:
            model.train()
            optimizer.zero_grad()

            batch_num = batch_num.to(device)
            batch_cat = batch_cat.to(device)

            Recon_X_num, Recon_X_cat, mu_z, std_z = model(batch_num, batch_cat)

            loss_mse, loss_ce, loss_kld, train_acc = compute_loss(
                batch_num, batch_cat, Recon_X_num, Recon_X_cat, mu_z, std_z
            )

            loss = loss_mse + loss_ce + beta * loss_kld
            loss.backward()
            optimizer.step()

            batch_length = batch_num.shape[0]
            curr_count += batch_length
            curr_loss_multi += loss_ce.item() * batch_length
            curr_loss_gauss += loss_mse.item() * batch_length
            curr_loss_kl += loss_kld.item() * batch_length

        num_loss = curr_loss_gauss / curr_count
        cat_loss = curr_loss_multi / curr_count
        kl_loss = curr_loss_kl / curr_count

        model.eval()
        with torch.no_grad():
            Recon_X_num, Recon_X_cat, mu_z, std_z = model(X_test_num, X_test_cat)

            val_mse_loss, val_ce_loss, val_kl_loss, val_acc = compute_loss(
                X_test_num, X_test_cat, Recon_X_num, Recon_X_cat, mu_z, std_z
            )
            val_loss = val_mse_loss.item() * 0 + val_ce_loss.item()

            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]["lr"]

            if new_lr != current_lr:
                current_lr = new_lr
                print(f"Learning rate updated: {current_lr}")

            train_loss = val_loss
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                patience = 0
                torch.save(model.state_dict(), model_save_path)
            else:
                patience += 1
                if patience == 10:
                    if beta > min_beta:
                        beta = beta * lambd

        print(
            "epoch: {}, beta = {:.6f}, Train MSE: {:.6f}, Train CE:{:.6f}, Train KL:{:.6f}, "
            "Val MSE:{:.6f}, Val CE:{:.6f}, Train ACC:{:6f}, Val ACC:{:6f}".format(
                epoch, beta, num_loss, cat_loss, kl_loss,
                val_mse_loss.item(), val_ce_loss.item(), train_acc.item(), val_acc.item()
            )
        )

    end_time = time.time()
    print("Training time: {:.4f} mins".format((end_time - start_time) / 60))

    # Saving latent embeddings
    with torch.no_grad():
        model.load_state_dict(torch.load(model_save_path))
        pre_encoder.load_weights(model)
        pre_decoder.load_weights(model)

        torch.save(pre_encoder.state_dict(), encoder_save_path)
        torch.save(pre_decoder.state_dict(), decoder_save_path)

        X_train_num = X_train_num.to(device)
        X_train_cat = X_train_cat.to(device)

        print("Successfully load and save the model!")

        train_z = pre_encoder(X_train_num, X_train_cat).detach().cpu().numpy()

        np.save(f"{ckpt_dir}/train_z.npy", train_z)

        print("Successfully save pretrained embeddings in disk!")


def train_diffusion(
    vae_ckpt_dir: str,
    ckpt_dir: str,
    device: str,
    num_epochs: int = 10001,
    lr: float = 1e-3,
    wd: float = 0.0,
    batch_size: int = 4096,
    dim_t: int = 1024,
    early_stop_patience: int = 500,
):
    """Original tabsyn/main.py main() with paths/hparams as arguments."""
    train_z = torch.tensor(np.load(f"{vae_ckpt_dir}/train_z.npy")).float()
    train_z = train_z[:, 1:, :]
    B, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim
    train_z = train_z.view(B, in_dim)

    os.makedirs(ckpt_dir, exist_ok=True)

    mean, std = train_z.mean(0), train_z.std(0)

    train_z = (train_z - mean) / 2
    train_data = train_z

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)

    denoise_fn = MLPDiffusion(in_dim, dim_t).to(device)

    num_params = sum(p.numel() for p in denoise_fn.parameters())
    print("the number of parameters", num_params)

    model = Model(denoise_fn=denoise_fn, hid_dim=train_z.shape[1]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=20)

    model.train()

    best_loss = float("inf")
    patience = 0
    start_time = time.time()
    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

        batch_loss = 0.0
        len_input = 0
        for batch in pbar:
            inputs = batch.float().to(device)
            loss = model(inputs)

            loss = loss.mean()

            batch_loss += loss.item() * len(inputs)
            len_input += len(inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"Loss": loss.item()})

        curr_loss = batch_loss / len_input
        scheduler.step(curr_loss)

        if curr_loss < best_loss:
            best_loss = curr_loss
            patience = 0
            torch.save(model.state_dict(), f"{ckpt_dir}/model.pt")
        else:
            patience += 1
            if patience == early_stop_patience:
                print("Early stopping")
                break

    end_time = time.time()
    print("Time: ", end_time - start_time)

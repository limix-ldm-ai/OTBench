"""Train-fitted support, coverage, and authenticity curves."""
from __future__ import annotations

import numpy as np
import pandas as pd


def alpha_precision_beta_recall(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
) -> dict[str, float]:
    """Alpha-precision / beta-recall / authenticity (Alaa et al., ICML 2022).

    MinMax-normalise on the real frame, then run the support curves with
    ``emb_center = mean(real)``. Inputs are expected to be **numeric** frames
    (e.g. the one-hot/numeric encoded pair from ``EvalFrames.onehot_pair``).
    """
    x_real = real_df.to_numpy(dtype=float)
    x_syn = syn_df.to_numpy(dtype=float)
    if x_real.shape[1] == 0 or len(x_real) == 0 or len(x_syn) == 0:
        return {"alpha_precision": float("nan"), "beta_recall": float("nan"), "authenticity": float("nan")}

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler().fit(x_real)
    x = scaler.transform(x_real)
    x_s = scaler.transform(x_syn)

    delta_precision_alpha, delta_coverage_beta, authenticity = _support_curves(x, x_s)
    return {
        "alpha_precision": float(delta_precision_alpha),
        "beta_recall": float(delta_coverage_beta),
        "authenticity": float(authenticity),
    }


def _support_curves(x: np.ndarray, x_syn: np.ndarray) -> tuple[float, float, float]:
    """Compute curves around the real-data mean with exact nearest neighbors."""
    from sklearn.neighbors import NearestNeighbors

    emb_center = np.mean(x, axis=0)

    n_steps = 30
    alphas = np.linspace(0, 1, n_steps)

    radii = np.quantile(np.sqrt(np.sum((x - emb_center) ** 2, axis=1)), alphas)
    synth_center = np.mean(x_syn, axis=0)
    synth_to_center = np.sqrt(np.sum((x_syn - emb_center) ** 2, axis=1))

    nbrs_real = NearestNeighbors(n_neighbors=2, n_jobs=1, p=2).fit(x)
    real_to_real, real_to_real_args = nbrs_real.kneighbors(x)

    nbrs_synth = NearestNeighbors(n_neighbors=1, n_jobs=1, p=2).fit(x_syn)
    real_to_synth, real_to_synth_args = nbrs_synth.kneighbors(x)

    # closest real point to each real point, excluding itself (column 1, not 0)
    real_to_real = real_to_real[:, 1].squeeze()
    real_to_real_args = real_to_real_args[:, 1].squeeze()
    real_to_synth = real_to_synth.squeeze()
    real_to_synth_args = real_to_synth_args.squeeze()

    real_synth_closest = x_syn[real_to_synth_args]
    real_synth_closest_d = np.sqrt(np.sum((real_synth_closest - synth_center) ** 2, axis=1))
    closest_synth_radii = np.quantile(real_synth_closest_d, alphas)

    alpha_precision_curve = []
    beta_coverage_curve = []
    for k in range(len(radii)):
        alpha_precision_curve.append(np.mean(synth_to_center <= radii[k]))
        beta_coverage_curve.append(
            np.mean((real_to_synth <= real_to_real) * (real_synth_closest_d <= closest_synth_radii[k]))
        )

    authenticity = float(np.mean(real_to_real[real_to_real_args] < real_to_synth))

    delta_precision_alpha = 1 - np.sum(np.abs(alphas - np.array(alpha_precision_curve))) / np.sum(alphas)
    delta_coverage_beta = 1 - np.sum(np.abs(alphas - np.array(beta_coverage_curve))) / np.sum(alphas)

    return float(delta_precision_alpha), float(delta_coverage_beta), authenticity

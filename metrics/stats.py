"""Shared statistical primitives for every task's metrics.

Deliberately dependency-light: NumPy / SciPy / scikit-learn only. The core
evaluation must not depend on an external synthetic-data evaluation framework,
both because those libraries change definitions between releases and because a
metric-redundancy study needs the formula to be inspectable in one place.

Each function names the definition it implements; where a paper formula exists
it is written out so the implementation can be checked against it.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-12


# --- univariate distances --------------------------------------------------
def ks_statistic(real: np.ndarray, synthetic: np.ndarray) -> float | None:
    """Two-sample Kolmogorov-Smirnov statistic ``sup_x |F_r(x) - F_s(x)|`` in [0, 1]."""
    real = _finite(real)
    synthetic = _finite(synthetic)
    if len(real) == 0 or len(synthetic) == 0:
        return None
    return float(stats.ks_2samp(real, synthetic, method="asymp").statistic)


def total_variation_distance(real: Sequence, synthetic: Sequence, domain=None) -> float | None:
    """``0.5 * sum_c |p_r(c) - p_s(c)|`` over the union of observed categories.

    Values outside ``domain`` (when given) are folded into a single
    ``__UNKNOWN__`` category, so inventing categories costs probability mass
    rather than silently expanding the support.
    """
    p = _category_distribution(real, domain)
    q = _category_distribution(synthetic, domain)
    if p is None or q is None:
        return None
    keys = sorted(set(p) | set(q))
    return float(0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys))


def jensen_shannon_divergence(real: Sequence, synthetic: Sequence, domain=None) -> float | None:
    """Base-2 Jensen-Shannon divergence between two categorical distributions.

    JSD(P||Q) = 0.5 KL(P||M) + 0.5 KL(Q||M), M = (P+Q)/2, in [0, 1] with log2.
    Unlike KL it is symmetric and finite when the supports differ, which is the
    normal situation when comparing generated to real categories.
    """
    p = _category_distribution(real, domain)
    q = _category_distribution(synthetic, domain)
    if p is None or q is None:
        return None
    keys = sorted(set(p) | set(q))
    p_vec = np.array([p.get(k, 0.0) for k in keys], dtype=float)
    q_vec = np.array([q.get(k, 0.0) for k in keys], dtype=float)
    m = 0.5 * (p_vec + q_vec)
    return float(0.5 * _kl(p_vec, m) + 0.5 * _kl(q_vec, m))


def wasserstein1(real: np.ndarray, synthetic: np.ndarray, scale: float | None = None) -> float | None:
    """1-Wasserstein distance, optionally divided by a train-fitted ``scale``."""
    real = _finite(real)
    synthetic = _finite(synthetic)
    if len(real) == 0 or len(synthetic) == 0:
        return None
    value = float(stats.wasserstein_distance(real, synthetic))
    if scale is None:
        return value
    return value / scale if scale > EPS else None


def robust_scale(values: np.ndarray) -> tuple[float, str]:
    """Standard deviation, falling back to IQR then range. Returns (scale, kind).

    The SD is the sample estimate (``ddof=1``), matching ``pandas.Series.std``.
    Every normalised metric in the benchmark routes through here, so the choice
    is made once: an NRMSE from the imputation task and one from the prediction
    task are divided by the same estimator.
    """
    values = _finite(values)
    if len(values) < 2:
        return 0.0, "none"
    std = float(np.std(values, ddof=1))
    if std > EPS:
        return std, "std"
    iqr = float(np.percentile(values, 75) - np.percentile(values, 25))
    if iqr > EPS:
        return iqr, "iqr"
    span = float(np.max(values) - np.min(values))
    return (span, "range") if span > EPS else (0.0, "none")


# --- dependence structure --------------------------------------------------
def cramers_v(a: Sequence, b: Sequence) -> float | None:
    """Cramer's V association between two categorical columns, in [0, 1].

    V = sqrt( chi2 / (n * min(k-1, r-1)) ) -- Cramer (1946).
    """
    table = pd.crosstab(pd.Series(a).astype(str), pd.Series(b).astype(str))
    if table.size == 0 or min(table.shape) < 2:
        return None
    chi2 = float(stats.chi2_contingency(table.to_numpy(), correction=False)[0])
    n = float(table.to_numpy().sum())
    denominator = n * (min(table.shape) - 1)
    if denominator <= EPS:
        return None
    return float(np.sqrt(chi2 / denominator))


def correlation_ratio(categories: Sequence, values: np.ndarray) -> float | None:
    """Correlation ratio eta between a categorical and a numerical column, in [0, 1].

    eta^2 = between-group sum of squares / total sum of squares.
    """
    frame = pd.DataFrame({"c": pd.Series(categories).astype(str).to_numpy(),
                          "v": pd.to_numeric(pd.Series(values), errors="coerce").to_numpy()})
    frame = frame.dropna()
    if frame["c"].nunique() < 2 or len(frame) < 2:
        return None
    grand_mean = frame["v"].mean()
    total_ss = float(((frame["v"] - grand_mean) ** 2).sum())
    if total_ss <= EPS:
        return None
    between_ss = float(
        sum(len(g) * (g["v"].mean() - grand_mean) ** 2 for _, g in frame.groupby("c"))
    )
    return float(np.sqrt(max(0.0, min(1.0, between_ss / total_ss))))


def association_matrix(frame: pd.DataFrame, numerical: Iterable[str], categorical: Iterable[str]) -> pd.DataFrame:
    """Mixed-type pairwise association matrix in [0, 1].

    num-num  : |Pearson r|
    cat-cat  : Cramer's V
    num-cat  : correlation ratio eta

    All three are magnitudes in [0, 1], so the matrix is directly comparable
    across pair types and a Frobenius difference between two such matrices is a
    meaningful "how different is the dependence structure" score.
    """
    numerical = [c for c in numerical if c in frame.columns]
    categorical = [c for c in categorical if c in frame.columns]
    columns = numerical + categorical
    matrix = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns, dtype=float)
    num_set = set(numerical)
    for i, left in enumerate(columns):
        for right in columns[i + 1:]:
            if left in num_set and right in num_set:
                value = _abs_pearson(frame[left], frame[right])
            elif left not in num_set and right not in num_set:
                value = cramers_v(frame[left], frame[right])
            elif left in num_set:
                value = correlation_ratio(frame[right], frame[left])
            else:
                value = correlation_ratio(frame[left], frame[right])
            value = float(value) if value is not None and np.isfinite(value) else np.nan
            matrix.loc[left, right] = value
            matrix.loc[right, left] = value
    return matrix


def matrix_difference(real: pd.DataFrame, synthetic: pd.DataFrame) -> tuple[float | None, float | None, int]:
    """(Frobenius norm, mean absolute difference, number of comparable pairs).

    Only the upper triangle is used, and only entries that are finite in both
    matrices; the pair count is returned so a small difference computed over
    two pairs cannot be mistaken for one computed over two hundred.
    """
    columns = [c for c in real.columns if c in synthetic.columns]
    if len(columns) < 2:
        return None, None, 0
    left = real.loc[columns, columns].to_numpy(dtype=float)
    right = synthetic.loc[columns, columns].to_numpy(dtype=float)
    triangle = np.triu(np.ones_like(left, dtype=bool), k=1)
    valid = triangle & np.isfinite(left) & np.isfinite(right)
    if not valid.any():
        return None, None, 0
    diff = left[valid] - right[valid]
    return float(np.sqrt(np.sum(diff ** 2))), float(np.mean(np.abs(diff))), int(valid.sum())


def relative_frobenius_distance(real: pd.DataFrame, synthetic: pd.DataFrame) -> float | None:
    """Relative Frobenius error on comparable off-diagonal associations."""
    columns = [column for column in real.columns if column in synthetic.columns]
    if len(columns) < 2:
        return None
    left = real.loc[columns, columns].to_numpy(dtype=float)
    right = synthetic.loc[columns, columns].to_numpy(dtype=float)
    triangle = np.triu(np.ones_like(left, dtype=bool), k=1)
    valid = triangle & np.isfinite(left) & np.isfinite(right)
    if not valid.any():
        return None
    denominator = float(np.sqrt(np.sum(left[valid] ** 2)))
    if denominator <= np.finfo(float).eps:
        return None
    return float(np.sqrt(np.sum((left[valid] - right[valid]) ** 2)) / denominator)


# --- proper scoring rules --------------------------------------------------
def crps_ensemble(y_true: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Per-row CRPS of an ensemble forecast (Gneiting & Raftery 2007, eq. 21).

    ``CRPS(F, y) = E|X - y| - 0.5 E|X - X'|``, estimated from ``m`` samples as
    ``(1/m) sum_i |x_i - y| - 1/(2 m^2) sum_i sum_j |x_i - x_j|``.

    The double sum is evaluated in O(m log m) via the sorted-sample identity
    ``sum_i sum_j |x_i - x_j| = 2 * sum_i (2i - m + 1) x_(i)`` (0-based ``i``).
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] != len(y_true):
        raise ValueError("samples must have shape (n_rows, n_samples) matching y_true")
    m = samples.shape[1]
    if m == 0:
        return np.full(len(y_true), np.nan)
    term1 = np.mean(np.abs(samples - y_true[:, None]), axis=1)
    ordered = np.sort(samples, axis=1)
    weights = (2.0 * np.arange(m) - m + 1.0)
    pairwise = 2.0 * np.sum(ordered * weights, axis=1)
    term2 = pairwise / (2.0 * m * m)
    return term1 - term2


def energy_score(y_true: np.ndarray, samples: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Per-row multivariate energy score (Gneiting & Raftery 2007, section 4.3).

    ``ES(F, y) = E||X - y||^beta - 0.5 E||X - X'||^beta``. This is the
    multivariate generalisation of CRPS and is the reason multiple imputation
    draws must stay joint: it scores the *row*, not each cell independently.

    ``y_true`` is ``(n, d)``; ``samples`` is ``(n, m, d)``.
    """
    y_true = np.asarray(y_true, dtype=float)
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 3 or samples.shape[0] != y_true.shape[0]:
        raise ValueError("samples must have shape (n_rows, n_draws, n_dims)")
    m = samples.shape[1]
    if m == 0:
        return np.full(y_true.shape[0], np.nan)
    diff_to_truth = np.linalg.norm(samples - y_true[:, None, :], axis=2) ** beta
    term1 = np.mean(diff_to_truth, axis=1)
    if m == 1:
        return term1
    pairwise = np.linalg.norm(samples[:, :, None, :] - samples[:, None, :, :], axis=3) ** beta
    term2 = 0.5 * pairwise.sum(axis=(1, 2)) / (m * m)
    return term1 - term2


def pinball_loss(y_true: np.ndarray, quantiles: np.ndarray, levels: Sequence[float]) -> np.ndarray:
    """Per-level average pinball (quantile) loss.

    ``rho_tau(u) = max(tau * u, (tau - 1) * u)`` with ``u = y - q_tau``.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1, 1)
    quantiles = np.asarray(quantiles, dtype=float)
    levels_arr = np.asarray(levels, dtype=float).reshape(1, -1)
    residual = y_true - quantiles
    losses = np.maximum(levels_arr * residual, (levels_arr - 1.0) * residual)
    return np.mean(losses, axis=0)


def brier_score(y_true_codes: np.ndarray, proba: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error against the one-hot label.

    ``(1/n) sum_i sum_k (p_ik - [y_i = k])^2`` (Brier 1950; multiclass form of
    Brier as in Gneiting & Raftery 2007). Range [0, 2]; binary probabilities
    give twice the usual binary Brier, so it is reported under an explicit
    multiclass name.
    """
    proba = np.asarray(proba, dtype=float)
    onehot = np.zeros_like(proba)
    rows = np.arange(len(y_true_codes))
    onehot[rows, np.asarray(y_true_codes, dtype=int)] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def expected_calibration_error(
    y_true_codes: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 15,
) -> tuple[float, list[dict]]:
    """Top-label ECE with a fixed number of equal-width confidence bins.

    Naeini, Cooper & Hauskrecht (2015) / Guo et al. (2017):
    ``ECE = sum_b (|B_b| / n) * |acc(B_b) - conf(B_b)|`` over bins of the
    predicted-class confidence. The per-bin reliability data is returned as
    well, because the scalar alone hides which side of the diagonal a model is
    on.
    """
    proba = np.asarray(proba, dtype=float)
    y_true_codes = np.asarray(y_true_codes, dtype=int)
    confidence = proba.max(axis=1)
    predicted = proba.argmax(axis=1)
    correct = (predicted == y_true_codes).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    reliability: list[dict] = []
    n = len(y_true_codes)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (confidence > lo) & (confidence <= hi) if lo > 0 else (confidence <= hi)
        count = int(in_bin.sum())
        if count == 0:
            reliability.append({"lower": float(lo), "upper": float(hi), "count": 0,
                                "accuracy": None, "confidence": None})
            continue
        accuracy = float(correct[in_bin].mean())
        mean_confidence = float(confidence[in_bin].mean())
        ece += (count / n) * abs(accuracy - mean_confidence)
        reliability.append({"lower": float(lo), "upper": float(hi), "count": count,
                            "accuracy": accuracy, "confidence": mean_confidence})
    return float(ece), reliability


def area_under_risk_coverage(losses: np.ndarray, confidence: np.ndarray) -> float:
    """AURC: mean selective risk over all coverage levels (Geifman et al. 2019).

    Rows are sorted by decreasing confidence; the risk at coverage ``k/n`` is the
    mean loss of the ``k`` most confident rows, and AURC averages that over
    ``k = 1..n``. Lower is better: a model whose confidence ranks its own errors
    last scores well even if its accuracy is unchanged.
    """
    losses = np.asarray(losses, dtype=float).reshape(-1)
    confidence = np.asarray(confidence, dtype=float).reshape(-1)
    if len(losses) == 0:
        return float("nan")
    order = np.argsort(-confidence, kind="stable")
    ordered = losses[order]
    cumulative_risk = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(np.mean(cumulative_risk))


# --- interval metrics ------------------------------------------------------
def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """PICP: fraction of true values inside [lower, upper]."""
    y_true = np.asarray(y_true, dtype=float)
    inside = (y_true >= np.asarray(lower, dtype=float)) & (y_true <= np.asarray(upper, dtype=float))
    return float(np.mean(inside))


def interval_width(lower: np.ndarray, upper: np.ndarray, scale: float | None = None) -> float:
    """MPIW, optionally normalised by a train-fitted scale (nMPIW)."""
    width = float(np.mean(np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)))
    if scale is None:
        return width
    return width / scale if scale > EPS else float("nan")


def quantile_crossing_rate(quantiles: np.ndarray) -> float:
    """Fraction of adjacent quantile pairs that are out of order.

    Independently fitted quantile regressions are not guaranteed monotone; a
    non-zero rate invalidates the interval metrics derived from them, so it is
    reported rather than repaired.
    """
    quantiles = np.asarray(quantiles, dtype=float)
    if quantiles.shape[1] < 2:
        return 0.0
    return float(np.mean(np.diff(quantiles, axis=1) < 0))


# ---------------------------------------------------------------------------
def _finite(values) -> np.ndarray:
    array = pd.to_numeric(pd.Series(np.asarray(values).reshape(-1)), errors="coerce").to_numpy(dtype=float)
    return array[np.isfinite(array)]


def _abs_pearson(a: pd.Series, b: pd.Series) -> float | None:
    frame = pd.DataFrame({
        "a": pd.to_numeric(a, errors="coerce"),
        "b": pd.to_numeric(b, errors="coerce"),
    }).dropna()
    if len(frame) < 2 or frame["a"].std() <= EPS or frame["b"].std() <= EPS:
        return None
    return float(abs(np.corrcoef(frame["a"], frame["b"])[0, 1]))


def _category_distribution(values: Sequence, domain=None) -> dict[str, float] | None:
    series = pd.Series(values).dropna().astype(str)
    if len(series) == 0:
        return None
    if domain is not None:
        allowed = {str(v) for v in domain}
        series = series.where(series.isin(allowed), "__UNKNOWN__")
    counts = series.value_counts(normalize=True)
    return {str(index): float(value) for index, value in counts.items()}


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / np.clip(q[mask], EPS, None))))

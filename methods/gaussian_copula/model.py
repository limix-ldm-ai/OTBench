"""A Gaussian copula over mixed tabular data, with conditional sampling.

The model is the textbook one (Sklar's theorem; the Gaussian-copula construction
used by Copulas/SDV): each column is mapped to a uniform by its own marginal CDF
and then to a standard normal by the probit, a single correlation matrix is
fitted in that latent space, and sampling inverts the chain.

    z ~ N(0, R)   ->   u = Phi(z)   ->   x = F_j^{-1}(u_j)

Everything the benchmark needs beyond plain sampling follows from one fact: the
conditional of a multivariate normal is again normal, in closed form. That gives
conditional sampling of any subset of columns given any other subset, which is
generation, imputation and prediction from *one* fitted joint model rather than
three separately trained ones.

Two modelling choices are worth stating because they bound what the model can
represent:

* **Numerical marginals are empirical**, inverted by interpolating the observed
  order statistics. No parametric family is imposed, so a bimodal or heavy-
  tailed column keeps its shape; the cost is that the model cannot extrapolate
  beyond the training range.
* **Categorical columns use the distributional transform.** A category is
  mapped to a uniform draw inside its CDF interval, which is the standard way to
  extend a copula to discrete margins. This makes the fitted latent correlation
  an attenuated version of the underlying association, so the model understates
  dependence involving low-cardinality categoricals. That is a property of the
  Gaussian copula on discrete data, not a bug in this implementation, and it is
  why the method is a strong baseline rather than a state-of-the-art generator.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-6


@dataclass
class ColumnMarginal:
    """One column's marginal, plus its forward and inverse CDF."""

    name: str
    kind: str                       # "numerical" | "categorical"
    quantiles: list = field(default_factory=list)     # sorted observed values
    categories: list = field(default_factory=list)
    cumulative: list = field(default_factory=list)    # CDF at each category

    def to_uniform(self, values: pd.Series, rng: np.random.Generator) -> np.ndarray:
        if self.kind == "numerical":
            numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
            grid = np.asarray(self.quantiles, dtype=float)
            if len(grid) < 2:
                return np.full(len(numeric), 0.5)
            # Empirical CDF by rank within the training order statistics.
            ranks = np.searchsorted(grid, numeric, side="right") / len(grid)
            ranks = np.where(np.isfinite(numeric), ranks, np.nan)
            return np.clip(ranks, EPS, 1.0 - EPS)

        # Distributional transform: u ~ Uniform(F(c-1), F(c)) for category c.
        lookup = {c: i for i, c in enumerate(self.categories)}
        upper = np.asarray(self.cumulative, dtype=float)
        lower = np.concatenate([[0.0], upper[:-1]])
        codes = values.astype(str).map(lookup).to_numpy()
        out = np.full(len(codes), np.nan)
        known = pd.notna(codes)
        idx = codes[known].astype(int)
        out[known] = lower[idx] + rng.random(known.sum()) * (upper[idx] - lower[idx])
        return np.clip(out, EPS, 1.0 - EPS)

    def from_uniform(self, u: np.ndarray) -> np.ndarray:
        u = np.clip(np.asarray(u, dtype=float), EPS, 1.0 - EPS)
        if self.kind == "numerical":
            grid = np.asarray(self.quantiles, dtype=float)
            if len(grid) == 0:
                return np.full(len(u), np.nan)
            if len(grid) == 1:
                return np.full(len(u), grid[0])
            positions = np.linspace(0.0, 1.0, len(grid))
            return np.interp(u, positions, grid)

        upper = np.asarray(self.cumulative, dtype=float)
        indices = np.searchsorted(upper, u, side="left")
        indices = np.clip(indices, 0, len(self.categories) - 1)
        return np.asarray([self.categories[int(i)] for i in indices], dtype=object)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "quantiles": self.quantiles,
                "categories": self.categories, "cumulative": self.cumulative}

    @classmethod
    def from_dict(cls, payload: dict) -> "ColumnMarginal":
        return cls(**payload)


@dataclass
class GaussianCopula:
    columns: list[str]
    marginals: dict[str, ColumnMarginal]
    correlation: np.ndarray
    n_train_rows: int = 0

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(cls, frame: pd.DataFrame, numerical: list[str], categorical: list[str],
            seed: int = 0, max_quantiles: int = 1000) -> "GaussianCopula":
        rng = np.random.default_rng(seed)
        columns = [c for c in frame.columns]
        marginals: dict[str, ColumnMarginal] = {}
        for column in columns:
            if column in set(numerical):
                values = pd.to_numeric(frame[column], errors="coerce").dropna()
                sorted_values = np.sort(values.to_numpy(float))
                if len(sorted_values) > max_quantiles:
                    # Subsample the order statistics on a regular grid: the inverse
                    # CDF is an interpolation anyway, and this keeps the checkpoint
                    # from scaling with the training set.
                    positions = np.linspace(0, len(sorted_values) - 1, max_quantiles)
                    sorted_values = sorted_values[positions.astype(int)]
                marginals[column] = ColumnMarginal(
                    column, "numerical", quantiles=sorted_values.tolist())
            else:
                counts = frame[column].astype(str).value_counts(normalize=True).sort_index()
                marginals[column] = ColumnMarginal(
                    column, "categorical",
                    categories=counts.index.tolist(),
                    cumulative=np.cumsum(counts.to_numpy(float)).tolist(),
                )

        latent = np.column_stack([
            stats.norm.ppf(marginals[c].to_uniform(frame[c], rng)) for c in columns
        ])
        finite = np.isfinite(latent).all(axis=1)
        latent = latent[finite]
        correlation = _shrunk_correlation(latent)
        return cls(columns=columns, marginals=marginals, correlation=correlation,
                   n_train_rows=int(len(frame)))

    # --------------------------------------------------------------- sample
    def sample(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        latent = rng.multivariate_normal(
            mean=np.zeros(len(self.columns)), cov=self.correlation, size=n,
            method="cholesky",
        )
        return self._decode(latent)

    def sample_conditional(
        self,
        observed: pd.DataFrame,
        targets: list[str],
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Sample ``targets`` given the observed values of the other columns.

        Uses the closed-form Gaussian conditional
        ``z_t | z_o ~ N(R_to R_oo^{-1} z_o, R_tt - R_to R_oo^{-1} R_ot)``,
        which is the same joint model as :meth:`sample`, not a second fit.
        """
        target_idx = [self.columns.index(c) for c in targets]
        observed_cols = [c for c in self.columns if c not in set(targets)]
        observed_idx = [self.columns.index(c) for c in observed_cols]

        if not observed_idx:
            return self.sample(len(observed), rng)[targets]

        z_obs = np.column_stack([
            stats.norm.ppf(self.marginals[c].to_uniform(observed[c], rng))
            for c in observed_cols
        ])
        # A cell that is itself missing carries no information; the median of the
        # latent (0) is the conditional mean given nothing, so it is neutral.
        z_obs = np.nan_to_num(z_obs, nan=0.0, posinf=0.0, neginf=0.0)

        r_tt = self.correlation[np.ix_(target_idx, target_idx)]
        r_to = self.correlation[np.ix_(target_idx, observed_idx)]
        r_oo = self.correlation[np.ix_(observed_idx, observed_idx)]
        solve = np.linalg.solve(r_oo, r_to.T).T          # R_to R_oo^{-1}

        mean = z_obs @ solve.T                            # (n, |targets|)
        covariance = r_tt - solve @ r_to.T
        covariance = _nearest_psd(covariance)

        noise = rng.multivariate_normal(
            np.zeros(len(target_idx)), covariance, size=len(observed), method="cholesky")
        latent = mean + noise
        return self._decode(latent, columns=targets)

    def conditional_mean(self, observed: pd.DataFrame, targets: list[str],
                         rng: np.random.Generator) -> pd.DataFrame:
        """The conditional latent mean, mapped back through the marginals.

        This is the point prediction: the median of the conditional predictive
        distribution, not its mean on the original scale. The two differ for a
        skewed marginal, and calling it a median is the accurate description.
        """
        target_idx = [self.columns.index(c) for c in targets]
        observed_cols = [c for c in self.columns if c not in set(targets)]
        observed_idx = [self.columns.index(c) for c in observed_cols]
        if not observed_idx:
            zeros = np.zeros((len(observed), len(targets)))
            return self._decode(zeros, columns=targets)

        z_obs = np.nan_to_num(np.column_stack([
            stats.norm.ppf(self.marginals[c].to_uniform(observed[c], rng))
            for c in observed_cols
        ]), nan=0.0, posinf=0.0, neginf=0.0)
        r_to = self.correlation[np.ix_(target_idx, observed_idx)]
        r_oo = self.correlation[np.ix_(observed_idx, observed_idx)]
        mean = z_obs @ np.linalg.solve(r_oo, r_to.T)
        return self._decode(mean, columns=targets)

    def conditional_category_probabilities(
        self, observed: pd.DataFrame, target: str, rng: np.random.Generator,
        n_samples: int = 200,
    ) -> tuple[np.ndarray, list]:
        """Class probabilities by Monte Carlo over the conditional latent.

        The latent conditional is exact and closed-form; turning it into category
        probabilities requires integrating over the distributional-transform
        interval of each category, which has no closed form here. Monte Carlo is
        the honest way to do it, and the sample count is recorded in the metadata
        so the resulting probabilities are known to be estimates.
        """
        marginal = self.marginals[target]
        categories = list(marginal.categories)
        counts = np.zeros((len(observed), len(categories)), dtype=float)
        index = {c: i for i, c in enumerate(categories)}
        for _ in range(n_samples):
            drawn = self.sample_conditional(observed, [target], rng)[target].astype(str)
            for row, value in enumerate(drawn):
                position = index.get(value)
                if position is not None:
                    counts[row, position] += 1.0
        totals = counts.sum(axis=1, keepdims=True)
        proba = np.divide(counts, totals, out=np.full_like(counts, 1.0 / len(categories)),
                          where=totals > 0)
        return proba, categories

    # -------------------------------------------------------------- helpers
    def _decode(self, latent: np.ndarray, columns: list[str] | None = None) -> pd.DataFrame:
        columns = columns or self.columns
        uniform = stats.norm.cdf(latent)
        return pd.DataFrame({
            column: self.marginals[column].from_uniform(uniform[:, position])
            for position, column in enumerate(columns)
        })

    # ------------------------------------------------------------ serialise
    def save(self, path: Path) -> None:
        payload = {
            "columns": self.columns,
            "marginals": {k: v.to_dict() for k, v in self.marginals.items()},
            "correlation": self.correlation.tolist(),
            "n_train_rows": self.n_train_rows,
        }
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: Path) -> "GaussianCopula":
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(
            columns=payload["columns"],
            marginals={k: ColumnMarginal.from_dict(v)
                       for k, v in payload["marginals"].items()},
            correlation=np.asarray(payload["correlation"], dtype=float),
            n_train_rows=payload["n_train_rows"],
        )


def _shrunk_correlation(latent: np.ndarray, shrinkage: float = 1e-3) -> np.ndarray:
    """Latent correlation, nudged toward the identity so it stays invertible."""
    if len(latent) < 2:
        return np.eye(latent.shape[1])
    correlation = np.corrcoef(latent, rowvar=False)
    correlation = np.atleast_2d(np.nan_to_num(correlation, nan=0.0))
    np.fill_diagonal(correlation, 1.0)
    identity = np.eye(len(correlation))
    return _nearest_psd((1.0 - shrinkage) * correlation + shrinkage * identity)


def _nearest_psd(matrix: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    """Clip negative eigenvalues so Cholesky sampling always succeeds."""
    symmetric = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(symmetric)
    values = np.clip(values, floor, None)
    return vectors @ np.diag(values) @ vectors.T

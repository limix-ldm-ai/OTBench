"""Train-fitted encodings and the one Gower distance used by every DCR variant.

The privacy metrics only mean something if "distance" means the same thing on
both sides of a ratio, so the Gower definition lives here once and every caller
uses it. Ranges and category sets are fitted on the *train* split, never on the
frames being compared.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

EPS = 1e-12


@dataclass
class GowerSpace:
    """A train-fitted Gower metric space over the schema's columns.

    Gower (1971): the distance between two rows is the mean over columns of a
    per-column distance -- ``|a - b| / range`` for numerical columns and
    ``0/1`` inequality for categorical ones. Every column therefore contributes
    at most 1, so mixed-type distances are commensurable.
    """

    numerical: list[str]
    categorical: list[str]
    ranges: dict[str, float]

    @classmethod
    def fit(cls, train: pd.DataFrame, schema) -> "GowerSpace":
        numerical = [c for c in schema.numerical_cols if c in train.columns]
        categorical = [c for c in schema.categorical_cols if c in train.columns]
        ranges = {}
        for col in numerical:
            values = pd.to_numeric(train[col], errors="coerce").dropna()
            span = float(values.max() - values.min()) if len(values) else 0.0
            ranges[col] = span if span > EPS else 1.0
        return cls(numerical=numerical, categorical=categorical, ranges=ranges)

    @property
    def n_columns(self) -> int:
        return len(self.numerical) + len(self.categorical)

    def encode(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (range-scaled numerical block, categorical code block)."""
        if self.numerical:
            num = np.column_stack([
                pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
                / self.ranges[col]
                for col in self.numerical
            ])
        else:
            num = np.zeros((len(frame), 0), dtype=float)
        if self.categorical:
            cat = np.column_stack([
                frame[col].astype("string").fillna("__MISSING__").to_numpy()
                for col in self.categorical
            ])
        else:
            cat = np.zeros((len(frame), 0), dtype=object)
        return num, cat

    def min_distances(self, query: pd.DataFrame, reference: pd.DataFrame,
                      block_size: int = 512) -> np.ndarray:
        """Distance from every ``query`` row to its closest ``reference`` row."""
        if self.n_columns == 0 or len(query) == 0 or len(reference) == 0:
            return np.full(len(query), np.nan)
        q_num, q_cat = self.encode(query)
        r_num, r_cat = self.encode(reference)
        out = np.empty(len(query), dtype=float)
        for start in range(0, len(query), block_size):
            stop = min(start + block_size, len(query))
            total = np.zeros((stop - start, len(reference)), dtype=float)
            if q_num.shape[1]:
                diff = np.abs(q_num[start:stop, None, :] - r_num[None, :, :])
                total += np.nansum(diff, axis=2)
            if q_cat.shape[1]:
                total += (q_cat[start:stop, None, :] != r_cat[None, :, :]).sum(axis=2)
            out[start:stop] = total.min(axis=1) / self.n_columns
        return out

    def argmin_distances(self, query: pd.DataFrame, reference: pd.DataFrame,
                         block_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
        """``(min distance, index of the closest reference row)`` per query row."""
        if self.n_columns == 0 or len(query) == 0 or len(reference) == 0:
            return np.full(len(query), np.nan), np.full(len(query), -1, dtype=int)
        q_num, q_cat = self.encode(query)
        r_num, r_cat = self.encode(reference)
        best = np.empty(len(query), dtype=float)
        where = np.empty(len(query), dtype=int)
        for start in range(0, len(query), block_size):
            stop = min(start + block_size, len(query))
            total = np.zeros((stop - start, len(reference)), dtype=float)
            if q_num.shape[1]:
                total += np.nansum(np.abs(q_num[start:stop, None, :] - r_num[None, :, :]), axis=2)
            if q_cat.shape[1]:
                total += (q_cat[start:stop, None, :] != r_cat[None, :, :]).sum(axis=2)
            where[start:stop] = total.argmin(axis=1)
            best[start:stop] = total.min(axis=1) / self.n_columns
        return best, where


def euclidean_min_distances(
    query: np.ndarray | sparse.spmatrix,
    reference: np.ndarray | sparse.spmatrix,
    block_size: int = 512,
    reference_block_size: int | None = None,
) -> np.ndarray:
    """Exact nearest-reference Euclidean distances with bounded intermediates.

    Uses ``||q-r||^2 = ||q||^2 + ||r||^2 - 2 q r^T`` and
    blocks *both* row dimensions.  CSR inputs therefore stay sparse until the
    small pairwise distance block is produced; no feature-sized 3-D array or
    full dense one-hot table is created.

    All arithmetic is float64.  ``reference_block_size`` defaults to
    ``block_size``, bounding the largest dense temporary to at most
    ``block_size * reference_block_size`` float64 values.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if reference_block_size is None:
        reference_block_size = block_size
    if reference_block_size <= 0:
        raise ValueError("reference_block_size must be positive")

    q = _float64_matrix(query)
    r = _float64_matrix(reference)
    if q.ndim != 2 or r.ndim != 2:
        raise ValueError("query and reference must be two-dimensional")
    if q.shape[1] != r.shape[1]:
        raise ValueError(
            f"query/reference feature mismatch: {q.shape[1]} != {r.shape[1]}"
        )

    n_query, n_features = q.shape
    n_reference = r.shape[0]
    if n_query == 0 or n_reference == 0 or n_features == 0:
        return np.full(n_query, np.nan, dtype=np.float64)

    # Exact CSR row signatures identify zero-distance matches
    # even when the norm identity leaves a tiny positive cancellation residual.
    if sparse.issparse(q) and sparse.issparse(r):
        exact_duplicate = _sparse_duplicate_rows(q, r)
    elif not sparse.issparse(q) and not sparse.issparse(r):
        exact_duplicate = _dense_duplicate_rows(q, r)
    else:
        exact_duplicate = None
    q_norm = _row_squared_norms(q)
    r_norm = _row_squared_norms(r)
    best_reference = np.full(n_query, -1, dtype=np.int64)

    for q_start in range(0, n_query, block_size):
        q_stop = min(q_start + block_size, n_query)
        q_block = q[q_start:q_stop]
        best = np.full(q_stop - q_start, np.inf, dtype=np.float64)
        best_index = np.full(q_stop - q_start, -1, dtype=np.int64)
        invalid_seen = np.zeros(q_stop - q_start, dtype=bool)
        q_block_norm = q_norm[q_start:q_stop]

        for r_start in range(0, n_reference, reference_block_size):
            r_stop = min(r_start + reference_block_size, n_reference)
            r_block = r[r_start:r_stop]
            scale = q_block_norm[:, None] + r_norm[None, r_start:r_stop]
            squared = scale - 2.0 * _dense_dot(q_block, r_block)

            # Cancellation may make a mathematically non-negative value a few
            # ulps negative.  Clamp negatives only: small *positive* distances
            # remain non-zero. Exact duplicate rows are restored to zero below
            # using direct row equality, not a numerical tolerance.
            np.maximum(squared, 0.0, out=squared)
            block_index = squared.argmin(axis=1)
            block_best = squared[np.arange(len(squared)), block_index]
            invalid_seen |= np.isnan(block_best)
            improve = (~invalid_seen) & (
                (best_index < 0) | (block_best < best)
            )
            best[improve] = block_best[improve]
            best_index[improve] = r_start + block_index[improve]
            best[invalid_seen] = np.nan
            best_index[invalid_seen] = -1

        best_reference[q_start:q_stop] = best_index

    # Recompute the selected distance by direct row subtraction. This costs
    # O(n_query * n_features) (O(nnz) for CSR), keeps memory bounded to one row,
    # and prevents cancellation from turning a genuine close neighbour to zero.
    out = _direct_selected_distances(q, r, best_reference)
    if exact_duplicate is not None:
        out[exact_duplicate] = 0.0
    return out


def _float64_matrix(matrix: np.ndarray | sparse.spmatrix):
    """Normalise matrix layout/dtype without densifying sparse inputs."""
    if sparse.issparse(matrix):
        result = matrix.astype(np.float64, copy=True).tocsr()
        result.sum_duplicates()
        result.eliminate_zeros()
        result.sort_indices()
        return result
    return np.asarray(matrix, dtype=np.float64)


def _row_squared_norms(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()
    return np.einsum("ij,ij->i", matrix, matrix, dtype=np.float64)


def _dense_dot(query_block, reference_block) -> np.ndarray:
    """Return only the current 2-D dot-product block as a dense ndarray."""
    if sparse.issparse(query_block):
        product = query_block @ reference_block.T
    elif sparse.issparse(reference_block):
        product = (reference_block @ query_block.T).T
    else:
        product = query_block @ reference_block.T
    if sparse.issparse(product):
        product = product.toarray()
    return np.asarray(product, dtype=np.float64)


def _direct_selected_distances(query, reference, selected: np.ndarray) -> np.ndarray:
    """Recompute selected pairs by subtraction, one bounded row at a time."""
    out = np.full(query.shape[0], np.nan, dtype=np.float64)
    both_sparse = sparse.issparse(query) and sparse.issparse(reference)
    both_dense = not sparse.issparse(query) and not sparse.issparse(reference)
    for row, reference_row in enumerate(selected):
        if reference_row < 0:
            continue
        if both_sparse:
            difference = query.getrow(row) - reference.getrow(reference_row)
            squared = float(difference.multiply(difference).sum())
        elif both_dense:
            difference = query[row] - reference[reference_row]
            squared = float(np.sum(difference ** 2))
        else:
            query_values = (
                query.getrow(row).toarray().ravel()
                if sparse.issparse(query)
                else query[row]
            )
            reference_values = (
                reference.getrow(reference_row).toarray().ravel()
                if sparse.issparse(reference)
                else reference[reference_row]
            )
            squared = float(np.sum((query_values - reference_values) ** 2))
        out[row] = np.sqrt(squared)
    return out


def _sparse_duplicate_rows(query, reference) -> np.ndarray:
    """Mark CSR query rows that are bit-exactly present in the reference."""
    reference_keys = {
        key
        for row in range(reference.shape[0])
        if (key := _sparse_row_key(reference, row)) is not None
    }
    return np.fromiter(
        (
            (key := _sparse_row_key(query, row)) is not None
            and key in reference_keys
            for row in range(query.shape[0])
        ),
        dtype=bool,
        count=query.shape[0],
    )


def _sparse_row_key(matrix, row: int) -> tuple[bytes, bytes] | None:
    start, stop = matrix.indptr[row:row + 2]
    indices = matrix.indices[start:stop].astype(np.int64, copy=False)
    values = matrix.data[start:stop]
    if not np.isfinite(values).all():
        return None
    return indices.tobytes(), values.tobytes()


def _dense_duplicate_rows(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Find exact dense duplicates without a rows x rows x features array."""
    by_digest: dict[bytes, list[int]] = {}
    for row in range(reference.shape[0]):
        digest = _dense_row_digest(reference[row])
        if digest is not None:
            by_digest.setdefault(digest, []).append(row)

    duplicate = np.zeros(query.shape[0], dtype=bool)
    for row in range(query.shape[0]):
        digest = _dense_row_digest(query[row])
        if digest is None:
            continue
        duplicate[row] = any(
            np.array_equal(query[row], reference[candidate])
            for candidate in by_digest.get(digest, ())
        )
    return duplicate


def _dense_row_digest(row: np.ndarray) -> bytes | None:
    if not np.isfinite(row).all():
        return None
    canonical = np.ascontiguousarray(row)
    # Direct subtraction treats -0.0 and +0.0 as equal; make their digest agree.
    signed_zero = (canonical == 0.0) & np.signbit(canonical)
    if signed_zero.any():
        canonical = canonical.copy()
        canonical[signed_zero] = 0.0
    digest = hashlib.blake2b(digest_size=16)
    digest.update(memoryview(canonical).cast("B"))
    return digest.digest()


def tpr_at_fpr(scores: np.ndarray, labels: np.ndarray, target_fpr: float = 0.01) -> float | None:
    """True-positive rate at a fixed false-positive rate.

    Averaged AUROC hides the regime that matters for a membership attack: an
    attacker only needs to be confidently right about a few records.
    """
    from sklearn.metrics import roc_curve

    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) < 2:
        return None
    fpr, tpr, _ = roc_curve(labels, np.asarray(scores, dtype=float))
    admissible = fpr <= target_fpr
    return float(tpr[admissible].max()) if admissible.any() else 0.0

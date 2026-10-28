# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""TabImpute (Feitelberg et al. 2025) — pretrained zero-shot imputation.

Runs the authors' ``tabimpute`` package. The model is a transformer that fills a
whole matrix in one forward pass, so unlike the foundation methods there is no
per-column estimator: the masked test frame is handed over as a matrix with NaN
at the masked cells and comes back complete.

Two consequences shape this adapter.

*Context rows.* The model imputes in context, using the other rows of the very
matrix it is given. A fixed train-only context is prepended to every fixed-size
test chunk and dropped again afterwards. That keeps the method inductive —
nothing is fitted on test — and bounds the V2 bar-distribution tensor without
changing context between chunks. Upstream's ``max_num_rows`` is not used because
V2 rejects it outright.

*Categoricals.* Upstream's own categorical wrapper (``TabImputeCategorical``) is
hardwired to the V1 model, so its scheme is reproduced here against V2: one-hot
per categorical column, all-NaN across the block for a missing cell, and softmax
+ argmax to read a category back out. The one difference is deliberate — the
category vocabulary is fitted on train rather than on the frame being imputed,
because taking it from the test frame would let the test split decide what the
model is allowed to predict.

V2 supports neither ``num_repeats > 1`` nor sampling: ``get_imputation`` returns
the median of the model's bar distribution. The method therefore declares
``point`` output only, and K draws are K copies of one deterministic answer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from otbench.methods.common import data as data_utils
from otbench.methods.common.base import ImputeContext, TabularMethod, Task
from otbench.methods.common.capabilities import UnsupportedRun
from otbench.methods.common.imputer_base import merge_masked_only
from otbench.methods.common.seeding import seed_everything
from otbench.methods.common.upstream import require_package

UPSTREAM_URL = "https://github.com/jacobf18/tabular"


class TabImputeMethod(TabularMethod):
    name = "tabimpute"
    tasks = frozenset({Task.IMPUTATION})
    trainable = False

    def impute(self, ctx: ImputeContext) -> pd.DataFrame:
        require_package("tabimpute", method="tabimpute", url=UPSTREAM_URL,
                        extra="imputation")
        from tabimpute.tabimpute_v2 import TabImputeV2

        seed_everything(ctx.seed)
        schema = ctx.bundle.schema
        columns = list(schema.all_columns)
        train = ctx.bundle.train.reset_index(drop=True)[columns]
        masked = ctx.masked_test.reset_index(drop=True)[columns]

        codec = _OneHotCodec.fit(train, schema)
        context = self._context_rows(train, int(ctx.seed))
        max_test_rows = int(self.cfg("impute", "max_test_rows_per_batch", 256))
        if max_test_rows <= 0:
            raise ValueError(f"max_test_rows_per_batch must be positive, got {max_test_rows}")
        n_bins = int(self.cfg("impute", "bar_distribution_bins", 5000))
        workspace_factor = float(self.cfg("impute", "distribution_workspace_factor", 3.0))
        max_estimated_bytes = int(
            float(self.cfg("impute", "max_estimated_gib", 16.0)) * (1024 ** 3)
        )
        _check_encoded_capacity(
            rows=len(context) + min(len(masked), max_test_rows),
            width=codec.width,
            max_cells=int(self.cfg("impute", "max_encoded_cells", 750_000)),
            n_bins=n_bins,
            dtype_bytes=np.dtype(np.float32).itemsize,
            workspace_factor=workspace_factor,
            max_estimated_bytes=max_estimated_bytes,
        )
        if masked.empty:
            return masked
        imputer = TabImputeV2(
            device=self.resolve_device(ctx),
            checkpoint_path=self.pretrained_path("model_ckpt"),
            verbose=bool(self.cfg("model", "verbose", False)),
        )
        context_matrix = codec.encode(context).astype(np.float32, copy=False)
        completed_chunks = []
        for start in range(0, len(masked), max_test_rows):
            stop = min(start + max_test_rows, len(masked))
            masked_chunk = masked.iloc[start:stop].reset_index(drop=True)
            _check_encoded_capacity(
                rows=len(context) + len(masked_chunk),
                width=codec.width,
                max_cells=int(self.cfg("impute", "max_encoded_cells", 750_000)),
                n_bins=n_bins,
                dtype_bytes=np.dtype(np.float32).itemsize,
                workspace_factor=workspace_factor,
                max_estimated_bytes=max_estimated_bytes,
            )
            test_matrix = codec.encode(masked_chunk).astype(np.float32, copy=False)
            matrix = np.concatenate([context_matrix, test_matrix], axis=0)
            filled = np.asarray(imputer.impute(matrix), dtype=np.float64)
            completed_chunks.append(codec.decode(filled[len(context):], masked_chunk))

        completed = (
            pd.concat(completed_chunks, ignore_index=True)
            if completed_chunks else masked.copy()
        )
        return merge_masked_only(masked, completed, ctx.mask.reset_index(drop=True))

    def _context_rows(self, train: pd.DataFrame, seed: int) -> pd.DataFrame:
        """Clean train rows prepended to the matrix as in-context support."""
        budget = self.cfg("impute", "max_context_rows", 2048)
        if budget is None or len(train) <= int(budget):
            return train
        rng = np.random.default_rng(int(self.cfg("impute", "train_seed", 0)) + seed)
        picked = np.sort(rng.choice(len(train), size=int(budget), replace=False))
        return train.iloc[picked].reset_index(drop=True)


class _OneHotCodec:
    """Upstream's one-hot scheme with a train-fitted category vocabulary."""

    def __init__(self, columns, categories, fills):
        self.columns = list(columns)
        self.categories = categories      # column -> list of category values
        self.fills = fills                # numeric column -> train mean
        self.blocks = {}                  # column -> (start, stop) in the matrix
        start = 0
        for column in self.columns:
            width = len(self.categories[column]) if column in self.categories else 1
            self.blocks[column] = (start, start + max(1, width))
            start += max(1, width)
        self.width = start

    @classmethod
    def fit(cls, train: pd.DataFrame, schema) -> "_OneHotCodec":
        categorical = set(schema.categorical_cols)
        categories, fills = {}, {}
        for column in schema.all_columns:
            if column in categorical:
                values = train[column].dropna().astype(str).unique()
                categories[column] = sorted(values)
            else:
                mean = pd.to_numeric(train[column], errors="coerce").mean()
                fills[column] = 0.0 if pd.isna(mean) else float(mean)
        return cls(schema.all_columns, categories, fills)

    def encode(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = np.full((len(frame), self.width), np.nan, dtype=np.float64)
        for column in self.columns:
            start, stop = self.blocks[column]
            values = frame[column]
            if column in self.categories:
                known = self.categories[column]
                if not known:
                    continue
                present = values.notna().to_numpy()
                codes = values.astype(str).map({c: i for i, c in enumerate(known)})
                block = np.zeros((len(frame), stop - start), dtype=np.float64)
                seen = present & codes.notna().to_numpy()
                block[seen, codes[seen].to_numpy(dtype=int)] = 1.0
                # A cell missing from the frame stays NaN across the whole block,
                # which is how upstream marks a categorical cell as unobserved.
                block[~present, :] = np.nan
                matrix[:, start:stop] = block
            else:
                matrix[:, start] = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
        return matrix

    def decode(self, matrix: np.ndarray, reference: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for column in self.columns:
            start, stop = self.blocks[column]
            if column in self.categories:
                known = self.categories[column]
                if not known:
                    out[column] = reference[column].to_numpy()
                    continue
                scores = np.nan_to_num(matrix[:, start:stop], nan=0.0)
                picked = np.asarray(known, dtype=object)[scores.argmax(axis=1)]
                out[column] = picked
            else:
                values = matrix[:, start]
                out[column] = np.where(np.isfinite(values), values, self.fills[column])
        completed = pd.DataFrame(out, index=reference.index)[self.columns]
        return data_utils.reset_rows(completed)


def _check_encoded_capacity(
    *,
    rows: int,
    width: int,
    max_cells: int,
    n_bins: int = 5000,
    dtype_bytes: int = 4,
    workspace_factor: float = 1.0,
    max_estimated_bytes: int | None = None,
) -> None:
    cells = int(rows) * int(width)
    if cells > int(max_cells):
        raise UnsupportedRun(
            "TabImpute V2 uses dense whole-matrix inference; the encoded matrix "
            f"would contain {cells:,} cells ({rows:,}x{width:,}), above the "
            f"validated limit {max_cells:,}. Reduce max_context_rows or use a "
            "method that supports this dataset scale."
        )
    estimated_bytes = int(
        cells * int(n_bins) * int(dtype_bytes) * float(workspace_factor)
    )
    if max_estimated_bytes is not None and estimated_bytes > int(max_estimated_bytes):
        gib = estimated_bytes / (1024 ** 3)
        limit_gib = int(max_estimated_bytes) / (1024 ** 3)
        raise UnsupportedRun(
            "TabImpute V2 bar-distribution inference is estimated to require "
            f"{gib:.2f} GiB for {rows:,}x{width:,} cells, {int(n_bins):,} bars, "
            f"and workspace factor {float(workspace_factor):g}; the configured "
            f"limit is {limit_gib:.2f} GiB. Reduce max_context_rows or "
            "max_test_rows_per_batch."
        )

"""Train-fitted layout between a schema frame and the flat matrix a model wants.

ReMasker, TabCSDI and DiffPuter all consume ``(n_rows, n_matrix_columns)`` floats
plus a cell mask, and all three widen a categorical column into more than one
matrix column. Only the *widening rule* and the *scaling* differ between them,
and those are decisions their papers make, so they stay in the adapters. What is
identical — which schema column owns which matrix columns, how a per-column mask
becomes a per-matrix-column mask, how a decoded matrix becomes a frame again —
lives here, once.

Two widening rules are provided because they are the two the three upstreams use:

``ordinal``
    One matrix column holding the category index. ReMasker's implementation is
    numerical-only, so this is the encoding that lets it run at all on a mixed
    frame; the adapter says so in its metadata.

``analog_bits``
    Matrix columns holding the binary expansion of the category index, the
    "analog bits" encoding of Chen et al. (2023) that both TabCSDI and DiffPuter
    adopt for categorical columns. The two upstreams differ on one detail and
    ``reserve_missing_code`` selects between them: DiffPuter numbers categories
    from 0 in ``(n_levels - 1).bit_length()`` bits, while TabCSDI numbers them
    from 1 in ``n_levels.bit_length()`` bits so that the all-zero pattern stays
    free to mean "missing".

Category vocabularies come from the training split only. A value that never
appeared in train has no code, and forcing one would silently relabel it, so it
decodes to the train mode instead and the adapter reports how often that
happened.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

CategoricalEncoding = Literal["ordinal", "analog_bits"]


@dataclass(frozen=True)
class Block:
    """The matrix columns owned by one schema column."""

    name: str
    is_categorical: bool
    start: int
    width: int
    levels: tuple = ()

    @property
    def stop(self) -> int:
        return self.start + self.width

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)


class BlockLayout:
    """A train-fitted frame <-> matrix mapping with an explicit block structure."""

    def __init__(self, blocks: list[Block], columns: list[str],
                 encoding: CategoricalEncoding, bit_values: tuple[float, float],
                 reserve_missing_code: bool = False) -> None:
        self.blocks = blocks
        self.columns = list(columns)
        self.encoding = encoding
        self.bit_low, self.bit_high = bit_values
        self.reserve_missing_code = reserve_missing_code
        self.width = blocks[-1].stop if blocks else 0
        self._by_name = {b.name: b for b in blocks}
        self._codebooks = {
            b.name: _codebook(len(b.levels), b.width, self.bit_low, self.bit_high,
                              reserve_missing_code)
            for b in blocks if b.is_categorical and encoding == "analog_bits"
        }

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(cls, train: pd.DataFrame, schema, *, encoding: CategoricalEncoding,
            bit_values: tuple[float, float] = (0.0, 1.0),
            reserve_missing_code: bool = False) -> "BlockLayout":
        categorical = set(schema.categorical_cols)
        blocks: list[Block] = []
        cursor = 0
        for column in schema.all_columns:
            if column not in train.columns:
                continue
            if column in categorical:
                levels = tuple(sorted(train[column].dropna().astype(str).unique()))
                if not levels:
                    levels = ("__MISSING__",)
                width = 1 if encoding == "ordinal" else _bit_width(
                    len(levels), reserve_missing_code)
                blocks.append(Block(column, True, cursor, width, levels))
            else:
                blocks.append(Block(column, False, cursor, 1))
            cursor = blocks[-1].stop
        return cls(blocks, [b.name for b in blocks], encoding, bit_values,
                   reserve_missing_code)

    # --------------------------------------------------------------- encode
    def encode(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(matrix, observed)`` where ``observed`` is True for known cells.

        Missing cells are left as ``0.0`` in the matrix, matching what all three
        upstreams do before handing the tensor to the model; the mask is what
        tells the model to ignore them.
        """
        n = len(frame)
        matrix = np.zeros((n, self.width), dtype=np.float32)
        observed = np.zeros((n, self.width), dtype=bool)
        for block in self.blocks:
            values = frame[block.name]
            present = values.notna().to_numpy()
            observed[:, block.slice] = present[:, None]
            if not block.is_categorical:
                numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
                matrix[:, block.start] = np.nan_to_num(numeric, nan=0.0)
                continue
            codes = self._codes(block, values)
            if self.encoding == "ordinal":
                matrix[:, block.start] = np.where(present, codes, 0.0)
            else:
                book = self._codebooks[block.name]
                rows = book[np.clip(codes, 0, len(block.levels) - 1)]
                matrix[:, block.slice] = np.where(present[:, None], rows, 0.0)
        return matrix, observed

    def expand_mask(self, mask: pd.DataFrame) -> np.ndarray:
        """Widen a per-schema-column boolean frame to per-matrix-column."""
        out = np.zeros((len(mask), self.width), dtype=bool)
        for block in self.blocks:
            if block.name in mask.columns:
                flags = mask[block.name].fillna(False).to_numpy(dtype=bool)
                out[:, block.slice] = flags[:, None]
        return out

    # --------------------------------------------------------------- decode
    def decode(self, matrix: np.ndarray, template: pd.DataFrame | None = None) -> pd.DataFrame:
        """Invert :meth:`encode`. Categorical blocks snap to the nearest code."""
        matrix = np.asarray(matrix, dtype=float)
        frame = pd.DataFrame(index=range(matrix.shape[0]))
        for block in self.blocks:
            if not block.is_categorical:
                frame[block.name] = matrix[:, block.start]
                continue
            if self.encoding == "ordinal":
                codes = np.rint(matrix[:, block.start])
            else:
                # Nearest codeword rather than a per-bit threshold: thresholding
                # can land on a bit pattern no category owns, and there is no
                # sane way to read that back as a label.
                book = self._codebooks[block.name]
                distance = ((matrix[:, block.slice][:, None, :] - book[None, :, :]) ** 2).sum(-1)
                codes = distance.argmin(axis=1)
            codes = np.clip(codes, 0, len(block.levels) - 1).astype(int)
            frame[block.name] = [block.levels[c] for c in codes]
        if template is not None:
            frame = _match_dtypes(frame, template)
        return frame

    # ------------------------------------------------------------- helpers
    def numerical_slices(self) -> list[int]:
        return [b.start for b in self.blocks if not b.is_categorical]

    def categorical_slices(self) -> list[slice]:
        return [b.slice for b in self.blocks if b.is_categorical]

    def block(self, name: str) -> Block:
        return self._by_name[name]

    def _codes(self, block: Block, values: pd.Series) -> np.ndarray:
        index = {level: i for i, level in enumerate(block.levels)}
        fallback = self._mode_code(block, values, index)
        return np.asarray(
            [index.get(str(v), fallback) if pd.notna(v) else 0 for v in values],
            dtype=int,
        )

    @staticmethod
    def _mode_code(block: Block, values: pd.Series, index: dict) -> int:
        known = [index[str(v)] for v in values.dropna() if str(v) in index]
        if not known:
            return 0
        return int(pd.Series(known).mode().iloc[0])


def _bit_width(n_levels: int, reserve_missing_code: bool) -> int:
    return max(1, n_levels.bit_length() if reserve_missing_code
               else (n_levels - 1).bit_length())


def _codebook(n_levels: int, width: int, low: float, high: float,
              reserve_missing_code: bool) -> np.ndarray:
    """Rows are the analog-bits code of each category index, in ``{low, high}``."""
    offset = 1 if reserve_missing_code else 0
    book = np.zeros((max(n_levels, 1), width), dtype=float)
    for level in range(n_levels):
        bits = format(level + offset, f"0{width}b")
        book[level] = [high if b == "1" else low for b in bits]
    return book


def _match_dtypes(frame: pd.DataFrame, template: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if column not in template.columns:
            continue
        dtype = template[column].dtype
        if pd.api.types.is_integer_dtype(dtype):
            out[column] = np.rint(pd.to_numeric(out[column], errors="coerce")).astype(dtype)
        elif pd.api.types.is_bool_dtype(dtype):
            out[column] = pd.to_numeric(out[column], errors="coerce") > 0.5
    return out

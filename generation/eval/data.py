# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from otbench.tabdata.schema import EvalSchema
from otbench.tabdata.io import read_frame_csv


@dataclass
class EvalFrames:
    """The (train, test, gen) triple a generation metric evaluates.

    ``gen`` is the schema-aligned frame: coerced to the schema's column set,
    order and dtypes so downstream models can consume it. ``gen_raw`` is the
    generator's untouched output — validity, missingness, parse-failure and
    exact-copy metrics must read that one, since alignment is exactly the step
    that erases those signals. ``gen_raw`` falls back to ``gen`` only when a
    caller supplied no raw artifact (e.g. evaluating a third-party CSV), and
    :attr:`has_raw` says which of the two happened.
    """

    train: pd.DataFrame
    test: pd.DataFrame
    gen: pd.DataFrame
    schema: EvalSchema
    gen_path: str | None = None
    gen_raw: pd.DataFrame | None = None
    gen_raw_path: str | None = None

    @property
    def has_raw(self) -> bool:
        return self.gen_raw is not None

    def raw_or_aligned(self) -> pd.DataFrame:
        return self.gen_raw if self.gen_raw is not None else self.gen

    def raw_schema_frame(self) -> tuple[pd.DataFrame | None, str | None]:
        """Return raw generated values suitable for distance/privacy metrics.

        Distance metrics may coerce representations (e.g. ``"1"`` to ``1.0``)
        but must not repair missing columns/cells.  If raw data cannot be
        represented in the declared schema, privacy metrics skip and validity
        metrics explain why.
        """
        if self.gen_raw is None:
            return None, "no raw generator artifact was supplied"
        missing = [column for column in self.columns if column not in self.gen_raw.columns]
        if missing:
            return None, f"raw generator artifact is missing schema columns {missing}"
        frame = self.gen_raw[self.columns].copy().reset_index(drop=True)
        invalid: dict[str, int] = {}
        for column in self.schema.numerical_cols:
            if column not in frame:
                continue
            parsed = pd.to_numeric(frame[column], errors="coerce")
            bad = parsed.isna()
            if bad.any():
                invalid[column] = int(bad.sum())
            frame[column] = parsed
        for column in self.schema.categorical_cols:
            if column in frame and frame[column].isna().any():
                invalid[column] = int(frame[column].isna().sum())
            if column in frame:
                frame[column] = frame[column].astype("string")
        if invalid:
            return None, (
                "raw generator artifact has missing or unparseable values; repairing them "
                f"would invalidate a raw privacy measurement: {invalid}"
            )
        return frame, None

    @property
    def columns(self) -> list[str]:
        return self.schema.all_columns

    def original(self, split: str) -> pd.DataFrame:
        return getattr(self, split)[self.columns].copy()

    def onehot_pair(self, real_split: str = "test") -> tuple[pd.DataFrame, pd.DataFrame]:
        real = self.original(real_split)
        syn = self.original("gen")
        return onehot_pair(real, syn, self.schema, train_df=self.train)

    def high_order_pair(self, real_split: str = "test") -> tuple[pd.DataFrame, pd.DataFrame]:
        """Encode real and generated features and target for high-order metrics."""
        real = self.original(real_split)
        syn = self.original("gen")
        return high_order_onehot(real, syn, self.schema, train_df=self.train)

    def ordinal_all(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return ordinal_frames(self.train, self.test, self.gen, self.schema)

    def ci_frame(self, split: str, n_bins: int = 5) -> pd.DataFrame:
        return discretize_for_ci(self.original(split), self.schema, n_bins=n_bins)


def read_csv(path: str | Path, schema=None) -> pd.DataFrame:
    return read_frame_csv(path, schema)


def list_gen_files(path: str | Path) -> list[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)]
    if not p.is_dir():
        raise FileNotFoundError(f"Generated data path does not exist: {path}")
    files = sorted(str(item) for item in p.iterdir() if item.suffix.lower() == ".csv")
    if not files:
        raise ValueError(f"No CSV files found under generated data directory: {path}")
    return files


def align_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    gen_df: pd.DataFrame,
    schema: EvalSchema,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Identifiers belong to the reference-data audit, not to the modeled table.
    # Requiring them from ``gen_df`` contradicts ``EvalSchema.all_columns``
    # (which deliberately excludes IDs) and makes every otherwise valid
    # generator fail on datasets such as adult-census and colleges.
    required = schema.all_columns
    missing = {
        "train": [col for col in required if col not in train_df.columns],
        "test": [col for col in required if col not in test_df.columns],
        "gen": [col for col in required if col not in gen_df.columns],
    }
    missing = {name: cols for name, cols in missing.items() if cols}
    if missing:
        raise ValueError(f"Input CSVs are missing schema columns: {missing}")

    cols = schema.all_columns
    return train_df[cols].copy(), test_df[cols].copy(), gen_df[cols].copy()


def coerce_and_impute(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    gen_df: pd.DataFrame,
    schema: EvalSchema,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = train_df.copy()
    test = test_df.copy()
    gen = gen_df.copy()

    for col in schema.numerical_cols:
        if col in schema.id_cols:
            continue
        train[col] = pd.to_numeric(train[col], errors="coerce")
        test[col] = pd.to_numeric(test[col], errors="coerce")
        gen[col] = pd.to_numeric(gen[col], errors="coerce")
        fill_value = train[col].mean()
        if pd.isna(fill_value):
            fill_value = 0.0
        train[col] = train[col].fillna(fill_value).astype(float)
        test[col] = test[col].fillna(fill_value).astype(float)
        gen[col] = gen[col].fillna(fill_value).astype(float)

    for col in schema.categorical_cols:
        if col in schema.id_cols:
            continue
        train[col] = _categorical_to_string(train[col])
        test[col] = _categorical_to_string(test[col])
        gen[col] = _categorical_to_string(gen[col])

    return train[schema.all_columns], test[schema.all_columns], gen[schema.all_columns]


def load_eval_frames(
    train_path: str | Path,
    test_path: str | Path,
    gen_path: str | Path,
    schema: EvalSchema,
    gen_raw_path: str | Path | None = None,
) -> EvalFrames:
    train_df = read_frame_csv(train_path, schema, strict=True)
    test_df = read_frame_csv(test_path, schema, strict=True)
    gen_df = read_csv(gen_path, schema)
    # Read before alignment touches anything, and keep it out of the coercion path.
    raw_df = read_csv(gen_raw_path, schema) if gen_raw_path is not None else None
    train_df, test_df, gen_df = align_columns(train_df, test_df, gen_df, schema)
    train_df, test_df, gen_df = coerce_and_impute(train_df, test_df, gen_df, schema)
    return EvalFrames(
        train=train_df,
        test=test_df,
        gen=gen_df,
        schema=schema,
        gen_path=str(gen_path),
        gen_raw=raw_df,
        gen_raw_path=str(gen_raw_path) if gen_raw_path is not None else None,
    )


def onehot_pair(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    schema: EvalSchema,
    train_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts_real: list[pd.DataFrame] = []
    parts_syn: list[pd.DataFrame] = []

    if schema.numerical_cols:
        real_num, syn_num = _scaled_numerical_pair(
            train_df if train_df is not None else real_df,
            real_df,
            syn_df,
            schema.numerical_cols,
        )
        parts_real.append(real_num)
        parts_syn.append(syn_num)

    if schema.categorical_cols:
        real_cat, syn_cat = _onehot_columns(
            real_df, syn_df, schema.categorical_cols, train_df=train_df
        )
        parts_real.append(real_cat)
        parts_syn.append(syn_cat)

    if not parts_real:
        return pd.DataFrame(index=real_df.index), pd.DataFrame(index=syn_df.index)

    return (
        pd.concat(parts_real, axis=1).reset_index(drop=True),
        pd.concat(parts_syn, axis=1).reset_index(drop=True),
    )


def sparse_onehot_pair(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    schema: EvalSchema,
    train_df: pd.DataFrame | None = None,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Return the privacy encoding as float64 CSR matrices.

    This is the sparse counterpart of :func:`onehot_pair`: numerical columns
    are standardised with the same train-fitted ``StandardScaler`` and precede
    categorical columns encoded with the same train-fitted category space and
    ``handle_unknown="ignore"`` contract.  It is deliberately separate from
    ``onehot_pair`` so dense consumers such as the high-order metric keep their
    existing representation.
    """
    real_parts: list[sparse.spmatrix] = []
    syn_parts: list[sparse.spmatrix] = []

    if schema.numerical_cols:
        real_num, syn_num = _scaled_numerical_pair(
            train_df if train_df is not None else real_df,
            real_df,
            syn_df,
            schema.numerical_cols,
        )
        # Missing encoded values contribute numeric zeros to distances.
        # conversion before making the small numerical block sparse.
        real_values = np.nan_to_num(real_num.to_numpy(dtype=np.float64))
        syn_values = np.nan_to_num(syn_num.to_numpy(dtype=np.float64))
        real_parts.append(sparse.csr_matrix(real_values, dtype=np.float64))
        syn_parts.append(sparse.csr_matrix(syn_values, dtype=np.float64))

    if schema.categorical_cols:
        encoder = _make_sparse_onehot_encoder()
        basis = train_df if train_df is not None else pd.concat(
            [real_df, syn_df], ignore_index=True
        )
        encoder.fit(basis[schema.categorical_cols].astype(str))
        real_parts.append(
            encoder.transform(real_df[schema.categorical_cols].astype(str)).tocsr()
        )
        syn_parts.append(
            encoder.transform(syn_df[schema.categorical_cols].astype(str)).tocsr()
        )

    if not real_parts:
        return (
            sparse.csr_matrix((len(real_df), 0), dtype=np.float64),
            sparse.csr_matrix((len(syn_df), 0), dtype=np.float64),
        )

    return (
        sparse.hstack(real_parts, format="csr", dtype=np.float64),
        sparse.hstack(syn_parts, format="csr", dtype=np.float64),
    )


def high_order_onehot(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    schema: EvalSchema,
    train_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Encode the joint feature/target distribution for high-order metrics.

    Categorical features are one-hot encoded. A categorical target uses reference-
    class encoding, dropping its final one-hot column when multiple classes exist.
    A numerical target is included with the other numerical columns. Numerical
    scaling fits ``train_df`` when supplied, otherwise ``real_df``.
    """
    target = schema.target_col
    # Numerical columns to include directly. ``schema.numerical_cols`` already
    # contains the target for regression (and excludes it for classification,
    # where the target lives in ``categorical_cols`` and is one-hot'd below), so
    # the regression target is naturally included here — see docstring.
    num_cols = list(schema.numerical_cols)
    feat_cat = [c for c in schema.categorical_cols if c != target]

    parts_real: list[pd.DataFrame] = []
    parts_syn: list[pd.DataFrame] = []

    if num_cols:
        real_num, syn_num = _scaled_numerical_pair(
            train_df if train_df is not None else real_df,
            real_df,
            syn_df,
            num_cols,
        )
        parts_real.append(real_num)
        parts_syn.append(syn_num)

    if feat_cat:
        real_cat, syn_cat = _onehot_columns(real_df, syn_df, feat_cat, train_df=train_df)
        parts_real.append(real_cat)
        parts_syn.append(syn_cat)

    # Categorical target -> one-hot minus last class (dummy encoding). A
    # numerical (regression) target is already included via num_cols above.
    if target in schema.categorical_cols:
        real_tgt, syn_tgt = _onehot_columns(real_df, syn_df, [target], train_df=train_df)
        if real_tgt.shape[1] > 1:  # drop the last class column (reference category)
            real_tgt = real_tgt.iloc[:, :-1]
            syn_tgt = syn_tgt.iloc[:, :-1]
        parts_real.append(real_tgt)
        parts_syn.append(syn_tgt)

    if not parts_real:
        return pd.DataFrame(index=real_df.index), pd.DataFrame(index=syn_df.index)

    return (
        pd.concat(parts_real, axis=1).reset_index(drop=True),
        pd.concat(parts_syn, axis=1).reset_index(drop=True),
    )


def _onehot_columns(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    columns: list[str],
    train_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot ``columns`` over a fixed, **train-fitted** category set.

    Fitting on real+syn would let the held-out test split (the usual ``real_df``)
    decide the feature space every metric is computed in, and would give a
    generator that invents categories extra dimensions to live in. With a
    train-fitted encoder (``handle_unknown="ignore"``) an invented category maps
    to the all-zero row and is counted by the validity metric instead.
    """
    encoder = _make_onehot_encoder()
    basis = train_df if train_df is not None else pd.concat([real_df, syn_df], ignore_index=True)
    encoder.fit(basis[columns].astype(str))
    names = list(encoder.get_feature_names_out(columns))
    real_enc = pd.DataFrame(encoder.transform(real_df[columns].astype(str)), columns=names)
    syn_enc = pd.DataFrame(encoder.transform(syn_df[columns].astype(str)), columns=names)
    return real_enc, syn_enc


def ordinal_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    gen_df: pd.DataFrame,
    schema: EvalSchema,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = train_df[schema.all_columns].copy()
    test = test_df[schema.all_columns].copy()
    gen = gen_df[schema.all_columns].copy()

    if schema.numerical_cols:
        train_num, test_num, gen_num = _scaled_numerical_frames(
            train,
            [train, test, gen],
            schema.numerical_cols,
        )
        for df, scaled in zip((train, test, gen), (train_num, test_num, gen_num), strict=True):
            df[schema.numerical_cols] = scaled[schema.numerical_cols].to_numpy()

    if schema.categorical_cols:
        # Train-fitted; unseen (test or generated) values become -1.
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoder.fit(train[schema.categorical_cols].astype(str))
        for df in (train, test, gen):
            df[schema.categorical_cols] = encoder.transform(df[schema.categorical_cols].astype(str))

    return train.astype(float), test.astype(float), gen.astype(float)


def _scaled_numerical_pair(
    train_df: pd.DataFrame,
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    real_scaled, syn_scaled = _scaled_numerical_frames(train_df, [real_df, syn_df], columns)
    return real_scaled, syn_scaled


def _scaled_numerical_frames(
    train_df: pd.DataFrame,
    frames: list[pd.DataFrame],
    columns: list[str],
) -> list[pd.DataFrame]:
    scaler = StandardScaler().fit(train_df[columns].astype(float))
    return [
        pd.DataFrame(
            scaler.transform(frame[columns].astype(float)),
            columns=columns,
        ).reset_index(drop=True)
        for frame in frames
    ]


def discretize_for_ci(df: pd.DataFrame, schema: EvalSchema, n_bins: int = 5) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in schema.all_columns:
        if col in schema.categorical_cols:
            out[col] = df[col].astype(str)
            continue

        values = pd.to_numeric(df[col], errors="coerce")
        if values.nunique(dropna=True) <= n_bins:
            out[col] = values.fillna(values.median()).astype(str)
            continue

        try:
            out[col] = pd.qcut(values.rank(method="first"), q=n_bins, labels=False, duplicates="drop").astype(str)
        except ValueError:
            out[col] = pd.cut(values, bins=n_bins, labels=False, duplicates="drop").astype(str)
        out[col] = out[col].fillna("__MISSING__").astype(str)
    return out


def sample_to_size(df: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    if len(df) == size:
        return df.reset_index(drop=True)
    return df.sample(n=size, replace=len(df) < size, random_state=seed).reset_index(drop=True)


def _categorical_to_string(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("__MISSING__").astype(str)


def _make_onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _make_sparse_onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(
            handle_unknown="ignore", sparse_output=True, dtype=np.float64
        )
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=np.float64)

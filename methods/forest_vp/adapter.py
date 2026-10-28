# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Forest-VP: XGBoost-based diffusion (Jolicoeur-Martineau et al., AISTATS 2024).

Runs the authors' ``ForestDiffusion`` package. Two upstream facts shape this
adapter and are not negotiable:

* ``diffusion_type`` is fixed to ``'vp'``. The package's ``impute()`` opens with
  ``assert self.diffusion_type != 'flow'``, so the flow variant is
  generation-only. Pinning 'vp' is what lets one checkpoint serve both tasks;
  the flow variant would need a separate registry entry to be honest about it.
* With ``label_y`` set, upstream fits an independent model *per class*. The
  registry declares ``auxiliary_models="per_class"`` for exactly this reason,
  and the fit cost recorded for the method is the cost of all of them.

Upstream consumes a numeric matrix and is told which columns are categorical,
binary or integer. The encoding here is fitted on train only, and the inverse
mapping is stored in the checkpoint so generation and imputation decode through
the same table.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from otbench.methods.common import checkpoints as ckpt_utils
from otbench.methods.common.base import (
    GenerateContext,
    ImputationOutput,
    ImputeContext,
    MethodContext,
    PredictContext,
    PredictionOutput,
    TabularMethod,
    Task,
    TrainContext,
)
from otbench.methods.common.capabilities import UnsupportedRun
from otbench.methods.common.imputer_base import merge_masked_only
from otbench.methods.common.seeding import seed_everything
from otbench.methods.common.upstream import require_package

UPSTREAM_URL = "https://github.com/SamsungSAILMontreal/ForestDiffusion"

_MODEL_FILE = "forest_vp.joblib"
_META_FILE = "forest_vp.json"


class ForestVPMethod(TabularMethod):
    name = "forest_vp"
    tasks = frozenset({Task.GENERATION, Task.IMPUTATION})
    trainable = True
    # The official label_y route fits one forest family per target class.  The
    # sibling forest_vp_joint adapter disables this so that target masking is a
    # genuine conditional draw from one joint model rather than a label leak.
    condition_on_categorical_target = True
    checkpoint_runtime_packages = (
        "ForestDiffusion", "xgboost", "joblib", "numpy", "pandas", "scikit-learn",
    )
    require_checkpoint_runtime_fingerprint = True

    # ------------------------------------------------------------------ train
    def train(self, ctx: TrainContext) -> None:
        import joblib

        seed_everything(ctx.train_seed)
        schema = ctx.bundle.schema
        train = ctx.bundle.train.reset_index(drop=True)[schema.all_columns]

        codec = _Codec.fit(
            train, schema,
            condition_on_target=self.condition_on_categorical_target,
        )
        matrix = codec.encode(train)

        resolved_device = self.resolve_device(ctx)
        model = _build(self, codec, matrix, ctx.train_seed, resolved_device)
        gpu_hist = bool(self.cfg("model", "gpu_hist", True))

        ckpt_dir = Path(ctx.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "codec": codec}, ckpt_dir / _MODEL_FILE)
        with (ckpt_dir / _META_FILE).open("w", encoding="utf-8") as f:
            json.dump({
                "diffusion_type": "vp",
                "diffusion_type_reason": "upstream impute() asserts the type is not 'flow'; "
                                         "'vp' is the only variant that supports both tasks",
                "n_auxiliary_models": codec.n_label_models,
                "label_conditioned": codec.label_column is not None,
                "family_variant": (
                    "label_y_per_class" if codec.label_column is not None else "strict_joint"
                ),
                "execution_backend": "xgboost_cuda" if gpu_hist else "xgboost_cpu",
                "resolved_device": resolved_device,
                "gpu_hist": gpu_hist,
                "tree_method": str(self.cfg("model", "tree_method", "hist")),
                "xgboost_nthread": int(self.cfg("model", "xgboost_nthread", 1)),
                "encoded_columns": codec.columns,
                "categorical_indexes": codec.categorical_indexes,
                "integer_indexes": codec.integer_indexes,
                "binary_indexes": codec.binary_indexes,
            }, f, indent=2, ensure_ascii=False)
        ckpt_utils.write_metadata(
            ckpt_dir, self.name, ctx.bundle, self.config, ctx.train_seed,
            artifacts={"model": _MODEL_FILE, "meta": _META_FILE},
            runtime_packages=self.checkpoint_runtime_packages,
        )

    # --------------------------------------------------------------- generate
    def generate(self, ctx: GenerateContext) -> pd.DataFrame:
        model, codec = self._load(ctx)
        n_samples = int(ctx.n_samples or len(ctx.bundle.train))
        batch_rows = _positive_batch_rows(
            self.cfg("generate", "batch_rows", 1024), n_samples,
        )

        np.random.seed(int(ctx.seed))  # upstream samples from the global state
        chunks = [
            np.asarray(model.generate(batch_size=stop - start), dtype=float)
            for start, stop in _row_slices(n_samples, batch_rows)
        ]
        raw = np.concatenate(chunks, axis=0) if chunks else np.empty((0, len(codec.columns)))
        return codec.decode(raw).head(n_samples).reset_index(drop=True)

    # ----------------------------------------------------------------- impute
    def impute(self, ctx: ImputeContext) -> pd.DataFrame:
        return self.impute_draws(ctx).draws[0]

    def impute_draws(self, ctx: ImputeContext) -> ImputationOutput:
        model, codec = self._load(ctx)
        masked = ctx.masked_test.reset_index(drop=True)[codec.schema_columns]
        k = max(1, int(ctx.trials))

        matrix = codec.encode(masked, allow_missing=True)
        label = codec.label_codes(masked) if codec.label_column else None
        batch_rows = _positive_batch_rows(
            self.cfg("impute", "batch_rows", 256), len(matrix),
        )

        np.random.seed(int(ctx.seed))
        matrix = _to_internal(model, matrix)
        chunks = []
        for start, stop in _row_slices(len(matrix), batch_rows):
            chunk_label = label[start:stop] if label is not None else None
            chunk = model.impute(
                k=k,
                X=matrix[start:stop],
                label_y=chunk_label,
                repaint=bool(self.cfg("impute", "repaint", True)),
                r=int(self.cfg("impute", "repaint_r", 5)),
                j=float(self.cfg("impute", "repaint_j", 0.1)),
            )
            chunk = np.asarray(chunk, dtype=float)
            if chunk.ndim == 2:  # upstream drops the leading axis when k == 1
                chunk = chunk[None, ...]
            if chunk.shape[0] != k:
                raise RuntimeError(
                    f"ForestDiffusion returned {chunk.shape[0]} draw(s), expected {k}."
                )
            chunks.append(chunk)
        imputed = np.concatenate(chunks, axis=1)

        # Upstream's postprocessing (inverse min-max, integer rounding, clipping
        # to the train range) touches every cell, not just the imputed ones, so
        # the completion is merged back under the mask exactly as the sklearn
        # imputers do. Observed values are never rewritten.
        draws = [merge_masked_only(masked, codec.decode(imputed[i]), ctx.mask)
                 for i in range(imputed.shape[0])]
        return ImputationOutput.from_frames(draws, metadata={
            "source": "native",
            "draw_strategy": "upstream ForestDiffusion.impute(k=K) with REPAINT; the K "
                             "completions are joint draws, not K independent runs",
            "n_draws": len(draws),
            "diffusion_type": "vp",
        })

    # --------------------------------------------------------------- predict
    def predict(self, ctx: PredictContext) -> PredictionOutput:
        """Predict by masking the target in the strict-joint family member.

        This method deliberately exists on the shared implementation but is
        registered only by ``forest_vp_joint``.  Calling it on the label_y
        variant would condition on the answer, so fail before sampling.
        """
        _, codec = self._load(ctx)
        if codec.label_column is not None:
            raise RuntimeError(
                "forest_vp label_y checkpoints cannot predict a masked target; "
                "use the independently trained forest_vp_joint method."
            )

        schema = ctx.bundle.schema
        test = ctx.bundle.test.reset_index(drop=True)[schema.all_columns]
        mask = pd.DataFrame(False, index=test.index, columns=test.columns)
        mask[schema.target_col] = True
        imputation = self.impute_draws(ImputeContext(
            bundle=ctx.bundle,
            config=ctx.config,
            checkpoint_dir=ctx.checkpoint_dir,
            train_seed=ctx.train_seed,
            seed=ctx.seed,
            device=ctx.device,
            masked_test=test.mask(mask),
            mask=mask,
            trials=max(1, int(ctx.n_samples)),
        ))

        target = schema.target_col
        shared = {
            "source": "native",
            "prediction_route": "strict joint model with target column masked",
            "n_samples": imputation.n_draws,
            "diffusion_type": "vp",
        }
        if schema.task_type == "regression" or target in set(schema.numerical_cols):
            samples = np.column_stack([
                pd.to_numeric(draw[target], errors="coerce").to_numpy(float)
                for draw in imputation.draws
            ])
            return PredictionOutput(
                y_pred=samples.mean(axis=1),
                y_samples=samples,
                metadata={**shared, "point_rule": "mean of native target-mask draws"},
            )

        labels = list(codec.categories[target])
        sampled = np.column_stack([
            draw[target].astype(str).to_numpy() for draw in imputation.draws
        ])
        proba = np.column_stack([(sampled == label).mean(axis=1) for label in labels])
        codes = proba.argmax(axis=1)
        return PredictionOutput(
            y_pred=np.asarray([labels[int(code)] for code in codes], dtype=object),
            y_proba=proba,
            y_score=proba[:, 1] if len(labels) == 2 else None,
            labels=labels,
            threshold=0.5 if len(labels) == 2 else None,
            threshold_source="fixed_0.5" if len(labels) == 2 else "argmax",
            metadata={**shared, "probability_rule": "class frequency over native draws"},
        )

    # -------------------------------------------------------------- internals
    def _load(self, ctx: MethodContext):
        import joblib

        payload = joblib.load(Path(ctx.checkpoint_dir) / _MODEL_FILE)
        return payload["model"], payload["codec"]


def _build(
    method: ForestVPMethod,
    codec: "_Codec",
    matrix: np.ndarray,
    seed: int,
    resolved_device: str,
):
    forest = require_package("ForestDiffusion", method=method.name, url=UPSTREAM_URL,
                             extra="imputation")
    label = codec.training_label_codes
    gpu_hist = bool(method.cfg("model", "gpu_hist", True))
    if gpu_hist and not str(resolved_device).startswith("cuda"):
        raise RuntimeError(
            f"{method.name} model.gpu_hist=true requires a CUDA device, but resolved "
            f"device is {resolved_device!r}. Set model.gpu_hist=false for an explicit CPU run."
        )
    n_jobs = int(method.cfg("model", "n_jobs", 1))
    if gpu_hist and n_jobs != 1:
        raise ValueError(
            f"{method.name} GPU training requires model.n_jobs=1; upstream uses this "
            "for joblib-parallel booster fits that would otherwise contend for one GPU."
        )
    xgboost_nthread = int(method.cfg("model", "xgboost_nthread", 1))
    if xgboost_nthread <= 0:
        raise ValueError("model.xgboost_nthread must be a positive integer")
    return forest.ForestDiffusionModel(
        matrix,
        label_y=label,
        # Fixed, not configurable: the imputation path asserts on it.
        diffusion_type="vp",
        n_t=int(method.cfg("model", "n_t", 50)),
        duplicate_K=int(method.cfg("model", "duplicate_K", 100)),
        n_estimators=int(method.cfg("model", "n_estimators", 100)),
        max_depth=int(method.cfg("model", "max_depth", 7)),
        eta=float(method.cfg("model", "eta", 0.3)),
        tree_method=str(method.cfg("model", "tree_method", "hist")),
        gpu_hist=gpu_hist,
        cat_indexes=codec.categorical_indexes,
        int_indexes=codec.integer_indexes,
        bin_indexes=codec.binary_indexes,
        n_jobs=n_jobs,
        nthread=xgboost_nthread,
        seed=int(seed),
    )


def _positive_batch_rows(value, total_rows: int) -> int:
    """Validate a deterministic row cap used by the memory-bounded adapters."""
    rows = int(value)
    if rows <= 0:
        raise ValueError(f"batch_rows must be positive, got {value!r}")
    return min(rows, max(1, int(total_rows)))


def _row_slices(total_rows: int, batch_rows: int):
    for start in range(0, int(total_rows), int(batch_rows)):
        yield start, min(start + int(batch_rows), int(total_rows))


def _to_internal(model, matrix: np.ndarray) -> np.ndarray:
    """Put a raw coded matrix into upstream's internal space.

    ``impute(X=...)`` does not preprocess X — it expects the representation the
    model was fitted on: categorical columns dummy-coded with the first level
    dropped, then min-max scaled to [-1, 1]. Passing raw columns instead is
    silently wrong whenever any categorical column has more than two levels,
    because the widths no longer match.

    The dummies are built from the fitted ``X_names_after`` rather than by
    calling ``pd.get_dummies`` again. Re-running it on the test frame would drop
    whichever level happens to come first *there*, so a column could encode a
    different level than it did during training.
    """
    matrix = np.asarray(matrix, dtype=float)
    cat_indexes = list(getattr(model, "cat_indexes", []) or [])
    if not cat_indexes:
        return model.scaler.transform(matrix)

    names_after = [str(name) for name in model.X_names_after]
    n_rows = matrix.shape[0]
    encoded = np.zeros((n_rows, len(names_after)), dtype=float)

    for position, name in enumerate(names_after):
        if "_" in name:
            prefix, level = name.split("_", 1)
            column = int(prefix)
            values = matrix[:, column]
            hit = values == float(level)
            encoded[:, position] = np.where(hit, 1.0, 0.0)
            # A missing category must stay missing in every one of its dummies,
            # otherwise it decodes as the dropped reference level instead of
            # being imputed.
            encoded[np.isnan(values), position] = np.nan
        else:
            encoded[:, position] = matrix[:, int(name)]

    return model.scaler.transform(encoded)


class _Codec:
    """Train-fitted mapping between the schema frame and upstream's matrix.

    Upstream wants floats plus index lists saying which columns are categorical,
    binary or integer. When a label column is used it is passed separately as
    ``label_y`` and upstream appends it back as the last column of its output,
    so the decoded column order is features-then-label, not schema order.
    """

    def __init__(self, schema_columns, columns, categories, integer_columns,
                 label_column, label_categories, label_is_numeric):
        self.schema_columns = list(schema_columns)
        self.columns = list(columns)
        self.categories = categories                # column -> list of values
        self.integer_columns = list(integer_columns)
        self.label_column = label_column
        self.label_categories = label_categories
        self.label_is_numeric = label_is_numeric
        self.training_label_codes = None

    @classmethod
    def fit(cls, train: pd.DataFrame, schema, *, condition_on_target: bool = True) -> "_Codec":
        categorical = [c for c in schema.categorical_cols if c in train.columns]
        numerical = [c for c in schema.numerical_cols if c in train.columns]

        # Upstream conditions on a single discrete label. Use the target when it
        # is categorical; a numerical target is just another column.
        label_column = (
            schema.target_col
            if condition_on_target and schema.target_col in categorical
            else None
        )
        feature_columns = [c for c in schema.all_columns if c != label_column]

        categories = {c: sorted(train[c].dropna().astype(str).unique())
                      for c in categorical if c != label_column}
        integer_columns = [
            c for c in numerical
            if train[c].dropna().apply(lambda v: float(v).is_integer()).all()
            and train[c].notna().any()
        ]
        label_categories = (sorted(train[label_column].dropna().astype(str).unique())
                            if label_column else [])

        codec = cls(schema.all_columns, feature_columns, categories, integer_columns,
                    label_column, label_categories, False)
        if label_column:
            codec.training_label_codes = codec.label_codes(train)
        return codec

    # -- index lists upstream asks for ------------------------------------
    @property
    def categorical_indexes(self) -> list[int]:
        return [i for i, c in enumerate(self.columns)
                if c in self.categories and len(self.categories[c]) > 2]

    @property
    def binary_indexes(self) -> list[int]:
        return [i for i, c in enumerate(self.columns)
                if c in self.categories and len(self.categories[c]) <= 2]

    @property
    def integer_indexes(self) -> list[int]:
        return [i for i, c in enumerate(self.columns) if c in self.integer_columns]

    @property
    def n_label_models(self) -> int:
        return max(1, len(self.label_categories))

    # -- encode / decode ---------------------------------------------------
    def label_codes(self, frame: pd.DataFrame) -> np.ndarray | None:
        if not self.label_column:
            return None
        index = {value: i for i, value in enumerate(self.label_categories)}
        values = frame[self.label_column].astype(str)
        # A test row whose label was never seen in train cannot be conditioned
        # on; upstream has no model for it, so it falls back to the first class
        # and the fallback is visible in the metadata rather than silent.
        return np.asarray([index.get(v, 0) for v in values], dtype=int)

    def encode(self, frame: pd.DataFrame, allow_missing: bool = False) -> np.ndarray:
        matrix = np.empty((len(frame), len(self.columns)), dtype=float)
        for i, column in enumerate(self.columns):
            values = frame[column]
            if column in self.categories:
                index = {value: code for code, value in enumerate(self.categories[column])}
                codes = [np.nan if pd.isna(v) else float(index.get(str(v), np.nan))
                         for v in values]
                matrix[:, i] = codes
            else:
                matrix[:, i] = pd.to_numeric(values, errors="coerce").to_numpy(float)
        if not allow_missing and np.isnan(matrix).any():
            # Upstream treats NaN as "impute me". A NaN reaching the fit path
            # would silently change what the model is trained on.
            raise UnsupportedRun(
                "forest_vp: the training frame has missing cells, which upstream would "
                "interpret as imputation targets. This adapter does not silently pre-impute "
                "training data because that would define a different method."
            )
        return matrix

    def decode(self, matrix: np.ndarray) -> pd.DataFrame:
        matrix = np.asarray(matrix, dtype=float)
        frame = pd.DataFrame(index=range(matrix.shape[0]))
        for i, column in enumerate(self.columns):
            values = matrix[:, i]
            if column in self.categories:
                levels = self.categories[column]
                codes = np.clip(np.rint(np.nan_to_num(values)), 0, len(levels) - 1).astype(int)
                frame[column] = [levels[c] for c in codes]
            elif column in self.integer_columns:
                frame[column] = np.rint(values).astype("int64")
            else:
                frame[column] = values

        if self.label_column:
            # Upstream appends label_y as the final column of its output.
            codes = np.clip(np.rint(np.nan_to_num(matrix[:, -1])), 0,
                            len(self.label_categories) - 1).astype(int)
            frame[self.label_column] = [self.label_categories[c] for c in codes]

        return frame[self.schema_columns]

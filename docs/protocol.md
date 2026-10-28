# Evaluation protocol

The registry describes implementation support; it is broader than any paper's
fixed analysis cohort. Dataset/task/column capabilities and available native
outputs determine which runs and metrics are meaningful. Retain metric status and
coverage when summarizing results.

## Repetitions and information access

- Generation uses independent checkpoint and sampling seeds 0, 1, 2. Keep both raw
  and aligned samples; validity, memorization, and privacy read the raw artifact.
- Imputation uses MCAR, MAR, MNAR, and `MIXED_MCAR_MAR_MNAR` at rates 0.1, 0.3, 0.5.
  Train, selection, mask, and imputation seeds are separate; the repeated protocol
  pairs them as `(s, s, s, s)` for `s = 0, 1, 2`. Native stochastic draws within each
  setting are distinct from those independent seed repetitions.
- Prediction uses independent checkpoint seeds 0, 1, 2. Sampling trials are an
  explicit method/protocol choice; one trial is descriptive and does not establish
  sampling stability.

`target_observed` and `feature_only` expose different information in imputation;
report them separately. Rubin validity and label stability require declared
scientific/stability protocols; `auto` applies the declared derivation rule.
An absent protocol is not an estimable score.

## Data and artifacts

Dataset schemas specify column types and CSV missing-value encoding. Typed I/O
preserves categorical strings such as `01` and `NA` while keeping actual missing
values separate. Shared mask bundles bind data, schema, column selection, and mask
content; stored artifact hashes validate the bundle before reuse.

Checkpoint/result reuse requires matching data, schema, config, code, runtime,
pretrained resources, and output artifacts. `--skip-existing` requests that
validated reuse. See [artifacts and recovery](artifacts.md) for stored contracts,
checkpoint archives, and recovery after inference.

Published summaries must preserve `skipped`/`failed` and non-estimable metrics,
not silently average available subsets as complete axes. A rebuilt store matches
the reference paper splits only after reference verification passes; see
[installation](installation.md).

## Method identities

`bn` outputs discretized-bin indices. `bn_original_units` decodes saved train-fitted
bin edges into original-unit midpoints. Their checkpoints and results are separate;
choose the intended identity explicitly in a fixed-method comparison.

Pretrained files are content-hashed on first use; within one process that identity
is cached only while their file-stat identity is unchanged. Run provenance
conservatively fingerprints the method tree and installed runtime, so unrelated
adapter or package edits can invalidate reuse.

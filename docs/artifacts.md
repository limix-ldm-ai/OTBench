# Artifacts and recovery

Paths below are relative to the selected workspace. `--workspace` takes precedence
over `OTBENCH_WORKSPACE`, then the current directory; explicit `--store`,
`--outputs`, and `--checkpoints` override their individual roots. A runner's returned
JSON records its concrete artifact paths and fingerprint suffix.

| Location | Purpose |
| --- | --- |
| `tabdata/store/{real,scm}/<dataset>/` | Source/split metadata, schema, and train/test CSVs |
| `checkpoints/<group>/<dataset>/train_seed<seed>/` | Fitted model artifacts and `metadata.json` |
| `checkpoints/<group>/pretrained/` | Externally provisioned resources, possibly shared across method identities |
| `outputs/generation/<method>/` | Immutable `gen_raw/`, aligned `gen/`, generation manifests, and metrics |
| `outputs/prediction/<method>/` | Prediction CSVs and metric records |
| `outputs/imputation/<method>/imputed/` | One CSV per committed imputation draw |
| `outputs/imputation/<method>/metrics/<dataset>/<setting>/run_<fingerprint>/` | `inference.json`, `run_state.json`, and enabled `metrics.json` |
| `outputs/imputation/masks/<dataset>/<mask-setting>/data_<identity>/` | Immutable selection, mask, masked input, and bundle manifest |

`<setting>` records scenario, rate, and train/selection/mask/impute seeds.
`<mask-setting>` records scenario, rate, selection seed, and mask seed. The
`data_<identity>` namespace also binds source data, schema, selection, and mask
content. `bundle.json` records that identity and hashes `selection.json`,
`mask.json`, and `masked.csv`.

## Provenance and checkpoints

Checkpoint metadata uses version 4, result-cache provenance version 2, imputation
evaluation metadata version 2, and mask-bundle identity version 2. Reuse validates
the recorded contract and artifact hashes. A nonempty file alone is insufficient.

Source and actual pretrained-resource identities participate in provenance. File
content hashes are cached within a process only while file-stat identity is
unchanged. Method source and installed-runtime identity are conservative: an
unrelated method or package edit can invalidate reuse.

Before retraining a nonempty checkpoint directory, the runner holds its checkpoint
lock and archives the complete directory, including metadata, under
`checkpoints/<group>/<dataset>/.history/train_seed<seed>-<UTC-time>-<uuid>/`.
It then trains at the requested path. The archive remains if training fails;
archive retention and disk usage are user-managed.

## Imputation commit and recovery

The runner validates each draw's shape, columns, preserved observations, and
finite required completions before accepting it. It commits draw CSVs and an
artifact-validated `inference.json` before metric setup/evaluation. The per-setting
state distinguishes `running`, `inferred`, `succeeded`, and `failed`.

If evaluation fails after inference was committed, the state is `failed` while the
validated draws remain available. Retry the same request with `--skip-existing`:
when the full request fingerprint and inference artifacts match, the runner loads
those draws and resumes evaluation without sampling again. Different data, config,
code, resources, or evaluation arguments can invalidate recovery. Grid preparation
failures mark affected settings as failed after preserving their committed draws.

Both evaluations are enabled by default. `--no-direct-eval` disables direct
metrics while downstream evaluation can still run; `--no-downstream-eval` disables
downstream work. Supplying both flags validates and commits imputed draws without
metric evaluation or a new `metrics.json`. Selecting a metric family limits enabled
work; unknown metric selectors fail during planning.

A succeeded run and an `ok` metric are different levels of evidence. Metric records
retain `ok`, `skipped`, and `failed`; skipped/non-estimable values are not zeros.
See [the evaluation protocol](protocol.md) before aggregating results.

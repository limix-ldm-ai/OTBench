# Tasks and outputs

Install the package and build the smoke dataset as shown in the README. These
examples use that offline mixed-schema dataset and CPU baseline methods. Runtime
metadata records the resolved configuration and protocol; benchmark-quality model
training needs the intended method parameters and evaluation cohort.

```bash
# Predictions and metrics.
otbench --task prediction --method random_forest --dataset breast_cancer \
  --workspace ./work --device cpu --seed 0

# Immutable raw samples, an aligned evaluation view, and metrics.
otbench --task generation --method independent_marginal --dataset breast_cancer \
  --workspace ./work --device cpu --seed 0 --eval

# One missingness setting, with direct and downstream evaluation.
otbench --task imputation --method simple_mean_mode --dataset breast_cancer \
  --workspace ./work --device cpu --scenario MCAR --rate 0.3 --seed 0
```

Use `otbench --help` for task controls. Flags for a different task are rejected;
`--metric` selects evaluation metrics for the chosen task.

## Workspace contents

| Location under the workspace | Contents |
| --- | --- |
| `tabdata/store/{real,scm}/<dataset>/` | Train/test CSVs, schema and source/split metadata |
| `checkpoints/<group>/<dataset>/train_seed<seed>/` | Fitted artifacts and checkpoint provenance |
| `checkpoints/<group>/pretrained/` | Externally provisioned foundation resources |
| `outputs/generation/<method>/` | `gen_raw/`, aligned `gen/`, generation manifests, metrics |
| `outputs/imputation/<method>/` | Imputed draws and per-setting metric/artifact records |
| `outputs/imputation/masks/` | Shared mask bundles identified by the input/protocol contract |
| `outputs/prediction/<method>/` | Prediction CSVs and metric records |

The returned JSON contains the concrete paths, including any fingerprint suffix.
Metric records distinguish `ok`, `skipped`, and `failed`; skipped values are not
zeros. A successful runner invocation does not make every optional metric estimable.

For imputation, `--trials` controls draws within one setting, while independent
seed repetitions measure repeatability. The grid entry keeps one method instance
alive across settings so eligible conditional models can reuse fits:

```bash
python -m otbench.imputation.grid_cli --method simple_mean_mode \
  --dataset breast_cancer --workspace ./work --device cpu \
  --scenarios MCAR MAR --rates 0.1 0.3 --seed 0
```

See [artifact versions and recovery](artifacts.md) for committed draws and cache
behavior, [the protocol](protocol.md) before pooling results, and
[the launchers](../scripts/README.md) for repeated or distributed execution.

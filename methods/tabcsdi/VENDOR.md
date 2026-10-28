# TabCSDI — vendored upstream

* Paper: Shuhan Zheng, Nontawat Charoenphakdee. *Diffusion Models for Missing
  Value Imputation in Tabular Data.* NeurIPS 2022 Table Representation Learning
  workshop.
* Source: <https://github.com/pfnet-research/TabCSDI> (itself built on
  <https://github.com/ermongroup/CSDI>)
* Snapshot identity: `sha256:682c3edb0d6f62872…` over the sorted vendored
  source paths and bytes; license: MIT (`model/LICENSE.md`).

## What is vendored

| File | Upstream path | Changes |
| --- | --- | --- |
| `model/main_model_table.py` | `src/main_model_table.py` | `from src.diff_models_table import` → relative |
| `model/diff_models_table.py` | `src/diff_models_table.py` | none |
| `model/utils_table.py` | `src/utils_table.py` | none (kept for reference; the adapter does not call it) |
| `model/upstream_census_analog.yaml` | `config/census_onehot_analog.yaml` | none |
| `model/LICENSE.md` | `LICENSE.md` | none |

The model, the noise schedule, `get_randmask`, `impute` and `evaluate` are all
upstream's and are called directly.

## What is not vendored

`dataset_census_*.py` and `exe_census_*.py`. Upstream ships the encoding as a
per-dataset script that reads a fixed CSV, writes pickles to a fixed directory
and hard-codes the categorical column indices, so it cannot be reused. The
adapter rebuilds the analog-bits variant of `dataset_census_analog.py`:

* `bits_num = int(log2(cat_num)) + 1` bits per categorical column, category
  indices numbered from 1 so the all-zero pattern stays free for "missing", bit
  values mapped to `2b - 1 ∈ {-1, +1}`;
* numerical columns scaled `(x - (min - 1)) / (max - min + 1)` with min/max over
  observed training cells.

Decoding is by nearest codeword rather than upstream's per-bit threshold at 0.
Thresholding can produce a pattern no category owns, which upstream tolerates
because `evaluate_analog` only counts exact matches and never has to name the
resulting category; producing a dataframe does.

`utils_table.train` is replaced by an equivalent loop in the adapter: same Adam
settings, same weight decay, same four-milestone `MultiStepLR`, minus the
TensorBoard/tqdm/`torch.save` side effects and the validation branch that only
prints.

## Dependencies

`torch` only.

# DiffPuter — vendored upstream

* Paper: Hengrui Zhang et al. *DiffPuter: Empowering Diffusion Models for
  Missing Data Imputation.* ICLR 2025.
* Source: <https://github.com/hengruizhang98/DiffPuter>
* Snapshot identity: `sha256:6bac335ce53a7a937…` over the sorted vendored
  source paths and bytes; license: MIT (`model/LICENSE`).

## What is vendored

| File | Upstream path | Changes |
| --- | --- | --- |
| `model/model.py` | `model.py` | `from diffusion_utils import EDMLoss` → relative |
| `model/diffusion_utils.py` | `diffusion_utils.py` | none |
| `model/LICENSE` | `LICENSE` | none |

`MLPDiffusion`, `Precond`, `EDMLoss` and the RePaint-style `impute_mask` sampler
are upstream's and are called directly.

## What is not vendored

`main.py` and `dataset.py`. Both are experiment scripts: `main.py` is built
around `argparse` and hard-coded `ckpt/…` paths, and `dataset.py` loads
pre-generated `.npy` masks and per-dataset JSON metadata from a fixed layout.

The adapter reproduces `main.py`'s EM loop (M-step = train the denoiser on the
current completion; E-step = `impute_mask`, averaged over `num_trials`), with
upstream's optimiser, `ReduceLROnPlateau` and patience-500 early stopping. Two
differences, both stated in the adapter docstring:

* **Warm start.** Upstream builds a fresh network each round and reloads the
  best checkpoint from disk. The adapter warm-starts the next M-step from the
  previous round's weights; `em.warm_start = false` restores the cold restart.
* **Inverse standardisation.** `main.py` un-standardises only the categorical
  block, because `get_eval` compares numerical columns in standardised space.
  Returning a dataframe requires inverting the whole matrix, so the adapter
  does.

The encoding follows `dataset.py`: categorical columns as
`(n_levels - 1).bit_length()` analog bits in `{0, 1}`, numerical columns as-is,
then `(x - mean) / std / 2` over observed cells. Upstream never has to decode
the bits (its metric ignores categorical accuracy); the adapter decodes by
nearest codeword.

## Dependencies

`torch`, `scipy` (imported by `diffusion_utils`).

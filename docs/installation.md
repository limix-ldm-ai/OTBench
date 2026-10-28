# Installation and pretrained weights

Python **3.12** is the supported interpreter for the package and launchers. Batch
scripts also require Bash 4 or newer; the smoke driver uses GNU `timeout`.
Use `requirements/constraints.txt` with every supported installation command to
preserve the validated numerical stack. The project's dependency metadata reads
`requirements/core.txt`; the `test`, `data`, and `metrics` extras read their matching
requirements files.

## Select an environment

| Need | Install after activating the environment |
| --- | --- |
| CPU baseline tasks and core tests | `python -m pip install -c requirements/constraints.txt -e '.[test]'` |
| OpenML parquet and supported SCM dataset builders | `python -m pip install -c requirements/constraints.txt -r requirements/data.txt` |
| SDMetrics parity and SCM metric tests | `python -m pip install -c requirements/constraints.txt -r requirements/metrics.txt` |
| Traditional/deep prediction packages | `python -m pip install -c requirements/constraints.txt -r requirements/prediction.txt` |
| Diffusion packages | Install the compatible PyTorch/CUDA build first, then `requirements/diffusion.txt` |
| Synth-plugin generation packages | `python -m pip install -c requirements/constraints.txt -r requirements/generation.txt` |
| Foundation inference packages | Install compatible PyTorch first, then `requirements/foundation.txt` |
| Additional imputers | Follow the explicit installation notes in `requirements/imputation.txt` |

These are method families, not a promise that every upstream package can coexist
in one environment. In particular, TabPFN v2/TabEBM uses the pinned TabPFN 2.1.2
API; TabPFN unsupervised uses its separate v3 extension stack. HyperImpute and
TabImpute have documented metadata conflicts and explicit `--no-deps` steps in
the imputation requirements. LimiX-2 includes a separate upstream environment
specification under `methods/limix2/model/`. Use isolated environments for those
stacks; retain their method identity and parameters when comparing results.

CARD and ReMasker upstream implementation files are not distributed. Follow
`methods/card/model/README.md` and `methods/remasker/model/README.md` to obtain
source directly from their project pages. Local availability does not grant
permission to include those files in a redistributed checkout, wheel, or sdist.
See [licensing](licensing.md) for the project and third-party terms.

Choose the PyTorch wheel/index for the actual CUDA allocation; no CUDA toolkit or
GPU memory size is inferred by this project. A selected method may need both its
task-family requirements and the diffusion/foundation requirements. See the
[generated reference](methods.md) for the actual optional import modules.

## Pretrained resources

Weights are never bundled or silently substituted. The default location is
`<workspace>/checkpoints/<weight-group>/pretrained/`. Resource names, config keys,
environment overrides, and shared weight groups are declared once in
`methods/common/resources.py`; adapters, readiness, and cache provenance use that
same description. Absolute method config paths and supported environment overrides
can point to an existing local resource. `--checkpoints` redirects the root.

```bash
python -m otbench.methods.common.readiness tabpfn --checkpoints ./work/checkpoints
```

This reports missing packages/files without importing a model or downloading
anything. `READY` establishes availability only; backend tests establish import/API
compatibility. Obtain each resource from the upstream linked in
`methods/common/registry.py`, using its access process and terms. An unused missing
prediction head need not prevent a runtime call that uses another head; whole-method
readiness conservatively checks all declared heads.

## Rebuild the evaluation cohort

`tabdata/manifest.yaml` describes 50 OpenML datasets and six SCMs, without rows.
Install the data requirements before running the builder. The reference manifest
does not include reference split seeds. The builder uses a declared OpenML target
or requires a unique target candidate from the recorded columns. It defaults to
seed 0, test fraction 0.2. A rebuilt
store reproduces paper splits only after `--verify-reference` passes.

The `healthcare`, `mehra`, and `sangiovese` SCMs require external
`tabdata/raw/<name>.csv` and `<name>.arcs.json` inputs from their network models;
those inputs/exporters are absent from this release. An existing compatible store
can always be selected with `--store`.

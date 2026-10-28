# OTBench

OTBench evaluates tabular **generation, imputation, and prediction** through shared
method, dataset, metric, and artifact contracts.

The source includes runners, adapters, metrics, dataset manifests, and launchers.
Datasets, pretrained weights, measured results, and paper analyses are supplied
separately.

Original OTBench code is distributed under the
[Apache License 2.0](LICENSE.txt). Third-party components retain their own licenses;
see [NOTICE.md](NOTICE.md) and the [licensing guide](docs/licensing.md).

## Install

Use **Python 3.12**. The default installation supports CPU baselines; optional model
families have separate requirements and may need isolated environments.

### Install from PyPI

Install the published package without cloning the repository:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install otbench-tabular
otbench --help
otbench --list-methods
```

The PyPI distribution is named `otbench-tabular`; the Python package and command
are named `otbench`. You can also invoke the CLI with `python -m otbench`.
The `data` and `metrics` extras add dependencies for the supported OpenML parquet /
SCM dataset builders and optional metrics, respectively:

```bash
python -m pip install 'otbench-tabular[data]'
python -m pip install 'otbench-tabular[metrics]'
```

Datasets and pretrained weights are not included in the package. See the offline
CPU example below for a first run that needs neither downloads nor a GPU.

### Install from source

For development, tests, or the validated numerical dependency stack, use a checkout
with the repository's constraints:

```bash
git clone https://github.com/Erisura/OTBench.git
cd OTBench
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements/constraints.txt -e '.[test]'
```

[Installation and weights](docs/installation.md) explains data dependencies,
optional method families, CUDA, and environment requirements.

## Run an offline CPU example

After either installation, build the bundled scikit-learn smoke dataset and
predict its target. Resolve the manifest from the installed package so the example
also works outside a source checkout. The smoke dataset is separate from the paper
cohort.

```bash
python -m otbench.tabdata.build_datasets \
  --manifest "$(python -c 'from importlib.resources import files; print(files("otbench.tabdata").joinpath("manifest_smoke.yaml"))')" \
  --store ./work/tabdata/store
otbench --task prediction --method random_forest --dataset breast_cancer \
  --workspace ./work --seed 0 --device cpu
```

The command prints JSON with artifact paths. Inspect `work/outputs/prediction/`
for predictions and metric records; fitted models go under `work/checkpoints/`.
See [tasks and outputs](docs/tasks.md) for generation and imputation examples.

## Choose a method or protocol

```bash
otbench --list-methods
otbench --help
```

- [Method reference](docs/methods.md): task support, optional packages, and resources.
- [Evaluation protocol](docs/protocol.md): seeds, draws, information access, and metric statuses.
- [Artifacts and recovery](docs/artifacts.md): stored outputs, provenance, and inference recovery.
- [Running at scale](docs/running-at-scale.md): launchers, workspace paths, and strict status checks.
- [Contributing](CONTRIBUTING.md): architecture, adding a method, tests, and distribution checks.

CARD and ReMasker upstream implementations are excluded because their inspected
source snapshots did not provide redistribution licenses. Their directories contain
instructions for obtaining the code directly from upstream for local use.

`--workspace` selects the artifact workspace; otherwise `OTBENCH_WORKSPACE`, then
the current directory, is used. Explicit `--store`, `--outputs`, and `--checkpoints`
override their individual roots.

## Validate

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  -m 'not optional_backend and not optional_metric'
```

This runs core contracts without training optional model backends. To exercise
installed backends and optional metric packages, run the full `tests/` suite.
Missing optional packages or external weights skip; an installed backend with a
broken import or API fails. Tests use synthetic fixtures; full training, dataset
downloads, and multi-node deployment checks are separate work.

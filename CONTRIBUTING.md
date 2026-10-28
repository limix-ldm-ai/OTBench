# Contributing

Use Python 3.12 and keep all runtime artifacts in a selected workspace. The core
runs offline on synthetic fixtures and scikit-learn datasets; model backends,
weights, and deployment checks are optional layers.

## Architecture and adding a method

The dependency direction is dataset/schema contracts → method contracts and
registry → task runners → metric/artifact records. `methods/common/base.py` owns
method contexts and outputs; `methods/common/registry.py` owns declared support.
The shared CLI in `run.py` dispatches to the three task runners. The imputation
grid is a separate execution strategy that keeps reusable conditional fits alive.

To add a method, implement an adapter and `config.toml`, register its capabilities
and upstream identity, declare optional imports in `methods/common/readiness.py`,
and declare external resources in `methods/common/resources.py`. Preserve native
output semantics and information access. Add contract tests through the real task
path and an isolated tiny smoke budget; do not substitute a fallback algorithm.
Regenerate `docs/methods.md` with:

```bash
python -m otbench.methods.common.readiness --format markdown > docs/methods.md
```

Keep comments focused on non-obvious behavior and protocol decisions. Document
public arguments from their actual implementation and retain upstream attribution
and license notices.

## Validation layers

```bash
python -m pip install -c requirements/constraints.txt -e '.[test]' build wheel
python -m pytest -p no:cacheprovider -m 'not optional_backend and not optional_metric'
# After provisioning the selected optional backends and resources:
python -m pytest -p no:cacheprovider tests
```

`optional_backend` covers real backend training/inference; `optional_metric`
covers third-party metric parity and SCM tests. Missing packages/resources may
skip. Installed backends that fail to import or execute must fail, so environment
breakage remains visible. `readiness` does not catch backend exceptions.

Build and inspect actual distributions before a release:

```bash
python -m build --no-isolation
python tests/check_release_artifacts.py dist
```

[Artifact documentation](docs/artifacts.md) describes stored contracts and recovery.

CI checks the core suite, the sdist support files, wheel resources, and a console
entrypoint installed from the wheel into a fresh environment. Full GPU training,
complete dataset reproduction, and real multi-node execution require separate
runs on the intended deployment.

## Licensing and release metadata

Original OTBench contributions are submitted under the
[Apache License 2.0](LICENSE.txt), unless a separate written
agreement states otherwise. Submit only code you have permission to contribute.
Keep third-party license texts and copyright notices, record source revisions and
local modifications, and mark modified upstream files prominently. The root license
does not replace third-party terms. Read [the licensing guide](docs/licensing.md)
before redistributing a fork, product, service, or resulting model.

Do not commit locally obtained CARD or ReMasker source: their upstream snapshots
provided no redistribution license. Keep their exclusion notices and check that
both the sdist and wheel contain no such implementation files. Apply the same rule
to any new unlicensed component. Refer users to the original project and its terms.

Keep the visible attribution **Built with StableAI LimiX**, ship the exact root
license and all applicable notices, and preserve notices when modifying files.
The release checker inspects the actual archives, not only the checkout. Its
presence checks do not replace reviewing the source and terms of new dependencies.

Contributor/citation metadata and the public repository URL still require real
maintainer information. The citation file identifies this software without
asserting an unpublished paper, DOI, repository URL, or release date.

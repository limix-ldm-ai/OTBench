# great implementation provenance

`model/tabular_great.py` and plugin integration follows Synthcity's Apache-2.0 implementation.
Reference source: <https://github.com/vanderschaarlab/synthcity/tree/v0.2.11/src/synthcity>. The unmodified
Apache License 2.0 from tag `v0.2.11` is included in `LICENSE.synthcity`.
The original import commit is unrecorded. The reference tag identifies verified
source lineage, not an exact-copy claim for every local file.

The files are modified for the OTBench namespace and benchmark data, device,
sampling, and checkpoint integration. Additional shared support notices are in
`../common/synth/VENDOR.md`. The root project license does not replace the
upstream license on these portions.

The language-model generator is provided by the separately installed `be-great` package (MIT): <https://github.com/tabularis-ai/be_great>. That package and its pretrained language-model weights are not vendored here and retain their respective terms.

Source and license references were checked on 2026-10-25.

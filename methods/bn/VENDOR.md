# bn implementation provenance

`plugin.py` follows Synthcity's Apache-2.0 implementation.
Reference source: <https://github.com/vanderschaarlab/synthcity/tree/v0.2.11/src/synthcity>. The unmodified
Apache License 2.0 from tag `v0.2.11` is included in `LICENSE.synthcity`.
The original import commit is unrecorded. The reference tag identifies verified
source lineage, not an exact-copy claim for every local file.

The files are modified for the OTBench namespace and benchmark data, device,
sampling, and checkpoint integration. Additional shared support notices are in
`../common/synth/VENDOR.md`. The root project license does not replace the
upstream license on these portions.

The Bayesian-network backend is the separately installed `pgmpy` package (MIT); its implementation is not vendored here. The `bn_original_units` method shares this plugin through its adapter.

Source and license references were checked on 2026-10-25.

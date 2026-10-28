# nflow implementation provenance

`model/flows.py` and the tabular/plugin integration follows Synthcity's Apache-2.0 implementation.
Reference source: <https://github.com/vanderschaarlab/synthcity/tree/v0.2.11/src/synthcity>. The unmodified
Apache License 2.0 from tag `v0.2.11` is included in `LICENSE.synthcity`.
The original import commit is unrecorded. The reference tag identifies verified
source lineage, not an exact-copy claim for every local file.

The files are modified for the OTBench namespace and benchmark data, device,
sampling, and checkpoint integration. Additional shared support notices are in
`../common/synth/VENDOR.md`. The root project license does not replace the
upstream license on these portions.

The separately installed `nflows` and `fflows` packages retain their own licenses. The wrapper imports their interfaces; their full implementations are not vendored here.

Source and license references were checked on 2026-10-25.

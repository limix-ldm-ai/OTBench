# tvae implementation provenance

`plugin.py`, `model/vae.py`, and their tabular integration follows Synthcity's Apache-2.0 implementation.
Reference source: <https://github.com/vanderschaarlab/synthcity/tree/v0.2.11/src/synthcity>. The unmodified
Apache License 2.0 from tag `v0.2.11` is included in `LICENSE.synthcity`.
The original import commit is unrecorded. The reference tag identifies verified
source lineage, not an exact-copy claim for every local file.

The files are modified for the OTBench namespace and benchmark data, device,
sampling, and checkpoint integration. Additional shared support notices are in
`../common/synth/VENDOR.md`. The root project license does not replace the
upstream license on these portions.

TVAE algorithm/project: <https://github.com/sdv-dev/CTGAN>. The bundled model is the Synthcity implementation, rather than a vendored copy of the current `sdv-dev/CTGAN` package. The latter currently uses BSL-1.1; its terms apply independently if that external software is obtained. No CTGAN package or BSL license text was removed by this documentation change.

Source and license references were checked on 2026-10-25.

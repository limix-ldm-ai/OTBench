# Synthcity-derived support code

Upstream: <https://github.com/vanderschaarlab/synthcity>.
Reference source: tag `v0.2.11`; the retained `version.py` declares `0.2.11`.
License: Apache License 2.0, reproduced without changes in `LICENSE` from
<https://github.com/vanderschaarlab/synthcity/blob/v0.2.11/LICENSE>.

The support code in this directory is an adapted subset of Synthcity. Related
Synthcity-style plugin/model code is located beside the benchmark methods,
including the CTGAN, TVAE, BN, GOGGLE, GReaT, normalizing-flow, and ARF
implementations described in the corresponding method `VENDOR.md` files.
The original import did not record a source commit. The version tag records a
verified source lineage, not a claim that every local file equals that tag.

Local modifications include the OTBench namespace, relocated method backends,
Pydantic compatibility updates, and benchmark-specific data, sampling, and
checkpoint integration. Existing upstream notices and licenses continue to
apply to the upstream portions; the project license does not replace them.

`plugins/core/models/RGCNConv.py` is a modified PyTorch Geometric relational
convolution implementation retained through the Synthcity source lineage.
PyTorch Geometric's MIT notice is preserved in `LICENSE.PyG` (reference license:
<https://github.com/pyg-team/pytorch_geometric/blob/2.4.0/LICENSE>). That release is
license-reference provenance; the exact earlier PyG source commit is unrecorded.

Licenses were checked against the official upstream locations on 2026-10-25.

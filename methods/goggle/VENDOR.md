# GOGGLE source and licenses

Algorithm/project: <https://github.com/tennisonliu/GOGGLE>.
The official project's MIT license is preserved in `LICENSE.GOGGLE`:
<https://github.com/tennisonliu/GOGGLE/blob/main/LICENSE>.

The bundled `model/goggle.py`, `model/tabular_goggle.py`, and plugin route are
adapted from the Synthcity implementation, whose reference source is
<https://github.com/vanderschaarlab/synthcity/tree/v0.2.11/src/synthcity>.
Synthcity's Apache License 2.0 is preserved in `LICENSE.synthcity`; shared support
code and the PyG notice are described in `../common/synth/VENDOR.md`.
The original import commit was not recorded; `v0.2.11` is verified lineage,
not an exact-copy claim for all files.

Local changes include OTBench imports, Pydantic compatibility, the DGL
GraphBolt compatibility gate, explicit device handling, and benchmark plugin
integration. These files are modified from upstream. The root project license
does not replace the upstream licenses.

The MIT project license and Apache source license were checked at their
official locations on 2026-10-25. The MIT file was obtained from upstream `main`;
its original vendoring revision is unrecorded.

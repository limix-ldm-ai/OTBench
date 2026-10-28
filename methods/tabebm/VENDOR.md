# TabEBM source and license

Upstream: <https://github.com/andreimargeloiu/TabEBM>.
Reference implementation:
<https://github.com/andreimargeloiu/TabEBM/blob/main/src/tabebm/TabEBM.py>.
License: Apache License 2.0; the unmodified upstream license is included as
`LICENSE`, from <https://github.com/andreimargeloiu/TabEBM/blob/main/LICENSE>.

`model/tabebm_model.py` is a modified copy of the official TabEBM implementation.
Local changes include explicit offline TabPFN checkpoint resolution, device and
memory controls, and benchmark generation integration. The plugin and adapter
connect that model to OTBench's shared Synthcity-derived support code; see
`../common/synth/VENDOR.md` for that support code's separate notices.

The original source commit was not recorded. The official reference source
and license were checked on 2026-10-25; its setup metadata reported version
`2025.08.19`. This reference does not assert exact equivalence to that version.
TabPFN code and pretrained weights remain subject to their own upstream terms;
no weights are included here. The root project license does not replace the
TabEBM license.

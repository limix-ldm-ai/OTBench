# NRGBoost integration and license

Upstream: <https://github.com/ajoo/nrgboost>.
Backend reference: `nrgboost` version `0.0.3`, installed separately.
The official MIT license is preserved as `LICENSE.NRGBoost`, retrieved from
<https://github.com/ajoo/nrgboost/blob/v0.0.3/LICENSE> on 2026-10-25.

This directory contains the benchmark adapter, plugin, and tabular wrapper;
`model/tabular_nrgboost.py` imports `Dataset` and `NRGBooster` from the separately
installed `nrgboost` package. The upstream NRGBoost backend is not vendored here.
Local wrapper integration includes train-fitted encoding, CPU/OpenMP selection,
thread control, and serialization into benchmark checkpoints. Shared
Synthcity-derived encoding/plugin support retains its Apache-2.0 notice in
`../common/synth/VENDOR.md`.

The license copy records the external backend's terms and does not relicense
that dependency under the root project license.

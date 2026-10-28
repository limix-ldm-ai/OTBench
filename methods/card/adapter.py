# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""OTBench placeholder for CARD; no upstream implementation is distributed."""
from __future__ import annotations

from otbench.methods.common.base import TabularMethod, Task
from otbench.methods.common.upstream import UpstreamUnavailable

UPSTREAM_URL = "https://github.com/XzwHan/CARD"


def _unavailable() -> UpstreamUnavailable:
    return UpstreamUnavailable(
        "CARD source is excluded from the OTBench distribution because the reviewed "
        "upstream source supplied no redistribution license. Obtain the code directly "
        f"from {UPSTREAM_URL} and verify its terms. See methods/card/model/README.md. "
        "This registry placeholder does not load or execute external code."
    )


class CARDMethod(TabularMethod):
    name = "card"
    tasks = frozenset({Task.PREDICTION})
    trainable = True

    def train(self, ctx) -> None:
        raise _unavailable()

    def predict(self, ctx):
        raise _unavailable()

    def is_trained(self, ctx) -> bool:
        return False

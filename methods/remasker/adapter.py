# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""OTBench placeholder for ReMasker; no upstream implementation is distributed."""
from __future__ import annotations

from otbench.methods.common.base import TabularMethod, Task
from otbench.methods.common.upstream import UpstreamUnavailable

UPSTREAM_URL = "https://github.com/tydusky/remasker"


def _unavailable() -> UpstreamUnavailable:
    return UpstreamUnavailable(
        "ReMasker source is excluded from the OTBench distribution because the reviewed "
        "upstream source supplied no redistribution license. Obtain the code directly "
        f"from {UPSTREAM_URL} and verify its terms. See methods/remasker/model/README.md. "
        "This registry placeholder does not load or execute external code."
    )


class ReMaskerMethod(TabularMethod):
    name = "remasker"
    tasks = frozenset({Task.IMPUTATION})
    trainable = True

    def train(self, ctx) -> None:
        raise _unavailable()

    def impute(self, ctx):
        raise _unavailable()

    def impute_draws(self, ctx):
        raise _unavailable()

    def is_trained(self, ctx) -> bool:
        return False

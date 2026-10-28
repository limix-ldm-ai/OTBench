# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""BN with train-fitted numeric bin edges and midpoint decoding in original units.

The ``bn`` and ``bn_original_units`` identities have distinct output spaces and
checkpoint groups. Their evaluation results must not be pooled.
"""
from otbench.methods.bn.adapter import BNMethod


class BNOriginalUnitsMethod(BNMethod):
    name = "bn_original_units"
    restore_numeric_units = True

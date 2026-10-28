# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Forest-VP strict-joint adapter.

Unlike :mod:`otbench.methods.forest_vp`, this family member never supplies
``label_y``.  The target therefore remains in the modeled matrix and prediction
is exactly the same operation as imputation with the whole target column masked.
It has its own registry name, config and checkpoint group because its fitted
artifact is not interchangeable with the per-class label-conditioned family.
"""
from __future__ import annotations

from otbench.methods.common.base import Task
from otbench.methods.forest_vp.adapter import ForestVPMethod


class ForestVPJointMethod(ForestVPMethod):
    name = "forest_vp_joint"
    tasks = frozenset({Task.GENERATION, Task.IMPUTATION, Task.PREDICTION})
    condition_on_categorical_target = False


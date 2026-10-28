# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""OTBench — unified tabular benchmark library.

Layout (one-way dependency, bottom to top):

    tabdata      : dataset store + loader + schema contract
    methods      : all methods behind one interface (methods.common) + per-method
                   implementations and TOML parameter definitions
    checkpoints  : training artifacts, checkpoints/<method>/<dataset>/train_seed<seed>/
                   (foundation models: checkpoints/<method>/pretrained/)
    generation / imputation / prediction : task runners, eval, outputs
    run.py       : unified entry — python -m otbench.run
"""
import os

# TabPFN requires SciPy array API support before SciPy is imported.
# Preserve an explicit environment setting.
os.environ.setdefault("SCIPY_ARRAY_API", "1")

__version__ = "0.2.0"

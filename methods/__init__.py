"""Methods layer: one common interface (methods.common) + per-method packages.

Each method lives in ``methods/<name>/`` with:
    adapter.py    — the TabularMethod implementation
    config.toml   — model/train/sample/impute parameter definition
    model/        — vendored original implementation (when applicable)
"""

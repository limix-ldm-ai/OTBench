# Modified for OTBench on 2026-10-25: package naming and release integration; see NOTICE.md.
"""Setuptools mapping for the OTBench source-root layout."""
from pathlib import Path
from setuptools import setup

root = Path(__file__).parent
source_roots = {"generation", "imputation", "prediction", "metrics", "methods", "tabdata", "scripts"}
packages = ["otbench"]
excluded = {"__pycache__", "store", "raw", "cache", "outputs", "checkpoints", "build", "dist"}
external_source_roots = {("methods", "card", "model"), ("methods", "remasker", "model")}


def external_source(path: Path) -> bool:
    return any(path.parts[:len(prefix)] == prefix for prefix in external_source_roots)


for path in sorted(root.rglob("*")):
    relative = path.relative_to(root)
    if path.is_dir() and relative.parts[0] in source_roots and not external_source(relative):
        if not any(part in excluded for part in relative.parts) and all(part.isidentifier() for part in relative.parts):
            packages.append("otbench." + ".".join(relative.parts))
license_files = {"LICENSE.txt", "NOTICE.md"}
license_files.update(
    path.relative_to(root).as_posix()
    for pattern in ("LICENSE*", "NOTICE*", "VENDOR.md")
    for path in (root / "methods").rglob(pattern)
    if path.is_file() and not external_source(path.relative_to(root))
)
setup(
    packages=packages,
    package_dir={"otbench": "."},
    package_data={
        "": ["*.toml", "*.json", "*.yaml", "*.yml", "*.sh", "*.md", "constraints.txt", "LICENSE*", "NOTICE*"],
        "otbench": ["docs/*.md", "docs/*.json"],
        "otbench.methods.card": ["model/README.md"],
        "otbench.methods.remasker": ["model/README.md"],
    },
    license_files=sorted(license_files),
)

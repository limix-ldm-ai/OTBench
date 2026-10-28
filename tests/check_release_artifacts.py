"""Check built distribution contents without installing optional backends."""
from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path


def check_artifacts(directory: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    wheels = list(directory.glob("*.whl"))
    sdists = list(directory.glob("*.tar.gz"))
    assert len(wheels) == len(sdists) == 1, "build into a clean dist/ directory"
    with zipfile.ZipFile(wheels[0]) as archive:
        names = set(archive.namelist())
        expected = ["run.py", "tabdata/manifest.yaml", "tabdata/manifest_smoke.yaml"]
        expected += [str(path.relative_to(root)) for path in (root / "methods").glob("*/config.toml")]
        expected += [str(path.relative_to(root)) for path in (root / "scripts").glob("*.sh")]
        assert all("otbench/" + path in names for path in expected), "wheel is missing runtime resources"
        entrypoint = next(path for path in names if path.endswith(".dist-info/entry_points.txt"))
        assert "otbench = otbench.run:main" in archive.read(entrypoint).decode()
    with tarfile.open(sdists[0]) as archive:
        names = {path.split("/", 1)[1].removeprefix("./") for path in archive.getnames() if "/" in path}
        expected = ["pytest.ini", "README.md", "NOTICE.md", "CITATION.cff"]
        expected += [str(path.relative_to(root)) for path in (root / "requirements").glob("*.txt")]
        expected += [str(path.relative_to(root)) for path in (root / "tests").glob("*.py")]
        assert all(path in names for path in expected), "sdist is missing install/test support files"
    print("Wheel runtime resources and sdist install/test support files are complete.")


if __name__ == "__main__":
    check_artifacts(Path(sys.argv[1] if len(sys.argv) > 1 else "dist"))

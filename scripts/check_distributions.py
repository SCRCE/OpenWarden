"""Validate OpenWarden wheel and source-distribution contents."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


SDIST_FILES = {
    ".env.example",
    ".gitignore",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "PKG-INFO",
    "README.md",
    "SECURITY.md",
    "pyproject.toml",
}
SDIST_DIRECTORIES = {"assets", "docs", "examples", "openwarden", "scripts", "tests"}
FORBIDDEN_PARTS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "brag-output",
    "dist",
}


def check_wheel(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()

    dist_info = {name.split("/", 1)[0] for name in names if ".dist-info/" in name}
    if len(dist_info) != 1:
        raise ValueError(f"{path}: expected one dist-info directory, found {sorted(dist_info)}")

    allowed_prefixes = ("openwarden/", f"{dist_info.pop()}/")
    unexpected = [name for name in names if not name.startswith(allowed_prefixes)]
    if unexpected:
        raise ValueError(f"{path}: unexpected wheel entries: {unexpected}")
    if "openwarden/py.typed" not in names:
        raise ValueError(f"{path}: missing openwarden/py.typed")


def check_sdist(path: Path) -> None:
    with tarfile.open(path, "r:gz") as archive:
        names = [member.name for member in archive.getmembers() if member.isfile()]

    roots = {PurePosixPath(name).parts[0] for name in names}
    if len(roots) != 1:
        raise ValueError(f"{path}: expected one archive root, found {sorted(roots)}")

    unexpected: list[str] = []
    for name in names:
        parts = PurePosixPath(name).parts[1:]
        if not parts:
            continue
        if any(part in FORBIDDEN_PARTS or part.startswith("brag-output") for part in parts):
            unexpected.append(name)
            continue
        if parts[0] not in SDIST_FILES | SDIST_DIRECTORIES:
            unexpected.append(name)

    if unexpected:
        raise ValueError(f"{path}: unexpected sdist entries: {unexpected}")
    required = {"openwarden", "tests", "README.md", "LICENSE", "pyproject.toml"}
    present = {PurePosixPath(name).parts[1] for name in names}
    missing = required - present
    if missing:
        raise ValueError(f"{path}: missing required sdist entries: {sorted(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archives", nargs="+", type=Path)
    args = parser.parse_args()

    for archive in args.archives:
        if archive.suffix == ".whl":
            check_wheel(archive)
        elif archive.name.endswith(".tar.gz"):
            check_sdist(archive)
        else:
            raise ValueError(f"Unsupported distribution: {archive}")
        print(f"validated {archive}")


if __name__ == "__main__":
    main()

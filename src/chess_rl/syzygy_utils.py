from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def _count_table_files(path: Path) -> int:
    return sum(1 for _ in path.glob("*.rtb*"))


def discover_syzygy_paths(
    explicit_paths: Iterable[str] | None,
    auto_discover: bool = True,
    cwd: Path | None = None,
) -> tuple[list[str], int]:
    search_roots: list[Path] = []
    seen: set[str] = set()

    def add_path(raw: str) -> None:
        candidate = Path(raw).expanduser()
        resolved = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if resolved in seen:
            return
        seen.add(resolved)
        search_roots.append(candidate)

    for path in explicit_paths or []:
        add_path(path)

    env_value = os.environ.get("SYZYGY_PATH", "")
    for path in env_value.split(os.pathsep):
        if path.strip():
            add_path(path.strip())

    if auto_discover:
        base = cwd or Path.cwd()
        defaults = [
            base / "syzygy",
            base / "tablebases",
            base / "data" / "syzygy",
            base / "data" / "tablebases",
            base / "artifacts" / "syzygy",
            Path.home() / "syzygy",
            Path.home() / "tablebases",
            Path.home() / ".local" / "share" / "syzygy",
        ]
        for path in defaults:
            add_path(str(path))

    valid: list[str] = []
    table_files = 0
    for path in search_roots:
        if not path.is_dir():
            continue
        count = _count_table_files(path)
        if count <= 0:
            continue
        valid.append(str(path))
        table_files += count

    return valid, table_files


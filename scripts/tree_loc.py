#!/usr/bin/env python3
"""Print the project's source tree with a line count per file.

Default roots are the project-owned code directories; pass others as arguments:

    python scripts/tree_loc.py
    python scripts/tree_loc.py planners/oracle utils
    python scripts/tree_loc.py --assets my_scenes      # include binary assets
    python scripts/tree_loc.py --git my_scenes         # add `git status --short` marks
    python scripts/tree_loc.py --plain my_scenes       # no counts, tree only

`lsd --tree` shows the same shape without counts; this script exists because the
line counts are what we compare between refactors.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_ROOTS = ("my_scenes", "planners", "robots")
SKIP_DIRS = {"__pycache__", ".git", ".venv", ".venv-sim", ".mypy_cache", ".pytest_cache",
             ".ruff_cache", ".ipynb_checkpoints", "node_modules"}
ASSET_SUFFIXES = {".png", ".jpg", ".jpeg", ".obj", ".mtl", ".stl", ".glb", ".gltf",
                  ".npy", ".npz", ".pt", ".pth", ".h5", ".mp4", ".gif", ".zip"}


def git_marks(root: Path) -> dict[str, str]:
    """`git status --short` letters per path, keyed by the path git prints."""
    try:
        out = subprocess.run(["git", "status", "--short", "--", str(root)],
                             capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}
    marks = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        status, path = line[:2].strip(), line[3:]
        if " -> " in path:
            path = path.split(" -> ")[-1]
        marks[path.strip().strip('"')] = status or "?"
    return marks


def count_lines(path: Path) -> int:
    try:
        with path.open("rb") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return 0


def collect(directory: Path, *, assets: bool) -> tuple[dict, int, int]:
    """Nested {name: node} for the surviving entries, plus (files, lines) below."""
    children: dict[str, dict] = {}
    files = lines = 0
    for path in sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        if path.name in SKIP_DIRS or path.name.endswith(".pyc"):
            continue
        if path.is_dir():
            sub, sub_files, sub_lines = collect(path, assets=assets)
            if not sub:  # nothing left inside: do not print an empty directory
                continue
            children[path.name] = {"dir": True, "children": sub}
            files += sub_files
            lines += sub_lines
        else:
            if not assets and path.suffix.lower() in ASSET_SUFFIXES:
                continue
            count = count_lines(path)
            children[path.name] = {"dir": False, "lines": count}
            files += 1
            lines += count
    return children, files, lines


def render(children: dict, *, prefix: str, loc: bool, marks: dict[str, str],
           parent: Path) -> None:
    items = list(children.items())
    for index, (name, node) in enumerate(items):
        last = index == len(items) - 1
        path = parent / name
        mark = marks.get(str(path))
        label = f"{name}{'/' if node['dir'] else ''}"
        if mark:
            label += f"  [{mark}]"
        if loc and not node["dir"]:
            label += f"  {node['lines']}"
        print(f"{prefix}{'└── ' if last else '├── '}{label}")
        if node["dir"]:
            render(node["children"], prefix=prefix + ("    " if last else "│   "),
                   loc=loc, marks=marks, parent=path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="*", default=list(DEFAULT_ROOTS))
    parser.add_argument("--assets", action="store_true", help="include binary assets")
    parser.add_argument("--plain", action="store_true", help="no per-file line counts")
    parser.add_argument("--git", action="store_true", help="mark changed paths")
    args = parser.parse_args(argv)

    total_files = total_lines = 0
    for name in args.roots:
        root = Path(name)
        if not root.exists():
            print(f"{name}: no such path", file=sys.stderr)
            continue
        if root.is_file():
            print(f"{name}  {count_lines(root)}")
            total_files += 1
            total_lines += count_lines(root)
            continue
        marks = git_marks(root) if args.git else {}
        children, files, lines = collect(root, assets=args.assets)
        print(f"{name}/")
        render(children, prefix="", loc=not args.plain, marks=marks, parent=root)
        print(f"  {files} files, {lines} lines\n")
        total_files += files
        total_lines += lines
    if len(args.roots) > 1:
        print(f"total: {total_files} files, {total_lines} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

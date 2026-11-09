"""Synchronize generated full metadata with the sole authored package version.

Run after editing pynbodyext/__init__.py, then regenerate uv.lock with ``uv lock``.
``--check`` is read-only and fails if any generated version field has drifted.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re
import sys
import tomllib

EXTRAS = ("image", "store", "dev", "bench")


def synchronize(root: Path, *, check: bool) -> int:
    tree = ast.parse((root / "pynbodyext/__init__.py").read_text(encoding="utf-8"))
    versions = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "__version__" for t in n.targets)]
    if len(versions) != 1 or not isinstance(versions[0], str) or not re.fullmatch(r"[A-Za-z0-9.!+_-]+", versions[0]):
        raise ValueError("expected one literal __version__ string")
    version = versions[0]
    member = root / "packages/pynbodyext/pyproject.toml"
    original = member.read_text(encoding="utf-8")
    data = tomllib.loads(original)
    project = data["project"]
    if (project["name"] != "pynbodyext" or set(project["optional-dependencies"]) != set(EXTRAS)
            or len(project["dependencies"]) != 2 or project["dependencies"][1] != "sqlalchemy>=2.0"
            or not project["dependencies"][0].startswith("pynbodyext-core==")
            or any(len(project["optional-dependencies"][e]) != 1 or
                   not project["optional-dependencies"][e][0].startswith(f"pynbodyext-core[{e}]==") for e in EXTRAS)):
        raise ValueError("unexpected full metadata shape; refusing to rewrite")
    updated, count = re.subn(r'(?m)^version = "[^"\n]+"$', f'version = "{version}"', original)
    if count != 1:
        raise ValueError("expected one generated version field")
    for suffix in ("", *(f"[{e}]" for e in EXTRAS)):
        updated, count = re.subn(r'"pynbodyext-core' + re.escape(suffix) + r'==[^"\n]+"',
                                 f'"pynbodyext-core{suffix}=={version}"', updated)
        if count != 1:
            raise ValueError("expected one exact generated core requirement per extra")
    if updated != original:
        if check:
            print("full distribution versions out of sync; run scripts/sync_distribution_versions.py then uv lock", file=sys.stderr)
            return 1
        member.write_text(updated, encoding="utf-8")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        raise SystemExit(synchronize(args.root, check=args.check))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"{exc}\n")

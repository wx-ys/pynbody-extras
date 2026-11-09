"""Inspect ADR-019 artifacts or smoke an installed distribution outside the checkout.

Examples:
  python scripts/check_installation_artifacts.py pair CORE.whl FULL.whl CORE.tar.gz --source SOURCE
  /tmp/venv/bin/python /absolute/path/scripts/check_installation_artifacts.py probe core native
"""
from __future__ import annotations

import argparse
import csv
import email
import hashlib
from importlib import machinery
import io
import json
from pathlib import Path
import sys
import tarfile
import zipfile


def full_wheel(path: Path) -> dict:
    from packaging.requirements import Requirement

    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        assert not any(n.startswith("pynbodyext/") for n in names), "full wheel owns import files"
        message = email.message_from_bytes(wheel.read(next(n for n in names if n.endswith("/METADATA"))))
        version = message["Version"]
        assert message["Name"] == "pynbodyext" and message["Requires-Python"] == ">=3.11"
        requirements = [Requirement(r) for r in message.get_all("Requires-Dist", [])]
        assert {str(r) for r in requirements if r.marker is None} == {f"pynbodyext-core=={version}", "sqlalchemy>=2.0"}
        assert all(r.url is None for r in requirements)
        for extra in ("image", "store", "dev", "bench"):
            forwarded = [r for r in requirements if r.marker and r.marker.evaluate({"extra": extra})]
            assert len(forwarded) == 1 and forwarded[0].name == "pynbodyext-core"
            assert forwarded[0].extras == {extra} and str(forwarded[0].specifier) == f"=={version}"
        tags = wheel.read(next(n for n in names if n.endswith("/WHEEL"))).decode()
        assert "Tag: py3-none-any" in tags
        return {"version": version, "files": names, "requirements": [str(r) for r in requirements]}


def pair(core: Path, full: Path, sdist: Path, source: Path) -> dict:
    from packaging.requirements import Requirement

    result = full_wheel(full)
    with zipfile.ZipFile(core) as wheel:
        names = wheel.namelist()
        assert not set(names) & set(result["files"]), "overlapping wheel paths"
        record = next(n for n in names if n.endswith("/RECORD"))
        recorded = {r[0] for r in csv.reader(io.StringIO(wheel.read(record).decode()))}
        assert set(names) == recorded, "wheel RECORD does not cover its file inventory"
        imports = {n for n in names if n.startswith("pynbodyext/")}
        source_files = {str(p.relative_to(source)) for p in (source / "pynbodyext").rglob("*.py")}
        assert source_files <= imports, f"missing source modules: {sorted(source_files - imports)}"
        native = [n for n in imports if any(n.endswith(s) for s in machinery.EXTENSION_SUFFIXES)]
        assert native, "supported native artifact has no compiled extension"
        css = "pynbodyext/core/calculate/display.css"
        assert b".pynbodyext-calc-card" in wheel.read(css)
        message = email.message_from_bytes(wheel.read(next(n for n in names if n.endswith("/METADATA"))))
        assert message["Name"] == "pynbodyext-core" and message["Version"] == result["version"]
        assert message["Requires-Python"] == ">=3.11"
        deps = [Requirement(r) for r in message.get_all("Requires-Dist", [])]
        assert not any(r.name.lower() == "sqlalchemy" and r.marker is None for r in deps)
        assert any(r.name.lower() == "sqlalchemy" and r.marker.evaluate({"extra": "store"}) for r in deps)
    with tarfile.open(sdist) as archive:
        members = {str(Path(*Path(n).parts[1:])) for n in archive.getnames()}
        required = {str(p.relative_to(source)) for p in (source / "cpp").rglob("*") if p.suffix in {".cpp", ".h", ".hpp"}}
        assert required <= members, f"missing sdist inputs: {sorted(required - members)}"
        assert css in members
        assert not any(n.endswith((".so", ".pyd")) or "/.venv/" in n or "/testdata/" in n for n in members)
    return {"version": result["version"], "import_files": len(imports), "full_import_files": 0,
            "overlap": [], "native": native, "sdist_native_inputs": len(required),
            "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (core, full, sdist)}}


def probe(mode: str, native_mode: str) -> dict:
    import importlib.metadata
    import importlib.resources
    import importlib.util
    import warnings

    import numpy as np
    import pynbody
    import pynbodyext
    from pynbody.array import SimArray
    from pynbodyext.properties import ParamSum

    package = Path(pynbodyext.__file__).resolve()
    assert not package.is_relative_to(Path(__file__).resolve().parents[1]), package
    assert package.is_relative_to(Path(sys.prefix).resolve()), package
    assert importlib.metadata.version("pynbodyext-core") == pynbodyext.__version__
    assert importlib.resources.files("pynbodyext.core.calculate").joinpath("display.css").is_file()
    if mode == "core":
        assert importlib.util.find_spec("sqlalchemy") is None
        assert not any(n == "pynbodyext.orm" or n.startswith("pynbodyext.orm.") for n in sys.modules)
    else:
        assert importlib.metadata.version("pynbodyext") == pynbodyext.__version__
        import sqlalchemy
        assert sqlalchemy.__version__
    sim = pynbody.new(dm=2)
    sim["mass"] = SimArray([6., 0.], "Msol")
    sim["pos"] = SimArray(np.zeros((2, 3)), "kpc")
    sim["smooth"] = SimArray([0., 0.], "kpc")
    assert ParamSum("mass").run(sim).value == 6.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import pynbodyext.gravity
    from pynbodyext.core.calculate import Bin1D
    from pynbodyext.util import deps
    bins = Bin1D("rxy", vmin="1 kpc", vmax="3 kpc", nbins=2)(sim)
    if native_mode == "native":
        import pynbodyext._native
        assert deps.GRAVITY_NATIVE_AVAILABLE
        assert len(bins["vcirc"]) == 2
    else:
        assert importlib.util.find_spec("pynbodyext._native") is None
        assert not deps.GRAVITY_NATIVE_AVAILABLE
        assert "is not built" in deps.native_unavailable_reason()
        assert any(issubclass(w.category, deps.NativeExtensionWarning) for w in caught)
        try:
            bins["vcirc"]
        except ImportError as exc:
            assert exc.name == "pynbodyext._native"
        else:
            raise AssertionError("vcirc must explain missing native")
    return {"package": str(package), "mode": mode, "native": deps.GRAVITY_NATIVE_AVAILABLE, "mass": 6}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    full = sub.add_parser("full-wheel")
    full.add_argument("wheel", type=Path)
    both = sub.add_parser("pair")
    both.add_argument("core", type=Path)
    both.add_argument("full", type=Path)
    both.add_argument("sdist", type=Path)
    both.add_argument("--source", type=Path, required=True)
    installed = sub.add_parser("probe")
    installed.add_argument("mode", choices=("core", "full"))
    installed.add_argument("native", choices=("native", "missing"))
    args = parser.parse_args()
    if args.command == "full-wheel":
        result = full_wheel(args.wheel)
    elif args.command == "pair":
        result = pair(args.core, args.full, args.sdist, args.source)
    else:
        result = probe(args.mode, args.native)
    print(json.dumps(result, indent=2))

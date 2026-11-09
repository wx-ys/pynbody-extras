"""ADR-019: one code owner, exact full/core coupling, standalone source artifacts."""
from __future__ import annotations

import ast
import email
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile

from packaging.requirements import Requirement
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXTRAS = {"image", "store", "dev", "bench"}


def metadata(path):
    return tomllib.loads(path.read_text())


def source_version(root=ROOT):
    tree = ast.parse((root / "pynbodyext/__init__.py").read_text())
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "__version__" for t in n.targets))


def test_core_owns_the_existing_import_package_without_mandatory_orm():
    data = metadata(ROOT / "pyproject.toml")
    assert data["project"]["name"] == "pynbodyext-core"
    assert data["project"]["requires-python"] == ">=3.11"
    deps = {Requirement(d).name.lower(): Requirement(d) for d in data["project"]["dependencies"]}
    assert "sqlalchemy" not in deps
    assert str(deps["pynbody"].specifier) == ">=2.8.0"
    assert data["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "pynbodyext.__version__"}
    assert set(data["project"]["optional-dependencies"]) == EXTRAS
    assert data["project"]["optional-dependencies"]["store"] == ["sqlalchemy>=2.0"]


def test_full_is_metadata_only_and_forwards_exact_version_extras():
    path = ROOT / "packages/pynbodyext/pyproject.toml"
    assert path.is_file(), "full member must exist"
    data = metadata(path)
    version = source_version()
    assert data["project"]["name"] == "pynbodyext"
    assert data["project"]["version"] == version
    assert data["project"]["requires-python"] == ">=3.11"
    assert set(data["project"]["dependencies"]) == {f"pynbodyext-core=={version}", "sqlalchemy>=2.0"}
    assert data["tool"]["setuptools"]["packages"] == []
    assert data["tool"]["uv"]["sources"] == {"pynbodyext-core": {"workspace": True}}
    assert data["project"]["optional-dependencies"] == {
        name: [f"pynbodyext-core[{name}]=={version}"] for name in EXTRAS}
    assert data["build-system"]["build-backend"] == "setuptools.build_meta"


def test_workspace_and_lock_cover_both_local_distributions():
    root = metadata(ROOT / "pyproject.toml")
    assert root.get("tool", {}).get("uv", {}).get("workspace", {}).get("members") == ["packages/pynbodyext"]
    packages = {p["name"]: p for p in metadata(ROOT / "uv.lock")["package"]}
    for name, path in [("pynbodyext-core", "."), ("pynbodyext", "packages/pynbodyext")]:
        assert name in packages
        if name == "pynbodyext":
            assert packages[name]["version"] == source_version()
        else:
            # uv leaves version unset for a dynamic editable member.
            assert "version" not in packages[name]
        assert packages[name]["source"] == {"editable": path}


def test_version_synchronizer_detects_drift_and_updates_all_requirements(tmp_path):
    script = ROOT / "scripts/sync_distribution_versions.py"
    assert script.is_file(), "version synchronization tool must exist"
    (tmp_path / "pynbodyext").mkdir()
    (tmp_path / "pynbodyext/__init__.py").write_text('__version__ = "9.8.7"\nraise RuntimeError("must not import")\n')
    member = tmp_path / "packages/pynbodyext/pyproject.toml"
    member.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "packages/pynbodyext/pyproject.toml", member)
    args = [sys.executable, str(script), "--root", str(tmp_path)]
    before = member.read_bytes()
    result = subprocess.run([*args, "--check"], capture_output=True, text=True)
    assert result.returncode == 1 and "out of sync" in result.stderr
    assert member.read_bytes() == before
    subprocess.run(args, check=True)
    data = metadata(member)["project"]
    assert data["version"] == "9.8.7"
    assert data["dependencies"] == ["pynbodyext-core==9.8.7", "sqlalchemy>=2.0"]
    assert data["optional-dependencies"] == {n: [f"pynbodyext-core[{n}]==9.8.7"] for n in EXTRAS}
    after = member.read_bytes()
    subprocess.run(args, check=True)
    assert member.read_bytes() == after
    subprocess.run([*args, "--check"], check=True)


@pytest.fixture(scope="module")
def source_artifacts(tmp_path_factory):
    """PEP 517 builds; copy inputs to avoid touching checkout egg-info/build state."""
    stage = tmp_path_factory.mktemp("distribution")
    source = stage / "source"
    source.mkdir()
    for name in ["pyproject.toml", "setup.py", "README.md", "LICENSE", "MANIFEST.in"]:
        path = ROOT / name
        if path.exists():
            shutil.copyfile(path, source / name)
    for name in ["pynbodyext", "cpp", "packages"]:
        if (ROOT / name).exists():
            shutil.copytree(ROOT / name, source / name,
                            ignore=shutil.ignore_patterns("__pycache__", "*.so", "*.pyd", "*.egg-info", "build", "dist"))
    env = {"PATH": os.environ["PATH"], "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", str(stage / "uv-cache"))}
    if "UV_OFFLINE" in os.environ:
        env["UV_OFFLINE"] = os.environ["UV_OFFLINE"]
    output = stage / "artifacts"
    subprocess.run(["uv", "build", "--sdist", "--out-dir", str(output),
                    "--default-index", "https://pypi.org/simple", str(source)], env=env, check=True,
                   capture_output=True, text=True)
    return source, output, env


@pytest.mark.slow
def test_actual_core_sdist_contains_native_headers_and_css(source_artifacts):
    source, output, _ = source_artifacts
    with tarfile.open(next(output.glob("*.tar.gz"))) as archive:
        files = {str(Path(*Path(n).parts[1:])) for n in archive.getnames()}
        required = {str(p.relative_to(source)) for p in (source / "cpp").rglob("*")
                    if p.suffix in {".h", ".hpp", ".cpp"}}
        assert required <= files, f"sdist missing native inputs: {sorted(required - files)}"
        assert {"cpp/gravity/octree.hpp", "cpp/image/scatter.hpp"} <= files
        css = "pynbodyext/core/calculate/display.css"
        assert {css, "setup.py", "pyproject.toml", "README.md", "LICENSE"} <= files
        name = next(n for n in archive.getnames() if n.endswith("/" + css))
        assert b".pynbodyext-calc-card" in archive.extractfile(name).read()


@pytest.mark.slow
def test_full_sdist_rebuild_has_exact_portable_metadata_and_no_code(source_artifacts, tmp_path):
    source, _, env = source_artifacts
    member = source / "packages/pynbodyext"
    assert member.is_dir(), "full member must exist"
    subprocess.run(["uv", "build", "--sdist", "--out-dir", str(tmp_path / "sdist"),
                    "--default-index", "https://pypi.org/simple", str(member)], env=env, check=True,
                   capture_output=True, text=True)
    sdist = next((tmp_path / "sdist").glob("*.tar.gz"))
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(tmp_path / "wheel"),
                    "--default-index", "https://pypi.org/simple", str(sdist)], env=env, check=True,
                   capture_output=True, text=True)
    with zipfile.ZipFile(next((tmp_path / "wheel").glob("*.whl"))) as wheel:
        assert not any(n.startswith("pynbodyext/") for n in wheel.namelist())
        msg = email.message_from_bytes(wheel.read(next(n for n in wheel.namelist() if n.endswith("/METADATA"))))
        assert msg["Name"] == "pynbodyext" and msg["Version"] == source_version()
        requirements = [Requirement(s) for s in msg.get_all("Requires-Dist")]
        base = {str(r) for r in requirements if r.marker is None}
        assert base == {f"pynbodyext-core=={source_version()}", "sqlalchemy>=2.0"}
        for name in EXTRAS:
            forwarded = [r for r in requirements if r.marker and r.marker.evaluate({"extra": name})]
            assert len(forwarded) == 1
            assert forwarded[0].name == "pynbodyext-core"
            assert forwarded[0].extras == {name}
            assert str(forwarded[0].specifier) == f"=={source_version()}"
        assert all(r.url is None for r in requirements)


def test_artifact_checker_rejects_code_ownership_in_full_wheel(tmp_path):
    script = ROOT / "scripts/check_installation_artifacts.py"
    assert script.is_file(), "artifact checker must exist"
    bad = tmp_path / "bad.whl"
    with zipfile.ZipFile(bad, "w") as wheel:
        wheel.writestr("pynbodyext/__init__.py", "")
        wheel.writestr("pynbodyext-0.0.1.dist-info/METADATA", "Name: pynbodyext\nVersion: 0.0.1\n")
    result = subprocess.run([sys.executable, str(script), "full-wheel", str(bad)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "owns import files" in result.stderr


@pytest.mark.parametrize('path', ['pyproject.toml', 'packages/pynbodyext/pyproject.toml'])
def test_build_backend_floor_supports_spdx_license_metadata(path):
    from packaging.version import Version
    data = metadata(ROOT / path)
    requirement = next(Requirement(r) for r in data['build-system']['requires'] if Requirement(r).name == 'setuptools')
    assert Version('77.0.0') in requirement.specifier
    assert Version('76.0.0') not in requirement.specifier

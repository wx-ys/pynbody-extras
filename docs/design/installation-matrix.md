# Installation matrix — TASK-0002 / ADR-019

2026-11-09. Final local source after FEAT-0005–FEAT-0012, version **0.0.1**.
No upload, package-name reservation, commit or version bump occurred.

## Ownership and version contract

Root distribution `pynbodyext-core` owns every `pynbodyext/**` Python/native/CSS
file, including optional ORM code. Full `pynbodyext` is a metadata-only workspace
member, requiring **pynbodyext-core==0.0.1** and **SQLAlchemy>=2.0**. It forwards
image/store/dev/bench extras with the same exact core version. Import paths remain
`pynbodyext.*`; image/powerbin stays optional even for full.

The source literal `pynbodyext/__init__.py:__version__` is authoritative. Run
`python scripts/sync_distribution_versions.py`, then `uv lock`, after changing it.
The script's `--check` is nonmutating. Full metadata has a generated static version
so its sdist builds independently. uv 0.9.17 leaves the dynamic editable core's
version unset in uv.lock; full's static lock version, actual built core METADATA
and its exact full dependency are verified separately. Lock regeneration preserved
the existing Tsinghua registry and existing dependency versions (59 packages).

Both build definitions require setuptools>=77.0.0, the minimum needed for their
SPDX `license = "MIT"` metadata (and root `license-files`). No runtime dependency
was added. See [setuptools metadata support](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html).

## Evidence location and environment

Final artifacts/logs: `/tmp/task0002-revalidated/`. Build driver:
`/tmp/task0002-build-revalidated.py`; full command records `build-commands.json`,
`install-commands.json` and `migration-commands.json` in that directory. These are temporary evidence,
not distributable project assets. The repository checker is
`scripts/check_installation_artifacts.py`; slow packaging tests rebuild actual
sdists without touching checkout build state.

The final artifacts were rebuilt after correcting invocation provenance to record
the code-owning core version independently of the optional full metapackage.
Earlier `/tmp/task0002-final` artifacts are superseded by this directory.

The source was copied directly from the current working tree, including all
untracked feature modules. `.so`, `.pyd`, pycache, build/dist and egg-info were
excluded before building. `source-sha256.json` fingerprints every staged source
input; this dirty snapshot is not represented by a commit ID alone. Final source
sdists/wheels include the corrected backend floor; earlier prototypes and the
unintended unpinned Python 3.14 build are excluded from final certification.

| Component | Tested value |
|---|---|
| OS / architecture | Linux 7.0.0-43-generic, x86_64 |
| Python | CPython 3.11.9; explicitly passed to every final build |
| uv / pip | uv 0.9.17; disposable pip 26.3 for source/migration |
| Isolated PEP 517 build constraints | setuptools 84.2.1, wheel 0.48.0, pybind11 3.1.0 |
| Compiler | g++ (Ubuntu 15.2.0-16ubuntu1) 15.2.0 |
| Runtime constraints | pynbody 2.8.0, numpy 2.3.5, scipy 1.16.3, matplotlib 3.10.7, SQLAlchemy 2.0.52 when requested |
| Full dependency inventories | `light-packages.log`, `full-packages.log`, `serial-packages.log`, `no-compiler-packages.log`, `extras-packages.log`, `sdist-packages.log` |

`/tmp/pynbody-packaging-spike/current-constraints.txt` fixes existing development
versions for resolver experiments; it does not preinstall dependencies. Install
requirements named both local wheel paths directly, avoiding similarly named
registry packages. Runtime commands ran from `/tmp` with PYTHONPATH removed,
without system-site packages. Artifact probes require the imported package to be
inside the target venv; source-install probes intentionally identify staged source.
Temporary optional pandas used for unrelated suite coverage was not in artifact
PYTHONPATH or base runtime dependencies.

A writable temporary uv cache was seeded from the read-only host cache. Resolver
installs explicitly used `https://pypi.org/simple`; the existing lock uses
`https://pypi.tuna.tsinghua.edu.cn/simple`. Source identity matters to offline cache
lookup: initial cache misses are recorded, followed by approved isolated network
retrieval for missing optional extras/build/source dependencies. No dependency
was installed globally or into the working project environment.

## Final artifact inventory

All listed builds exited 0 through isolated PEP 517. Native presence was separately
asserted; optional build exit 0 alone is insufficient.

| Artifact, relative to evidence directory | Bytes | SHA256 |
|---|---:|---|
| `wheels/pynbodyext_core-0.0.1-cp311-cp311-linux_x86_64.whl` (from sdist) | 593660 | `3265250f70c7fe63d463f79f88d724aacfbc354e3b349441a3e345d7ba0509c1` |
| `wheels/pynbodyext-0.0.1-py3-none-any.whl` (from sdist) | 1239 | `f4c5b77cb33d5a1b9d847fbd8bf72ce78242e80cc0bd75ba94c469fff8bb23b1` |
| `sdists/pynbodyext_core-0.0.1.tar.gz` | 377976 | `d8616b2dfdf0b5a706100828b2c94236461124af5d6ccd3ff18b60f95de07fbb` |
| `sdists/pynbodyext-0.0.1.tar.gz` | 1374 | `1b13854e7ce1bf43367d8ea59d51a75eb396f10a3cbd61b2a3acf8c444c915cf` |
| `serial/pynbodyext_core-0.0.1-cp311-cp311-linux_x86_64.whl` | 591318 | `281f63ee1d3560a8bef06c5d54eeb29661696e839d2877afd1846f0a0bd9fcac` |
| `no-compiler/pynbodyext_core-0.0.1-cp311-cp311-linux_x86_64.whl` | 409123 | `2d9a464efd0b5e6925b4f65a52db017a4dc4e1f9bfc0c2a07d8720c22cae197d` |

Direct-source core/full wheels also built successfully; their hashes are in
`artifact-hashes.json`. ZIP/RECORD inventory: core has **126 import/resource files**,
full has **zero**, complete wheel-path intersection is empty. Core's sdist contains
all **20 C++ source/header inputs**, including gravity/octree.hpp and image/scatter.hpp.
CSS exists in sdist, wheel and installed resources with the card selector content.
Both wheels' actual METADATA preserve Python>=3.11 and exact full/core coupling;
all extras have proper markers and no local/workspace URLs leak into Requires-Dist.
The no-compiler wheel retains a platform tag from the optional extension build
configuration; no universal-wheel portability claim is made for it.

## Reproduce builds and installs

These commands use the recorded directories; replace them with new empty staging
and environment paths when reproducing. No prebuilt extension may enter SOURCE.

```bash
# Root and member source sdists (repeat for packages/pynbodyext).
UV_CACHE_DIR=/tmp/task0002-cache uv build --python /home/yxi/Simulation/pynbody-extras/.venv/bin/python --offline --default-index https://pypi.org/simple --build-constraints /tmp/task0002-revalidated/build-constraints.txt --sdist --out-dir /tmp/task0002-revalidated/sdists /tmp/task0002-revalidated/source
# Each source archive is rebuilt independently, outside its workspace.
UV_CACHE_DIR=/tmp/task0002-cache uv build --python /home/yxi/Simulation/pynbody-extras/.venv/bin/python --offline --default-index https://pypi.org/simple --build-constraints /tmp/task0002-revalidated/build-constraints.txt --wheel --out-dir /tmp/task0002-revalidated/wheels /tmp/task0002-revalidated/sdists/pynbodyext_core-0.0.1.tar.gz
# Full archive equivalent uses pynbodyext-0.0.1.tar.gz.
# Normal resolver install of both local artifacts; light supplies only core.
UV_CACHE_DIR=/tmp/task0002-cache uv pip install --python /tmp/task0002-revalidated/full-env/bin/python --offline --default-index https://pypi.org/simple --constraint /tmp/pynbody-packaging-spike/current-constraints.txt /tmp/task0002-revalidated/wheels/pynbodyext_core-0.0.1-cp311-cp311-linux_x86_64.whl /tmp/task0002-revalidated/wheels/pynbodyext-0.0.1-py3-none-any.whl
```

The full member's sdist was rebuilt without root README/version-script/source
access. Direct installation of **both sdists** also resolved/build-isolated
successfully with `uv pip install --build-constraints ... CORE.tar.gz FULL.tar.gz`.
Build logs include `core-sdist-wheel.log`, `full-sdist-wheel.log`,
`core-source-wheel.log`, `full-source-wheel.log`, `core-serial-wheel.log`,
`core-no-compiler-wheel.log`; install log `sdist-install.log` proves the frontend
source-install path. Forced serial used `PYNBODYEXT_OPENMP=0`. No-compiler used
nonexistent `CC=/tmp/task0002-missing-cc`, `CXX=/tmp/task0002-missing-cxx` with fresh
sdist extraction. No `--no-deps` installation was used.

## Observed matrix

| Linux x86_64 / CPython 3.11.9 experiment | Result | Evidence |
|---|---|---|
| Fresh core wheel resolver install | Passed | `light-install.log`, `light-smoke.log`, SQLAlchemy spec absent, mass 6 |
| Fresh full wheel resolver install + ORM SQLite | Passed | `full-install.log`, `full-allfeatures.log`, `final-allfeatures.log` |
| Core[store] and full invocation package versions | Passed | `core-store-provenance.log` records core 0.0.1/full null; `full-provenance.log` records both 0.0.1 |
| Core/full wheel from each sdist | Passed | PEP 517 logs above; actual installed native/CSS |
| Direct core/full sdist install | Passed | `sdist-install.log`, `sdist-smoke.log`, `sdist-check.log` |
| Full all four extras | Passed after cache retrieval | `extras-install.log`, `extras-smoke.log`, `extras-check.log`; earlier cache retrieval log in `/tmp/task0002-final` |
| Full removal preserves all core files | Passed | `owned-before.json` equals `owned-after.json`; calculator/native/CSS still work |
| Remove core while full remains | Expected dependency failure, then repaired | `broken-full-check.log` exit 1; `repaired-full-check.log` exit 0 |
| Remove both distributions | Passed | `removed-imports.log`: import spec absent |
| Default OpenMP native | Passed | build flag log, `native-ldd.log` (libgomp), native vcirc call |
| Forced serial native | Passed | serial build log, `serial-ldd.log`, `serial-smoke.log` |
| No compiler / optional native unavailable | Passed | no binary in wheel, visible NativeExtensionWarning/reason, vcirc ImportError, mass 6, CSS available |
| Missing/unloadable native loader | Passed (simulated) | unchanged `tests/test_native_availability.py`; not a foreign-platform test |
| Pip lightweight editable source | Passed | `/tmp/task0002-pip-core-source.log`, `/tmp/task0002-pip-core-smoke.log`; no SQLAlchemy |
| Pip full editable source pair with image extra | Passed | `pip-full-source.log`, `pip-full-smoke.log` |
| uv development and lightweight source sync | Passed offline with filled cache | `uv-dev-source.log`, `uv-light-source.log`, `uv-dev-smoke.log`, `uv-light-smoke.log`; working `.venv` untouched |
| Old single-package 0.0.1 -> new pair, uninstall old first | Passed | `migration-*.log`; new full removal leaves core operational |
| Same-version direct pip --upgrade | **Failed ownership contract despite exit 0** | `migration-direct-upgrade.log`, `migration-direct-state.log`; old full persists |

The unmodified `/tmp/pynbody-full-workflow.py` ran from `/tmp` in final full-wheel
environment, without PYTHONPATH: **11 Runs / 4 cells**, ordered duplicate selection
`[2,0,2]`, Catalogue host relation, Job replay/cache hit, StudyReport/SampleTable
round trips, immutable saved values **12 -> 24**, current-cache deletion retaining
history, and external-file overwrite reported `changed` without touching bytes.
The same joint workflow passed for the full editable source recipes. Fresh
core[store] and full resolver installs separately created real SQLite Runs and
verified the persisted core/full package-version fields; core-only metadata is no
longer mistaken for an unknown code version.

## Supported source installation and migration

From a checkout in a fresh environment:

```bash
# Full, including optional image support; both local projects in one operation.
python -m pip install -e . -e './packages/pynbodyext[image]'
# Lightweight in a separate fresh environment:
python -m pip install -e .
# Development (also used by make setup):
uv sync --locked --all-packages --all-extras
# Lightweight uv source environment:
uv sync --locked --package pynbodyext-core --no-default-groups
```

The exact source commands tested against the staged checkout were (the first pip
lightweight installation preceded the provenance-only correction; the final uv
lightweight source probe used the corrected source):

```bash
/tmp/task0002-pip-env/bin/python -m pip install --index-url https://pypi.org/simple --constraint /tmp/pynbody-packaging-spike/current-constraints.txt --editable /tmp/task0002-final/source
/tmp/task0002-pip-env/bin/python -m pip install --cache-dir /tmp/task0002-pip-cache --index-url https://pypi.org/simple --constraint /tmp/pynbody-packaging-spike/current-constraints.txt --editable /tmp/task0002-revalidated/pip-source --editable '/tmp/task0002-revalidated/pip-source/packages/pynbodyext[image]'
UV_PROJECT_ENVIRONMENT=/tmp/task0002-revalidated/uv-dev-env UV_CACHE_DIR=/tmp/task0002-cache uv sync --offline --locked --all-packages --all-extras --python /home/yxi/Simulation/pynbody-extras/.venv/bin/python --project /tmp/task0002-revalidated/source
UV_PROJECT_ENVIRONMENT=/tmp/task0002-revalidated/uv-light-env UV_CACHE_DIR=/tmp/task0002-cache uv sync --offline --locked --package pynbodyext-core --no-default-groups --python /home/yxi/Simulation/pynbody-extras/.venv/bin/python --project /tmp/task0002-revalidated/source
```

The earlier uv development retry downloaded missing lock-pinned mirror artifacts;
both final-source syncs then succeeded offline. These are distinct temporary
environments. The smoke probes show their installed source path, and core has no
SQLAlchemy. Installed wheel RECORDs were also checked after core-then-full
installation: 135 core entries, 8 full entries, intersection empty
(`/tmp/task0002-final/lifecycle-3.log`, unchanged ownership mechanism); final core
file hashes also match before/after full removal (`owned-before.json`,
`owned-after.json`).

The root now names core; `.[image,store]` adds dependencies to core but does not
install full's metadata. Pip does not interpret uv workspace source overrides;
installing only the member can seek core remotely. Existing environments retain
SQLAlchemy unless it is separately removed; a fresh light environment proves the
contract. Uninstalling full never promises automatic dependency removal.

The migration experiment built the old baseline source copy in
`/tmp/pynbody-packaging-spike/original` into an actual 0.0.1 legacy wheel. It was
installed with its store extra so all dependency requirements were satisfied
before the direct-upgrade experiment. With pip 26.3, direct local wheel upgrade
exited 0 yet printed that full was already installed with the same version;
only core was installed. The retained old full RECORD still owned **119** import
paths, overlapping core. That is not a safe migration result. An earlier attempt
without the store dependency also failed resolution under `--no-index`; it is
an environment prerequisite failure, not evidence of ownership behavior.

Use the verified order in a disposable test environment before applying it to an
older installation (replace paths with the actual matching artifacts):

```bash
python -m pip uninstall -y pynbodyext
python -m pip install /absolute/path/pynbodyext_core-0.0.1-cp311-cp311-linux_x86_64.whl /absolute/path/pynbodyext-0.0.1-py3-none-any.whl
python -m pip check
```

If an attempted mixed installation already lost core files, explicitly reinstall
the core artifact after old full has been removed, then install full. A force
reinstall may be needed when versions match. No version bump, remote release
upgrade, registry availability or other installer-version ordering was assumed.

## Platform and Python limits

| Platform | Python 3.11.9 full/core + native available/unavailable | Other Python >=3.11 combinations |
|---|---|---|
| Linux x86_64 (this host) | Passed as detailed above | **Not run** |
| Linux other architectures | **Not run** | **Not run** |
| macOS Intel / Apple Silicon | **Not run** | **Not run** |
| Windows | **Not run** | **Not run** |

Metadata Python>=3.11/pynbody>=2.8.0 remains the compatibility baseline. This table
certifies only the observed host/interpreter; resolution or a compiled uninstalled
Python 3.14 artifact does not establish runtime support. macOS OpenMP README
instructions remain prerequisites, not fresh platform certification.

## Regression evidence and final checks

Initial six contract tests failed for the old layout/missing member/synchronizer
and actual missing headers (`/tmp/task0002-red.log`). The artifact checker rejects
an adversarial code-owning full wheel (`/tmp/task0002-checker-red.log`). Two
backend-floor regressions failed before raising setuptools to >=77.0.0
(`/tmp/task0002-backend-red.log`). Final packaging tests: **9 passed**
(`/tmp/task0002-green.log`). Final standard `make check`: **1679 passed, 2 skipped, 7 deselected**, including
lint/typecheck/metadata-lock/docs checks (`/tmp/task0002-check-final.log`). Final
standard `make test-all`: **1686 passed, 2 skipped**
(`/tmp/task0002-test-all-final.log`). Final optional-pandas-enabled full suite:
**1688 passed, zero skipped**, independently run by the root agent with
`UV_CACHE_DIR=/tmp/task0002-cache UV_OFFLINE=1 PYTHONPATH=/tmp/pynbody-test-libs make test-all`
(`/tmp/pynbody-final-pandas-all.log`). Native C++ tests: **ALL PASS**
(`/tmp/task0002-cpp.log`). The TASK handoff records the final review state; docs validation runs again after
updating that state/index.

中文：仅实际验证 Linux x86_64 / Python 3.11.9，完整/轻量、sdist 重建、可选 native、
卸载归属与保守旧包迁移通过；其他平台和 Python 组合未运行。旧同版本 pip upgrade
虽然退出 0，仍保留旧 full 的文件归属，因此不计为通过。未发布、未改用户工作环境。

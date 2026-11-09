# TASK-0002 installation matrix implementation plan

**Goal:** Implement ADR-019 and verify the two distributions using newly built
artifacts after FEAT-0012 and its predecessors are complete.

**Architecture:** The root project becomes `pynbodyext-core`, owning every
`pynbodyext/**` source, native and CSS path. The metadata-only workspace member
`packages/pynbodyext` provides full installation through an exact core dependency
and SQLAlchemy >=2.0. Imports remain `pynbodyext.*`.

**Tools:** Python >=3.11, setuptools/PEP 517, uv workspace/lock, pytest, C++17.

**Execution boundary:** This document is preparation only. Do not change production
packaging, tests, source, work-item status or CONTEXT until the sequential feature
work through FEAT-0012 is complete. ADR-019 already approves the names and ownership;
no new architecture approval is required for that decision. Use inline execution
with checkpoints when this work starts. Preserve the active checkout and existing
user changes, including the pre-existing untracked `.github/`. Do not commit, push,
upload, reserve package names or infer a version bump from this plan.

## Read and preserve

Read AGENTS.md, CONTEXT.md, TASK-0002 metadata/both locales/latest handoff,
ADR-005, ADR-019 and `docs/design/packaging-decision.md` before execution. Preserve
`tests/test_native_availability.py` and the CSS contract in
`tests/test_calculate_render.py`; these describe real import and resource behavior.
There is no existing dedicated setup/build test: artifact tests must add coverage.
Check the final source and acceptance criteria again after FEAT-0012; prototype
artifacts predate that work.

## Files and responsibilities

| File | Planned responsibility |
|---|---|
| `pyproject.toml` | Rename root distribution; declare workspace; preserve core dependencies/extras, dynamic version and package data |
| `packages/pynbodyext/pyproject.toml` (new) | Standalone full metadata/build definition; exact core dependencies and forwarded extras; no import packages |
| `scripts/sync_distribution_versions.py` (new) | Read the literal source version without importing the package; write/check generated full version requirements |
| `MANIFEST.in` (new) | Include required C++ sources/headers in core sdist |
| `uv.lock` | Resolver-generated workspace identities, versions and dependencies |
| `Makefile` | Check metadata synchronization and synchronize both development members |
| `tests/test_installation_contract.py` (new) | Fast version/metadata contracts plus explicit slow actual-artifact regressions |
| `scripts/check_installation_artifacts.py` (new) | Outside-checkout artifact inventory and installation/lifecycle smoke driver; no publication |
| `README.md`, `README.zh-CN.md` | Matching supported source/full/core/development installation commands and migration guidance |
| `docs/design/installation-matrix.md` (new) | Reproducible commands, tool versions, artifact hashes, logs and per-cell results |
| `docs/work/TASK-0002/{en.md,zh-CN.md,meta.json,handoff.md}` | Acceptance evidence and handoff after implementation |

Change `setup.py` only if a failing artifact test shows it is needed beyond the
manifest; preserve optional-native behavior and its OpenMP/serial probing. No
runtime/layer redesign or new runtime dependency is part of this work. Declare
any newly required tooling in pyproject and regenerate uv.lock together.

## 1. Make the packaging contract fail first

- [ ] Add fast contract tests parsing TOML with `tomllib` and versions with AST;
  importing `pynbodyext` to obtain a packaging version is unnecessary. Assert root
  name `pynbodyext-core`, Python >=3.11 and unchanged pynbody >=2.8.0 baseline.
- [ ] Assert full has exactly its base requirements: `pynbodyext-core==<version>`
  and `sqlalchemy>=2.0` (compare normalized requirement names). Core base must not
  require SQLAlchemy; its `store` extra must. For each of image/store/dev/bench,
  full forwards `pynbodyext-core[<extra>]==<version>`; compare all three version
  locations without accepting a range or a direct local URL.
- [ ] Assert full explicitly has `packages = []`, matching Python baseline and
  an independent build definition. Assert the workspace lists this member and
  its core source uses `workspace = true` only for local resolution.
- [ ] Add an actual sdist inventory regression, marked `slow`: build into a
  temporary directory, open the tar archive, and require all current `cpp/**/*.hpp`,
  `cpp/**/*.h` and C++ source files needed by setup.py, plus CSS, setup.py,
  pyproject, README and LICENSE. Compare archive members to the source inventory,
  not merely the text of MANIFEST.in. At least `cpp/gravity/octree.hpp` and
  `cpp/image/scatter.hpp` must be asserted explicitly.
- [ ] Run `.venv/bin/python -m pytest tests/test_installation_contract.py -m 'not slow' -q`
  and the explicit slow sdist inventory test. Preserve red output: baseline has
  the old project name/no member and actual sdist omits C++ headers. A dependency
  cache/network failure is an environment failure, not the expected red result.

Keep build/network work out of `make test` through the existing `slow` marker.
The slow test uses a controlled cached PEP 517 build environment prepared for the
matrix, and errors clearly when prerequisites are absent; do not silently skip a
claimed artifact check. Fast tests must run without installing packaging tools
into the user's environment or performing network access.

## 2. Implement deterministic metadata and lock synchronization

- [ ] Preserve `pynbodyext/__init__.py:__version__` as the single authored version.
  Rename only root distribution metadata; do not rename import paths or duplicate
  code into the member. Add this root workspace declaration:

  ```toml
  [tool.uv.workspace]
  members = ["packages/pynbodyext"]
  ```

- [ ] Give the full member static generated metadata so its sdist builds without
  access to the root source, root README or synchronization script. At the current
  version, the decisive content is:

  ```toml
  [build-system]
  requires = ["setuptools>=61", "wheel"]
  build-backend = "setuptools.build_meta"

  [project]
  name = "pynbodyext"
  version = "0.0.1"
  description = "Full pynbodyext installation with ORM support"
  requires-python = ">=3.11"
  dependencies = ["pynbodyext-core==0.0.1", "sqlalchemy>=2.0"]

  [project.optional-dependencies]
  image = ["pynbodyext-core[image]==0.0.1"]
  store = ["pynbodyext-core[store]==0.0.1"]
  dev = ["pynbodyext-core[dev]==0.0.1"]
  bench = ["pynbodyext-core[bench]==0.0.1"]

  [tool.setuptools]
  packages = []

  [tool.uv.sources]
  pynbodyext-core = { workspace = true }
  ```

  These values use the source version at execution time; do not hardcode a new
  version. Avoid an external README/license-file reference that a member sdist
  cannot contain. Preserve appropriate project authors/URLs/license metadata
  without making full import or copy the source package.
- [ ] Implement `scripts/sync_distribution_versions.py` with default write mode
  and `--check` nonmutating mode. Use stdlib AST to require one literal source
  version; generate/check the member version, base exact requirement and all
  four forwarded-extra exact requirements. Reject an unexpected member shape
  rather than rewriting unrelated metadata. Do not execute package code, edit
  uv.lock text/hashes, or invent a second authored version source.
- [ ] Test the script against a temporary fixture: a changed source version makes
  `--check` fail, write mode updates every generated value, repeated write is
  byte-identical, and subsequent `--check` passes. The real checkout remains
  unchanged by these tests.
- [ ] Add manifest content `recursive-include cpp *.cpp *.hpp *.h`; retain CSS
  package-data configuration. Exclude stale built `.so`/`.pyd`, egg-info and
  build outputs from the clean staging copy, so binaries must be freshly built.
- [ ] Run `.venv/bin/python scripts/sync_distribution_versions.py`, then `uv lock`
  without an indiscriminate dependency upgrade. Preserve the repository index
  policy; the lock currently uses the Tsinghua mirror. Review both local package
  entries: root core and member full have the source version, correct editable
  paths, exact metadata dependencies and extra forwarding. Check `uv lock --check`
  and `.venv/bin/python scripts/sync_distribution_versions.py --check`.
- [ ] Update `make setup` to `uv sync --all-packages --all-extras`; make check must
  run the nonmutating metadata check before verification. Version-update procedure:
  edit the source literal, run sync script, run `uv lock`, run both checks, then
  rebuild both distributions together. No hand-edited lock hashes.

## 3. Build both projects from clean source and from their sdists

- [ ] Stage the final current source in a new `/tmp` directory, excluding `.venv`,
  testdata, caches, existing native binaries, build/dist and egg-info. Record the
  source revision plus dirty-source inventory/hash; do not describe a dirty
  snapshot using a commit hash alone. Build logs and artifacts stay in `/tmp`.
- [ ] Use an isolated PEP 517 frontend, not just direct `build_meta` calls. Installed
  uv supports `uv build --sdist --out-dir <dir> <source>` and
  `uv build --wheel --out-dir <dir> <sdist-path>`. Build each core/full sdist, then
  build each wheel from its archive outside the workspace. Also build source
  wheels for comparison. Record exact expanded commands and exit codes.
- [ ] Prepare build dependencies separately from runtime constraints. Existing
  .venv lacks setuptools; cached setuptools 84.0.0, wheel 0.48.0 and pybind11
  3.1.0 were used by the spike via temporary PYTHONPATH. That backend-only probe
  does not prove isolated PEP 517 resolution. Use a disposable build environment
  and explicit build constraints/index/cache; never install globally. Isolated
  build environment failure must be resolved or recorded failed before claiming
  PEP 517 success.
- [ ] Inspect built METADATA, WHEEL and ZIP RECORD, not only pyproject. Parse
  Requires-Dist with `packaging.requirements.Requirement`; validate full's exact
  base/extra dependencies and markers, core's optional SQLAlchemy, matching
  versions and Requires-Python. Reject repository file URLs/workspace paths in
  published dependency metadata. Require full wheel `py3-none-any` and no import
  packages. Require core alone owns every `pynbodyext/**` entry, including orm,
  native and CSS; compare complete ZIP and installed RECORD path intersections.
- [ ] Inspect both sdists for standalone rebuild inputs and no accidental `.venv`,
  testdata, secrets or prebuilt native. Require all native headers in core tar;
  require CSS with expected `.pynbodyext-calc-card` content in core tar and wheel.
- [ ] For the supported Linux native build, require a newly produced extension
  suffix from `importlib.machinery.EXTENSION_SUFFIXES`, successful installed
  import, and a small native operation through the public API. Build exit 0 is
  insufficient because `optional=True` can swallow compiler failure. Assert the
  same for the wheel rebuilt from core sdist.

Use the new slow artifact tests/driver for these contracts. For a deliberate
negative check, remove a header in a throwaway source copy and verify the sdist
inventory test fails; never weaken existing native tests to accommodate it.

## 4. Verify resolver installs and source documentation

- [ ] Create distinct temporary venvs for full, core, source examples, lifecycle
  and migration. Use no system-site packages; unset PYTHONPATH for all runtime
  probes, use a working directory outside the repository, and assert module
  `__file__` is inside that venv. Do not reuse or uninstall the working `.venv`.
- [ ] Install built wheels with normal dependency resolution and both artifacts
  available in a local `--find-links` directory. Do not use `--no-deps` or manually
  preinstall SQLAlchemy in the full test. Use runtime constraints from the known
  development versions, recording the complete resulting package inventory.
  Constraints alone do not install packages, so the light environment must still
  have no SQLAlchemy. Record index/cache configuration alongside results.
- [ ] Full probe: find both distributions at the source version, import
  SQLAlchemy/ORM, calculate mass 6 for a two-particle snapshot, run an ORM SQLite
  fixture through AnalysisRepository, and fetch the stored result. Adapt
  `/tmp/pynbody-artifact-smoke.py` to final public APIs without relaxing behavior.
- [ ] Core probe: find only the core distribution, `find_spec('sqlalchemy') is None`,
  no ORM import triggered by calculator use, calculate the same mass, and read
  packaged CSS. Both supported-native probes import the actual extension.
- [ ] Run `uv pip check --python <venv-python>` after each valid install. Validate
  all forwarded extras in built metadata and resolve/install full `[image]` and
  `[store]` locally; inspect dev/bench forwarding as well. Full means ORM is
  included; optional powerbin/image dependencies remain optional.
- [ ] Test these supported clone/source commands in fresh temporary environments,
  then put matching English/Chinese instructions in the READMEs:

  ```bash
  # Full from a checkout: supply both local projects in one resolver operation.
  python -m pip install -e . -e './packages/pynbodyext[image]'
  # Lightweight from a fresh environment: root now names the core distribution.
  python -m pip install -e .
  # Development workspace (all projects and extras).
  uv sync --locked --all-packages --all-extras
  # Optional uv lightweight source environment, with no store extra requested.
  uv sync --locked --package pynbodyext-core --no-default-groups
  ```

  Run these in disposable copies/venvs, not the working .venv. Validate pip
  commands with pip in a disposable environment in addition to uv artifact
  installs; uv workspace source overrides are not interpreted by pip. Installing
  only `./packages/pynbodyext` with pip can seek core on an index, so that is not
  the documented checkout recipe. Root `.[image,store]` installs core plus extras,
  but does not install the full distribution metadata. Switching to core in an
  existing environment does not automatically uninstall SQLAlchemy.
- [ ] Describe wheel/local-sdist installs using both built artifacts and local
  find-links. Do not imply candidate names are available on PyPI today. Future
  registry examples may name `pynbodyext`/`pynbodyext-core` but must say publication
  has not occurred. Preserve Python/pynbody/native prerequisites and both README
  development command equivalences.

## 5. Prove ownership and test the old single-distribution transition

- [ ] Start with core, install full, hash representative/all core RECORD-owned
  files, uninstall only full, and compare those files and core metadata. Fresh
  subprocesses must still calculate, read CSS and load native; dependency check
  succeeds. Do not promise automatic removal of SQLAlchemy or other dependencies.
- [ ] Reinstall full; intentionally uninstall core only. Require dependency check
  failure naming the missing exact core requirement. Reinstall the matching core
  artifact and require full smoke/check recovery. Then remove both projects and
  verify no `pynbodyext/**` or either dist-info remains except installer-generated
  cache handling; do not remove third-party dependencies as part of this check.
- [ ] Obtain/build the old single-distribution artifact from the recorded baseline
  snapshot in `/tmp`, with its version/hash recorded; it owns the import files.
  Never fabricate a version bump or change project version merely to make the
  upgrade test easier. Current source version is 0.0.1, which may equal the old
  version. A bare `pip install --upgrade pynbodyext` can be a no-op at that version
  and is not evidence of migration.
- [ ] In a separate disposable environment test explicit direct-artifact in-place
  replacement of the old wheel by new full/core, including forced replacement
  when versions match. Inspect installed RECORD ownership and every core file
  after replacement and after uninstalling full. Capture pip/uv operation order;
  installing core before uninstalling old full can leave old RECORD able to
  delete core files. Empty new-wheel intersection alone proves nothing here.
- [ ] Always verify a conservative migration recipe: uninstall the old
  `pynbodyext` before introducing core, then install both new local artifacts
  together with dependency resolution and run full smoke/ownership checks. If an
  in-place experiment loses files, document the observed failure and this verified
  recovery path; do not label that transition passed. Where a prior failed attempt
  left core installed but damaged, explicitly reinstall core/full artifacts.
  Repeat relevant lifecycle checks for editable source installs when they have
  different installed RECORD behavior. Version bumps/real registry upgrades and
  release orchestration require a later publication decision.

## 6. Verify optional-native and serial builds independently

- [ ] From a clean core sdist, build a wheel with unusable `CC`/`CXX` paths and
  separate empty build output/cache. Require success through optional-native
  fallback, no packaged native binary, calculator/CSS smoke success, visible
  NativeExtensionWarning, unavailable reason, and the documented ImportError
  for `bins['vcirc']`. Import that fallback wheel in its own venv to prevent
  accidental native reuse. This tests genuine no-compiler packaging.
- [ ] Build another fresh sdist wheel with `PYNBODYEXT_OPENMP=0`; require actual
  native import/use and a serial-build log. This tests serial native capability,
  not a missing compiler. Keep the default OpenMP case distinct and record its
  compiler/runtime versions. Reuse the existing simulated missing/broken loader
  tests for unloadable native behavior; they do not certify macOS or Windows.
- [ ] Do not broaden native flag/platform support based on inspection alone.
  Windows/macOS/other Python runtime results remain not-run unless executed on
  those systems. Any unexpected setup.py compatibility issue is recorded with
  evidence and handled within the accepted optional-native contract.

## 7. Final matrix and handoff

- [ ] Record source fingerprint, artifact SHA256s, exact Python/OS/architecture,
  uv/pip/build backend/compiler versions, dependency inventories, command logs,
  exits and expected values in `docs/design/installation-matrix.md`. Cover fresh
  core/full; wheel/sdist; source examples; extras; lifecycle; old transition;
  default OpenMP/forced serial/no-compiler. Distinguish expected dependency-check
  failure experiments from broken acceptance results.
- [ ] List Linux/macOS/Windows and each claimed supported Python combination with
  passed/failed/not-run cells; Python >=3.11 is a metadata requirement, not proof
  that every future interpreter has been tested. Separate pure-Python and native
  results. Record blockers for not-run cells without inventing certification.
- [ ] Run fast installation contracts, explicit slow artifact regressions,
  `make check`, `make test-all` and `make docs-check`; run `make cpp-test` on the
  tested native Linux path. Tests added for packaging must exercise actual
  outputs, not duplicate implementation strings. Capture final red -> green
  evidence and avoid stale results from prototype artifacts.
- [ ] Review against ADR-019 and the bilingual TASK acceptance criteria. Update
  both locales and handoff, retain review status until merged, and leave a
  concrete changed-file summary plus remaining platform limitations. Request
  review within existing authorization; do not initiate publication or Git work.

## Existing evidence and constraints

`/tmp/pynbody-packaging-spike/` contains prototype core/full wheels and sdists.
Original sdist omitted every C++ header; rebuilding returned exit 0 while missing
`gravity/octree.hpp` prevented producing native. Later prototype full installation
with normal dependency resolution succeeded over the network using explicit
`https://pypi.org/simple` and known development-version constraints. A separate
light installation then succeeded offline using the matching index/cache; both
outside-repository smoke probes calculated 6, read CSS and imported native. Full
also ran the ORM SQLite save/fetch fixture, while light had no SQLAlchemy.

The repository lock/default mirror and the explicit PyPI cache are different
sources: offline cache misses with the former do not invalidate the successful
latter experiment or prove unavailable dependencies. Record source selection; do
not rewrite the lock's registry URLs just to reuse a cache. Temporary optional
pandas at `/tmp/pynbody-test-libs` is test infrastructure, not a production or
light-install requirement; do not pass that PYTHONPATH into artifact probes.

All these are prototype evidence. The final feature source must be rebuilt and
revalidated, especially isolated PEP 517 sdist builds, uninstall/migration ordering,
extra forwarding and no-compiler/serial paths, which the prototype did not prove.

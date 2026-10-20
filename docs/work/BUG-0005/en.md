# BUG-0005: Native availability is decided by find_spec, not by loading pynbodyext._native

## Context

Raised by a review comment on PR #25 (BUG-0003), left open at merge.
`pynbodyext/util/deps.py` sets

```python
GRAVITY_NATIVE_AVAILABLE = module_available("pynbodyext._native")  # find_spec(...) is not None
```

`find_spec` only checks that a module file exists. A `_native*.so` that is
present but cannot be loaded (wrong architecture, a missing `libomp.dylib`,
an ABI mismatch) still gives `True`. Then:

- `pynbodyext/gravity/__init__.py` imports `.base`, whose `from
  pynbodyext._native import ...` raises, so `import pynbodyext.gravity`
  fails instead of degrading with the existing `ImportWarning`;
- `bins["vcirc"]` passes the BUG-0003 check and fails on the gravity import.

`_sph_scatter.native_pair_builder()` already does it right: it tries the
import and falls back on `ImportError`.

## Goal

"Available" means "imports successfully". A broken extension behaves like a
missing one, and the reason is kept for error messages. When the extension is
unusable, importing `pynbodyext.gravity` emits a warning that is visible by
default (also answers BUG-0004's second open question).

## Actual behavior

Not observed yet on a real machine; follows from the code (see Context).

## Expected behavior

- With a present-but-broken `_native`, `GRAVITY_NATIVE_AVAILABLE` is `False`,
  `import pynbodyext.gravity` succeeds with the `ImportWarning`, and
  `bins["vcirc"]` raises the BUG-0003 error, which also carries the original
  import error.
- With a working or missing extension, behaviour is unchanged.

## Reproduction

```python
# Simulate a broken extension: a spec exists but import fails.
import sys, importlib.util, types
# e.g. monkeypatch importlib.util.find_spec to return a spec for
# "pynbodyext._native" and sys.modules["pynbodyext._native"] = None,
# then reload pynbodyext.util.deps and pynbodyext.gravity.
```

## Non-goals

- Fixing the macOS build (BUG-0004).
- Changing which features use the extension.

## Acceptance criteria

- [x] `deps` decides native availability by attempting
      `import pynbodyext._native` (once, cached), not by `find_spec`.
- [x] The import error is kept (e.g. `deps.GRAVITY_NATIVE_ERROR`) and
      appended to the `vcirc` and `pynbodyext.gravity` messages.
      (`deps.native_import_error()`, `deps.native_unavailable_reason()`.)
- [ ] A test simulates a present-but-unloadable extension and checks that
      `import pynbodyext.gravity` succeeds and `bins["vcirc"]` raises the
      clear `ImportError`; it runs on every machine.
- [ ] `tests/test_package_layers_contract.py` still passes (importing
      `_native` from `util.deps` must not break the layer rules).
- [ ] `make check` passes with and without the extension.

## Design notes

- `util/deps.py` is imported early; a real import of `_native` there changes
  import-time cost and layering. Alternative: keep the flag lazy (a cached
  function `gravity_native_available()`), and keep `GRAVITY_NATIVE_AVAILABLE`
  as a compatibility alias.
- BUG-0003's test monkeypatches `deps.GRAVITY_NATIVE_AVAILABLE`; keep that
  working or update the test together with the requirement.
- `_native` also hosts the SPH scatter engine; consider one shared helper so
  gravity and scatter agree on availability.

## Decision

- `util/deps.py`: `native_import_error()` imports `pynbodyext._native` once
  (cached) and returns the `ImportError` or `None`;
  `GRAVITY_NATIVE_AVAILABLE = native_import_error() is None` (still a module
  attribute, so BUG-0003's monkeypatch keeps working). Eager, at `deps`
  import: `deps` is only imported by `gravity`, `plot.image.adaptive` and
  inside `_vcirc`, and loading the extension is cheap.
  `native_unavailable_reason()` says "not built" (`ModuleNotFoundError` for
  `pynbodyext._native`) or "present but failed to load (<error>)", plus how
  to rebuild.
- New `deps.NativeExtensionWarning(UserWarning)`, shown by default.
  `pynbodyext.gravity` emits it at import when the extension is unusable,
  replacing the hidden `ImportWarning`. Filter with
  `warnings.filterwarnings("ignore", category=NativeExtensionWarning)`.
- `bins["vcirc"]`: the `ImportError` message now ends with the reason and is
  chained (`from`) to the original import error.
- `import pynbodyext` itself does not warn: users who never touch gravity or
  `vcirc` should not see it. `_sph_scatter.native_pair_builder()` keeps its own
  silent fallback.

## Test plan

- Red: a test with a fake spec + failing import shows
  `GRAVITY_NATIVE_AVAILABLE is True` and `import pynbodyext.gravity` raising.
- Green: the same test passes after the change; `make check`.

## Outcome

## Open questions

- Is the extra import at `deps` import time acceptable, or should the check
  be lazy?

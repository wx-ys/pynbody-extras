# Context

Single-context repo. This file defines the vocabulary used across the calculator
and result-store subsystems. Decisions live in [`docs/adr/`].

## Glossary

Use these terms exactly. Don't drift to synonyms.

- **Calculator DAG** — the composed execution graph (`CalculatorBase` +
  mixins, `PropertyBase`, `FilterBase`, `TransformBase`, `Pipeline`). Each node
  produces a `ResultNode`; the graph is resolved and executed by `EvalEngine`.
- **`CalculatorSignature`** (`result/signature.py`) — a frozen, structured
  description of a calculator (class + `deps` + parameters), with a content
  hash and a JSON text. *Not* a display name.
- **`sim_signature`** — a pluggable address tuple describing *which* simulation
  object a result belongs to, e.g. `("sim", "snapshot_103", "halo_0")`. Produced
  by `EvalEngine.make_sim_signature(sim)`.
- **`engine.make_sim_signature`** — the stable-identity seam. Default is
  id-based (`("sim", id(sim))`); a persistence-aware caller injects an address
  provider for restart-stable identity.
- **Content-addressed key** — the composite store identity:
  `(sim_signature, calculator_signature.short_hash())`. Two identical
  computations derive the same key; a changed calculator or object derives a
  different one.
- **`ResultStore`** (`store/base.py`) — the persistence seam. `store`/`fetch`/
  `get`/`has`/`delete`/`list_refs` over four abstract primitives
  (`_put`, `_get_by_key`, `_delete`, `_list`).
- **Record** (`store/base.py::ResultRecord`) — one self-describing, codec-encoded
  row (value + named + provenance), serializable via `to_json`/`from_json`.
- **`ValueCodec`** (`store/codec.py`) — portable value serialization (numpy,
  pynbody `SimArray` with units, scalars, nested structures, pickle fallback).
- **tangos** — the Python package this store is modeled on; a relational DB that
  interns name strings. We deliberately diverge (see ADR-0001).
- **Repr style** — how a calculator object renders in a notebook, set by
  `display.set_repr_style`: `rich` (cards, `<details>`, badges), `github`
  (sanitizer-safe `<table>` / `<pre>` / `<h4>`) or `plain` (text only).
  *Not* a data property: every style shows the same fields, only the markup
  differs.
- **`InfoView`** (`display.py`) — mixin for a value object that renders itself.
  The subclass defines `_display_rows()` once; the mixin derives the compact
  `__repr__` and the per-style HTML. Adopted by `ProvenanceInfo`, `PerfSummary`,
  `ValueSummary`, `ErrorInfo`, `PhaseRecord`, `AccessObservation` and the result
  dict views.
- **`TextReport`** (`display.py`) — a `str` subclass returned by the view
  attributes (`Result.execution_tree` / `.performance` / `.cache`,
  `CalculatorBase.dependency_tree`, the `Result.report_*()` builders). It behaves
  like `str` everywhere but also renders as HTML, so one name serves both the
  text and the notebook view.
- **Section attribute** — the rule tying the two display layers together: every
  collapsible section of the rich card is reachable as an attribute whose name is
  the section title in snake_case (``Execution tree`` -> ``.execution_tree``).
  The `github`/`plain` card lists exactly those names, so a reader drills down by
  typing an attribute, never by calling a method. Card sections render from the
  attribute's own rows, so the two cannot drift.
- **View-only attribute** — a public attribute that exists only to render
  something already reachable elsewhere. Deliberately avoided: the detail lives
  on the object itself (via `InfoView`) or on the report method (via
  `TextReport`). See ADR-0002.

## Persistence entry points

The store seam is reachable from the live execution paths:

```python
engine = EvalEngine(sim_identity=lambda sim: ("sim", "snapshot_103", "halo_0"))
engine.run(calc, sim, store=store)                    # direct engine
calc.run(sim, store=store, sim_identity=path_identity)  # calculator-level
calc(sim, store=store, sim_identity=path_identity)    # __call__ alias
```

The result is persisted automatically on a clean run keyed by
`(sim_identity(sim), root CalculatorSignature.short_hash())`. `store` and
`sim_identity` are both optional, so existing callers are unaffected.

Loading back is the mirror image — `store.load(sim_signature=...,
calculator_signature_text=...)` deserializes the stored signature text into a
runnable calculator and returns `(calculator, result)` (or `None` on a miss):

```python
calculator, result = store.load(
    sim_signature=("sim", "snapshot_103", "halo_0"),
    calculator_signature_text=record.calculator_signature_text,
)
rerun = calculator.run(sim)   # the recipe is fully rebuilt from the text
```

## Relevant decisions

- [ADR-0001](docs/adr/0001-result-store-content-addressed-by-signature.md): the
  result store is keyed by content-addressed signatures rather than interned
  names — better than tangos because identity derives from what was *computed*
  (and stays stable across restarts via `engine.make_sim_signature`).
- [ADR-0002](docs/adr/0002-display-protocol-render-existing-objects.md): display
  comes from the existing objects (`InfoView` rows, `TextReport` methods) rather
  than parallel view-only attributes, so `rich` and `github` cannot drift apart.
- [ADR-0004](docs/adr/0004-image-layer-takes-plain-2d-arrays.md): the image
  layer (`plot/image/`) takes plain 2-D arrays rather than calculator nodes, so
  any map producer can feed it; `ImageData` is the only place that knows the
  calculator layer.
- [ADR-0005](docs/adr/0005-image-data-is-the-pipeline-object.md): the image layer
  is class-centric — one `ImageData` composed from capability mixins, free
  functions underneath — and every image drawn anywhere in the package goes
  through it.
- [ADR-0006](docs/adr/0006-image-capabilities-are-composed-views.md): the
  capabilities are *views* reached as properties, found
  through a registry so a new family is a new module rather than a new base class,
  and colour bars are docked to their panel by one helper.
- [ADR-0007](docs/adr/0007-image-orientation-is-type-driven.md): orientation is
  decided by the type (`as_image` transposes a `BinsArray`), so
  `adaptive.bin(bins.s["count"])` means what it looks like, and a raw transposed
  grid is named in the error.
- [ADR-0008](docs/adr/0008-image-noise-models.md): two explicit, seeded noise
  models — Gaussian (`sigma` or `snr`) and Poisson (`exposure`, `background`) —
  rather than hidden noise in the display path.
- [ADR-0009](docs/adr/0009-display-is-a-family-too.md): `display` is a capability
  family like the others, so no operation has a flat spelling and `ImageData`
  carries only what it *is* (supersedes decision 4 of ADR-0006).
- [ADR-0010](docs/adr/0010-two-entry-points-display-and-process.md): an image
  has two entry points — `image.display.*` for showing it and
  `image.process.*` for working on it — with the families nested inside the
  second.
- [ADR-0011](docs/adr/0011-the-colormap-is-saurons.md): the colour map is Cappellari &
  Emsellem's SAURON map, reproduced from its published table bit-for-bit and
  registered under its own name, `"sauron"` (`"sauron_r"`), with the source
  credited in the code.

## Image layer (`pynbodyext/plot/image/`)

The vocabulary for turning a 2-D map into a figure. Code lives in
`pynbodyext/plot/image/`. Every operation is a free function over a plain `numpy`
2-D array, and the same operations are methods of `ImageData`, which is the
object a caller normally holds.

- **`ImageData`** (`plot/image/data.py`) — a 2-D array plus the metadata needed
  to display and measure it. Geometry is per axis: `x_edges`/`y_edges` are the
  bin edges (unevenly spaced allowed — logarithmic, quantile, or explicit) and
  `extent` (`(xmin, xmax, ymin, ymax)`) is shorthand for evenly spaced bins.
  Metadata is per axis too: `x_units`/`y_units` may differ and `x_label`/`y_label`
  name the axes, while `label`/`units` describe the values (and go on the colour
  bar). `ImageData.from_bins(bins2d, query)` reads one query of a 2-D
  `BinNDResult` together with its grid, units and axis names; `.with_data(...)`
  carries the metadata through a processing chain; `.pixel_size` is the
  `(dy, dx)` needed to express a kernel width in physical units, and refuses when
  a direction has bins of unequal width. Display accordingly:
  `.display.imshow()` for evenly spaced bins, `.display.pcolormesh()` for arbitrary
  edges, `.display.draw()` to let it pick.
  `ImageData` is a single class (no capability base classes). Each family is a
  **view** — `SmoothOps`, `PsfOps`, `ComposeOps`, `AdaptiveOps` — reached as a
  property of `image.process`: `image.process.smooth.gaussian(fwhm=2)`,
  `image.process.psf.convolve(fwhm=3)`, `image.process.noise.poisson(...)`,
  `image.process.compose(other)`, `image.process.adaptive.bin(signal)`.
  Display is the other entry point: `image.display.normalize/to_rgba/draw/imshow/
  pcolormesh/contour/add_colorbar`. Every operation has exactly one spelling.
- **`ImageDataView`** (`plot/image/ops.py`) — "an `ImageData` seen from one angle":
  it holds the image and forwards its data and metadata (`data`, `shape`, `ndim`,
  `extent`, edges, centers, uniformity, `pixel_size`, units, labels) plus
  `add_colorbar`. Every view of an image derives from it, so nothing forwards the
  same property twice. **`ImageOps`** extends it with the two things a capability
  family needs: `derive()` (return a new image with the operation recorded) and
  `kernel_scale()` (pixel size, or `None` on uneven bins). `AdaptiveMap` derives
  from it directly — it *is* a view of its painted map, not a capability.
- **`register_ops`** / `ImageData.register_ops("tessellation", TessellationOps)`
  adds a capability family through the registry, so a new processing stage is a new
  module and a registration — no base-class list to edit
  (`ImageData.views()` lists what is registered; an instance's `.ops` is a different
  thing — the chain of operations already applied to *that* image).
- **`ImageOp`** (`plot/image/ops.py`, next to the views and the registry) — one
  recorded step of a chain (`name`, `params`), appended to `ImageData.ops` by every
  processing method and summarised by `repr(image)`.
- Methods that produce an image return a new
  `ImageData` — geometry and units intact — with the call appended to
  **`ops`** (`tuple[ImageOp, ...]`, `ImageOp(name, params)`), which is what
  `repr(image)` summarises.
- **Two entry points: `display` and `process`.** `ProcessOps` (in
  `ops.py`, the module that also holds the views and the registry) exposes the five
  families that change or measure the values; `DisplayOps` exposes the seven
  operations that show them. So `dir(image)` is short and unambiguous —
  `image.process.smooth.gaussian(...)` versus `image.display.imshow(...)` — and
  `ImageData` itself carries only what it *is*: values, geometry, units, labels,
  `ops`, `with_data`, `from_bins`/`from_bins_array`/`as_image`, the registry, and
  those two properties.
- **Colour bars** are docked to their panel, not floated: `add_colorbar(artist |
  image, loc=…)` (also `image.display.add_colorbar(…)`, and `colorbar=True|loc` +
  `colorbar_kwargs` on `display.draw`/`imshow`/`pcolormesh`/`contour`) uses
  `make_axes_locatable` with a shared divider per panel, so `size="5%"`,
  `pad=0.05`, `label_pad` and `tick_label_size` are under the caller's control and
  several bars (e.g. the two sides of `compose.imshow`) coexist.
- **Scales are explicit and shared.** `display.draw`/`imshow`/`pcolormesh`/
  `contour` and `add_colorbar` take `log=True` (or a `norm=`), so a map that spans
  decades — a density, say — is drawn and labelled on one scale: `log=True` builds a
  `LogNorm` over the positive values (or explicit `vmin`/`vmax`), contour levels
  become geometric, the colour bar comes out logarithmic on every side, and asking
  for a scale a drawn artist does not have is an error rather than a mislabelled
  bar. `symmetric=True` (zero-centred) and `log=True` are mutually exclusive.
  Matplotlib's *string* spellings are understood too — `norm="log"` behaves exactly
  like `log=True` (it is resolved to a norm instance by `cmaps.as_norm`, so the
  levels become geometric as well), and `norm="symlog"`, `"logit"`, `"asinh"` and
  `"linear"` are accepted where this matplotlib has them; an unknown name is an
  error rather than a silently linear scale.
- **Every user-facing method documents its own arguments.** The free functions stay
  the detailed reference, but a method may not answer a parameter with "as in
  :func:`the_function`" — it lists its own `Parameters`, `Returns` and an `Examples`
  block, and names the function that implements it. `tests/test_plot_image_docs.py`
  enforces that, rejecting pointer-style parameter entries, because the methods are
  what users read.
  A `MapStyle` may carry a `norm` too (or a `stretch`, not both), and
  `MapStyle.norm_for(data)` — used by `compose.imshow` — returns a norm matching the
  stretch exactly (`FuncNorm` over the data range, per `stretch_functions`), so a
  stretched map's colour bar no longer shows a linear scale.
- **Annotations name real types.** The vocabulary is `plot/image/_types.py` —
  `MapLike` (an image, a binned array or a plain array), `MaskLike`, `KernelWidth`
  (a number, a pynbody unit, or a `(y, x)` pair) — plus `numpy.typing.ArrayLike`,
  the `matplotlib` artist classes and `UnitLike`; a signature never answers with a
  bare `Any` (it stays only inside a container, `dict[str, Any]`, and on forwarded
  `**kwargs`, where it is honest). The bridge is typed on both sides:
  `BinsArray.image` is an `ImageData`, and `ImageData.from_bins_array` takes a
  `BinsArray`. `tests/test_plot_image_typehints.py` keeps both from drifting.
  An `Artist` is anything a drawing method returns — an image, a mesh, a contour
  set, a scatter `PathCollection`, or a bar — so a scatter profile can go straight
  to `add_colorbar` without a type error. The annotations are deferred (PEP 563)
  and the names they use are imported under `TYPE_CHECKING`, which the lint config
  requires (`ruff` `TC`); resolving them therefore takes
  `plot/image/_types.resolve_type_hints`, which imports what they name — matplotlib's
  drawing stack, the calculator — only when asked. That is deliberate: importing
  `pynbodyext.plot.image` stays free of `matplotlib.pyplot` and of the calculator
  (about 350 ms and 270 ms), and the tests pin it.
- **Contours** are a display verb alongside `imshow`: `image.display.contour(ax=ax,
  levels=…, filled=…)` (and `AdaptiveMap.contour`) draws on the bin centres, so it
  is correct for uneven bins and overlays an existing image when passed the same
  axes. The artist is a `ContourSet`, which `add_colorbar` accepts directly.
- **Orientation is type-driven.** A binned array is `(x, y)`; an image is
  `(row=y, column=x)`. `as_image(bins2d["mass.sum"])` — and every function that
  takes a second map (`adaptive.bin(signal)`, `compose(other)`, `adaptive_bin_map`)
  — recognises a binned array by its `bins`/`grid` attributes and transposes it, so
  `bins.s["count"]` means what it looks like. A *raw* array cannot be told apart
  from an image, so when its shape is the transpose of the image's, the error says
  exactly that instead of failing obscurely (silent when the grid is square).
  The rule holds for the **methods** (signals, noise maps, masks, the other side of
  a composite): they accept an image-like input. The free functions underneath take
  arrays you have already oriented, which is the array-level contract.
- **`BinNDResult.imshow`** (`core/calculate/bins/plot.py`) — a one-line bridge:
  it builds `ImageData.from_bins(self, query)` and calls `.display.draw()`. A bin grid is
  `(x, y)` and an image is `(row=y, column=x)`, so `from_bins` transposes; uneven
  bins therefore come out as a `pcolormesh` rather than being forced onto a
  regular pixel grid. No image drawing is implemented in the calculator layer.
  The same bridge is reachable from the array itself: `bins2d["mass.sum"].image`
  (a `BinsArray` property) hands back the `ImageData`, so
  `bins2d["mass.sum"].image.display.draw()` or `.display.contour(ax=ax)` need no
  further import.
- **Adaptive bin map** (`AdaptiveMap`, `plot/image/adaptive.py`) — a partition of
  a map into regions of comparable *capacity* made with PowerBin (centroidal
  power diagrams, the successor of Voronoi binning). The capacity is the
  `signal` map, or `(S/N)²` when a `noise` map is given. `.bin_num` assigns every
  binned pixel to a region, `.bin_value`/`.bin_capacity`/`.bin_count` describe the
  regions, and `.value` paints the region values back at full resolution with
  unbinned pixels left non-finite. *Not* a smoothing operation: it never mixes
  pixels into an average they do not belong to.
  It carries the geometry of the grid it was binned *from* (`x_edges`/`y_edges`,
  per-axis units and labels), and `.to_image_data()` hands that on; `.display.draw()`
  draws even grids with `imshow` and uneven ones with `pcolormesh`.
- **Map mask** (`plot/image/compose.py`) — a pair of complementary soft masks in
  `[0, 1]` split by a line at `line_angle`, with a transition ramp whose width is
  a fraction of the image diagonal (`width`, so the look is resolution
  independent; `width <= 0` gives a hard split). Drives `blend_images`,
  `blend_stack` and `compose_maps`, which stitch two maps drawn with different
  colour maps into one RGBA image whose alpha is the coverage.
- **`sauron_cmap`** (registered as `"sauron"`/`"sauron_r"`, `plot/image/cmaps.py`)
  — **Michele Cappellari & Eric Emsellem's SAURON colormap** (Leiden, 2001): black
  → blue → cyan → **green on zero (0.50)** → yellow → red → light grey (0.9, not
  white). `SAURON_POSITIONS` (11 control points, symmetric about 0.5) and
  `SAURON_RGB` are the published table, and the map is built from it exactly as the
  reference implementation (`plotbin/sauron_colormap.py`, Cappellari 2014-2024,
  https://purl.org/cappellari) does — a test asserts the two look-up tables are
  identical whenever `plotbin` is installed, so `cmap="sauron"` means the same
  colours whoever registered it first. This is the map used for the stellar
  velocity fields of the SAURON and ATLAS³D surveys; cite that lineage, not this
  package, for the colours. `vel_cmap`/`vel_cmap_r` are aliases. `to_rgba` maps a
  2-D array plus a stretch to RGBA.
- **PSF** (`plot/image/psf.py`) — the observational point-spread function:
  `gaussian_psf(fwhm=..., e=..., theta=...)` builds a kernel, `convolve_psf`
  applies it (forward direction, for comparing a model to data), and
  `wiener_deconvolve` / `richardson_lucy` invert it (visualisation only — they
  amplify noise).
- **Noise** (`plot/image/noise.py`, `image.noise.*`) — what the map looks like once
  it is *counted*. `noise.gaussian(sigma=… | snr=…)` adds Gaussian white noise (a
  scalar or per-pixel σ map, or σ = |data| / snr), and
  `noise.poisson(exposure=…, background=…)` samples counting noise at a given
  exposure (bigger = deeper; variance `(data + background) / exposure`). Both are
  seeded (`rng=`), leave non-finite pixels alone, honour a mask, and record what
  they did in `.ops`.

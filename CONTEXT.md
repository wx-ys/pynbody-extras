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
  capabilities are *views* reached as properties (`image.smooth.gaussian`), found
  through a registry so a new family is a new module rather than a new base class,
  and colour bars are docked to their panel by one helper.
- [ADR-0007](docs/adr/0007-image-orientation-is-type-driven.md): orientation is
  decided by the type (`as_image` transposes a `BinsArray`), so
  `adaptive.bin(bins.s["count"])` means what it looks like, and a raw transposed
  grid is named in the error.

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
  `.imshow()` for evenly spaced bins, `.pcolormesh()` for arbitrary edges.
  `ImageData` is a single class (no capability base classes). Each family is a
  **view** — `SmoothOps`, `PsfOps`, `ComposeOps`, `AdaptiveOps` — reached as a
  property: `image.smooth.gaussian(fwhm=2)`, `image.psf.convolve(fwhm=3)`,
  `image.compose(other)`, `image.adaptive.bin(signal, target_nbins=200)`.
  Single-call operations stay methods of `ImageData` itself (`normalize`,
  `to_rgba`, `draw`, `imshow`, `pcolormesh`, `add_colorbar`), so no operation has
  two spellings.
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
  (`ImageData.operations()` lists what is registered).
- Methods that produce an image return a new
  `ImageData` — geometry and units intact — with the call appended to
  **`ops`** (`tuple[ImageOp, ...]`, `ImageOp(name, params)`), which is what
  `repr(image)` summarises.
- **Colour bars** are docked to their panel, not floated: `add_colorbar(artist |
  image, loc=…)` (also `image.add_colorbar(…)`, and `colorbar=True|loc` +
  `colorbar_kwargs` on `draw`/`imshow`/`pcolormesh`/`AdaptiveMap.imshow`) uses
  `make_axes_locatable` with a shared divider per panel, so `size="5%"`,
  `pad=0.05`, `label_pad` and `tick_label_size` are under the caller's control and
  several bars (e.g. the two sides of `compose.imshow`) coexist.
- **Contours** are a display verb alongside `imshow`: `image.contour(ax=ax,
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
  it builds `ImageData.from_bins(self, query)` and calls `.draw()`. A bin grid is
  `(x, y)` and an image is `(row=y, column=x)`, so `from_bins` transposes; uneven
  bins therefore come out as a `pcolormesh` rather than being forced onto a
  regular pixel grid. No image drawing is implemented in the calculator layer.
- **Adaptive bin map** (`AdaptiveMap`, `plot/image/adaptive.py`) — a partition of
  a map into regions of comparable *capacity* made with PowerBin (centroidal
  power diagrams, the successor of Voronoi binning). The capacity is the
  `signal` map, or `(S/N)²` when a `noise` map is given. `.bin_num` assigns every
  binned pixel to a region, `.bin_value`/`.bin_capacity`/`.bin_count` describe the
  regions, and `.value` paints the region values back at full resolution with
  unbinned pixels left non-finite. *Not* a smoothing operation: it never mixes
  pixels into an average they do not belong to.
  It carries the geometry of the grid it was binned *from* (`x_edges`/`y_edges`,
  per-axis units and labels), and `.to_image_data()` hands that on; `.imshow()`
  draws even grids with `imshow` and uneven ones with `pcolormesh`.
- **Map mask** (`plot/image/compose.py`) — a pair of complementary soft masks in
  `[0, 1]` split by a line at `line_angle`, with a transition ramp whose width is
  a fraction of the image diagonal (`width`, so the look is resolution
  independent; `width <= 0` gives a hard split). Drives `blend_images`,
  `blend_stack` and `compose_maps`, which stitch two maps drawn with different
  colour maps into one RGBA image whose alpha is the coverage.
- **`K_B_C_G_Y_R_W`** (`plot/image/cmaps.py`) — the velocity colour map, black →
  blue → cyan → **green at zero velocity (position 0.50)** → yellow → red →
  white, with stops at `[0.00, 0.18, 0.44, 0.50, 0.56, 0.84, 1.00]`. Registered
  with matplotlib, so `cmap="K_B_C_G_Y_R_W"` works; `vel_cmap` is the same
  object. `to_rgba` maps a 2-D array plus a stretch to RGBA.
- **PSF** (`plot/image/psf.py`) — the observational point-spread function:
  `gaussian_psf(fwhm=..., e=..., theta=...)` builds a kernel, `convolve_psf`
  applies it (forward direction, for comparing a model to data), and
  `wiener_deconvolve` / `richardson_lucy` invert it (visualisation only — they
  amplify noise).

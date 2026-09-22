"""SPH-smoothed queries over a binned result: ``bins.sph_render[...]``.

The strict binning asks "which particles are in this cell"; ``sph_render`` asks
the same question with the particles smeared by their SPH kernel, so the answers
have to agree with the strict ones where they should (units, totals) and with
pynbody's own kernel where they can be checked exactly.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pynbody
import pytest
from pynbody.array import SimArray
from pynbody.sph import kernels

from pynbodyext.core.calculate import Bin1D
from pynbodyext.core.calculate.bins import BinNDResult, SphRender
from pynbodyext.core.calculate.bins.statistics import Percentile

#: Cell size of the 2-D test grids: 10 kpc across, 20 bins.
SPAN = 10.0
NBINS = 20
EDGE = SPAN / NBINS


def make_sim(n: int = 4000, *, h: float = 0.5, spread: float = 3.5, stars: int = 0) -> pynbody.SimSnap:
    """A small snapshot with hand-set smoothing lengths (no kdtree needed)."""
    rng = np.random.default_rng(7)
    sim = pynbody.new(dm=n, star=stars)
    for family, count in ((sim.dm, n), (sim.star, stars)):
        if not count:
            continue
        family["pos"] = SimArray(rng.uniform(-spread, spread, (count, 3)), "kpc")
        family["mass"] = SimArray(rng.uniform(0.5, 2.0, count), "Msol")
        family["vz"] = SimArray(rng.normal(0.0, 100.0, count), "km s**-1")
        family["smooth"] = SimArray(np.full(count, h), "kpc")
    return sim


def make_bins(sim: pynbody.SimSnap, *, mode: str = "linear"):
    """A 2-D (x, y) binning covering the test grid."""
    x = Bin1D("x", vmin=-SPAN / 2, vmax=SPAN / 2, nbins=NBINS, alias="x", mode=mode)
    y = Bin1D("y", vmin=-SPAN / 2, vmax=SPAN / 2, nbins=NBINS, alias="y", mode=mode)
    return (x @ y)(sim)


def one_particle(h: float, *, nbins: int = 21, span: float = 10.5):
    """One particle at the centre of a grid whose middle cell is centred on it."""
    sim = pynbody.new(dm=1)
    sim["pos"] = SimArray(np.array([[0.0, 0.0, 0.0]]), "kpc")
    sim["mass"] = SimArray([1.0], "Msol")
    sim["vz"] = SimArray([10.0], "km s**-1")
    sim["smooth"] = SimArray([h], "kpc")
    x = Bin1D("x", vmin=-span / 2, vmax=span / 2, nbins=nbins, alias="x")
    y = Bin1D("y", vmin=-span / 2, vmax=span / 2, nbins=nbins, alias="y")
    return (x @ y)(sim)


def reference_quantile(
    position: np.ndarray,
    mass: np.ndarray,
    value: np.ndarray,
    *,
    h: float,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    percentile: float,
    weight_by_mass: bool,
) -> np.ndarray:
    """Brute force: every particle over every cell its kernel reaches, by hand."""
    centres_x = 0.5 * (x_edges[:-1] + x_edges[1:])
    centres_y = 0.5 * (y_edges[:-1] + y_edges[1:])
    area = float(x_edges[1] - x_edges[0]) * float(y_edges[1] - y_edges[0])
    table = np.asarray(kernels.Kernel2D(kernels.CubicSplineKernel()).get_samples(dtype=float))
    statistic = Percentile(f"p{percentile:g}", percentile)
    reference = np.full((len(centres_x), len(centres_y)), np.nan)
    for i, x_centre in enumerate(centres_x):
        for j, y_centre in enumerate(centres_y):
            distance = np.hypot(position[:, 0] - x_centre, position[:, 1] - y_centre)
            index = ((distance / h) ** 2 / 4.0 * len(table)).astype(int)
            kernel_value = np.where(index < len(table), table[np.clip(index, 0, len(table) - 1)], 0.0)
            weight = kernel_value / h**2 * area
            if weight_by_mass:
                weight = weight * mass
            reference[i, j] = statistic(value[weight > 0.0], weight[weight > 0.0])
    return reference


# ---------------------------------------------------------------------------
# what can and cannot be rendered
# ---------------------------------------------------------------------------


def test_a_radial_profile_cannot_be_rendered() -> None:
    """One axis, and not a spatial one: there is no geometry to smear onto."""
    sim = make_sim(200)
    sim["r"] = SimArray(np.linalg.norm(np.asarray(sim["pos"]), axis=1), "kpc")
    bins = Bin1D("r", vmin=0.0, vmax=5.0, nbins=5)(sim)

    with pytest.raises(ValueError, match="2 or 3 axes"):
        bins.sph_render["count"]


def test_a_non_spatial_pair_cannot_be_rendered() -> None:
    """Two axes that are not x/y/z have no sky to render onto."""
    sim = make_sim(200)
    bins = (Bin1D("x", vmin=-3, vmax=3, nbins=4, alias="x") @ Bin1D("vz", vmin=-300, vmax=300, nbins=4, alias="vz"))(
        sim
    )

    with pytest.raises(ValueError, match="spatial"):
        bins.sph_render["count"]


def test_uneven_bins_cannot_be_rendered() -> None:
    """The renderer needs one cell shape, so the bins have to be uniform."""
    sim = make_sim(500)
    rng = np.random.default_rng(1)
    sim["x"] = SimArray(rng.uniform(0.2, 5.0, len(sim)), "kpc")
    x = Bin1D("x", vmin=0.2, vmax=5.0, nbins=6, alias="x", mode="log")
    y = Bin1D("y", vmin=-SPAN / 2, vmax=SPAN / 2, nbins=6, alias="y")
    bins = (x @ y)(sim)

    with pytest.raises(ValueError, match="evenly spaced"):
        bins.sph_render["count"]


def test_unusable_smoothing_lengths_are_rejected() -> None:
    """Zeros (or NaN) in ``smooth`` cannot be rendered, and say so."""
    rng = np.random.default_rng(0)
    sim = pynbody.new(dm=200)
    sim["pos"] = SimArray(rng.uniform(-3, 3, (200, 3)), "kpc")
    sim["mass"] = SimArray(np.ones(200), "Msol")
    sim["smooth"] = SimArray(np.zeros(200), "kpc")

    with pytest.raises(ValueError, match="positive smoothing lengths"):
        make_bins(sim).sph_render["count"]


def test_quantiles_are_not_implemented_yet() -> None:
    """Statistics that are neither kernel sums nor quantiles are rejected."""
    bins = make_bins(make_sim(500))

    with pytest.raises(KeyError):
        bins.sph_render["vz.nonsense"]


def test_unknown_keys_are_rejected() -> None:
    bins = make_bins(make_sim(200))

    with pytest.raises(KeyError, match="Unknown sph_render query"):
        bins.sph_render["nonsense"]


def test_one_view_answers_the_kernel_sums_and_the_quantiles() -> None:
    """The statistic picks the engine; there is no second entry point to remember."""
    bins = make_bins(make_sim(500))

    mean = np.asarray(bins.sph_render["vz.mean"])
    median = np.asarray(bins.sph_render["vz.median"])

    assert mean.shape == median.shape == (NBINS, NBINS)
    assert np.isfinite(mean).any() and np.isfinite(median).any()


# ---------------------------------------------------------------------------
# the two-dimensional (projected) engine
# ---------------------------------------------------------------------------


def test_count_integrates_the_kernel_to_the_particle_number() -> None:
    """``count`` is the kernel-integrated number of particles, and fractional."""
    count = np.asarray(make_bins(make_sim(3000)).sph_render["count"])

    assert count.shape == (NBINS, NBINS)
    assert count.sum() == pytest.approx(3000, rel=0.02)
    assert not np.allclose(count, np.round(count)), "a smoothed count should be fractional"


def test_a_render_is_a_neighbour_estimate_not_a_cell_census() -> None:
    """A particle outside the grid still reaches into it, as SPH interpolation does."""
    sim = make_sim(500, h=0.5)
    inside = make_bins(sim).sph_render["count"]
    # one more particle just outside the grid: the strict bins cannot see it ...
    extra = pynbody.new(dm=1)
    extra["pos"] = SimArray(np.array([[-SPAN / 2 - 0.3, 0.0, 0.0]]), "kpc")
    extra["mass"] = SimArray([1.0], "Msol")
    extra["smooth"] = SimArray([0.5], "kpc")
    assert float(np.asarray(make_bins(extra)["count"]).sum()) == 0.0
    # ... but its kernel reaches the edge cells of the render.
    assert float(np.asarray(make_bins(extra).sph_render["count"]).sum()) > 0.0
    assert inside.shape == (NBINS, NBINS)


def test_mass_sum_keeps_the_units_and_the_total_of_the_strict_query() -> None:
    """The smoothed ``mass.sum`` is comparable with ``bins["mass.sum"]``."""
    sim = make_sim(3000)
    bins = make_bins(sim)

    strict = bins["mass.sum"]
    smoothed = bins.sph_render["mass.sum"]

    assert smoothed.units == strict.units
    assert np.isnan(np.asarray(strict)).any(), "an empty strict cell is NaN, as everywhere else"
    assert np.nansum(np.asarray(smoothed)) == pytest.approx(np.nansum(np.asarray(strict)), rel=0.02)
    assert np.nansum(np.asarray(smoothed)) == pytest.approx(float(np.asarray(sim["mass"]).sum()), rel=0.02)
    assert smoothed.shape_bins == strict.shape_bins


def test_the_projection_matches_pynbody_kernel_sampling_exactly() -> None:
    """Our render is pynbody's: sample the projected kernel at each cell centre."""
    h, nbins, span = 1.0, 21, 10.5
    bins = one_particle(h, nbins=nbins, span=span)
    kernel = kernels.Kernel2D(kernels.CubicSplineKernel())

    centres = -span / 2 + (np.arange(nbins) + 0.5) * (span / nbins)
    grids = np.meshgrid(centres, centres, indexing="ij")
    distance = np.hypot(*grids)
    reference = np.zeros_like(distance)
    inside = distance < 2 * h
    reference[inside] = [kernel.get_value(r / h) / h**2 for r in distance[inside]]
    reference *= (span / nbins) ** 2

    # pynbody samples its kernel from a 0.02-step table, so agreement is to the
    # table's resolution rather than to machine precision.
    np.testing.assert_allclose(np.asarray(bins.sph_render["count"]), reference, rtol=0.02, atol=1e-3)


def test_mean_is_the_kernel_weighted_average() -> None:
    """``mean`` is ``Σ w f / Σ w``, and a constant field survives it."""
    sim = make_sim(3000)
    bins = make_bins(sim)

    plain = np.asarray(bins.sph_render["vz.mean"])
    weighted = np.asarray(bins.sph_render["vz.mean@mass"])
    absolute = np.asarray(bins.sph_render["vz.abs.mean"])

    assert plain.shape == (NBINS, NBINS)
    assert bins.sph_render["vz.mean"].units == bins["vz.mean"].units
    assert not np.allclose(plain, weighted), "@mass should change the answer"
    assert np.all(absolute[np.isfinite(absolute)] >= 0.0)
    assert np.nanmean(np.abs(absolute)) > np.nanmean(np.abs(plain))

    sim["const"] = SimArray(np.full(len(sim), 7.0), "Msol")
    constant = np.asarray(make_bins(sim).sph_render["const.mean"])
    populated = np.asarray(make_bins(sim).sph_render["count"]) > 1e-6
    np.testing.assert_allclose(constant[populated], 7.0, rtol=1e-6)


# ---------------------------------------------------------------------------
# integration with the rest of the result
# ---------------------------------------------------------------------------


def test_a_family_subresult_renders_only_its_own_particles() -> None:
    bins = make_bins(make_sim(2000, stars=800))

    every = float(np.asarray(bins.sph_render["count"]).sum())
    stars = float(np.asarray(bins.star.sph_render["count"]).sum())

    assert stars == pytest.approx(800, rel=0.05)
    assert stars < every
    assert bins.star.sph_render["count"].shape_bins == bins["count"].shape_bins


def test_the_view_is_cached_on_the_result_and_so_are_its_renders() -> None:
    bins = make_bins(make_sim(500))

    assert isinstance(bins.sph_render, SphRender)
    assert bins.sph_render is bins.sph_render
    assert bins.sph_render["count"] is bins.sph_render["count"]


def test_a_rendered_map_flows_into_the_image_layer() -> None:
    """The result is a BinsArray, so ``.image`` and the image layer just work."""
    rendered = make_bins(make_sim(2000)).sph_render["mass.sum"]

    assert rendered.image.shape == rendered.shape[::-1]
    assert rendered.image.extent is not None


# ---------------------------------------------------------------------------
# the three-dimensional engine
# ---------------------------------------------------------------------------


def test_a_volume_render_places_the_kernel_on_the_cell_grid() -> None:
    """Three axes render onto the cell grid; a single particle is analytic."""
    h, nbins, span = 1.0, 11, 5.5
    sim = pynbody.new(dm=1)
    sim["pos"] = SimArray(np.array([[0.0, 0.0, 0.0]]), "kpc")
    sim["mass"] = SimArray([1.0], "Msol")
    sim["smooth"] = SimArray([h], "kpc")
    axes = [Bin1D(prop, vmin=-span / 2, vmax=span / 2, nbins=nbins, alias=prop) for prop in ("x", "y", "z")]
    bins = (axes[0] @ axes[1] @ axes[2])(sim)

    count = np.asarray(bins.sph_render["count"])
    edge = span / nbins
    middle = nbins // 2

    assert count.shape == (nbins, nbins, nbins)
    assert count[middle, middle, middle] == pytest.approx(kernels.CubicSplineKernel().value(0.0, h) * edge**3, rel=1e-6)
    assert count.sum() == pytest.approx(1.0, rel=0.05)


def test_a_volume_render_matches_a_direct_kernel_sum() -> None:
    """With equal y/z resolutions, pynbody's 3-D grid is what it claims to be."""
    h, nbins, span, n = 0.7, 9, 5.4, 120
    rng = np.random.default_rng(11)
    sim = pynbody.new(dm=n)
    sim["pos"] = SimArray(rng.uniform(-2, 2, (n, 3)), "kpc")
    sim["mass"] = SimArray(rng.uniform(0.5, 2.0, n), "Msol")
    sim["smooth"] = SimArray(np.full(n, h), "kpc")
    axes = [Bin1D(prop, vmin=-span / 2, vmax=span / 2, nbins=nbins, alias=prop) for prop in ("x", "y", "z")]
    bins = (axes[0] @ axes[1] @ axes[2])(sim)

    rendered = np.asarray(bins.sph_render["mass.sum"])

    edge = span / nbins
    centres = -span / 2 + (np.arange(nbins) + 0.5) * edge
    grid = np.meshgrid(centres, centres, centres, indexing="ij")
    kernel = kernels.CubicSplineKernel()
    position = np.asarray(sim["pos"])
    mass = np.asarray(sim["mass"])
    reference = np.zeros((nbins, nbins, nbins))
    for atom, weight in zip(position, mass, strict=True):
        distance = np.sqrt(
            (grid[0] - atom[0]) ** 2 + (grid[1] - atom[1]) ** 2 + (grid[2] - atom[2]) ** 2
        )
        reference += weight * edge**3 * kernel.value(distance, h)

    # pynbody samples its kernel from a 0.02-step lookup table, so the agreement
    # is to the table's resolution rather than to machine precision: about a
    # percent of the peak, and closer than that in the bulk.
    peak = reference.max()
    assert rendered.sum() == pytest.approx(reference.sum(), rel=0.01)
    np.testing.assert_allclose(rendered, reference, rtol=0.03, atol=0.02 * peak)


def test_an_unequal_z_resolution_renders_the_grid_it_was_asked_for() -> None:
    """``nz != ny`` used to be misplaced (pynbody sized z pixels with ``ny``)."""
    h, span = 0.5, 3.0
    counts = {"x": 4, "y": 6, "z": 3}
    atom = np.array([0.1, 0.1, 1.0])  # inside cell (2, 3, 2) of the grid below
    sim = pynbody.new(dm=1)
    sim["pos"] = SimArray([atom], "kpc")
    sim["mass"] = SimArray([1.0], "Msol")
    sim["smooth"] = SimArray([h], "kpc")
    axes = [Bin1D(prop, vmin=-span / 2, vmax=span / 2, nbins=counts[prop], alias=prop) for prop in ("x", "y", "z")]
    bins = (axes[0] @ axes[1] @ axes[2])(sim)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the upstream bug this used to warn about is fixed
        count = np.asarray(bins.sph_render["count"])

    assert count.shape == (4, 6, 3)
    assert np.unravel_index(count.argmax(), count.shape) == (2, 3, 2)
    assert count[:, :, 2].sum() == pytest.approx(count.sum()), "the kernel belongs in the containing z layer"

    centre = np.array(
        [-span / 2 + (index + 0.5) * (span / counts[prop]) for prop, index in zip(("x", "y", "z"), (2, 3, 2))]
    )
    volume = (span / counts["x"]) * (span / counts["y"]) * (span / counts["z"])
    expected = kernels.CubicSplineKernel().value(float(np.linalg.norm(centre - atom)), h) * volume
    # the renderer reads its kernel from a 0.02-step table, as the sibling tests note
    assert count[2, 3, 2] == pytest.approx(expected, rel=0.03)


# ---------------------------------------------------------------------------
# the kernel-weighted quantiles
# ---------------------------------------------------------------------------


def test_a_median_of_one_particle_is_that_particle() -> None:
    """One particle, one neighbour: the weighted median is its value, exactly."""
    sim = pynbody.new(dm=1)
    sim["pos"] = SimArray(np.array([[0.0, 0.0, 0.0]]), "kpc")
    sim["mass"] = SimArray([1.0], "Msol")
    sim["vz"] = SimArray([42.0], "km s**-1")
    sim["smooth"] = SimArray([1.0], "kpc")
    bins = make_bins(sim)

    median = np.asarray(bins.sph_render["vz.median"])
    count = np.asarray(bins.sph_render["count"])

    np.testing.assert_allclose(median[count > 0.0], 42.0)
    assert np.isnan(median[count == 0.0]).all(), "cells the kernel misses stay empty"


def test_a_constant_field_has_the_same_median_as_mean() -> None:
    """Whatever the weights are, a constant field is its own quantiles."""
    sim = make_sim(2000)
    sim["const"] = SimArray(np.full(len(sim), 7.0), "km s**-1")
    bins = make_bins(sim)
    render = SphRender(bins)

    count = np.asarray(bins.sph_render["count"])
    for key in ("const.median", "const.p16", "const.mean"):
        values = np.asarray(render[key])
        np.testing.assert_allclose(values[count > 0.0], 7.0, rtol=1e-6, err_msg=key)


def test_quantiles_match_a_brute_force_weighted_quantile() -> None:
    """Every neighbour, every weight, computed by hand in the test."""
    h, nbins, span, n = 0.6, 6, 6.0, 200
    rng = np.random.default_rng(5)
    sim = pynbody.new(dm=n)
    sim["pos"] = SimArray(rng.uniform(-2, 2, (n, 3)), "kpc")
    sim["mass"] = SimArray(rng.uniform(0.5, 2.0, n), "Msol")
    sim["vz"] = SimArray(rng.normal(0.0, 100.0, n), "km s**-1")
    sim["smooth"] = SimArray(np.full(n, h), "kpc")
    x = Bin1D("x", vmin=-span / 2, vmax=span / 2, nbins=nbins, alias="x")
    y = Bin1D("y", vmin=-span / 2, vmax=span / 2, nbins=nbins, alias="y")
    bins = (x @ y)(sim)

    rendered = np.asarray(SphRender(bins)["vz.abs.p16@mass"])

    reference = reference_quantile(
        np.asarray(sim["pos"]),
        np.asarray(sim["mass"]),
        np.abs(np.asarray(sim["vz"])),
        h=h,
        x_edges=np.linspace(-span / 2, span / 2, nbins + 1),
        y_edges=np.linspace(-span / 2, span / 2, nbins + 1),
        percentile=16.0,
        weight_by_mass=True,
    )

    # The engine sums the neighbours in its own order and the reference in particle
    # order, so the two agree to floating-point, not bit-for-bit.
    np.testing.assert_allclose(rendered, reference, rtol=1e-6, atol=1e-6)


def test_a_grid_taller_than_one_slab_is_scattered_correctly() -> None:
    """Slabs are an implementation detail: the answer cannot depend on the split."""
    h, nbins_x, nbins_y, span, n = 0.6, 40, 4, 8.0, 400
    rng = np.random.default_rng(17)
    sim = pynbody.new(dm=n)
    sim["pos"] = SimArray(rng.uniform(-3, 3, (n, 3)), "kpc")
    sim["mass"] = SimArray(rng.uniform(0.5, 2.0, n), "Msol")
    sim["vz"] = SimArray(rng.normal(0.0, 100.0, n), "km s**-1")
    sim["smooth"] = SimArray(np.full(n, h), "kpc")
    x = Bin1D("x", vmin=-span / 2, vmax=span / 2, nbins=nbins_x, alias="x")
    y = Bin1D("y", vmin=-span / 2, vmax=span / 2, nbins=nbins_y, alias="y")
    bins = (x @ y)(sim)

    rendered = np.asarray(SphRender(bins)["vz.median"])

    reference = reference_quantile(
        np.asarray(sim["pos"]),
        np.asarray(sim["mass"]),
        np.asarray(sim["vz"]),
        h=h,
        x_edges=np.linspace(-span / 2, span / 2, nbins_x + 1),
        y_edges=np.linspace(-span / 2, span / 2, nbins_y + 1),
        percentile=50.0,
        weight_by_mass=False,
    )
    assert rendered.shape == (nbins_x, nbins_y)
    # rows 0-31 are one slab and the rest another, so this pins the row bookkeeping
    np.testing.assert_allclose(rendered, reference, rtol=1e-4, atol=1e-4)


def test_quantiles_respect_transforms_weights_and_units() -> None:
    sim = make_sim(3000)
    bins = make_bins(sim)

    render = SphRender(bins)
    absolute = render["vz.abs.p16"]
    weighted = render["vz.abs.p16@mass"]

    assert absolute.units == bins["vz.mean"].units
    assert np.nanmin(np.asarray(absolute)) >= 0.0, "|vz| percentiles cannot be negative"
    assert not np.allclose(absolute, weighted), "@mass should move the percentile"
    # p50 and median are the same statistic under two names
    np.testing.assert_allclose(
        np.asarray(render["vz.p50"]), np.asarray(render["vz.median"]), equal_nan=True
    )


def test_a_quantile_then_a_kernel_sum_still_works() -> None:
    """pynbody caches its kernel table per kernel: ask for it in its own dtype."""
    bins = make_bins(make_sim(500))

    median = np.asarray(bins.sph_render["vz.median"])
    mean = np.asarray(bins.sph_render["vz.mean"])

    assert median.shape == mean.shape == (NBINS, NBINS)


def test_the_compiled_pair_builder_matches_the_numpy_one() -> None:
    """The C++ kernel is an optimisation of the fallback, not a second definition."""
    from pynbodyext.core.calculate.bins.sph_render import (
        _kernel_table,
        _kernel_weights,
        _native_pair_builder,
        _scatter_pairs,
    )

    builder = _native_pair_builder()
    if builder is None:
        pytest.skip("the optional C++ extension is not built")
    sim = make_sim(4000, h=0.5)
    bins = make_bins(sim)
    render = SphRender(bins)
    plan = render._layout()
    particle, shift = render._images(plan)
    position = plan["position"][particle][:, [0, 1]] + shift
    smoothing = render._smoothing(plan["smooth"])[particle]
    support = 2.0 * smoothing
    centres = [np.asarray(plan["axes"][prop].edges[:-1] + plan["axes"][prop].edges[1:]) / 2 for prop in plan["present"]]
    origins = [float(plan["axes"][prop].edges[0]) for prop in plan["present"]]
    widths = [plan["widths"][prop] for prop in plan["present"]]
    strides = [NBINS, 1]
    kernel = render._kernel(projected=True)
    measure = plan["measure"]
    rng = np.random.default_rng(3)
    value = rng.normal(0.0, 100.0, len(position))
    extra = rng.uniform(0.5, 2.0, len(position))

    native = builder(
        np.ascontiguousarray(position),
        np.ascontiguousarray(smoothing),
        np.ascontiguousarray(support),
        np.ascontiguousarray(value),
        np.ascontiguousarray(extra),
        _kernel_table(kernel),
        int(getattr(kernel, "h_power", 3)),
        float(measure),
        origins,
        widths,
        [NBINS, NBINS],
        strides,
        0,
        NBINS,
        0,
    )
    cell, distance, found = _scatter_pairs(position, smoothing, support, centres, origins, widths, strides, 0, NBINS)
    fallback = (
        cell,
        value[found],
        _kernel_weights(np.sqrt(distance), smoothing[found], kernel) * measure * extra[found],
    )

    assert len(native[0]) == len(fallback[0]) > 0
    compiled = np.column_stack([np.asarray(part) for part in native])
    interpreted = np.column_stack([np.asarray(part) for part in fallback])
    # Compare as multisets: a lexicographic sort of (cell, value, weight) is the
    # same list whichever engine produced it.  The slight tolerance covers the
    # fallback's sqrt-then-square round trip into the kernel lookup.
    order = np.lexsort((compiled[:, 2], compiled[:, 1], compiled[:, 0]))
    other = np.lexsort((interpreted[:, 2], interpreted[:, 1], interpreted[:, 0]))
    np.testing.assert_allclose(compiled[order], interpreted[other], rtol=1e-9, atol=1e-12)


def test_the_numpy_fallback_answers_the_same_quantile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the optional extension the NumPy path still answers the same query."""
    import pynbodyext.core.calculate.bins.sph_render as module

    sim = make_sim(3000, h=0.5)
    bins = make_bins(sim)
    native = np.asarray(SphRender(bins)["vz.median"])

    monkeypatch.setattr(module, "_native_pair_builder", lambda: None)
    fallback = np.asarray(SphRender(bins)["vz.median"])

    assert np.isfinite(fallback).sum() == np.isfinite(native).sum() > 0
    np.testing.assert_allclose(fallback, native, rtol=1e-6, atol=1e-6, equal_nan=True)


# ---------------------------------------------------------------------------
# derived quantities: densities and the properties that opt in with allow_sph
# ---------------------------------------------------------------------------


def make_two_family_sim(n: int = 2000, *, h: float = 0.5) -> pynbody.SimSnap:
    """A snapshot whose gas and dark matter can be smoothed separately."""
    rng = np.random.default_rng(31)
    sim = pynbody.new(dm=n, gas=n)
    for family in (sim.dm, sim.gas):
        family["pos"] = SimArray(rng.uniform(-3.5, 3.5, (n, 3)), "kpc")
        family["mass"] = SimArray(rng.uniform(0.5, 2.0, n), "Msol")
        family["smooth"] = SimArray(np.full(n, h), "kpc")
    return sim


def test_a_density_query_divides_the_smoothed_map_by_the_cell_measure() -> None:
    """``<field>.density`` is the strict suffix, with the smoothed numerator."""
    bins = make_bins(make_sim(3000))
    density = bins.sph_render["mass.sum.density"]

    expected = np.asarray(bins.sph_render["mass.sum"]) / np.asarray(bins["measure"])

    np.testing.assert_allclose(np.asarray(density), expected)
    assert density.units == bins["mass.sum.density"].units
    # a smoothed numerator is a different map from the strict one
    assert not np.allclose(
        np.nan_to_num(np.asarray(density)), np.nan_to_num(np.asarray(bins["mass.sum.density"]))
    )


def test_a_derived_property_declared_allow_sph_is_smoothed() -> None:
    """``gas_fraction`` runs its own callback against the smoothed queries."""
    bins = make_bins(make_two_family_sim())
    rendered = np.asarray(bins.sph_render["gas_fraction"])

    with np.errstate(invalid="ignore", divide="ignore"):
        expected = np.asarray(bins.gas.sph_render["mass.sum"]) / np.asarray(bins.sph_render["mass.sum"])

    np.testing.assert_allclose(rendered, expected, rtol=1e-12, equal_nan=True)
    # each family is smoothed with its own particles, so this is not the strict ratio
    assert not np.allclose(np.nan_to_num(rendered), np.nan_to_num(np.asarray(bins["gas_fraction"])))


def test_a_user_registered_property_can_opt_in_to_sph() -> None:
    """The flag is public: registering is how a plugin says "smooth this too"."""
    from pynbodyext.core.calculate.bins.extensions import BIN_RESULT_EXTENSIONS

    @BinNDResult.derived("dm_mass_fraction", allow_sph=True, overwrite=True)
    def dm_mass_fraction(result: Any) -> np.ndarray:
        return result.dm["mass.sum"] / result["mass.sum"]

    try:
        bins = make_bins(make_two_family_sim())

        with np.errstate(invalid="ignore", divide="ignore"):
            rendered = np.asarray(bins.sph_render["dm_mass_fraction"])
            expected = np.asarray(bins.dm.sph_render["mass.sum"]) / np.asarray(bins.sph_render["mass.sum"])

        assert dm_mass_fraction is not None
        np.testing.assert_allclose(rendered, expected, rtol=1e-12, equal_nan=True)
    finally:
        # a test-only property must not leak into the rest of the suite
        BIN_RESULT_EXTENSIONS._derived_specs[BinNDResult].pop("dm_mass_fraction", None)


def test_a_derived_property_without_allow_sph_is_refused() -> None:
    """Order-dependent properties are refused by name, not quietly mixed in."""
    bins = make_bins(make_sim(500))

    with pytest.raises(KeyError, match="allow_sph=True"):
        bins.sph_render["enclosed_mass"]


def test_geometry_is_the_same_map_on_both_views() -> None:
    """The grid is the one the strict query used, so its measure is not smoothed."""
    bins = make_bins(make_sim(300))

    np.testing.assert_allclose(np.asarray(bins.sph_render["measure"]), np.asarray(bins["measure"]))


# ---------------------------------------------------------------------------
# periodic boxes
# ---------------------------------------------------------------------------


def make_periodic_sim(boxsize: float, *, span: float = 1000.0, h: float = 20.0, count: int = 1):
    """A snapshot whose coordinates sit in ``[-L/2, L/2)``, as gadget writes them.

    ``boxsize=0`` leaves the snapshot non-periodic, for the control case.
    """
    rng = np.random.default_rng(13)
    sim = pynbody.new(dm=count)
    sim["pos"] = SimArray(rng.uniform(-span / 2, span / 2, (count, 3)), "kpc")
    if count == 1:
        sim["pos"] = SimArray(np.array([[-span / 2 + 0.1, 0.0, 0.0]]), "kpc")
    sim["mass"] = SimArray(np.full(count, 1.0), "Msol")
    sim["vz"] = SimArray(np.full(count, 5.0), "km s**-1")
    sim["smooth"] = SimArray(np.full(count, h), "kpc")
    if boxsize > 0:
        sim.properties["boxsize"] = boxsize
    axes = [Bin1D(prop, vmin=-span / 2, vmax=span / 2, nbins=32, alias=prop) for prop in ("x", "y")]
    return (axes[0] @ axes[1])(sim)


def test_a_quantile_accepts_coordinates_outside_the_box() -> None:
    """Snapshots put particles outside [0, L); the scatter handles that."""
    bins = make_periodic_sim(1000.0, count=200)

    median = np.asarray(bins.sph_render["vz.median"])

    assert median.shape == (32, 32)
    assert np.isfinite(median).any()


def test_the_box_wraps_the_quantile_neighbours_as_the_sums_do() -> None:
    """A particle just inside one edge reaches the other, in both engines."""
    bins = make_periodic_sim(1000.0)

    count = np.asarray(bins.sph_render["count"])
    median = np.asarray(bins.sph_render["vz.median"])

    for cell in (0, -1):  # either side of the periodic boundary
        assert count[cell, 16] > 0.0
        assert median[cell, 16] == 5.0
    # without a box, the far side is simply far away
    open_bins = make_periodic_sim(0.0)
    assert np.asarray(open_bins.sph_render["count"])[-1, 16] == 0.0
    assert np.isnan(np.asarray(open_bins.sph_render["vz.median"])[-1, 16])


def test_a_mixed_particle_set_points_at_the_family_with_smoothing() -> None:
    """Only some families carry smoothing lengths; say which way out."""
    rng = np.random.default_rng(21)
    sim = pynbody.new(dm=400, gas=400)
    for family, count in ((sim.dm, 400), (sim.gas, 400)):
        family["pos"] = SimArray(rng.uniform(-3.5, 3.5, (count, 3)), "kpc")
        family["mass"] = SimArray(rng.uniform(0.5, 2.0, count), "Msol")
        family["vz"] = SimArray(rng.normal(0.0, 100.0, count), "km s**-1")
    sim.gas["smooth"] = SimArray(np.full(400, 0.5), "kpc")  # only the gas is SPH

    with pytest.raises(ValueError, match="family"):
        make_bins(sim).sph_render["count"]

    assert make_bins(sim.gas).sph_render["count"].shape == (NBINS, NBINS)

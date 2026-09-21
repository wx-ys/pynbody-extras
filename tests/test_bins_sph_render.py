"""SPH-smoothed queries over a binned result: ``bins.sph_render[...]``.

The strict binning asks "which particles are in this cell"; ``sph_render`` asks
the same question with the particles smeared by their SPH kernel, so the answers
have to agree with the strict ones where they should (units, totals) and with
pynbody's own kernel where they can be checked exactly.
"""

from __future__ import annotations

import warnings

import numpy as np
import pynbody
import pytest
from pynbody.array import SimArray
from pynbody.sph import kernels

from pynbodyext.core.calculate import Bin1D
from pynbodyext.core.calculate.bins import SphRender
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


def test_an_unequal_z_resolution_warns_about_pynbody_grid_bug() -> None:
    """pynbody sizes z pixels with ``ny``; say so rather than hide it."""
    sim = make_sim(300, h=0.5)
    axes = [
        Bin1D(prop, vmin=-SPAN / 2, vmax=SPAN / 2, nbins=nbins, alias=prop)
        for prop, nbins in (("x", 5), ("y", 5), ("z", 3))
    ]
    bins = (axes[0] @ axes[1] @ axes[2])(sim)

    with pytest.warns(UserWarning, match="z pixels"):
        bins.sph_render["count"]

    equal = [
        Bin1D(prop, vmin=-SPAN / 2, vmax=SPAN / 2, nbins=5, alias=prop) for prop in ("x", "y", "z")
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a matching resolution must stay quiet
        (equal[0] @ equal[1] @ equal[2])(sim).sph_render["count"]


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

    median = np.asarray(SphRender(bins, neighbours=1)["vz.median"])
    count = np.asarray(bins.sph_render["count"])

    np.testing.assert_allclose(median[count > 0.0], 42.0)
    assert np.isnan(median[count == 0.0]).all(), "cells the kernel misses stay empty"


def test_a_constant_field_has_the_same_median_as_mean() -> None:
    """Whatever the weights are, a constant field is its own quantiles."""
    sim = make_sim(2000)
    sim["const"] = SimArray(np.full(len(sim), 7.0), "km s**-1")
    bins = make_bins(sim)

    count = np.asarray(bins.sph_render["count"])
    for key in ("const.median", "const.p16", "const.mean"):
        values = np.asarray(bins.sph_render[key])
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

    rendered = np.asarray(SphRender(bins, neighbours=n)["vz.abs.p16@mass"])

    edge = span / nbins
    centres = -span / 2 + (np.arange(nbins) + 0.5) * edge
    table = np.asarray(kernels.Kernel2D(kernels.CubicSplineKernel()).get_samples(dtype=float))
    q2 = np.arange(len(table)) * 0.02
    position = np.asarray(sim["pos"])
    mass, vz = np.asarray(sim["mass"]), np.abs(np.asarray(sim["vz"]))
    statistic = Percentile("p16", 16.0)
    reference = np.full((nbins, nbins), np.nan)
    for i, x_centre in enumerate(centres):
        for j, y_centre in enumerate(centres):
            distance = np.hypot(position[:, 0] - x_centre, position[:, 1] - y_centre)
            weight = np.interp((distance / h) ** 2, q2, table) / h**2 * edge**2 * mass
            reference[i, j] = statistic(vz[weight > 0.0], weight[weight > 0.0])

    np.testing.assert_allclose(rendered, reference, rtol=1e-9, atol=1e-9)


def test_quantiles_respect_transforms_weights_and_units() -> None:
    sim = make_sim(3000)
    bins = make_bins(sim)

    absolute = bins.sph_render["vz.abs.p16"]
    weighted = bins.sph_render["vz.abs.p16@mass"]

    assert absolute.units == bins["vz.mean"].units
    assert np.nanmin(np.asarray(absolute)) >= 0.0, "|vz| percentiles cannot be negative"
    assert not np.allclose(absolute, weighted), "@mass should move the percentile"
    # p50 and median are the same statistic under two names
    np.testing.assert_allclose(
        np.asarray(bins.sph_render["vz.p50"]), np.asarray(bins.sph_render["vz.median"]), equal_nan=True
    )


def test_the_neighbour_count_must_be_positive() -> None:
    bins = make_bins(make_sim(200))

    with pytest.raises(ValueError, match="neighbours"):
        SphRender(bins, neighbours=0)


def test_a_quantile_then_a_kernel_sum_still_works() -> None:
    """pynbody caches its kernel table per kernel: ask for it in its own dtype."""
    bins = make_bins(make_sim(500))

    median = np.asarray(bins.sph_render["vz.median"])
    mean = np.asarray(bins.sph_render["vz.mean"])

    assert median.shape == mean.shape == (NBINS, NBINS)


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
    """scipy's periodic tree wants [0, L); snapshots are not written that way."""
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

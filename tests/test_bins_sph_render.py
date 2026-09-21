"""SPH-smoothed queries over a binned result: ``bins.sph_render[...]``.

The strict binning asks "which particles are in this cell"; ``sph_render`` asks
the same question with the particles smeared by their SPH kernel, so the answers
have to agree with the strict ones where they should (units, totals) and with
pynbody's own kernel where they can be checked exactly.
"""

from __future__ import annotations

import numpy as np
import pynbody
import pytest
from pynbody.array import SimArray
from pynbody.sph import kernels

from pynbodyext.core.calculate import Bin1D
from pynbodyext.core.calculate.bins import SphRender

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
    """A kernel-weighted median needs neighbour lists, not two kernel sums."""
    bins = make_bins(make_sim(500))

    with pytest.raises(NotImplementedError, match="quantile"):
        bins.sph_render["vz.median"]


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

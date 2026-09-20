"""Tests for ``pynbodyext.plot.image.compose`` (masking and map stitching)."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

from pynbodyext.plot.image.cmaps import K_B_C_G_Y_R_W, to_rgba
from pynbodyext.plot.image.compose import blend_images, blend_stack, compose_maps, create_map_mask, imshow_compose


def gradient(shape: tuple[int, int] = (20, 30)) -> np.ndarray:
    """A map that increases along x, so the two halves of a split are distinct."""
    return np.tile(np.linspace(0.0, 1.0, shape[1]), (shape[0], 1))


def test_create_map_mask_returns_complementary_unit_masks() -> None:
    show = np.zeros((12, 16))

    mask1, mask2 = create_map_mask(show, line_angle=45, width=0.1)

    assert mask1.shape == mask2.shape == show.shape
    assert mask1.min() >= 0.0 and mask1.max() <= 1.0
    np.testing.assert_allclose(mask1 + mask2, 1.0)


def test_create_map_mask_hard_split_is_binary() -> None:
    mask1, mask2 = create_map_mask(np.zeros((8, 8)), line_angle=45, width=0.0)

    assert set(np.unique(mask1)) == {0.0, 1.0}
    np.testing.assert_allclose(mask1 + mask2, 1.0)
    assert mask1[0, 0] == 0.0  # below the diagonal
    assert mask1[-1, -1] == 1.0  # above it


def test_create_map_mask_soft_split_ramps_across_the_line() -> None:
    mask1, _ = create_map_mask(np.zeros((41, 41)), line_angle=0.0, width=0.2)

    column = mask1[20]
    transition = column[(column > 0.0) & (column < 1.0)]
    assert transition.size > 1  # a real ramp, not a step
    assert column[0] == 0.0 and column[-1] == 1.0
    assert np.all(np.diff(column) >= 0.0)


def test_create_map_mask_wider_ramp_is_smoother() -> None:
    narrow, _ = create_map_mask(np.zeros((41, 41)), line_angle=0.0, width=0.05)
    wide, _ = create_map_mask(np.zeros((41, 41)), line_angle=0.0, width=0.5)

    partial = lambda mask: int(((mask > 0.0) & (mask < 1.0)).sum())  # noqa: E731
    assert partial(wide) > partial(narrow)


def test_create_map_mask_angle_selects_the_axis() -> None:
    horizontal, _ = create_map_mask(np.zeros((10, 10)), line_angle=0.0, width=0.0)
    vertical, _ = create_map_mask(np.zeros((10, 10)), line_angle=90.0, width=0.0)

    # A zero-angle line splits along x, so every row is identical...
    np.testing.assert_allclose(horizontal, np.tile(horizontal[0], (10, 1)))
    # ...and a 90-degree one splits along y, so every column is identical.
    np.testing.assert_allclose(vertical, np.tile(vertical[:, :1], (1, 10)))
    np.testing.assert_allclose(vertical, horizontal.T)


def test_create_map_mask_accepts_rgb_arrays_and_shapes() -> None:
    rgb = np.zeros((6, 9, 3))

    from_rgb = create_map_mask(rgb)
    from_shape = create_map_mask((6, 9))

    assert from_rgb[0].shape == (6, 9)
    np.testing.assert_allclose(from_rgb[0], from_shape[0])
    np.testing.assert_allclose(from_rgb[1], from_shape[1])


def test_create_map_mask_honours_an_explicit_centre() -> None:
    mask1, _ = create_map_mask((11, 11), line_angle=0.0, width=0.0, center=(0.0, 0.0))

    assert mask1[0, 0] == 0.0
    assert mask1[0, -1] == 1.0
    np.testing.assert_allclose(mask1[:, 0], 0.0)  # the line moved from the centre to the left edge
    np.testing.assert_allclose(mask1[:, 1:], 1.0)


def test_create_map_mask_rejects_non_image_input() -> None:
    with pytest.raises(ValueError, match="2-D"):
        create_map_mask(np.zeros(5))


# ---------------------------------------------------------------------------
# blending
# ---------------------------------------------------------------------------


def test_blend_images_crossfades_by_the_mask() -> None:
    image1 = np.ones((4, 4)) * 4.0
    image2 = np.ones((4, 4)) * 8.0
    mask = np.full((4, 4), 0.25)

    blended = blend_images(image1, image2, mask)

    np.testing.assert_allclose(blended, 0.25 * 4.0 + 0.75 * 8.0)


def test_blend_images_broadcasts_the_mask_over_colour_channels() -> None:
    image1 = np.zeros((3, 3, 3))
    image2 = np.ones((3, 3, 3))
    mask = np.full((3, 3), 0.5)

    blended = blend_images(image1, image2, mask)

    assert blended.shape == (3, 3, 3)
    np.testing.assert_allclose(blended, 0.5)


def test_blend_images_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="shape"):
        blend_images(np.zeros((4, 4)), np.zeros((4, 5)), np.zeros((4, 4)))


def test_blend_stack_weights_each_layer() -> None:
    layers = [np.full((2, 2), 4.0), np.full((2, 2), 8.0)]
    weights = [np.full((2, 2), 0.25), np.full((2, 2), 0.75)]

    stacked = blend_stack(layers, weights)

    np.testing.assert_allclose(stacked, 7.0)


def test_blend_stack_leaves_unsupported_pixels_empty() -> None:
    layers = [np.ones((2, 2)), np.ones((2, 2))]
    weights = [np.zeros((2, 2)), np.zeros((2, 2))]

    stacked = blend_stack(layers, weights)

    np.testing.assert_allclose(stacked, 0.0)


def test_blend_stack_validates_weight_shapes() -> None:
    with pytest.raises(ValueError, match="does not match"):
        blend_stack([np.zeros((2, 2))], [np.zeros((3, 3))])


# ---------------------------------------------------------------------------
# compose_maps / imshow_compose
# ---------------------------------------------------------------------------


def test_compose_maps_returns_an_rgba_image() -> None:
    composed = compose_maps(gradient(), gradient()[:, ::-1])

    assert composed.shape == (20, 30, 4)
    assert 0.0 <= composed.min() and composed.max() <= 1.0


def test_compose_maps_uses_each_cmap_on_its_own_side() -> None:
    data1 = gradient()
    data2 = gradient()

    # A zero-angle line leaves map 1 on the right and map 2 on the left.
    composed = compose_maps(data1, data2, cmap1="viridis", cmap2="magma", line_angle=0.0, width=0.0)

    left = to_rgba(data1, "viridis", vmin=0.0, vmax=1.0)[..., :3]
    right = to_rgba(data2, "magma", vmin=0.0, vmax=1.0)[..., :3]
    np.testing.assert_allclose(composed[:, -1, :3], left[:, -1])
    np.testing.assert_allclose(composed[:, 0, :3], right[:, 0])
    assert not np.allclose(composed[:, 0], composed[:, -1])


def test_compose_maps_crossfades_in_the_transition_band() -> None:
    data1 = np.ones((21, 21))
    data2 = np.zeros((21, 21))

    composed = compose_maps(
        data1,
        data2,
        cmap1="gray",
        cmap2="gray",
        vmin1=0.0,
        vmax1=1.0,
        vmin2=0.0,
        vmax2=1.0,
        line_angle=180.0,  # map 1 (white) on the left, map 2 (black) on the right
        width=0.4,
    )

    centremost = composed[10, 10, 0]
    assert composed[10, 0, 0] > centremost > composed[10, -1, 0]


def test_compose_maps_keeps_empty_bins_transparent() -> None:
    data1 = np.full((8, 8), 0.5)
    data2 = np.full((8, 8), 0.5)
    data2[0, 0] = np.nan
    mask = np.zeros((8, 8))  # everything comes from the second map

    composed = compose_maps(data1, data2, mask=mask)

    assert composed[0, 0, 3] == 0.0
    assert composed[4, 4, 3] == 1.0


def test_compose_maps_accepts_an_explicit_mask() -> None:
    data1 = np.zeros((4, 4))
    data2 = np.ones((4, 4))
    mask = np.zeros((4, 4))

    composed = compose_maps(data1, data2, cmap1="gray", cmap2="gray", vmin2=0.0, vmax2=1.0, mask=mask)

    np.testing.assert_allclose(composed[..., 0], 1.0)  # only the second (white) map shows


def test_compose_maps_validates_the_mask_shape() -> None:
    with pytest.raises(ValueError, match="mask"):
        compose_maps(np.zeros((4, 4)), np.zeros((4, 4)), mask=np.zeros((2, 2)))


def test_compose_maps_defaults_to_the_velocity_cmap() -> None:
    data = gradient()

    composed = compose_maps(data, data, vmin1=0.0, vmax1=1.0, vmin2=0.0, vmax2=1.0)

    np.testing.assert_allclose(composed[..., :3], np.asarray(K_B_C_G_Y_R_W(data))[..., :3])


def test_imshow_compose_draws_the_image_and_two_colorbars() -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        artist = imshow_compose(
            gradient(),
            gradient()[:, ::-1],
            ax=ax,
            extent=(0.0, 30.0, 0.0, 20.0),
            label1="gas density",
            label2="dark matter density",
        )
        assert tuple(artist.get_extent()) == (0.0, 30.0, 0.0, 20.0)
        assert len(fig.axes) == 3  # the map plus one colour bar per map
    finally:
        plt.close(fig)


def test_imshow_compose_can_skip_the_colorbars() -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        imshow_compose(gradient(), gradient(), ax=ax, colorbars=False)
        assert len(fig.axes) == 1
    finally:
        plt.close(fig)

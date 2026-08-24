from __future__ import annotations

import numpy as np
import pytest

from pynbodyext.gravity import Gravity, KernelKind


@pytest.fixture
def particles():
    rng = np.random.default_rng(42)
    n = 256
    pos = rng.uniform(-1.0, 1.0, (n, 3))
    mass = 1.0 + rng.uniform(0.0, 1.0, n)
    return pos, mass


def test_tree_accelerations_match_direct(particles):
    pos, mass = particles
    g = Gravity(pos, mass, leaf_capacity=8, multipole_order=2)
    acc_tree = g.tree_accelerations(theta=0.0, threads=1)
    acc_direct = g.direct_accelerations(threads=1)
    np.testing.assert_allclose(acc_tree, acc_direct, atol=1e-10)


def test_tree_potentials_match_direct(particles):
    pos, mass = particles
    g = Gravity(pos, mass, leaf_capacity=8, multipole_order=2)
    pot_tree = g.tree_potentials(theta=0.0, threads=1)
    pot_direct = g.direct_potentials(threads=1)
    np.testing.assert_allclose(pot_tree, pot_direct, atol=1e-10)


def test_at_points_match_direct(particles):
    pos, mass = particles
    g = Gravity(pos, mass, leaf_capacity=8, multipole_order=2)
    rng = np.random.default_rng(7)
    targets = rng.uniform(-1.0, 1.0, (128, 3))
    np.testing.assert_allclose(
        g.tree_accelerations(positions=targets, theta=0.0, threads=1),
        g.direct_accelerations(positions=targets, threads=1),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        g.tree_potentials(positions=targets, theta=0.0, threads=1),
        g.direct_potentials(positions=targets, threads=1),
        atol=1e-10,
    )


def test_multipole_order_convergence_accel(particles):
    pos, mass = particles
    rng = np.random.default_rng(1)
    n = 800
    big_pos = rng.uniform(-1.0, 1.0, (n, 3))
    big_mass = 1.0 + rng.uniform(0.0, 1.0, n)
    g = Gravity(big_pos, big_mass, leaf_capacity=64)
    direct = g.direct_accelerations(threads=1)
    errs = []
    for order in (0, 3, 4, 5):
        acc = g.tree_accelerations(theta=0.7, leaf_capacity=64, multipole_order=order, threads=1)
        errs.append(np.sqrt(np.mean(np.sum((acc - direct) ** 2, axis=1))))
    assert errs == sorted(errs, reverse=True)  # non-increasing with order
    assert errs[-1] <= 0.8 * errs[0]  # order-5 at least 20% better than order-0


def test_multipole_order_convergence_potential(particles):
    pos, mass = particles
    rng = np.random.default_rng(1)
    n = 800
    big_pos = rng.uniform(-1.0, 1.0, (n, 3))
    big_mass = 1.0 + rng.uniform(0.0, 1.0, n)
    g = Gravity(big_pos, big_mass, leaf_capacity=64)
    direct = g.direct_potentials(threads=1)
    errs = []
    for order in (0, 2, 3, 4, 5):
        pot = g.tree_potentials(theta=0.7, leaf_capacity=64, multipole_order=order, threads=1)
        errs.append(np.sqrt(np.mean((pot - direct) ** 2)))
    assert errs == sorted(errs, reverse=True)
    assert errs[-1] <= 0.8 * errs[0]


def test_plummer_softening_matches_numpy(particles):
    pos, mass = particles
    eps = 0.05
    g = Gravity(pos, mass, softening=eps, kernel=KernelKind.Plummer, leaf_capacity=8, multipole_order=2)
    pot = g.direct_potentials(threads=1)
    # NumPy reference: phi_i = -sum_j m_j / sqrt(r_ij^2 + eps^2), excluding self.
    d = pos[:, None, :] - pos[None, :, :]
    r2 = np.sum(d * d, axis=-1)
    inv = 1.0 / np.sqrt(r2 + eps**2)
    np.fill_diagonal(inv, 0.0)
    ref = -np.sum(mass[None, :] * inv, axis=1)
    np.testing.assert_allclose(pot, ref, atol=1e-12)


def test_spline_softening_matches_numpy(particles):
    # Port of the W2 kernel. Use a softened (non-Newtonian) pair, u = r/h in [0.5, 1),
    # so this genuinely exercises the spline polynomial (differs from Newtonian).
    pos = np.array([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]])
    mass = np.array([1.0, 1.0])
    h = 0.2
    g = Gravity(pos, mass, softening=h, kernel=KernelKind.Spline)
    pot = g.direct_potentials(threads=1)
    u = 0.15 / h
    # W2(u) for 0.5 <= u < 1 (note the POSITIVE (1/15)(1/u) first term):
    W2 = (1.0/15.0)*(1.0/u) + (32.0/3.0)*u**2 - 16.0*u**3 + (48.0/5.0)*u**4 - (32.0/15.0)*u**5 - 16.0/5.0
    ref = W2 * (1.0 / h)
    np.testing.assert_allclose(pot, ref, atol=1e-12)

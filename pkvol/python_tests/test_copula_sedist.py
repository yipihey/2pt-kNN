"""Copula and SEDist correctness tests."""

import numpy as np
import pytest

import pkvol
from pkvol.copula import shifted_legendre, compute_copula_summary, coefficients_for_pk


def test_shifted_legendre_orthogonality():
    # Discrete orthogonality on a fine grid.
    x = np.linspace(0, 1, 4001)
    dx = x[1] - x[0]
    # Should integrate to 1 (P_n^2 has unit norm under our scaling).
    for n in range(0, 5):
        p = shifted_legendre(n, x)
        # Use trapezoidal weight for accuracy.
        w = np.full_like(x, dx)
        w[0] *= 0.5
        w[-1] *= 0.5
        norm = (p ** 2 * w).sum()
        assert abs(norm - 1.0) < 5e-3


def test_copula_independent_is_near_zero():
    # Build a P table whose distribution is separable -> empirical copula
    # equals U V exactly -> Legendre coefficients vanish.
    n_u, n_v = 11, 13
    u_marg = np.linspace(0.5, 1.0, n_u)
    v_marg = np.linspace(0.3, 1.1, n_v)
    p = np.outer(u_marg, v_marg)
    s = compute_copula_summary(p)
    for mn, c in s["coeffs"].items():
        assert abs(c) < 1e-10, f"mode {mn} coef {c}"


def test_copula_picks_up_dependence():
    # Strong positive correlation -> non-zero residual.
    rng = np.random.default_rng(0)
    n_u, n_v = 9, 9
    p = rng.uniform(0.1, 1.0, size=(n_u, n_v))
    # Inject diagonal: amplify the diagonal cells.
    for i in range(min(n_u, n_v)):
        p[i, i] *= 5.0
    s = compute_copula_summary(p)
    # The (1, 1) coefficient (dominant) should be non-trivial.
    assert abs(s["coeffs"][(1, 1)]) > 1e-3


def test_coefficients_for_pk_shape():
    arr = np.random.default_rng(42).uniform(0.0, 1.0, size=(4, 5, 6))
    out = coefficients_for_pk(arr)
    assert out["coeffs"].shape == (4, 6)


def test_compressed_cdf_1d_round_trip():
    samples = np.random.default_rng(1).normal(size=200)
    c = pkvol.CompressedCdf1D.from_samples(samples)
    # Evaluate at a known point: the median should map to ~0.5.
    med = np.median(samples)
    assert abs(c.evaluate(np.array([med]))[0] - 0.5) < 0.02
    d = pkvol.CompressedCdf1D.from_dict(c.to_dict())
    np.testing.assert_array_equal(c.x, d.x)
    np.testing.assert_array_equal(c.y, d.y)


def test_compressed_cdf_2d_bilinear():
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    F = np.outer(x, y)
    c = pkvol.CompressedCdf2D(x=x, y=y, F=F)
    qx = np.array([0.25, 0.5, 0.75])
    qy = np.array([0.25, 0.5, 0.75])
    got = c.evaluate(qx, qy)
    np.testing.assert_allclose(got, qx * qy, atol=1e-12)

"""Marks, randoms, and edge-cut tests."""

import numpy as np
import pytest

import pkvol


def test_marks_multiplicative_doubles_F():
    # Same catalog twice, with mark = 2.0 -> the weighted count doubles,
    # so K_q*2 >= k iff K_q >= k/2. With k=2 and integer K, k/2 = 1.
    rng = np.random.default_rng(7)
    N = 500
    ra = rng.uniform(0, 1.0, N)
    dec = rng.uniform(-0.5, 0.5, N)
    z = rng.uniform(0.0, 1.0, N)
    n_q = 30
    ra_q = rng.uniform(0.1, 0.9, n_q)
    dec_q = rng.uniform(-0.4, 0.4, n_q)

    theta_edges = np.array([0.05, 0.1])
    z_edges = np.array([0.3, 0.7])
    k_values = np.array([2.0])

    a = pkvol.measure_pk(ra, dec, z, None, ra_q, dec_q, theta_edges, z_edges, k_values, 0.2)
    marks = np.full(N, 2.0)
    b = pkvol.measure_pk(ra, dec, z, None, ra_q, dec_q, theta_edges, z_edges, k_values, 0.2,
                         marks=marks, mark_mode="multiplicative")
    # K' = 2 K. So 1[K' >= 2] == 1[K >= 1].
    k_values_b = np.array([1.0])
    a_k1 = pkvol.measure_pk(ra, dec, z, None, ra_q, dec_q, theta_edges, z_edges, k_values_b, 0.2)
    np.testing.assert_allclose(b["F"][0], a_k1["F"][0], atol=1e-12)


def test_marks_threshold_filters():
    rng = np.random.default_rng(11)
    N = 400
    ra = rng.uniform(0, 1.0, N)
    dec = rng.uniform(-0.5, 0.5, N)
    z = rng.uniform(0.0, 1.0, N)
    marks = rng.uniform(0, 1, N)
    n_q = 40
    ra_q = rng.uniform(0.1, 0.9, n_q)
    dec_q = rng.uniform(-0.4, 0.4, n_q)

    theta_edges = np.array([0.05, 0.1])
    z_edges = np.array([0.3, 0.7])
    k_values = np.array([1.0, 2.0])

    full = pkvol.measure_pk(ra, dec, z, None, ra_q, dec_q, theta_edges, z_edges, k_values, 0.2)
    sub = pkvol.measure_pk(ra, dec, z, None, ra_q, dec_q, theta_edges, z_edges, k_values, 0.2,
                           marks=marks, mark_mode="threshold", mark_cut=0.5)

    # threshold should reduce or keep counts -> F_k(threshold) <= F_k(full)
    assert np.all(sub["F"] <= full["F"] + 1e-12)


def test_quantile_mode():
    rng = np.random.default_rng(0)
    N = 200
    ra = rng.uniform(0, 1.0, N)
    dec = rng.uniform(-0.5, 0.5, N)
    z = rng.uniform(0.0, 1.0, N)
    marks = rng.uniform(0, 1, N)

    w_full = pkvol.apply_marks(None, marks, mode="quantile", quantile=(0.0, 1.0), n_galaxies=N)
    np.testing.assert_allclose(w_full, np.ones(N))
    w_top = pkvol.apply_marks(np.ones(N), marks, mode="quantile", quantile=(0.5, 1.0))
    assert (w_top > 0).sum() == N // 2 or (w_top > 0).sum() == N // 2 + 1


def test_subsample_randoms_reproducible():
    a = pkvol.subsample_randoms(1000, 50, seed=42)
    b = pkvol.subsample_randoms(1000, 50, seed=42)
    np.testing.assert_array_equal(a, b)
    assert a.size == 50
    assert len(set(a.tolist())) == 50
    c = pkvol.subsample_randoms(1000, 50, seed=43)
    assert not np.array_equal(a, c)


def test_lambda_per_query_matches_brute():
    rng = np.random.default_rng(2)
    N = 300
    ra = rng.uniform(0, 2 * np.pi, N)
    dec = np.arcsin(rng.uniform(-1, 1, N))
    z = rng.uniform(0.1, 0.9, N)

    n_q = 20
    ra_q = rng.uniform(0, 2 * np.pi, n_q)
    dec_q = np.arcsin(rng.uniform(-1, 1, n_q))

    theta_edges = np.array([0.1, 0.2, 0.3])
    z_edges = np.array([0.3, 0.5, 0.7])
    lam = pkvol.measure_lambda(
        ra, dec, z, ra_q, dec_q, theta_edges, z_edges, theta_max=0.4
    )

    # Brute force.
    s_dec = np.sin((dec[None, :] - dec_q[:, None]) / 2)
    s_ra = np.sin((ra[None, :] - ra_q[:, None]) / 2)
    h = s_dec ** 2 + np.cos(dec_q[:, None]) * np.cos(dec[None, :]) * s_ra ** 2
    theta = 2.0 * np.arcsin(np.minimum(np.sqrt(h), 1.0))

    n_int = z_edges.size - 1
    n_u = theta_edges.size
    expected = np.zeros((n_q, n_u, n_int))
    for q in range(n_q):
        for a, te in enumerate(theta_edges):
            sel = theta[q] <= te
            for ell in range(n_int):
                inn = sel & (z > z_edges[ell]) & (z <= z_edges[ell + 1])
                expected[q, a, ell] = inn.sum()
    np.testing.assert_allclose(lam, expected, atol=1e-12)


def test_apply_edge_cut():
    lam = np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
    expected = np.array([1.0, 2.0, 3.0])  # broadcastable
    mask = pkvol.apply_edge_cut(lam, expected, f_min=0.6)
    assert mask[0, 0] is np.bool_(True) or mask[0, 0]
    assert not mask[1, 0]

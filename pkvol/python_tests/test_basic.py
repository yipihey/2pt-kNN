"""Basic correctness tests for the Python API."""

import numpy as np
import pytest

import pkvol


@pytest.fixture(scope="module")
def small_random_catalog():
    rng = np.random.default_rng(123)
    N = 800
    ra = rng.uniform(0, 2 * np.pi, N)
    dec = np.arcsin(rng.uniform(-1, 1, N))
    z = rng.uniform(0.1, 0.9, N)
    w = rng.uniform(0.5, 1.5, N)

    n_q = 100
    ra_q = rng.uniform(0, 2 * np.pi, n_q)
    dec_q = np.arcsin(rng.uniform(-1, 1, n_q))
    qw = rng.uniform(0.8, 1.2, n_q)
    return ra, dec, z, w, ra_q, dec_q, qw


def haversine(ra1, dec1, ra2, dec2):
    s_dec = np.sin((dec2 - dec1) / 2.0)
    s_ra = np.sin((ra2 - ra1) / 2.0)
    h = s_dec ** 2 + np.cos(dec1) * np.cos(dec2) * s_ra ** 2
    return 2.0 * np.arcsin(np.minimum(np.sqrt(h), 1.0))


def brute_pk(ra, dec, z, w, ra_q, dec_q, qw, theta_edges, z_edges, k_values, theta_max):
    n_q = ra_q.size
    n_u = theta_edges.size
    n_z = z_edges.size
    n_int = n_z - 1
    n_k = k_values.size
    f_sum = np.zeros((n_k + 1, n_u, n_int))
    k_ext = np.append(k_values, np.maximum(np.floor(k_values[-1] + 1), k_values[-1] + 1))
    for q in range(n_q):
        theta = haversine(ra, dec, ra_q[q], dec_q[q])
        mask = theta <= theta_max
        loc_theta = theta[mask]
        loc_z = z[mask]
        loc_w = w[mask]
        K = np.zeros((n_u, n_z))
        for a, u in enumerate(theta_edges):
            sel_a = loc_theta <= u
            for b, ze in enumerate(z_edges):
                sel = sel_a & (loc_z <= ze)
                K[a, b] = loc_w[sel].sum()
        for a in range(n_u):
            for ell in range(n_int):
                shell = K[a, ell + 1] - K[a, ell]
                for j, kj in enumerate(k_ext):
                    if shell >= kj:
                        f_sum[j, a, ell] += qw[q]
    inv = 1.0 / qw.sum()
    F = f_sum[:n_k] * inv
    P = F - f_sum[1:] * inv
    return F, P


def test_brute_force_match(small_random_catalog):
    ra, dec, z, w, ra_q, dec_q, qw = small_random_catalog
    theta_edges = np.array([0.1, 0.15, 0.2, 0.3])
    z_edges = np.array([0.2, 0.4, 0.6, 0.8])
    k_values = np.array([1.0, 2.0, 5.0])
    theta_max = 0.4

    out = pkvol.measure_pk(
        ra, dec, z, w, ra_q, dec_q, theta_edges, z_edges, k_values, theta_max,
        query_weights=qw,
    )
    F_b, P_b = brute_pk(ra, dec, z, w, ra_q, dec_q, qw, theta_edges, z_edges, k_values, theta_max)
    np.testing.assert_allclose(out["F"], F_b, atol=1e-12)
    np.testing.assert_allclose(out["P"], P_b, atol=1e-12)


def test_backends_agree(small_random_catalog):
    ra, dec, z, w, ra_q, dec_q, qw = small_random_catalog
    theta_edges = np.array([0.1, 0.2, 0.3])
    z_edges = np.array([0.2, 0.5, 0.8])
    k_values = np.array([1.0, 3.0])
    theta_max = 0.5
    a = pkvol.measure_pk(ra, dec, z, w, ra_q, dec_q, theta_edges, z_edges, k_values, theta_max,
                         query_weights=qw, backend="ecdf")
    b = pkvol.measure_pk(ra, dec, z, w, ra_q, dec_q, theta_edges, z_edges, k_values, theta_max,
                         query_weights=qw, backend="histogram")
    np.testing.assert_allclose(a["F"], b["F"], atol=1e-12)
    np.testing.assert_allclose(a["P"], b["P"], atol=1e-12)


def test_angular_variables_consistent(small_random_catalog):
    """Theta vs theta^2 should give identical results when edges map consistently."""
    ra, dec, z, w, ra_q, dec_q, _ = small_random_catalog
    theta_edges = np.array([0.05, 0.1, 0.2, 0.3])
    z_edges = np.array([0.3, 0.7])
    k_values = np.array([1.0])
    theta_max = 0.4
    a = pkvol.measure_pk(ra, dec, z, w, ra_q, dec_q, theta_edges, z_edges, k_values, theta_max,
                         angular_variable="theta")
    b = pkvol.measure_pk(ra, dec, z, w, ra_q, dec_q, theta_edges, z_edges, k_values, theta_max,
                         angular_variable="theta2")
    c = pkvol.measure_pk(ra, dec, z, w, ra_q, dec_q, theta_edges, z_edges, k_values, theta_max,
                         angular_variable="omega")
    np.testing.assert_allclose(a["F"], b["F"], atol=1e-12)
    np.testing.assert_allclose(a["F"], c["F"], atol=1e-12)


def test_haversine_periodicity():
    # Through bindings.
    h = pkvol._pkvol.haversine_py(0.0, 0.0, 2 * np.pi, 0.0)
    assert abs(h) < 1e-9


def test_query_radius_via_python():
    rng = np.random.default_rng(0)
    N = 200
    ra = rng.uniform(0, 2 * np.pi, N)
    dec = np.arcsin(rng.uniform(-1, 1, N))
    out = pkvol._pkvol.query_radius_py(ra, dec, 0.0, 0.0, 0.5)
    out = np.sort(np.asarray(out))
    brute = np.where(haversine(ra, dec, 0.0, 0.0) <= 0.5)[0]
    np.testing.assert_array_equal(out, brute)


def test_self_exclusion():
    # Two galaxies; query coincides with the first galaxy.
    ra_g = np.array([0.0, 0.001])
    dec_g = np.array([0.0, 0.0])
    z_g = np.array([0.5, 0.5])
    w_g = np.array([1.0, 1.0])
    theta_edges = np.array([0.01])
    z_edges = np.array([0.4, 0.6])
    k_values = np.array([1.0])
    out_no = pkvol.measure_pk(ra_g, dec_g, z_g, w_g, np.array([0.0]), np.array([0.0]),
                              theta_edges, z_edges, k_values, theta_max=0.02)
    out_yes = pkvol.measure_pk(ra_g, dec_g, z_g, w_g, np.array([0.0]), np.array([0.0]),
                               theta_edges, z_edges, k_values, theta_max=0.02,
                               exclude_self=True)
    # Without self-exclusion: K_shell = 2 -> F_1 = 1.
    # With self-exclusion: K_shell = 1 -> F_1 = 1 (still >= 1).
    assert out_no["F"][0, 0, 0] == 1.0
    assert out_yes["F"][0, 0, 0] == 1.0
    # But for k=2 we should differ.
    k_values2 = np.array([1.0, 2.0])
    out_no = pkvol.measure_pk(ra_g, dec_g, z_g, w_g, np.array([0.0]), np.array([0.0]),
                              theta_edges, z_edges, k_values2, theta_max=0.02)
    out_yes = pkvol.measure_pk(ra_g, dec_g, z_g, w_g, np.array([0.0]), np.array([0.0]),
                               theta_edges, z_edges, k_values2, theta_max=0.02,
                               exclude_self=True)
    assert out_no["F"][1, 0, 0] == 1.0   # K_shell = 2 >= 2
    assert out_yes["F"][1, 0, 0] == 0.0  # K_shell = 1, not >= 2


def test_p_telescopes():
    rng = np.random.default_rng(7)
    N = 300
    ra = rng.uniform(0, 1.0, N)
    dec = rng.uniform(-0.5, 0.5, N)
    z = rng.uniform(0.0, 1.0, N)
    n_q = 50
    ra_q = rng.uniform(0.1, 0.9, n_q)
    dec_q = rng.uniform(-0.4, 0.4, n_q)
    theta_edges = np.array([0.05, 0.1])
    z_edges = np.array([0.3, 0.7])
    k_values = np.arange(1, 8, dtype=float)
    out = pkvol.measure_pk(ra, dec, z, None, ra_q, dec_q, theta_edges, z_edges, k_values, 0.2)
    # P_k = F_k - F_{k+1}; sum(P_k) = F_1 - F_{k_max+1}.
    # So cumulative sum of P from top to bottom equals F.
    cum = np.cumsum(out["P"][::-1], axis=0)[::-1]
    # cum[j] = F_j - F_{k_max + 1}
    diff = out["F"] - cum
    assert np.all(diff >= -1e-12)
    # All entries identical: it's F_{k_max+1}.
    F_kmax_plus_1 = diff[0]
    np.testing.assert_allclose(
        diff, np.broadcast_to(F_kmax_plus_1, diff.shape), atol=1e-12
    )

"""Tests for the conditional, median-centered P_k pipeline (pkvol.conditional)."""

from __future__ import annotations

import numpy as np
import pytest

import pkvol


@pytest.fixture(scope="module")
def small_uniform_catalogs():
    """A small uniform catalog and matching random over a tangent-plane patch."""
    rng = np.random.default_rng(42)
    L = 0.5
    n_d = 600
    n_r = 4 * n_d
    ra_d = rng.uniform(0, L, n_d)
    dec_d = rng.uniform(-L / 2, L / 2, n_d)
    z_d = rng.uniform(0.1, 0.9, n_d)
    ra_r = rng.uniform(0, L, n_r)
    dec_r = rng.uniform(-L / 2, L / 2, n_r)
    z_r = rng.uniform(0.1, 0.9, n_r)
    return dict(
        ra_d=ra_d, dec_d=dec_d, z_d=z_d,
        ra_r=ra_r, dec_r=dec_r, z_r=z_r,
        L=L,
    )


def _common_grids():
    theta_edges = np.array([0.005, 0.01, 0.02, 0.04])      # radians
    z_edges = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    s_tilde_edges = np.linspace(-1.0, 1.0, 11)
    eta_tilde_edges = np.linspace(-1.0, 1.0, 9)
    z_query_edges = np.array([0.1, 0.5, 0.9])
    return theta_edges, z_edges, s_tilde_edges, eta_tilde_edges, z_query_edges


def test_shapes_R(small_uniform_catalogs):
    cats = small_uniform_catalogs
    theta_edges, z_edges, s_e, e_e, zq_e = _common_grids()
    out = pkvol.measure_pk_R(
        ra_gal=cats["ra_d"], dec_gal=cats["dec_d"], z_gal=cats["z_d"],
        ra_random=cats["ra_r"], dec_random=cats["dec_r"], z_random=cats["z_r"],
        theta_edges=theta_edges, z_edges=z_edges,
        s_tilde_edges=s_e, eta_tilde_edges=e_e,
        z_query_edges=zq_e,
        n_query_subsample=300,
        k_max=5,
    )
    n_k, n_s, n_e, n_zq = 5, s_e.size - 1, e_e.size - 1, zq_e.size - 1
    assert out["P"].shape == (n_k, n_s, n_e, n_zq)
    assert out["F"].shape == (n_k, n_s, n_e, n_zq)
    assert out["denominator"].shape == (n_k, n_s, n_e, n_zq)
    assert out["s_med"].shape == (n_k, n_zq)
    assert out["eta_med"].shape == (n_k, n_zq)
    assert out["k_values"].tolist() == [1, 2, 3, 4, 5]
    assert out["query_type"] == "R"
    assert "diagnostics" in out


def test_shapes_D(small_uniform_catalogs):
    cats = small_uniform_catalogs
    theta_edges, z_edges, s_e, e_e, zq_e = _common_grids()
    out = pkvol.measure_pk_D(
        ra_gal=cats["ra_d"], dec_gal=cats["dec_d"], z_gal=cats["z_d"],
        ra_random=cats["ra_r"], dec_random=cats["dec_r"], z_random=cats["z_r"],
        theta_edges=theta_edges, z_edges=z_edges,
        s_tilde_edges=s_e, eta_tilde_edges=e_e,
        z_query_edges=zq_e,
        k_max=4,
    )
    assert out["query_type"] == "D"
    assert out["P"].shape[0] == 4
    # alpha = sum_w_data / sum_w_random
    n_d = cats["ra_d"].size
    n_r = cats["ra_r"].size
    np.testing.assert_allclose(out["alpha"], n_d / n_r)


def test_alpha_with_random_subsample(small_uniform_catalogs):
    """alpha should reflect the *full* random catalog weight, not a subsample."""
    cats = small_uniform_catalogs
    theta_edges, z_edges, s_e, e_e, zq_e = _common_grids()
    # Use only half the randoms — alpha changes accordingly.
    n_r = cats["ra_r"].size
    half = n_r // 2
    out = pkvol.measure_pk_R(
        ra_gal=cats["ra_d"], dec_gal=cats["dec_d"], z_gal=cats["z_d"],
        ra_random=cats["ra_r"][:half], dec_random=cats["dec_r"][:half], z_random=cats["z_r"][:half],
        theta_edges=theta_edges, z_edges=z_edges,
        s_tilde_edges=s_e, eta_tilde_edges=e_e,
        z_query_edges=zq_e,
        n_query_subsample=200,
        k_max=3,
    )
    np.testing.assert_allclose(out["alpha"], cats["ra_d"].size / half)


def test_marginal_F_monotone_in_k(small_uniform_catalogs):
    """Marginalised over (s_tilde, eta_tilde) — i.e. the unconditional F_k(z) —
    is monotone non-increasing in k. (Pointwise F_k(s_tilde, eta_tilde) need
    NOT be monotone, because each k uses k-specific medians and therefore
    indexes different cells.)
    """
    cats = small_uniform_catalogs
    theta_edges, z_edges, s_e, e_e, zq_e = _common_grids()
    out = pkvol.measure_pk_R(
        ra_gal=cats["ra_d"], dec_gal=cats["dec_d"], z_gal=cats["z_d"],
        ra_random=cats["ra_r"], dec_random=cats["dec_r"], z_random=cats["z_r"],
        theta_edges=theta_edges, z_edges=z_edges,
        s_tilde_edges=np.linspace(-3.0, 3.0, 31),       # wide enough to catch all cells
        eta_tilde_edges=np.linspace(-3.0, 3.0, 31),
        z_query_edges=zq_e,
        n_query_subsample=400,
        k_max=5,
    )
    F_num = out["F"] * out["denominator"]
    F_marg = np.nansum(F_num, axis=(1, 2)) / np.maximum(np.nansum(out["denominator"], axis=(1, 2)), 1e-30)
    for zb in range(F_marg.shape[1]):
        diffs = np.diff(F_marg[:, zb])
        assert np.all(diffs <= 1e-9), f"marginal F not monotone in k at z_bin {zb}: {F_marg[:, zb]}"


def test_self_exclusion_changes_N(small_uniform_catalogs):
    """D-style with exclude_self=True should give different (smaller) F_k than without."""
    cats = small_uniform_catalogs
    theta_edges, z_edges, s_e, e_e, zq_e = _common_grids()
    common_kwargs = dict(
        ra_gal=cats["ra_d"], dec_gal=cats["dec_d"], z_gal=cats["z_d"],
        ra_random=cats["ra_r"], dec_random=cats["dec_r"], z_random=cats["z_r"],
        theta_edges=theta_edges, z_edges=z_edges,
        s_tilde_edges=s_e, eta_tilde_edges=e_e,
        z_query_edges=zq_e,
        k_max=4,
    )
    out_excl = pkvol.measure_pk_D(exclude_self=True, **common_kwargs)
    out_inc = pkvol.measure_pk_D(exclude_self=False, **common_kwargs)
    # Different in F at small k because excluded self contributed +1.
    diff = np.nansum(np.abs(out_excl["F"] - out_inc["F"]))
    assert diff > 0.0, "Excluding self should change F_k for D-style queries."


def test_alpha_with_marks():
    """alpha should equal (sum mark-applied data weights) / (sum mark-applied random weights)."""
    rng = np.random.default_rng(7)
    L = 0.4
    n_d, n_r = 200, 800
    ra_d = rng.uniform(0, L, n_d); dec_d = rng.uniform(-L/2, L/2, n_d); z_d = rng.uniform(0.1, 0.9, n_d)
    ra_r = rng.uniform(0, L, n_r); dec_r = rng.uniform(-L/2, L/2, n_r); z_r = rng.uniform(0.1, 0.9, n_r)
    marks_d = rng.uniform(0.0, 2.0, n_d)
    marks_r = rng.uniform(0.0, 2.0, n_r)
    theta_edges, z_edges, s_e, e_e, zq_e = _common_grids()
    out = pkvol.measure_pk_R(
        ra_gal=ra_d, dec_gal=dec_d, z_gal=z_d,
        ra_random=ra_r, dec_random=dec_r, z_random=z_r,
        marks_gal=marks_d, mark_mode_gal="multiplicative",
        marks_random=marks_r, mark_mode_random="multiplicative",
        theta_edges=theta_edges, z_edges=z_edges,
        s_tilde_edges=s_e, eta_tilde_edges=e_e,
        z_query_edges=zq_e,
        n_query_subsample=200,
        k_max=2,
    )
    expected = float(marks_d.sum() / marks_r.sum())
    np.testing.assert_allclose(out["alpha"], expected, rtol=1e-12)


def test_uniform_smed_increases_with_k(small_uniform_catalogs):
    """For a uniform catalog with random density matched, s_med(k) should be a
    monotone-non-decreasing function of k (larger k requires larger Lambda).
    """
    cats = small_uniform_catalogs
    theta_edges, z_edges, s_e, e_e, zq_e = _common_grids()
    out = pkvol.measure_pk_R(
        ra_gal=cats["ra_d"], dec_gal=cats["dec_d"], z_gal=cats["z_d"],
        ra_random=cats["ra_r"], dec_random=cats["dec_r"], z_random=cats["z_r"],
        theta_edges=theta_edges, z_edges=z_edges,
        s_tilde_edges=s_e, eta_tilde_edges=e_e,
        z_query_edges=zq_e,
        n_query_subsample=600,
        k_max=4,
    )
    s_med = out["s_med"]                           # (k, z_bin)
    # Take a z_bin with finite medians.
    for zb in range(s_med.shape[1]):
        col = s_med[:, zb]
        if np.isfinite(col).all():
            diffs = np.diff(col)
            assert np.all(diffs >= -1e-9), f"s_med not non-decreasing at z_bin {zb}: {col}"
            return
    pytest.skip("no z_bin had all finite medians (catalog too sparse)")


def test_centering_recovers_absolute(small_uniform_catalogs):
    """Adding s_med to s_tilde edges should recover absolute log10(Lambda) range."""
    cats = small_uniform_catalogs
    theta_edges, z_edges, s_e, e_e, zq_e = _common_grids()
    out = pkvol.measure_pk_R(
        ra_gal=cats["ra_d"], dec_gal=cats["dec_d"], z_gal=cats["z_d"],
        ra_random=cats["ra_r"], dec_random=cats["dec_r"], z_random=cats["z_r"],
        theta_edges=theta_edges, z_edges=z_edges,
        s_tilde_edges=s_e, eta_tilde_edges=e_e,
        z_query_edges=zq_e,
        n_query_subsample=500,
        k_max=2,
    )
    # Reconstruct absolute s edges for k=0 (k_value=1), z_bin 0.
    s_abs = s_e + out["s_med"][0, 0]
    assert np.isfinite(s_abs).all()
    # The midpoint should be a plausible log10 of expected count for the smallest aperture.
    assert s_abs.min() > -10.0 and s_abs.max() < 10.0


def test_default_F_ref():
    z = np.array([0.1, 0.5, 1.0])
    expected = z * np.sqrt(1.0 + z)
    np.testing.assert_allclose(pkvol.default_F_ref(z), expected)


def test_log_base_e_vs_10(small_uniform_catalogs):
    """Choosing log_base='e' rescales s_med by ln(10); centered shape should match."""
    cats = small_uniform_catalogs
    theta_edges, z_edges, s_e, e_e, zq_e = _common_grids()
    common = dict(
        ra_gal=cats["ra_d"], dec_gal=cats["dec_d"], z_gal=cats["z_d"],
        ra_random=cats["ra_r"], dec_random=cats["dec_r"], z_random=cats["z_r"],
        theta_edges=theta_edges, z_edges=z_edges,
        z_query_edges=zq_e,
        n_query_subsample=300,
        k_max=2,
    )
    out10 = pkvol.measure_pk_R(s_tilde_edges=s_e, eta_tilde_edges=e_e, log_base="10", **common)
    out_e = pkvol.measure_pk_R(
        s_tilde_edges=s_e * np.log(10.0), eta_tilde_edges=e_e * np.log(10.0),
        log_base="e", **common,
    )
    # Medians scale by log(10) when switching from log10 to ln.
    np.testing.assert_allclose(
        out_e["s_med"], out10["s_med"] * np.log(10.0),
        rtol=1e-9, atol=1e-12,
    )
    np.testing.assert_allclose(
        out_e["eta_med"], out10["eta_med"] * np.log(10.0),
        rtol=1e-9, atol=1e-12,
    )


def test_sparse_k_values_matches_contiguous(small_uniform_catalogs):
    """Sparse k_values e.g. [1, 3, 6] must produce F, P, medians identical to the
    same k's sliced from a contiguous k_max=7 run.
    """
    cats = small_uniform_catalogs
    theta_edges, z_edges, s_e, e_e, zq_e = _common_grids()
    common = dict(
        ra_gal=cats["ra_d"], dec_gal=cats["dec_d"], z_gal=cats["z_d"],
        ra_random=cats["ra_r"], dec_random=cats["dec_r"], z_random=cats["z_r"],
        theta_edges=theta_edges, z_edges=z_edges,
        s_tilde_edges=s_e, eta_tilde_edges=e_e,
        z_query_edges=zq_e,
        n_query_subsample=400,
        subsample_seed=2026,
    )
    out_full = pkvol.measure_pk_R(k_max=7, **common)
    out_sparse = pkvol.measure_pk_R(k_values=[1, 3, 6], **common)
    pick = np.array([0, 2, 5])    # 1->0, 3->2, 6->5 in the contiguous array
    np.testing.assert_array_equal(out_sparse["k_values"], np.array([1, 3, 6]))
    np.testing.assert_allclose(out_sparse["F"], out_full["F"][pick], equal_nan=True)
    np.testing.assert_allclose(out_sparse["P"], out_full["P"][pick], equal_nan=True)
    np.testing.assert_allclose(out_sparse["s_med"], out_full["s_med"][pick], equal_nan=True)
    np.testing.assert_allclose(out_sparse["eta_med"], out_full["eta_med"][pick], equal_nan=True)
    np.testing.assert_allclose(
        out_sparse["denominator"], out_full["denominator"][pick], equal_nan=True
    )


def test_sparse_k_log_spaced(small_uniform_catalogs):
    """Log-spaced k_values like [1, 3, 10] should run and shape-match user input."""
    cats = small_uniform_catalogs
    theta_edges, z_edges, s_e, e_e, zq_e = _common_grids()
    out = pkvol.measure_pk_R(
        ra_gal=cats["ra_d"], dec_gal=cats["dec_d"], z_gal=cats["z_d"],
        ra_random=cats["ra_r"], dec_random=cats["dec_r"], z_random=cats["z_r"],
        theta_edges=theta_edges, z_edges=z_edges,
        s_tilde_edges=s_e, eta_tilde_edges=e_e,
        z_query_edges=zq_e,
        n_query_subsample=300,
        k_values=[1, 3, 10],
    )
    assert out["P"].shape[0] == 3
    assert out["k_values"].tolist() == [1, 3, 10]
    # F_next must be present and same shape as F.
    assert out["F_next"].shape == out["F"].shape
    assert out["denominator_next"].shape == out["denominator"].shape
    # P = F - F_next pointwise.
    np.testing.assert_allclose(
        out["P"], out["F"] - out["F_next"], equal_nan=True
    )


def test_sparse_k_unsorted_dedup(small_uniform_catalogs):
    """Unsorted/duplicate k_values are sorted and de-duplicated."""
    cats = small_uniform_catalogs
    theta_edges, z_edges, s_e, e_e, zq_e = _common_grids()
    out = pkvol.measure_pk_R(
        ra_gal=cats["ra_d"], dec_gal=cats["dec_d"], z_gal=cats["z_d"],
        ra_random=cats["ra_r"], dec_random=cats["dec_r"], z_random=cats["z_r"],
        theta_edges=theta_edges, z_edges=z_edges,
        s_tilde_edges=s_e, eta_tilde_edges=e_e,
        z_query_edges=zq_e,
        n_query_subsample=200,
        k_values=[5, 1, 3, 1, 5],
    )
    assert out["k_values"].tolist() == [1, 3, 5]


def test_sparse_k_invalid():
    """k_values must be positive integers."""
    rng = np.random.default_rng(3)
    L = 0.3
    n = 50
    ra = rng.uniform(0, L, n); dec = rng.uniform(-L/2, L/2, n); z = rng.uniform(0.1, 0.9, n)
    theta_edges = np.array([0.01, 0.02])
    z_edges = np.array([0.1, 0.5, 0.9])
    s_e = np.linspace(-1, 1, 5); e_e = np.linspace(-1, 1, 5); zq_e = np.array([0.1, 0.9])
    common = dict(
        ra_gal=ra, dec_gal=dec, z_gal=z,
        ra_random=ra, dec_random=dec, z_random=z,
        theta_edges=theta_edges, z_edges=z_edges,
        s_tilde_edges=s_e, eta_tilde_edges=e_e,
        z_query_edges=zq_e,
        n_query_subsample=20,
    )
    with pytest.raises(ValueError):
        pkvol.measure_pk_R(k_values=[0, 1], **common)
    with pytest.raises(ValueError):
        pkvol.measure_pk_R(k_values=[-1, 2], **common)
    with pytest.raises(ValueError):
        pkvol.measure_pk_R(k_values=[], **common)


def test_diagnostics_keys(small_uniform_catalogs):
    cats = small_uniform_catalogs
    theta_edges, z_edges, s_e, e_e, zq_e = _common_grids()
    out = pkvol.measure_pk_R(
        ra_gal=cats["ra_d"], dec_gal=cats["dec_d"], z_gal=cats["z_d"],
        ra_random=cats["ra_r"], dec_random=cats["dec_r"], z_random=cats["z_r"],
        theta_edges=theta_edges, z_edges=z_edges,
        s_tilde_edges=s_e, eta_tilde_edges=e_e,
        z_query_edges=zq_e,
        n_query_subsample=200,
        k_max=3,
    )
    d = out["diagnostics"]
    for key in ("fraction_used", "mean_lambda", "mean_N", "mean_eta", "n_query", "n_data", "n_random"):
        assert key in d
    assert d["fraction_used"].shape == (3, zq_e.size - 1)

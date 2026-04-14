#![cfg(feature = "theory")]

use twopoint::theory::*;

#[test]
fn cumulative_profile_converges_to_poisson_at_large_r() {
    let cosmo = Cosmology::planck2018();
    let mut ws = Workspace::new(2000);
    ws.update_cosmology(&cosmo);
    let ip = IntegrationParams::fast();
    let nbar = 1e-3; // galaxies per (Mpc/h)^3

    // At very large R, xi_bar -> 0, so N(<R) -> nbar * V(R)
    let r_large: Vec<f64> = vec![200.0, 300.0, 400.0];
    let profile = theory_cumulative_profile(&r_large, nbar, &ws, &ip);

    for (i, &r) in r_large.iter().enumerate() {
        let v = 4.0 / 3.0 * std::f64::consts::PI * r * r * r;
        let poisson = nbar * v;
        let ratio = profile.counts[i] / poisson;
        // At R > 200 Mpc/h, xi_bar should be < 1%, so ratio ~ 1.0
        assert!(
            (ratio - 1.0).abs() < 0.02,
            "At R={} Mpc/h, ratio={:.4} (expected ~1.0)",
            r, ratio
        );
    }
}

#[test]
fn predicted_cdf_is_monotonic_and_bounded() {
    let cosmo = Cosmology::planck2018();
    let mut ws = Workspace::new(2000);
    ws.update_cosmology(&cosmo);
    let ip = IntegrationParams::fast();
    let nbar = 1e-3;

    let r_grid: Vec<f64> = (1..=50).map(|i| i as f64 * 2.0).collect();

    for k in [1, 4, 16] {
        let mut prev = 0.0_f64;
        for &r in &r_grid {
            let cdf = predicted_cdf(k, r, nbar, &ws, &ip);
            assert!(cdf >= 0.0 && cdf <= 1.0, "CDF out of [0,1]: k={}, r={}, cdf={}", k, r, cdf);
            assert!(cdf >= prev - 1e-12, "CDF not monotonic: k={}, r={}", k, r);
            prev = cdf;
        }
    }
}

#[test]
fn theory_at_knn_scale_returns_sensible_values() {
    let cosmo = Cosmology::planck2018();
    let nbar = 1e-3;

    for k in [1, 8, 64] {
        let result = theory_at_knn_scale(k, nbar, &cosmo);
        assert!(result.r > 0.0);
        assert!(result.sigma2_lin > 0.0);
        assert!(result.sigma2_zel > 0.0);
        assert!(result.sigma2_j > 0.0);
        // sigma2_j should be close to sigma2_zel (corrections are small at large R)
        let ratio = result.sigma2_j / result.sigma2_zel;
        assert!(ratio > 0.5 && ratio < 2.0,
            "sigma2_j/sigma2_zel = {:.3} at k={} (R={:.1} Mpc/h)",
            ratio, k, result.r);
    }
}

#[test]
fn sigma2j_regression_planck2018_r10() {
    // Reference value for Planck 2018 at R=10 Mpc/h
    let cosmo = Cosmology::planck2018();
    let result = sigma2_j_detailed(&cosmo, 10.0, 2, 0.0, 0.0);

    // sigma2_lin at R=10 should be around 0.56 for Planck 2018
    assert!(
        (result.sigma2_lin - 0.56).abs() < 0.1,
        "sigma2_lin(R=10) = {:.4}, expected ~0.56",
        result.sigma2_lin
    );
    // xi_bar should be positive at R=10
    assert!(result.xi_bar > 0.0, "xi_bar(R=10) should be positive");
}

// ── kNN CDF prediction tests ────────────────────────────────────────────

#[test]
fn rknn_cdf_physical_monotonic_and_bounded() {
    let cosmo = Cosmology::planck2018();
    let mut ws = Workspace::new(2000);
    ws.update_cosmology(&cosmo);
    let ip = IntegrationParams::fast();
    let nbar = 1e-3;
    let params = KnnCdfParams::default();

    let r_values: Vec<f64> = (1..=40).map(|i| i as f64 * 1.5).collect();

    for k in [1, 4, 16] {
        let pred = rknn_cdf_physical(k, nbar, &r_values, &ws, &ip, &params);
        assert_eq!(pred.k, k);
        assert!(pred.r_lag > 0.0);
        assert!(pred.sigma_lin > 0.0);

        let mut prev = 0.0_f64;
        for (i, &c) in pred.cdf.iter().enumerate() {
            assert!(c >= 0.0 && c <= 1.0,
                "CDF out of [0,1]: k={}, r={:.1}, cdf={:.4}", k, r_values[i], c);
            assert!(c >= prev - 1e-10,
                "CDF not monotonic: k={}, r={:.1}", k, r_values[i]);
            prev = c;
        }
    }
}

#[test]
fn predict_rknn_cdfs_matches_knn_cdfs_format() {
    let cosmo = Cosmology::planck2018();
    let mut ws = Workspace::new(2000);
    ws.update_cosmology(&cosmo);
    let ip = IntegrationParams::fast();
    let nbar = 1e-3;
    let params = KnnCdfParams::default();

    let k_values = vec![1, 4, 16];
    let r_values: Vec<f64> = (1..=20).map(|i| i as f64 * 2.0).collect();

    let cdfs = predict_rknn_cdfs(&k_values, nbar, &r_values, &ws, &ip, &params);

    // Check structure matches KnnCdfs
    assert_eq!(cdfs.k_values, k_values);
    assert_eq!(cdfs.r_values, r_values);
    assert_eq!(cdfs.cdf_values.len(), k_values.len());
    for cdf_row in &cdfs.cdf_values {
        assert_eq!(cdf_row.len(), r_values.len());
    }
}

#[test]
fn dknn_biased_differs_from_unbiased() {
    let cosmo = Cosmology::planck2018();
    let mut ws = Workspace::new(2000);
    ws.update_cosmology(&cosmo);
    let ip = IntegrationParams::fast();
    let nbar = 1e-3;
    let params = KnnCdfParams::default();

    let k = 8;
    let r_values: Vec<f64> = (1..=20).map(|i| i as f64 * 2.0).collect();

    let unbiased = rknn_cdf_physical(k, nbar, &r_values, &ws, &ip, &params);
    let biased = dknn_cdf_biased(k, nbar, 1.5, &r_values, &ws, &ip, &params);

    // Biased CDF should differ from unbiased
    let max_diff: f64 = unbiased.cdf.iter().zip(&biased.cdf)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(max_diff > 0.01,
        "Biased and unbiased CDFs should differ significantly, max_diff={:.4}", max_diff);
}

#[test]
fn doroshkevich_predictor_vs_erlang() {
    // The Doroshkevich predictor should give different results from Erlang
    // at cosmological densities where clustering matters
    let cosmo = Cosmology::planck2018();
    let mut ws = Workspace::new(2000);
    ws.update_cosmology(&cosmo);
    let ip = IntegrationParams::fast();
    let nbar = 1e-3;
    let params = KnnCdfParams::default();

    let k = 4;
    let r_lag = knn_to_radius(k, nbar);

    // Test at r near the Lagrangian radius where Doroshkevich effects are visible
    let r_test = r_lag * 1.0;

    let doroshkevich_cdf_val = rknn_cdf_physical(k, nbar, &[r_test], &ws, &ip, &params).cdf[0];
    let erlang_cdf_val = twopoint::diagnostics::erlang_cdf(k, r_test, nbar);

    // They should differ because Doroshkevich includes the full J distribution
    // while Erlang assumes pure Poisson
    let diff = (doroshkevich_cdf_val - erlang_cdf_val).abs();
    assert!(diff > 0.001,
        "Doroshkevich CDF ({:.4}) should differ from Erlang ({:.4}) at R_L, diff={:.4}",
        doroshkevich_cdf_val, erlang_cdf_val, diff);
}

//! Bridge between perturbation theory predictions (pt crate) and kNN measurements.
//!
//! This module connects the Lagrangian PT predictions (σ²_J, ξ̄, S₃)
//! to the kNN estimator's data types (CumulativeProfile, KnnResiduals).

pub use pt::{
    Cosmology, Workspace, Sigma2JDetailed, XibarJDetailed,
    knn_to_radius, mass_to_radius, radius_to_mass, radius_to_k_eff,
    sigma2_j_detailed, sigma2_j_at_masses, sigma2_j_plot,
    xibar_j_full, xibar_j_plot, xibar_j_full_rsd, xibar_j_plot_rsd,
    xibar_j_full_bias, xibar_j_plot_bias,
    xibar_j_full_rsd_bias, xibar_j_plot_rsd_bias,
};
pub use pt::integrals::{IntegrationParams, RsdParams, BispecIntegrals};
pub use pt::knn_cdf::{
    KnnCdfParams, KnnCdfPrediction, KnnCdfPoissonCorrected,
    rknn_cdf_zel, rknn_cdf_corrected, rknn_cdf_physical,
    rknn_cdf_with_poisson,
    dknn_cdf_biased,
    multi_k_rknn_cdfs, multi_k_dknn_cdfs,
    TiltedKnnParams, TiltedKnnPrediction, PairType,
    rknn_cdf_tilted, dknn_cdf_tilted, knn_cdf_tilted, knn_cdf_tilted_bias,
    poisson_correct_cdf,
};
pub use pt::doroshkevich::{
    doroshkevich_cdf, doroshkevich_cdf_biased,
    doroshkevich_pdf, doroshkevich_pdf_biased,
    find_sigma_eff,
    TiltedPass, doroshkevich_tilted_pass, doroshkevich_tilted_moments,
    fit_tilt_to_moments, cdf_from_pdf, uniform_j_grid,
    UnbiasedCrossMoments, doroshkevich_unbiased_cross_moments,
    BiasParams, BiasedPass, doroshkevich_biased_polynomial_pass,
    xibar_tree_bias_expanded,
};

use pt::integrals::xi_bar_ws;
use crate::estimator::{CumulativeProfile, KnnCdfs, volume};
use crate::diagnostics::erlang_cdf;

/// Build a theory-predicted cumulative neighbor count profile from ξ̄(R).
///
/// N_theory(<r) = n̄ · V(r) · (1 + ξ̄(r))
///
/// This feeds directly into `RrMode::Empirical(profile)` to replace
/// stochastic RR pair counts with a perturbation theory prediction.
pub fn theory_cumulative_profile(
    r_grid: &[f64],
    nbar: f64,
    ws: &Workspace,
    ip: &IntegrationParams,
) -> CumulativeProfile {
    let radii = r_grid.to_vec();
    let counts: Vec<f64> = r_grid
        .iter()
        .map(|&r| {
            let xb = xi_bar_ws(r, ws, ip);
            nbar * volume(r) * (1.0 + xb)
        })
        .collect();
    CumulativeProfile { radii, counts }
}

/// Theory-predicted kNN CDF incorporating clustering via ξ̄(R).
///
/// CDF_k^theory(r) = Erlang_k(r; n̄_eff)
/// where n̄_eff = n̄ · (1 + ξ̄(r))
///
/// This is the leading-order correction to the Poisson CDF: clustering
/// raises the effective density, shifting the CDF to smaller radii.
pub fn predicted_cdf(
    k: usize,
    r: f64,
    nbar: f64,
    ws: &Workspace,
    ip: &IntegrationParams,
) -> f64 {
    let xb = xi_bar_ws(r, ws, ip);
    let nbar_eff = nbar * (1.0 + xb);
    erlang_cdf(k, r, nbar_eff)
}

/// Create a prediction closure for use with `KnnResiduals::from_cdfs`.
///
/// ```ignore
/// let predict = theory::make_predictor(nbar, &ws, &ip);
/// let residuals = KnnResiduals::from_cdfs(&cdf_dd, predict);
/// ```
pub fn make_predictor<'a>(
    nbar: f64,
    ws: &'a Workspace,
    ip: &'a IntegrationParams,
) -> impl Fn(usize, f64) -> f64 + 'a {
    move |k: usize, r: f64| predicted_cdf(k, r, nbar, ws, ip)
}

/// Look up all perturbation theory predictions at the smoothing scale
/// corresponding to the k-th nearest neighbor at mean density n̄.
///
/// Returns σ²_lin, σ²_Zel, σ²_J, ξ̄, S₃ at R_k = (3k / 4πn̄)^{1/3}.
pub fn theory_at_knn_scale(
    k_neighbours: usize,
    nbar: f64,
    cosmo: &Cosmology,
) -> Sigma2JDetailed {
    let r = knn_to_radius(k_neighbours, nbar);
    sigma2_j_detailed(cosmo, r, 2, 0.0, 0.0)
}

// ── kNN CDF predictions from Doroshkevich/LPT ──────────────────────────

/// Predict R-kNN CDFs in the same format as measured CDFs (`KnnCdfs`).
///
/// This produces theory predictions directly comparable to empirical
/// `KnnCdfs` from the estimator. For each k value, the Doroshkevich/LPT
/// CDF is evaluated at the given r grid.
pub fn predict_rknn_cdfs(
    k_values: &[usize],
    nbar: f64,
    r_values: &[f64],
    ws: &Workspace,
    ip: &IntegrationParams,
    cdf_params: &KnnCdfParams,
) -> KnnCdfs {
    let predictions = multi_k_rknn_cdfs(k_values, nbar, r_values, ws, ip, cdf_params);
    KnnCdfs {
        r_values: r_values.to_vec(),
        k_values: k_values.to_vec(),
        cdf_values: predictions.into_iter().map(|p| p.cdf).collect(),
        n_queries: 0, // theory, not from queries
    }
}

/// Predict D-kNN CDFs for biased tracers in `KnnCdfs` format.
pub fn predict_dknn_cdfs(
    k_values: &[usize],
    nbar_ref: f64,
    b1: f64,
    r_values: &[f64],
    ws: &Workspace,
    ip: &IntegrationParams,
    cdf_params: &KnnCdfParams,
) -> KnnCdfs {
    let predictions = multi_k_dknn_cdfs(k_values, nbar_ref, b1, r_values, ws, ip, cdf_params);
    KnnCdfs {
        r_values: r_values.to_vec(),
        k_values: k_values.to_vec(),
        cdf_values: predictions.into_iter().map(|p| p.cdf).collect(),
        n_queries: 0,
    }
}

/// Create a prediction closure for `KnnResiduals::from_cdfs` that uses
/// the full Doroshkevich/LPT CDF instead of the Erlang (Poisson) baseline.
///
/// This replaces `make_predictor` (which uses the ξ̄-shifted Erlang) with
/// the exact Jacobian CDF from the Doroshkevich distribution.
///
/// ```ignore
/// let predict = theory::make_doroshkevich_predictor(nbar, &ws, &ip, &cdf_params);
/// let residuals = KnnResiduals::from_cdfs(&cdf_dd, predict);
/// ```
pub fn make_doroshkevich_predictor<'a>(
    nbar: f64,
    ws: &'a Workspace,
    ip: &'a IntegrationParams,
    cdf_params: &'a KnnCdfParams,
) -> impl Fn(usize, f64) -> f64 + 'a {
    move |k: usize, r: f64| {
        let pred = rknn_cdf_physical(k, nbar, &[r], ws, ip, cdf_params);
        pred.cdf[0]
    }
}

/// Compute theory predictions on the same radial grid as a measurement.
///
/// Returns a Vec of `Sigma2JDetailed` for each radius in the grid,
/// suitable for overlaying theory curves on measurement plots.
pub fn theory_on_grid(
    r_grid: &[f64],
    cosmo: &Cosmology,
) -> Vec<Sigma2JDetailed> {
    r_grid
        .iter()
        .map(|&r| sigma2_j_detailed(cosmo, r, 2, 0.0, 0.0))
        .collect()
}

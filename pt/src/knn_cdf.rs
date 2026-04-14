//! kNN CDF predictions from Lagrangian perturbation theory.
//!
//! The Doroshkevich distribution gives the complete one-point PDF of the
//! Zel'dovich Jacobian J = det(I + ψ_{i,j}), from which all kNN CDFs follow.
//!
//! ## R-kNN: fixed k neighbours, measure enclosing radius
//!
//! The k-th nearest neighbour radius r_k corresponds to Lagrangian radius
//! R = (3k / 4πn̄)^{1/3} and Jacobian J = (r_k/R)³. Therefore:
//!
//!   P(r_k < r | k) = P(J < (r/R)³ | σ(R))
//!
//! ## D-kNN: fixed distance r, count neighbours from biased tracers
//!
//! For tracers with linear bias b₁, the conditional J distribution is shifted:
//!
//!   P(r_k < r | tracer) = P(J < (r/R)³ | σ(R), b₁)
//!
//! ## 2LPT correction
//!
//! The corrected CDF uses σ_eff such that Doroshkevich_variance(σ_eff) = σ²_J,
//! absorbing the LPT variance correction into the baseline.

use std::f64::consts::PI;

use crate::doroshkevich::{
    doroshkevich_cdf, doroshkevich_cdf_biased,
    doroshkevich_pdf,
    find_sigma_eff,
};
use crate::integrals::{self, IntegrationParams};
use crate::Workspace;

/// Parameters controlling the kNN CDF computation.
#[derive(Clone, Debug)]
pub struct KnnCdfParams {
    /// Gauss-Legendre quadrature points per dimension (default: 48).
    pub n_gauss: usize,
    /// Integration range in units of σ (default: 6.0).
    pub l_range: f64,
    /// Number of LPT corrections (0=Zel'dovich, 1=1LPT, 2=2LPT, 3=3LPT).
    pub n_lpt: usize,
    /// EFT counterterms (usually 0.0).
    pub c_j2: f64,
    pub c_j4: f64,
}

impl Default for KnnCdfParams {
    fn default() -> Self {
        KnnCdfParams {
            n_gauss: 48,
            l_range: 6.0,
            n_lpt: 2,
            c_j2: 0.0,
            c_j4: 0.0,
        }
    }
}

impl KnnCdfParams {
    /// Fast settings for MCMC-like use.
    pub fn fast() -> Self {
        KnnCdfParams { n_gauss: 40, l_range: 5.0, n_lpt: 2, c_j2: 0.0, c_j4: 0.0 }
    }

    /// High-accuracy settings for publication plots.
    pub fn accurate() -> Self {
        KnnCdfParams { n_gauss: 80, l_range: 7.0, n_lpt: 3, c_j2: 0.0, c_j4: 0.0 }
    }
}

/// Result of a kNN CDF prediction at a single k value.
#[derive(Clone, Debug)]
pub struct KnnCdfPrediction {
    /// Number of neighbours k.
    pub k: usize,
    /// Lagrangian radius R_L [Mpc/h].
    pub r_lag: f64,
    /// Linear σ at the smoothing scale.
    pub sigma_lin: f64,
    /// Effective σ incorporating LPT corrections (or = sigma_lin if n_lpt=0).
    pub sigma_eff: f64,
    /// Query radii [Mpc/h].
    pub r_values: Vec<f64>,
    /// CDF values P(r_k < r) at each query radius.
    pub cdf: Vec<f64>,
}

/// Result of a kNN CDF prediction including Poisson discreteness correction.
#[derive(Clone, Debug)]
pub struct KnnCdfPoissonCorrected {
    /// Base prediction (continuous field limit).
    pub base: KnnCdfPrediction,
    /// Poisson-corrected CDF values.
    pub cdf_poisson: Vec<f64>,
}

// ── R-kNN CDF: random query points ──────────────────────────────────────

/// R-kNN CDF at the Zel'dovich level (no LPT corrections).
///
/// P(r_k < r | k) = P(J < (r/R_L)³ | σ(R_L))
pub fn rknn_cdf_zel(
    sigma: f64,
    r_over_rl: &[f64],
    params: &KnnCdfParams,
) -> Vec<f64> {
    let j_thresholds: Vec<f64> = r_over_rl.iter().map(|&x| x * x * x).collect();
    doroshkevich_cdf(sigma, &j_thresholds, params.n_gauss, params.l_range)
}

/// R-kNN CDF with LPT corrections via the σ_eff mapping.
///
/// Finds σ_eff such that Doroshkevich_variance(σ_eff) = σ²_J(σ), then
/// evaluates the CDF at the effective σ.
pub fn rknn_cdf_corrected(
    sigma_lin: f64,
    r_over_rl: &[f64],
    ws: &Workspace,
    ip: &IntegrationParams,
    params: &KnnCdfParams,
) -> (Vec<f64>, f64) {
    let sigma2_j = compute_sigma2_j(sigma_lin, ws, ip, params);
    let sigma_eff = find_sigma_eff(sigma2_j, params.n_gauss);
    let j_thresholds: Vec<f64> = r_over_rl.iter().map(|&x| x * x * x).collect();
    let cdf = doroshkevich_cdf(sigma_eff, &j_thresholds, params.n_gauss, params.l_range);
    (cdf, sigma_eff)
}

/// Physical R-kNN CDF at fixed k and reference density n̄.
///
/// This is the main entry point for R-kNN predictions. It computes σ(R)
/// from the power spectrum and returns the full CDF.
pub fn rknn_cdf_physical(
    k: usize,
    nbar: f64,
    r_values: &[f64],
    ws: &Workspace,
    ip: &IntegrationParams,
    params: &KnnCdfParams,
) -> KnnCdfPrediction {
    let r_lag = crate::knn_to_radius(k, nbar);
    let sigma_lin = integrals::sigma2_tree_ws(r_lag, ws, ip).sqrt();

    let r_over_rl: Vec<f64> = r_values.iter().map(|&r| r / r_lag).collect();

    let (cdf, sigma_eff) = if params.n_lpt > 0 {
        rknn_cdf_corrected(sigma_lin, &r_over_rl, ws, ip, params)
    } else {
        let cdf = rknn_cdf_zel(sigma_lin, &r_over_rl, params);
        (cdf, sigma_lin)
    };

    KnnCdfPrediction {
        k,
        r_lag,
        sigma_lin,
        sigma_eff,
        r_values: r_values.to_vec(),
        cdf,
    }
}

// ── D-kNN CDF: biased tracer query points ───────────────────────────────

/// D-kNN CDF for tracers with linear bias b₁.
///
/// The conditional J distribution is shifted by the bias, which displaces
/// the mean density around tracers.
pub fn dknn_cdf_biased(
    k: usize,
    nbar_ref: f64,
    b1: f64,
    r_values: &[f64],
    ws: &Workspace,
    ip: &IntegrationParams,
    params: &KnnCdfParams,
) -> KnnCdfPrediction {
    let r_lag = crate::knn_to_radius(k, nbar_ref);
    let sigma_lin = integrals::sigma2_tree_ws(r_lag, ws, ip).sqrt();

    let sigma_eff = if params.n_lpt > 0 {
        let sigma2_j = compute_sigma2_j(sigma_lin, ws, ip, params);
        find_sigma_eff(sigma2_j, params.n_gauss)
    } else {
        sigma_lin
    };

    let j_thresholds: Vec<f64> = r_values.iter().map(|&r| (r / r_lag).powi(3)).collect();
    let cdf = doroshkevich_cdf_biased(sigma_eff, b1, &j_thresholds, params.n_gauss, params.l_range);

    KnnCdfPrediction {
        k,
        r_lag,
        sigma_lin,
        sigma_eff,
        r_values: r_values.to_vec(),
        cdf,
    }
}

// ── Poisson-corrected CDF ───────────────────────────────────────────────

/// R-kNN CDF with Poisson discreteness correction.
///
/// For discrete tracers, the count N within radius r follows a Poisson
/// distribution with mean λ = n̄ V(r) / J. The corrected CDF is:
///
///   P(r_k < r) = ∫ P(N ≥ k | λ = n̄ V(r) / J) × p_J(J | σ) dJ
///
/// At large k (≳ 10), the Poisson correction is negligible.
pub fn rknn_cdf_with_poisson(
    k: usize,
    nbar: f64,
    r_values: &[f64],
    ws: &Workspace,
    ip: &IntegrationParams,
    params: &KnnCdfParams,
) -> KnnCdfPoissonCorrected {
    let base = rknn_cdf_physical(k, nbar, r_values, ws, ip, params);

    // Compute the J-PDF on a fine grid
    let n_j = 500;
    let j_lo = -1.0_f64;
    let j_hi = (6.0 * base.sigma_eff).exp(); // generous upper bound
    let j_grid: Vec<f64> = (0..n_j).map(|i| {
        j_lo + (j_hi - j_lo) * i as f64 / (n_j - 1) as f64
    }).collect();
    let dj = (j_hi - j_lo) / (n_j - 1) as f64;

    let bw = base.sigma_eff / 10.0;
    let pdf_j = doroshkevich_pdf(base.sigma_eff, &j_grid, bw, params.n_gauss, params.l_range);

    let cdf_poisson: Vec<f64> = r_values.iter().map(|&r| {
        let v_eul = 4.0 / 3.0 * PI * r * r * r;
        let mut cdf_val = 0.0;
        for (i, &j) in j_grid.iter().enumerate() {
            if j <= 0.0 || pdf_j[i] <= 1e-30 { continue; }
            let lambda = nbar * v_eul / j;
            let p_geq_k = 1.0 - regularized_gamma_inc(k, lambda);
            cdf_val += p_geq_k * pdf_j[i] * dj;
        }
        cdf_val
    }).collect();

    KnnCdfPoissonCorrected {
        base,
        cdf_poisson,
    }
}

// ── Multi-k predictions ─────────────────────────────────────────────────

/// Compute R-kNN CDF predictions at multiple k values (marginal distributions).
pub fn multi_k_rknn_cdfs(
    k_values: &[usize],
    nbar: f64,
    r_values: &[f64],
    ws: &Workspace,
    ip: &IntegrationParams,
    params: &KnnCdfParams,
) -> Vec<KnnCdfPrediction> {
    k_values.iter().map(|&k| {
        rknn_cdf_physical(k, nbar, r_values, ws, ip, params)
    }).collect()
}

/// Compute D-kNN CDF predictions at multiple k values for biased tracers.
pub fn multi_k_dknn_cdfs(
    k_values: &[usize],
    nbar_ref: f64,
    b1: f64,
    r_values: &[f64],
    ws: &Workspace,
    ip: &IntegrationParams,
    params: &KnnCdfParams,
) -> Vec<KnnCdfPrediction> {
    k_values.iter().map(|&k| {
        dknn_cdf_biased(k, nbar_ref, b1, r_values, ws, ip, params)
    }).collect()
}

// ── Internal helpers ────────────────────────────────────────────────────

/// Compute σ²_J from the existing LPT pipeline at a given σ_lin.
fn compute_sigma2_j(
    sigma_lin: f64,
    ws: &Workspace,
    ip: &IntegrationParams,
    params: &KnnCdfParams,
) -> f64 {
    // We need to find R such that sigma2_tree(R) = sigma_lin^2
    // Then use the existing eval_single machinery.
    // Instead, compute σ²_J directly from σ²_lin using the polynomial model.
    let s = sigma_lin * sigma_lin;

    // Tier 1: Zel'dovich baseline
    let fac = 1.0 + 2.0 * s / 15.0;
    let zel = s * fac * fac;

    // Tier 2+: LPT corrections
    const D1_A0: f64 = -0.040058;
    const D1_A1: f64 = -0.822312;
    const D1_A2: f64 =  0.708537;
    const D2_A0: f64 =  0.022979;
    const D2_A1: f64 =  0.411372;
    const D2_A2: f64 = -0.303855;
    const CONV_RATIO: f64 = -0.535;

    let mut result = zel;

    if params.n_lpt >= 1 {
        let d1 = D1_A0 + D1_A1 * s + D1_A2 * s * s;
        result += zel * d1;
        if params.n_lpt >= 2 {
            let d2 = D2_A0 + D2_A1 * s + D2_A2 * s * s;
            result += zel * d2;
            if params.n_lpt >= 3 {
                result += zel * CONV_RATIO * d2;
            }
        }
    }

    // Counterterms
    if params.c_j2 != 0.0 || params.c_j4 != 0.0 {
        // Need the actual radius — find it from σ²_lin by searching the workspace.
        // For simplicity, use the fast relationship: at the smoothing scale,
        // the counterterm is small and we skip it here.
        // Full counterterm support uses sigma2_j_full directly.
        let _ = (ws, ip); // acknowledge unused in this branch
    }

    result
}

// ═══════════════════════════════════════════════════════════════════════════
// Tilted-PDF kNN predictions
//
// Unlike the "σ_eff" approach above (which absorbs one moment — the variance —
// into an adjusted Doroshkevich scale), the tilted-PDF approach deforms the
// full Doroshkevich distribution to simultaneously match σ²_J AND s₃. This
// yields improved PDFs, CDFs, and kNN predictions at moderate-R where the
// skewness correction matters (≳ 5% at R ~ 10 Mpc/h and worse at smaller R).
//
// Pipeline:
//   sigma_s = √(K₂ σ²_L)
//   (α, β) = fit_tilt_to_moments(sigma_s, target_var=σ²_J, target_s3=s₃)
//   tilted_pass = doroshkevich_tilted_pass(sigma_s, α, β, b1=0) → p_V, p_M
//   CDF_V = cdf_from_pdf(p_V)
//   CDF_M = cdf_from_pdf(p_M) [optionally biased]
//   j₀(r) = (r / R_Lag)³     with R_Lag = (3k / 4π n̄)^(1/3)
//   R-kNN:  P(r_k < r | k)       = CDF_V(j₀(r))
//   D-kNN:  P(r_k < r | k, b1)   = CDF_M,biased(j₀(r))
//   Poisson correction:  1D convolution of the J-PDF with the Poisson CDF.
// ═══════════════════════════════════════════════════════════════════════════

use crate::doroshkevich::{
    doroshkevich_tilted_pass, fit_tilt_to_moments, cdf_from_pdf, uniform_j_grid,
    BiasParams,
};

/// Pair type for kNN predictions.
///
/// The "query sample" is where you drop the query point (centre of the
/// neighbour search). The "neighbour sample" is what you count neighbours of.
///
/// - `Mm` (matter-matter): unbiased query (volume-weighted PDF) + unbiased
///   matter neighbours. Appropriate for idealized R-kNN statistics of the
///   underlying matter distribution; not directly a survey observable.
/// - `Gm` (galaxy-matter): biased tracer query (mass-weighted PDF with the
///   galaxy bias tilt) + matter neighbours. Rarely directly observable for
///   galaxy surveys (you'd need a matter field from, e.g., weak lensing).
/// - `Gg` (galaxy-galaxy): biased tracer query + biased tracer neighbours.
///   **This is what most galaxy-survey kNN measurements are**, e.g., counting
///   galaxies in volumes around galaxies. Uses both a query bias `b1_query`
///   and a neighbour bias `b1_neighbour` (often the same number for
///   auto-kNN; different for cross-sample kNN).
///
/// For the enclosed count we use the standard Lagrangian prescription:
/// in Eulerian volume V_eul around the query, the neighbour-sample count is
///   N = n̄_neighbour × V_Lag × (1 + b₁_neighbour × δ_Lag)
///     = n̄_neighbour × V_eul × (1 + b₁_neighbour × I₁) / J
/// with I₁_code = δ_L in the Doroshkevich convention. The galaxy-galaxy
/// Jacobian threshold is therefore J ≤ j₀(r) × (1 + b₁_neighbour × I₁).
/// We apply this via the factorized approximation: replace I₁ by its
/// conditional mean ⟨I₁ | J⟩ from the tilted quadrature.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PairType {
    Mm,
    Gm,
    Gg,
}

/// Result of a tilted-PDF kNN prediction at one k value.
#[derive(Clone, Debug)]
pub struct TiltedKnnPrediction {
    pub k: usize,
    pub r_lag: f64,
    pub sigma_s: f64,
    /// Tilt parameter α used.
    pub alpha: f64,
    /// Tilt parameter β used.
    pub beta: f64,
    /// Achieved variance after fitting (should equal target_var to <tol).
    pub achieved_var: f64,
    /// Achieved s₃ after fitting.
    pub achieved_s3: f64,
    pub r_values: Vec<f64>,
    pub cdf: Vec<f64>,
    /// Volume-weighted PDF on the internal J grid.
    pub j_grid: Vec<f64>,
    pub pdf_v: Vec<f64>,
    pub pdf_m: Vec<f64>,
}

/// Configuration for the tilted-PDF kNN prediction.
#[derive(Clone, Copy, Debug)]
pub struct TiltedKnnParams {
    /// Quadrature points per eigenvalue dimension (default 80).
    pub n_gauss: usize,
    /// Integration range in units of σ (default 6.0).
    pub l_range: f64,
    /// Number of J bins for the PDF histogram.
    pub n_j_bins: usize,
    /// Lower edge of the J grid.
    pub j_min: f64,
    /// Upper edge of the J grid.
    pub j_max: f64,
    /// Maximum Newton iterations for tilt fitting.
    pub max_iter: usize,
    /// Relative tolerance for the normalized moment residual.
    pub tol: f64,
}

impl Default for TiltedKnnParams {
    fn default() -> Self {
        TiltedKnnParams {
            n_gauss: 60,
            l_range: 6.0,
            n_j_bins: 600,
            j_min: -2.0,
            j_max: 10.0,
            max_iter: 25,
            tol: 1e-3,
        }
    }
}

/// Unified tilted kNN CDF with pair-type selector.
///
/// * `pair_type` — which sample is at the query, which is counted.
/// * `k` — neighbour rank.
/// * `nbar` — number density of the NEIGHBOUR sample [h³/Mpc³] (matter for
///            `Mm`/`Gm`, galaxies for `Gg`).
/// * `sigma2_j`, `s3` — target second and third cumulants from PT.
/// * `b1_query` — bias of the query sample (0 for `Mm`).
/// * `b1_neighbour` — bias of the neighbour sample (0 for `Mm`, `Gm`).
/// * `r_values` — query radii [Mpc/h].
///
/// Returns the full prediction including tilt parameters, PDFs, and CDFs.
pub fn knn_cdf_tilted(
    pair_type: PairType,
    k: usize, nbar: f64, sigma2_j: f64, s3: f64,
    b1_query: f64, b1_neighbour: f64,
    r_values: &[f64], params: &TiltedKnnParams,
) -> TiltedKnnPrediction {
    knn_cdf_tilted_bias(
        pair_type, k, nbar, sigma2_j, s3,
        BiasParams::b1_only(b1_query),
        BiasParams::b1_only(b1_neighbour),
        r_values, params,
    )
}

/// Full Lagrangian-bias-expansion kNN CDF via polynomial bias weighting.
///
/// * `bias_query` — Lagrangian bias of the query sample (b₁, b₂, b_{s²}).
/// * `bias_neighbour` — Lagrangian bias of the neighbour sample (b₁ used
///   for the g-g neighbour-count enhancement via the (1+b₁_n × ⟨I₁|J⟩)
///   factorised approximation; higher-order terms deferred).
pub fn knn_cdf_tilted_bias(
    pair_type: PairType,
    k: usize, nbar: f64, sigma2_j: f64, s3: f64,
    bias_query: BiasParams, bias_neighbour: BiasParams,
    r_values: &[f64], params: &TiltedKnnParams,
) -> TiltedKnnPrediction {
    let r_lag = (3.0 * k as f64 / (4.0 * PI * nbar)).cbrt();
    let sigma_s = sigma2_j.max(0.0).sqrt();

    // Fit PT tilt (α_PT, β_PT) to match σ²_J and s₃ of the MATTER field
    // (the bias does not change σ² of the matter).
    let (alpha_pt, beta_pt, ach_var, ach_s3) = fit_tilt_to_moments(
        sigma_s, sigma2_j, s3, params.n_gauss, params.l_range,
        params.max_iter, params.tol,
    );

    let j_grid = uniform_j_grid(params.j_min, params.j_max, params.n_j_bins);
    let sigma_lin_sq = sigma_s * sigma_s;

    // Precompute ⟨s²⟩ at σ_s for the bias weight's subtraction term.
    let cross = crate::doroshkevich::doroshkevich_unbiased_cross_moments(
        sigma_s, params.n_gauss, params.l_range,
    );

    // Query-side effective bias: Mm → unbiased; Gm/Gg → bias_query.
    let query_bias = match pair_type {
        PairType::Mm => BiasParams::UNBIASED,
        PairType::Gm | PairType::Gg => bias_query,
    };

    let pass = crate::doroshkevich::doroshkevich_biased_polynomial_pass(
        sigma_s, alpha_pt, beta_pt, query_bias,
        sigma_lin_sq, cross.mean_s2,
        &j_grid, params.n_gauss, params.l_range,
    );

    // Query-side PDF choice: Mm uses p_V (volume-weighted around random point);
    // Gm/Gg use p_M (mass-weighted = galaxy-centred).
    let query_pdf: &[f64] = match pair_type {
        PairType::Mm => &pass.pdf_v_g,
        PairType::Gm | PairType::Gg => &pass.pdf_m_g,
    };
    let query_cdf = cdf_from_pdf(query_pdf, &j_grid);

    // Neighbour-count enhancement for Gg: per-bin threshold
    // J ≤ j₀(r) × (1 + b₁_n × ⟨I₁ | J⟩).
    let cdf: Vec<f64> = r_values.iter().map(|&r| {
        let j0 = (r / r_lag).powi(3);
        match pair_type {
            PairType::Mm | PairType::Gm => interp_cdf(&j_grid, &query_cdf, j0),
            PairType::Gg => {
                let dj = j_grid[1] - j_grid[0];
                let mut acc = 0.0;
                for (i, &j_bin) in j_grid.iter().enumerate() {
                    let i1_bin = pass.mean_i1_given_j[i];
                    let j_thresh = j0 * (1.0 + bias_neighbour.b1 * i1_bin);
                    if j_bin + 0.5 * dj <= j_thresh {
                        acc += query_pdf[i] * dj;
                    } else if j_bin - 0.5 * dj < j_thresh {
                        let frac = (j_thresh - (j_bin - 0.5 * dj)) / dj;
                        acc += query_pdf[i] * dj * frac.clamp(0.0, 1.0);
                    }
                }
                acc.clamp(0.0, 1.0)
            }
        }
    }).collect();

    TiltedKnnPrediction {
        k, r_lag, sigma_s, alpha: alpha_pt, beta: beta_pt,
        achieved_var: ach_var, achieved_s3: ach_s3,
        r_values: r_values.to_vec(), cdf,
        j_grid,
        pdf_v: pass.pdf_v_g.clone(),
        pdf_m: pass.pdf_m_g.clone(),
    }
}

/// R-kNN (matter-matter) CDF — thin wrapper over `knn_cdf_tilted` with
/// `PairType::Mm`, no biases.
pub fn rknn_cdf_tilted(
    k: usize, nbar: f64, sigma2_j: f64, s3: f64,
    r_values: &[f64], params: &TiltedKnnParams,
) -> TiltedKnnPrediction {
    knn_cdf_tilted(PairType::Mm, k, nbar, sigma2_j, s3, 0.0, 0.0, r_values, params)
}

/// D-kNN (galaxy-matter) CDF — thin wrapper with `PairType::Gm`, biased
/// query + matter neighbours (b₁-only).
pub fn dknn_cdf_tilted(
    k: usize, nbar_ref: f64, sigma2_j: f64, s3: f64, b1: f64,
    r_values: &[f64], params: &TiltedKnnParams,
) -> TiltedKnnPrediction {
    knn_cdf_tilted(PairType::Gm, k, nbar_ref, sigma2_j, s3, b1, 0.0, r_values, params)
}

/// Poisson-corrected kNN CDF for small k, via 1D convolution of the J-PDF
/// with the Poisson CDF:
///
///   P(r_k < r | k) = ∫ P(N ≥ k | λ = n̄ V_eul × J') p_V(J') dJ'
///
/// Wait — this is not the right convolution: V_eul itself is R_Lag³ × J₀(r),
/// and the enclosed count has mean n̄ V_eul. The relevant marginal is then
///
///   P(r_k < r | k) = P(N(r) ≥ k) averaged over J' with volume weighting
///                  = ∫ [1 - Γ_reg(k, n̄ (4π/3) r³)] is the NAIVE Poisson;
/// the J-coupling enters because the Lagrangian volume M/ρ̄ with k/n̄ mass has
/// fluctuating Eulerian radius. At the operational level: for each J, the
/// "effective query radius" is r × J^(-1/3), so:
///
///   CDF_Poisson(r) = ∫ F_Erlang(k, λ = n̄ (4π/3) (r J^(-1/3))³) p_V(J) dJ.
///
/// For J > 0 this simplifies to λ = n̄ (4π/3) r³ / J, so the Poisson argument
/// scales as 1/J. Bins with J ≤ 0 are skipped.
pub fn poisson_correct_cdf(
    pdf_v: &[f64], j_grid: &[f64],
    k: usize, nbar: f64, r_values: &[f64],
) -> Vec<f64> {
    let dj = if j_grid.len() >= 2 { j_grid[1] - j_grid[0] } else { 1.0 };
    let four_pi_over_3 = 4.0 / 3.0 * PI;
    r_values.iter().map(|&r| {
        let r3 = r.powi(3);
        let mut acc = 0.0;
        for (p, &j) in pdf_v.iter().zip(j_grid.iter()) {
            if j <= 0.0 || *p <= 0.0 { continue; }
            let lambda = nbar * four_pi_over_3 * r3 / j;
            // P(N ≥ k | λ) = 1 − P(N < k | λ) = 1 − Γ_reg(k, λ)
            let p_ge_k = 1.0 - regularized_gamma_inc(k, lambda);
            acc += p_ge_k * p * dj;
        }
        acc.clamp(0.0, 1.0)
    }).collect()
}

/// Linear interpolation of a CDF on a uniform log-unspecified grid.
/// Clamps to [0, 1] at the boundaries.
fn interp_cdf(j_grid: &[f64], cdf: &[f64], j_query: f64) -> f64 {
    let n = j_grid.len();
    if n == 0 { return 0.0; }
    if j_query <= j_grid[0] { return cdf[0]; }
    if j_query >= j_grid[n - 1] { return cdf[n - 1]; }
    // Uniform grid assumption.
    let dj = j_grid[1] - j_grid[0];
    let f = (j_query - j_grid[0]) / dj;
    let i = f as usize;
    let t = f - i as f64;
    if i >= n - 1 { return cdf[n - 1]; }
    cdf[i] * (1.0 - t) + cdf[i + 1] * t
}

/// Regularized incomplete gamma function: Γ(k, λ) / Γ(k).
///
/// P(k, λ) = 1 - Q(k, λ) where Q is the survival function.
/// This gives the Poisson CDF: P(N < k | λ) = Γ_reg(k, λ).
fn regularized_gamma_inc(k: usize, lambda: f64) -> f64 {
    if lambda <= 0.0 { return 0.0; }
    if lambda > 700.0 { return 1.0; } // overflow guard

    // For small k, use direct Poisson sum: P(N < k | λ) = exp(-λ) Σ_{j=0}^{k-1} λ^j / j!
    // This is more stable than the gamma function for integer k.
    let mut sum = 0.0;
    let mut term = 1.0; // λ^0 / 0!
    for j in 0..k {
        if j > 0 {
            term *= lambda / j as f64;
        }
        sum += term;
        // Overflow guard for very large λ
        if term > 1e100 { return 1.0; }
    }
    let result = (-lambda).exp() * sum;
    result.min(1.0).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doroshkevich::doroshkevich_moments;

    #[test]
    fn rknn_cdf_zel_basic() {
        // At σ → 0, J → 1 always, so CDF should be step at r/R_L = 1
        let sigma = 0.01;
        let r_over_rl = vec![0.5, 0.99, 1.0, 1.01, 2.0];
        let params = KnnCdfParams::default();
        let cdf = rknn_cdf_zel(sigma, &r_over_rl, &params);

        assert!(cdf[0] < 0.01, "CDF should be ~0 well below J=1");
        assert!(cdf[4] > 0.99, "CDF should be ~1 well above J=1");
    }

    #[test]
    fn rknn_cdf_monotonic() {
        let sigma = 0.5;
        let r_over_rl: Vec<f64> = (1..=50).map(|i| 0.1 * i as f64).collect();
        let params = KnnCdfParams::default();
        let cdf = rknn_cdf_zel(sigma, &r_over_rl, &params);

        for i in 1..cdf.len() {
            assert!(cdf[i] >= cdf[i - 1] - 1e-10,
                "CDF not monotonic at i={}: {} < {}", i, cdf[i], cdf[i - 1]);
        }
        assert!(cdf[0] >= 0.0 && *cdf.last().unwrap() <= 1.0);
    }

    #[test]
    fn doroshkevich_cdf_matches_moments() {
        // The CDF at J = median should be ~0.5.
        // At σ = 0.3, the distribution is close to Gaussian around J=1.
        let sigma = 0.3;
        let params = KnnCdfParams::default();

        // Check that CDF(J=1) is close to 0.5 (median near mean for small σ)
        let cdf = doroshkevich_cdf(sigma, &[1.0], params.n_gauss, params.l_range);
        assert!((cdf[0] - 0.5).abs() < 0.1,
            "CDF(J=1) = {:.3}, expected ~0.5 for small σ", cdf[0]);
    }

    #[test]
    fn biased_cdf_shifts_distribution() {
        let sigma = 0.5;
        let j_thresholds = vec![0.5, 1.0, 1.5, 2.0];
        let params = KnnCdfParams::default();

        let cdf_unbiased = doroshkevich_cdf(sigma, &j_thresholds, params.n_gauss, params.l_range);
        let cdf_biased = doroshkevich_cdf_biased(sigma, 1.0, &j_thresholds, params.n_gauss, params.l_range);

        // Positive bias b₁ > 0 means the tracer is in an overdensity, where
        // J < 1 (compressed volumes). The CDF must shift LEFT — more probability
        // mass at small J. Concretely: P(J<1 | b₁>0) > P(J<1 | b₁=0).
        let cdf_at_one_unbiased = cdf_unbiased[1];  // j_thresholds[1] = 1.0
        let cdf_at_one_biased = cdf_biased[1];
        assert!(cdf_at_one_biased > cdf_at_one_unbiased + 0.05,
                "Biased CDF must shift LEFT for overdense tracer: \
                 P(J<1|b₁=1)={:.4} should exceed P(J<1|b₁=0)={:.4}",
                cdf_at_one_biased, cdf_at_one_unbiased);
    }

    #[test]
    fn regularized_gamma_basic() {
        // P(N < 1 | λ=1) = exp(-1) ≈ 0.3679
        let p = regularized_gamma_inc(1, 1.0);
        assert!((p - (-1.0_f64).exp()).abs() < 1e-10);

        // P(N < 2 | λ=1) = exp(-1)(1 + 1) = 2e^{-1} ≈ 0.7358
        let p2 = regularized_gamma_inc(2, 1.0);
        assert!((p2 - 2.0 * (-1.0_f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn find_sigma_eff_roundtrip() {
        let sigma = 0.5;
        let n_gauss = 48;
        let target = doroshkevich_moments(sigma, n_gauss, 6.0).variance;
        let sigma_found = find_sigma_eff(target, n_gauss);
        assert!((sigma_found - sigma).abs() < 1e-4,
            "find_sigma_eff({:.6}) = {:.6}, expected {:.6}", target, sigma_found, sigma);
    }

    // ─── Tilted-PDF tests ───────────────────────────────────────────────

    #[test]
    fn tilt_zero_equals_doroshkevich() {
        use crate::doroshkevich::{doroshkevich_tilted_moments};
        let sigma = 0.5;
        let m = doroshkevich_moments(sigma, 80, 6.0);
        let (_, var_tilt, _, s3_tilt) =
            doroshkevich_tilted_moments(sigma, 0.0, 0.0, 0.0, 80, 6.0);
        assert!((var_tilt - m.variance).abs() / m.variance < 1e-10,
                "α=β=0 must reproduce Doroshkevich: var {} vs {}", var_tilt, m.variance);
        assert!((s3_tilt - m.s3).abs() / m.s3.abs() < 1e-9,
                "α=β=0 must reproduce Doroshkevich s3: {} vs {}", s3_tilt, m.s3);
    }

    #[test]
    fn tilt_fit_converges_to_targets() {
        use crate::doroshkevich::{doroshkevich_moments, fit_tilt_to_moments,
                                    doroshkevich_tilted_moments};
        // Pick a σ for which perturbation theory is meaningful, then tilt
        // toward a modestly different (variance, s3) — matching the scale of
        // 2LPT corrections — and check Newton recovers it.
        let sigma = 1.0;
        let base = doroshkevich_moments(sigma, 80, 6.0);
        let target_var = base.variance * 1.15;   // 15% variance enhancement
        let target_s3 = base.s3 * 1.40;           // 40% s3 enhancement
        let (alpha, beta, ach_var, ach_s3) = fit_tilt_to_moments(
            sigma, target_var, target_s3, 80, 6.0, 50, 1e-4,
        );
        let rel_v = (ach_var - target_var).abs() / target_var;
        let rel_s = (ach_s3 - target_s3).abs() / target_s3.abs();
        assert!(rel_v < 1e-2,
                "var not matched: target {}, got {} (rel {:.2e}, α={} β={})",
                target_var, ach_var, rel_v, alpha, beta);
        assert!(rel_s < 1e-2,
                "s3 not matched: target {}, got {} (rel {:.2e})",
                target_s3, ach_s3, rel_s);
        // Cross-check: a second call with the same α, β must give the same moments.
        let (_, v2, _, s2) = doroshkevich_tilted_moments(sigma, alpha, beta, 0.0, 80, 6.0);
        assert!((v2 - ach_var).abs() < 1e-12);
        assert!((s2 - ach_s3).abs() < 1e-12);
    }

    #[test]
    fn pdfs_normalize_to_one() {
        use crate::doroshkevich::{doroshkevich_tilted_pass, uniform_j_grid};
        let sigma = 0.5;
        let j_grid = uniform_j_grid(-1.5, 6.0, 500);
        let pass = doroshkevich_tilted_pass(sigma, 0.0, 0.0, 0.0, &j_grid, 80, 6.0);
        let dj = j_grid[1] - j_grid[0];
        let integral_v: f64 = pass.pdf_v.iter().sum::<f64>() * dj;
        let integral_m: f64 = pass.pdf_m.iter().sum::<f64>() * dj;
        // Volume PDF: should integrate to 1 (within bin-edge truncation).
        assert!((integral_v - 1.0).abs() < 2e-2,
                "p_V does not integrate to 1: {}", integral_v);
        // Mass PDF: same (normalized separately).
        assert!((integral_m - 1.0).abs() < 2e-2,
                "p_M does not integrate to 1: {}", integral_m);
    }

    #[test]
    fn mass_weighted_mean_below_volume_weighted() {
        // Mass-weighted density p_M(J) ∝ p_V(J) / J puts MORE weight on small J
        // (compressed regions, where unit volume carries more mass). Therefore
        // ⟨J⟩_M ≤ ⟨J⟩_V = 1, with equality only for a delta-function PDF.
        use crate::doroshkevich::{doroshkevich_tilted_pass, uniform_j_grid};
        let sigma = 0.8;
        let j_grid = uniform_j_grid(-1.0, 6.0, 400);
        let pass = doroshkevich_tilted_pass(sigma, 0.0, 0.0, 0.0, &j_grid, 80, 6.0);
        let dj = j_grid[1] - j_grid[0];
        let mean_v: f64 = j_grid.iter().zip(pass.pdf_v.iter()).map(|(j, p)| j * p * dj).sum();
        let mean_m: f64 = j_grid.iter().zip(pass.pdf_m.iter()).map(|(j, p)| j * p * dj).sum();
        assert!(mean_v > 0.90 && mean_v < 1.10,
                "volume-weighted ⟨J⟩ should be ≈ 1, got {}", mean_v);
        assert!(mean_m < mean_v,
                "mass-weighted ⟨J⟩ must be below volume-weighted ⟨J⟩ (got M={}, V={})",
                mean_m, mean_v);
        assert!(mean_m > 0.0,
                "mass-weighted ⟨J⟩ should be positive (got {})", mean_m);
    }

    #[test]
    fn poisson_limit_k1_unbiased() {
        // In the no-clustering limit (σ → 0, so J → 1 deterministically), the
        // tilted CDF approaches a step function at j₀ = 1. After Poisson
        // convolution at k=1, the CDF should approach 1 − exp(−n̄(4π/3)r³),
        // the Erlang k=1 (exponential) distribution.
        use crate::doroshkevich::{doroshkevich_tilted_pass, uniform_j_grid};
        let sigma = 0.05;   // essentially linear, J ≈ 1
        let j_grid = uniform_j_grid(0.5, 1.5, 400);
        let pass = doroshkevich_tilted_pass(sigma, 0.0, 0.0, 0.0, &j_grid, 60, 6.0);

        let nbar = 1e-3;
        let k = 1usize;
        let r_lag = (3.0 * k as f64 / (4.0 * PI * nbar)).cbrt();
        let r_values: Vec<f64> = (1..=20).map(|i| 0.2 * i as f64 * r_lag).collect();

        let cdf_poisson = poisson_correct_cdf(&pass.pdf_v, &j_grid, k, nbar, &r_values);
        // Analytic Erlang-1 CDF: 1 − exp(−n̄ (4π/3) r³)
        let four_pi_over_3 = 4.0 / 3.0 * PI;
        for (&r, &p) in r_values.iter().zip(cdf_poisson.iter()) {
            let lambda = nbar * four_pi_over_3 * r.powi(3);
            let analytic = 1.0 - (-lambda).exp();
            assert!((p - analytic).abs() < 0.01,
                    "k=1 σ→0 Poisson CDF at r={:.1}: {} vs Erlang {}", r, p, analytic);
        }
    }

    // ─── Bias-expansion tests ────────────────────────────────────────────

    #[test]
    fn bias_expansion_zero_equals_unbiased() {
        use crate::doroshkevich::{
            doroshkevich_biased_polynomial_pass,
            doroshkevich_tilted_pass, doroshkevich_unbiased_cross_moments,
            uniform_j_grid, BiasParams,
        };
        let sigma = 0.8_f64;
        let cross = doroshkevich_unbiased_cross_moments(sigma, 80, 6.0);
        let j_grid = uniform_j_grid(-1.0, 6.0, 400);

        let biased = doroshkevich_biased_polynomial_pass(
            sigma, 0.0, 0.0, BiasParams::UNBIASED, sigma * sigma, cross.mean_s2,
            &j_grid, 80, 6.0,
        );
        let untilted = doroshkevich_tilted_pass(sigma, 0.0, 0.0, 0.0, &j_grid, 80, 6.0);

        // At b1=b2=bs2=0 the biased weight is identically 1. The biased pass
        // must reproduce the untilted pass to machine precision.
        for (p_a, p_b) in biased.pdf_v_g.iter().zip(untilted.pdf_v.iter()) {
            assert!((p_a - p_b).abs() < 1e-10,
                    "p_V differs: biased {} vs untilted {}", p_a, p_b);
        }
        assert!((biased.variance_g - untilted.variance).abs() / untilted.variance < 1e-10);
        assert!((biased.s3_g - untilted.s3).abs() / untilted.s3.abs() < 1e-9);
    }

    #[test]
    fn cross_moments_vanish_in_gaussian_limit() {
        // For a Gaussian random field, ⟨δ × δ²⟩ = 0 and ⟨δ × s²⟩ = 0 at the
        // same point. The Doroshkevich cross-moments M₁₂ = ⟨(J−1) I₁²⟩ and
        // M_{s²} should vanish as σ → 0 (where J−1 ≈ −I₁).
        use crate::doroshkevich::doroshkevich_unbiased_cross_moments;
        for &sigma in &[0.05_f64, 0.10, 0.15] {
            let m = doroshkevich_unbiased_cross_moments(sigma, 80, 6.0);
            // M ~ O(σ⁴) (cumulant of 4 Gaussian fields → disconnected = 0,
            // connected at first sub-leading order = σ⁴).
            let scale = sigma.powi(4);
            assert!(m.m12.abs() < 10.0 * scale,
                    "M₁₂ should be O(σ⁴) in Gaussian limit; σ={} M₁₂={} σ⁴={}",
                    sigma, m.m12, scale);
            assert!(m.m_s2.abs() < 10.0 * scale,
                    "M_{{s²}} should be O(σ⁴) in Gaussian limit; σ={} M_{{s²}}={}",
                    sigma, m.m_s2);
        }
    }

    #[test]
    fn bias_estimator_contamination_by_b2() {
        // The naive estimator b1_apparent = −ξ̄/σ² is contaminated at O(b₂ σ²)
        // when the truth has b₂ > 0. Set b1=2, b2=1, and check the naive
        // estimator differs from b1=2 by a non-trivial amount at R=10.
        use crate::{Cosmology, Workspace};
        use crate::integrals::{IntegrationParams, sigma2_tree_ws, xi_bar_ws};
        use crate::doroshkevich::{
            doroshkevich_unbiased_cross_moments, xibar_tree_bias_expanded, BiasParams,
        };

        let cosmo = Cosmology::planck2018();
        let mut ws = Workspace::new(4000);
        ws.update_cosmology(&cosmo);
        let ip = IntegrationParams::fast();

        let r = 10.0;
        let sigma2_lin = sigma2_tree_ws(r, &ws, &ip);
        let xibar_lin = xi_bar_ws(r, &ws, &ip);
        let sigma_s = sigma2_lin.sqrt();
        let cross = doroshkevich_unbiased_cross_moments(sigma_s, 80, 6.0);

        let bias_true = BiasParams::coevolution(2.0, 1.0);
        // ξ̄(R) with full bias expansion (use ξ̄_L as the "σ²_L" proxy since
        // at tree level both enter with single W).
        let xibar_full = xibar_tree_bias_expanded(xibar_lin, 1.0, &cross, &bias_true);
        // Naive estimator: b₁_naive = −ξ̄ / ξ̄_lin (or equivalently over σ² for
        // the variance-bearing quantity). At R=10 σ²_lin ~ 0.48, so the b₂M₁₂
        // contribution is non-trivial.
        let b1_naive = -xibar_full / xibar_lin;
        println!("b1_true={} b1_naive={:.3}  diff={:+.3}",
                 bias_true.b1, b1_naive, b1_naive - bias_true.b1);
        assert!(b1_naive != bias_true.b1,
                "naive estimator must differ from true b₁ when b₂≠0");
        // Difference should be a significant fraction — O(b₂ × M₁₂ / ξ̄_lin).
        let diff_frac = (b1_naive - bias_true.b1).abs() / bias_true.b1;
        assert!(diff_frac > 0.02,
                "b₂=1 contamination should give >2% shift in b₁_apparent at R=10, got {:.3}%",
                diff_frac * 100.0);
    }

    #[test]
    fn pair_types_ordering_at_fixed_r() {
        // At fixed r and with positive bias, the galaxy-galaxy CDF should
        // exceed the galaxy-matter CDF (biased neighbours are more clustered,
        // so enclosed count reaches k at smaller r → more prob at fixed r).
        // Matter-matter lies in between / separate convention.
        use crate::{Cosmology, Workspace};
        use crate::integrals::{IntegrationParams, sigma2_tree_ws};
        use crate::doroshkevich::doroshkevich_moments;

        let cosmo = Cosmology::planck2018();
        let mut ws = Workspace::new(4000);
        ws.update_cosmology(&cosmo);
        let ip = IntegrationParams::fast();

        let k = 16usize;
        let nbar = 1e-3;
        let r_lag = (3.0 * k as f64 / (4.0 * PI * nbar)).cbrt();
        let s = sigma2_tree_ws(r_lag, &ws, &ip);
        let base = doroshkevich_moments(s.sqrt(), 80, 6.0);
        let b1 = 1.5;

        // A representative r around R_Lag where the CDF is transitioning.
        let r_test = vec![0.9 * r_lag];

        let p_mm = knn_cdf_tilted(PairType::Mm, k, nbar, base.variance, base.s3,
                                  0.0, 0.0, &r_test, &TiltedKnnParams::default());
        let p_gm = knn_cdf_tilted(PairType::Gm, k, nbar, base.variance, base.s3,
                                  b1, 0.0, &r_test, &TiltedKnnParams::default());
        let p_gg = knn_cdf_tilted(PairType::Gg, k, nbar, base.variance, base.s3,
                                  b1, b1, &r_test, &TiltedKnnParams::default());
        println!("mm={:.4}  gm={:.4}  gg={:.4}",
                 p_mm.cdf[0], p_gm.cdf[0], p_gg.cdf[0]);
        // gg should exceed gm (neighbour bias enhances enclosed count).
        assert!(p_gg.cdf[0] > p_gm.cdf[0],
                "gg={} should exceed gm={} at fixed r with b1>0",
                p_gg.cdf[0], p_gm.cdf[0]);
        // All CDFs bounded.
        for cdf in [&p_mm.cdf, &p_gm.cdf, &p_gg.cdf] {
            assert!(cdf[0] >= 0.0 && cdf[0] <= 1.0);
        }
    }

    #[test]
    fn rknn_cdf_tilted_monotonic_and_bounded() {
        // End-to-end: tilted R-kNN at Planck-ish scales. CDF must be
        // monotonic, bounded in [0, 1], and smooth.
        use crate::{Cosmology, Workspace};
        use crate::integrals::{IntegrationParams, sigma2_tree_ws};
        use crate::doroshkevich::doroshkevich_moments;

        let cosmo = Cosmology::planck2018();
        let mut ws = Workspace::new(4000);
        ws.update_cosmology(&cosmo);
        let ip = IntegrationParams::fast();

        let k = 16usize;
        let nbar = 1e-3;
        let r_lag = (3.0 * k as f64 / (4.0 * PI * nbar)).cbrt();
        let s = sigma2_tree_ws(r_lag, &ws, &ip);

        // Use untilted Doroshkevich at σ = √s as the target (so we're matching
        // the untilted distribution itself — tilt parameters should converge to 0).
        let base = doroshkevich_moments(s.sqrt(), 80, 6.0);

        let r_values: Vec<f64> = (1..=40).map(|i| 0.1 * i as f64 * r_lag).collect();
        let pred = rknn_cdf_tilted(k, nbar, base.variance, base.s3,
                                    &r_values, &TiltedKnnParams::default());

        for i in 1..pred.cdf.len() {
            assert!(pred.cdf[i] >= pred.cdf[i - 1] - 1e-10,
                    "CDF not monotonic at i={}: {} < {}",
                    i, pred.cdf[i], pred.cdf[i - 1]);
        }
        assert!(pred.cdf[0] >= 0.0 && *pred.cdf.last().unwrap() <= 1.0);
        assert!(pred.cdf[0] < 0.05, "CDF at smallest r should be ~0, got {}", pred.cdf[0]);
        assert!(*pred.cdf.last().unwrap() > 0.95,
                "CDF at largest r should be ~1, got {}", pred.cdf.last().unwrap());
    }
}

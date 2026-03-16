//! Non-Poissonian diagnostics from kNN residuals.
//!
//! The residuals Δ_k(r₀) = CDF^measured_kNN(r₀) − CDF^{Poisson|ξ}_kNN(r₀)
//! encode non-Gaussian and non-Poissonian structure at each scale.
//! From these residuals we extract:
//!
//! - δσ²_N(V) = excess variance of counts in cells
//! - α_SN(V) = scale-dependent shot-noise rescaling for covariance
//! - Factorial cumulants C_j(V)
//! - σ²_NL(R) = full nonlinear density variance (model-free)

/// Residual between measured and Poisson|ξ predicted kNN-CDFs.
#[derive(Debug, Clone)]
pub struct KnnResiduals {
    /// Sphere radii at which residuals are evaluated
    pub r0: Vec<f64>,
    /// Δ_k(r₀) for each k = 1..k_max, indexed as [k-1][r0_idx]
    pub delta_k: Vec<Vec<f64>>,
    /// k_max
    pub k_max: usize,
}

/// Scale-dependent shot-noise parameter.
#[derive(Debug, Clone)]
pub struct AlphaSn {
    /// Sphere radii (or volumes)
    pub r0: Vec<f64>,
    /// α_SN(V) at each radius
    pub alpha: Vec<f64>,
}

use crate::ladder::KnnCdfSummary;

impl KnnResiduals {
    /// Compute residuals Δ_k(r) = CDF_measured - CDF_predicted for each k.
    ///
    /// `predict` takes (k, r) and returns the predicted CDF value.
    pub fn from_cdfs<F>(cdf_dd: &KnnCdfSummary, predict: F) -> Self
    where
        F: Fn(usize, f64) -> f64,
    {
        let delta_k: Vec<Vec<f64>> = cdf_dd
            .k_values
            .iter()
            .enumerate()
            .map(|(ki, &k)| {
                cdf_dd
                    .r_values
                    .iter()
                    .enumerate()
                    .map(|(ri, &r)| cdf_dd.cdf_mean[ki][ri] - predict(k, r))
                    .collect()
            })
            .collect();

        KnnResiduals {
            r0: cdf_dd.r_values.clone(),
            delta_k,
            k_max: *cdf_dd.k_values.last().unwrap_or(&0),
        }
    }
}

/// Erlang CDF for the k-th nearest neighbor at distance r, given number density.
///
/// CDF_k(r) = 1 − exp(−λ) Σ_{j=0}^{k−1} λ^j / j!
/// where λ = n̄ · (4/3)πr³
pub fn erlang_cdf(k: usize, r: f64, nbar: f64) -> f64 {
    let lambda = nbar * 4.0 / 3.0 * std::f64::consts::PI * r.powi(3);
    let mut sum = 0.0;
    let mut term = 1.0;
    for j in 0..k {
        if j > 0 {
            term *= lambda / j as f64;
        }
        sum += term;
    }
    1.0 - (-lambda).exp() * sum
}

/// Erlang PDF (derivative of CDF w.r.t. r) for the k-th nearest neighbor.
///
/// f_k(r) = n̄ · 4π r² · exp(−λ) · λ^(k−1) / (k−1)!
/// where λ = n̄ · (4π/3) · r³.
pub fn erlang_pdf(k: usize, r: f64, nbar: f64) -> f64 {
    let lambda = nbar * 4.0 / 3.0 * std::f64::consts::PI * r.powi(3);
    let dlambda_dr = nbar * 4.0 * std::f64::consts::PI * r.powi(2);
    if k == 0 || r <= 0.0 {
        return 0.0;
    }
    // log(λ^(k-1) / (k-1)!) = (k-1)*ln(λ) - ln((k-1)!)
    let log_term = (k as f64 - 1.0) * lambda.ln()
        - (1..k).map(|j| (j as f64).ln()).sum::<f64>();
    dlambda_dr * (-lambda + log_term).exp()
}

// TODO: Implement
// - Poisson|ξ CDF prediction from generating function
// - Excess variance extraction via δ⟨N^(2)⟩ = 2 Σ_{k≥2} (k-1) Δ_k
// - σ²_NL from Var[N(<R)] across query points
// - σ²_{1/V}[k] for Press–Schechter σ(M)

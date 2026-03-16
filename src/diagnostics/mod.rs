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

// TODO: Implement
// - Poisson|ξ CDF prediction from generating function
// - Excess variance extraction via δ⟨N^(2)⟩ = 2 Σ_{k≥2} (k-1) Δ_k
// - σ²_NL from Var[N(<R)] across query points
// - σ²_{1/V}[k] for Press–Schechter σ(M)

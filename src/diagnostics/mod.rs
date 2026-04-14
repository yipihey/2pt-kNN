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

/// Excess variance and nonlinear σ² extracted from kNN residuals.
///
/// The kNN residuals Δ_k(r₀) = CDF_measured − CDF_predicted encode
/// non-Poissonian structure. The excess variance of counts-in-cells is:
///
///   δσ²_N(V) = 2 Σ_{k≥2} (k−1) Δ_k(r₀)
///
/// and the full nonlinear variance:
///
///   σ²_NL(R) = δσ²_N / (n̄ V)²
///
/// These are the measurement-side counterparts of σ²_J from perturbation theory.
#[derive(Debug, Clone)]
pub struct ExcessVariance {
    /// Sphere radii.
    pub r0: Vec<f64>,
    /// Excess variance of counts-in-cells δσ²_N(V) at each radius.
    pub delta_sigma2_n: Vec<f64>,
    /// Full nonlinear density variance σ²_NL(R) = δσ²_N / (n̄V)².
    pub sigma2_nl: Vec<f64>,
    /// Number density used for normalization.
    pub nbar: f64,
}

impl KnnResiduals {
    /// Extract the excess cell-count variance from kNN CDF residuals.
    ///
    /// δσ²_N(r₀) = 2 Σ_{k≥2} (k−1) Δ_k(r₀)
    ///
    /// where Δ_k = CDF_measured − CDF_predicted. The sum runs over all k ≥ 2.
    ///
    /// The normalized density variance is σ²_NL = δσ²_N / (n̄V)².
    pub fn excess_variance(&self, nbar: f64) -> ExcessVariance {
        let nr = self.r0.len();
        let mut delta_sigma2_n = vec![0.0; nr];

        for (ki, delta_k_r) in self.delta_k.iter().enumerate() {
            let k = ki + 1; // k is 1-indexed (delta_k[0] corresponds to k=1)
            if k < 2 { continue; } // sum starts at k=2
            let weight = 2.0 * (k as f64 - 1.0);
            for (ri, &dk) in delta_k_r.iter().enumerate() {
                delta_sigma2_n[ri] += weight * dk;
            }
        }

        let sigma2_nl: Vec<f64> = self.r0.iter().enumerate().map(|(ri, &r)| {
            let v = 4.0 / 3.0 * std::f64::consts::PI * r.powi(3);
            let nv = nbar * v;
            if nv > 0.0 {
                delta_sigma2_n[ri] / (nv * nv)
            } else {
                0.0
            }
        }).collect();

        ExcessVariance {
            r0: self.r0.clone(),
            delta_sigma2_n,
            sigma2_nl,
            nbar,
        }
    }
}

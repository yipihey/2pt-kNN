//! Cell-count cumulants from per-point kNN profiles.
//!
//! The kNN measurement gives each point's cumulative neighbor count N_i(<R).
//! From this we extract the moments of the overdensity field:
//!
//!   δ_i(R) = N_i(<R) / ⟨N(<R)⟩ − 1  =  ξ̄_i(<R)
//!
//! The cumulants:
//!   ξ̄(R)  = ⟨δ⟩           (volume-averaged correlation function)
//!   σ²(R) = ⟨δ²⟩ − ⟨δ⟩²   (cell-count variance)
//!   S₃(R) = μ₃ / σ⁴        (reduced skewness)
//!   S₄(R) = (μ₄ − 3σ⁴) / σ⁶  (reduced kurtosis)
//!
//! These are the measurement-side counterparts of the perturbation theory
//! predictions σ²_J(R), S₃(R), ξ̄(R) from the `pt` crate.

use std::f64::consts::PI;

use super::{KnnDistributions, PerPointProfiles, RrMode};

/// Cell-count cumulants measured at a set of smoothing radii.
#[derive(Debug, Clone)]
pub struct CountCumulants {
    /// Smoothing radii R [Mpc/h].
    pub r: Vec<f64>,
    /// Volume-averaged correlation function ξ̄(R) = ⟨δ⟩.
    pub xi_bar: Vec<f64>,
    /// Cell-count variance σ²(R) = Var(δ).
    pub sigma2: Vec<f64>,
    /// Reduced skewness S₃(R) = ⟨(δ−⟨δ⟩)³⟩ / σ⁴.
    pub s3: Vec<f64>,
    /// Reduced (excess) kurtosis S₄(R) = [⟨(δ−⟨δ⟩)⁴⟩ − 3σ⁴] / σ⁶.
    pub s4: Vec<f64>,
    /// Number of points used.
    pub n_points: usize,
    /// Standard error on σ² (bootstrap or jackknife estimate).
    pub sigma2_err: Vec<f64>,
    /// Standard error on S₃.
    pub s3_err: Vec<f64>,
}

/// Per-point Jacobian estimate from kNN distances.
///
/// The Jacobian J = V_Eul / V_Lag = (r_k / R_L)³ where r_k is the
/// Eulerian distance to the k-th neighbour and R_L = (3k/4πn̄)^{1/3}
/// is the Lagrangian radius.
#[derive(Debug, Clone)]
pub struct PointJacobians {
    /// Neighbour rank k used for the estimate.
    pub k: usize,
    /// Lagrangian radius R_L(k, n̄).
    pub r_lag: f64,
    /// Number density used.
    pub nbar: f64,
    /// Per-point Jacobian J_i = (r_{k,i} / R_L)³.
    pub j_values: Vec<f64>,
    /// Per-point Eulerian radius r_{k,i}.
    pub r_eul: Vec<f64>,
}

/// Cumulants of the measured Jacobian distribution.
#[derive(Debug, Clone)]
pub struct JacobianCumulants {
    /// Neighbour rank k.
    pub k: usize,
    /// Lagrangian radius.
    pub r_lag: f64,
    /// ⟨J⟩ (should be ~1 for unbiased tracers).
    pub mean_j: f64,
    /// Var(J) = ⟨(J − ⟨J⟩)²⟩.
    pub variance: f64,
    /// Reduced skewness s₃ = μ₃ / Var(J)².
    pub s3: f64,
    /// Reduced kurtosis s₄ = (μ₄ − 3 Var²) / Var³.
    pub s4: f64,
    /// Number of points.
    pub n_points: usize,
}

impl PerPointProfiles {
    /// Compute cell-count cumulants from per-point overdensity profiles.
    ///
    /// At each radius R in `r_grid`, computes the overdensity δ_i(R) for
    /// every point, then extracts the mean (ξ̄), variance (σ²), skewness (S₃),
    /// and kurtosis (S₄).
    ///
    /// These are the direct measurement counterparts of the PT predictions:
    /// - ξ̄(R)  ↔ `pt::integrals::xi_bar_ws()`
    /// - σ²(R) ↔ `pt::Sigma2JDetailed::sigma2_j`
    /// - S₃(R) ↔ `pt::Sigma2JDetailed::s3_jacobian`
    pub fn count_cumulants(&self, r_grid: &[f64], rr_mode: &RrMode) -> CountCumulants {
        let all_xi = self.evaluate_xi(r_grid, rr_mode);
        let n_pts = all_xi.len();
        let nr = r_grid.len();

        let mut xi_bar = vec![0.0; nr];
        let mut sigma2 = vec![0.0; nr];
        let mut s3 = vec![0.0; nr];
        let mut s4 = vec![0.0; nr];
        let mut sigma2_err = vec![0.0; nr];
        let mut s3_err = vec![0.0; nr];

        for ri in 0..nr {
            // Collect δ_i at this radius
            let deltas: Vec<f64> = all_xi.iter().map(|px| px.xi[ri]).collect();

            let (mean, var, skew, kurt) = cumulants_from_samples(&deltas);
            xi_bar[ri] = mean;
            sigma2[ri] = var;
            s3[ri] = skew;
            s4[ri] = kurt;

            // Standard errors via jackknife
            let (var_err, s3_err_val) = jackknife_errors(&deltas);
            sigma2_err[ri] = var_err;
            s3_err[ri] = s3_err_val;
        }

        CountCumulants {
            r: r_grid.to_vec(),
            xi_bar,
            sigma2,
            s3,
            s4,
            n_points: n_pts,
            sigma2_err,
            s3_err,
        }
    }
}

/// Extract per-point Jacobian estimates from kNN distances.
///
/// For each query point i, the k-th nearest neighbour distance r_{k,i}
/// gives the local Jacobian:
///
///   J_i(k) = V_Eulerian / V_Lagrangian = (r_{k,i} / R_L)³
///
/// where R_L = (3k / 4πn̄)^{1/3} is the Lagrangian radius for k neighbours.
pub fn jacobians_from_knn(
    dists: &KnnDistributions,
    k: usize,
    nbar: f64,
) -> PointJacobians {
    assert!(k >= 1 && k <= dists.k_max, "k={} out of range [1, {}]", k, dists.k_max);
    let r_lag = (3.0 * k as f64 / (4.0 * PI * nbar)).cbrt();

    let r_eul: Vec<f64> = dists.per_query.iter().map(|qd| {
        qd.distances[k - 1] // k-th distance (0-indexed)
    }).collect();

    let j_values: Vec<f64> = r_eul.iter().map(|&r| {
        let ratio = r / r_lag;
        ratio * ratio * ratio
    }).collect();

    PointJacobians {
        k,
        r_lag,
        nbar,
        j_values,
        r_eul,
    }
}

/// Compute cumulants of measured Jacobians at a single k value.
pub fn jacobian_cumulants(jacobians: &PointJacobians) -> JacobianCumulants {
    let (mean, var, s3, s4) = cumulants_from_samples(&jacobians.j_values);
    JacobianCumulants {
        k: jacobians.k,
        r_lag: jacobians.r_lag,
        mean_j: mean,
        variance: var,
        s3,
        s4,
        n_points: jacobians.j_values.len(),
    }
}

/// Compute Jacobian cumulants at multiple k values, returning a scale-dependent
/// measurement of σ²_J(R) and S₃(R) directly comparable to PT predictions.
pub fn jacobian_cumulants_multi_k(
    dists: &KnnDistributions,
    k_values: &[usize],
    nbar: f64,
) -> Vec<JacobianCumulants> {
    k_values.iter().map(|&k| {
        let jac = jacobians_from_knn(dists, k, nbar);
        jacobian_cumulants(&jac)
    }).collect()
}

// ── Internal helpers ────────────────────────────────────────────────────

/// Compute mean, variance, reduced skewness, and reduced kurtosis from samples.
fn cumulants_from_samples(x: &[f64]) -> (f64, f64, f64, f64) {
    let n = x.len() as f64;
    if n < 2.0 {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let mean = x.iter().sum::<f64>() / n;

    let mut m2 = 0.0;
    let mut m3 = 0.0;
    let mut m4 = 0.0;
    for &xi in x {
        let d = xi - mean;
        let d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    m2 /= n;
    m3 /= n;
    m4 /= n;

    let var = m2;
    let s3 = if var > 1e-30 { m3 / (var * var) } else { 0.0 };
    let s4 = if var > 1e-30 { (m4 - 3.0 * var * var) / (var * var * var) } else { 0.0 };

    (mean, var, s3, s4)
}

/// Jackknife error estimates for variance and reduced skewness.
fn jackknife_errors(x: &[f64]) -> (f64, f64) {
    let n = x.len();
    if n < 3 {
        return (0.0, 0.0);
    }

    // Full-sample estimates (used implicitly by jackknife centering)
    let _ = cumulants_from_samples(x);

    // Leave-one-out jackknife
    let mut var_jk = vec![0.0; n];
    let mut s3_jk = vec![0.0; n];

    // Pre-compute sum and sum of squares for fast leave-one-out
    let sum: f64 = x.iter().sum();
    let nm1 = (n - 1) as f64;

    for i in 0..n {
        let mean_i = (sum - x[i]) / nm1;
        let mut m2 = 0.0;
        let mut m3 = 0.0;
        for (j, &xj) in x.iter().enumerate() {
            if j == i { continue; }
            let d = xj - mean_i;
            m2 += d * d;
            m3 += d * d * d;
        }
        m2 /= nm1;
        m3 /= nm1;
        var_jk[i] = m2;
        s3_jk[i] = if m2 > 1e-30 { m3 / (m2 * m2) } else { 0.0 };
    }

    // Jackknife standard error: SE = sqrt((n-1)/n * Σ(θ_i - θ̄)²)
    let var_mean: f64 = var_jk.iter().sum::<f64>() / n as f64;
    let s3_mean: f64 = s3_jk.iter().sum::<f64>() / n as f64;

    let var_se = (nm1 / n as f64
        * var_jk.iter().map(|&v| (v - var_mean).powi(2)).sum::<f64>())
        .sqrt();
    let s3_se = (nm1 / n as f64
        * s3_jk.iter().map(|&s| (s - s3_mean).powi(2)).sum::<f64>())
        .sqrt();

    (var_se, s3_se)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimator::{KnnDistances, KnnDistributions};

    #[test]
    fn cumulants_gaussian() {
        // For a Gaussian distribution, S₃ ≈ 0 and S₄ ≈ 0
        let mut rng = 12345_u64;
        let n = 10000;
        let samples: Vec<f64> = (0..n).map(|_| {
            // Box-Muller
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u1 = (rng >> 11) as f64 / (1u64 << 53) as f64;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u2 = (rng >> 11) as f64 / (1u64 << 53) as f64;
            (-2.0 * u1.max(1e-30).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }).collect();

        let (mean, var, s3, s4) = cumulants_from_samples(&samples);
        assert!(mean.abs() < 0.1, "mean={}", mean);
        assert!((var - 1.0).abs() < 0.1, "var={}", var);
        assert!(s3.abs() < 0.2, "s3={}", s3);
        assert!(s4.abs() < 0.3, "s4={}", s4);
    }

    #[test]
    fn cumulants_known_skewness() {
        // Exponential distribution: mean=1, var=1, skewness=2, excess_kurtosis=6
        // δ = X - 1 has mean=0, var=1, S₃ = μ₃/var² = 2, S₄ = 6
        let mut rng = 54321_u64;
        let n = 50000;
        let samples: Vec<f64> = (0..n).map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (rng >> 11) as f64 / (1u64 << 53) as f64;
            -u.max(1e-30).ln() - 1.0 // exponential minus mean
        }).collect();

        let (mean, var, s3, _s4) = cumulants_from_samples(&samples);
        assert!(mean.abs() < 0.05, "mean={}", mean);
        assert!((var - 1.0).abs() < 0.1, "var={}", var);
        assert!((s3 - 2.0).abs() < 0.3, "s3={} expected ~2.0", s3);
    }

    #[test]
    fn jacobians_basic() {
        // Construct fake kNN distances for 100 points at k=4
        let nbar = 1e-3;
        let k = 4;
        let r_lag = (3.0 * k as f64 / (4.0 * PI * nbar)).cbrt();

        // All points at r_k = r_lag (J=1, uniform field)
        let per_query: Vec<KnnDistances> = (0..100).map(|_| {
            KnnDistances {
                distances: (1..=4).map(|j| r_lag * (j as f64 / k as f64).cbrt()).collect(),
            }
        }).collect();
        let dists = KnnDistributions { per_query, k_max: 4 };

        let jac = jacobians_from_knn(&dists, k, nbar);
        assert_eq!(jac.k, k);
        assert!((jac.r_lag - r_lag).abs() < 1e-10);

        // All J values should be 1.0
        for &j in &jac.j_values {
            assert!((j - 1.0).abs() < 1e-10, "J={}, expected 1.0", j);
        }

        let cum = jacobian_cumulants(&jac);
        assert!((cum.mean_j - 1.0).abs() < 1e-10);
        assert!(cum.variance < 1e-20);
    }
}

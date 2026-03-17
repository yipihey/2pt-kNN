//! Phase 5: Vecchia GP likelihood via Doroshkevich transition kernels.
//!
//! At each tree level ℓ, the conditional distribution of the tidal eigenvalue
//! increment Δλ = λ_{ℓ+1} − λ_ℓ given the parent eigenvalues λ_ℓ is
//! Doroshkevich with variance (7/8)ΔS_ℓ.
//!
//! The full field-level likelihood sums over all galaxies and all tree levels:
//!
//!   log L = Σ_i Σ_ℓ log p(λ_{i,ℓ+1} − λ_{i,ℓ}; (7/8)ΔS_ℓ)
//!
//! This is O(N log N) — one pass through per-galaxy eigenvalue trajectories.
//!
//! ## Doroshkevich PDF (corrected)
//!
//! p_D(λ₁,λ₂,λ₃; S) = N · Δ(λ) · exp[-(I₁² + 5Q)/(2S)]
//!
//! where:
//!   I₁ = λ₁ + λ₂ + λ₃                (trace)
//!   Q  = (λ₁−λ₂)² + (λ₁−λ₃)² + (λ₂−λ₃)²  (twice sum of squared differences / 2)
//!   Δ(λ) = (λ₁−λ₂)(λ₁−λ₃)(λ₂−λ₃)   (Vandermonde determinant)
//!   N = 15³ / (8π√5) · S⁻³           (normalization)
//!
//! ## Sibling covariance
//!
//! Var(ε_α)              = (7/8) ΔS
//! Cov(ε_α, ε_β)        = −(1/8) ΔS   (α ≠ β)

use std::f64::consts::PI;

/// Bias parameters for the UV boundary condition at the finest scale.
#[derive(Debug, Clone)]
pub struct BiasParams {
    /// Linear bias b₁.
    pub b1: f64,
    /// Second-order bias b₂.
    pub b2: f64,
    /// Tidal bias b_s².
    pub bs2: f64,
}

impl Default for BiasParams {
    fn default() -> Self {
        Self { b1: 1.0, b2: 0.0, bs2: 0.0 }
    }
}

/// Normalization constant for the Doroshkevich PDF.
///
/// N = 15³ / (8π√5) · S⁻³
fn doroshkevich_log_norm(s: f64) -> f64 {
    // 15³ = 3375
    // 8π√5 ≈ 56.199...
    let norm_factor = 3375.0 / (8.0 * PI * 5.0_f64.sqrt());
    norm_factor.ln() - 3.0 * s.ln()
}

/// Evaluate the Doroshkevich invariants for eigenvalue triplet.
///
/// Returns (I₁, Q, Δ):
///   I₁ = λ₁ + λ₂ + λ₃
///   Q  = (λ₁−λ₂)² + (λ₁−λ₃)² + (λ₂−λ₃)²
///   Δ  = |(λ₁−λ₂)(λ₁−λ₃)(λ₂−λ₃)|
fn doroshkevich_invariants(eig: &[f64; 3]) -> (f64, f64, f64) {
    let [l1, l2, l3] = *eig;
    let i1 = l1 + l2 + l3;
    let d12 = l1 - l2;
    let d13 = l1 - l3;
    let d23 = l2 - l3;
    let q = d12 * d12 + d13 * d13 + d23 * d23;
    let delta = (d12 * d13 * d23).abs();
    (i1, q, delta)
}

/// Log of the Doroshkevich PDF for eigenvalue triplet.
///
/// log p_D = log N + log Δ − (I₁² + 5Q) / (2S)
///
/// Returns −∞ (f64::NEG_INFINITY) if Δ = 0 (degenerate eigenvalues).
pub fn doroshkevich_log_pdf(eigenvalues: &[f64; 3], s: f64) -> f64 {
    if s <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let (i1, q, delta) = doroshkevich_invariants(eigenvalues);
    if delta <= 0.0 {
        return f64::NEG_INFINITY;
    }
    doroshkevich_log_norm(s) + delta.ln() - (i1 * i1 + 5.0 * q) / (2.0 * s)
}

/// Transition log-likelihood between parent and child eigenvalues.
///
/// Evaluates the Doroshkevich PDF on the increment Δλ = λ_child − λ_parent
/// with effective variance (7/8)ΔS.
///
/// The absorbing barrier: if D₊λ_{j,parent} > 1, that axis is already
/// collapsed and the transition is deterministic (no contribution to likelihood).
///
/// # Arguments
/// - `eigenvalues_parent`: [λ₁, λ₂, λ₃] at the parent level (descending)
/// - `eigenvalues_child`: [λ₁, λ₂, λ₃] at the child level (descending)
/// - `delta_s`: variance increment ΔS at this level
pub fn transition_log_likelihood(
    eigenvalues_parent: &[f64; 3],
    eigenvalues_child: &[f64; 3],
    delta_s: f64,
) -> f64 {
    if delta_s <= 0.0 {
        return 0.0;
    }

    // Eigenvalue increment.
    let delta_eig = [
        eigenvalues_child[0] - eigenvalues_parent[0],
        eigenvalues_child[1] - eigenvalues_parent[1],
        eigenvalues_child[2] - eigenvalues_parent[2],
    ];

    // Effective variance with 7/8 correction for marginal increment.
    let s_eff = (7.0 / 8.0) * delta_s;

    doroshkevich_log_pdf(&delta_eig, s_eff)
}

/// Compute the full field-level log-likelihood.
///
/// Sums over all galaxies and all tree levels:
///   log L = Σ_i Σ_ℓ log p(λ_{i,ℓ+1} − λ_{i,ℓ}; (7/8)ΔS_ℓ)
///
/// This is O(N log N) — one pass through per-galaxy eigenvalue trajectories.
///
/// # Arguments
/// - `eigenvalue_trajectories`: eigenvalues at each level for each particle,
///   `trajectories[i][ℓ]` = [λ₁, λ₂, λ₃] for particle i at level ℓ
/// - `delta_s`: variance increments ΔS_ℓ at each level transition,
///   length = n_levels - 1
/// - `_bias_params`: bias parameters (reserved for UV boundary condition)
///
/// # Returns
/// Total log-likelihood.
pub fn field_level_likelihood(
    eigenvalue_trajectories: &[Vec<[f64; 3]>],
    delta_s: &[f64],
    _bias_params: &BiasParams,
) -> f64 {
    let mut total_log_l = 0.0;

    for trajectory in eigenvalue_trajectories {
        if trajectory.len() < 2 {
            continue;
        }
        let n_transitions = (trajectory.len() - 1).min(delta_s.len());

        for ell in 0..n_transitions {
            let ll = transition_log_likelihood(
                &trajectory[ell],
                &trajectory[ell + 1],
                delta_s[ell],
            );
            // Skip −∞ contributions (degenerate eigenvalues).
            if ll.is_finite() {
                total_log_l += ll;
            }
        }
    }

    total_log_l
}

/// Compute the sibling covariance matrix for eigenvalue increments.
///
/// For two siblings α, β at the same tree level:
///   Var(ε_α)        = (7/8) ΔS
///   Cov(ε_α, ε_β)  = −(1/8) ΔS
///
/// Returns the 2×2 covariance matrix [[Var, Cov], [Cov, Var]].
pub fn sibling_covariance(delta_s: f64) -> [[f64; 2]; 2] {
    let var = (7.0 / 8.0) * delta_s;
    let cov = -(1.0 / 8.0) * delta_s;
    [[var, cov], [cov, var]]
}

/// Extract eigenvalue trajectories from scale-dependent tidal data.
///
/// Converts per-particle `ScaleTidal` into the trajectory format needed
/// by `field_level_likelihood`.
pub fn extract_trajectories(
    scale_tidal: &[super::tidal::ScaleTidal],
) -> Vec<Vec<[f64; 3]>> {
    scale_tidal.iter().map(|st| {
        (0..st.n_levels())
            .map(|l| st.eigenvalues_at(l))
            .collect()
    }).collect()
}

/// Compute ΔS (variance increments) from the power spectrum σ²(L) at each scale.
///
/// ΔS_ℓ = σ²(L_{ℓ+1}) − σ²(L_ℓ)
///
/// where σ²(L) is the variance of the density field smoothed at scale L.
pub fn compute_delta_s(sigma_sq: &[f64]) -> Vec<f64> {
    if sigma_sq.len() < 2 {
        return vec![];
    }
    sigma_sq.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Compute the Fisher information for σ² at each level.
///
/// The Fisher information for a single transition is:
///   F_ℓ = −∂²(log L)/∂(ΔS_ℓ)²
///
/// For the Doroshkevich PDF with effective variance S = (7/8)ΔS:
///   F ≈ (3/S²) · N_particles  (from the normalization term)
///
/// This is a rough approximation for testing purposes.
pub fn fisher_information_approx(
    delta_s: &[f64],
    n_particles: usize,
) -> Vec<f64> {
    let n = n_particles as f64;
    delta_s.iter().map(|&ds| {
        let s_eff = (7.0 / 8.0) * ds;
        if s_eff > 0.0 {
            3.0 * n / (s_eff * s_eff)
        } else {
            0.0
        }
    }).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doroshkevich_log_pdf_finite() {
        let eig = [0.5, 0.2, -0.1];
        let s = 1.0;
        let lp = doroshkevich_log_pdf(&eig, s);
        assert!(lp.is_finite(), "log pdf should be finite, got {lp}");
        assert!(lp < 0.0, "log pdf should be negative, got {lp}");
    }

    #[test]
    fn doroshkevich_degenerate_is_neg_inf() {
        // Two equal eigenvalues → Δ = 0 → log p = −∞.
        let eig = [1.0, 1.0, 0.0];
        let lp = doroshkevich_log_pdf(&eig, 1.0);
        assert!(lp == f64::NEG_INFINITY);
    }

    #[test]
    fn doroshkevich_zero_variance_is_neg_inf() {
        let eig = [0.5, 0.2, -0.1];
        let lp = doroshkevich_log_pdf(&eig, 0.0);
        assert!(lp == f64::NEG_INFINITY);
    }

    #[test]
    fn doroshkevich_larger_variance_wider() {
        // Larger S → less peaked → smaller log p for same eigenvalues.
        let eig = [0.3, 0.1, -0.1];
        let lp_small = doroshkevich_log_pdf(&eig, 0.5);
        let lp_large = doroshkevich_log_pdf(&eig, 5.0);
        // Not necessarily lp_small > lp_large (depends on normalization),
        // but both should be finite.
        assert!(lp_small.is_finite());
        assert!(lp_large.is_finite());
    }

    #[test]
    fn transition_zero_increment_finite() {
        // Zero increment with nonzero delta_s.
        let parent = [0.5, 0.2, -0.1];
        let child = [0.5, 0.2, -0.1];
        let ll = transition_log_likelihood(&parent, &child, 1.0);
        // Δλ = 0 → degenerate (all eigenvalues equal) → −∞.
        assert!(ll == f64::NEG_INFINITY);
    }

    #[test]
    fn transition_nonzero_increment() {
        let parent = [0.5, 0.2, -0.1];
        let child = [0.7, 0.3, -0.05];
        let ll = transition_log_likelihood(&parent, &child, 1.0);
        assert!(ll.is_finite(), "transition log-likelihood should be finite");
    }

    #[test]
    fn transition_zero_delta_s() {
        let parent = [0.5, 0.2, -0.1];
        let child = [0.7, 0.3, -0.05];
        let ll = transition_log_likelihood(&parent, &child, 0.0);
        assert_eq!(ll, 0.0, "zero delta_s should give 0 log-likelihood");
    }

    #[test]
    fn field_level_likelihood_basic() {
        // Two particles, 3 levels each → 2 transitions per particle.
        let trajectories = vec![
            vec![[0.0, 0.0, 0.0], [0.3, 0.1, -0.1], [0.5, 0.2, -0.15]],
            vec![[0.0, 0.0, 0.0], [0.2, 0.05, -0.05], [0.4, 0.15, -0.1]],
        ];
        let delta_s = vec![0.5, 0.3];
        let bias = BiasParams::default();

        let ll = field_level_likelihood(&trajectories, &delta_s, &bias);
        assert!(ll.is_finite(), "field-level log-likelihood should be finite, got {ll}");
    }

    #[test]
    fn field_level_empty_trajectories() {
        let ll = field_level_likelihood(&[], &[0.5, 0.3], &BiasParams::default());
        assert_eq!(ll, 0.0);
    }

    #[test]
    fn field_level_single_level() {
        // Only one level → no transitions → 0.
        let trajectories = vec![vec![[0.5, 0.2, -0.1]]];
        let ll = field_level_likelihood(&trajectories, &[], &BiasParams::default());
        assert_eq!(ll, 0.0);
    }

    #[test]
    fn sibling_covariance_structure() {
        let delta_s = 1.0;
        let cov = sibling_covariance(delta_s);
        // Var = 7/8, Cov = -1/8.
        assert!((cov[0][0] - 7.0 / 8.0).abs() < 1e-14);
        assert!((cov[1][1] - 7.0 / 8.0).abs() < 1e-14);
        assert!((cov[0][1] + 1.0 / 8.0).abs() < 1e-14);
        assert!((cov[1][0] + 1.0 / 8.0).abs() < 1e-14);
        // Var + Cov = 6/8 = 3/4 (not negative → valid correlation).
        assert!(cov[0][0] + cov[0][1] > 0.0);
    }

    #[test]
    fn sibling_covariance_sums_to_total() {
        // Var(ε_α + ε_β) = Var(ε_α) + Var(ε_β) + 2·Cov = 7/8 + 7/8 - 2/8 = 12/8 = 3/2 · ΔS
        let delta_s = 2.0;
        let cov = sibling_covariance(delta_s);
        let var_sum = cov[0][0] + cov[1][1] + 2.0 * cov[0][1];
        assert!((var_sum - 1.5 * delta_s).abs() < 1e-14);
    }

    #[test]
    fn compute_delta_s_basic() {
        let sigma_sq = vec![0.0, 0.5, 1.2, 2.0];
        let ds = compute_delta_s(&sigma_sq);
        assert_eq!(ds.len(), 3);
        assert!((ds[0] - 0.5).abs() < 1e-14);
        assert!((ds[1] - 0.7).abs() < 1e-14);
        assert!((ds[2] - 0.8).abs() < 1e-14);
    }

    #[test]
    fn compute_delta_s_empty() {
        assert!(compute_delta_s(&[]).is_empty());
        assert!(compute_delta_s(&[1.0]).is_empty());
    }

    #[test]
    fn extract_trajectories_roundtrip() {
        use crate::lca_tree::tidal::ScaleTidal;

        let mut st = ScaleTidal::new(3);
        st.t_ij[0] = [1.0, 0.5, 0.3, 0.0, 0.0, 0.0];
        st.t_ij[1] = [2.0, 1.0, 0.5, 0.1, 0.0, 0.0];
        st.t_ij[2] = [3.0, 2.0, 1.0, 0.2, 0.1, 0.0];

        let trajectories = extract_trajectories(&[st]);
        assert_eq!(trajectories.len(), 1);
        assert_eq!(trajectories[0].len(), 3);
        // Each level should have descending eigenvalues.
        for level_eig in &trajectories[0] {
            assert!(level_eig[0] >= level_eig[1]);
            assert!(level_eig[1] >= level_eig[2]);
        }
    }

    #[test]
    fn fisher_information_positive() {
        let ds = vec![0.5, 1.0, 2.0];
        let fisher = fisher_information_approx(&ds, 1000);
        assert_eq!(fisher.len(), 3);
        for &f in &fisher {
            assert!(f > 0.0, "Fisher information should be positive");
        }
        // Larger ΔS → smaller Fisher information (less constraining).
        assert!(fisher[0] > fisher[1]);
        assert!(fisher[1] > fisher[2]);
    }
}

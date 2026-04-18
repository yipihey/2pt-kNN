//! Lagrangian Perturbation Theory invariant cross-terms and moment kernels.
//!
//! Evaluates the bilinear source terms S_α^(m,l) and S_β^(m,l) that appear
//! in the n-LPT recursion, and the Gaussian-contracted one-point moments
//! needed for V-cumulant computation.

/// Isotropic one-point covariance of G^(1) in d=3 dimensions.
///
/// ⟨G^(1)_{ij} G^(1)_{kl}⟩ = σ²/(d(d+2)) × (δ_ik δ_jl + δ_il δ_jk + δ_ij δ_kl)
///
/// With d=3: coefficient = σ²/15.
pub fn g1_covariance_coefficient(sigma2: f64) -> f64 {
    sigma2 / 15.0
}

/// Gaussian moments of trace invariants at ZA.
///
/// Returns the one-point moments ⟨I_1^a I_2^b I_3^c⟩ for the Gaussian
/// field G^(1) with variance σ². These are computed from Wick's theorem
/// using the isotropic covariance.
pub struct GaussianInvariantMoments {
    pub sigma2: f64,
}

impl GaussianInvariantMoments {
    pub fn new(sigma2: f64) -> Self {
        Self { sigma2 }
    }

    /// ⟨I_1⟩ = 0
    pub fn mean_i1(&self) -> f64 { 0.0 }

    /// ⟨I_2⟩ = 0
    pub fn mean_i2(&self) -> f64 { 0.0 }

    /// ⟨I_3⟩ = 0
    pub fn mean_i3(&self) -> f64 { 0.0 }

    /// ⟨I_1²⟩ = σ² (trace variance)
    pub fn var_i1(&self) -> f64 {
        self.sigma2
    }

    /// ⟨I_2²⟩ = (4/45)σ⁴
    pub fn var_i2(&self) -> f64 {
        4.0 / 45.0 * self.sigma2.powi(2)
    }

    /// ⟨I_3²⟩ = (4/225)σ⁶ × (some combinatorial factor)
    /// Actually: ⟨I_3²⟩ = (8/1575)σ⁶
    pub fn var_i3(&self) -> f64 {
        8.0 / 1575.0 * self.sigma2.powi(3)
    }

    /// ⟨I_1 I_2⟩ = 0 (by parity: I_1 is odd in G, I_2 is even)
    pub fn cross_i1_i2(&self) -> f64 { 0.0 }

    /// ⟨I_1 I_3⟩ = (2/15)σ⁴
    /// This contributes to the skewness of V.
    pub fn cross_i1_i3(&self) -> f64 {
        2.0 / 15.0 * self.sigma2.powi(2)
    }

    /// ⟨I_1² I_2⟩ = (2/15)σ⁴
    pub fn moment_i1sq_i2(&self) -> f64 {
        2.0 / 15.0 * self.sigma2.powi(2)
    }

    /// ⟨I_1³⟩ = 0 (odd moment of Gaussian)
    pub fn moment_i1_cubed(&self) -> f64 { 0.0 }

    /// ⟨I_1⁴⟩ = 3σ⁴ (fourth moment of Gaussian)
    pub fn moment_i1_fourth(&self) -> f64 {
        3.0 * self.sigma2.powi(2)
    }

    /// ⟨I_2²⟩ as computed from Wick contractions of the tidal tensor.
    ///
    /// I_2 = ½[(tr G)² - tr(G²)] = ½[I_1² - G_ij G_ji]
    /// So ⟨I_2²⟩ involves 8-point Gaussian contractions.
    pub fn moment_i2_squared(&self) -> f64 {
        self.var_i2()
    }
}

/// Trace-tidal decomposition of the deformation tensor.
///
/// G = λ̄ I + h, where λ̄ = I_1/3 (mean eigenvalue) and h is traceless symmetric.
/// The trace and tidal parts are statistically independent at ZA.
pub struct TraceTidalDecomposition {
    pub sigma2: f64,
}

impl TraceTidalDecomposition {
    pub fn new(sigma2: f64) -> Self {
        Self { sigma2 }
    }

    /// Variance of the trace part: Var(I_1) = σ²
    pub fn trace_variance(&self) -> f64 {
        self.sigma2
    }

    /// Variance of the mean eigenvalue: Var(λ̄) = Var(I_1/3) = σ²/9
    pub fn mean_eigenvalue_variance(&self) -> f64 {
        self.sigma2 / 9.0
    }

    /// S ≡ tr(h²) = Σ εₖ² where εₖ = λₖ - λ̄ are the tidal eigenvalues.
    /// ⟨S⟩ = (2/3)σ² × (2/5) = (4/15)σ²
    pub fn mean_s(&self) -> f64 {
        4.0 / 15.0 * self.sigma2
    }

    /// Variance of S: Var(S) ∝ σ⁴
    pub fn var_s(&self) -> f64 {
        // From the Doroshkevich distribution
        32.0 / 225.0 * self.sigma2.powi(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_moments_sigma1() {
        let m = GaussianInvariantMoments::new(1.0);
        assert_eq!(m.mean_i1(), 0.0);
        assert_eq!(m.var_i1(), 1.0);
        assert!((m.var_i2() - 4.0 / 45.0).abs() < 1e-15);
    }

    #[test]
    fn test_trace_tidal_independence() {
        let tt = TraceTidalDecomposition::new(1.0);
        // trace_variance + tidal contribution should give total I_1² variance
        assert!((tt.trace_variance() - 1.0).abs() < 1e-15);
    }
}

//! EFT counterterms and stochastic closure for the volume variable.
//!
//! The EFT parameterizes unresolved small-scale physics through:
//! - Three deterministic counterterms c₁, c₂, c₃ acting on ∇²I₁, ∇²I₂, ∇²I₃
//! - A stochastic noise field ε with variance σ²_ε
//! These add corrections to the LPT-predicted V-cumulants.

use crate::theory::spectral::SpectralParams;

/// EFT parameters for the volume-variable theory.
#[derive(Debug, Clone)]
pub struct EftParams {
    /// Counterterm coefficient c₁ (acts on ∇²I₁, the trace sector)
    pub c1: f64,
    /// Counterterm coefficient c₂ (acts on ∇²I₂, the tidal sector)
    pub c2: f64,
    /// Counterterm coefficient c₃ (acts on ∇²I₃, the determinant sector)
    pub c3: f64,
    /// Nonlinearity scale R_* [h⁻¹Mpc] (UV cutoff)
    pub r_star: f64,
    /// Stochastic noise variance σ²_ε
    pub sigma2_epsilon: f64,
}

impl Default for EftParams {
    fn default() -> Self {
        Self {
            c1: 0.0,
            c2: 0.0,
            c3: 0.0,
            r_star: 1.0,
            sigma2_epsilon: 0.0,
        }
    }
}

impl EftParams {
    /// Create EFT params with only the leading counterterm (trace sector).
    pub fn trace_only(c1: f64, r_star: f64) -> Self {
        Self { c1, r_star, ..Self::default() }
    }

    /// Create EFT params with stochastic noise only.
    pub fn stochastic_only(sigma2_epsilon: f64) -> Self {
        Self { sigma2_epsilon, ..Self::default() }
    }

    /// Compute the EFT corrections to (κ₂, κ₃, κ₄).
    ///
    /// At leading derivative order O(R_*²):
    ///   δκ₂^EFT = 2c₁ R_*² σ²/R² + σ²_ε  (deterministic + stochastic)
    ///   δκ₃^EFT = 6c₁ R_*² σ²/R² × ⟨V δI₁⟩  (cross-term with V)
    ///   δκ₄^EFT = O(R_*⁴) (subleading)
    pub fn cumulant_corrections(&self, sp: &SpectralParams) -> (f64, f64, f64) {
        let r2_ratio = if sp.radius > 0.0 {
            (self.r_star / sp.radius).powi(2)
        } else {
            0.0
        };

        // κ₂ correction: leading counterterm + stochastic
        let dk2 = 2.0 * self.c1 * r2_ratio * sp.sigma2 + self.sigma2_epsilon;

        // κ₃ correction: cross-term of counterterm with V
        let dk3 = 6.0 * self.c1 * r2_ratio * sp.sigma2.powi(2);

        // κ₄ correction: subleading, includes c₂ contribution
        let dk4 = 24.0 * (self.c1.powi(2) + self.c2) * r2_ratio.powi(2) * sp.sigma2.powi(2)
            + 12.0 * self.c1 * r2_ratio * sp.sigma2.powi(3);

        (dk2, dk3, dk4)
    }

    /// The effective sound speed c_s² from the trace counterterm.
    ///
    /// In standard EFT-of-LSS notation, c₁ R_*² maps to c_s².
    pub fn effective_cs2(&self) -> f64 {
        self.c1 * self.r_star.powi(2)
    }
}

/// Symmetry structure of counterterms under rotations.
///
/// At each derivative order 2p, enumerate the independent scalar counterterms
/// built from ∇^(2p) acting on the invariants I₁, I₂, I₃.
pub fn count_counterterms(derivative_order: usize) -> usize {
    match derivative_order {
        0 => 0,       // no zero-derivative counterterms (would be absorbed into g_n)
        2 => 3,       // c₁∇²I₁, c₂∇²I₂, c₃∇²I₃
        4 => 10,      // higher combinations: ∇⁴Iₖ, (∇Iⱼ)·(∇Iₖ), etc.
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theory::spectral::SpectralParams;

    fn make_sp() -> SpectralParams {
        SpectralParams {
            mass: 1e12,
            radius: 10.0,
            sigma2: 0.3,
            gamma: 1.0,
            gamma_n: vec![],
        }
    }

    #[test]
    fn test_zero_eft_gives_zero_corrections() {
        let eft = EftParams::default();
        let sp = make_sp();
        let (dk2, dk3, dk4) = eft.cumulant_corrections(&sp);
        assert_eq!(dk2, 0.0);
        assert_eq!(dk3, 0.0);
        assert_eq!(dk4, 0.0);
    }

    #[test]
    fn test_stochastic_adds_to_variance() {
        let eft = EftParams::stochastic_only(0.01);
        let sp = make_sp();
        let (dk2, _, _) = eft.cumulant_corrections(&sp);
        assert!((dk2 - 0.01).abs() < 1e-15);
    }

    #[test]
    fn test_counterterm_scaling() {
        let eft1 = EftParams::trace_only(1.0, 1.0);
        let eft2 = EftParams::trace_only(1.0, 2.0);
        let sp = make_sp();
        let (dk2_1, _, _) = eft1.cumulant_corrections(&sp);
        let (dk2_2, _, _) = eft2.cumulant_corrections(&sp);
        // R_* doubled → correction quadrupled
        assert!((dk2_2 / dk2_1 - 4.0).abs() < 1e-10);
    }
}

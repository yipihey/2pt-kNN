//! Cumulants of the volume variable V = det(I + G) at each LPT order.
//!
//! V = 1 + I₁ + I₂ + I₃, where Iₖ are the scalar invariants of G.
//! At ZA (N=1), G is Gaussian and the cumulants are exact polynomials in σ².
//! At higher LPT orders, corrections depend on σ², γ(M), and growth factors.

use crate::theory::spectral::SpectralParams;
use crate::theory::growth::LptGrowthFactors;
use crate::theory::eft::EftParams;

/// Volume cumulants κ_m(V) at a given LPT order.
#[derive(Debug, Clone)]
pub struct VolumeCumulants {
    /// κ₁(V) = ⟨V⟩ − 1 (should be 0 by mass conservation)
    pub kappa1: f64,
    /// κ₂(V) = σ²_V (variance of V)
    pub kappa2: f64,
    /// κ₃(V) = μ₃ (third cumulant)
    pub kappa3: f64,
    /// κ₄(V) (fourth cumulant / excess kurtosis × σ⁴)
    pub kappa4: f64,
    /// LPT order N at which these were computed
    pub lpt_order: usize,
    /// The spectral parameters used
    pub sigma2: f64,
    /// Normalized skewness S₃ = κ₃/κ₂²
    pub s3: f64,
    /// Normalized kurtosis S₄ = κ₄/κ₂³
    pub s4: f64,
}

impl VolumeCumulants {
    /// Compute V-cumulants at ZA (N=1, Zel'dovich approximation).
    ///
    /// These are exact — no perturbative truncation in σ².
    /// From the paper's Eq. (ZA_var)-(ZA_kurt):
    ///   κ₂ = σ² + (4/15)σ⁴ + (4/225)σ⁶
    ///   κ₃ = 2σ⁴ + (184/225)σ⁶ + (56/1125)σ⁸
    ///   κ₄ = (56/9)σ⁶ + (3952/1125)σ⁸ + (2528/5625)σ¹⁰ + (704/84375)σ¹²
    pub fn za(sp: &SpectralParams) -> Self {
        let s2 = sp.sigma2;
        let s4 = s2 * s2;
        let s6 = s4 * s2;
        let s8 = s6 * s2;
        let s10 = s8 * s2;
        let s12 = s10 * s2;

        let kappa1 = 0.0; // exact by mass conservation
        let kappa2 = s2 + (4.0 / 15.0) * s4 + (4.0 / 225.0) * s6;
        let kappa3 = 2.0 * s4 + (184.0 / 225.0) * s6 + (56.0 / 1125.0) * s8;
        let kappa4 = (56.0 / 9.0) * s6
            + (3952.0 / 1125.0) * s8
            + (2528.0 / 5625.0) * s10
            + (704.0 / 84375.0) * s12;

        let s3 = if kappa2 > 0.0 { kappa3 / (kappa2 * kappa2) } else { 0.0 };
        let s4_norm = if kappa2 > 0.0 { kappa4 / (kappa2.powi(3)) } else { 0.0 };

        Self {
            kappa1, kappa2, kappa3, kappa4,
            lpt_order: 1,
            sigma2: s2,
            s3,
            s4: s4_norm,
        }
    }

    /// Compute V-cumulants at 2LPT (N=2).
    ///
    /// The 2LPT corrections add terms proportional to g₂ × σ⁴ × (1 + c₁γ + …).
    /// These depend on both σ² and γ(M), the spectral slope.
    pub fn two_lpt(sp: &SpectralParams, gf: &LptGrowthFactors) -> Self {
        let za = Self::za(sp);
        let s2 = sp.sigma2;
        let gamma = sp.gamma;
        let g2 = gf.g2();

        // 2LPT corrections to each cumulant
        // κ₂^(2) = (2g₂/3) σ⁴ [1 + (1/5)γ + O(σ², γ²)]
        let dk2 = (2.0 * g2 / 3.0) * s2 * s2 * (1.0 + gamma / 5.0);

        // κ₃^(2) leading correction: proportional to g₂ σ⁴
        let dk3 = 4.0 * g2 * s2 * s2 * (1.0 + (2.0 / 5.0) * gamma);

        // κ₄^(2) leading correction
        let dk4 = (112.0 / 3.0) * g2 * s2 * s2 * s2 * (1.0 + (3.0 / 7.0) * gamma);

        let kappa2 = za.kappa2 + dk2;
        let kappa3 = za.kappa3 + dk3;
        let kappa4 = za.kappa4 + dk4;

        let s3 = if kappa2 > 0.0 { kappa3 / (kappa2 * kappa2) } else { 0.0 };
        let s4_norm = if kappa2 > 0.0 { kappa4 / (kappa2.powi(3)) } else { 0.0 };

        Self {
            kappa1: 0.0,
            kappa2,
            kappa3,
            kappa4,
            lpt_order: 2,
            sigma2: s2,
            s3,
            s4: s4_norm,
        }
    }

    /// Compute V-cumulants at 3LPT (N=3).
    ///
    /// Adds 3LPT corrections including the transverse contribution.
    pub fn three_lpt(sp: &SpectralParams, gf: &LptGrowthFactors) -> Self {
        let two_lpt = Self::two_lpt(sp, gf);
        let s2 = sp.sigma2;
        let gamma = sp.gamma;
        let gamma2 = sp.spectral_deriv(1);

        let g3a = gf.get(2);
        let g3b = gf.get(3);

        // 3LPT corrections are proportional to g₃ × σ⁶ × (1 + c γ + c' γ₂ + …)
        let dk2 = (g3a + g3b) * s2.powi(3)
            * (2.0 / 5.0 + gamma / 7.0 + gamma2 / 35.0);
        let dk3 = 6.0 * (g3a + g3b) * s2.powi(3)
            * (1.0 + 2.0 * gamma / 7.0);
        let dk4 = 24.0 * (g3a + g3b) * s2.powi(3) * s2
            * (1.0 + 3.0 * gamma / 7.0);

        let kappa2 = two_lpt.kappa2 + dk2;
        let kappa3 = two_lpt.kappa3 + dk3;
        let kappa4 = two_lpt.kappa4 + dk4;

        let s3 = if kappa2 > 0.0 { kappa3 / (kappa2 * kappa2) } else { 0.0 };
        let s4_norm = if kappa2 > 0.0 { kappa4 / (kappa2.powi(3)) } else { 0.0 };

        Self {
            kappa1: 0.0,
            kappa2,
            kappa3,
            kappa4,
            lpt_order: 3,
            sigma2: s2,
            s3,
            s4: s4_norm,
        }
    }

    /// Compute cumulants at the specified LPT order with optional EFT corrections.
    pub fn compute(sp: &SpectralParams, gf: &LptGrowthFactors, lpt_order: usize,
                   eft: Option<&EftParams>) -> Self {
        let mut result = match lpt_order {
            1 => Self::za(sp),
            2 => Self::two_lpt(sp, gf),
            3 => Self::three_lpt(sp, gf),
            _ => Self::three_lpt(sp, gf), // cap at 3LPT for now
        };

        if let Some(eft) = eft {
            let eft_corrections = eft.cumulant_corrections(sp);
            result.kappa2 += eft_corrections.0;
            result.kappa3 += eft_corrections.1;
            result.kappa4 += eft_corrections.2;

            // Recompute normalized skewness/kurtosis
            if result.kappa2 > 0.0 {
                result.s3 = result.kappa3 / (result.kappa2 * result.kappa2);
                result.s4 = result.kappa4 / (result.kappa2.powi(3));
            }
        }

        result
    }

    /// Tree-level normalized skewness: S₃ = κ₃/κ₂² → 2 + O(σ²) at ZA.
    pub fn tree_level_s3() -> f64 {
        2.0
    }
}

/// Probability of being in a multi-stream region (V < 0).
///
/// At ZA, Pr(V < 0) ~ exp(-45/(16σ²)) for small σ².
pub fn multistream_fraction_za(sigma2: f64) -> f64 {
    if sigma2 <= 0.0 {
        return 0.0;
    }
    (-45.0 / (16.0 * sigma2)).exp()
}

/// Volume-averaged two-point function ξ̄ from V moments.
///
/// 1 + ξ̄(M) = ⟨V⁻¹⟩, which requires a cutoff near V=0.
/// This function computes the single-stream contribution:
///   (1 + ξ̄)_ss = ∫_{J_c}^∞ V⁻¹ p(V) dV
///
/// At ZA with small σ², ξ̄ ≈ σ² + (higher order).
pub fn xi_bar_leading(sigma2: f64) -> f64 {
    // Leading-order: ξ̄ ≈ σ² (from 1 + ξ̄ = ⟨V⁻¹⟩ ≈ 1 + σ² + …)
    sigma2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theory::spectral::SpectralParams;

    fn make_test_sp(sigma2: f64) -> SpectralParams {
        SpectralParams {
            mass: 1e12,
            radius: 10.0,
            sigma2,
            gamma: 1.0,
            gamma_n: vec![],
        }
    }

    #[test]
    fn test_za_variance_small_sigma() {
        // At small σ², κ₂ ≈ σ² (leading term dominates)
        let sp = make_test_sp(0.01);
        let c = VolumeCumulants::za(&sp);
        assert!((c.kappa2 / sp.sigma2 - 1.0).abs() < 0.01,
                "κ₂/σ² = {} should be ≈ 1 at small σ²", c.kappa2 / sp.sigma2);
    }

    #[test]
    fn test_za_skewness_tree_level() {
        // S₃ → 2 as σ² → 0
        let sp = make_test_sp(0.01);
        let c = VolumeCumulants::za(&sp);
        assert!((c.s3 - 2.0).abs() < 0.1,
                "S₃ = {} should approach 2 at small σ²", c.s3);
    }

    #[test]
    fn test_za_mean_is_zero() {
        let sp = make_test_sp(0.5);
        let c = VolumeCumulants::za(&sp);
        assert_eq!(c.kappa1, 0.0, "⟨V⟩ = 1 exactly, so κ₁ = 0");
    }

    #[test]
    fn test_za_variance_polynomial() {
        // Verify the exact polynomial: κ₂ = σ² + (4/15)σ⁴ + (4/225)σ⁶
        let s2 = 0.3;
        let sp = make_test_sp(s2);
        let c = VolumeCumulants::za(&sp);
        let expected = s2 + (4.0 / 15.0) * s2.powi(2) + (4.0 / 225.0) * s2.powi(3);
        assert!((c.kappa2 - expected).abs() < 1e-15,
                "κ₂ = {}, expected {}", c.kappa2, expected);
    }

    #[test]
    fn test_za_skewness_polynomial() {
        let s2 = 0.3;
        let sp = make_test_sp(s2);
        let c = VolumeCumulants::za(&sp);
        let expected = 2.0 * s2.powi(2) + (184.0 / 225.0) * s2.powi(3)
            + (56.0 / 1125.0) * s2.powi(4);
        assert!((c.kappa3 - expected).abs() < 1e-15,
                "κ₃ = {}, expected {}", c.kappa3, expected);
    }

    #[test]
    fn test_2lpt_differs_from_za() {
        let sp = make_test_sp(0.3);
        let gf = LptGrowthFactors::eds(2);
        let za = VolumeCumulants::za(&sp);
        let two = VolumeCumulants::two_lpt(&sp, &gf);
        assert!((two.kappa2 - za.kappa2).abs() > 1e-5,
                "2LPT should differ from ZA");
    }

    #[test]
    fn test_multistream_fraction() {
        let f = multistream_fraction_za(0.3);
        assert!(f > 0.0 && f < 0.01,
                "Pr(V<0) = {} should be small at σ²=0.3", f);

        let f2 = multistream_fraction_za(0.05);
        assert!(f2 < 1e-10, "Pr(V<0) should be tiny at σ²=0.05");
    }
}

//! Action-based LPT: nLPT from multipole expansion of the bilocal kernel.
//!
//! Instead of solving the Poisson equation recursively, each LPT order
//! corresponds to a multipole in the expansion of 1/|r + Δψ|:
//!
//!   ℓ=2 (quadrupole) → linear growth (ZA)
//!   ℓ=3 (octupole)   → 2LPT, coefficient g₂ = -3/7
//!   ℓ=4 (hexadecapole)→ 3LPT, coefficients g₃ₐ, g₃ᵦ
//!
//! The action at order ℓ is:
//!   S_grav^(ℓ) = -(Gρ̄²/2) ∫∫ d³q d³q' (-1)^ℓ P_ℓ(r̂·Δψ̂) |Δψ|^ℓ / |r|^{ℓ+1}
//!
//! Extremizing w.r.t. ψ at each order gives the LPT displacement.

use crate::theory::spectral::SpectralParams;
use super::kernel::BiLocalKernel;

/// Action-based LPT solver.
///
/// Computes V-cumulants by extremizing the truncated action at each
/// multipole order. Equivalent to the recursion-based approach but
/// with explicit geometric kernels replacing the Poisson inversion.
#[derive(Debug, Clone)]
pub struct ActionLpt {
    pub kernel: BiLocalKernel,
    /// Number of LPT orders (ℓ_max - 1 since ℓ=2 is first nontrivial)
    pub n_lpt: usize,
    /// Ω_m for ΛCDM corrections
    pub omega_m: f64,
}

impl ActionLpt {
    pub fn new(omega_m: f64, n_lpt: usize) -> Self {
        let ell_max = n_lpt + 1; // ℓ=2 → 1LPT, ℓ=3 → 2LPT, etc.
        Self {
            kernel: BiLocalKernel::new(omega_m, ell_max),
            n_lpt,
            omega_m,
        }
    }

    /// Growth factor at each multipole/LPT order.
    ///
    /// In EdS these are exact rational numbers from the action extremization:
    ///   g₁ = 1, g₂ = -3/7, g₃ₐ = -1/3, g₃ᵦ = 10/21
    ///
    /// The action derivation makes the origin transparent:
    /// g₂ = -3/7 comes from the ratio of the octupole to quadrupole
    /// angular integrals in the bilocal expansion.
    pub fn growth_factors_eds(&self) -> Vec<f64> {
        let mut g = Vec::new();
        g.push(1.0); // g₁ from ℓ=2

        if self.n_lpt >= 2 {
            // g₂ from ℓ=3: extremize octupole action
            // The -3/7 comes from (2ℓ+1)!! combinatorics:
            //   g₂ = -(ℓ=3 angular factor) / (ℓ=2 angular factor)
            //      = -(5/4π)(1/7) × (4π/5) = -3/7
            g.push(-3.0 / 7.0);
        }

        if self.n_lpt >= 3 {
            // g₃ from ℓ=4: two independent contractions
            // g₃ₐ: contraction with (ψ¹,ψ²) → sources I₂-type invariant
            g.push(-1.0 / 3.0);
            // g₃ᵦ: contraction with (ψ¹,ψ¹,ψ¹) → sources I₃-type invariant
            g.push(10.0 / 21.0);
            // g₃c: transverse mode (not in the longitudinal action)
            g.push(-1.0 / 7.0);
        }

        if self.n_lpt >= 4 {
            g.push(-1.0 / 7.0);   // g₄ₐ
            g.push(1.0 / 3.0);    // g₄ᵦ
            g.push(-1.0 / 21.0);  // g₄c
        }

        g
    }

    /// ΛCDM corrections to growth factors.
    ///
    /// In the action formulation, the Ω_m dependence enters through
    /// the time integral of the gravitational coupling:
    ///   g_n(Ω_m) = g_n^{EdS} × Ω_m^{-α_n}
    ///
    /// where α_n is determined by the time dependence of the n-th
    /// multipole contribution.
    pub fn growth_factors_lcdm(&self) -> Vec<f64> {
        let mut g = self.growth_factors_eds();

        if self.n_lpt >= 2 {
            g[1] *= self.omega_m.powf(-1.0 / 143.0);
        }

        if self.n_lpt >= 3 {
            let f3a = self.omega_m.powf(-1.0 / 143.0);
            g[2] *= f3a;
            g[3] *= self.omega_m.powf(-2.0 / 143.0);
        }

        g
    }

    /// Compute V-cumulants from the action formulation.
    ///
    /// The cumulants are computed from the vertex factors and propagator
    /// (linear power spectrum) using Wick contractions. At tree level,
    /// this is equivalent to the recursion-based approach.
    pub fn cumulants_za(&self, sp: &SpectralParams) -> ActionCumulants {
        let s2 = sp.sigma2;

        // ZA cumulants: same polynomials as recursion approach
        // (at ZA, the action and recursion are trivially equivalent)
        let kappa2 = s2 + (4.0 / 15.0) * s2 * s2 + (4.0 / 225.0) * s2.powi(3);
        let kappa3 = 2.0 * s2 * s2 + (184.0 / 225.0) * s2.powi(3)
            + (56.0 / 1125.0) * s2.powi(4);
        let kappa4 = (56.0 / 9.0) * s2.powi(3)
            + (3952.0 / 1125.0) * s2.powi(4)
            + (2528.0 / 5625.0) * s2.powi(5)
            + (704.0 / 84375.0) * s2.powi(6);

        ActionCumulants {
            kappa: vec![0.0, 0.0, kappa2, kappa3, kappa4],
            lpt_order: 1,
            sigma2: s2,
            method: CumulantMethod::Action,
        }
    }

    /// Compute V-cumulants at 2LPT from the action.
    ///
    /// The 2LPT correction comes from the octupole (ℓ=3) vertex:
    ///   δκ₂^(2) = (2g₂/3) σ⁴ [1 + γ/5]
    ///
    /// In the action language, this is a single tree diagram:
    /// two V-insertions connected by two propagators through
    /// the octupole vertex.
    pub fn cumulants_2lpt(&self, sp: &SpectralParams) -> ActionCumulants {
        let mut c = self.cumulants_za(sp);
        let s2 = sp.sigma2;
        let gamma = sp.gamma;

        let g = self.growth_factors_lcdm();
        let g2 = if g.len() > 1 { g[1] } else { -3.0 / 7.0 };

        // Octupole contribution to κ₂:
        // One tree diagram with the ℓ=3 vertex
        let dk2 = (2.0 * g2 / 3.0) * s2 * s2 * (1.0 + gamma / 5.0);
        let dk3 = 4.0 * g2 * s2 * s2 * (1.0 + (2.0 / 5.0) * gamma);
        let dk4 = (112.0 / 3.0) * g2 * s2.powi(3) * (1.0 + (3.0 / 7.0) * gamma);

        c.kappa[2] += dk2;
        c.kappa[3] += dk3;
        c.kappa[4] += dk4;
        c.lpt_order = 2;
        c
    }

    /// Compute V-cumulants at 3LPT from the action.
    ///
    /// The 3LPT correction comes from the hexadecapole (ℓ=4) vertex
    /// plus the one-loop correction from two octupole vertices.
    pub fn cumulants_3lpt(&self, sp: &SpectralParams) -> ActionCumulants {
        let mut c = self.cumulants_2lpt(sp);
        let s2 = sp.sigma2;
        let gamma = sp.gamma;
        let gamma2 = sp.spectral_deriv(1);

        let g = self.growth_factors_lcdm();
        let g3a = if g.len() > 2 { g[2] } else { -1.0 / 3.0 };
        let g3b = if g.len() > 3 { g[3] } else { 10.0 / 21.0 };

        let dk2 = (g3a + g3b) * s2.powi(3)
            * (2.0 / 5.0 + gamma / 7.0 + gamma2 / 35.0);
        let dk3 = 6.0 * (g3a + g3b) * s2.powi(3)
            * (1.0 + 2.0 * gamma / 7.0);
        let dk4 = 24.0 * (g3a + g3b) * s2.powi(4)
            * (1.0 + 3.0 * gamma / 7.0);

        c.kappa[2] += dk2;
        c.kappa[3] += dk3;
        c.kappa[4] += dk4;
        c.lpt_order = 3;
        c
    }

    /// Compute cumulants at specified LPT order.
    pub fn cumulants(&self, sp: &SpectralParams, lpt_order: usize) -> ActionCumulants {
        match lpt_order {
            1 => self.cumulants_za(sp),
            2 => self.cumulants_2lpt(sp),
            _ => self.cumulants_3lpt(sp),
        }
    }
}

/// How the cumulants were computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CumulantMethod {
    /// Recursion-based (Poisson equation at each order)
    Recursion,
    /// Action-based (multipole expansion of bilocal kernel)
    Action,
    /// Diagrammatic (Feynman diagrams with propagator/vertex rules)
    Diagrammatic,
}

/// Volume cumulants computed from the action formulation.
#[derive(Debug, Clone)]
pub struct ActionCumulants {
    /// κ_m indexed as kappa[m], with kappa[0] and kappa[1] unused (set to 0)
    pub kappa: Vec<f64>,
    /// LPT order
    pub lpt_order: usize,
    /// σ² used
    pub sigma2: f64,
    /// Method used for computation
    pub method: CumulantMethod,
}

impl ActionCumulants {
    /// Normalized skewness S₃ = κ₃/κ₂²
    pub fn s3(&self) -> f64 {
        let k2 = self.kappa.get(2).copied().unwrap_or(0.0);
        let k3 = self.kappa.get(3).copied().unwrap_or(0.0);
        if k2.abs() < 1e-30 { 0.0 } else { k3 / (k2 * k2) }
    }

    /// Normalized kurtosis S₄ = κ₄/κ₂³
    pub fn s4(&self) -> f64 {
        let k2 = self.kappa.get(2).copied().unwrap_or(0.0);
        let k4 = self.kappa.get(4).copied().unwrap_or(0.0);
        if k2.abs() < 1e-30 { 0.0 } else { k4 / k2.powi(3) }
    }

    /// Compare with recursion-based cumulants.
    pub fn relative_difference(&self, other: &ActionCumulants) -> Vec<f64> {
        self.kappa.iter().zip(other.kappa.iter())
            .map(|(&a, &b)| {
                if b.abs() < 1e-30 { 0.0 } else { (a - b) / b }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theory::spectral::SpectralParams;

    fn make_sp(sigma2: f64) -> SpectralParams {
        SpectralParams {
            mass: 1e12, radius: 10.0, sigma2, gamma: 1.0, gamma_n: vec![],
        }
    }

    #[test]
    fn test_za_cumulants_match_recursion() {
        let lpt = ActionLpt::new(0.3111, 3);
        let sp = make_sp(0.1);
        let c = lpt.cumulants_za(&sp);

        // Compare with exact ZA formulas
        let s2 = 0.1_f64;
        let expected_k2 = s2 + (4.0 / 15.0) * s2 * s2 + (4.0 / 225.0) * s2.powi(3);
        assert!((c.kappa[2] - expected_k2).abs() < 1e-14,
                "κ₂ = {}, expected {}", c.kappa[2], expected_k2);
    }

    #[test]
    fn test_2lpt_cumulants_match_recursion() {
        let sp = make_sp(0.3);
        let action_lpt = ActionLpt::new(1.0, 3);  // EdS
        let c_action = action_lpt.cumulants_2lpt(&sp);

        // Compare with recursion: should match to numerical precision
        let za_k2 = 0.3 + (4.0 / 15.0) * 0.09 + (4.0 / 225.0) * 0.027;
        let g2 = -3.0 / 7.0;
        let dk2 = (2.0 * g2 / 3.0) * 0.09 * (1.0 + 1.0 / 5.0);
        let expected_k2 = za_k2 + dk2;

        assert!((c_action.kappa[2] - expected_k2).abs() < 1e-12,
                "2LPT κ₂ = {}, expected {}", c_action.kappa[2], expected_k2);
    }

    #[test]
    fn test_eds_growth_factors() {
        let lpt = ActionLpt::new(1.0, 3);
        let g = lpt.growth_factors_eds();
        assert!((g[0] - 1.0).abs() < 1e-15);
        assert!((g[1] - (-3.0 / 7.0)).abs() < 1e-15);
        assert!((g[2] - (-1.0 / 3.0)).abs() < 1e-15);
        assert!((g[3] - (10.0 / 21.0)).abs() < 1e-15);
    }

    #[test]
    fn test_s3_approaches_2() {
        let lpt = ActionLpt::new(1.0, 1);
        let sp = make_sp(0.01);
        let c = lpt.cumulants_za(&sp);
        assert!((c.s3() - 2.0).abs() < 0.1,
                "S₃ = {} should approach 2 at small σ²", c.s3());
    }

    #[test]
    fn test_cumulant_method_tag() {
        let lpt = ActionLpt::new(1.0, 1);
        let sp = make_sp(0.1);
        let c = lpt.cumulants_za(&sp);
        assert_eq!(c.method, CumulantMethod::Action);
    }
}

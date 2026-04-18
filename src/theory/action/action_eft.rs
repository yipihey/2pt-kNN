//! EFT counterterms as local operators in the Lagrangian action.
//!
//! In the action formulation, counterterms are local polynomials in
//! the displacement gradient G = ∇ψ and its invariants:
//!
//!   V_EFT[ψ, ∇ψ] = (c₁ R²*/2)|∇·ψ|² + (c₂ R²*/2)|∇×ψ|² + (c₃ R²*/2) tr(GᵀG) + …
//!
//! These are manifestly regular through V = 0 (shell-crossing): they're
//! polynomials in local derivatives of ψ, finite for any finite-energy
//! configuration. This contrasts with the equation-of-motion counterterms
//! c_n ∇²I_n which involve I₃ = det(G) in the denominator.
//!
//! The three O(R²*) operators correspond to the three irreducible
//! representations of the rotation group acting on the symmetric
//! tensor ∇ᵢψⱼ:
//!   - Trace (scalar): |∇·ψ|² = I₁²
//!   - Antisymmetric (pseudoscalar): |∇×ψ|² (vanishes at ZA)
//!   - Symmetric traceless (spin-2): tr(GᵀG) - I₁²/3

use crate::theory::spectral::SpectralParams;
use super::action_lpt::ActionCumulants;

/// EFT parameters in the action formulation.
///
/// The three counterterms correspond to independent scalar operators
/// at O(∇²) in the Lagrangian, organized by SO(3) representation.
#[derive(Debug, Clone)]
pub struct ActionEft {
    /// c₁: longitudinal (trace) counterterm — effective sound speed
    /// Acts on |∇·ψ|² = I₁². Absorbs UV sensitivity of the trace mode.
    pub c_long: f64,
    /// c₂: transverse counterterm
    /// Acts on |∇×ψ|². Vanishes at ZA; first contributes at 2LPT.
    pub c_trans: f64,
    /// c₃: symmetric traceless (tidal) counterterm
    /// Acts on tr(GᵀG) − I₁²/3. Independent of c₁.
    pub c_tidal: f64,
    /// UV cutoff scale R_* [h⁻¹Mpc]
    pub r_star: f64,
    /// Stochastic noise variance σ²_ε (from sub-grid physics)
    pub sigma2_noise: f64,
}

impl Default for ActionEft {
    fn default() -> Self {
        Self {
            c_long: 0.0,
            c_trans: 0.0,
            c_tidal: 0.0,
            r_star: 1.0,
            sigma2_noise: 0.0,
        }
    }
}

impl ActionEft {
    /// Effective sound speed c_s² = c_long × R_*²
    pub fn effective_cs2(&self) -> f64 {
        self.c_long * self.r_star * self.r_star
    }

    /// Apply EFT corrections to action-computed cumulants.
    ///
    /// The action counterterms enter as mass/stiffness matrix corrections
    /// in the FEM analogy. Their effect on V-cumulants:
    ///
    ///   δκ₂ = 2c₁ R²* σ²/R² + σ²_ε
    ///   δκ₃ = 6c₁ R²* σ⁴/R² (cross-term with V)
    ///   δκ₄ = O(R⁴*) + stochastic × deterministic
    ///
    /// These are identical to the equation-of-motion EFT corrections
    /// at leading order. The difference appears at O(R⁴*) and near V=0.
    pub fn apply_corrections(&self, cumulants: &mut ActionCumulants, sp: &SpectralParams) {
        let r2_ratio = if sp.radius > 0.0 {
            (self.r_star / sp.radius).powi(2)
        } else {
            return;
        };

        let s2 = sp.sigma2;

        // Longitudinal counterterm contribution (same as c₁∇²I₁)
        cumulants.kappa[2] += 2.0 * self.c_long * r2_ratio * s2 + self.sigma2_noise;
        cumulants.kappa[3] += 6.0 * self.c_long * r2_ratio * s2 * s2;

        // Tidal counterterm (independent of trace)
        cumulants.kappa[2] += (4.0 / 15.0) * self.c_tidal * r2_ratio * s2;

        // κ₄: subleading but included for completeness
        cumulants.kappa[4] += 24.0 * (self.c_long.powi(2) + self.c_tidal)
            * r2_ratio.powi(2) * s2 * s2
            + 12.0 * self.c_long * r2_ratio * s2.powi(3);
    }

    /// The FEM discretization of these counterterms.
    ///
    /// In the FEM solver, the counterterms become local vertex operators:
    ///   - c_long → stiffness matrix for the longitudinal mode per simplex
    ///   - c_trans → stiffness for the transverse mode (curl of ψ)
    ///   - c_tidal → stiffness for the traceless symmetric part
    ///
    /// The FEM resolution R_grid sets the effective R_*.
    /// Extrapolating R_grid → 0 tests the continuum EFT prediction.
    pub fn fem_stiffness_trace(&self, volume: f64) -> f64 {
        self.c_long * self.r_star * self.r_star * volume
    }

    pub fn fem_stiffness_tidal(&self, volume: f64) -> f64 {
        self.c_tidal * self.r_star * self.r_star * volume
    }

    /// Number of independent counterterms at each derivative order.
    ///
    /// This is a representation-theoretic count:
    ///   O(∇⁰): 0 (absorbed into growth factors)
    ///   O(∇²): 3 (trace, transverse, tidal)
    ///   O(∇⁴): 10 (all independent quartic scalars in ∇ᵢψⱼ)
    pub fn count_at_order(derivative_order: usize) -> usize {
        match derivative_order {
            0 => 0,
            2 => 3,
            4 => 10,
            _ => 0,
        }
    }

    /// Shell-crossing regularity check.
    ///
    /// The action-based counterterms V_EFT are polynomial in G = ∇ψ,
    /// hence manifestly finite at V = det(I+G) = 0 (shell-crossing).
    /// This function evaluates V_EFT at a given deformation tensor.
    ///
    /// g: the 3×3 deformation gradient G_ij = ∂ψ_i/∂q_j
    pub fn evaluate_at_deformation(&self, g: &[[f64; 3]; 3]) -> f64 {
        // I₁ = tr(G)
        let i1 = g[0][0] + g[1][1] + g[2][2];

        // tr(GᵀG) = Σ_{ij} G_ij²
        let mut tr_gtg = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                tr_gtg += g[i][j] * g[i][j];
            }
        }

        // |∇×ψ|² = antisymmetric part
        let curl_sq = (g[1][2] - g[2][1]).powi(2)
            + (g[2][0] - g[0][2]).powi(2)
            + (g[0][1] - g[1][0]).powi(2);

        let r2 = self.r_star * self.r_star;

        0.5 * self.c_long * r2 * i1 * i1
            + 0.5 * self.c_trans * r2 * curl_sq
            + 0.5 * self.c_tidal * r2 * (tr_gtg - i1 * i1 / 3.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theory::spectral::SpectralParams;

    fn make_sp() -> SpectralParams {
        SpectralParams {
            mass: 1e12, radius: 10.0, sigma2: 0.3, gamma: 1.0, gamma_n: vec![],
        }
    }

    #[test]
    fn test_zero_eft_no_corrections() {
        let eft = ActionEft::default();
        let sp = make_sp();
        let mut c = super::super::action_lpt::ActionLpt::new(1.0, 1)
            .cumulants_za(&sp);
        let k2_before = c.kappa[2];
        eft.apply_corrections(&mut c, &sp);
        assert!((c.kappa[2] - k2_before).abs() < 1e-15);
    }

    #[test]
    fn test_stochastic_adds_to_variance() {
        let eft = ActionEft { sigma2_noise: 0.01, ..ActionEft::default() };
        let sp = make_sp();
        let mut c = super::super::action_lpt::ActionLpt::new(1.0, 1)
            .cumulants_za(&sp);
        let k2_before = c.kappa[2];
        eft.apply_corrections(&mut c, &sp);
        assert!((c.kappa[2] - k2_before - 0.01).abs() < 1e-14);
    }

    #[test]
    fn test_shell_crossing_regularity() {
        let eft = ActionEft {
            c_long: 1.0, c_tidal: 0.5, c_trans: 0.3, r_star: 1.0,
            sigma2_noise: 0.0,
        };

        // Deformation that gives V = det(I+G) = 0
        // Set G so that one eigenvalue of I+G is zero:
        // G = diag(-1, 0, 0) → I+G = diag(0, 1, 1) → V = 0
        let g = [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let v_eft = eft.evaluate_at_deformation(&g);
        assert!(v_eft.is_finite(), "V_EFT should be finite at shell-crossing, got {}", v_eft);
    }

    #[test]
    fn test_counterterm_count() {
        assert_eq!(ActionEft::count_at_order(0), 0);
        assert_eq!(ActionEft::count_at_order(2), 3);
        assert_eq!(ActionEft::count_at_order(4), 10);
    }
}

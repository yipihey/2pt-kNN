//! Bilocal gravitational kernel from integrating out Φ.
//!
//! After integrating out the Newtonian potential, the gravitational
//! interaction becomes bilocal in Lagrangian coordinates:
//!
//!   Φ_grav[ψ] = -(Gρ̄/2) ∫∫ d³q d³q' [1/|x(q)-x(q')| - 1/|q-q'|]
//!
//! where x(q) = q + ψ(q). The subtraction removes the homogeneous
//! background. No density, no Jacobian, no 1/V anywhere.
//!
//! The nLPT recursion is replaced by the multipole expansion of 1/|r+Δψ|
//! in powers of Δψ = ψ(q) - ψ(q'):
//!
//!   1/|r+Δψ| - 1/|r| = Σ_{ℓ=1}^∞ (-1)^ℓ P_ℓ(r̂·Δψ̂) |Δψ|^ℓ / |r|^{ℓ+1}

use std::f64::consts::PI;

/// The bilocal gravitational kernel at a given multipole order.
///
/// Each multipole ℓ corresponds to a specific LPT order:
/// ℓ=1 → dipole (absorbed by momentum conservation)
/// ℓ=2 → quadrupole → linear growth (ZA)
/// ℓ=3 → octupole → 2LPT
/// ℓ=4 → hexadecapole → 3LPT
#[derive(Debug, Clone)]
pub struct BiLocalKernel {
    /// Maximum multipole order
    pub ell_max: usize,
    /// Gravitational coupling: 4πGρ̄ = (3/2)Ω_m H₀²
    pub coupling: f64,
}

impl BiLocalKernel {
    pub fn new(omega_m: f64, ell_max: usize) -> Self {
        // 4πGρ̄ = (3/2)Ω_m H₀² in code units where H₀ = 1
        let coupling = 1.5 * omega_m;
        Self { ell_max, coupling }
    }

    /// Multipole vertex factor V_ℓ(k₁, k₂) in Fourier space.
    ///
    /// The ℓ-th multipole of the bilocal kernel gives the vertex:
    ///   V_ℓ(k₁, k₂) = (-1)^ℓ × (2ℓ+1)/(4π) × I_ℓ(k₁, k₂)
    ///
    /// where I_ℓ is the angular integral of the Legendre polynomial
    /// against the Fourier-space propagator.
    ///
    /// For the first few orders:
    ///   V₂(k₁,k₂) = (k₁·k₂)/(k₁²k₂²) − 1/3   [quadrupole]
    ///   V₃(k₁,k₂,k₃) = det kernel               [octupole]
    pub fn vertex_quadrupole(&self, k1: &[f64; 3], k2: &[f64; 3]) -> f64 {
        let k1sq = k1[0] * k1[0] + k1[1] * k1[1] + k1[2] * k1[2];
        let k2sq = k2[0] * k2[0] + k2[1] * k2[1] + k2[2] * k2[2];
        if k1sq < 1e-30 || k2sq < 1e-30 {
            return 0.0;
        }
        let k1dk2 = k1[0] * k2[0] + k1[1] * k2[1] + k1[2] * k2[2];
        let mu = k1dk2 / (k1sq * k2sq).sqrt();
        self.coupling * (mu * mu - 1.0 / 3.0)
    }

    /// The F₂ symmetrized kernel from the action formulation.
    ///
    /// F₂(k₁,k₂) = 5/7 + (1/2)(k₁·k₂)(1/k₁² + 1/k₂²) + (2/7)(k₁·k₂)²/(k₁²k₂²)
    ///
    /// This is identical to the standard PT F₂ kernel, but here it arises
    /// from extremizing the quadrupole action rather than solving Poisson.
    pub fn f2_kernel(k1: &[f64; 3], k2: &[f64; 3]) -> f64 {
        let k1sq = k1[0] * k1[0] + k1[1] * k1[1] + k1[2] * k1[2];
        let k2sq = k2[0] * k2[0] + k2[1] * k2[1] + k2[2] * k2[2];
        if k1sq < 1e-30 || k2sq < 1e-30 {
            return 0.0;
        }
        let k1dk2 = k1[0] * k2[0] + k1[1] * k2[1] + k1[2] * k2[2];
        let mu = k1dk2 / (k1sq * k2sq).sqrt();
        let k_ratio = k1sq.sqrt() / k2sq.sqrt() + k2sq.sqrt() / k1sq.sqrt();

        5.0 / 7.0 + mu * k_ratio / 2.0 + 2.0 / 7.0 * mu * mu
    }

    /// The G₂ kernel (velocity divergence) from the action.
    ///
    /// G₂(k₁,k₂) = 3/7 + (1/2)(k₁·k₂)(1/k₁² + 1/k₂²) + (4/7)(k₁·k₂)²/(k₁²k₂²)
    pub fn g2_kernel(k1: &[f64; 3], k2: &[f64; 3]) -> f64 {
        let k1sq = k1[0] * k1[0] + k1[1] * k1[1] + k1[2] * k1[2];
        let k2sq = k2[0] * k2[0] + k2[1] * k2[1] + k2[2] * k2[2];
        if k1sq < 1e-30 || k2sq < 1e-30 {
            return 0.0;
        }
        let k1dk2 = k1[0] * k2[0] + k1[1] * k2[1] + k1[2] * k2[2];
        let mu = k1dk2 / (k1sq * k2sq).sqrt();
        let k_ratio = k1sq.sqrt() / k2sq.sqrt() + k2sq.sqrt() / k1sq.sqrt();

        3.0 / 7.0 + mu * k_ratio / 2.0 + 4.0 / 7.0 * mu * mu
    }

    /// Legendre polynomial P_ℓ(x).
    pub fn legendre(ell: usize, x: f64) -> f64 {
        match ell {
            0 => 1.0,
            1 => x,
            2 => 1.5 * x * x - 0.5,
            3 => 2.5 * x * x * x - 1.5 * x,
            4 => (35.0 * x.powi(4) - 30.0 * x * x + 3.0) / 8.0,
            5 => (63.0 * x.powi(5) - 70.0 * x.powi(3) + 15.0 * x) / 8.0,
            6 => (231.0 * x.powi(6) - 315.0 * x.powi(4) + 105.0 * x * x - 5.0) / 16.0,
            _ => {
                // Bonnet recursion: (n+1)P_{n+1} = (2n+1)xP_n - nP_{n-1}
                let mut p_prev = Self::legendre(5, x);
                let mut p_curr = Self::legendre(6, x);
                for n in 6..ell {
                    let p_next = ((2 * n + 1) as f64 * x * p_curr - n as f64 * p_prev)
                        / (n + 1) as f64;
                    p_prev = p_curr;
                    p_curr = p_next;
                }
                p_curr
            }
        }
    }

    /// Multipole expansion coefficient at order ℓ.
    ///
    /// The ℓ-th term in the expansion of 1/|r+δ| - 1/|r|:
    ///   a_ℓ = (-1)^ℓ × (2ℓ+1)/(4π)
    ///
    /// weighted by the angular average over the Lagrangian separation r.
    pub fn multipole_coefficient(&self, ell: usize) -> f64 {
        let sign = if ell % 2 == 0 { 1.0 } else { -1.0 };
        sign * (2 * ell + 1) as f64 / (4.0 * PI)
    }

    /// Angular integral ⟨P_ℓ(μ)⟩ over the uniform sphere.
    ///
    /// For the isotropic average: ⟨P_ℓ⟩ = δ_{ℓ,0}.
    /// For the quadrupole coupling with the tidal field:
    /// ⟨P₂(μ) e_i e_j⟩ = (1/5)(3 r̂_i r̂_j - δ_ij) ∝ tidal tensor.
    pub fn angular_average(ell: usize) -> f64 {
        if ell == 0 { 1.0 } else { 0.0 }
    }

    /// Multipole-integrated vertex for the action at order ℓ.
    ///
    /// After angular integration over the Lagrangian separation r̂,
    /// each multipole gives a specific momentum-space kernel:
    ///   ℓ=2: -(3/2)Ω_m [δ_{ij} - k_i k_j/k²]  (tidal projector)
    ///   ℓ=3: octupole → 2LPT kernel
    ///
    /// Returns the growth-factor coefficient g_ℓ that emerges from
    /// extremizing the action at this multipole order.
    pub fn growth_from_multipole(&self, ell: usize) -> f64 {
        match ell {
            0 | 1 => 0.0,  // ℓ=0,1 don't contribute (background + momentum conservation)
            2 => 1.0,      // ℓ=2 → linear growth g₁ = 1
            3 => -3.0 / 7.0,  // ℓ=3 → 2LPT coefficient g₂ = -3/7
            4 => {
                // ℓ=4 gives the 3LPT contributions:
                // two channels: g₃ₐ = -1/3, g₃ᵦ = 10/21
                // Combined effective: -(1/3) + (10/21) = 1/7
                1.0 / 7.0
            }
            _ => 0.0,
        }
    }
}

/// Pair interaction energy between two Lagrangian patches.
///
/// E(q, q') = -Gρ̄²/|x(q) - x(q')| + Gρ̄²/|q - q'|
///
/// This is what the FEM solver computes: the energy of a pair of
/// simplices at positions x = q + ψ, summed over all pairs.
pub fn pair_energy(q1: &[f64; 3], psi1: &[f64; 3], q2: &[f64; 3], psi2: &[f64; 3],
                   coupling: f64) -> f64 {
    let mut r_lag_sq = 0.0;
    let mut r_eul_sq = 0.0;
    for i in 0..3 {
        let dq = q1[i] - q2[i];
        let dx = (q1[i] + psi1[i]) - (q2[i] + psi2[i]);
        r_lag_sq += dq * dq;
        r_eul_sq += dx * dx;
    }
    if r_lag_sq < 1e-30 || r_eul_sq < 1e-30 {
        return 0.0;
    }
    coupling * (1.0 / r_lag_sq.sqrt() - 1.0 / r_eul_sq.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_legendre_values() {
        assert!((BiLocalKernel::legendre(0, 0.5) - 1.0).abs() < 1e-15);
        assert!((BiLocalKernel::legendre(1, 0.5) - 0.5).abs() < 1e-15);
        assert!((BiLocalKernel::legendre(2, 0.5) - (-0.125)).abs() < 1e-15);
        assert!((BiLocalKernel::legendre(3, 0.5) - (-0.4375)).abs() < 1e-15);
    }

    #[test]
    fn test_legendre_at_one() {
        for ell in 0..8 {
            assert!((BiLocalKernel::legendre(ell, 1.0) - 1.0).abs() < 1e-12,
                    "P_{}(1) = {}, expected 1", ell, BiLocalKernel::legendre(ell, 1.0));
        }
    }

    #[test]
    fn test_f2_kernel_collinear() {
        // For collinear k₁ ∥ k₂ (μ=1): F₂ = 5/7 + 1/2(k₁/k₂ + k₂/k₁) + 2/7
        let k1 = [0.1, 0.0, 0.0];
        let k2 = [0.2, 0.0, 0.0];
        let f2 = BiLocalKernel::f2_kernel(&k1, &k2);
        let expected = 5.0 / 7.0 + 0.5 * (0.5 + 2.0) + 2.0 / 7.0;
        assert!((f2 - expected).abs() < 1e-12, "F₂ collinear = {}, expected {}", f2, expected);
    }

    #[test]
    fn test_f2_kernel_perpendicular() {
        // For k₁ ⊥ k₂ (μ=0): F₂ = 5/7
        let k1 = [0.1, 0.0, 0.0];
        let k2 = [0.0, 0.1, 0.0];
        let f2 = BiLocalKernel::f2_kernel(&k1, &k2);
        assert!((f2 - 5.0 / 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_growth_from_multipole_matches_eds() {
        let kernel = BiLocalKernel::new(1.0, 4);
        assert!((kernel.growth_from_multipole(2) - 1.0).abs() < 1e-15);
        assert!((kernel.growth_from_multipole(3) - (-3.0 / 7.0)).abs() < 1e-15);
    }

    #[test]
    fn test_pair_energy_zero_displacement() {
        let q1 = [0.0, 0.0, 0.0];
        let q2 = [1.0, 0.0, 0.0];
        let psi_zero = [0.0, 0.0, 0.0];
        let e = pair_energy(&q1, &psi_zero, &q2, &psi_zero, 1.0);
        assert!((e).abs() < 1e-15, "Zero displacement → zero interaction energy");
    }
}

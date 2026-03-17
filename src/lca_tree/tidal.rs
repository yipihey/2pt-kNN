//! Per-particle tidal tensor accumulator.
//!
//! During the FMM downward pass, each particle accumulates contributions
//! to its external tidal tensor T_ab from far-field source nodes.  The
//! accumulated tensor can then be diagonalized to extract Doroshkevich
//! eigenvalues at each scale.

use super::eigen::{Sym3x3, SYM3_ZERO, sym3_add, sym3x3_eigenvalues};

/// Per-particle tidal tensor accumulator.
///
/// Stores the symmetric 3×3 tidal tensor T_ab = ∂²Φ/∂x_a∂x_b
/// as 6 independent components: [T_xx, T_yy, T_zz, T_xy, T_xz, T_yz].
#[derive(Debug, Clone)]
pub struct TidalAccum {
    /// Symmetric tidal tensor [T_xx, T_yy, T_zz, T_xy, T_xz, T_yz].
    pub t_ij: Sym3x3,
}

impl Default for TidalAccum {
    fn default() -> Self {
        Self { t_ij: SYM3_ZERO }
    }
}

impl TidalAccum {
    /// Create a zeroed accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a tidal tensor contribution.
    #[inline]
    pub fn add_contribution(&mut self, t: &Sym3x3) {
        sym3_add(&mut self.t_ij, t);
    }

    /// Extract eigenvalues of the accumulated tidal tensor.
    ///
    /// Returns [λ₁, λ₂, λ₃] in descending order.
    pub fn eigenvalues(&self) -> [f64; 3] {
        sym3x3_eigenvalues(self.t_ij)
    }

    /// Reset to zero.
    pub fn reset(&mut self) {
        self.t_ij = SYM3_ZERO;
    }
}

/// Scale-dependent tidal tensor: stores cumulative tidal tensor at each
/// tree depth, so eigenvalues can be extracted at multiple smoothing scales.
#[derive(Debug, Clone)]
pub struct ScaleTidal {
    /// Cumulative tidal tensor through each depth level.
    /// `t_ij[ell]` = tidal tensor including contributions from depths 0..=ell.
    pub t_ij: Vec<Sym3x3>,
}

impl ScaleTidal {
    /// Create with a given number of depth levels, all zeroed.
    pub fn new(n_levels: usize) -> Self {
        Self {
            t_ij: vec![SYM3_ZERO; n_levels],
        }
    }

    /// Extract eigenvalues at a given depth level.
    pub fn eigenvalues_at(&self, level: usize) -> [f64; 3] {
        sym3x3_eigenvalues(self.t_ij[level])
    }

    /// Number of stored levels.
    pub fn n_levels(&self) -> usize {
        self.t_ij.len()
    }
}

/// Doroshkevich eigenvalue trajectory: eigenvalues at each stored scale.
#[derive(Debug, Clone)]
pub struct EigenvalueTrajectory {
    /// Eigenvalues at each level: [λ₁, λ₂, λ₃] in descending order.
    pub eigenvalues: Vec<[f64; 3]>,
    /// Characteristic scale (e.g., cell size) at each level.
    pub scales: Vec<f64>,
}

impl EigenvalueTrajectory {
    /// Extract from a ScaleTidal accumulator.
    pub fn from_scale_tidal(st: &ScaleTidal, scales: &[f64]) -> Self {
        assert_eq!(st.n_levels(), scales.len());
        let eigenvalues: Vec<[f64; 3]> = (0..st.n_levels())
            .map(|l| st.eigenvalues_at(l))
            .collect();
        Self {
            eigenvalues,
            scales: scales.to_vec(),
        }
    }
}

/// Cosmic web type based on number of collapsed axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WebType {
    Void,
    Pancake,
    Filament,
    Halo,
}

impl WebType {
    /// Classify based on eigenvalues and linear growth factor.
    ///
    /// At 1LPT: axis i is collapsed if D₊λᵢ > 1.
    pub fn classify_1lpt(eigenvalues: &[f64; 3], d_plus: f64) -> Self {
        let n_collapsed = eigenvalues.iter().filter(|&&l| d_plus * l > 1.0).count();
        match n_collapsed {
            0 => WebType::Void,
            1 => WebType::Pancake,
            2 => WebType::Filament,
            3 => WebType::Halo,
            _ => unreachable!(),
        }
    }

    /// Classify at 2LPT order with ε = -3/7 correction.
    ///
    /// Axis i is collapsed if D₊λᵢ > 1 + ε·λⱼ·λₖ (where j,k are the other axes).
    pub fn classify_2lpt(eigenvalues: &[f64; 3], d_plus: f64) -> Self {
        let eps = -3.0 / 7.0;
        let [l1, l2, l3] = *eigenvalues;
        let c1 = d_plus * l1 > 1.0 + eps * l2 * l3;
        let c2 = d_plus * l2 > 1.0 + eps * l1 * l3;
        let c3 = d_plus * l3 > 1.0 + eps * l1 * l2;
        let n_collapsed = [c1, c2, c3].iter().filter(|&&c| c).count();
        match n_collapsed {
            0 => WebType::Void,
            1 => WebType::Pancake,
            2 => WebType::Filament,
            3 => WebType::Halo,
            _ => unreachable!(),
        }
    }
}

/// Normalize eigenvalues to dimensionless Doroshkevich form.
///
/// λ̂ᵢ = λᵢ / σ_T where σ_T² = (3H₀²Ωₘ/2a)² σ²(L)
///
/// For simplicity, this takes the raw eigenvalue variance σ² directly.
pub fn normalize_eigenvalues(eigenvalues: &[f64; 3], sigma_sq: f64) -> [f64; 3] {
    if sigma_sq <= 0.0 {
        return *eigenvalues;
    }
    let sigma = sigma_sq.sqrt();
    [eigenvalues[0] / sigma, eigenvalues[1] / sigma, eigenvalues[2] / sigma]
}

/// Invert measured Eulerian tidal eigenvalues to 1LPT Lagrangian eigenvalues.
///
/// At leading order: λᵢ^(1) = T_ii^E / c
/// where c is the Poisson proportionality constant.
pub fn invert_to_1lpt(eig_eulerian: &[f64; 3], proportionality: f64) -> [f64; 3] {
    let c = proportionality;
    [eig_eulerian[0] / c, eig_eulerian[1] / c, eig_eulerian[2] / c]
}

/// Invert measured Eulerian tidal eigenvalues to 2LPT Lagrangian eigenvalues.
///
/// λᵢ^(1) = T_ii^E/c − ε[(T_ii^E/c)² − (T_jj^E/c)(T_kk^E/c)]
/// where ε = -3/7 is the 2LPT correction.
pub fn invert_to_2lpt(eig_eulerian: &[f64; 3], proportionality: f64, epsilon: f64) -> [f64; 3] {
    let c = proportionality;
    let t = [eig_eulerian[0] / c, eig_eulerian[1] / c, eig_eulerian[2] / c];
    [
        t[0] - epsilon * (t[0] * t[0] - t[1] * t[2]),
        t[1] - epsilon * (t[1] * t[1] - t[0] * t[2]),
        t[2] - epsilon * (t[2] * t[2] - t[0] * t[1]),
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tidal_accum_add_and_eigenvalues() {
        let mut acc = TidalAccum::new();
        // Add diagonal tensor: diag(3, 2, 1)
        acc.add_contribution(&[3.0, 2.0, 1.0, 0.0, 0.0, 0.0]);
        let eig = acc.eigenvalues();
        assert!((eig[0] - 3.0).abs() < 1e-14);
        assert!((eig[1] - 2.0).abs() < 1e-14);
        assert!((eig[2] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn tidal_accum_reset() {
        let mut acc = TidalAccum::new();
        acc.add_contribution(&[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]);
        acc.reset();
        for &v in &acc.t_ij {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn web_type_1lpt_classification() {
        // All eigenvalues small → Void
        assert_eq!(WebType::classify_1lpt(&[0.1, 0.05, 0.01], 1.0), WebType::Void);
        // One large → Pancake
        assert_eq!(WebType::classify_1lpt(&[2.0, 0.5, 0.1], 1.0), WebType::Pancake);
        // Two large → Filament
        assert_eq!(WebType::classify_1lpt(&[2.0, 1.5, 0.1], 1.0), WebType::Filament);
        // All large → Halo
        assert_eq!(WebType::classify_1lpt(&[3.0, 2.0, 1.5], 1.0), WebType::Halo);
    }

    #[test]
    fn web_type_2lpt_differs() {
        // The 2LPT correction with ε = -3/7 should allow more "collapse"
        // since ε is negative (product of eigenvalues enters).
        let eig = [1.2, 0.8, 0.6];
        let w1 = WebType::classify_1lpt(&eig, 1.0);
        let w2 = WebType::classify_2lpt(&eig, 1.0);
        // At 1LPT: D₊λ₁ = 1.2 > 1 → Pancake
        assert_eq!(w1, WebType::Pancake);
        // At 2LPT: threshold for axis 1 is 1 + (-3/7)*0.8*0.6 = 1 - 0.206 = 0.794
        // D₊λ₁ = 1.2 > 0.794 ✓ and D₊λ₂ = 0.8 > 1 + (-3/7)*1.2*0.6 = 0.691 ✓
        assert!(matches!(w2, WebType::Filament | WebType::Halo));
    }

    #[test]
    fn inversion_round_trip_1lpt() {
        let original = [3.0, 2.0, 1.0];
        let c = 1.5;
        let eulerian = [original[0] * c, original[1] * c, original[2] * c];
        let recovered = invert_to_1lpt(&eulerian, c);
        for i in 0..3 {
            assert!((recovered[i] - original[i]).abs() < 1e-14);
        }
    }

    #[test]
    fn scale_tidal_basic() {
        let mut st = ScaleTidal::new(3);
        st.t_ij[0] = [1.0, 0.5, 0.3, 0.0, 0.0, 0.0];
        st.t_ij[1] = [2.0, 1.0, 0.5, 0.1, 0.0, 0.0];
        st.t_ij[2] = [3.0, 2.0, 1.0, 0.2, 0.1, 0.0];

        assert_eq!(st.n_levels(), 3);
        let eig0 = st.eigenvalues_at(0);
        assert!(eig0[0] >= eig0[1]);
        assert!(eig0[1] >= eig0[2]);
    }
}

//! Per-node multipole moments for gravity kernel evaluation.
//!
//! Each tree node stores source moments for the multipole expansion of the
//! gravitational potential.  These are computed from particle positions and
//! weights relative to the node's center of mass:
//!
//! - **Monopole**: M = Σ wᵢ (total mass)
//! - **Dipole**:   M^a = Σ wᵢ (xᵢ - x̄)^a (first moment about center)
//! - **Quadrupole**: M^{ab} = Σ wᵢ (xᵢ - x̄)^a (xᵢ - x̄)^b (second moment)
//!
//! The existing `LcaNode` stores monopole weight (W_C).  This module adds
//! dipole and quadrupole storage, which are needed for higher-order tidal
//! tensor evaluation.
//!
//! # Dual interpretation
//!
//! The monopole `W_C` serves double duty:
//! - **ξ estimator**: overdensity weight for pair decomposition
//! - **Gravity kernel**: source mass for potential expansion
//!
//! The dipole and quadrupole are purely for the gravity kernel.

use super::eigen::Sym3x3;

/// Multipole moments for a tree node, expanded about the node center.
///
/// These are the *source* moments used in the far-field expansion of the
/// gravitational potential Φ(x) = -G Σ mᵢ/|x - xᵢ|.
#[derive(Debug, Clone)]
pub struct NodeMultipole {
    /// Center of expansion (typically center of mass or bounding box center).
    pub center: [f64; 3],
    /// Monopole: total mass M = Σ wᵢ.
    pub monopole: f64,
    /// Dipole: M^a = Σ wᵢ (xᵢ - center)^a.
    /// If expanded about the center of mass, this is zero.
    pub dipole: [f64; 3],
    /// Traceless quadrupole: Q^{ab} = Σ wᵢ [(xᵢ-center)^a (xᵢ-center)^b - |xᵢ-center|²δ_{ab}/3].
    /// Stored as [Q_xx, Q_yy, Q_zz, Q_xy, Q_xz, Q_yz].
    /// Note: Q_xx + Q_yy + Q_zz = 0 by construction.
    pub quadrupole: Sym3x3,
    /// Full (non-traceless) second moment: I^{ab} = Σ wᵢ (xᵢ-center)^a (xᵢ-center)^b.
    /// Used for error estimates and opening-angle criteria.
    pub inertia: Sym3x3,
    /// Maximum distance from center to any particle in this node's subtree.
    pub radius: f64,
}

impl Default for NodeMultipole {
    fn default() -> Self {
        Self {
            center: [0.0; 3],
            monopole: 0.0,
            dipole: [0.0; 3],
            quadrupole: [0.0; 6],
            inertia: [0.0; 6],
            radius: 0.0,
        }
    }
}

/// Multipole order for tidal tensor evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultipoleOrder {
    Monopole,
    Dipole,
    Quadrupole,
}

/// Compute multipole moments for a set of particles about their center of mass.
///
/// Returns a `NodeMultipole` with the center set to the center of mass,
/// dipole = 0 (by definition), and the quadrupole computed from particle
/// positions relative to the COM.
pub fn compute_multipole(
    positions: &[[f64; 3]],
    weights: &[f64],
    start: usize,
    end: usize,
) -> NodeMultipole {
    if start >= end {
        return NodeMultipole::default();
    }

    // Pass 1: compute center of mass and total mass.
    let mut total_mass = 0.0;
    let mut com = [0.0f64; 3];
    for i in start..end {
        let w = weights[i];
        total_mass += w;
        for k in 0..3 {
            com[k] += w * positions[i][k];
        }
    }
    if total_mass > 0.0 {
        for k in 0..3 {
            com[k] /= total_mass;
        }
    }

    // Pass 2: compute quadrupole and inertia about COM.
    let mut inertia = [0.0f64; 6]; // [I_xx, I_yy, I_zz, I_xy, I_xz, I_yz]
    let mut radius_sq = 0.0f64;

    for i in start..end {
        let w = weights[i];
        let dx = positions[i][0] - com[0];
        let dy = positions[i][1] - com[1];
        let dz = positions[i][2] - com[2];

        inertia[0] += w * dx * dx;
        inertia[1] += w * dy * dy;
        inertia[2] += w * dz * dz;
        inertia[3] += w * dx * dy;
        inertia[4] += w * dx * dz;
        inertia[5] += w * dy * dz;

        let r2 = dx * dx + dy * dy + dz * dz;
        if r2 > radius_sq {
            radius_sq = r2;
        }
    }

    // Traceless quadrupole: Q^{ab} = I^{ab} - (tr I / 3) δ_{ab}
    let trace_3 = (inertia[0] + inertia[1] + inertia[2]) / 3.0;
    let quadrupole = [
        inertia[0] - trace_3,
        inertia[1] - trace_3,
        inertia[2] - trace_3,
        inertia[3],
        inertia[4],
        inertia[5],
    ];

    NodeMultipole {
        center: com,
        monopole: total_mass,
        // Dipole is zero when expanded about COM.
        dipole: [0.0; 3],
        quadrupole,
        inertia,
        radius: radius_sq.sqrt(),
    }
}

/// Shift multipole moments from one center to another.
///
/// Given moments about center `c_old`, compute moments about `c_new`.
/// This is needed when merging child node moments into a parent.
///
/// At monopole order: M is unchanged.
/// At dipole order: M'^a = M^a + M · Δx^a where Δx = c_old - c_new.
/// At quadrupole order: Q'^{ab} = Q^{ab} + M^a Δx^b + M^b Δx^a + M Δx^a Δx^b - trace correction.
pub fn shift_multipole(src: &NodeMultipole, new_center: &[f64; 3]) -> NodeMultipole {
    let dx = [
        src.center[0] - new_center[0],
        src.center[1] - new_center[1],
        src.center[2] - new_center[2],
    ];
    let m = src.monopole;

    // Shifted dipole: d'^a = d^a + M·Δx^a
    let dipole = [
        src.dipole[0] + m * dx[0],
        src.dipole[1] + m * dx[1],
        src.dipole[2] + m * dx[2],
    ];

    // Shifted inertia: I'^{ab} = I^{ab} + d^a·Δx^b + d^b·Δx^a + M·Δx^a·Δx^b
    let d = &src.dipole;
    let inertia = [
        src.inertia[0] + 2.0 * d[0] * dx[0] + m * dx[0] * dx[0],
        src.inertia[1] + 2.0 * d[1] * dx[1] + m * dx[1] * dx[1],
        src.inertia[2] + 2.0 * d[2] * dx[2] + m * dx[2] * dx[2],
        src.inertia[3] + d[0] * dx[1] + d[1] * dx[0] + m * dx[0] * dx[1],
        src.inertia[4] + d[0] * dx[2] + d[2] * dx[0] + m * dx[0] * dx[2],
        src.inertia[5] + d[1] * dx[2] + d[2] * dx[1] + m * dx[1] * dx[2],
    ];

    let trace_3 = (inertia[0] + inertia[1] + inertia[2]) / 3.0;
    let quadrupole = [
        inertia[0] - trace_3,
        inertia[1] - trace_3,
        inertia[2] - trace_3,
        inertia[3],
        inertia[4],
        inertia[5],
    ];

    let r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
    let radius = src.radius + r2.sqrt();

    NodeMultipole {
        center: *new_center,
        monopole: m,
        dipole,
        quadrupole,
        inertia,
        radius,
    }
}

/// Merge two child multipoles into a parent multipole about a given center.
pub fn merge_multipoles(
    left: &NodeMultipole,
    right: &NodeMultipole,
    parent_center: &[f64; 3],
) -> NodeMultipole {
    let left_shifted = shift_multipole(left, parent_center);
    let right_shifted = shift_multipole(right, parent_center);

    let monopole = left_shifted.monopole + right_shifted.monopole;
    let dipole = [
        left_shifted.dipole[0] + right_shifted.dipole[0],
        left_shifted.dipole[1] + right_shifted.dipole[1],
        left_shifted.dipole[2] + right_shifted.dipole[2],
    ];
    let mut inertia = [0.0f64; 6];
    for k in 0..6 {
        inertia[k] = left_shifted.inertia[k] + right_shifted.inertia[k];
    }

    let trace_3 = (inertia[0] + inertia[1] + inertia[2]) / 3.0;
    let quadrupole = [
        inertia[0] - trace_3,
        inertia[1] - trace_3,
        inertia[2] - trace_3,
        inertia[3],
        inertia[4],
        inertia[5],
    ];

    let radius = left_shifted.radius.max(right_shifted.radius);

    NodeMultipole {
        center: *parent_center,
        monopole,
        dipole,
        quadrupole,
        inertia,
        radius,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_particle_monopole() {
        let positions = [[1.0, 2.0, 3.0]];
        let weights = [5.0];
        let mp = compute_multipole(&positions, &weights, 0, 1);

        assert!((mp.monopole - 5.0).abs() < 1e-14);
        assert!((mp.center[0] - 1.0).abs() < 1e-14);
        assert!((mp.center[1] - 2.0).abs() < 1e-14);
        assert!((mp.center[2] - 3.0).abs() < 1e-14);
        // Dipole about COM is zero.
        for &d in &mp.dipole {
            assert!(d.abs() < 1e-14);
        }
        // Quadrupole of single particle at COM is zero.
        for &q in &mp.quadrupole {
            assert!(q.abs() < 1e-14);
        }
    }

    #[test]
    fn two_particles_com() {
        let positions = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let weights = [1.0, 1.0];
        let mp = compute_multipole(&positions, &weights, 0, 2);

        assert!((mp.monopole - 2.0).abs() < 1e-14);
        assert!((mp.center[0] - 1.0).abs() < 1e-14);
        assert!((mp.center[1]).abs() < 1e-14);
        // Dipole about COM should be zero.
        for &d in &mp.dipole {
            assert!(d.abs() < 1e-14);
        }
        // Inertia: each particle at distance 1 from COM along x.
        // I_xx = 2 × 1² = 2, I_yy = I_zz = 0
        assert!((mp.inertia[0] - 2.0).abs() < 1e-14);
        assert!((mp.inertia[1]).abs() < 1e-14);
    }

    #[test]
    fn traceless_quadrupole() {
        let positions = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -2.0, 0.0],
        ];
        let weights = [1.0; 4];
        let mp = compute_multipole(&positions, &weights, 0, 4);

        // Trace of quadrupole should be zero.
        let trace = mp.quadrupole[0] + mp.quadrupole[1] + mp.quadrupole[2];
        assert!(trace.abs() < 1e-14, "trace = {trace}");
    }

    #[test]
    fn shift_preserves_monopole() {
        let positions = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let weights = [2.0, 3.0];
        let mp = compute_multipole(&positions, &weights, 0, 2);

        let new_center = [10.0, 20.0, 30.0];
        let shifted = shift_multipole(&mp, &new_center);

        assert!((shifted.monopole - mp.monopole).abs() < 1e-14);
        assert_eq!(shifted.center, new_center);
    }

    #[test]
    fn shift_creates_dipole() {
        // Shift from COM → dipole should be non-zero.
        let positions = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let weights = [1.0, 1.0];
        let mp = compute_multipole(&positions, &weights, 0, 2);

        // COM is at (1,0,0). Shift to origin.
        let shifted = shift_multipole(&mp, &[0.0, 0.0, 0.0]);
        // Dipole: d'^x = 0 + M × (1-0) = 2.0
        assert!((shifted.dipole[0] - 2.0).abs() < 1e-14);
    }

    #[test]
    fn merge_two_particles() {
        let pos_l = [[0.0, 0.0, 0.0]];
        let wt_l = [1.0];
        let pos_r = [[4.0, 0.0, 0.0]];
        let wt_r = [1.0];

        let ml = compute_multipole(&pos_l, &wt_l, 0, 1);
        let mr = compute_multipole(&pos_r, &wt_r, 0, 1);

        let parent_center = [2.0, 0.0, 0.0];
        let merged = merge_multipoles(&ml, &mr, &parent_center);

        assert!((merged.monopole - 2.0).abs() < 1e-14);
        // Dipole about (2,0,0): left at (-2,0,0), right at (2,0,0) → net = 0
        assert!(merged.dipole[0].abs() < 1e-14);
        // Inertia I_xx = 1×4 + 1×4 = 8
        assert!((merged.inertia[0] - 8.0).abs() < 1e-14);
    }

    #[test]
    fn empty_range() {
        let mp = compute_multipole(&[], &[], 0, 0);
        assert_eq!(mp.monopole, 0.0);
        assert_eq!(mp.radius, 0.0);
    }
}

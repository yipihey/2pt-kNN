//! FMM-style gravity kernel: tidal tensor from tree multipole moments.
//!
//! Given a source node with multipole moments (monopole M, dipole M^a,
//! quadrupole Q^{ab}) and a field point x_i, computes the tidal tensor
//! contribution T_{ab}(x_i) = ∂²Φ/∂x_a∂x_b from the multipole expansion.
//!
//! # Multipole expansion of the tidal tensor
//!
//! At monopole order:
//!   T_{ab}^{(0)} = M/r⁵ (3 r_a r_b − r² δ_{ab})
//!
//! At dipole order, add:
//!   T_{ab}^{(1)} = contribution from ∂/∂x_c [T_{ab}^{(0)}] · M^c
//!
//! At quadrupole order, add O(r⁻⁵) terms from Q^{ab}.
//!
//! All formulas use the convention Φ = −G Σ m_i/|x − x_i| with G=1.

use super::eigen::Sym3x3;
use super::multipole::{NodeMultipole, MultipoleOrder};

/// Compute the tidal tensor contribution from a source node at a field point.
///
/// # Arguments
/// - `source`: multipole moments of the source node
/// - `field_point`: position where the tidal tensor is evaluated
/// - `order`: truncation order of the multipole expansion
///
/// # Returns
/// Symmetric tidal tensor [T_xx, T_yy, T_zz, T_xy, T_xz, T_yz].
pub fn tidal_from_node(
    source: &NodeMultipole,
    field_point: &[f64; 3],
    order: MultipoleOrder,
) -> Sym3x3 {
    let rx = field_point[0] - source.center[0];
    let ry = field_point[1] - source.center[1];
    let rz = field_point[2] - source.center[2];
    let r2 = rx * rx + ry * ry + rz * rz;

    if r2 < 1e-30 {
        return [0.0; 6];
    }

    let r = r2.sqrt();
    let r_inv = 1.0 / r;
    let r2_inv = r_inv * r_inv;
    let r3_inv = r2_inv * r_inv;
    let r5_inv = r3_inv * r2_inv;

    let rv = [rx, ry, rz];

    // ---- Monopole: T_{ab}^{(0)} = M (3 r_a r_b / r^5 - δ_{ab} / r^3) ----
    let m = source.monopole;
    let mut t = [
        m * (3.0 * rv[0] * rv[0] * r5_inv - r3_inv),  // T_xx
        m * (3.0 * rv[1] * rv[1] * r5_inv - r3_inv),  // T_yy
        m * (3.0 * rv[2] * rv[2] * r5_inv - r3_inv),  // T_zz
        m * 3.0 * rv[0] * rv[1] * r5_inv,              // T_xy
        m * 3.0 * rv[0] * rv[2] * r5_inv,              // T_xz
        m * 3.0 * rv[1] * rv[2] * r5_inv,              // T_yz
    ];

    if order == MultipoleOrder::Monopole {
        return t;
    }

    // ---- Dipole: T_{ab}^{(1)} = ∂_c T_{ab}^{(0)} × d^c ----
    // ∂_c [M(3 r_a r_b / r^5 − δ_{ab} / r^3)]
    //   = M [ 3(δ_{ac} r_b + δ_{bc} r_a) / r^5
    //         − 15 r_a r_b r_c / r^7
    //         + 3 δ_{ab} r_c / r^5 ]
    // Contracted with d^c:
    //   T^{(1)}_{ab} = 3/r^5 (d_a r_b + d_b r_a)
    //                  − 15/r^7 r_a r_b (d·r)
    //                  + 3/r^5 δ_{ab} (d·r)
    let d = &source.dipole;
    let r7_inv = r5_inv * r2_inv;
    let d_dot_r = d[0] * rv[0] + d[1] * rv[1] + d[2] * rv[2];

    // Pairs: (a,b) → index mapping: (0,0)→0, (1,1)→1, (2,2)→2, (0,1)→3, (0,2)→4, (1,2)→5
    let ab_pairs: [(usize, usize); 6] = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)];
    let delta = |a: usize, b: usize| -> f64 { if a == b { 1.0 } else { 0.0 } };

    for (idx, &(a, b)) in ab_pairs.iter().enumerate() {
        t[idx] += 3.0 * r5_inv * (d[a] * rv[b] + d[b] * rv[a])
            - 15.0 * r7_inv * rv[a] * rv[b] * d_dot_r
            + 3.0 * r5_inv * delta(a, b) * d_dot_r;
    }

    if order == MultipoleOrder::Dipole {
        return t;
    }

    // ---- Quadrupole: T_{ab}^{(2)} from traceless quadrupole Q^{cd} ----
    // The quadrupole contribution to the potential is:
    //   Φ^{(2)} = −(1/2) Q^{cd} ∂_c ∂_d (1/r) = −(1/2) Q^{cd} (3 r_c r_d / r^5 − δ_{cd} / r^3)
    //
    // The tidal tensor from this is ∂_a ∂_b Φ^{(2)}:
    //   T_{ab}^{(2)} = −(1/2) Q^{cd} ∂_a ∂_b (3 r_c r_d / r^5 − δ_{cd} / r^3)
    //
    // Computing ∂_a ∂_b [3 r_c r_d / r^5]:
    //   = 3[δ_{ac}δ_{bd} + δ_{ad}δ_{bc}]/r^5
    //     − 15[δ_{ac} r_b r_d + δ_{ad} r_b r_c + δ_{bc} r_a r_d + δ_{bd} r_a r_c]/r^7
    //     + 105 r_a r_b r_c r_d / r^9
    //
    // Computing ∂_a ∂_b [δ_{cd} / r^3]:
    //   = δ_{cd} [3 r_a r_b / r^5 − δ_{ab} / r^3]
    //
    // For the traceless quadrupole (tr Q = 0), the δ_{cd} terms cancel.
    let q = &source.quadrupole;
    let r9_inv = r7_inv * r2_inv;

    // Precompute Q contracted with r: Q_r[a] = Q^{ac} r_c
    let qr = [
        q[0] * rv[0] + q[3] * rv[1] + q[4] * rv[2],
        q[3] * rv[0] + q[1] * rv[1] + q[5] * rv[2],
        q[4] * rv[0] + q[5] * rv[1] + q[2] * rv[2],
    ];
    // r^T Q r = Σ Q^{cd} r_c r_d
    let rqr = qr[0] * rv[0] + qr[1] * rv[1] + qr[2] * rv[2];
    // tr(Q) — should be zero for traceless, but include for robustness.
    let _tr_q = q[0] + q[1] + q[2];

    // Full Q tensor for indexing.
    let q_mat = |a: usize, b: usize| -> f64 {
        match (a, b) {
            (0, 0) => q[0], (1, 1) => q[1], (2, 2) => q[2],
            (0, 1) | (1, 0) => q[3],
            (0, 2) | (2, 0) => q[4],
            (1, 2) | (2, 1) => q[5],
            _ => 0.0,
        }
    };

    for (idx, &(a, b)) in ab_pairs.iter().enumerate() {
        // Derivation (see module-level docs for details):
        //
        // Φ^{(2)} = -(1/2) Q^{cd} ∂_c ∂_d (1/r)
        //         = -(1/2) Q^{cd} (3 r_c r_d / r^5 − δ_{cd} / r^3)
        //
        // T_{ab}^{(2)} = ∂_a ∂_b Φ^{(2)} = -(1/2) Q^{cd} ∂_a ∂_b [3 r_c r_d / r^5 − δ_{cd} / r^3]
        //
        // After contracting with Q^{cd} and using tr(Q)=0:
        // T_{ab}^{(2)} = -(3/2) [2 Q_{ab}/r^5
        //                        − 5(δ_{ab} rQr + 2 r_b qr[a] + 2 r_a qr[b])/r^7
        //                        + 35 r_a r_b rQr / r^9]

        let t2 = -1.5 * (
            2.0 * q_mat(a, b) * r5_inv
            - 5.0 * r7_inv * (delta(a, b) * rqr + 2.0 * rv[b] * qr[a] + 2.0 * rv[a] * qr[b])
            + 35.0 * rv[a] * rv[b] * rqr * r9_inv
        );
        // Traceless correction (should be ~0 but include for robustness).
        // + 0.5 * tr_q * (3.0 * rv[a] * rv[b] * r5_inv - delta(a, b) * r3_inv);

        t[idx] += t2;
    }

    t
}

/// Compute tidal tensor from direct particle-by-particle summation.
///
/// T_{ab}(x) = Σᵢ wᵢ (3 rᵢ_a rᵢ_b / rᵢ⁵ − δ_{ab} / rᵢ³)
///
/// Used as ground truth for testing the multipole expansion.
pub fn tidal_direct(
    field_point: &[f64; 3],
    positions: &[[f64; 3]],
    weights: &[f64],
    softening_sq: f64,
) -> Sym3x3 {
    let mut t = [0.0f64; 6];
    let delta = |a: usize, b: usize| -> f64 { if a == b { 1.0 } else { 0.0 } };
    let ab_pairs: [(usize, usize); 6] = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)];

    for (i, pos) in positions.iter().enumerate() {
        let rx = field_point[0] - pos[0];
        let ry = field_point[1] - pos[1];
        let rz = field_point[2] - pos[2];
        let r2 = rx * rx + ry * ry + rz * rz + softening_sq;

        if r2 < 1e-30 {
            continue;
        }

        let r = r2.sqrt();
        let r3_inv = 1.0 / (r * r2);
        let r5_inv = r3_inv / r2;
        let rv = [rx, ry, rz];
        let w = weights[i];

        for (idx, &(a, b)) in ab_pairs.iter().enumerate() {
            t[idx] += w * (3.0 * rv[a] * rv[b] * r5_inv - delta(a, b) * r3_inv);
        }
    }

    t
}

/// Compute tidal tensor with periodic minimum-image convention.
pub fn tidal_direct_periodic(
    field_point: &[f64; 3],
    positions: &[[f64; 3]],
    weights: &[f64],
    box_size: f64,
    softening_sq: f64,
) -> Sym3x3 {
    let mut t = [0.0f64; 6];
    let delta_fn = |a: usize, b: usize| -> f64 { if a == b { 1.0 } else { 0.0 } };
    let ab_pairs: [(usize, usize); 6] = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)];
    let half_box = 0.5 * box_size;

    for (i, pos) in positions.iter().enumerate() {
        let mut rv = [
            field_point[0] - pos[0],
            field_point[1] - pos[1],
            field_point[2] - pos[2],
        ];
        for k in 0..3 {
            if rv[k] > half_box { rv[k] -= box_size; }
            else if rv[k] < -half_box { rv[k] += box_size; }
        }
        let r2 = rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2] + softening_sq;

        if r2 < 1e-30 {
            continue;
        }

        let r = r2.sqrt();
        let r3_inv = 1.0 / (r * r2);
        let r5_inv = r3_inv / r2;
        let w = weights[i];

        for (idx, &(a, b)) in ab_pairs.iter().enumerate() {
            t[idx] += w * (3.0 * rv[a] * rv[b] * r5_inv - delta_fn(a, b) * r3_inv);
        }
    }

    t
}

/// Barnes-Hut style opening angle criterion.
///
/// Returns true if the source node should be opened (i.e., it is too close
/// for the multipole approximation to be accurate).
///
/// Criterion: r > θ⁻¹ × h, where h = source.radius (max particle distance
/// from center) and θ is the opening angle parameter.
#[inline]
pub fn should_open(source: &NodeMultipole, field_point: &[f64; 3], theta: f64) -> bool {
    let rx = field_point[0] - source.center[0];
    let ry = field_point[1] - source.center[1];
    let rz = field_point[2] - source.center[2];
    let r2 = rx * rx + ry * ry + rz * rz;
    let h = source.radius;
    // Open if h/r > θ, i.e., h² > θ² r²
    h * h > theta * theta * r2
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::multipole::compute_multipole;
    use super::super::eigen::sym3_frob_sq;

    #[test]
    fn point_mass_tidal() {
        // Single particle at origin, field point at (r, 0, 0).
        let source = NodeMultipole {
            center: [0.0, 0.0, 0.0],
            monopole: 1.0,
            dipole: [0.0; 3],
            quadrupole: [0.0; 6],
            inertia: [0.0; 6],
            radius: 0.0,
        };
        let r = 5.0;
        let fp = [r, 0.0, 0.0];

        let t = tidal_from_node(&source, &fp, MultipoleOrder::Monopole);

        // T_xx = M(3·r²/r⁵ − 1/r³) = M(3/r³ − 1/r³) = 2M/r³
        let expected_txx = 2.0 / (r * r * r);
        assert!((t[0] - expected_txx).abs() < 1e-14, "T_xx = {}, expected {}", t[0], expected_txx);

        // T_yy = M(0 − 1/r³) = −M/r³
        let expected_tyy = -1.0 / (r * r * r);
        assert!((t[1] - expected_tyy).abs() < 1e-14, "T_yy = {}", t[1]);

        // T_zz = T_yy
        assert!((t[2] - expected_tyy).abs() < 1e-14);

        // Off-diagonal should be zero.
        assert!(t[3].abs() < 1e-14); // T_xy
        assert!(t[4].abs() < 1e-14); // T_xz
        assert!(t[5].abs() < 1e-14); // T_yz
    }

    #[test]
    fn traceless_tidal() {
        // The tidal tensor should be traceless (for a source in vacuum).
        let source = NodeMultipole {
            center: [1.0, 2.0, 3.0],
            monopole: 5.0,
            dipole: [0.1, -0.2, 0.3],
            quadrupole: [0.5, -0.2, -0.3, 0.1, 0.05, -0.15],
            inertia: [0.0; 6],
            radius: 1.0,
        };
        let fp = [10.0, 12.0, 8.0];

        for order in [MultipoleOrder::Monopole, MultipoleOrder::Dipole, MultipoleOrder::Quadrupole] {
            let t = tidal_from_node(&source, &fp, order);
            let trace = t[0] + t[1] + t[2];
            assert!(
                trace.abs() < 1e-10,
                "trace at {:?} order = {:.2e}",
                order,
                trace
            );
        }
    }

    #[test]
    fn monopole_matches_direct_for_point_mass() {
        let positions = [[3.0, 4.0, 5.0]];
        let weights = [2.0];
        let fp = [10.0, 0.0, 0.0];

        let t_direct = tidal_direct(&fp, &positions, &weights, 0.0);

        let mp = compute_multipole(&positions, &weights, 0, 1);
        let t_mono = tidal_from_node(&mp, &fp, MultipoleOrder::Monopole);

        for i in 0..6 {
            assert!(
                (t_direct[i] - t_mono[i]).abs() < 1e-12,
                "component {}: direct={}, mono={}",
                i, t_direct[i], t_mono[i]
            );
        }
    }

    #[test]
    fn multipole_convergence_with_distance() {
        // Cluster of particles near origin, field point at varying distances.
        // Error should scale as O(h/r)^{p+1} where p is the multipole order.
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-0.5, 0.5, 0.3],
        ];
        let weights = [1.0, 2.0, 1.5, 0.8, 1.2];

        let mp = compute_multipole(&positions, &weights, 0, 5);

        let mut prev_mono_err = f64::MAX;
        let mut prev_quad_err = f64::MAX;

        for &dist in &[10.0, 20.0, 40.0] {
            let fp = [dist, 0.3 * dist, 0.1 * dist]; // off-axis

            let t_direct = tidal_direct(&fp, &positions, &weights, 0.0);
            let t_mono = tidal_from_node(&mp, &fp, MultipoleOrder::Monopole);
            let t_quad = tidal_from_node(&mp, &fp, MultipoleOrder::Quadrupole);

            let err_mono = sym3_frob_err(&t_direct, &t_mono);
            let err_quad = sym3_frob_err(&t_direct, &t_quad);

            // Errors should decrease with distance.
            assert!(
                err_mono < prev_mono_err * 1.1,  // allow small noise
                "monopole error not decreasing: {} vs {}",
                err_mono, prev_mono_err
            );
            assert!(
                err_quad < prev_quad_err * 1.1,
                "quadrupole error not decreasing: {} vs {}",
                err_quad, prev_quad_err
            );

            // Both errors should be small at these distances.
            assert!(
                err_mono < 0.1,
                "monopole error too large at dist={}: {}",
                dist, err_mono
            );
            assert!(
                err_quad < 0.1,
                "quadrupole error too large at dist={}: {}",
                dist, err_quad
            );

            prev_mono_err = err_mono;
            prev_quad_err = err_quad;
        }
    }

    #[test]
    fn shell_theorem() {
        // Uniform shell of particles → tidal tensor inside should be ~0.
        // 26 points on a cube shell at distance 10 from origin.
        let r = 10.0;
        let mut positions = Vec::new();
        for &x in &[-r, 0.0, r] {
            for &y in &[-r, 0.0, r] {
                for &z in &[-r, 0.0, r] {
                    if x == 0.0 && y == 0.0 && z == 0.0 { continue; }
                    positions.push([x, y, z]);
                }
            }
        }
        let n = positions.len();
        let weights = vec![1.0; n];

        let fp = [0.0, 0.0, 0.0];
        let t = tidal_direct(&fp, &positions, &weights, 0.0);

        // Due to cubic symmetry, the tidal tensor at origin should be zero.
        // (Not exactly a spherical shell, but cubic symmetry gives the same result.)
        for i in 0..6 {
            assert!(
                t[i].abs() < 1e-12,
                "shell theorem: T[{}] = {:.2e}",
                i, t[i]
            );
        }
    }

    #[test]
    fn opening_angle_criterion() {
        let source = NodeMultipole {
            center: [0.0, 0.0, 0.0],
            monopole: 1.0,
            dipole: [0.0; 3],
            quadrupole: [0.0; 6],
            inertia: [0.0; 6],
            radius: 1.0,
        };

        // Close field point: should open.
        assert!(should_open(&source, &[1.5, 0.0, 0.0], 0.5));
        // Far field point: should not open.
        assert!(!should_open(&source, &[100.0, 0.0, 0.0], 0.5));
    }

    fn sym3_frob_err(a: &Sym3x3, b: &Sym3x3) -> f64 {
        let diff = [
            a[0] - b[0], a[1] - b[1], a[2] - b[2],
            a[3] - b[3], a[4] - b[4], a[5] - b[5],
        ];
        let norm_a = sym3_frob_sq(a).sqrt().max(1e-30);
        sym3_frob_sq(&diff).sqrt() / norm_a
    }
}

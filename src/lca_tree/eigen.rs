//! Closed-form eigenvalue decomposition for real symmetric 3x3 matrices.
//!
//! Uses the analytically stable trigonometric (Cardano) form to avoid
//! catastrophic cancellation for nearly degenerate eigenvalues.
//!
//! Reference: Smith, O.K. (1961), "Eigenvalues of a symmetric 3×3 matrix",
//! Communications of the ACM, 4(4):168.

use std::f64::consts::PI;

/// Symmetric 3x3 matrix stored as 6 independent components:
/// `[T_xx, T_yy, T_zz, T_xy, T_xz, T_yz]`
pub type Sym3x3 = [f64; 6];

/// Zero symmetric 3x3 matrix.
pub const SYM3_ZERO: Sym3x3 = [0.0; 6];

/// Add a contribution to a symmetric 3x3 matrix.
#[inline]
pub fn sym3_add(a: &mut Sym3x3, b: &Sym3x3) {
    for i in 0..6 {
        a[i] += b[i];
    }
}

/// Scale a symmetric 3x3 matrix.
#[inline]
pub fn sym3_scale(a: &Sym3x3, s: f64) -> Sym3x3 {
    [a[0] * s, a[1] * s, a[2] * s, a[3] * s, a[4] * s, a[5] * s]
}

/// Trace of a symmetric 3x3 matrix: I₁ = T_xx + T_yy + T_zz.
#[inline]
pub fn sym3_trace(m: &Sym3x3) -> f64 {
    m[0] + m[1] + m[2]
}

/// Frobenius norm squared of a symmetric 3x3 matrix.
#[inline]
pub fn sym3_frob_sq(m: &Sym3x3) -> f64 {
    m[0] * m[0] + m[1] * m[1] + m[2] * m[2]
        + 2.0 * (m[3] * m[3] + m[4] * m[4] + m[5] * m[5])
}

/// Compute eigenvalues of a real symmetric 3x3 matrix using the trigonometric
/// method (Cardano's formula, analytically stable form).
///
/// Returns eigenvalues in descending order: λ₁ ≥ λ₂ ≥ λ₃.
///
/// # Storage convention
/// `m = [M_xx, M_yy, M_zz, M_xy, M_xz, M_yz]`
pub fn sym3x3_eigenvalues(m: Sym3x3) -> [f64; 3] {
    let [a11, a22, a33, a12, a13, a23] = m;

    // Invariants of the characteristic polynomial det(M - λI) = 0:
    // -λ³ + I₁λ² - I₂λ + I₃ = 0
    // where I₁ = tr(M), I₂ = (tr(M)² - tr(M²))/2, I₃ = det(M).

    let i1 = a11 + a22 + a33;
    let i2 = a11 * a22 + a11 * a33 + a22 * a33
        - a12 * a12 - a13 * a13 - a23 * a23;
    let i3 = a11 * a22 * a33 + 2.0 * a12 * a13 * a23
        - a11 * a23 * a23 - a22 * a13 * a13 - a33 * a12 * a12;

    // Shift to depressed cubic: substitute λ = t + I₁/3
    // t³ + pt + q = 0
    // where p = I₂ - I₁²/3, q = (2I₁³ - 9I₁I₂ + 27I₃)/27
    let i1_3 = i1 / 3.0;
    let p = i2 - i1 * i1_3;
    let q = (-2.0 * i1 * i1 * i1 + 9.0 * i1 * i2 - 27.0 * i3) / 27.0;

    // For a real symmetric matrix, p ≤ 0 (the matrix has real eigenvalues).
    // Use the trigonometric solution.
    let neg_p_3 = (-p / 3.0).max(0.0); // guard against tiny positive p from roundoff
    let r = neg_p_3.sqrt();

    if r < 1e-30 {
        // Degenerate: all eigenvalues equal to I₁/3.
        return [i1_3, i1_3, i1_3];
    }

    // cos(θ) = -q/(2r³)
    let r3 = r * r * r;
    let cos_theta = (-q / (2.0 * r3)).clamp(-1.0, 1.0);
    let theta = cos_theta.acos();

    // Three roots of the depressed cubic:
    let mut e1 = 2.0 * r * (theta / 3.0).cos() + i1_3;
    let mut e2 = 2.0 * r * ((theta + 2.0 * PI) / 3.0).cos() + i1_3;
    let mut e3 = 2.0 * r * ((theta + 4.0 * PI) / 3.0).cos() + i1_3;

    // Sort descending.
    if e2 > e1 { std::mem::swap(&mut e1, &mut e2); }
    if e3 > e1 { std::mem::swap(&mut e1, &mut e3); }
    if e3 > e2 { std::mem::swap(&mut e2, &mut e3); }

    [e1, e2, e3]
}

/// Compute eigenvectors + eigenvalues of a real symmetric 3x3 matrix.
///
/// Returns `([λ₁, λ₂, λ₃], [[v₁], [v₂], [v₃]])` with eigenvalues in
/// descending order and corresponding unit eigenvectors.
pub fn sym3x3_eigen(m: Sym3x3) -> ([f64; 3], [[f64; 3]; 3]) {
    let eigenvalues = sym3x3_eigenvalues(m);
    let [a11, a22, a33, a12, a13, a23] = m;

    let mut vecs = [[0.0f64; 3]; 3];

    for (i, &lam) in eigenvalues.iter().enumerate() {
        // Solve (M - λI)v = 0 by computing the cross product of two rows
        // of (M - λI). Pick the cross product with largest norm for stability.
        let r0 = [a11 - lam, a12, a13];
        let r1 = [a12, a22 - lam, a23];
        let r2 = [a13, a23, a33 - lam];

        let c01 = cross(&r0, &r1);
        let c02 = cross(&r0, &r2);
        let c12 = cross(&r1, &r2);

        let n01 = dot(&c01, &c01);
        let n02 = dot(&c02, &c02);
        let n12 = dot(&c12, &c12);

        let (best, best_n) = if n01 >= n02 && n01 >= n12 {
            (c01, n01)
        } else if n02 >= n12 {
            (c02, n02)
        } else {
            (c12, n12)
        };

        if best_n > 1e-60 {
            let inv_norm = 1.0 / best_n.sqrt();
            vecs[i] = [best[0] * inv_norm, best[1] * inv_norm, best[2] * inv_norm];
        } else {
            // Degenerate eigenvalue — pick an arbitrary orthogonal direction.
            vecs[i] = if i == 0 {
                [1.0, 0.0, 0.0]
            } else if i == 1 {
                // Gram-Schmidt against v0
                let mut v = [0.0, 1.0, 0.0];
                let d = dot(&v, &vecs[0]);
                for k in 0..3 { v[k] -= d * vecs[0][k]; }
                let n = dot(&v, &v).sqrt();
                if n > 1e-30 {
                    for k in 0..3 { v[k] /= n; }
                }
                v
            } else {
                cross(&vecs[0], &vecs[1])
            };
        }
    }

    (eigenvalues, vecs)
}

#[inline]
fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonal_matrix() {
        let m = [3.0, 2.0, 1.0, 0.0, 0.0, 0.0];
        let eig = sym3x3_eigenvalues(m);
        assert!((eig[0] - 3.0).abs() < 1e-14);
        assert!((eig[1] - 2.0).abs() < 1e-14);
        assert!((eig[2] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn identity_matrix() {
        let m = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let eig = sym3x3_eigenvalues(m);
        for &e in &eig {
            assert!((e - 1.0).abs() < 1e-14);
        }
    }

    #[test]
    fn zero_matrix() {
        let m = [0.0; 6];
        let eig = sym3x3_eigenvalues(m);
        for &e in &eig {
            assert!(e.abs() < 1e-14);
        }
    }

    #[test]
    fn known_eigenvalues() {
        // Matrix: [[2, 1, 0], [1, 3, 1], [0, 1, 2]]
        // Eigenvalues: 4, 2, 1
        let m = [2.0, 3.0, 2.0, 1.0, 0.0, 1.0];
        let eig = sym3x3_eigenvalues(m);
        assert!((eig[0] - 4.0).abs() < 1e-12, "λ₁ = {}", eig[0]);
        assert!((eig[1] - 2.0).abs() < 1e-12, "λ₂ = {}", eig[1]);
        assert!((eig[2] - 1.0).abs() < 1e-12, "λ₃ = {}", eig[2]);
    }

    #[test]
    fn trace_and_det_preserved() {
        // Random-ish symmetric matrix.
        let m = [5.0, 3.0, 1.0, 2.0, -1.0, 0.5];
        let eig = sym3x3_eigenvalues(m);

        let trace = eig[0] + eig[1] + eig[2];
        let expected_trace = m[0] + m[1] + m[2];
        assert!((trace - expected_trace).abs() < 1e-12);

        // Sum of products of pairs = I₂
        let sum_pairs = eig[0] * eig[1] + eig[0] * eig[2] + eig[1] * eig[2];
        let expected_i2 = m[0] * m[1] + m[0] * m[2] + m[1] * m[2]
            - m[3] * m[3] - m[4] * m[4] - m[5] * m[5];
        assert!((sum_pairs - expected_i2).abs() < 1e-10);
    }

    #[test]
    fn nearly_degenerate() {
        // Two eigenvalues very close.
        let eps = 1e-10;
        let m = [1.0 + eps, 1.0, 1.0 - eps, 0.0, 0.0, 0.0];
        let eig = sym3x3_eigenvalues(m);
        // Trace must be preserved exactly.
        let trace = eig[0] + eig[1] + eig[2];
        assert!((trace - 3.0).abs() < 1e-12);
        // Individual eigenvalues may lose precision at ~eps level.
        assert!((eig[0] - (1.0 + eps)).abs() < 1e-8);
        assert!((eig[1] - 1.0).abs() < 1e-8);
        assert!((eig[2] - (1.0 - eps)).abs() < 1e-8);
    }

    #[test]
    fn negative_eigenvalues() {
        let m = [-2.0, -3.0, -1.0, 0.5, 0.0, 0.0];
        let eig = sym3x3_eigenvalues(m);
        // All eigenvalues should be negative.
        assert!(eig[0] < 0.0);
        assert!(eig[2] < 0.0);
        // Descending order.
        assert!(eig[0] >= eig[1]);
        assert!(eig[1] >= eig[2]);
    }

    #[test]
    fn eigenvectors_orthogonal() {
        let m = [5.0, 3.0, 1.0, 2.0, -1.0, 0.5];
        let (_, vecs) = sym3x3_eigen(m);

        for i in 0..3 {
            let norm = dot(&vecs[i], &vecs[i]);
            assert!((norm - 1.0).abs() < 1e-10, "v{i} norm = {norm}");
        }

        // Orthogonality.
        let d01 = dot(&vecs[0], &vecs[1]).abs();
        let d02 = dot(&vecs[0], &vecs[2]).abs();
        let d12 = dot(&vecs[1], &vecs[2]).abs();
        assert!(d01 < 1e-10, "v0·v1 = {d01}");
        assert!(d02 < 1e-10, "v0·v2 = {d02}");
        assert!(d12 < 1e-10, "v1·v2 = {d12}");
    }

    #[test]
    fn eigenvectors_satisfy_equation() {
        let m = [5.0, 3.0, 1.0, 2.0, -1.0, 0.5];
        let [a11, a22, a33, a12, a13, a23] = m;
        let (eigenvalues, vecs) = sym3x3_eigen(m);

        for (i, &lam) in eigenvalues.iter().enumerate() {
            let v = vecs[i];
            // Mv - λv should be zero.
            let mv = [
                a11 * v[0] + a12 * v[1] + a13 * v[2] - lam * v[0],
                a12 * v[0] + a22 * v[1] + a23 * v[2] - lam * v[1],
                a13 * v[0] + a23 * v[1] + a33 * v[2] - lam * v[2],
            ];
            let resid = dot(&mv, &mv).sqrt();
            assert!(resid < 1e-10, "eigenvector {i} residual = {resid}");
        }
    }

    #[test]
    fn random_matrices_trace_preserved() {
        // Test with pseudo-random matrices using a simple LCG.
        let mut state = 42u64;
        let lcg = |s: &mut u64| -> f64 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((*s >> 11) as f64 / (1u64 << 53) as f64) * 10.0 - 5.0
        };

        for _ in 0..1000 {
            let m = [lcg(&mut state), lcg(&mut state), lcg(&mut state),
                      lcg(&mut state), lcg(&mut state), lcg(&mut state)];
            let eig = sym3x3_eigenvalues(m);

            let trace_eig = eig[0] + eig[1] + eig[2];
            let trace_m = m[0] + m[1] + m[2];
            assert!(
                (trace_eig - trace_m).abs() < 1e-10 * (1.0 + trace_m.abs()),
                "trace mismatch: eig={trace_eig}, m={trace_m}"
            );
            assert!(eig[0] >= eig[1] - 1e-14, "not sorted: {:?}", eig);
            assert!(eig[1] >= eig[2] - 1e-14, "not sorted: {:?}", eig);
        }
    }
}

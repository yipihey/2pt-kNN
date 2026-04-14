//! Savitzky-Golay smoothing and differentiation.
//!
//! Fits a local polynomial at each grid point via least-squares, yielding
//! simultaneous smoothing and analytic differentiation. At boundaries,
//! asymmetric (one-sided) windows are used automatically.
//!
//! The input signal must be sampled on a **uniform** grid with spacing `h`.

/// Apply a Savitzky-Golay filter, returning both the smoothed signal and its
/// first derivative (in physical units, i.e. dy/dx not dy/di).
///
/// At each point, a polynomial of degree `poly_order` is fit to the nearest
/// `2 * half_window + 1` data points (or fewer at boundaries). The polynomial
/// is centered at the evaluation point, so c₀ = smoothed value and c₁ = dy/dx.
///
/// # Panics
/// Panics if `y.len() < poly_order + 1`.
pub fn sg_smooth_diff(
    y: &[f64],
    h: f64,
    half_window: usize,
    poly_order: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = y.len();
    assert!(
        n >= poly_order + 1,
        "need at least poly_order + 1 = {} points, got {}",
        poly_order + 1,
        n,
    );
    let m = half_window.min((n - 1) / 2);

    let mut smoothed = vec![0.0; n];
    let mut deriv = vec![0.0; n];

    for i in 0..n {
        // Window bounds: symmetric where possible, clamped at edges
        let lo = i.saturating_sub(m);
        let hi = (i + m).min(n - 1);
        let win_size = hi - lo + 1;
        let p_eff = poly_order.min(win_size - 1);
        let np1 = p_eff + 1;

        // Build normal equations for polynomial centered at point i:
        //   p(t) = c₀ + c₁t + c₂t² + ...    where t = (j − i)·h
        // JᵀJ·c = Jᵀy
        let mut jtj = vec![vec![0.0; np1]; np1];
        let mut jty = vec![0.0; np1];

        for j in lo..=hi {
            let t = (j as f64 - i as f64) * h;
            // Precompute powers t^0, t^1, ..., t^{2*p_eff}
            let n_powers = 2 * p_eff + 1;
            let mut tp = vec![1.0; n_powers];
            for k in 1..n_powers {
                tp[k] = tp[k - 1] * t;
            }
            // JᵀJ[a][b] = Σ_j t_j^{a+b}
            for a in 0..np1 {
                for b in 0..np1 {
                    jtj[a][b] += tp[a + b];
                }
                jty[a] += tp[a] * y[j];
            }
        }

        let c = solve_small(&mut jtj, &mut jty);
        smoothed[i] = c[0];
        deriv[i] = if np1 > 1 { c[1] } else { 0.0 };
    }

    (smoothed, deriv)
}

/// Solve a small linear system Ax = b via Gaussian elimination with partial pivoting.
///
/// Mutates `a` and `b` in place; returns the solution vector.
fn solve_small(a: &mut [Vec<f64>], b: &mut [f64]) -> Vec<f64> {
    let n = b.len();

    // Forward elimination
    for col in 0..n {
        // Partial pivoting
        let mut best = col;
        let mut best_val = a[col][col].abs();
        for row in (col + 1)..n {
            let v = a[row][col].abs();
            if v > best_val {
                best_val = v;
                best = row;
            }
        }
        if best != col {
            a.swap(col, best);
            b.swap(col, best);
        }

        let pivot = a[col][col];
        if pivot.abs() < 1e-30 {
            continue;
        }

        for row in (col + 1)..n {
            let factor = a[row][col] / pivot;
            for j in col..n {
                a[row][j] -= factor * a[col][j];
            }
            b[row] -= factor * b[col];
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i][j] * x[j];
        }
        x[i] = if a[i][i].abs() > 1e-30 {
            sum / a[i][i]
        } else {
            0.0
        };
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sg_recovers_quadratic() {
        // y = 3 + 2x + 0.5x², SG with poly_order >= 2 should recover exactly
        let n = 51;
        let h = 0.1;
        let y: Vec<f64> = (0..n).map(|i| {
            let x = i as f64 * h;
            3.0 + 2.0 * x + 0.5 * x * x
        }).collect();

        let (smoothed, deriv) = sg_smooth_diff(&y, h, 5, 2);

        for i in 0..n {
            let x = i as f64 * h;
            let expected_y = 3.0 + 2.0 * x + 0.5 * x * x;
            let expected_dy = 2.0 + x;
            assert!(
                (smoothed[i] - expected_y).abs() < 1e-10,
                "smoothed[{}]: expected {}, got {}",
                i, expected_y, smoothed[i],
            );
            assert!(
                (deriv[i] - expected_dy).abs() < 1e-8,
                "deriv[{}]: expected {}, got {}",
                i, expected_dy, deriv[i],
            );
        }
    }

    #[test]
    fn test_sg_recovers_cubic() {
        // y = x³, with poly_order=3 should recover exactly
        let n = 41;
        let h = 0.25;
        let y: Vec<f64> = (0..n).map(|i| {
            let x = i as f64 * h;
            x * x * x
        }).collect();

        let (smoothed, deriv) = sg_smooth_diff(&y, h, 4, 3);

        // Check interior points (boundaries may have reduced accuracy)
        for i in 4..(n - 4) {
            let x = i as f64 * h;
            let expected_y = x * x * x;
            let expected_dy = 3.0 * x * x;
            assert!(
                (smoothed[i] - expected_y).abs() < 1e-8,
                "smoothed[{}]: expected {}, got {}",
                i, expected_y, smoothed[i],
            );
            assert!(
                (deriv[i] - expected_dy).abs() < 1e-6,
                "deriv[{}]: expected {}, got {}",
                i, expected_dy, deriv[i],
            );
        }
    }

    #[test]
    fn test_sg_smooths_noisy_signal() {
        // y = sin(x) + noise, SG should reduce noise
        let n = 101;
        let h = 0.1;
        let y_clean: Vec<f64> = (0..n).map(|i| {
            let x = i as f64 * h;
            x.sin()
        }).collect();

        // Add deterministic "noise"
        let y_noisy: Vec<f64> = y_clean.iter().enumerate().map(|(i, &y)| {
            y + 0.1 * if i % 3 == 0 { 1.0 } else if i % 3 == 1 { -1.0 } else { 0.5 }
        }).collect();

        let (smoothed, _) = sg_smooth_diff(&y_noisy, h, 5, 3);

        // Smoothed signal should be closer to clean than noisy is
        let mse_noisy: f64 = y_noisy.iter().zip(y_clean.iter())
            .map(|(a, b)| (a - b).powi(2)).sum::<f64>() / n as f64;
        let mse_smoothed: f64 = smoothed.iter().zip(y_clean.iter())
            .map(|(a, b)| (a - b).powi(2)).sum::<f64>() / n as f64;

        assert!(
            mse_smoothed < mse_noisy * 0.5,
            "SG should reduce noise: MSE noisy={:.4}, smoothed={:.4}",
            mse_noisy, mse_smoothed,
        );
    }

    #[test]
    fn test_sg_boundary_handling() {
        // Constant signal: boundaries should be exact
        let n = 20;
        let h = 1.0;
        let y = vec![5.0; n];

        let (smoothed, deriv) = sg_smooth_diff(&y, h, 5, 2);

        for i in 0..n {
            assert!(
                (smoothed[i] - 5.0).abs() < 1e-12,
                "smoothed[{}] = {} (expected 5.0)", i, smoothed[i],
            );
            assert!(
                deriv[i].abs() < 1e-12,
                "deriv[{}] = {} (expected 0.0)", i, deriv[i],
            );
        }
    }
}

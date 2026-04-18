//! Volume PDF p(V) from the cumulant hierarchy.
//!
//! Two methods are provided:
//!
//! 1. **Saddle-point** (default): Uses the cumulant generating function
//!    K(t) = κ₂t²/2 + κ₃t³/6 + κ₄t⁴/24 and finds the saddle t* for each V.
//!    Guaranteed positive, smooth, and well-behaved at all V.
//!
//! 2. **Edgeworth expansion** at tunable order: truncated asymptotic series
//!    around a Gaussian. Oscillates and goes negative in the tails — useful
//!    for studying convergence, not for production PDF estimates.

use std::f64::consts::PI;
use crate::theory::cumulants::VolumeCumulants;

/// Volume PDF evaluated from the cumulant hierarchy.
#[derive(Debug, Clone)]
pub struct VolumePdf {
    /// Cumulants used to construct this PDF
    pub cumulants: VolumeCumulants,
}

impl VolumePdf {
    pub fn new(cumulants: VolumeCumulants) -> Self {
        Self { cumulants }
    }

    /// Evaluate p(V) via saddle-point approximation (default, positive-definite).
    pub fn evaluate(&self, v: f64) -> f64 {
        self.evaluate_saddlepoint(v)
    }

    /// Saddle-point approximation using the CGF.
    ///
    /// K(t) = κ₂t²/2 + κ₃t³/6 + κ₄t⁴/24  (CGF of W = V − 1)
    /// Find t* with K'(t*) = w, then p(V) = exp(K(t*) − t*w) / √(2π K''(t*)).
    pub fn evaluate_saddlepoint(&self, v: f64) -> f64 {
        let k2 = self.cumulants.kappa2;
        let k3 = self.cumulants.kappa3;
        let k4 = self.cumulants.kappa4;
        if k2 <= 0.0 {
            return 0.0;
        }

        let w = v - 1.0;

        // Newton's method: solve K'(t) = κ₂t + κ₃t²/2 + κ₄t³/6 = w
        let mut t = w / k2;
        for _ in 0..50 {
            let kp = k2 * t + k3 * t * t / 2.0 + k4 * t.powi(3) / 6.0;
            let kpp = k2 + k3 * t + k4 * t * t / 2.0;
            if kpp.abs() < 1e-30 {
                break;
            }
            let dt = (kp - w) / kpp;
            t -= dt;
            if dt.abs() < 1e-14 * (1.0 + t.abs()) {
                break;
            }
        }

        let k_val = k2 * t * t / 2.0 + k3 * t.powi(3) / 6.0 + k4 * t.powi(4) / 24.0;
        let kpp = k2 + k3 * t + k4 * t * t / 2.0;

        if kpp <= 0.0 {
            return 0.0;
        }

        let log_p = k_val - t * w - 0.5 * (2.0 * PI * kpp).ln();
        if log_p < -500.0 {
            return 0.0;
        }
        log_p.exp()
    }

    /// Edgeworth expansion at specified truncation order (0–8).
    ///
    /// The expansion is p(V) = φ(x)/σ × [1 + Σ cₙ Heₙ(x)] where x=(V−1)/σ
    /// and the coefficients come from exp(λ₃t³ + λ₄t⁴):
    ///
    /// | Order | New term(s)                            |
    /// |-------|----------------------------------------|
    /// |   0   | Gaussian                               |
    /// |   1   | + λ₃ He₃                               |
    /// |   2   | + λ₄ He₄                               |
    /// |   3   | + λ₃²/2 He₆                            |
    /// |   4   | + λ₃λ₄ He₇                             |
    /// |   5   | + λ₄²/2 He₈                            |
    /// |   6   | + λ₃³/6 He₉                            |
    /// |   7   | + λ₃²λ₄/2 He₁₀                        |
    /// |   8   | + λ₃λ₄²/2 He₁₁ + (λ₃⁴/24+λ₄³/6) He₁₂ |
    pub fn evaluate_at_order(&self, v: f64, order: usize) -> f64 {
        let sigma_v = self.cumulants.kappa2.sqrt();
        if sigma_v <= 0.0 {
            return 0.0;
        }

        let x = (v - 1.0) / sigma_v;
        let phi = gaussian_density(x);

        let k3 = self.cumulants.kappa3;
        let k4 = self.cumulants.kappa4;
        let s3 = sigma_v.powi(3);
        let s4 = sigma_v.powi(4);

        let lam3 = k3 / (6.0 * s3);
        let lam4 = k4 / (24.0 * s4);

        let mut correction = 1.0;

        if order >= 1 {
            correction += lam3 * hermite(3, x);
        }
        if order >= 2 {
            correction += lam4 * hermite(4, x);
        }
        if order >= 3 {
            correction += lam3 * lam3 / 2.0 * hermite(6, x);
        }
        if order >= 4 {
            correction += lam3 * lam4 * hermite(7, x);
        }
        if order >= 5 {
            correction += lam4 * lam4 / 2.0 * hermite(8, x);
        }
        if order >= 6 {
            correction += lam3.powi(3) / 6.0 * hermite(9, x);
        }
        if order >= 7 {
            correction += lam3 * lam3 * lam4 / 2.0 * hermite(10, x);
        }
        if order >= 8 {
            correction += lam3 * lam4 * lam4 / 2.0 * hermite(11, x);
            correction += (lam3.powi(4) / 24.0 + lam4.powi(3) / 6.0) * hermite(12, x);
        }

        (phi / sigma_v) * correction
    }

    /// Evaluate p(V) on a grid of V values.
    pub fn evaluate_grid(&self, v_values: &[f64]) -> Vec<f64> {
        v_values.iter().map(|&v| self.evaluate(v)).collect()
    }

    /// Generate a default V grid centered on V=1 with extent ±4σ_V.
    pub fn default_grid(&self, n_points: usize) -> Vec<f64> {
        let sigma_v = self.cumulants.kappa2.sqrt();
        let v_min = (1.0 - 4.0 * sigma_v).max(-0.5);
        let v_max = 1.0 + 5.0 * sigma_v;
        let dv = (v_max - v_min) / (n_points - 1) as f64;
        (0..n_points).map(|i| v_min + i as f64 * dv).collect()
    }

    /// Compute ⟨V^m⟩ from the PDF (numerical integration).
    pub fn moment(&self, m: i32, n_points: usize) -> f64 {
        let grid = self.default_grid(n_points);
        let pdf = self.evaluate_grid(&grid);
        let dv = if grid.len() >= 2 {
            grid[1] - grid[0]
        } else {
            return 0.0;
        };

        grid.iter()
            .zip(pdf.iter())
            .map(|(&v, &p)| v.powi(m) * p * dv)
            .sum()
    }

    /// Fraction of probability in multi-stream region (V < 0).
    pub fn multistream_fraction(&self, n_points: usize) -> f64 {
        let sigma_v = self.cumulants.kappa2.sqrt();
        let v_min = -4.0 * sigma_v;
        let dv = -v_min / n_points as f64;

        (0..n_points)
            .map(|i| {
                let v = v_min + (i as f64 + 0.5) * dv;
                self.evaluate(v) * dv
            })
            .sum()
    }
}

/// Standard normal density φ(x) = exp(−x²/2) / √(2π).
fn gaussian_density(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Probabilist's Hermite polynomial He_n(x) via three-term recursion.
///
/// He₀ = 1, He₁ = x, He_{n+1}(x) = x He_n(x) − n He_{n−1}(x).
fn hermite(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return x;
    }
    let mut h_prev = 1.0;
    let mut h_curr = x;
    for k in 1..n {
        let h_next = x * h_curr - k as f64 * h_prev;
        h_prev = h_curr;
        h_curr = h_next;
    }
    h_curr
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theory::spectral::SpectralParams;

    #[test]
    fn test_gaussian_limit() {
        let c = VolumeCumulants {
            kappa1: 0.0,
            kappa2: 0.1,
            kappa3: 0.0,
            kappa4: 0.0,
            lpt_order: 1,
            sigma2: 0.1,
            s3: 0.0,
            s4: 0.0,
        };
        let pdf = VolumePdf::new(c);
        let p_at_mean = pdf.evaluate(1.0);
        let expected = 1.0 / (2.0 * PI * 0.1).sqrt();
        assert!(
            (p_at_mean - expected).abs() / expected < 1e-6,
            "saddle-point at mean: {} vs expected {}", p_at_mean, expected,
        );
    }

    #[test]
    fn test_pdf_normalizes() {
        let sp = SpectralParams {
            mass: 1e12,
            radius: 10.0,
            sigma2: 0.1,
            gamma: 1.0,
            gamma_n: vec![],
        };
        let c = VolumeCumulants::za(&sp);
        let pdf = VolumePdf::new(c);
        let integral = pdf.moment(0, 2000);
        // Saddle-point is an asymptotic approximation, not exactly normalized.
        // At σ²=0.1 it's typically within a few percent.
        assert!(
            (integral - 1.0).abs() < 0.05,
            "PDF integral = {}, should be ≈ 1",
            integral
        );
    }

    #[test]
    fn test_pdf_mean() {
        let sp = SpectralParams {
            mass: 1e12,
            radius: 10.0,
            sigma2: 0.1,
            gamma: 1.0,
            gamma_n: vec![],
        };
        let c = VolumeCumulants::za(&sp);
        let pdf = VolumePdf::new(c);
        let mean = pdf.moment(1, 2000);
        assert!(
            (mean - 1.0).abs() < 0.05,
            "⟨V⟩ = {}, should be ≈ 1",
            mean
        );
    }

    #[test]
    fn test_saddlepoint_positive() {
        let sp = SpectralParams {
            mass: 1e12,
            radius: 10.0,
            sigma2: 0.3,
            gamma: 1.0,
            gamma_n: vec![],
        };
        let c = VolumeCumulants::za(&sp);
        let pdf = VolumePdf::new(c);
        for i in -20..=40 {
            let v = i as f64 * 0.1;
            let p = pdf.evaluate(v);
            assert!(
                p >= 0.0,
                "saddle-point gave negative p({}) = {}",
                v, p
            );
        }
    }

    #[test]
    fn test_edgeworth_order0_is_gaussian() {
        let c = VolumeCumulants {
            kappa1: 0.0,
            kappa2: 0.2,
            kappa3: 0.1,
            kappa4: 0.05,
            lpt_order: 1,
            sigma2: 0.2,
            s3: 0.1 / 0.04,
            s4: 0.05 / 0.008,
        };
        let pdf = VolumePdf::new(c);
        let p0 = pdf.evaluate_at_order(1.0, 0);
        let expected = 1.0 / (2.0 * PI * 0.2).sqrt();
        assert!(
            (p0 - expected).abs() / expected < 1e-10,
            "order 0 at mean: {} vs {}",
            p0, expected,
        );
    }

    #[test]
    fn test_hermite_polynomials() {
        let x = 2.0;
        assert!((hermite(0, x) - 1.0).abs() < 1e-15);
        assert!((hermite(1, x) - 2.0).abs() < 1e-15);
        assert!((hermite(2, x) - 3.0).abs() < 1e-15); // x²−1 = 3
        assert!((hermite(3, x) - 2.0).abs() < 1e-15); // x³−3x = 8−6 = 2
        assert!((hermite(4, x) - (-5.0)).abs() < 1e-14); // x⁴−6x²+3 = 16−24+3 = −5
    }
}

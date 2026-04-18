//! Volume PDF p(V) from the cumulant hierarchy.
//!
//! Reconstructs the probability density of V from its cumulants using
//! the Edgeworth expansion around a Gaussian reference. This is the
//! one-point distribution at fixed Lagrangian mass M.

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

    /// Evaluate p(V) at a single value using the Edgeworth expansion.
    ///
    /// p(V) ≈ φ(x)/σ_V × [1 + (S₃/6) H₃(x) + (S₄/24) H₄(x) + (S₃²/72) H₆(x)]
    ///
    /// where x = (V - 1)/σ_V and φ is the standard normal density.
    pub fn evaluate(&self, v: f64) -> f64 {
        let sigma_v = self.cumulants.kappa2.sqrt();
        if sigma_v <= 0.0 {
            return 0.0;
        }

        let x = (v - 1.0) / sigma_v;
        let phi = gaussian_density(x);
        let s3 = self.cumulants.s3;
        let s4 = self.cumulants.s4;

        let correction = 1.0
            + (s3 / 6.0) * hermite_h3(x)
            + (s4 / 24.0) * hermite_h4(x)
            + (s3 * s3 / 72.0) * hermite_h6(x);

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
        let dv = if grid.len() >= 2 { grid[1] - grid[0] } else { return 0.0 };

        grid.iter().zip(pdf.iter())
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

/// Standard normal density φ(x) = exp(-x²/2) / √(2π).
fn gaussian_density(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Probabilist's Hermite polynomial H₃(x) = x³ - 3x.
fn hermite_h3(x: f64) -> f64 {
    x.powi(3) - 3.0 * x
}

/// Probabilist's Hermite polynomial H₄(x) = x⁴ - 6x² + 3.
fn hermite_h4(x: f64) -> f64 {
    x.powi(4) - 6.0 * x * x + 3.0
}

/// Probabilist's Hermite polynomial H₆(x) = x⁶ - 15x⁴ + 45x² - 15.
fn hermite_h6(x: f64) -> f64 {
    x.powi(6) - 15.0 * x.powi(4) + 45.0 * x * x - 15.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theory::spectral::SpectralParams;

    #[test]
    fn test_gaussian_limit() {
        // When S₃ = S₄ = 0, the PDF should be Gaussian
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
        assert!((p_at_mean - expected).abs() / expected < 1e-10);
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
        assert!((integral - 1.0).abs() < 0.02,
                "PDF integral = {}, should be ≈ 1", integral);
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
        assert!((mean - 1.0).abs() < 0.02,
                "⟨V⟩ = {}, should be ≈ 1", mean);
    }
}

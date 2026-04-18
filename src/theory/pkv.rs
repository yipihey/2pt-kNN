//! The P_k(V) observable: joint distribution of volume and tracer count.
//!
//! P_k(V) is a probability density in V at each fixed integer k ≥ 0.
//! It is the natural observable of kNN statistics: the volume distribution
//! at fixed neighbor count. Its two moment hierarchies — moments of V at
//! fixed k, and factorial moments of k at fixed V — carry cosmological
//! and bias information on orthogonal axes.

use std::f64::consts::PI;
use crate::theory::volume_pdf::VolumePdf;
use crate::theory::bias::LagrangianBias;

/// The P_k(V) observable at a given mass scale.
#[derive(Debug, Clone)]
pub struct PkV {
    /// Volume PDF p(V) at this mass scale
    pub volume_pdf: VolumePdf,
    /// Mean tracer number density n̄_t [h³/Mpc³]
    pub nbar: f64,
    /// Lagrangian bias parameters (for D→D predictions)
    pub bias: Option<LagrangianBias>,
}

impl PkV {
    pub fn new(volume_pdf: VolumePdf, nbar: f64) -> Self {
        Self { volume_pdf, nbar, bias: None }
    }

    pub fn with_bias(mut self, bias: LagrangianBias) -> Self {
        self.bias = Some(bias);
        self
    }

    /// P_k^RD(V) for random-to-data configuration.
    ///
    /// P_k(V) = ∫ dδ_t p(δ_t; V) Π_k(λ(δ_t, V))
    /// where λ = n̄ V (1 + δ_t) and Π_k is Poisson PMF.
    ///
    /// For unbiased tracers (δ_t = 0 everywhere):
    ///   P_k^RD(V) = p(V) × Poisson(k | n̄ V)
    pub fn pk_rd(&self, k: usize, v: f64) -> f64 {
        let pv = self.volume_pdf.evaluate(v);
        if v <= 0.0 {
            return 0.0;
        }
        let lambda = self.nbar * v;
        pv * poisson_pmf(k, lambda)
    }

    /// P_k^DD(V) for data-to-data configuration.
    ///
    /// Includes an extra factor (1 + δ_t(q₀)) for tracer-weighted queries.
    /// At linear order in bias:
    ///   P_k^DD(V) ≈ P_k^RD(V) × [1 + b₁ × ⟨δ_L | V⟩]
    pub fn pk_dd(&self, k: usize, v: f64) -> f64 {
        let p_rd = self.pk_rd(k, v);
        if let Some(ref bias) = self.bias {
            let delta_cond = bias.conditional_delta(v);
            p_rd * (1.0 + delta_cond)
        } else {
            p_rd
        }
    }

    /// Volume moments ⟨V^m⟩_k at fixed k (R→D).
    ///
    /// These map to the volume cumulants κ_m(V; M_k) after Poisson deconvolution.
    pub fn volume_moment_rd(&self, k: usize, m: i32, n_points: usize) -> f64 {
        let grid = self.volume_pdf.default_grid(n_points);
        let dv = if grid.len() >= 2 { grid[1] - grid[0] } else { return 0.0 };

        let mut num = 0.0;
        let mut den = 0.0;
        for &v in &grid {
            let pk = self.pk_rd(k, v);
            num += v.powi(m) * pk * dv;
            den += pk * dv;
        }
        if den > 0.0 { num / den } else { 0.0 }
    }

    /// Factorial moments ⟨k^[n]⟩_V at fixed V (R→D).
    ///
    /// ⟨k^[n]⟩_V = (n̄V)^n × ⟨(1 + δ_t)^n⟩_V
    /// For unbiased tracers: ⟨k^[n]⟩_V = (n̄V)^n.
    pub fn factorial_moment_rd(&self, v: f64, n: usize) -> f64 {
        let lambda = self.nbar * v;
        lambda.powi(n as i32)
    }

    /// DD/RD ratio — direct observable of Lagrangian bias.
    ///
    /// P_k^DD(V) / P_k^RD(V) - 1 = ⟨δ_t | V, k⟩
    pub fn dd_over_rd_ratio(&self, k: usize, v: f64) -> f64 {
        let rd = self.pk_rd(k, v);
        if rd <= 0.0 {
            return 0.0;
        }
        self.pk_dd(k, v) / rd - 1.0
    }

    /// Void-tail linear bias test.
    ///
    /// At large V: log(P_DD/P_RD) → −3b₁(V^{1/3} − 1)
    /// Returns the predicted log-ratio for the void tail.
    pub fn void_tail_log_ratio(&self, v: f64) -> f64 {
        if let Some(ref bias) = self.bias {
            -3.0 * bias.b1 * (v.cbrt() - 1.0)
        } else {
            0.0
        }
    }

    /// Erlang deconvolution: extract clustering signal from factorial moments.
    ///
    /// ⟨k^[n]⟩_V / (n̄V)^n = ⟨(1+δ_t)^n⟩_V
    /// = 1 + n⟨δ_t⟩_V + C(n,2)⟨δ_t²⟩_V + …
    pub fn clustering_signal(&self, v: f64, n: usize) -> f64 {
        if let Some(ref bias) = self.bias {
            let delta = bias.conditional_delta(v);
            let delta2 = bias.conditional_delta2(v);
            match n {
                0 => 1.0,
                1 => 1.0 + delta,
                2 => 1.0 + 2.0 * delta + delta2,
                _ => 1.0 + n as f64 * delta,
            }
        } else {
            1.0
        }
    }
}

/// Poisson probability mass function: Pr(K=k | λ) = λ^k e^{-λ} / k!
fn poisson_pmf(k: usize, lambda: f64) -> f64 {
    if lambda <= 0.0 {
        return if k == 0 { 1.0 } else { 0.0 };
    }
    let log_pmf = k as f64 * lambda.ln() - lambda - log_factorial(k);
    log_pmf.exp()
}

/// ln(k!) via Stirling for large k, exact for small k.
fn log_factorial(k: usize) -> f64 {
    if k <= 20 {
        (1..=k).map(|i| (i as f64).ln()).sum()
    } else {
        let n = k as f64;
        n * n.ln() - n + 0.5 * (2.0 * PI * n).ln()
    }
}

/// Erlang CDF for the k-th nearest neighbor at distance r.
///
/// This is the CDF of the volume containing exactly k points in a Poisson process.
pub fn erlang_cdf_volume(k: usize, v: f64, nbar: f64) -> f64 {
    let lambda = nbar * v;
    if lambda <= 0.0 {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut term = 1.0;
    for j in 0..k {
        if j > 0 {
            term *= lambda / j as f64;
        }
        sum += term;
    }
    1.0 - (-lambda).exp() * sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theory::cumulants::VolumeCumulants;
    use crate::theory::spectral::SpectralParams;

    #[test]
    fn test_poisson_pmf_normalization() {
        let lambda = 5.0;
        let sum: f64 = (0..30).map(|k| poisson_pmf(k, lambda)).sum();
        assert!((sum - 1.0).abs() < 1e-8, "Poisson PMF sum = {}", sum);
    }

    #[test]
    fn test_poisson_pmf_mean() {
        let lambda = 3.0;
        let mean: f64 = (0..30).map(|k| k as f64 * poisson_pmf(k, lambda)).sum();
        assert!((mean - lambda).abs() < 1e-10, "Poisson mean = {}", mean);
    }

    #[test]
    fn test_pk_rd_positive() {
        let sp = SpectralParams {
            mass: 1e12, radius: 10.0, sigma2: 0.1, gamma: 1.0, gamma_n: vec![],
        };
        let c = VolumeCumulants::za(&sp);
        let pdf = VolumePdf::new(c);
        let pkv = PkV::new(pdf, 1e-3);
        let p = pkv.pk_rd(1, 1.0);
        assert!(p > 0.0, "P_1^RD(V=1) should be positive");
    }
}

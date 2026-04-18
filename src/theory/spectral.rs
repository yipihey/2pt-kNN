//! Spectral parameters: σ²(M), γ(M), and higher derivatives.
//!
//! These are the N spectral numbers that parametrize V-cumulants at LPT order N.
//! σ²(M) is the linear variance at mass scale M; γ(M) = −3 d ln σ²/d ln M
//! is the effective spectral slope; γ₂, γ₃, … are its derivatives.

use std::f64::consts::PI;
use crate::theory::cosmology::{CosmologyProvider, tophat_window, mass_to_radius};

/// Spectral parameters at a given mass scale M.
#[derive(Debug, Clone)]
pub struct SpectralParams {
    /// Mass scale M [h⁻¹M☉]
    pub mass: f64,
    /// Tophat radius R [h⁻¹Mpc]
    pub radius: f64,
    /// Linear variance σ²(M) = D²(a) × σ²_lin(M, z=0)
    pub sigma2: f64,
    /// Effective slope γ(M) = −3 d ln σ²/d ln M
    pub gamma: f64,
    /// Higher derivatives γ_n = dⁿγ/d(ln M)ⁿ, indexed as [n-2] → γ₂, γ₃, …
    pub gamma_n: Vec<f64>,
}

impl SpectralParams {
    /// Compute spectral parameters at mass M using the given cosmology at scale factor a.
    ///
    /// `n_derivs` controls how many spectral derivatives beyond γ to compute
    /// (0 = just σ² and γ; 1 = also γ₂; etc.)
    pub fn compute(cosmo: &dyn CosmologyProvider, mass: f64, a: f64, n_derivs: usize) -> Self {
        let rho = cosmo.mean_density();
        let radius = mass_to_radius(mass, rho);

        let sigma2 = compute_sigma2(cosmo, radius, a);
        let gamma = compute_gamma(cosmo, mass, a, rho);

        let gamma_n = (0..n_derivs)
            .map(|n| compute_gamma_n(cosmo, mass, a, rho, n + 2))
            .collect();

        Self { mass, radius, sigma2, gamma, gamma_n }
    }

    /// Get σ (root variance), useful for perturbative expansions.
    pub fn sigma(&self) -> f64 {
        self.sigma2.sqrt()
    }

    /// Get the n-th spectral derivative. γ₀ ≡ γ by convention.
    pub fn spectral_deriv(&self, n: usize) -> f64 {
        match n {
            0 => self.gamma,
            k => self.gamma_n.get(k - 1).copied().unwrap_or(0.0),
        }
    }
}

/// Compute σ²(R, a) via numerical integration of P(k) W²(kR) k² dk / (2π²).
///
/// Uses Gauss-Legendre quadrature in ln k for better sampling of the oscillatory integrand.
fn compute_sigma2(cosmo: &dyn CosmologyProvider, radius: f64, a: f64) -> f64 {
    let lnk_min = -7.0_f64;
    let lnk_max = (50.0 / radius).ln().max(3.0);
    let n_points = 512;

    let dlnk = (lnk_max - lnk_min) / n_points as f64;
    let mut sum = 0.0;

    for i in 0..n_points {
        let lnk = lnk_min + (i as f64 + 0.5) * dlnk;
        let k = lnk.exp();
        let kr = k * radius;
        let w = tophat_window(kr);
        let pk = cosmo.power_spectrum(k, a);
        sum += k * k * k * pk * w * w * dlnk;
    }

    sum / (2.0 * PI * PI)
}

/// Compute γ(M) = −3 d ln σ²/d ln M via central finite difference.
fn compute_gamma(cosmo: &dyn CosmologyProvider, mass: f64, a: f64, rho: f64) -> f64 {
    let eps = 0.01;
    let m_plus = mass * (1.0 + eps);
    let m_minus = mass * (1.0 - eps);
    let r_plus = mass_to_radius(m_plus, rho);
    let r_minus = mass_to_radius(m_minus, rho);

    let s2_plus = compute_sigma2(cosmo, r_plus, a);
    let s2_minus = compute_sigma2(cosmo, r_minus, a);

    -3.0 * (s2_plus.ln() - s2_minus.ln()) / (m_plus.ln() - m_minus.ln())
}

/// Compute higher spectral derivatives γ_n via iterated finite differences.
fn compute_gamma_n(cosmo: &dyn CosmologyProvider, mass: f64, a: f64, rho: f64, n: usize) -> f64 {
    let eps = 0.02;
    let m_plus = mass * (1.0 + eps);
    let m_minus = mass * (1.0 - eps);

    if n <= 1 {
        return compute_gamma(cosmo, mass, a, rho);
    }

    let g_plus = compute_gamma_n(cosmo, m_plus, a, rho, n - 1);
    let g_minus = compute_gamma_n(cosmo, m_minus, a, rho, n - 1);

    (g_plus - g_minus) / (m_plus.ln() - m_minus.ln())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theory::cosmology::{SyrenCosmology, CosmoParams};

    #[test]
    fn test_sigma2_positive() {
        let cosmo = SyrenCosmology::new(CosmoParams::planck2018());
        let sp = SpectralParams::compute(&cosmo, 1e12, 1.0, 0);
        assert!(sp.sigma2 > 0.0, "σ²(M) must be positive, got {}", sp.sigma2);
    }

    #[test]
    fn test_sigma2_decreases_with_mass() {
        let cosmo = SyrenCosmology::new(CosmoParams::planck2018());
        let sp_small = SpectralParams::compute(&cosmo, 1e10, 1.0, 0);
        let sp_large = SpectralParams::compute(&cosmo, 1e14, 1.0, 0);
        assert!(sp_small.sigma2 > sp_large.sigma2,
                "σ²(M) should decrease with M: σ²(10¹⁰)={} vs σ²(10¹⁴)={}",
                sp_small.sigma2, sp_large.sigma2);
    }

    #[test]
    fn test_gamma_positive() {
        let cosmo = SyrenCosmology::new(CosmoParams::planck2018());
        let sp = SpectralParams::compute(&cosmo, 1e12, 1.0, 0);
        assert!(sp.gamma > 0.0, "γ(M) should be positive for CDM, got {}", sp.gamma);
    }
}

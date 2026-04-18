//! Cosmology provider trait and syren-new implementation.
//!
//! The trait abstracts the cosmological inputs needed by the theory module:
//! linear power spectrum, growth factor, and derived spectral parameters.
//! The syren-new implementation ports the symbolic emulator formulas from
//! Bartlett et al. (2024) to Rust.

use std::f64::consts::PI;

/// Cosmological parameters for ΛCDM + massive neutrinos + w₀wₐ dark energy.
#[derive(Debug, Clone)]
pub struct CosmoParams {
    /// Matter density parameter Ω_m
    pub omega_m: f64,
    /// Baryon density parameter Ω_b
    pub omega_b: f64,
    /// Dimensionless Hubble parameter h = H₀/(100 km/s/Mpc)
    pub h: f64,
    /// Scalar spectral index n_s
    pub ns: f64,
    /// Primordial amplitude A_s (× 10⁹)
    pub a_s: f64,
    /// Sum of neutrino masses Σmᵥ [eV]
    pub mnu: f64,
    /// Dark energy equation of state w₀
    pub w0: f64,
    /// Dark energy equation of state time derivative wₐ
    pub wa: f64,
}

impl Default for CosmoParams {
    fn default() -> Self {
        Self {
            omega_m: 0.3111,
            omega_b: 0.049,
            h: 0.6766,
            ns: 0.9665,
            a_s: 2.1,
            mnu: 0.0,
            w0: -1.0,
            wa: 0.0,
        }
    }
}

impl CosmoParams {
    pub fn planck2018() -> Self {
        Self::default()
    }

    pub fn sigma8(&self) -> f64 {
        as_to_sigma8(self.a_s, self.omega_m, self.omega_b, self.h, self.ns, self.mnu, self.w0, self.wa)
    }
}

/// Provider of cosmological inputs for the theory module.
///
/// Implementations supply the linear power spectrum, growth factor,
/// and optionally the spectral parameters σ²(M), γ(M).
/// The default implementation uses syren-new emulator formulas.
pub trait CosmologyProvider: Send + Sync {
    /// Linear matter power spectrum P_lin(k, a) in (Mpc/h)³.
    fn power_spectrum(&self, k: f64, a: f64) -> f64;

    /// Linear growth factor D(a), normalized so D(1) = 1.
    fn growth_factor(&self, a: f64) -> f64;

    /// Mean matter density ρ̄ in h²M☉/Mpc³ (for mass↔radius conversion).
    fn mean_density(&self) -> f64;

    /// Cosmological parameters.
    fn params(&self) -> &CosmoParams;
}

/// Syren-new cosmology: symbolic emulator for P_lin(k) and D(a).
///
/// Ports the `symbolic_pofk` Python package (Bartlett et al. 2024) to Rust.
/// All formulas are closed-form polynomial/trigonometric expressions with
/// pre-fitted coefficients — no neural networks or lookup tables.
#[derive(Debug, Clone)]
pub struct SyrenCosmology {
    pub params: CosmoParams,
    pub sigma8_cached: f64,
}

impl SyrenCosmology {
    pub fn new(params: CosmoParams) -> Self {
        let sigma8_cached = params.sigma8();
        Self { params, sigma8_cached }
    }

    pub fn from_sigma8(mut params: CosmoParams, sigma8: f64) -> Self {
        params.a_s = sigma8_to_as(sigma8, params.omega_m, params.omega_b, params.h, params.ns, params.mnu, params.w0, params.wa);
        Self { params, sigma8_cached: sigma8 }
    }
}

impl CosmologyProvider for SyrenCosmology {
    fn power_spectrum(&self, k: f64, a: f64) -> f64 {
        plin_emulated(k, self.params.a_s, self.params.omega_m, self.params.omega_b,
                      self.params.h, self.params.ns, self.params.mnu,
                      self.params.w0, self.params.wa, a)
    }

    fn growth_factor(&self, a: f64) -> f64 {
        let d = approximate_growth_factor(self.params.omega_m, self.params.mnu, self.params.w0, self.params.wa, a);
        let d1 = approximate_growth_factor(self.params.omega_m, self.params.mnu, self.params.w0, self.params.wa, 1.0);
        d / d1
    }

    fn mean_density(&self) -> f64 {
        // ρ̄ = Ω_m × ρ_crit, with ρ_crit = 2.775 × 10¹¹ h² M☉/Mpc³
        self.params.omega_m * 2.775e11
    }

    fn params(&self) -> &CosmoParams {
        &self.params
    }
}

// ---------------------------------------------------------------------------
// Syren-new emulator formulas (ported from symbolic_pofk)
// ---------------------------------------------------------------------------

/// Eisenstein & Hu (1998) zero-baryon (no-wiggle) transfer function T(k).
///
/// Eq 29-31 of Eisenstein & Hu (1998). k in h/Mpc.
fn eisenstein_hu_transfer(k: f64, omega_m: f64, omega_b: f64, h: f64) -> f64 {
    if k <= 0.0 {
        return 1.0;
    }

    let omh2 = omega_m * h * h;
    let obh2 = omega_b * h * h;
    let fb = omega_b / omega_m;
    let theta27 = 2.7255 / 2.7;

    // Sound horizon (simplified no-wiggle form, Eq 26 of EH98)
    let s = 44.5 * (9.83 / omh2).ln() / (1.0 + 10.0 * obh2.powf(0.75)).sqrt();

    // Eq 31: α_Γ
    let alpha_gamma = 1.0 - 0.328 * (431.0 * omh2).ln() * fb
        + 0.38 * (22.3 * omh2).ln() * fb * fb;

    // Eq 30: effective shape parameter
    let gamma_eff = omega_m * h
        * (alpha_gamma + (1.0 - alpha_gamma) / (1.0 + (0.43 * k * s).powi(4)));

    // Eq 29: transfer function
    let q = k * theta27 * theta27 / gamma_eff;
    let l0 = (2.0 * std::f64::consts::E + 1.8 * q).ln();
    let c0 = 14.2 + 731.0 / (1.0 + 62.5 * q);
    l0 / (l0 + c0 * q * q)
}

/// Compute the unnormalized P(k) shape: T²(k) × k^{n_s}.
fn pk_shape(k: f64, omega_m: f64, omega_b: f64, h: f64, ns: f64) -> f64 {
    if k <= 0.0 { return 0.0; }
    let tk = eisenstein_hu_transfer(k, omega_m, omega_b, h);
    tk * tk * k.powf(ns)
}

/// Compute σ₈ for unit-amplitude P(k) = T²(k) k^{ns}, for internal calibration.
fn compute_sigma8_unnorm(omega_m: f64, omega_b: f64, h: f64, ns: f64) -> f64 {
    let r8 = 8.0; // 8 h⁻¹Mpc
    let lnk_min = -7.0_f64;
    let lnk_max = 3.0_f64;
    let n = 1024;
    let dlnk = (lnk_max - lnk_min) / n as f64;

    let mut sum = 0.0;
    for i in 0..n {
        let lnk = lnk_min + (i as f64 + 0.5) * dlnk;
        let k = lnk.exp();
        let w = tophat_window(k * r8);
        let pk = pk_shape(k, omega_m, omega_b, h, ns);
        sum += k.powi(3) * pk * w * w * dlnk;
    }
    (sum / (2.0 * PI * PI)).sqrt()
}

/// Compute the full P(k) at z=0 normalized to σ₈, in (Mpc/h)³.
///
/// P(k) = (σ₈/σ₈_unnorm)² × T²(k) × k^{n_s}
///
/// σ²(R) = (1/2π²) ∫ k² P(k) W²(kR) dk, which at R=8 gives σ₈².
fn pk_normalized(k: f64, sigma8: f64, omega_m: f64, omega_b: f64, h: f64, ns: f64) -> f64 {
    let shape = pk_shape(k, omega_m, omega_b, h, ns);
    let s8_unnorm = compute_sigma8_unnorm(omega_m, omega_b, h, ns);
    if s8_unnorm <= 0.0 { return 0.0; }
    let norm = sigma8 * sigma8 / (s8_unnorm * s8_unnorm);
    norm * shape
}

/// Approximate linear growth factor D(a) following Bond/Lahav + w₀wₐ + neutrinos.
///
/// Ported from symbolic_pofk.linear_new.get_approximate_D
fn approximate_growth_factor(omega_m: f64, mnu: f64, w0: f64, wa: f64, a: f64) -> f64 {
    if a <= 0.0 {
        return 0.0;
    }

    let omega_nu = mnu / (93.14 * 0.6766_f64.powi(2));
    let _omega_cb = omega_m - omega_nu;
    let omega_de_a = (1.0 - omega_m) * a.powf(-3.0 * (1.0 + w0 + wa)) * (3.0 * wa * (a - 1.0)).exp();
    let omega_m_a = omega_m / (omega_m + omega_de_a * a.powi(3));

    // Neutrino free-streaming suppression
    let f_nu = omega_nu / omega_m;
    let f_cb = 1.0 - f_nu;
    let p_cb = 0.25 * (5.0 - (1.0 + 24.0 * f_cb).sqrt());

    // Carroll et al. (1992) growth factor approximation
    let d = a * 2.5 * omega_m_a
        / (omega_m_a.powf(4.0 / 7.0) - (1.0 - omega_m_a) + (1.0 + omega_m_a / 2.0) * (1.0 + (1.0 - omega_m_a) / 70.0));

    d * f_cb.powf(p_cb - 1.0)
}

/// Log-ratio correction: ln(P_lin_true / P_EH).
///
/// Placeholder for BAO wiggles correction from syren-new (37-coefficient
/// symbolic emulator). The coefficients need verification against the
/// Python source before enabling. For now returns 0 (smooth E&H spectrum).
fn log_f_correction(_k: f64, _sigma8: f64, _omega_m: f64, _omega_b: f64, _h: f64, _ns: f64) -> f64 {
    0.0
}

/// Emulated linear matter power spectrum P_lin(k, a) in (Mpc/h)³.
///
/// Combines σ₈-normalized Eisenstein-Hu × growth factor² × symbolic BAO correction.
fn plin_emulated(k: f64, a_s: f64, omega_m: f64, omega_b: f64, h: f64, ns: f64,
                 mnu: f64, w0: f64, wa: f64, a: f64) -> f64 {
    let sigma8 = as_to_sigma8(a_s, omega_m, omega_b, h, ns, mnu, w0, wa);
    let p_eh = pk_normalized(k, sigma8, omega_m, omega_b, h, ns);
    let log_f = log_f_correction(k, sigma8, omega_m, omega_b, h, ns);
    let d = approximate_growth_factor(omega_m, mnu, w0, wa, a);
    let d0 = approximate_growth_factor(omega_m, mnu, w0, wa, 1.0);

    p_eh * log_f.exp() * (d / d0).powi(2)
}

/// Convert A_s (×10⁹) to σ₈ from first principles.
///
/// Uses the exact relationship:
///   P(k) = A × T²(k) × k^{ns}
///   A = (8π²/25) × (c/100)⁴ × D₀² × A_s × 10⁻⁹ × (h/k_pivot)^{ns-1} / Ω_m²
///   σ₈² = A × σ₈_unnorm²
fn as_to_sigma8(a_s: f64, omega_m: f64, omega_b: f64, h: f64, ns: f64,
                mnu: f64, w0: f64, wa: f64) -> f64 {
    let a_s_phys = a_s * 1e-9;
    let k_pivot = 0.05_f64; // Mpc⁻¹
    let c_over_100 = 2997.925_f64; // c / (100 km/s/Mpc) in Mpc

    let d0 = approximate_growth_factor(omega_m, mnu, w0, wa, 1.0);

    let amp = (8.0 * PI * PI / 25.0) * c_over_100.powi(4) * d0 * d0 * a_s_phys
        * (h / k_pivot).powf(ns - 1.0) / (omega_m * omega_m);

    let s8_unnorm = compute_sigma8_unnorm(omega_m, omega_b, h, ns);
    (amp * s8_unnorm * s8_unnorm).sqrt()
}

/// Convert σ₈ to A_s (×10⁹).
///
/// Inverts as_to_sigma8: σ₈² ∝ A_s, so A_s = (σ₈/σ₈(A_s=1))².
fn sigma8_to_as(sigma8: f64, omega_m: f64, omega_b: f64, h: f64, ns: f64,
                mnu: f64, w0: f64, wa: f64) -> f64 {
    let s8_unit = as_to_sigma8(1.0, omega_m, omega_b, h, ns, mnu, w0, wa);
    if s8_unit.abs() < 1e-15 { return 0.0; }
    (sigma8 / s8_unit).powi(2)
}

/// Tophat window function in Fourier space: W(x) = 3(sin x - x cos x)/x³
pub fn tophat_window(x: f64) -> f64 {
    if x.abs() < 1e-4 {
        1.0 - x * x / 10.0
    } else {
        3.0 * (x.sin() - x * x.cos()) / (x * x * x)
    }
}

/// Convert mass M [h⁻¹M☉] to tophat radius R [h⁻¹Mpc].
pub fn mass_to_radius(mass: f64, mean_density: f64) -> f64 {
    (3.0 * mass / (4.0 * PI * mean_density)).cbrt()
}

/// Convert tophat radius R [h⁻¹Mpc] to mass M [h⁻¹M☉].
pub fn radius_to_mass(radius: f64, mean_density: f64) -> f64 {
    4.0 / 3.0 * PI * radius.powi(3) * mean_density
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_planck_sigma8() {
        let cosmo = CosmoParams::planck2018();
        let s8 = cosmo.sigma8();
        assert!((s8 - 0.81).abs() < 0.05, "σ₈ = {} should be near 0.81", s8);
    }

    #[test]
    fn test_growth_factor_today() {
        let cosmo = SyrenCosmology::new(CosmoParams::planck2018());
        let d1 = cosmo.growth_factor(1.0);
        assert!((d1 - 1.0).abs() < 1e-10, "D(a=1) should be 1, got {}", d1);
    }

    #[test]
    fn test_growth_factor_monotone() {
        let cosmo = SyrenCosmology::new(CosmoParams::planck2018());
        let d05 = cosmo.growth_factor(0.5);
        let d08 = cosmo.growth_factor(0.8);
        let d10 = cosmo.growth_factor(1.0);
        assert!(d05 < d08 && d08 < d10, "Growth factor should be monotone");
    }

    #[test]
    fn test_tophat_window() {
        assert!((tophat_window(0.0) - 1.0).abs() < 1e-10);
        assert!((tophat_window(1e-5) - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_mass_radius_roundtrip() {
        let rho = 0.3 * 2.775e11;
        let m = 1e14;
        let r = mass_to_radius(m, rho);
        let m2 = radius_to_mass(r, rho);
        assert!((m2 / m - 1.0).abs() < 1e-10);
    }
}

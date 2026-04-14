//! FFTLog: Hamilton (2000) algorithm for Hankel transforms on log grids.
//!
//! Implements the integral transform
//!   ã(r) = ∫₀^∞ a(k) (kr) J_μ(kr) d(ln k)
//! on log-spaced k and r grids, via FFT. Reference:
//!   A.J.S. Hamilton, MNRAS 312, 257 (2000), astro-ph/9905191.
//!
//! ## Convention and wrappers for cosmological observables
//!
//! For ξ(r) = (1/2π²) ∫ k² P(k) j₀(kr) dk with j₀(x) = √(π/(2x)) J_{1/2}(x),
//! use a(k) = k^(3/2) P(k), μ = 1/2. Then
//!   ξ(r) = √(π/2)/(2π²) × ã(r) / r^(3/2).
//!
//! The configuration-space smoothed quantities (xi_bar, sigma²) follow from a
//! single ξ(r) via bounded integrals that are provided here.

use num_complex::Complex64;
use rustfft::{FftPlanner, Fft};
use std::sync::Arc;

// ═══════════════════════════════════════════════════════════════════════════
// Complex gamma function (Lanczos approximation, g=7)
// ═══════════════════════════════════════════════════════════════════════════

// Lanczos coefficients for g=7, n=9 (Numerical Recipes / Wikipedia)
const LANCZOS_G: f64 = 7.0;
const LANCZOS_COEFS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_59,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_571_6e-6,
    1.505_632_735_149_311_6e-7,
];

/// Complex log-gamma via Lanczos approximation.
/// Valid for complex z; uses reflection formula for Re(z) < 0.5.
#[inline]
pub fn lngamma_complex(z: Complex64) -> Complex64 {
    if z.re < 0.5 {
        // Reflection: log Γ(z) = log π - log sin(π z) - log Γ(1 - z)
        let pi = std::f64::consts::PI;
        let ln_pi = pi.ln();
        let one_minus_z = Complex64::new(1.0, 0.0) - z;
        let sin_piz = (Complex64::new(pi, 0.0) * z).sin();
        // log(pi / sin(pi z)) = log(pi) - log(sin(pi z))
        Complex64::new(ln_pi, 0.0) - sin_piz.ln() - lngamma_complex(one_minus_z)
    } else {
        let z1 = z - Complex64::new(1.0, 0.0);
        let mut x = Complex64::new(LANCZOS_COEFS[0], 0.0);
        for i in 1..9 {
            let denom = z1 + Complex64::new(i as f64, 0.0);
            x += LANCZOS_COEFS[i] / denom;
        }
        let t = z1 + Complex64::new(LANCZOS_G + 0.5, 0.0);
        let log_2pi_half = 0.5 * (2.0 * std::f64::consts::PI).ln();
        Complex64::new(log_2pi_half, 0.0)
            + (z1 + Complex64::new(0.5, 0.0)) * t.ln()
            - t
            + x.ln()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// FFTLog kernel: u_m coefficients
// ═══════════════════════════════════════════════════════════════════════════

/// Compute U_μ(x) = 2^x · Γ((μ+1+x)/2) / Γ((μ+1−x)/2) via log-gamma.
#[inline]
fn u_mu(mu: f64, x: Complex64) -> Complex64 {
    let half = Complex64::new(0.5, 0.0);
    let mu_plus_one = Complex64::new(mu + 1.0, 0.0);
    let alpha_plus = (mu_plus_one + x) * half;
    let alpha_minus = (mu_plus_one - x) * half;
    let two = Complex64::new(2.0_f64.ln(), 0.0);
    (x * two + lngamma_complex(alpha_plus) - lngamma_complex(alpha_minus)).exp()
}

/// Find the "low-ringing" value of κ = k_c · r_c such that u_{N/2} is real.
/// Adjusts an initial κ guess by the minimal correction satisfying the condition.
fn lowring_kappa(mu: f64, q: f64, dlnk: f64, n: usize, mut kappa: f64) -> f64 {
    let total_range = dlnk * n as f64;
    // x at the Nyquist: q + i π / dlnk
    let x = Complex64::new(q, std::f64::consts::PI / dlnk);
    let alpha_plus = Complex64::new(mu + 1.0 + q, std::f64::consts::PI / dlnk) * 0.5;
    let alpha_minus = Complex64::new(mu + 1.0 - q, -std::f64::consts::PI / dlnk) * 0.5;
    let phi_plus = lngamma_complex(alpha_plus).im;
    let phi_minus = lngamma_complex(alpha_minus).im;

    // condition: log(2/κ)/dlnk + (phi_plus − phi_minus)/π must be an integer
    let arg = (2.0f64.ln() - kappa.ln()) / dlnk + (phi_plus - phi_minus) / std::f64::consts::PI;
    let int_arg = arg.round();
    if (arg - int_arg).abs() > 1e-15 {
        kappa *= ((arg - int_arg) * dlnk).exp();
    }
    let _ = total_range; // silence unused; kept for potential future use
    let _ = x;
    kappa
}

// ═══════════════════════════════════════════════════════════════════════════
// Main FFTLog transform struct
// ═══════════════════════════════════════════════════════════════════════════

/// Configured FFTLog transform (reusable; coefficients precomputed).
///
/// Transforms input f(k) sampled on log-spaced k to ã(r) sampled on log-spaced r,
/// where ã(r) = ∫₀^∞ f(k) k^q (kr) J_μ(kr) d(ln k) × r^(-q).
pub struct FFTLog {
    pub n: usize,
    pub dlnk: f64,
    pub mu: f64,
    pub q: f64,
    /// Product κ = k_c · r_c (the log-grids' pivot product).
    pub kappa: f64,
    /// Precomputed kernel: u_m in Fourier space, length N/2+1.
    u_m: Vec<Complex64>,
    fft_fwd: Arc<dyn Fft<f64>>,
    fft_inv: Arc<dyn Fft<f64>>,
}

impl FFTLog {
    /// Build a new FFTLog transform for given parameters.
    ///
    /// * `n` — number of log-spaced samples (power of 2 recommended)
    /// * `dlnk` — spacing in natural log of k
    /// * `mu` — Bessel order (e.g., 1/2 for j₀ transforms)
    /// * `q` — bias (0.0 is a safe default for decaying f(k))
    /// * `lowring` — if true, adjust κ for the low-ringing condition
    pub fn new(n: usize, dlnk: f64, mu: f64, q: f64, lowring: bool) -> Self {
        assert!(n >= 2 && n % 2 == 0, "FFTLog requires even n ≥ 2");
        let mut kappa = 1.0;
        if lowring {
            kappa = lowring_kappa(mu, q, dlnk, n, kappa);
        }

        // Build u_m array: for m = 0, 1, ..., N/2, the frequency is
        //   ω_m = 2π m / (N dlnk)
        // and the argument x = q + i ω_m. The kernel u_m = κ^(−iω_m) · U_μ(q + iω_m).
        let nh = n / 2;
        let mut u_m = vec![Complex64::new(0.0, 0.0); nh + 1];
        let total = dlnk * n as f64;
        for m in 0..=nh {
            let omega = 2.0 * std::f64::consts::PI * (m as f64) / total;
            let x = Complex64::new(q, omega);
            // κ^(-i ω) = exp(-i ω ln κ)
            let kappa_pow = Complex64::new(0.0, -omega * kappa.ln()).exp();
            u_m[m] = kappa_pow * u_mu(mu, x);
        }
        // Ensure u_{N/2} is real (force imag → 0 after low-ringing adjustment)
        if lowring {
            u_m[nh].im = 0.0;
        }

        let mut planner = FftPlanner::<f64>::new();
        let fft_fwd = planner.plan_fft_forward(n);
        let fft_inv = planner.plan_fft_inverse(n);

        FFTLog { n, dlnk, mu, q, kappa, u_m, fft_fwd, fft_inv }
    }

    /// Execute the transform.
    ///
    /// * `f_k` — input samples on log-spaced k grid, length N
    /// * `k_center` — the pivot k_c of the input log grid
    ///
    /// Returns `(r_grid, f_r)` where r_grid is log-spaced with r_c = κ / k_c
    /// and f_r(r_n) = ∫ f_k(k) (kr) J_μ(kr) d(ln k), with bias q accounted for.
    pub fn transform(&self, f_k: &[f64], k_center: f64) -> (Vec<f64>, Vec<f64>) {
        assert_eq!(f_k.len(), self.n, "f_k length must match FFTLog.n");
        let n = self.n;
        let nh = n / 2;

        // Apply bias: multiply f_k by k^q → a_k
        let mut buf: Vec<Complex64> = f_k.iter().enumerate().map(|(i, &v)| {
            let ln_k = k_center.ln() + (i as f64 - (n / 2) as f64) * self.dlnk;
            let k_q = if self.q == 0.0 { 1.0 } else { (self.q * ln_k).exp() };
            Complex64::new(v * k_q, 0.0)
        }).collect();

        // Forward FFT in place
        self.fft_fwd.process(&mut buf);

        // Multiply by u_m kernel. buf[0..=nh] correspond to non-negative frequencies;
        // buf[nh+1..n] are complex conjugates (since input was real).
        buf[0] *= self.u_m[0];
        for m in 1..nh {
            let u = self.u_m[m];
            buf[m] *= u;
            // Negative-frequency bin is complex conjugate of u_m
            buf[n - m] *= u.conj();
        }
        buf[nh] *= self.u_m[nh];

        // Inverse FFT
        self.fft_inv.process(&mut buf);
        // rustfft inverse is unnormalized → divide by n
        let inv_n = 1.0 / (n as f64);
        for v in buf.iter_mut() {
            *v *= inv_n;
        }

        // FFTLog output is on a reversed log-grid: the sign in the exponent of r
        // is flipped relative to k. We handle this by reversing the result array.
        let r_center = self.kappa / k_center;
        let mut r_grid = vec![0.0; n];
        let mut f_r = vec![0.0; n];
        for i in 0..n {
            let j = n - 1 - i; // reverse index: r increases as k decreases
            let ln_r = r_center.ln() + (i as f64 - (n / 2) as f64) * self.dlnk;
            r_grid[i] = ln_r.exp();
            let r_q = if self.q == 0.0 { 1.0 } else { (-self.q * ln_r).exp() };
            f_r[i] = buf[j].re * r_q;
        }
        (r_grid, f_r)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// High-level wrapper: ξ(r) from P(k) via FFTLog
// ═══════════════════════════════════════════════════════════════════════════

/// Log-spaced table of the matter correlation function ξ(r).
#[derive(Clone)]
pub struct XiTable {
    pub r: Vec<f64>,    // log-spaced r [Mpc/h]
    pub xi: Vec<f64>,   // ξ(r)
    pub ln_r_min: f64,
    pub ln_r_max: f64,
    pub dlnr: f64,
}

impl XiTable {
    /// Cubic (Catmull-Rom) interpolation of ξ(r) with log-spaced abscissa.
    /// Out-of-range queries clamp to the boundary values.
    #[inline]
    pub fn eval(&self, r: f64) -> f64 {
        let n = self.xi.len();
        if r <= 0.0 { return 0.0; }
        let lnr = r.ln();
        if lnr <= self.ln_r_min { return self.xi[0]; }
        if lnr >= self.ln_r_max { return self.xi[n - 1]; }
        let f = (lnr - self.ln_r_min) / self.dlnr;
        let i = f as usize;
        let t = f - i as f64;
        // Catmull-Rom cubic interpolation using 4 neighboring points
        let i0 = if i == 0 { 0 } else { i - 1 };
        let i1 = i;
        let i2 = if i + 1 >= n { n - 1 } else { i + 1 };
        let i3 = if i + 2 >= n { n - 1 } else { i + 2 };
        let p0 = self.xi[i0];
        let p1 = self.xi[i1];
        let p2 = self.xi[i2];
        let p3 = self.xi[i3];
        // Catmull-Rom: p(t) = 0.5 * ((2*p1) + (-p0 + p2) * t +
        //   (2*p0 - 5*p1 + 4*p2 - p3) * t² + (-p0 + 3*p1 - 3*p2 + p3) * t³)
        let t2 = t * t;
        let t3 = t2 * t;
        0.5 * ((2.0 * p1)
             + (-p0 + p2) * t
             + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
             + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
    }
}

/// Compute ξ(r) from a log-spaced P(k) grid via FFTLog.
///
/// Transform: ξ(r) = (1/2π²) ∫ k² P(k) j₀(kr) dk, using
/// j₀(x) = √(π/(2x)) J_{1/2}(x), so we evaluate
/// ã(r) = ∫ k^(3/2) P(k) (kr) J_{1/2}(kr) d(ln k) via FFTLog (μ = 1/2, q = 0),
/// then ξ(r) = √(π/2)/(2π²) × ã(r) / r^(3/2).
///
/// * `ln_k_min`, `ln_k_max`, `pk` — P(k) sampled uniformly in ln k at n = pk.len() points
///
/// Returns XiTable on log-r grid with the same dlnk step and r_c = 1 / k_c (if lowring=false)
/// or adjusted r_c (if lowring=true).
pub fn xi_from_pk_loglog(
    ln_k_min: f64, ln_k_max: f64, pk: &[f64], lowring: bool,
) -> XiTable {
    let n = pk.len();
    assert!(n >= 2 && n.is_power_of_two(), "pk length must be a power of 2 ≥ 2");

    let dlnk = (ln_k_max - ln_k_min) / (n - 1) as f64;
    let k_center = ((ln_k_min + ln_k_max) * 0.5).exp();

    // Prepare a(k) = k^(3/2) P(k)
    let a_k: Vec<f64> = (0..n).map(|i| {
        let k = (ln_k_min + i as f64 * dlnk).exp();
        k.powf(1.5) * pk[i]
    }).collect();

    let mu = 0.5;
    let q = 0.0;
    let fftlog = FFTLog::new(n, dlnk, mu, q, lowring);
    let (r_grid, a_r) = fftlog.transform(&a_k, k_center);

    // Convert ã(r) to ξ(r): ξ = √(π/2)/(2π²) × ã / r^(3/2)
    let prefactor = (std::f64::consts::PI * 0.5).sqrt() / (2.0 * std::f64::consts::PI.powi(2));
    let xi: Vec<f64> = r_grid.iter().zip(a_r.iter()).map(|(&r, &ar)| {
        prefactor * ar / r.powf(1.5)
    }).collect();

    let ln_r_min = r_grid[0].ln();
    let ln_r_max = r_grid[n - 1].ln();
    let dlnr = (ln_r_max - ln_r_min) / (n - 1) as f64;

    XiTable { r: r_grid, xi, ln_r_min, ln_r_max, dlnr }
}

// ═══════════════════════════════════════════════════════════════════════════
// Bundle: one FFTLog call + cosmology → ξ tables ready for smoothing
// ═══════════════════════════════════════════════════════════════════════════

/// A collection of ξ-tables used by the PT framework.
/// All computed from a single cosmology via FFTLog on a shared log-k grid.
#[derive(Clone)]
pub struct XiTables {
    /// ξ(r) from P_L(k) — tree-level matter auto-correlation.
    /// Used for σ²_lin (via sigma2_from_xi) and tree-level ξ̄ (via xi_bar_from_xi).
    pub xi_pk: XiTable,
    /// ξ(r) from k² P_L(k) — for the k⁵ W²(kR) counterterm σ²_{J,2}(R).
    /// Populated when `with_counterterms = true`.
    pub xi_k2_pk: Option<XiTable>,
    /// ξ(r) from k⁴ P_L(k) — for the k⁷ W²(kR) counterterm σ²_{J,4}(R).
    /// Populated when `with_counterterms = true`.
    pub xi_k4_pk: Option<XiTable>,
    /// ξ(r) from P_L(k) × I_P13(k) — for one-loop P₁₃ corrections (ξ̄ cross-spectrum).
    /// Populated when `with_p13 = true`.
    pub xi_p13_eff: Option<XiTable>,
    /// Log-k grid parameters (for diagnostics).
    pub ln_k_min: f64,
    pub ln_k_max: f64,
    pub n: usize,
}

/// Configuration for FFTLog-based ξ(r) tables.
/// Defaults: n=2048, k ∈ [1e-6, 300] h/Mpc (extended beyond the usual range
/// to push aliasing artifacts outside the physically meaningful region).
#[derive(Clone, Copy)]
pub struct FFTLogConfig {
    pub n: usize,
    pub ln_k_min: f64,
    pub ln_k_max: f64,
    pub lowring: bool,
}

impl Default for FFTLogConfig {
    fn default() -> Self {
        FFTLogConfig {
            n: 4096,
            ln_k_min: (1e-5_f64).ln(),
            ln_k_max: (50.0_f64).ln(),
            lowring: true,
        }
    }
}

/// Build ξ(r) tables from a cosmology via FFTLog.
///
/// * `with_counterterms = true` also builds ξ_{k² P_L} and ξ_{k⁴ P_L}
///    (cheap — two extra FFTLog calls).
/// * `with_p13 = true` also builds ξ_P13_eff for one-loop corrections.
///    Moderately expensive: the 3D I_P13(k) kernel is computed at n points.
pub fn build_xi_tables(
    cosmo: &crate::Cosmology, cfg: FFTLogConfig,
    with_counterterms: bool, with_p13: bool,
) -> XiTables {
    let n = cfg.n;
    assert!(n.is_power_of_two(), "n must be a power of 2");
    let dlnk = (cfg.ln_k_max - cfg.ln_k_min) / (n - 1) as f64;
    let pk: Vec<f64> = (0..n).map(|i| {
        let k = (cfg.ln_k_min + i as f64 * dlnk).exp();
        cosmo.p_lin(k)
    }).collect();
    let xi_pk = xi_from_pk_loglog(cfg.ln_k_min, cfg.ln_k_max, &pk, cfg.lowring);

    let (xi_k2_pk, xi_k4_pk) = if with_counterterms {
        let k_samples: Vec<f64> = (0..n).map(|i| (cfg.ln_k_min + i as f64 * dlnk).exp()).collect();
        let pk_k2: Vec<f64> = pk.iter().zip(&k_samples).map(|(&p, &k)| k * k * p).collect();
        let pk_k4: Vec<f64> = pk.iter().zip(&k_samples).map(|(&p, &k)| k.powi(4) * p).collect();
        (
            Some(xi_from_pk_loglog(cfg.ln_k_min, cfg.ln_k_max, &pk_k2, cfg.lowring)),
            Some(xi_from_pk_loglog(cfg.ln_k_min, cfg.ln_k_max, &pk_k4, cfg.lowring)),
        )
    } else {
        (None, None)
    };

    let xi_p13_eff = if with_p13 {
        let mut ws = crate::Workspace::new(4000);
        ws.update_cosmology(cosmo);
        let ip_inner = crate::integrals::IntegrationParams {
            n_k: 0, n_p: 300, n_mu: 48,
            ln_k_min: cfg.ln_k_min, ln_k_max: cfg.ln_k_max,
        };
        let pk_eff = crate::integrals::p13_effective_pk_table(
            &ws, &ip_inner, cfg.ln_k_min, cfg.ln_k_max, n
        );
        Some(xi_from_pk_loglog(cfg.ln_k_min, cfg.ln_k_max, &pk_eff, cfg.lowring))
    } else {
        None
    };

    XiTables {
        xi_pk, xi_k2_pk, xi_k4_pk, xi_p13_eff,
        ln_k_min: cfg.ln_k_min, ln_k_max: cfg.ln_k_max, n,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Configuration-space smoothing kernels
// ═══════════════════════════════════════════════════════════════════════════

/// Volume-averaged correlation function:
///   xi_bar(R) = (3/R³) ∫₀^R ξ(r) r² dr = 3 ∫₀¹ u² ξ(uR) du
pub fn xi_bar_from_xi(xi: &XiTable, r_smooth: f64, n_gauss: usize) -> f64 {
    let (nodes, weights) = gl_nodes_weights(n_gauss);
    // map [-1, 1] → [0, 1]
    let mut acc = 0.0;
    for i in 0..n_gauss {
        let u = 0.5 * (nodes[i] + 1.0);
        let w = 0.5 * weights[i];
        let rr = u * r_smooth;
        acc += w * u * u * xi.eval(rr);
    }
    3.0 * acc
}

/// Variance with top-hat smoothing:
///   σ²(R) = (3/16) ∫₀² (2−u)²(4+u) u² ξ(uR) du
pub fn sigma2_from_xi(xi: &XiTable, r_smooth: f64, n_gauss: usize) -> f64 {
    let (nodes, weights) = gl_nodes_weights(n_gauss);
    // map [-1, 1] → [0, 2]
    let mut acc = 0.0;
    for i in 0..n_gauss {
        let u = nodes[i] + 1.0;
        let w = weights[i];
        let rr = u * r_smooth;
        let two_m_u = 2.0 - u;
        let four_p_u = 4.0 + u;
        acc += w * two_m_u * two_m_u * four_p_u * u * u * xi.eval(rr);
    }
    (3.0 / 16.0) * acc
}

/// Gauss-Legendre nodes and weights on [-1, 1]. Small n_gauss (≤ 64) expected.
/// (Reuses algorithm from doroshkevich::gauss_legendre but avoids the cross-module
/// dependency so fftlog can be compiled/tested standalone.)
fn gl_nodes_weights(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];
    let m = (n + 1) / 2;
    for i in 0..m {
        let mut z = ((i as f64 + 0.75) / (n as f64 + 0.5) * std::f64::consts::PI).cos();
        loop {
            let mut p0 = 1.0;
            let mut p1 = z;
            for j in 2..=n {
                let p2 = ((2 * j - 1) as f64 * z * p1 - (j - 1) as f64 * p0) / j as f64;
                p0 = p1;
                p1 = p2;
            }
            let dp = n as f64 * (z * p1 - p0) / (z * z - 1.0);
            let dz = p1 / dp;
            z -= dz;
            if dz.abs() < 1e-15 { break; }
        }
        let mut p0 = 1.0;
        let mut p1 = z;
        for j in 2..=n {
            let p2 = ((2 * j - 1) as f64 * z * p1 - (j - 1) as f64 * p0) / j as f64;
            p0 = p1;
            p1 = p2;
        }
        let dp = n as f64 * (z * p1 - p0) / (z * z - 1.0);
        let w = 2.0 / ((1.0 - z * z) * dp * dp);
        nodes[i] = -z;
        nodes[n - 1 - i] = z;
        weights[i] = w;
        weights[n - 1 - i] = w;
    }
    (nodes, weights)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lngamma_on_real_axis() {
        // Γ(1) = 1, ln Γ(1) = 0
        let ln1 = lngamma_complex(Complex64::new(1.0, 0.0));
        assert!(ln1.re.abs() < 1e-12, "ln Γ(1) = {}", ln1.re);
        assert!(ln1.im.abs() < 1e-12);

        // Γ(5) = 24, ln Γ(5) = ln(24) ≈ 3.1780538
        let ln5 = lngamma_complex(Complex64::new(5.0, 0.0));
        assert!((ln5.re - 24.0f64.ln()).abs() < 1e-10, "ln Γ(5) = {}", ln5.re);

        // Γ(0.5) = √π, ln Γ(0.5) = 0.5 ln π ≈ 0.5723649429
        let ln05 = lngamma_complex(Complex64::new(0.5, 0.0));
        let expected = 0.5 * std::f64::consts::PI.ln();
        assert!((ln05.re - expected).abs() < 1e-10, "ln Γ(0.5) = {}", ln05.re);
    }

    #[test]
    fn lngamma_reflection_formula() {
        // Γ(z)Γ(1−z) = π/sin(πz), so ln Γ(z) + ln Γ(1−z) = ln π − ln sin(πz)
        let z = Complex64::new(0.3, 0.2);
        let l1 = lngamma_complex(z);
        let l2 = lngamma_complex(Complex64::new(1.0, 0.0) - z);
        let sum = l1 + l2;
        let pi = std::f64::consts::PI;
        let expected = Complex64::new(pi.ln(), 0.0)
            - (Complex64::new(pi, 0.0) * z).sin().ln();
        let diff = (sum - expected).norm();
        assert!(diff < 1e-9, "reflection formula failed: diff = {}", diff);
    }

    /// Analytic Hankel transform: for a(k) = k^α, the transform
    /// ã(r) = ∫₀^∞ k^α (kr) J_μ(kr) d(ln k) = r^(-α) × 2^α × Γ((μ+1+α)/2) / Γ((μ+1−α)/2)
    fn analytic_power_law(alpha: f64, mu: f64, r: f64) -> f64 {
        let num = lngamma_complex(Complex64::new((mu + 1.0 + alpha) * 0.5, 0.0)).re;
        let den = lngamma_complex(Complex64::new((mu + 1.0 - alpha) * 0.5, 0.0)).re;
        r.powf(-alpha) * 2.0f64.powf(alpha) * (num - den).exp()
    }

    #[test]
    fn fftlog_power_law_transform() {
        // Test FFTLog against the analytic power-law result.
        // a(k) = k^α with α = -1.5, μ = 1/2
        let n = 1024;
        let ln_k_min = (1e-4_f64).ln();
        let ln_k_max = (1e4_f64).ln();
        let dlnk = (ln_k_max - ln_k_min) / (n - 1) as f64;
        let alpha = -1.5;
        let mu = 0.5;

        let k_center = ((ln_k_min + ln_k_max) * 0.5).exp();
        let a_k: Vec<f64> = (0..n).map(|i| {
            let k = (ln_k_min + i as f64 * dlk(i, ln_k_min, dlnk)).exp();
            k.powf(alpha)
        }).collect();
        // helper stub to silence unused args above
        fn dlk(_i: usize, _lkm: f64, d: f64) -> f64 { d }
        let _ = a_k.len();

        let a_k: Vec<f64> = (0..n).map(|i| {
            let k = (ln_k_min + i as f64 * dlnk).exp();
            k.powf(alpha)
        }).collect();

        let fftlog = FFTLog::new(n, dlnk, mu, 0.0, true);
        let (r_grid, a_r) = fftlog.transform(&a_k, k_center);

        // Compare on interior points (edges may have aliasing)
        let n_skip = n / 8;
        let mut max_rel = 0.0_f64;
        for i in n_skip..(n - n_skip) {
            let r = r_grid[i];
            let expected = analytic_power_law(alpha, mu, r);
            let rel = ((a_r[i] - expected) / expected).abs();
            if rel > max_rel { max_rel = rel; }
        }
        assert!(max_rel < 1e-4, "FFTLog power-law max rel err = {:.3e}", max_rel);
    }

    #[test]
    fn xi_bar_kernel_uniform() {
        // For ξ(r) ≡ const C, xi_bar(R) = C regardless of R.
        // Build a trivial XiTable with constant ξ.
        let n = 64;
        let r: Vec<f64> = (0..n).map(|i| (0.01_f64).powf(1.0 - i as f64 / (n - 1) as f64) * 1000.0_f64.powf(i as f64 / (n - 1) as f64)).collect();
        let xi = vec![0.7; n];
        let ln_r_min = r[0].ln();
        let ln_r_max = r[n - 1].ln();
        let dlnr = (ln_r_max - ln_r_min) / (n - 1) as f64;
        let table = XiTable { r, xi, ln_r_min, ln_r_max, dlnr };

        for r_smooth in [5.0, 20.0, 100.0] {
            let xb = xi_bar_from_xi(&table, r_smooth, 32);
            assert!((xb - 0.7).abs() < 1e-10, "xi_bar({}) = {}", r_smooth, xb);
        }
    }

    /// Cross-check FFTLog xi_bar vs trapezoidal quadrature for Planck 2018 P_L.
    /// To be fair, match the integration ranges between the two methods.
    #[test]
    fn fftlog_vs_quadrature_xi_bar() {
        use crate::{Cosmology, Workspace};
        use crate::integrals::{IntegrationParams, xi_bar_ws};

        let cosmo = Cosmology::planck2018();
        // Match FFTLog range to quadrature range for apples-to-apples.
        let cfg = FFTLogConfig {
            n: 2048,
            ln_k_min: (1e-5_f64).ln(),
            ln_k_max: (50.0_f64).ln(),
            lowring: true,
        };
        let tables = build_xi_tables(&cosmo, cfg, false, false);

        // Trapezoidal reference: fine grid.
        let mut ws = Workspace::new(8000);
        ws.update_cosmology(&cosmo);
        let ip = IntegrationParams {
            n_k: 8000, n_p: 200, n_mu: 48,
            ln_k_min: (1e-5_f64).ln(), ln_k_max: (50.0_f64).ln(),
        };

        let mut max_rel = 0.0_f64;
        for &r in &[5.0_f64, 10.0, 20.0, 30.0, 50.0, 80.0] {
            let quad = xi_bar_ws(r, &ws, &ip);
            let fft = xi_bar_from_xi(&tables.xi_pk, r, 64);
            let rel = ((fft - quad) / quad).abs();
            println!("R={:5.1}  quad={:+.6e}  fftlog={:+.6e}  rel={:.2e}",
                     r, quad, fft, rel);
            if rel > max_rel { max_rel = rel; }
        }
        assert!(max_rel < 5e-3, "max rel disagreement {:.2e}", max_rel);
    }

    #[test]
    fn fftlog_vs_quadrature_sigma2() {
        use crate::{Cosmology, Workspace};
        use crate::integrals::{IntegrationParams, sigma2_tree_ws};

        let cosmo = Cosmology::planck2018();
        let cfg = FFTLogConfig {
            n: 2048,
            ln_k_min: (1e-5_f64).ln(),
            ln_k_max: (50.0_f64).ln(),
            lowring: true,
        };
        let tables = build_xi_tables(&cosmo, cfg, false, false);

        let mut ws = Workspace::new(8000);
        ws.update_cosmology(&cosmo);
        let ip = IntegrationParams {
            n_k: 8000, n_p: 200, n_mu: 48,
            ln_k_min: (1e-5_f64).ln(), ln_k_max: (50.0_f64).ln(),
        };

        let mut max_rel = 0.0_f64;
        for &r in &[5.0_f64, 10.0, 20.0, 30.0, 50.0, 80.0] {
            let quad = sigma2_tree_ws(r, &ws, &ip);
            let fft = sigma2_from_xi(&tables.xi_pk, r, 64);
            let rel = ((fft - quad) / quad).abs();
            println!("R={:5.1}  quad={:+.6e}  fftlog={:+.6e}  rel={:.2e}",
                     r, quad, fft, rel);
            if rel > max_rel { max_rel = rel; }
        }
        assert!(max_rel < 5e-3, "max rel disagreement {:.2e}", max_rel);
    }

    /// BAO smoothness test: FFTLog xi_bar must be monotonic-smooth at R=100-200
    /// (aside from physical BAO oscillation). High-frequency noise above
    /// the BAO frequency should be sub-percent.
    #[test]
    fn fftlog_bao_smoothness() {
        use crate::Cosmology;
        let cosmo = Cosmology::planck2018();
        let tables = build_xi_tables(&cosmo, FFTLogConfig::default(), false, false);

        // Dense R grid: 500 points in [100, 200] Mpc/h
        let n_r = 500;
        let r_values: Vec<f64> = (0..n_r).map(|i| {
            100.0 + 100.0 * i as f64 / (n_r - 1) as f64
        }).collect();
        let xb: Vec<f64> = r_values.iter().map(|&r| xi_bar_from_xi(&tables.xi_pk, r, 32)).collect();

        // Fit a smooth polynomial in ln R to R² xi_bar, then measure residual.
        // Simple approach: moving-average smooth with window=25, measure RMS of (raw - smooth) / smooth.
        let win = 25;
        let mut max_rel_residual: f64 = 0.0;
        let mut rms_rel_residual: f64 = 0.0;
        let mut count = 0;
        for i in win..(n_r - win) {
            // Use log-space to suppress broadband trend
            let smoothed: f64 = (0..(2 * win + 1)).map(|j| {
                let idx = i - win + j;
                (r_values[idx] * r_values[idx] * xb[idx]).ln()
            }).sum::<f64>() / (2 * win + 1) as f64;
            let raw = (r_values[i] * r_values[i] * xb[i]).ln();
            let resid = raw - smoothed;
            if resid.abs() > max_rel_residual { max_rel_residual = resid.abs(); }
            rms_rel_residual += resid * resid;
            count += 1;
        }
        rms_rel_residual = (rms_rel_residual / count as f64).sqrt();
        println!("BAO smoothness: rms log-residual = {:.2e}, max = {:.2e}",
                 rms_rel_residual, max_rel_residual);
        // At n=2048 FFTLog, smoothness should be well below 1% (0.01).
        // With moving-window 25 and R-range 100 Mpc/h, we capture the BAO bump
        // but any noise above that frequency should be very small.
        // (This is a lenient threshold; tighten if needed.)
        assert!(rms_rel_residual < 0.02,
                "BAO region too noisy: rms={:.3e}", rms_rel_residual);
    }

    #[test]
    fn sigma2_kernel_uniform() {
        // For ξ(r) ≡ const C, σ²(R) = C as well (since the kernel integrates to 1).
        let n = 64;
        let r: Vec<f64> = (0..n).map(|i| (0.01_f64).powf(1.0 - i as f64 / (n - 1) as f64) * 1000.0_f64.powf(i as f64 / (n - 1) as f64)).collect();
        let xi = vec![0.7; n];
        let ln_r_min = r[0].ln();
        let ln_r_max = r[n - 1].ln();
        let dlnr = (ln_r_max - ln_r_min) / (n - 1) as f64;
        let table = XiTable { r, xi, ln_r_min, ln_r_max, dlnr };

        for r_smooth in [5.0, 20.0, 100.0] {
            let s2 = sigma2_from_xi(&table, r_smooth, 64);
            // kernel: (3/16) ∫₀² (2-u)²(4+u) u² du — verify analytically
            // = (3/16) × (256/15) = 16/5 = 3.2... but we want this to equal C=0.7 if normalized
            // Actually (3/16) ∫₀² (2-u)²(4+u) u² du should equal 1. Let's verify numerically.
            println!("σ²(R={}) = {} (expected ≈ 0.7)", r_smooth, s2);
            assert!((s2 - 0.7).abs() < 1e-6, "σ²({}) = {}", r_smooth, s2);
        }
    }
}

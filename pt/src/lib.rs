//! sigma2j: Perturbative σ²_J(M) for Lagrangian volume statistics.
//!
//! Two-tier architecture:
//!   Tier 1: Doroshkevich/Zel'dovich baseline  Var(J) = S(1+2S/15)²
//!   Tier 2: LPT loop corrections from P_L(k)
//!
//! # Fast path (MCMC):
//! ```ignore
//! let mut ws = Workspace::new(2000);
//! let masses = vec![1e14, 3e14, 1e15, 3e15, 1e16];
//! let mut out = vec![0.0; masses.len()];
//! // Per MCMC step:
//! let cosmo = Cosmology::new(om, ob, h, ns, s8);
//! ws.update_cosmology(&cosmo);
//! sigma2_j_at_masses(&cosmo, &masses, 2, 20.0, 0.0, &ws, &mut out);
//! ```
//!
//! # Rich path (plotting):
//! ```ignore
//! let results = sigma2_j_plot(&cosmo, 2, 20.0, 0.0);
//! for r in &results { println!("{} {} {}", r.r, r.sigma2_lin, r.sigma2_j); }
//! ```

pub mod cosmology;
pub mod doroshkevich;
pub mod fftlog;
pub mod integrals;
pub mod knn_cdf;

pub use cosmology::Cosmology;
pub use integrals::IntegrationParams;

const RHO_CRIT_H2: f64 = 2.775e11;

// ── Workspace ────────────────────────────────────────────────────────────

/// Pre-allocated P_L(k) table. Create once, call update_cosmology per step.
pub struct Workspace {
    pk_table: Vec<f64>,
    ln_pk_table: Vec<f64>,
    ln_k: Vec<f64>,
    n: usize,
    ln_k_min: f64,
    ln_k_max: f64,
}

impl Workspace {
    pub fn new(n: usize) -> Self {
        let ln_k_min = (1e-5_f64).ln();
        let ln_k_max = (100.0_f64).ln();
        let mut ln_k = vec![0.0; n];
        for i in 0..n {
            ln_k[i] = ln_k_min + (ln_k_max - ln_k_min) * i as f64 / (n - 1) as f64;
        }
        Workspace { pk_table: vec![0.0; n], ln_pk_table: vec![0.0; n], ln_k, n, ln_k_min, ln_k_max }
    }

    pub fn update_cosmology(&mut self, cosmo: &Cosmology) {
        for i in 0..self.n {
            let k = self.ln_k[i].exp();
            let p = cosmo.p_lin(k);
            self.pk_table[i] = p;
            self.ln_pk_table[i] = if p > 0.0 { p.ln() } else { -100.0 };
        }
    }

    #[inline(always)]
    pub fn p_lin(&self, k: f64) -> f64 {
        if k <= 0.0 { return 0.0; }
        let lk = k.ln();
        if lk < self.ln_k_min || lk > self.ln_k_max { return 0.0; }
        let f = (lk - self.ln_k_min) / (self.ln_k_max - self.ln_k_min) * (self.n - 1) as f64;
        let i = f as usize;
        if i >= self.n - 1 { return self.pk_table[self.n - 1]; }
        let t = f - i as f64;
        (self.ln_pk_table[i] * (1.0 - t) + self.ln_pk_table[i + 1] * t).exp()
    }
}

// ── Geometry ─────────────────────────────────────────────────────────────

/// M = (4π/3) ρ̄ R³ where ρ̄ = Ωm × ρ_crit,0 h². Exact in Lagrangian coordinates.
#[inline(always)]
pub fn mass_to_radius(m: f64, omega_m: f64) -> f64 {
    (m / (4.0 / 3.0 * std::f64::consts::PI * RHO_CRIT_H2 * omega_m)).cbrt()
}

#[inline(always)]
pub fn radius_to_mass(r: f64, omega_m: f64) -> f64 {
    4.0 / 3.0 * std::f64::consts::PI * RHO_CRIT_H2 * omega_m * r * r * r
}

/// Effective wavenumber k_eff = π/R [h/Mpc].
///
/// This is the scale at which the top-hat window W(kR) transitions from
/// ~1 to ~0. It is NOT a Fourier-space cutoff — just a convenient label
/// for the EFT audience who think in k-space. The actual information
/// content comes from the integral over all k weighted by W²(kR).
#[inline(always)]
pub fn radius_to_k_eff(r: f64) -> f64 {
    std::f64::consts::PI / r
}

/// Inverse: R = π/k_eff.
#[inline(always)]
pub fn k_eff_to_radius(k: f64) -> f64 {
    std::f64::consts::PI / k
}

/// For D-kNN measurements: k neighbours at mean density n̄ enclose
/// mass M_k = k × m_particle. This gives R_k = (3k/(4π n̄))^{1/3}.
#[inline(always)]
pub fn knn_to_radius(k_neighbours: usize, n_bar: f64) -> f64 {
    (3.0 * k_neighbours as f64 / (4.0 * std::f64::consts::PI * n_bar)).cbrt()
}

/// Generate a default fine grid of masses for plotting (log-spaced).
/// Range: 10^{12.5} to 10^{17.5} Msun/h (50 points).
pub fn default_masses() -> Vec<f64> {
    (0..50).map(|i| {
        let log_m = 12.5 + 5.0 * i as f64 / 49.0;
        10.0_f64.powf(log_m)
    }).collect()
}

// ── Fast path ────────────────────────────────────────────────────────────

pub fn sigma2_j_at_masses(
    cosmo: &Cosmology, masses: &[f64], n_lpt: usize,
    c_j2: f64, c_j4: f64, ws: &Workspace, out: &mut [f64],
) {
    use rayon::prelude::*;
    let ip = IntegrationParams::fast();
    let om = cosmo.omega_m;
    out.par_iter_mut().zip(masses.par_iter()).for_each(|(o, &m)| {
        *o = eval_single(mass_to_radius(m, om), n_lpt, c_j2, c_j4, None, ws, &ip);
    });
}

/// Redshift-space version: all quantities Kaiser-enhanced.
pub fn sigma2_j_at_masses_rsd(
    cosmo: &Cosmology, masses: &[f64], n_lpt: usize,
    c_j2: f64, c_j4: f64, rsd: &integrals::RsdParams,
    ws: &Workspace, out: &mut [f64],
) {
    use rayon::prelude::*;
    let ip = IntegrationParams::fast();
    let om = cosmo.omega_m;
    out.par_iter_mut().zip(masses.par_iter()).for_each(|(o, &m)| {
        *o = eval_single(mass_to_radius(m, om), n_lpt, c_j2, c_j4, Some(rsd), ws, &ip);
    });
}

pub fn sigma2_j_at_radii(
    _cosmo: &Cosmology, radii: &[f64], n_lpt: usize,
    c_j2: f64, c_j4: f64, ws: &Workspace, out: &mut [f64],
) {
    use rayon::prelude::*;
    let ip = IntegrationParams::fast();
    out.par_iter_mut().zip(radii.par_iter()).for_each(|(o, &r)| {
        *o = eval_single(r, n_lpt, c_j2, c_j4, None, ws, &ip);
    });
}

/// Core single-radius evaluation using pre-built P_L table.
///
/// Three-tier model:
///   Tier 0: sigma^2_lin = S (tree)
///   Tier 1: Zel'dovich baseline = S * (1 + 2S/15)^2
///   Tier 2+: LPT corrections = Zel * [1 + D1(S) + D2(S) + D3(S)]
///   Counterterms: c_J^2 * sigma^2_{J,2}
///
/// When `rsd` is provided, all quantities are Kaiser-enhanced and
/// the Zel'dovich baseline uses the redshift-space S.
///
/// Calibrated from DISCO-DJ 5LPT (128^3, L=1000 Mpc/h, Planck 2018).
#[inline]
fn eval_single(
    r: f64, n_lpt: usize, c_j2: f64, c_j4: f64,
    rsd: Option<&integrals::RsdParams>,
    ws: &Workspace, ip: &IntegrationParams,
) -> f64 {
    let s_real = integrals::sigma2_tree_ws(r, ws, ip);

    // Apply Kaiser enhancement for RSD
    let s = match rsd {
        Some(p) => integrals::kaiser_sigma2(p.f) * s_real,
        None => s_real,
    };

    // Tier 1: Doroshkevich baseline (works in both real and redshift space)
    let fac = 1.0 + 2.0 * s / 15.0;
    let zel = s * fac * fac;

    // Tier 2+: LPT corrections as polynomial in S
    const D1_A0: f64 = -0.040058;
    const D1_A1: f64 = -0.822312;
    const D1_A2: f64 =  0.708537;
    const D2_A0: f64 =  0.022979;
    const D2_A1: f64 =  0.411372;
    const D2_A2: f64 = -0.303855;
    const CONV_RATIO: f64 = -0.535;

    let mut result = zel;

    if n_lpt >= 1 {
        let d1 = D1_A0 + D1_A1 * s + D1_A2 * s * s;
        result += zel * d1;

        if n_lpt >= 2 {
            let d2 = D2_A0 + D2_A1 * s + D2_A2 * s * s;
            result += zel * d2;

            if n_lpt >= 3 {
                let d3 = CONV_RATIO * d2;
                result += zel * d3;
            }
        }
    }

    // Counterterms
    if c_j2 != 0.0 {
        let ct_base = integrals::sigma2_jn_ws(r, 2, ws, ip);
        let ct_factor = match rsd {
            Some(p) => integrals::kaiser_sigma2(p.f),  // leading-order RSD for counterterm
            None => 1.0,
        };
        result += c_j2 * ct_factor * ct_base;
    }
    if c_j4 != 0.0 {
        let ct_base = integrals::sigma2_jn_ws(r, 4, ws, ip);
        let ct_factor = match rsd {
            Some(p) => integrals::kaiser_sigma2(p.f),
            None => 1.0,
        };
        result += c_j4 * ct_factor * ct_base;
    }

    result
}

// ── Rich path ────────────────────────────────────────────────────────────

/// Detailed result with all contributions exposed for plotting.
#[derive(Clone, Debug)]
pub struct Sigma2JDetailed {
    /// Lagrangian smoothing radius R [Mpc/h].
    pub r: f64,
    /// Enclosed mass M = (4π/3) ρ̄ R³ [Msun/h]. Exact.
    pub mass: f64,
    /// Effective wavenumber k_eff = π/R [h/Mpc]. Convenience label for EFT plots.
    pub k_eff: f64,
    /// Tree-level = σ²_lin(R).
    pub sigma2_lin: f64,
    /// Doroshkevich/Zel'dovich baseline = S(1+2S/15)².
    pub sigma2_zel: f64,
    /// Final σ²_J with all corrections.
    pub sigma2_j: f64,

    // ── D-kNN first cumulant ──
    /// Volume-averaged correlation function ξ̄(R) = ∫ k³/(2π²) W(kR) P_L dk.
    pub xi_bar: f64,
    /// P₁₃ one-loop correction to ξ̄, smoothed with single W(kR).
    pub xibar_p13: f64,

    // ── Polynomial LPT corrections (fast, calibrated) ──
    pub d1: f64,
    pub d2: f64,
    pub d3: f64,

    // ── Loop integral decomposition (diagnostic) ──
    pub p22: f64,
    pub two_p13: f64,
    pub counterterm: f64,

    // ── Third cumulant (bispectrum) building blocks ──
    /// Tree-level matter skewness S₃(R) = 6 ∫∫ F2 PL PL W₁W₂W₁₂.
    pub s3_matter: f64,
    /// Tree-level Jacobian skewness S₃^(J)(R) using F2^(J) kernel.
    pub s3_jacobian: f64,
    /// All bias-operator bispectrum integrals.
    pub bispec: Option<integrals::BispecIntegrals>,

    pub truncation_error: f64,
    pub elapsed_ns: u64,
}

/// Compute σ²_J with full diagnostic output including ξ̄ and S₃.
///
/// Convenience wrapper that enables all diagnostics. Use `sigma2_j_full` to
/// opt out of the 3D diagnostic integrals for faster batch evaluation.
pub fn sigma2_j_detailed(
    cosmo: &Cosmology, r: f64, n_lpt: usize, c_j2: f64, c_j4: f64,
) -> Sigma2JDetailed {
    sigma2_j_full(cosmo, r, n_lpt, c_j2, c_j4, true, false)
}

/// Full computation with optional diagnostic and bispectrum integrals.
///
/// `compute_diagnostics`: when false, the 3D loop diagnostics (`p22`, `two_p13`)
/// and the tree-level S₃ bispectrum integrals (`s3_matter`, `s3_jacobian`) are
/// skipped; their fields are zero in the returned struct. Skipping saves
/// ~15 ms/R on typical hardware. Use `true` only when you actually read these
/// fields.
pub fn sigma2_j_full(
    cosmo: &Cosmology, r: f64, n_lpt: usize, c_j2: f64, c_j4: f64,
    compute_diagnostics: bool, compute_bispec: bool,
) -> Sigma2JDetailed {
    let ip = IntegrationParams::default();
    let mut ws = Workspace::new(3000);
    ws.update_cosmology(cosmo);
    let xi_tables = fftlog::build_xi_tables(cosmo, fftlog::FFTLogConfig::default(),
                                             false, false, false);
    sigma2_j_with_workspace(&ws, &xi_tables, &ip, cosmo.omega_m,
                            r, n_lpt, c_j2, c_j4, compute_diagnostics, compute_bispec)
}

/// Shared-workspace variant: assumes ws + xi_tables have been built for the cosmology.
///
/// Uses FFTLog-backed 1D integrals for σ²_lin(R) and ξ̄(R). The counterterms
/// σ²_{J,2}(R), σ²_{J,4}(R) use the workspace's trapezoidal quadrature. The 3D
/// diagnostic integrals (P₂₂, P₁₃, S₃) are only computed when
/// `compute_diagnostics = true` — they contribute nothing to σ²_J itself (the
/// calibrated polynomial d₁..d₃ already captures the one-loop corrections) and
/// are optional display-only fields.
pub fn sigma2_j_with_workspace(
    ws: &Workspace, xi_tables: &fftlog::XiTables, ip: &IntegrationParams,
    omega_m: f64, r: f64, n_lpt: usize, c_j2: f64, c_j4: f64,
    compute_diagnostics: bool, compute_bispec: bool,
) -> Sigma2JDetailed {
    let t0 = std::time::Instant::now();

    // ── Second cumulants (FFTLog) ──
    let s = fftlog::sigma2_from_xi(&xi_tables.xi_pk, r, 64);
    let fac = 1.0 + 2.0 * s / 15.0;
    let zel = s * fac * fac;

    // ξ̄(R) — single W integral (FFTLog, cheap)
    let xi_bar = fftlog::xi_bar_from_xi(&xi_tables.xi_pk, r, 64);

    // LPT corrections — polynomial d₁..d₃ calibrated against DISCO-DJ 5LPT.
    // These give σ²_J directly; no 3D loop is involved in the returned σ²_J.
    const D1_A0: f64 = -0.040058;
    const D1_A1: f64 = -0.822312;
    const D1_A2: f64 =  0.708537;
    const D2_A0: f64 =  0.022979;
    const D2_A1: f64 =  0.411372;
    const D2_A2: f64 = -0.303855;
    const CONV_RATIO: f64 = -0.535;

    let d1_val = if n_lpt >= 1 { D1_A0 + D1_A1 * s + D1_A2 * s * s } else { 0.0 };
    let d2_val = if n_lpt >= 2 { D2_A0 + D2_A1 * s + D2_A2 * s * s } else { 0.0 };
    let d3_val = if n_lpt >= 3 { CONV_RATIO * d2_val } else { 0.0 };

    let mut v = zel * (1.0 + d1_val + d2_val + d3_val);

    // Counterterms — always via trapezoidal (1D, cheap; usually 0).
    let ct = c_j2 * integrals::sigma2_jn_ws(r, 2, ws, ip)
           + c_j4 * integrals::sigma2_jn_ws(r, 4, ws, ip);
    v += ct;

    // ── Diagnostic loop integrals (expensive 3D, skippable) ──
    let (p22, tp13) = if compute_diagnostics {
        (
            0.1803 * integrals::sigma2_p22_raw_ws(r, ws, ip),
            -1.070 * integrals::sigma2_p13_raw_ws(r, ws, ip),
        )
    } else {
        (0.0, 0.0)
    };

    // P₁₃ correction for ξ̄ — FFTLog (cheap) if xi_p13_eff populated; else 0
    // when diagnostics are off (the trapezoidal path is expensive).
    let xibar_p13 = if let Some(ref xi_p13) = xi_tables.xi_p13_eff {
        -1.070 * fftlog::xi_bar_from_xi(xi_p13, r, 64)
    } else if compute_diagnostics {
        -1.070 * integrals::xibar_p13_raw_ws(r, ws, ip)
    } else {
        0.0
    };

    // ── Third cumulants (expensive 3D, skippable) ──
    let (s3_m, s3_j) = if compute_diagnostics {
        (
            integrals::s3_tree_matter(r, ws, ip),
            integrals::s3_tree_jacobian(r, ws, ip),
        )
    } else {
        (0.0, 0.0)
    };

    // Bias integrals (extra-expensive, independent flag)
    let bispec = if compute_bispec {
        Some(integrals::s3_bias_integrals(r, ws, ip))
    } else {
        None
    };

    let last_d = if n_lpt >= 3 { d3_val }
                 else if n_lpt >= 2 { d2_val }
                 else { d1_val };
    let trunc = (CONV_RATIO * last_d * zel).abs();

    Sigma2JDetailed {
        r,
        mass: radius_to_mass(r, omega_m),
        k_eff: radius_to_k_eff(r),
        sigma2_lin: s,
        sigma2_zel: zel,
        sigma2_j: v,
        xi_bar,
        xibar_p13,
        d1: d1_val, d2: d2_val, d3: d3_val,
        p22, two_p13: tp13, counterterm: ct,
        s3_matter: s3_m,
        s3_jacobian: s3_j,
        bispec,
        truncation_error: trunc,
        elapsed_ns: t0.elapsed().as_nanos() as u64,
    }
}

pub fn default_radii() -> Vec<f64> {
    (0..50).map(|i| {
        let f = i as f64 / 49.0;
        (3.0_f64.ln() * (1.0 - f) + 200.0_f64.ln() * f).exp()
    }).collect()
}

/// Plot on a fine radius grid (R axis).
///
/// `compute_diagnostics`: when false, skips the ~15 ms/R 3D diagnostic integrals
/// (P₂₂, P₁₃, S₃_matter, S₃_jacobian) and zeros their fields. σ²_J itself is
/// unaffected. Use `true` only if you consume those diagnostic fields.
pub fn sigma2_j_plot(
    cosmo: &Cosmology, n_lpt: usize, c_j2: f64, c_j4: f64,
    compute_diagnostics: bool,
) -> Vec<Sigma2JDetailed> {
    use rayon::prelude::*;
    let ip = IntegrationParams::default();
    let mut ws = Workspace::new(3000);
    ws.update_cosmology(cosmo);
    let xi_tables = fftlog::build_xi_tables(cosmo, fftlog::FFTLogConfig::default(),
                                             false, false, false);
    default_radii().par_iter()
        .map(|&r| sigma2_j_with_workspace(&ws, &xi_tables, &ip, cosmo.omega_m,
                                          r, n_lpt, c_j2, c_j4,
                                          compute_diagnostics, false))
        .collect()
}

/// Plot on a fine mass grid (M axis). The natural coordinate for this framework.
/// See `sigma2_j_plot` for the `compute_diagnostics` flag semantics.
pub fn sigma2_j_plot_masses(
    cosmo: &Cosmology, n_lpt: usize, c_j2: f64, c_j4: f64,
    compute_diagnostics: bool,
) -> Vec<Sigma2JDetailed> {
    use rayon::prelude::*;
    let ip = IntegrationParams::default();
    let mut ws = Workspace::new(3000);
    ws.update_cosmology(cosmo);
    let om = cosmo.omega_m;
    let xi_tables = fftlog::build_xi_tables(cosmo, fftlog::FFTLogConfig::default(),
                                             false, false, false);
    default_masses().par_iter()
        .map(|&m| sigma2_j_with_workspace(&ws, &xi_tables, &ip, om,
                                          mass_to_radius(m, om), n_lpt, c_j2, c_j4,
                                          compute_diagnostics, false))
        .collect()
}

/// Plot on a user-specified mass grid.
/// See `sigma2_j_plot` for the `compute_diagnostics` flag semantics.
pub fn sigma2_j_plot_at_masses(
    cosmo: &Cosmology, masses: &[f64], n_lpt: usize, c_j2: f64, c_j4: f64,
    compute_diagnostics: bool,
) -> Vec<Sigma2JDetailed> {
    use rayon::prelude::*;
    let ip = IntegrationParams::default();
    let mut ws = Workspace::new(3000);
    ws.update_cosmology(cosmo);
    let om = cosmo.omega_m;
    let xi_tables = fftlog::build_xi_tables(cosmo, fftlog::FFTLogConfig::default(),
                                             false, false, false);
    masses.par_iter()
        .map(|&m| sigma2_j_with_workspace(&ws, &xi_tables, &ip, om,
                                          mass_to_radius(m, om), n_lpt, c_j2, c_j4,
                                          compute_diagnostics, false))
        .collect()
}

// ── Full ξ̄_J prediction ─────────────────────────────────────────────────

/// Result of the full ξ̄_J prediction with all three layers decomposed.
///
/// Returned by both real-space (`xibar_j_full`) and redshift-space (`xibar_j_full_rsd`)
/// entry points. In redshift space:
/// - `sigma2_lin` contains the Kaiser-enhanced σ²_s = K₂(f) × σ²_L(R).
/// - `xibar_tree` contains -b₁ × K₁(f) × ξ̄_L(R).
/// - `xibar_zel` uses the biased Doroshkevich at σ_s (Kaiser-enhanced).
/// - `xibar_1loop` has the leading-order Kaiser factor K₁(f) applied.
/// - `epsilon` = (3/7)² × σ²_s (Kaiser-enhanced).
#[derive(Clone, Debug)]
pub struct XibarJDetailed {
    /// Smoothing radius R [Mpc/h].
    pub r: f64,
    /// Linear bias b₁ used (tracer-matter cross-correlation coefficient).
    pub b1: f64,
    /// Quadratic bias b₂ used (Lagrangian bias expansion).
    pub b2: f64,
    /// Tidal bias b_{s²} used (Lagrangian bias expansion).
    pub bs2: f64,
    /// Growth rate f used (0 in real-space runs).
    pub f_growth: f64,
    /// Tree-level: −K₁ b₁ σ²_L + b₂ M₁₂ + b_{s²} M_{s²}(R).
    pub xibar_tree: f64,
    /// Exact Zel'dovich baseline: ⟨J-1⟩ from biased Doroshkevich at σ_s.
    /// (Currently uses b₁ exponential tilt; b₂/b_{s²} tilts deferred to
    /// polynomial-weight PDF path — see `doroshkevich_biased_polynomial_pass`.)
    pub xibar_zel: f64,
    /// One-loop P₁₃ correction (leading-order K₁ factor in RSD).
    pub xibar_1loop: f64,
    /// Full prediction: Zel'dovich + geometric series of loop corrections.
    pub xibar_full: f64,
    /// σ²_s(R) at this scale (Kaiser-enhanced in RSD, else σ²_L).
    pub sigma2_lin: f64,
    /// Expansion parameter ε = (3/7)² σ²_s for convergence diagnostic.
    pub epsilon: f64,
    /// Unbiased cross-moment M₁₂(R) = ⟨(J−1) I₁²⟩_Zel. Enters ξ̄ at O(b₂).
    pub m12: f64,
    /// Unbiased cross-moment M_{s²}(R). Enters ξ̄ at O(b_{s²}).
    pub m_s2: f64,
    /// One-loop b₂ correction: b₂ × Ξ_{b₂}(R). Zero when b₂ = 0.
    pub xibar_1loop_b2: f64,
    /// One-loop b_{s²} correction: b_{s²} × Ξ_{bs²}(R). Zero when b_{s²} = 0.
    pub xibar_1loop_bs2: f64,
}

/// Compute ξ̄_J at a single radius with full three-layer prediction.
///
/// Layer 1: Exact Zel'dovich baseline from biased Doroshkevich quadrature.
/// Layer 2: One-loop P₁₃ correction (cross-spectrum with single W).
/// Layer 3: Geometric series in ε = (3/7)² σ²_s.
///
/// # Parameters
/// * `b1` — linear bias of the tracer population
/// * `n_corrections` — number of geometric series terms (0 = Zel'dovich only,
///   1 = +one-loop, 2+ = higher-order resummation)
/// Standard integration parameters used by xibar_j_full:
/// - hi-res 1D grid for single-W smoothed integrals (ξ̄, σ²_lin)
/// - moderate 3D grid for the P₁₃ loop (sub-percent contribution)
fn xibar_j_integration_params() -> (IntegrationParams, IntegrationParams) {
    let ip_hires = IntegrationParams {
        n_k: 4000, n_p: 200, n_mu: 48,
        ln_k_min: (1e-5_f64).ln(), ln_k_max: (50.0_f64).ln(),
    };
    let ip_loop = IntegrationParams {
        n_k: 400, n_p: 200, n_mu: 48,
        ln_k_min: (1e-5_f64).ln(), ln_k_max: (50.0_f64).ln(),
    };
    (ip_hires, ip_loop)
}

/// Core single-radius ξ̄_J computation using fully FFTLog-based smoothed integrals.
/// All R-dependent integrals go through the xi_tables (one-time precomputed).
///
/// The tree-level biased prediction uses the full Lagrangian bias expansion:
///   ξ̄_tree = −b₁ K₁ ξ̄_L + b₂ M₁₂ + b_{s²} M_{s²}
/// where M₁₂ and M_{s²} are Zel'dovich cross-moments computed at σ_s.
///
/// When `rsd` is `Some(f)`, apply Kaiser enhancement:
///   σ²_s = K₂ × σ²_L,   K₁ = 1 + f/3,   ε = (3/7)² σ²_s.
fn xibar_j_with_workspace(
    ws: &Workspace, xi_tables: &fftlog::XiTables,
    ip_hires: &IntegrationParams, ip_loop: &IntegrationParams,
    r: f64, bias: doroshkevich::BiasParams, n_corrections: usize,
    rsd: Option<&integrals::RsdParams>,
) -> XibarJDetailed {
    // Real-space ingredients (always from FFTLog tree-level tables)
    let sigma2_lin_real = fftlog::sigma2_from_xi(&xi_tables.xi_pk, r, 64);
    let xi_bar_real = fftlog::xi_bar_from_xi(&xi_tables.xi_pk, r, 64);

    // Kaiser factors (1.0 in real space)
    let f_growth = rsd.map(|p| p.f).unwrap_or(0.0);
    let k2 = integrals::kaiser_sigma2(f_growth);
    let k1 = integrals::kaiser_xibar(f_growth);

    // σ²_s (Kaiser-enhanced) — drives Doroshkevich, cross-moments, and ε.
    let sigma2_s = k2 * sigma2_lin_real;
    let sigma_s = sigma2_s.max(0.0).sqrt();

    // Unbiased cross-moments at σ_s — for b₂ and b_{s²} tree pieces.
    let cross = doroshkevich::doroshkevich_unbiased_cross_moments(sigma_s, 80, 6.0);

    // Tree level: full Lagrangian bias expansion.
    // Matter piece −b₁ K₁ ξ̄_L uses the SINGLE-W ξ̄, consistent with the
    // cross-correlation structure we were already using in the b₁-only code.
    let xibar_tree_b1 = -bias.b1 * k1 * xi_bar_real;
    let xibar_tree_b2 = bias.b2 * cross.m12;
    let xibar_tree_bs2 = bias.bs2 * cross.m_s2;
    let xibar_tree = xibar_tree_b1 + xibar_tree_b2 + xibar_tree_bs2;

    // Layer 1: exact Zel'dovich with biased Doroshkevich at σ_s.
    // NOTE: currently uses the b₁-only exponential tilt; b₂/b_{s²} tilts are
    // implemented in the polynomial-weight PDF path used by the tilted-PDF
    // kNN predictions. This ξ̄_zel therefore only uses b₁.
    let xibar_zel = doroshkevich::doroshkevich_xibar_biased(sigma_s, bias.b1, 80, 6.0);

    // Layer 2: one-loop P₁₃ (single W). Leading-order RSD enhancement: K₁.
    let xibar_1loop_raw_real = if let Some(ref xi_p13) = xi_tables.xi_p13_eff {
        -1.070 * fftlog::xi_bar_from_xi(xi_p13, r, 64)
    } else {
        -1.070 * integrals::xibar_p13_raw_ws(r, ws, ip_loop)
    };
    let xibar_1loop = bias.b1 * k1 * xibar_1loop_raw_real;
    let _ = ip_hires;  // reserved for fallback paths

    // Layer 3: geometric series in ε = (3/7)² σ²_s
    let eps = (3.0 / 7.0_f64).powi(2) * sigma2_s;

    // One-loop b₂ and b_{s²} corrections from the biased F₂ kernel in the
    // cross-spectrum P₂₂-type piece. R-independent effective "pseudo P(k)"s
    // are precomputed in xi_tables and converted to ξ(r) via FFTLog; the
    // single-W R smoothing is the same as for the tree-level ξ̄.
    let xibar_1loop_b2 = if bias.b2 != 0.0 {
        if let Some(ref xi_b2) = xi_tables.xi_p22_cross_b2 {
            bias.b2 * k1 * fftlog::xi_bar_from_xi(xi_b2, r, 64)
        } else { 0.0 }
    } else { 0.0 };
    let xibar_1loop_bs2 = if bias.bs2 != 0.0 {
        if let Some(ref xi_bs2) = xi_tables.xi_p22_cross_bs2 {
            bias.bs2 * k1 * fftlog::xi_bar_from_xi(xi_bs2, r, 64)
        } else { 0.0 }
    } else { 0.0 };

    let xibar_full = if n_corrections == 0 {
        xibar_zel + xibar_1loop_b2 + xibar_1loop_bs2
    } else {
        let mut series_sum = xibar_1loop;
        let mut term = xibar_1loop;
        for _ in 1..n_corrections {
            term *= -eps;
            series_sum += term;
        }
        // b₂ / b_{s²} pieces are distinct loop structures, not in the Zel'dovich
        // geometric-series ladder — they add linearly on top of the resummed
        // b₁ tower.
        xibar_zel + series_sum + xibar_1loop_b2 + xibar_1loop_bs2
    };

    XibarJDetailed {
        r,
        b1: bias.b1, b2: bias.b2, bs2: bias.bs2, f_growth,
        xibar_tree, xibar_zel, xibar_1loop, xibar_full,
        sigma2_lin: sigma2_s,
        epsilon: eps,
        m12: cross.m12, m_s2: cross.m_s2,
        xibar_1loop_b2, xibar_1loop_bs2,
    }
}

/// Real-space ξ̄_J prediction at a single radius (three-layer, b₁ only).
pub fn xibar_j_full(
    cosmo: &Cosmology, r: f64, b1: f64, n_corrections: usize,
) -> XibarJDetailed {
    xibar_j_full_bias(cosmo, r, doroshkevich::BiasParams::b1_only(b1), n_corrections)
}

/// Real-space ξ̄_J prediction at a single radius with full Lagrangian bias.
pub fn xibar_j_full_bias(
    cosmo: &Cosmology, r: f64, bias: doroshkevich::BiasParams, n_corrections: usize,
) -> XibarJDetailed {
    let (ip_hires, ip_loop) = xibar_j_integration_params();
    let mut ws = Workspace::new(4000);
    ws.update_cosmology(cosmo);
    let xi_tables = fftlog::build_xi_tables(cosmo, fftlog::FFTLogConfig::default(), false, true, true);
    xibar_j_with_workspace(&ws, &xi_tables, &ip_hires, &ip_loop, r, bias, n_corrections, None)
}

/// Redshift-space ξ̄_J prediction at a single radius (b₁-only).
pub fn xibar_j_full_rsd(
    cosmo: &Cosmology, r: f64, b1: f64, rsd: &integrals::RsdParams,
    n_corrections: usize,
) -> XibarJDetailed {
    xibar_j_full_rsd_bias(cosmo, r, doroshkevich::BiasParams::b1_only(b1), rsd, n_corrections)
}

/// Redshift-space ξ̄_J prediction with full Lagrangian bias.
pub fn xibar_j_full_rsd_bias(
    cosmo: &Cosmology, r: f64, bias: doroshkevich::BiasParams,
    rsd: &integrals::RsdParams, n_corrections: usize,
) -> XibarJDetailed {
    let (ip_hires, ip_loop) = xibar_j_integration_params();
    let mut ws = Workspace::new(4000);
    ws.update_cosmology(cosmo);
    let xi_tables = fftlog::build_xi_tables(cosmo, fftlog::FFTLogConfig::default(), false, true, true);
    xibar_j_with_workspace(&ws, &xi_tables, &ip_hires, &ip_loop, r, bias, n_corrections, Some(rsd))
}

/// Compute ξ̄_J on a grid of radii (real space, b₁-only).
pub fn xibar_j_plot(
    cosmo: &Cosmology, radii: &[f64], b1: f64, n_corrections: usize,
) -> Vec<XibarJDetailed> {
    xibar_j_plot_bias(cosmo, radii, doroshkevich::BiasParams::b1_only(b1), n_corrections)
}

/// Compute ξ̄_J on a grid of radii with full Lagrangian bias.
pub fn xibar_j_plot_bias(
    cosmo: &Cosmology, radii: &[f64], bias: doroshkevich::BiasParams, n_corrections: usize,
) -> Vec<XibarJDetailed> {
    use rayon::prelude::*;
    let (ip_hires, ip_loop) = xibar_j_integration_params();
    let mut ws = Workspace::new(4000);
    ws.update_cosmology(cosmo);
    let xi_tables = fftlog::build_xi_tables(cosmo, fftlog::FFTLogConfig::default(), false, true, true);
    radii.par_iter()
        .map(|&r| xibar_j_with_workspace(&ws, &xi_tables, &ip_hires, &ip_loop,
                                         r, bias, n_corrections, None))
        .collect()
}

/// Compute ξ̄_J in redshift space on a grid of radii (b₁-only).
pub fn xibar_j_plot_rsd(
    cosmo: &Cosmology, radii: &[f64], b1: f64, rsd: &integrals::RsdParams,
    n_corrections: usize,
) -> Vec<XibarJDetailed> {
    xibar_j_plot_rsd_bias(cosmo, radii, doroshkevich::BiasParams::b1_only(b1), rsd, n_corrections)
}

/// Compute ξ̄_J in redshift space with full Lagrangian bias.
pub fn xibar_j_plot_rsd_bias(
    cosmo: &Cosmology, radii: &[f64], bias: doroshkevich::BiasParams,
    rsd: &integrals::RsdParams, n_corrections: usize,
) -> Vec<XibarJDetailed> {
    use rayon::prelude::*;
    let (ip_hires, ip_loop) = xibar_j_integration_params();
    let mut ws = Workspace::new(4000);
    ws.update_cosmology(cosmo);
    let xi_tables = fftlog::build_xi_tables(cosmo, fftlog::FFTLogConfig::default(), false, true, true);
    radii.par_iter()
        .map(|&r| xibar_j_with_workspace(&ws, &xi_tables, &ip_hires, &ip_loop,
                                         r, bias, n_corrections, Some(rsd)))
        .collect()
}

#[cfg(test)]
mod diagnostics_flag_tests {
    use super::*;

    /// Regression: σ²_J and all non-diagnostic fields must be identical whether
    /// compute_diagnostics is on or off. Only p22, two_p13, s3_matter,
    /// s3_jacobian differ (zero vs populated).
    #[test]
    fn diagnostics_flag_preserves_sigma2_j() {
        let cosmo = Cosmology::planck2018();
        let masses = [1e13_f64, 1e14, 1e15];
        let with = sigma2_j_plot_at_masses(&cosmo, &masses, 3, 0.0, 0.0, true);
        let without = sigma2_j_plot_at_masses(&cosmo, &masses, 3, 0.0, 0.0, false);
        assert_eq!(with.len(), without.len());
        for (a, b) in with.iter().zip(without.iter()) {
            assert_eq!(a.sigma2_j, b.sigma2_j, "sigma2_j differs at M={}", a.mass);
            assert_eq!(a.sigma2_zel, b.sigma2_zel);
            assert_eq!(a.sigma2_lin, b.sigma2_lin);
            assert_eq!(a.xi_bar, b.xi_bar);
            assert_eq!(a.d1, b.d1);
            assert_eq!(a.d2, b.d2);
            assert_eq!(a.d3, b.d3);
            assert_eq!(a.counterterm, b.counterterm);
            // Diagnostics: nonzero when on, exactly zero when off
            assert!(a.p22.abs() > 0.0, "p22 should be nonzero when diagnostics=true");
            assert_eq!(b.p22, 0.0, "p22 should be 0 when diagnostics=false");
            assert!(a.two_p13.abs() > 0.0);
            assert_eq!(b.two_p13, 0.0);
            assert!(a.s3_matter.abs() > 0.0);
            assert_eq!(b.s3_matter, 0.0);
            assert!(a.s3_jacobian.abs() > 0.0);
            assert_eq!(b.s3_jacobian, 0.0);
        }
    }
}

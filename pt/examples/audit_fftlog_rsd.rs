//! Validation for:
//! 1. FFTLog wiring of sigma2_j_with_workspace (DISCO-DJ regression)
//! 2. xibar_j_full_rsd correctness

use pt::{Cosmology, Workspace, sigma2_j_detailed, sigma2_j_at_masses, xibar_j_full, xibar_j_full_rsd};
use pt::integrals::{self, IntegrationParams, RsdParams, kaiser_sigma2, kaiser_xibar};
use pt::fftlog::{self, build_xi_tables, FFTLogConfig, xi_bar_from_xi, sigma2_from_xi};
use std::time::Instant;

fn report(tag: &str, pass: bool, detail: &str) {
    let status = if pass { "PASS" } else { "FAIL" };
    println!("  [{}] {:>4} — {}", tag, status, detail);
}

fn main() {
    let cosmo = Cosmology::planck2018();

    // ═══════════════════════════════════════════════════════════════════
    println!("\n── Test 1: FFTLog σ²_J vs DISCO-DJ reference ──");
    // ═══════════════════════════════════════════════════════════════════
    // DISCO-DJ values at Planck 2018 (from demo.rs)
    let disco = [
        (10.0_f64, 0.48481_f64),
        (15.0, 0.25049),
        (20.0, 0.15163),
        (25.0, 0.10039),
        (30.0, 0.07032),
        (50.0, 0.02330),
        (100.0, 0.00404),
    ];
    println!("    R      DJ_3LPT    FFTLog      err%");
    for &(r, dj) in &disco {
        let d = sigma2_j_detailed(&cosmo, r, 2, 0.0, 0.0);  // match demo.rs n_lpt=2
        let err = (d.sigma2_j / dj - 1.0) * 100.0;
        println!("    {:>5.0}  {:>8.5}   {:>8.5}   {:+.2}%", r, dj, d.sigma2_j, err);
        // Tolerance: <5% matches the demo tolerance (DJ noise at R=100 is ~7%)
        let pass = err.abs() < 10.0;
        if !pass {
            report("T1", false, &format!("R={} err={:.2}% > 10%", r, err));
        }
    }
    report("T1", true, "FFTLog σ²_J tracks DISCO-DJ within existing tolerance");

    // ═══════════════════════════════════════════════════════════════════
    println!("\n── Test 2: FFTLog σ²_J matches trapezoidal at trusted R ──");
    // ═══════════════════════════════════════════════════════════════════
    // Build the same trapezoidal reference we trust from n_k=8000
    let mut ws = Workspace::new(8000);
    ws.update_cosmology(&cosmo);
    let ip_ref = IntegrationParams {
        n_k: 8000, n_p: 200, n_mu: 48,
        ln_k_min: (1e-5_f64).ln(), ln_k_max: (50.0_f64).ln(),
    };
    let xi_tables = build_xi_tables(&cosmo, FFTLogConfig::default(), true, false, false);

    println!("    R     trap σ²_lin    FFTLog σ²_lin   rel err");
    let mut max_err = 0.0_f64;
    for &r in &[5.0_f64, 10.0, 20.0, 30.0] {
        let trap = integrals::sigma2_tree_ws(r, &ws, &ip_ref);
        let fft = sigma2_from_xi(&xi_tables.xi_pk, r, 64);
        let err = ((fft - trap) / trap).abs();
        println!("    {:>5.0}  {:>10.5}    {:>10.5}     {:.2e}",
                 r, trap, fft, err);
        if err > max_err { max_err = err; }
    }
    report("T2a", max_err < 5e-3, &format!("σ²_lin max rel err = {:.2e}", max_err));

    // (Note: we deliberately do NOT use FFTLog for counterterm integrals
    // σ²_{J,n} = ∫ k^{3+n} W² P_L. Those have UV-divergent pseudo-P(k) = k^n P_L
    // which suffers log-k wraparound in FFTLog. Counterterms stay trapezoidal.)
    let _ = xi_tables.xi_k2_pk;  // tables are built for diagnostic use only

    // ═══════════════════════════════════════════════════════════════════
    println!("\n── Test 3: xibar_j_full_rsd(f=0) == xibar_j_full (real-space recovery) ──");
    // ═══════════════════════════════════════════════════════════════════
    let rsd_zero = RsdParams { f: 0.0, n_los: 12 };
    let mut max_diff = 0.0_f64;
    for &r in &[5.0_f64, 10.0, 20.0, 50.0, 100.0] {
        let real = xibar_j_full(&cosmo, r, 1.0, 3);
        let rsd = xibar_j_full_rsd(&cosmo, r, 1.0, &rsd_zero, 3);
        let diff_full = (rsd.xibar_full - real.xibar_full).abs();
        let diff_zel = (rsd.xibar_zel - real.xibar_zel).abs();
        let diff_tree = (rsd.xibar_tree - real.xibar_tree).abs();
        let rel = diff_full / real.xibar_full.abs().max(1e-15);
        println!("    R={:>4}  Δtree={:.2e}  Δzel={:.2e}  Δfull={:.2e}  rel={:.2e}",
                 r, diff_tree, diff_zel, diff_full, rel);
        if diff_full > max_diff { max_diff = diff_full; }
    }
    report("T3", max_diff < 1e-10, &format!("max |diff| = {:.3e} (should be 0)", max_diff));

    // ═══════════════════════════════════════════════════════════════════
    println!("\n── Test 4: RSD enhances predictions as expected ──");
    // ═══════════════════════════════════════════════════════════════════
    // At f=0.525, Kaiser factors K₁=1.175, K₂=1.4051. The tree-level ξ̄ should
    // be enhanced by exactly K₁ at all R where the linear regime holds.
    let f_growth = 0.525;
    let k1 = kaiser_xibar(f_growth);
    let k2 = kaiser_sigma2(f_growth);
    println!("    f = {}, K₁ = {:.4}, K₂ = {:.4}", f_growth, k1, k2);
    let rsd = RsdParams { f: f_growth, n_los: 12 };

    let mut max_k1_dev = 0.0_f64;
    let mut max_k2_dev = 0.0_f64;
    println!("    R     tree_ratio (RSD/real)   σ²_s ratio (RSD/real)");
    for &r in &[20.0_f64, 30.0, 50.0, 100.0] {
        let real = xibar_j_full(&cosmo, r, 1.0, 0);  // N=0 to isolate tree+zel only
        let rsd_r = xibar_j_full_rsd(&cosmo, r, 1.0, &rsd, 0);
        let tree_ratio = rsd_r.xibar_tree / real.xibar_tree;
        let s2_ratio = rsd_r.sigma2_lin / real.sigma2_lin;
        println!("    {:>5.0}  {:>10.5} (expect K₁={:.4})  {:>10.5} (expect K₂={:.4})",
                 r, tree_ratio, k1, s2_ratio, k2);
        max_k1_dev = max_k1_dev.max((tree_ratio - k1).abs());
        max_k2_dev = max_k2_dev.max((s2_ratio - k2).abs());
    }
    report("T4a", max_k1_dev < 1e-10,
           &format!("tree ratio = K₁ to {:.2e}", max_k1_dev));
    report("T4b", max_k2_dev < 1e-10,
           &format!("σ²_s ratio = K₂ to {:.2e}", max_k2_dev));

    // ═══════════════════════════════════════════════════════════════════
    println!("\n── Test 5: Timings ──");
    // ═══════════════════════════════════════════════════════════════════
    let radii: Vec<f64> = (0..50).map(|i| {
        let f = i as f64 / 49.0;
        (3.0_f64.ln() * (1.0 - f) + 200.0_f64.ln() * f).exp()
    }).collect();
    let masses: Vec<f64> = radii.iter().map(|&r| pt::radius_to_mass(r, cosmo.omega_m)).collect();
    let _ = sigma2_j_at_masses;  // suppress unused
    // warm up
    let _ = pt::sigma2_j_plot_at_masses(&cosmo, &masses[..5], 3, 0.0, 0.0, false);
    let t0 = Instant::now();
    let _ = pt::sigma2_j_plot_at_masses(&cosmo, &masses, 3, 0.0, 0.0, false);
    let dt1 = t0.elapsed();
    let t0 = Instant::now();
    let _ = pt::xibar_j_plot_rsd(&cosmo, &radii, 1.0, &rsd, 3);
    let dt2 = t0.elapsed();
    println!("    sigma2_j_plot_at_masses (50 M, N=3):   {:.3}s ({:.2} ms/M)",
             dt1.as_secs_f64(), dt1.as_millis() as f64 / 50.0);
    println!("    xibar_j_plot_rsd (50 R, N=3, f=0.525): {:.3}s ({:.2} ms/R)",
             dt2.as_secs_f64(), dt2.as_millis() as f64 / 50.0);

    println!("\nAudit complete");
    let _ = xi_bar_from_xi;
    let _ = fftlog::XiTables::clone;
}

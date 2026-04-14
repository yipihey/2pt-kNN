//! Comprehensive self-audit of the pt crate.
//! Covers: Parts B (convergence), C (physical sanity), E1/E4 (cross-checks), F (RSD).

use pt::{Cosmology, Workspace, mass_to_radius};
use pt::integrals::{
    self, IntegrationParams, RsdParams,
    kaiser_sigma2, kaiser_xibar,
};
use pt::doroshkevich::{
    doroshkevich_moments, doroshkevich_xibar_biased, doroshkevich_cdf_biased,
    sigma2_zel_perturbative,
};
use pt::fftlog::{build_xi_tables, FFTLogConfig, xi_bar_from_xi, sigma2_from_xi};

fn header(section: &str) {
    println!("\n══════════════════════════════════════════════════════════════");
    println!("  {}", section);
    println!("══════════════════════════════════════════════════════════════");
}

fn report(tag: &str, pass: bool, detail: &str) {
    let status = if pass { "PASS" } else { "FAIL" };
    println!("  [{}] {:>6} — {}", tag, status, detail);
}

fn main() {
    let cosmo = Cosmology::planck2018();

    // ═══════════════════════════════════════════════════════════════════
    header("B1. Doroshkevich quadrature convergence");
    // ═══════════════════════════════════════════════════════════════════
    for &sigma in &[0.5_f64, 1.0, 1.5, 2.0] {
        let mut prev_var = 0.0;
        let mut max_rel_delta = 0.0f64;
        for &n_g in &[40_usize, 60, 80, 100, 120] {
            let m = doroshkevich_moments(sigma, n_g, 6.0);
            let mean_j_err = (m.mean_j - 1.0).abs();
            print!("    σ={:.1}  n_g={:>3}  Var={:.8}  <J>−1={:+.2e}",
                   sigma, n_g, m.variance, m.mean_j - 1.0);
            if prev_var > 0.0 {
                let rd = ((m.variance - prev_var) / prev_var).abs();
                if n_g >= 60 && rd > max_rel_delta { max_rel_delta = rd; }
                print!("  Δ={:.2e}", rd);
            }
            println!();
            prev_var = m.variance;
            // mass conservation check
            if mean_j_err > 1e-5 {
                report("B1", false, &format!("⟨J⟩=1 fails at σ={} n_g={}: {:.3e}",
                                             sigma, n_g, mean_j_err));
            }
        }
        let pass = max_rel_delta < 1e-4;
        report("B1", pass, &format!("σ={:.1}: max Δ(n_g≥60) = {:.2e} (tol 1e-4)",
                                    sigma, max_rel_delta));
    }

    // ═══════════════════════════════════════════════════════════════════
    header("B3. σ_L(R=8) vs input σ_8 = 0.8111");
    // ═══════════════════════════════════════════════════════════════════
    let mut ws = Workspace::new(8000);
    ws.update_cosmology(&cosmo);
    let ip_fine = IntegrationParams {
        n_k: 8000, n_p: 200, n_mu: 48,
        ln_k_min: (1e-5_f64).ln(), ln_k_max: (50.0_f64).ln(),
    };
    let s2_at_8 = integrals::sigma2_tree_ws(8.0, &ws, &ip_fine);
    let sigma8_computed = s2_at_8.sqrt();
    let sigma8_rel = (sigma8_computed - cosmo.sigma8).abs() / cosmo.sigma8;
    println!("    Computed σ₈ = {:.6}  (input {:.4})  rel err = {:.2e}",
             sigma8_computed, cosmo.sigma8, sigma8_rel);
    report("B3a", sigma8_rel < 1e-3, "σ₈ matches input");

    // Print σ_L(R) at standard radii
    println!("    σ_L(R) at standard R:");
    for &r in &[1.0_f64, 2.0, 5.0, 10.0, 20.0, 50.0] {
        let s2 = integrals::sigma2_tree_ws(r, &ws, &ip_fine);
        println!("      R={:>4.0}   σ_L = {:.6}", r, s2.sqrt());
    }

    // W(kR→0) limit test
    let w_small = pt::cosmology::top_hat(1e-8);
    let w_exact_small = 1.0 - (1e-8_f64).powi(2) / 10.0;
    report("B3b", (w_small - w_exact_small).abs() < 1e-15,
           &format!("W(kR→0)={:.14} matches Taylor {:.14}", w_small, w_exact_small));

    // ═══════════════════════════════════════════════════════════════════
    header("C1. Limiting behaviours");
    // ═══════════════════════════════════════════════════════════════════
    // C1a: xibar_biased(b1=0) = 0 (mass conservation)
    for &sigma in &[0.1_f64, 0.5, 1.0, 1.5] {
        let xb = doroshkevich_xibar_biased(sigma, 0.0, 80, 6.0);
        report("C1a", xb.abs() < 1e-10,
               &format!("ξ̄_zel(σ={}, b₁=0) = {:.3e} (should be ≈0)", sigma, xb));
    }

    // C1b: small-b₁ limit of ξ̄_zel
    // From our earlier analysis: ξ̄_zel ≈ -6 b₁ σ²_lin (with ~6× enhancement vs naive).
    // Verify the ratio is ~6.0 at small σ (as our previous tests showed).
    let sigma_small = 0.1;
    let b1_small = 0.01;
    let xb_s = doroshkevich_xibar_biased(sigma_small, b1_small, 80, 6.0);
    let xb_tree_naive = -b1_small * sigma_small.powi(2);
    let enhancement = xb_s / xb_tree_naive;
    println!("    σ={}, b₁={}: ⟨J-1⟩ = {:.3e},  naive tree = {:.3e},  ratio = {:.3}",
             sigma_small, b1_small, xb_s, xb_tree_naive, enhancement);
    report("C1b", (enhancement - 6.0).abs() < 0.1,
           &format!("small-b₁ ratio ⟨J-1⟩/(-b₁σ²) = {:.3} (expected ≈6.0)", enhancement));

    // ═══════════════════════════════════════════════════════════════════
    header("C2. Monotonicity");
    // ═══════════════════════════════════════════════════════════════════
    // C2a: σ²(R) monotonically decreasing in R
    let xi_tables = build_xi_tables(&cosmo, FFTLogConfig::default(), false, false, false);
    let r_grid: Vec<f64> = (0..30).map(|i| {
        let f = i as f64 / 29.0;
        (2.0_f64.ln() * (1.0 - f) + 200.0_f64.ln() * f).exp()
    }).collect();
    let sigma2_vals: Vec<f64> = r_grid.iter().map(|&r| sigma2_from_xi(&xi_tables.xi_pk, r, 64)).collect();
    let mut mono = true;
    for i in 1..sigma2_vals.len() {
        if sigma2_vals[i] > sigma2_vals[i-1] { mono = false; break; }
    }
    report("C2a", mono, "σ²(R) monotonically decreasing");

    // C2b: ξ̄(R, b₁>0) must be negative
    let xibar_1 = doroshkevich_xibar_biased(0.5, 1.0, 80, 6.0);
    report("C2b", xibar_1 < 0.0,
           &format!("ξ̄_zel(σ=0.5, b₁=1) = {:.6e} (must be <0)", xibar_1));

    // ═══════════════════════════════════════════════════════════════════
    header("C3. Doroshkevich vs perturbative crosscheck");
    // ═══════════════════════════════════════════════════════════════════
    // C3a: σ=0.3 should match perturbatively to < 0.1%
    for &sigma in &[0.3_f64, 0.5, 1.0, 1.5] {
        let m = doroshkevich_moments(sigma, 80, 6.0);
        let s2_exact = m.variance;
        let s2_pert = sigma2_zel_perturbative(sigma * sigma);
        let rel = (s2_pert - s2_exact) / s2_exact;
        println!("    σ={:.1}  Var_exact={:.6}  Var_pert={:.6}  rel={:+.4}%",
                 sigma, s2_exact, s2_pert, rel * 100.0);
        if sigma <= 0.3 {
            report("C3a", rel.abs() < 1e-3,
                   &format!("σ={:.1}: exact/pert disagreement = {:.2e} (tol 1e-3)", sigma, rel.abs()));
        } else if sigma >= 1.5 {
            // Perturbative should overshoot by ~10-15%
            report("C3b", rel > 0.05 && rel < 0.5,
                   &format!("σ={:.1}: pert overshoot = {:+.1}% (expected 5-50%)", sigma, rel * 100.0));
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    header("F1. Kaiser factors");
    // ═══════════════════════════════════════════════════════════════════
    // F1: Verify K₁, K₂ at f=0.525
    let f = 0.525;
    let k2 = kaiser_sigma2(f);
    let k1 = kaiser_xibar(f);
    let k2_expected = 1.0 + 2.0 * f / 3.0 + f * f / 5.0;
    let k1_expected = 1.0 + f / 3.0;
    println!("    f = {}", f);
    println!("    K₂ = {:.6} (expected 1+2f/3+f²/5 = {:.6})", k2, k2_expected);
    println!("    K₁ = {:.6} (expected 1+f/3   = {:.6})", k1, k1_expected);
    report("F1a", (k2 - k2_expected).abs() < 1e-10, "K₂ formula");
    report("F1b", (k1 - k1_expected).abs() < 1e-10, "K₁ formula");
    report("F1c", (k2 - 1.4051).abs() < 1e-3, &format!("K₂ @f=0.525 = {:.4} (≈1.4051)", k2));
    report("F1d", (k1 - 1.1750).abs() < 1e-3, &format!("K₁ @f=0.525 = {:.4} (≈1.1750)", k1));

    // F2: Numerical verification by μ-integration
    let n_mu = 2001;
    let dmu = 2.0 / (n_mu - 1) as f64;
    let mut int_k2 = 0.0_f64;
    let mut int_k1 = 0.0_f64;
    for i in 0..n_mu {
        let mu = -1.0 + i as f64 * dmu;
        let w = if i == 0 || i == n_mu - 1 { 0.5 } else { 1.0 };
        let kaiser_fac = 1.0 + f * mu * mu;
        int_k2 += w * kaiser_fac * kaiser_fac;
        int_k1 += w * kaiser_fac;
    }
    int_k2 *= dmu / 2.0;  // average over μ ∈ [-1,1] is divide by 2
    int_k1 *= dmu / 2.0;
    report("F1e", (int_k2 - k2).abs() < 1e-4, &format!("⟨(1+fμ²)²⟩_μ = {:.5} vs K₂ = {:.5}", int_k2, k2));
    report("F1f", (int_k1 - k1).abs() < 1e-4, &format!("⟨(1+fμ²)⟩_μ = {:.5} vs K₁ = {:.5}", int_k1, k1));

    // ═══════════════════════════════════════════════════════════════════
    header("F2. Real-space recovery (f=0)");
    // ═══════════════════════════════════════════════════════════════════
    // At f=0, K₂ = K₁ = 1; σ²_s = σ²_L; ξ̄_s = ξ̄_real
    let k2_at_0 = kaiser_sigma2(0.0);
    let k1_at_0 = kaiser_xibar(0.0);
    report("F2a", (k2_at_0 - 1.0).abs() < 1e-15, &format!("K₂(f=0) = {:.15} (must be 1)", k2_at_0));
    report("F2b", (k1_at_0 - 1.0).abs() < 1e-15, &format!("K₁(f=0) = {:.15} (must be 1)", k1_at_0));

    let rsd_zero_f = RsdParams { f: 0.0, n_los: 12 };
    let xi_real = integrals::xi_bar_ws(20.0, &ws, &ip_fine);
    let xi_rsd = integrals::xi_bar_rsd(20.0, &rsd_zero_f, &ws, &ip_fine);
    report("F2c", (xi_real - xi_rsd).abs() < 1e-10,
           &format!("xi_bar_rsd(f=0) = xi_bar_real: {:.6e} vs {:.6e}", xi_real, xi_rsd));

    // ═══════════════════════════════════════════════════════════════════
    header("C4. One-loop sign — P₂₂ positive, P₁₃ negative");
    // ═══════════════════════════════════════════════════════════════════
    // Use moderate R where loops are sensible
    for &r in &[5.0_f64, 10.0, 20.0] {
        let p22_raw = integrals::sigma2_p22_raw_ws(r, &ws, &ip_fine);
        let p13_raw = integrals::sigma2_p13_raw_ws(r, &ws, &ip_fine);
        let p22 = 0.1803 * p22_raw;      // diagnostic prefactor
        let tp13 = -1.070 * p13_raw;      // diagnostic prefactor (includes negative sign)
        let tree = integrals::sigma2_tree_ws(r, &ws, &ip_fine);
        println!("    R={:>4}  tree={:>9.5}  P₂₂={:+.5}  2P₁₃={:+.5}  1-loop/tree={:+.3}%",
                 r, tree, p22, tp13, (p22 + tp13) / tree * 100.0);
        report("C4a", p22 > 0.0, &format!("R={}: P₂₂ > 0", r));
        report("C4b", tp13 < 0.0, &format!("R={}: 2P₁₃ < 0", r));
    }

    // ═══════════════════════════════════════════════════════════════════
    header("E4. kNN CDF normalization and bounds");
    // ═══════════════════════════════════════════════════════════════════
    let sigma_test = 0.5;
    let j_grid: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();  // 0 to 10
    let cdf = doroshkevich_cdf_biased(sigma_test, 0.0, &j_grid, 80, 6.0);
    let cdf_monotone = cdf.windows(2).all(|w| w[1] >= w[0] - 1e-12);
    report("E4a", cdf_monotone, "Doroshkevich CDF monotonic");
    report("E4b", cdf[0] < 1e-6, &format!("CDF(J=0) = {:.3e} (→0)", cdf[0]));
    report("E4c", cdf.last().unwrap() > &0.999, &format!("CDF(J=10) = {:.6} (→1)", cdf.last().unwrap()));

    // CDF at J=1 should be close to 0.5 for small σ (symmetric around J=1)
    let cdf_half = doroshkevich_cdf_biased(sigma_test, 0.0, &[1.0], 80, 6.0);
    println!("    σ={}: P(J<1) = {:.6}", sigma_test, cdf_half[0]);

    // ═══════════════════════════════════════════════════════════════════
    header("Audit complete");
    // ═══════════════════════════════════════════════════════════════════
}

use pt::{Cosmology, Workspace};
use pt::integrals::{IntegrationParams, xi_bar_ws, sigma2_tree_ws};
use pt::fftlog::{build_xi_tables, FFTLogConfig, xi_bar_from_xi, sigma2_from_xi};

fn main() {
    let cosmo = Cosmology::planck2018();

    println!("=== Convergence check at R = 80 Mpc/h ===");
    println!("\n--- Quadrature (xi_bar) ---");
    for &n_k in &[2000, 4000, 8000, 16000, 32000] {
        let mut ws = Workspace::new(n_k);
        ws.update_cosmology(&cosmo);
        let ip = IntegrationParams {
            n_k, n_p: 200, n_mu: 48,
            ln_k_min: (1e-5_f64).ln(), ln_k_max: (50.0_f64).ln(),
        };
        let r = xi_bar_ws(80.0, &ws, &ip);
        println!("  n_k={:>6}  xi_bar(80) = {:.9}", n_k, r);
    }

    println!("\n--- FFTLog (xi_bar) ---");
    for &n in &[1024, 2048, 4096, 8192, 16384] {
        let cfg = FFTLogConfig {
            n, ln_k_min: (1e-5_f64).ln(), ln_k_max: (50.0_f64).ln(), lowring: true,
        };
        let tables = build_xi_tables(&cosmo, cfg, false, false, false);
        let r = xi_bar_from_xi(&tables.xi_pk, 80.0, 64);
        println!("  n={:>6}  xi_bar(80) = {:.9}", n, r);
    }

    println!("\n=== Wider k-range, FFTLog ===");
    for &(kmin_log, kmax_log, n) in &[
        (-6.0, 2.0, 4096),
        (-7.0, 3.0, 8192),
        (-8.0, 3.0, 8192),
    ] {
        let cfg = FFTLogConfig {
            n,
            ln_k_min: 10_f64.powf(kmin_log).ln(),
            ln_k_max: 10_f64.powf(kmax_log).ln(),
            lowring: true,
        };
        let tables = build_xi_tables(&cosmo, cfg, false, false, false);
        let r = xi_bar_from_xi(&tables.xi_pk, 80.0, 64);
        println!("  k=[1e{}, 1e{}], n={:>6}  xi_bar(80) = {:.9}",
                 kmin_log, kmax_log, n, r);
    }
}

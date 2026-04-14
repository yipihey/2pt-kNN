//! Timing benchmark for the pt crate production paths.
use pt::{Cosmology, sigma2_j_plot_at_masses, xibar_j_plot};
use std::time::Instant;

fn main() {
    let cosmo = Cosmology::new(0.3089, 0.0486, 0.6774, 0.9667, 0.8159);

    // 50 R values, log-spaced
    let radii: Vec<f64> = (0..50).map(|i| {
        let f = i as f64 / 49.0;
        (3.0_f64.ln() * (1.0 - f) + 200.0_f64.ln() * f).exp()
    }).collect();
    let masses: Vec<f64> = radii.iter().map(|&r| pt::radius_to_mass(r, cosmo.omega_m)).collect();

    // Warm up (build once)
    let _ = xibar_j_plot(&cosmo, &radii[..5], 1.0, 3);
    let _ = sigma2_j_plot_at_masses(&cosmo, &masses[..5], 3, 0.0, 0.0, false);

    // σ²_J at 50 masses, diagnostics=false (production fast path)
    let t0 = Instant::now();
    let _ = sigma2_j_plot_at_masses(&cosmo, &masses, 3, 0.0, 0.0, false);
    let dt_sigma_fast = t0.elapsed();

    // σ²_J at 50 masses, diagnostics=true (with 3D P22/P13/S3 integrals)
    let t0 = Instant::now();
    let _ = sigma2_j_plot_at_masses(&cosmo, &masses, 3, 0.0, 0.0, true);
    let dt_sigma_diag = t0.elapsed();

    // ξ̄_J at 50 R values (full 3-layer + FFTLog P₁₃)
    let t0 = Instant::now();
    let _ = xibar_j_plot(&cosmo, &radii, 1.0, 3);
    let dt_xibar = t0.elapsed();

    println!("Timing (Planck-like cosmo, 50 R values):");
    println!("  sigma2_j_plot_at_masses (N=3, fast):         {:.3}s  ({:.2} ms/R)",
             dt_sigma_fast.as_secs_f64(), dt_sigma_fast.as_millis() as f64 / 50.0);
    println!("  sigma2_j_plot_at_masses (N=3, diagnostics):  {:.3}s  ({:.2} ms/R)",
             dt_sigma_diag.as_secs_f64(), dt_sigma_diag.as_millis() as f64 / 50.0);
    let speedup = dt_sigma_diag.as_secs_f64() / dt_sigma_fast.as_secs_f64().max(1e-9);
    println!("  → diagnostics=false speedup:                {:.1}×", speedup);
    println!("  xibar_j_plot (N=3, FFTLog+P₁₃):              {:.3}s  ({:.2} ms/R)",
             dt_xibar.as_secs_f64(), dt_xibar.as_millis() as f64 / 50.0);

    // Large batch: 400 R values, BAO-dense
    let r_bao: Vec<f64> = {
        let r_log: Vec<f64> = (0..150).map(|i| {
            let f = i as f64 / 149.0;
            (3.0_f64.ln() * (1.0 - f) + 300.0_f64.ln() * f).exp()
        }).collect();
        let r_bao: Vec<f64> = (0..250).map(|i| 60.0 + 100.0 * i as f64 / 249.0).collect();
        let mut v: Vec<f64> = r_log.into_iter().chain(r_bao.into_iter()).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
        v
    };
    let t0 = Instant::now();
    let _ = xibar_j_plot(&cosmo, &r_bao, 1.0, 3);
    let dt_bao = t0.elapsed();
    println!();
    println!("  xibar_j_plot, {} R values (BAO-dense): {:.3}s  ({:.2} ms/R)",
             r_bao.len(), dt_bao.as_secs_f64(),
             dt_bao.as_millis() as f64 / r_bao.len() as f64);
}

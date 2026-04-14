//! Reference table of predictions at Planck 2018 cosmology, z=0, f=0.525.
//!
//! Cosmology per user's audit spec:
//!   Ω_m = 0.3089, σ₈ = 0.8159, n_s = 0.9667, h = 0.6774, Ω_b = 0.0486

use pt::{Cosmology, radius_to_mass, mass_to_radius};
use pt::integrals::kaiser_sigma2;
use pt::{xibar_j_full, sigma2_j_detailed};
use std::time::Instant;

fn main() {
    let cosmo = Cosmology::new(0.3089, 0.0486, 0.6774, 0.9667, 0.8159);
    let f_growth = 0.525;
    let k2 = kaiser_sigma2(f_growth);

    // Spec R values
    let radii = [3.0_f64, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0];

    println!("# Reference Predictions — Planck 2018 (per audit spec)");
    println!("#   Ωm={} Ωb={} h={} ns={} σ8={}  z=0, f={}",
             cosmo.omega_m, cosmo.omega_b, cosmo.h, cosmo.n_s, cosmo.sigma8, f_growth);
    println!("#   Kaiser K₂(f=0.525) = {:.4}  (σ²_s = K₂ × σ²_L)", k2);
    println!();
    println!("# R [Mpc/h]  M [Msun/h]   σ²_J       σ²_Zel    σ²_lin    ξ̄(b₁=1)    ξ̄(b₁=2)    σ²_J·σ²_s  trunc_err");

    let start = Instant::now();
    for &r in &radii {
        let m = radius_to_mass(r, cosmo.omega_m);

        // σ²_J path (polynomial LPT, N=3)
        let s = sigma2_j_detailed(&cosmo, r, 3, 0.0, 0.0);

        // ξ̄_J with N=3 corrections, two b₁ values
        let xi1 = xibar_j_full(&cosmo, r, 1.0, 3);
        let xi2 = xibar_j_full(&cosmo, r, 2.0, 3);

        // Expansion parameter and truncation uncertainty
        // ε = (3/7)² × σ²_s for the physical interpretation (Kaiser-enhanced)
        let eps_s = (3.0 / 7.0_f64).powi(2) * k2 * s.sigma2_lin;
        // For real-space (as currently implemented), ε = (3/7)² σ²_L
        let _eps_real = xi1.epsilon;
        // Truncation error: |Δ_1loop| × ε^N / baseline
        // For ξ̄: next term ≈ |xibar_1loop| × ε^N
        let trunc_xibar = xi1.xibar_1loop.abs() * xi1.epsilon.powi(3);

        println!("{:>6.1}   {:>10.3e}   {:+.5}    {:+.5}   {:+.5}   {:+.5}    {:+.5}    {:+.5}    {:.3e}",
                 r, m,
                 s.sigma2_j, s.sigma2_zel, s.sigma2_lin,
                 xi1.xibar_full, xi2.xibar_full,
                 eps_s, trunc_xibar);
    }
    let elapsed = start.elapsed();
    println!();
    println!("# Wall-clock: {} R values in {:.3}s  ({:.1} ms/R)",
             radii.len(), elapsed.as_secs_f64(),
             elapsed.as_millis() as f64 / radii.len() as f64);

    // s₃ scan at the requested R values
    println!();
    println!("# s₃ matter and Jacobian skewness");
    println!("# R [Mpc/h]  s₃_matter  s₃_jacobian   s3_jacobian/σ²_J²");
    for &r in &radii {
        // Need detailed computation with compute_bispec=false (but s3 is always computed)
        let s = sigma2_j_detailed(&cosmo, r, 3, 0.0, 0.0);
        let s3_red = s.s3_jacobian / s.sigma2_j.powi(2);
        println!("{:>6.1}   {:+.5e}   {:+.5e}   {:.4}",
                 r, s.s3_matter, s.s3_jacobian, s3_red);
    }
    let _ = mass_to_radius;  // suppress unused
}

use pt::cosmology::Cosmology;
use pt::*;

fn main() {
    let cosmo = Cosmology::planck2018();
    println!("pt v0.2 — Perturbative σ²_J(M)");
    println!("  Ωm={:.4} Ωb={:.4} h={:.4} ns={:.4} σ8={:.4}",
             cosmo.omega_m, cosmo.omega_b, cosmo.h, cosmo.n_s, cosmo.sigma8);

    // ── Fast path (MCMC) ─────────────────────────────────────────────
    println!("\n═══ Fast path: polynomial D_n model (MCMC inner loop) ═══");
    let masses: Vec<f64> = vec![1e13, 3e13, 1e14, 3e14, 1e15, 3e15, 1e16, 3e16, 1e17];
    let mut out = vec![0.0; masses.len()];

    let mut ws = Workspace::new(2000);
    ws.update_cosmology(&cosmo);

    // Warm up
    sigma2_j_at_masses(&cosmo, &masses, 2, 0.0, 0.0, &ws, &mut out);

    // Benchmark
    let n_calls = 100;
    let t0 = std::time::Instant::now();
    for _ in 0..n_calls {
        sigma2_j_at_masses(&cosmo, &masses, 2, 0.0, 0.0, &ws, &mut out);
    }
    let total_us = t0.elapsed().as_micros();
    let per_call_us = total_us as f64 / n_calls as f64;
    println!("  {} masses × {} calls = {:.1} µs/call ({:.0} µs/mass)",
             masses.len(), n_calls, per_call_us, per_call_us / masses.len() as f64);

    println!("\n  {:>12} {:>6} {:>10} {:>10} {:>10} {:>8}",
             "M [Msun/h]", "R", "σ²_lin", "σ²_Zel", "σ²_J", "J/lin");
    for (&m, &s2) in masses.iter().zip(out.iter()) {
        let r = mass_to_radius(m, cosmo.omega_m);
        let s_lin = {
            let mut tmp = [0.0];
            sigma2_j_at_radii(&cosmo, &[r], 0, 0.0, 0.0, &ws, &mut tmp);
            // Zel = S(1+2S/15)^2, need to invert for S...
            // Just compute tree directly
            let ip = pt::integrals::IntegrationParams::fast();
            pt::integrals::sigma2_tree_ws(r, &ws, &ip)
        };
        let zel = s_lin * (1.0 + 2.0 * s_lin / 15.0).powi(2);
        println!("  {:12.3e} {:6.1} {:10.5} {:10.5} {:10.5} {:8.4}",
                 m, r, s_lin, zel, s2, s2 / s_lin);
    }

    // ── Rich path ────────────────────────────────────────────────────
    println!("\n═══ Rich path: dual-axis output (M exact, k_eff convenience label) ═══");
    // Demo reads s3_jacobian below → request diagnostics.
    let results = sigma2_j_plot_at_masses(
        &cosmo, &[1e13, 3e13, 1e14, 3e14, 1e15, 3e15, 1e16, 3e16, 1e17], 2, 0.0, 0.0, true);
    println!("  {:>11} {:>6} {:>7} {:>9} {:>9} {:>9} {:>9} {:>9}",
             "M [Msun/h]", "R", "k_eff", "σ²_lin", "σ²_Zel", "σ²_J", "ξ̄(R)", "S3_J");
    println!("  {:>11} {:>6} {:>7} {:>9} {:>9} {:>9} {:>9} {:>9}",
             "", "[Mpc/h]", "[h/Mpc]", "", "", "", "", "");
    for d in &results {
        println!("  {:11.3e} {:6.1} {:7.3} {:9.5} {:9.5} {:9.5} {:9.5} {:9.2e}",
                 d.mass, d.r, d.k_eff, d.sigma2_lin, d.sigma2_zel, d.sigma2_j,
                 d.xi_bar, d.s3_jacobian);
    }

    // ── Full bias programme at one scale ─────────────────────────────
    println!("\n═══ Full bias programme: all bispectrum integrals at R=20 ═══");
    let d = sigma2_j_full(&cosmo, 20.0, 2, 0.0, 0.0, true, true);
    if let Some(ref bi) = d.bispec {
        println!("  I_F2     = {:12.6e}  (gravity)", bi.i_f2);
        println!("  I_F2J    = {:12.6e}  (Jacobian)", bi.i_f2j);
        println!("  I_delta2 = {:12.6e}  (b₂ local quad. bias)", bi.i_delta2);
        println!("  I_s2     = {:12.6e}  (bs² tidal bias)", bi.i_s2);
        println!("  I_nabla  = {:12.6e}  (b_∇²δ derivative bias)", bi.i_nabla);
        println!("  I_cs2    = {:12.6e}  (cs² EFT counterterm)", bi.i_cs2);
        println!();
        println!("  S₃_matter  = 6×I_F2  = {:12.6e}", bi.s3_matter());
        println!("  S₃_Jacobian= 6×I_F2J = {:12.6e}", bi.s3_jacobian());
        println!();
        // Show that I_nabla and I_cs2 have different R-dependence
        // (this is the b_{∇²δ} vs cs² degeneracy breaking)
        println!("  Ratio I_nabla/I_F2 = {:8.4}  (b_∇²δ angular weight)", bi.i_nabla / bi.i_f2);
        println!("  Ratio I_cs2/I_F2   = {:8.4}  (cs² angular weight)", bi.i_cs2 / bi.i_f2);
        println!("  → These differ → degeneracy broken by S₃");
        println!();
        println!("  Ratio I_delta2/I_F2 = {:8.4}  (b₂ isotropic)", bi.i_delta2 / bi.i_f2);
        println!("  Ratio I_s2/I_F2     = {:8.4}  (bs² quadrupolar)", bi.i_s2 / bi.i_f2);
        println!("  → These differ → b₂ vs bs² degeneracy broken");
        println!();
        println!("  Time: {:.1} ms (including all 6 triple-W integrals)", d.elapsed_ns as f64 / 1e6);
    }

    // ── Comparison with DISCO-DJ ─────────────────────────────────────
    println!("\n═══ Validation vs DISCO-DJ 5LPT ═══");
    // DISCO-DJ measured: (R, sigma^2_J at 1LPT, 2LPT, 3LPT)
    let disco = vec![
        (10.0, 0.47226, 0.55190, 0.40163, 0.48481),
        (15.0, 0.25255, 0.27617, 0.22125, 0.25049),
        (20.0, 0.15425, 0.16338, 0.13855, 0.15163),
        (25.0, 0.10227, 0.10646, 0.09358, 0.10039),
        (30.0, 0.07161, 0.07376, 0.06642, 0.07032),
        (50.0, 0.02362, 0.02387, 0.02258, 0.02330),
        (100.0,0.00409, 0.00408, 0.00399, 0.00404),
    ];
    println!("  {:>5} {:>9} {:>9} {:>9}   {:>9} {:>9} {:>7}",
             "R", "DJ_1LPT", "DJ_3LPT", "crate_3L",
             "DJ_Zel/S", "cr_Zel/S", "err%");
    for &(r, s, j1, _j2, j3) in &disco {
        let d = sigma2_j_detailed(&cosmo, r, 2, 0.0, 0.0);
        // Our sigma2_lin should match DISCO-DJ's S
        let err = if j3 > 1e-10 { (d.sigma2_j / j3 - 1.0) * 100.0 } else { 0.0 };
        println!("  {:5.0} {:9.5} {:9.5} {:9.5}   {:9.4} {:9.4} {:+7.2}",
                 r, j1, j3, d.sigma2_j,
                 j1/s, d.sigma2_zel/d.sigma2_lin, err);
    }

    // ── RSD demo ─────────────────────────────────────────────────────
    println!("\n═══ Redshift-space distortions (z=0, f≈0.53) ═══");
    let rsd = pt::integrals::RsdParams::from_cosmology(cosmo.omega_m, 0.0);
    println!("  f = {:.4}", rsd.f);
    println!("  Kaiser σ² factor = {:.4}", pt::integrals::kaiser_sigma2(rsd.f));
    println!("  Kaiser ξ̄ factor  = {:.4}", pt::integrals::kaiser_xibar(rsd.f));

    let mut out_rsd = vec![0.0; masses.len()];
    sigma2_j_at_masses_rsd(&cosmo, &masses, 2, 0.0, 0.0, &rsd, &ws, &mut out_rsd);

    println!("\n  {:>12} {:>10} {:>10} {:>8}",
             "M [Msun/h]", "σ²_J real", "σ²_J rsd", "ratio");
    for (i, &m) in masses.iter().enumerate() {
        let ratio = out_rsd[i] / out[i];
        println!("  {:12.3e} {:10.5} {:10.5} {:8.4}", m, out[i], out_rsd[i], ratio);
    }
}

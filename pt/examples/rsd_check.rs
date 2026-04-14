use pt::cosmology::Cosmology;
use pt::integrals::*;
use pt::Workspace;

fn main() {
    let cosmo = Cosmology::planck2018();
    let mut ws = Workspace::new(2000);
    ws.update_cosmology(&cosmo);
    let ip = IntegrationParams::fast();
    let rsd = RsdParams::from_cosmology(cosmo.omega_m, 0.0);

    println!("Bias degeneracy breaking: real space vs redshift space");
    println!("f = {:.4}\n", rsd.f);
    
    for &r in &[15.0, 20.0, 30.0, 50.0] {
        // Real space
        let bi_real = s3_bias_integrals(r, &ws, &ip);
        // Redshift space
        let bi_rsd = s3_bias_integrals_rsd(r, &rsd, &ws, &ip);
        
        println!("R = {} Mpc/h:", r);
        println!("  REAL SPACE:");
        println!("    I_nabla/I_F2 = {:.5}    I_cs2/I_F2 = {:.5}    ratio = {:.3}", 
                 bi_real.i_nabla/bi_real.i_f2, bi_real.i_cs2/bi_real.i_f2,
                 bi_real.i_cs2/bi_real.i_nabla);
        println!("    I_delta2/I_F2 = {:.5}   I_s2/I_F2 = {:.5}    ratio = {:.3}",
                 bi_real.i_delta2/bi_real.i_f2, bi_real.i_s2/bi_real.i_f2,
                 bi_real.i_delta2/bi_real.i_s2);
        println!("  REDSHIFT SPACE:");
        println!("    I_nabla/I_F2 = {:.5}    I_cs2/I_F2 = {:.5}    ratio = {:.3}",
                 bi_rsd.i_nabla/bi_rsd.i_f2, bi_rsd.i_cs2/bi_rsd.i_f2,
                 bi_rsd.i_cs2/bi_rsd.i_nabla);
        println!("    I_delta2/I_F2 = {:.5}   I_s2/I_F2 = {:.5}    ratio = {:.3}",
                 bi_rsd.i_delta2/bi_rsd.i_f2, bi_rsd.i_s2/bi_rsd.i_f2,
                 bi_rsd.i_delta2/bi_rsd.i_s2);
        println!("  DEGENERACY CHECK:");
        let nabla_cs_distinct = (bi_rsd.i_cs2/bi_rsd.i_nabla - 1.0).abs() > 0.01;
        let b2_s2_distinct = (bi_rsd.i_delta2/bi_rsd.i_s2).abs() > 2.0;
        println!("    b_nabla vs cs2: {} (ratio = {:.3})",
                 if nabla_cs_distinct { "BROKEN ✓" } else { "degenerate ✗" },
                 bi_rsd.i_cs2/bi_rsd.i_nabla);
        println!("    b2 vs bs2:      {} (ratio = {:.1})",
                 if b2_s2_distinct { "BROKEN ✓" } else { "degenerate ✗" },
                 bi_rsd.i_delta2/bi_rsd.i_s2);
        println!();
    }
}

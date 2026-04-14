//! Diagnostic: print I_{b2}(k) and I_{bs2}(k) at several k.
use pt::{Cosmology, Workspace};
use pt::integrals::{IntegrationParams, p22_cross_b2_inner_kernel, p22_cross_bs2_inner_kernel};

fn main() {
    let cosmo = Cosmology::planck2018();
    let mut ws = Workspace::new(4000);
    ws.update_cosmology(&cosmo);
    let ip = IntegrationParams {
        n_k: 0, n_p: 300, n_mu: 48,
        ln_k_min: (1e-5_f64).ln(), ln_k_max: (50.0_f64).ln(),
    };
    println!("{:>10}  {:>12}  {:>14}  {:>14}", "k", "P_L(k)", "I_b2(k)", "I_bs2(k)");
    for &k in &[0.001_f64, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0] {
        let plk = cosmo.p_lin(k);
        let i_b2 = p22_cross_b2_inner_kernel(k, &ws, &ip);
        let i_bs2 = p22_cross_bs2_inner_kernel(k, &ws, &ip);
        println!("{:>10.4}  {:>12.3e}  {:>14.5e}  {:>14.5e}", k, plk, i_b2, i_bs2);
    }
}

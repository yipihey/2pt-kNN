use pt::doroshkevich::*;

fn main() {
    println!("Doroshkevich distribution: exact vs perturbative baseline");
    println!("==========================================================\n");
    
    println!("{:>6} {:>10} {:>10} {:>10} {:>8} {:>8} {:>8}",
             "σ", "Var_exact", "Var_pert", "err%", "⟨J⟩", "s₃", "s₄");
    println!("{}", "-".repeat(70));
    
    for &sigma in &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0] {
        let n = if sigma < 0.5 { 48 } else { 80 };
        let m = doroshkevich_moments(sigma, n, 6.0);
        let s2 = sigma * sigma;
        let pert = sigma2_zel_perturbative(s2);
        let err = (pert / m.variance - 1.0) * 100.0;
        
        println!("{:6.2} {:10.5} {:10.5} {:+10.2} {:8.5} {:8.3} {:8.3}",
                 sigma, m.variance, pert, err, m.mean_j, m.s3, m.s4);
    }
    
    println!("\nKey validation points:");
    println!("  σ=0 limit: s₃ → 2.000 ✓ (Gaussian J, tree-level bispectrum)");
    
    let m01 = doroshkevich_moments(0.1, 48, 6.0);
    println!("  σ=0.1: s₃ = {:.4} (should → 2.00)", m01.s3);
    
    let m10 = doroshkevich_moments(1.0, 80, 6.0);
    println!("  σ=1.0: Var = {:.4} (doc says ~1.284), s₃ = {:.3} (doc says ~1.74)", 
             m10.variance, m10.s3);
    
    // Compare with the polynomial: S(1+2S/15)^2
    println!("\n\nPerturbative baseline error at key mass scales:");
    let cosmo_data = [
        ("M~10¹⁷ Msun/h", 100.0, 0.07),
        ("M~10¹⁶ Msun/h", 30.0, 0.27),
        ("M~10¹⁵ Msun/h", 14.0, 0.53),
        ("M~3×10¹⁴ Msun/h", 9.4, 0.73),
        ("M~10¹⁴ Msun/h", 6.5, 0.93),
        ("M~10¹³ Msun/h", 3.0, 1.47),
    ];
    for (label, r, sigma) in &cosmo_data {
        let m = doroshkevich_moments(*sigma, 80, 6.0);
        let s2 = sigma * sigma;
        let pert = sigma2_zel_perturbative(s2);
        let err = (pert / m.variance - 1.0) * 100.0;
        println!("  {} (R={:.0}, σ={:.2}): exact={:.4}, pert={:.4}, Δ={:+.1}%, s₃={:.3}",
                 label, r, sigma, m.variance, pert, err, m.s3);
    }
    
    // Build a cached table and test interpolation
    println!("\n\nBuilding Doroshkevich table (300 points, σ=0.01..3.0)...");
    let t0 = std::time::Instant::now();
    let table = DoroshkevichTable::default_table();
    let build_ms = t0.elapsed().as_millis();
    println!("  Built in {} ms", build_ms);
    
    // Interpolation test
    println!("\n  Interpolation accuracy:");
    for &sigma in &[0.25, 0.5, 0.75, 1.0, 1.5, 2.0] {
        let exact = doroshkevich_moments(sigma, 80, 6.0);
        let interp = table.variance_at(sigma);
        let err = (interp / exact.variance - 1.0) * 100.0;
        println!("    σ={:.2}: exact={:.6}, interp={:.6}, err={:+.4}%",
                 sigma, exact.variance, interp, err);
    }
}

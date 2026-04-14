use pt::doroshkevich::*;

fn main() {
    // Test at σ=1 where MC gives Var=1.2847, s3=1.74
    let sigma = 1.0;
    
    // Test with increasing quadrature points
    for &n in &[20, 40, 60, 80, 100] {
        let m = doroshkevich_moments(sigma, n, 6.0);
        println!("n={:3}: <J>={:.6} Var={:.6} s3={:.4}", n, m.mean_j, m.variance, m.s3);
    }
    
    // Test with different l_range
    println!();
    for &l in &[4.0, 6.0, 8.0, 10.0] {
        let m = doroshkevich_moments(sigma, 80, l);
        println!("L={:.0}: <J>={:.6} Var={:.6} s3={:.4}", l, m.mean_j, m.variance, m.s3);
    }
    
    // Test at small sigma where perturbative should be very close
    println!();
    let sigma = 0.1;
    let m = doroshkevich_moments(sigma, 80, 6.0);
    let pert = sigma * sigma * (1.0 + 2.0 * sigma * sigma / 15.0_f64).powi(2);
    println!("σ=0.1: <J>={:.6}, Var_exact={:.6}, Var_pert={:.6}, ratio={:.4}",
             m.mean_j, m.variance, pert, m.variance / pert);
    println!("  (should be: <J>=1.0000, Var≈0.01, ratio≈1.00)");
    
    // The problem might be that the integral is over ORDERED eigenvalues
    // but the PDF needs to account for the 3! permutations.
    // Let's check: with the gap variables, we integrate over the ordered region
    // (λ1 ≥ λ2 ≥ λ3) with the Vandermonde = g1*g2*(g1+g2).
    // The full integral over ALL orderings gives 6 × (ordered integral).
    // But the Vandermonde already accounts for the ordering — it's the
    // absolute value of the determinant, which is positive in the ordered region.
    // So we should NOT multiply by 6.
    // 
    // But maybe the normalisation N in the Doroshkevich PDF already includes
    // the factor of 6? Let me check by computing s0 (the normalisation integral)
    // and seeing if it gives 1/6 or 1.
    
    let sigma = 0.3;
    let m = doroshkevich_moments(sigma, 80, 6.0);
    println!("\nσ=0.3: <J>={:.6}, Var={:.6}", m.mean_j, m.variance);
    println!("  Perturbative: {:.6}", 0.09 * (1.0 + 2.0*0.09/15.0_f64).powi(2));
}

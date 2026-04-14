use pt::doroshkevich::doroshkevich_moments;

fn main() {
    // Test 1: the Gauss-Legendre implementation via a known integral
    // ∫₋₁¹ x² dx = 2/3
    // ∫₋₁¹ exp(-x²) dx = √π erf(1) = 1.4936...
    
    // I can't directly access gauss_legendre from here (it's private).
    // Instead, test doroshkevich_moments at σ → 0 where everything should be perturbative.
    
    // At very small σ: J ≈ 1 - (λ1+λ2+λ3) = 1 - I1
    // <J> = 1, Var(J) = Var(I1) = 3σ²/5 + 6σ²/15 = σ² (in our normalisation)
    // Wait: Var(I1) = Var(ψ_11+ψ_22+ψ_33) 
    //     = 3×σ²/5 + 6×σ²/15 (three diagonal variances + six cross terms × covariance)
    //     = 3σ²/5 + 6σ²/15 = 3σ²/5 + 2σ²/5 = σ²
    // So at σ→0: Var(J) → σ². The perturbative formula also gives σ² at leading order. ✓
    
    // At σ=0.01: 
    let sigma = 0.01;
    let m = doroshkevich_moments(sigma, 48, 6.0);
    println!("σ=0.01: <J>={:.8}, Var={:.8e}, Var/σ²={:.6}", 
             m.mean_j, m.variance, m.variance / (sigma*sigma));
    println!("  (should be: <J>=1.000, Var/σ²=1.000)");
    
    // Something is clearly wrong with the normalisation if <J> ≠ 1 even at tiny σ.
    // The Vandermonde factor might need to be squared (GUE vs GOE).
    // For the GOE (real symmetric matrices), the eigenvalue PDF has |Δ|^1.
    // For the GUE, it's |Δ|^2.
    // Our matrix is REAL and SYMMETRIC, so GOE: |Δ|^1. 
    // But the standard Doroshkevich distribution uses |Δ|^1 times the Gaussian.
    // The issue might be that the PDF written in the document uses an UNNORMALIZED
    // form, and the normalisation s0 handles it. Let me check if s0 is reasonable.
    
    // Actually: the fact that <J> varies with n_gauss and l_range suggests
    // the integral ISN'T converging. Either the GL is wrong or the range is wrong.
    
    // Let me try a much smaller sigma where everything is Gaussian
    let sigma = 0.001;
    for &n in &[20, 40, 80] {
        let m = doroshkevich_moments(sigma, n, 10.0);
        println!("σ={}, n={}: <J>={:.8}, Var={:.4e}, Var/σ²={:.4}", 
                 sigma, n, m.mean_j, m.variance, m.variance / (sigma*sigma));
    }
}

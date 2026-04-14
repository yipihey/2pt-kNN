fn main() {
    // Implement GL directly here for testing
    let n = 10;
    let (nodes, weights) = gauss_legendre(n);
    
    // Test: ∫₋₁¹ 1 dx = 2
    let sum: f64 = weights.iter().sum();
    println!("∫₋₁¹ 1 dx = {} (should be 2.0)", sum);
    
    // Test: ∫₋₁¹ x² dx = 2/3
    let sum2: f64 = nodes.iter().zip(weights.iter()).map(|(x,w)| x*x*w).sum();
    println!("∫₋₁¹ x² dx = {} (should be 0.6667)", sum2);
    
    // Test: ∫₋₁¹ x⁴ dx = 2/5
    let sum4: f64 = nodes.iter().zip(weights.iter()).map(|(x,w)| x.powi(4)*w).sum();
    println!("∫₋₁¹ x⁴ dx = {} (should be 0.4000)", sum4);
    
    // Test: ∫₀^∞ exp(-x²) dx = √π/2 via change of var
    // ∫₋₁¹ exp(-(a(1+t)/2)²) × a/2 dt for a = 5 (range [0, 5])
    let a = 5.0;
    let n = 40;
    let (nodes, weights) = gauss_legendre(n);
    let val: f64 = nodes.iter().zip(weights.iter()).map(|(t, w)| {
        let x = a * (1.0 + t) / 2.0;
        (-x*x).exp() * a / 2.0 * w
    }).sum();
    println!("∫₀⁵ exp(-x²) dx = {} (should be {:.6})", val, std::f64::consts::PI.sqrt()/2.0);
    
    // Print first 5 nodes and weights for n=10
    let (n10, w10) = gauss_legendre(10);
    println!("\nGL(10) nodes: {:?}", &n10[..5]);
    println!("GL(10) weights: {:?}", &w10[..5]);
}

fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];
    let m = (n + 1) / 2;
    for i in 0..m {
        let mut z = ((i as f64 + 0.75) / (n as f64 + 0.5) * std::f64::consts::PI).cos();
        loop {
            let mut p0 = 1.0;
            let mut p1 = z;
            for j in 2..=n {
                let p2 = ((2 * j - 1) as f64 * z * p1 - (j - 1) as f64 * p0) / j as f64;
                p0 = p1;
                p1 = p2;
            }
            let dp = n as f64 * (z * p1 - p0) / (z * z - 1.0);
            let dz = p1 / dp;
            z -= dz;
            if dz.abs() < 1e-15 { break; }
        }
        // Recompute P_n and P_{n-1} at converged z
        let mut p0 = 1.0;
        let mut p1 = z;
        for j in 2..=n {
            let p2 = ((2 * j - 1) as f64 * z * p1 - (j - 1) as f64 * p0) / j as f64;
            p0 = p1;
            p1 = p2;
        }
        let dp = n as f64 * (z * p1 - p0) / (z * z - 1.0);
        let w = 2.0 / ((1.0 - z * z) * dp * dp);
        nodes[i] = -z;
        nodes[n - 1 - i] = z;
        weights[i] = w;
        weights[n - 1 - i] = w;
    }
    (nodes, weights)
}

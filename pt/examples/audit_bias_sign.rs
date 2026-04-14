//! Audit: does doroshkevich_cdf_biased and xibar_biased use consistent bias sign?
//! Correct convention: b₁ > 0 means tracer in overdensity ⇒ J < 1 ⇒ CDF shifts LEFT.

use pt::doroshkevich::{doroshkevich_cdf, doroshkevich_cdf_biased, doroshkevich_xibar_biased, doroshkevich_moments};

fn main() {
    let sigma = 0.5;

    // Test 1: Does xibar_biased give negative value for b1 > 0?
    // Physical expectation: biased tracer in overdensity → J < 1 → ⟨J-1⟩ < 0.
    let xb_pos = doroshkevich_xibar_biased(sigma, 1.0, 80, 6.0);
    let xb_zero = doroshkevich_xibar_biased(sigma, 0.0, 80, 6.0);
    println!("Test 1: xibar_biased");
    println!("  b1=0.0: <J-1> = {:.6e} (should be ~0)", xb_zero);
    println!("  b1=1.0: <J-1> = {:.6e} (should be NEGATIVE for overdense tracer)", xb_pos);
    println!("  ⇒ {}", if xb_pos < 0.0 && xb_zero.abs() < 1e-10 { "PASS" } else { "FAIL" });
    println!();

    // Test 2: Does cdf_biased shift LEFT for b1 > 0?
    // CDF(J < 1 | biased, b1>0) should be GREATER than CDF(J < 1 | unbiased).
    let j_thresh = vec![0.3, 0.7, 1.0, 1.3, 1.7];
    let cdf_0 = doroshkevich_cdf(sigma, &j_thresh, 80, 6.0);
    let cdf_pos = doroshkevich_cdf_biased(sigma, 1.0, &j_thresh, 80, 6.0);
    println!("Test 2: cdf_biased direction");
    println!("  J      P(J<j | b1=0)    P(J<j | b1=1)    Δ(b1=1 - b1=0)");
    for i in 0..j_thresh.len() {
        let d = cdf_pos[i] - cdf_0[i];
        println!("  {:5.2}  {:>12.6}   {:>12.6}   {:>+9.6}",
                 j_thresh[i], cdf_0[i], cdf_pos[i], d);
    }
    let cdf_at_1_shift = cdf_pos[2] - cdf_0[2];
    println!("  Δ at J=1: {:+.6}", cdf_at_1_shift);
    println!("  Expected: POSITIVE (more probability at J < 1 for overdense tracers)");
    println!("  ⇒ {}", if cdf_at_1_shift > 0.0 { "PASS (sign agrees with xibar_biased)" } else { "FAIL (sign disagrees — BUG)" });
    println!();

    // Test 3: Sanity check: <J>=1 for unbiased
    let m = doroshkevich_moments(sigma, 80, 6.0);
    println!("Test 3: Mass conservation ⟨J⟩ = 1 (b1=0)");
    println!("  ⟨J⟩ = {:.10}", m.mean_j);
    println!("  ⇒ {}", if (m.mean_j - 1.0).abs() < 1e-6 { "PASS" } else { "FAIL" });
}

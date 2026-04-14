//! Exact Doroshkevich distribution: non-perturbative baseline for all J cumulants.
//!
//! The Zel'dovich Jacobian J = det(I + ψ_{i,j}) has eigenvalues (1-λ₁)(1-λ₂)(1-λ₃)
//! where λ_a are the eigenvalues of the Gaussian deformation tensor ψ_{i,j}.
//! The Doroshkevich distribution p(λ₁,λ₂,λ₃) is the exact PDF of these eigenvalues,
//! from which all one-point cumulants of J are computed by numerical quadrature.
//!
//! This is the EXACT Zel'dovich baseline — no perturbative truncation.
//! Valid at all σ, including σ > 1 where the polynomial S(1+2S/15)² fails.

/// Cumulants of the Zel'dovich Jacobian from the Doroshkevich distribution.
#[derive(Clone, Debug)]
pub struct DoroshkevichMoments {
    /// σ = sqrt(σ²_lin) used for the computation.
    pub sigma: f64,
    /// ⟨J⟩ — should be 1.000 (mass conservation).
    pub mean_j: f64,
    /// Var(J) = ⟨(J-1)²⟩.
    pub variance: f64,
    /// μ₃ = ⟨(J-1)³⟩ (third central moment).
    pub mu3: f64,
    /// μ₄ = ⟨(J-1)⁴⟩ - 3 Var(J)² (excess kurtosis × σ⁴).
    pub mu4: f64,
    /// Reduced skewness s₃ = μ₃ / Var(J)².
    pub s3: f64,
    /// Reduced kurtosis s₄ = μ₄ / Var(J)³.
    pub s4: f64,
}

/// Compute exact Doroshkevich moments at a given σ.
///
/// Uses Gauss-Legendre quadrature with change of variables:
///   λ₃ = e₃ (smallest eigenvalue)
///   λ₂ = e₃ + g₂ (gap ≥ 0)
///   λ₁ = e₃ + g₂ + g₁ (gap ≥ 0)
///
/// The Vandermonde factor g₁·g₂·(g₁+g₂) is manifestly positive.
///
/// # Parameters
/// * `sigma` — sqrt(σ²_lin(R))
/// * `n_gauss` — quadrature points per dimension (40 for σ<0.5, 80 for high accuracy)
/// * `l_range` — integration range in units of σ (6.0 captures tails to 10⁻⁸)
pub fn doroshkevich_moments(sigma: f64, n_gauss: usize, l_range: f64) -> DoroshkevichMoments {
    if sigma <= 0.0 {
        return DoroshkevichMoments {
            sigma: 0.0, mean_j: 1.0, variance: 0.0,
            mu3: 0.0, mu4: 0.0, s3: 0.0, s4: 0.0,
        };
    }

    let (nodes, weights) = gauss_legendre(n_gauss);
    let sig2 = sigma * sigma;

    // Map nodes to physical ranges
    let e3_lo = -l_range * sigma;
    let e3_hi = l_range * sigma;
    let e3_scale = (e3_hi - e3_lo) / 2.0;
    let e3_mid = (e3_hi + e3_lo) / 2.0;

    let gap_hi = 2.0 * l_range * sigma;
    let gap_scale = gap_hi / 2.0;  // map [-1,1] → [0, gap_hi]
    let gap_mid = gap_hi / 2.0;

    let mut s0 = 0.0_f64; // normalisation
    let mut s1 = 0.0_f64; // ⟨J⟩
    let mut s2 = 0.0_f64; // ⟨J²⟩
    let mut s3 = 0.0_f64; // ⟨J³⟩
    let mut s4 = 0.0_f64; // ⟨J⁴⟩

    for i3 in 0..n_gauss {
        let e3 = e3_mid + e3_scale * nodes[i3];
        let w3 = e3_scale * weights[i3];

        for i2 in 0..n_gauss {
            let g2 = gap_mid + gap_scale * nodes[i2];
            if g2 <= 0.0 { continue; }
            let wg2 = gap_scale * weights[i2];
            let e2 = e3 + g2;

            for i1 in 0..n_gauss {
                let g1 = gap_mid + gap_scale * nodes[i1];
                if g1 <= 0.0 { continue; }
                let wg1 = gap_scale * weights[i1];
                let e1 = e2 + g1;

                // Invariants
                let i1_inv = e1 + e2 + e3;
                let i2_inv = e1 * e2 + e1 * e3 + e2 * e3;

                // CORRECTED exponent: -3 I₁²/σ² + (15/2) I₂/σ²
                let exp_arg = -3.0 * (i1_inv * i1_inv) / sig2
                            + 7.5 * i2_inv / sig2;
                if exp_arg < -200.0 { continue; }

                let vandermonde = g1 * g2 * (g1 + g2);
                let f = exp_arg.exp() * vandermonde;
                let w = w3 * wg2 * wg1;
                let fw = f * w;

                let j = (1.0 - e1) * (1.0 - e2) * (1.0 - e3);

                s0 += fw;
                s1 += fw * j;
                let j2 = j * j;
                s2 += fw * j2;
                s3 += fw * j2 * j;
                s4 += fw * j2 * j2;
            }
        }
    }

    if s0 <= 0.0 {
        return DoroshkevichMoments {
            sigma, mean_j: 1.0, variance: sig2,
            mu3: 0.0, mu4: 0.0, s3: 0.0, s4: 0.0,
        };
    }

    let mean_j = s1 / s0;
    let m2 = s2 / s0;
    let m3 = s3 / s0;
    let m4 = s4 / s0;

    // Central moments
    let var = m2 - mean_j * mean_j;
    let mu3_val = m3 - 3.0 * m2 * mean_j + 2.0 * mean_j.powi(3);
    let mu4_val = m4 - 4.0 * m3 * mean_j + 6.0 * m2 * mean_j * mean_j
                - 3.0 * mean_j.powi(4);
    let excess_kurtosis = mu4_val - 3.0 * var * var;

    let s3_red = if var > 1e-30 { mu3_val / (var * var) } else { 0.0 };
    let s4_red = if var > 1e-30 { excess_kurtosis / (var * var * var) } else { 0.0 };

    DoroshkevichMoments {
        sigma, mean_j, variance: var,
        mu3: mu3_val, mu4: excess_kurtosis,
        s3: s3_red, s4: s4_red,
    }
}

/// Compute ξ̄_{J,Zel}(σ, b₁) — the exact Zel'dovich conditional mean of (J-1)
/// around a tracer with linear bias b₁.
///
/// Uses the same Gauss-Legendre quadrature as `doroshkevich_moments`, but with:
/// - Integrand: (J - 1) instead of (J - 1)²
/// - Weight tilted by exp(6 b₁ Δ I₁ / σ²) where Δ = -b₁ σ²
/// - Normalised by the tilted partition function
///
/// Validation:
/// - At b₁ = 0: returns 0 (mass conservation ⟨J⟩ = 1)
/// - At small b₁: returns ≈ -b₁ σ² (tree-level result)
pub fn doroshkevich_xibar_biased(
    sigma: f64,
    b1: f64,
    n_gauss: usize,
    l_range: f64,
) -> f64 {
    if sigma <= 0.0 {
        return 0.0;
    }

    let (nodes, weights) = gauss_legendre(n_gauss);
    let sig2 = sigma * sigma;
    // Bias conditioning: biased tracers have ⟨δ_L⟩ = b₁σ², and in code
    // convention I₁ = δ_L, so ⟨I₁⟩ = +b₁σ². The Gaussian tilt is exp(+6b₁I₁).
    let delta_shift = b1 * sig2;

    let e3_lo = -l_range * sigma;
    let e3_hi = l_range * sigma;
    let e3_scale = (e3_hi - e3_lo) / 2.0;
    let e3_mid = (e3_hi + e3_lo) / 2.0;

    let gap_hi = 2.0 * l_range * sigma;
    let gap_scale = gap_hi / 2.0;
    let gap_mid = gap_hi / 2.0;

    let mut numerator = 0.0_f64;  // ∫ (J-1) × tilted_p_D
    let mut denominator = 0.0_f64; // ∫ tilted_p_D (normalisation)

    for i3 in 0..n_gauss {
        let e3 = e3_mid + e3_scale * nodes[i3];
        let w3 = e3_scale * weights[i3];

        for i2 in 0..n_gauss {
            let g2 = gap_mid + gap_scale * nodes[i2];
            if g2 <= 0.0 { continue; }
            let wg2 = gap_scale * weights[i2];
            let e2 = e3 + g2;

            for i1 in 0..n_gauss {
                let g1 = gap_mid + gap_scale * nodes[i1];
                if g1 <= 0.0 { continue; }
                let wg1 = gap_scale * weights[i1];
                let e1 = e2 + g1;

                // Invariants
                let i1_inv = e1 + e2 + e3;
                let i2_inv = e1 * e2 + e1 * e3 + e2 * e3;

                // Doroshkevich exponent + bias tilt
                let mut exp_arg = -3.0 * (i1_inv * i1_inv) / sig2
                                + 7.5 * i2_inv / sig2;
                if delta_shift != 0.0 {
                    exp_arg += 6.0 * i1_inv * delta_shift / sig2;
                }

                if exp_arg < -200.0 { continue; }

                let vandermonde = g1 * g2 * (g1 + g2);
                let fw = exp_arg.exp() * vandermonde * w3 * wg2 * wg1;

                let j = (1.0 - e1) * (1.0 - e2) * (1.0 - e3);

                denominator += fw;
                numerator += fw * (j - 1.0);
            }
        }
    }

    if denominator <= 0.0 {
        return 0.0;
    }

    numerator / denominator
}

/// Pre-computed table of Doroshkevich moments for fast interpolation.
pub struct DoroshkevichTable {
    sigma_values: Vec<f64>,
    variance: Vec<f64>,
    mu3: Vec<f64>,
    mu4: Vec<f64>,
    s3: Vec<f64>,
    n: usize,
}

impl DoroshkevichTable {
    /// Build the table for σ ∈ [sigma_min, sigma_max] with n_points entries.
    /// Uses n_gauss=80 quadrature for high accuracy.
    pub fn new(sigma_min: f64, sigma_max: f64, n_points: usize) -> Self {
        let mut sigma_values = Vec::with_capacity(n_points);
        let mut variance = Vec::with_capacity(n_points);
        let mut mu3 = Vec::with_capacity(n_points);
        let mut mu4 = Vec::with_capacity(n_points);
        let mut s3 = Vec::with_capacity(n_points);

        for i in 0..n_points {
            let f = i as f64 / (n_points - 1) as f64;
            let sig = sigma_min + (sigma_max - sigma_min) * f;
            let n_gl = if sig < 0.5 { 48 } else { 80 };
            let m = doroshkevich_moments(sig, n_gl, 6.0);

            sigma_values.push(sig);
            variance.push(m.variance);
            mu3.push(m.mu3);
            mu4.push(m.mu4);
            s3.push(m.s3);
        }

        DoroshkevichTable { sigma_values, variance, mu3, mu4, s3, n: n_points }
    }

    /// Default table: σ ∈ [0.01, 3.0], 300 points.
    pub fn default_table() -> Self {
        Self::new(0.01, 3.0, 300)
    }

    /// Interpolate variance at given σ using linear interpolation in σ.
    pub fn variance_at(&self, sigma: f64) -> f64 {
        self.interp(sigma, &self.variance)
    }

    /// Interpolate μ₃ at given σ.
    pub fn mu3_at(&self, sigma: f64) -> f64 {
        self.interp(sigma, &self.mu3)
    }

    /// Interpolate reduced skewness s₃ at given σ.
    pub fn s3_at(&self, sigma: f64) -> f64 {
        self.interp(sigma, &self.s3)
    }

    fn interp(&self, sigma: f64, values: &[f64]) -> f64 {
        if sigma <= self.sigma_values[0] {
            return values[0];
        }
        if sigma >= self.sigma_values[self.n - 1] {
            return values[self.n - 1];
        }
        let f = (sigma - self.sigma_values[0])
              / (self.sigma_values[self.n - 1] - self.sigma_values[0])
              * (self.n - 1) as f64;
        let i = f as usize;
        if i >= self.n - 1 { return values[self.n - 1]; }
        let t = f - i as f64;
        values[i] * (1.0 - t) + values[i + 1] * t
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CDF and PDF of the Zel'dovich Jacobian
// ═══════════════════════════════════════════════════════════════════════════

/// CDF of the Zel'dovich Jacobian: P(J < j₀ | σ) for each threshold in `j_thresholds`.
///
/// Uses the same Gauss-Legendre quadrature as the moments computation.
/// The CDF is computed by accumulating the weight where J(λ) < j₀ in the
/// three-eigenvalue integral.
///
/// # Parameters
/// * `sigma` — sqrt(σ²_lin(R))
/// * `j_thresholds` — sorted ascending thresholds at which to evaluate the CDF
/// * `n_gauss` — quadrature points per dimension (40-80)
/// * `l_range` — integration range in units of σ
pub fn doroshkevich_cdf(
    sigma: f64,
    j_thresholds: &[f64],
    n_gauss: usize,
    l_range: f64,
) -> Vec<f64> {
    doroshkevich_cdf_biased(sigma, 0.0, j_thresholds, n_gauss, l_range)
}

/// CDF of the Zel'dovich Jacobian conditioned on linear bias b₁.
///
/// For a tracer with linear bias b₁, the conditional distribution shifts the
/// mean trace of the eigenvalue sum. In the code convention I₁_code = δ_L, so
/// biased tracers have ⟨δ_L⟩ = b₁ σ², hence ⟨I₁⟩ = +b₁ σ². The Gaussian tilt
/// is therefore exp(+6 b₁ I₁), equivalent to:
///
///   exp_arg += 6 I₁ Δ / σ²   with Δ = +b₁ σ²
///
/// For b₁ > 0, this enhances overdense configurations (I₁ > 0), which have
/// compressed volumes (J < 1), correctly shifting the CDF leftward.
pub fn doroshkevich_cdf_biased(
    sigma: f64,
    b1: f64,
    j_thresholds: &[f64],
    n_gauss: usize,
    l_range: f64,
) -> Vec<f64> {
    let n_thresh = j_thresholds.len();
    if sigma <= 0.0 || n_thresh == 0 {
        // At σ=0, J=1 deterministically
        return j_thresholds.iter().map(|&j0| if 1.0 < j0 { 1.0 } else { 0.0 }).collect();
    }

    let (nodes, weights) = gauss_legendre(n_gauss);
    let sig2 = sigma * sigma;
    // Correct sign: ⟨I₁⟩ = +b₁σ² for overdense tracers (I₁_code = δ_L).
    let delta_shift = b1 * sig2;

    let e3_lo = -l_range * sigma;
    let e3_hi = l_range * sigma;
    let e3_scale = (e3_hi - e3_lo) / 2.0;
    let e3_mid = (e3_hi + e3_lo) / 2.0;

    let gap_hi = 2.0 * l_range * sigma;
    let gap_scale = gap_hi / 2.0;
    let gap_mid = gap_hi / 2.0;

    let mut cdf = vec![0.0; n_thresh];
    let mut norm = 0.0;

    for i3 in 0..n_gauss {
        let e3 = e3_mid + e3_scale * nodes[i3];
        let w3 = e3_scale * weights[i3];

        for i2 in 0..n_gauss {
            let g2 = gap_mid + gap_scale * nodes[i2];
            if g2 <= 0.0 { continue; }
            let wg2 = gap_scale * weights[i2];
            let e2 = e3 + g2;

            for i1 in 0..n_gauss {
                let g1 = gap_mid + gap_scale * nodes[i1];
                if g1 <= 0.0 { continue; }
                let wg1 = gap_scale * weights[i1];
                let e1 = e2 + g1;

                let i1_inv = e1 + e2 + e3;
                let i2_inv = e1 * e2 + e1 * e3 + e2 * e3;

                let mut exp_arg = -3.0 * (i1_inv * i1_inv) / sig2
                                + 7.5 * i2_inv / sig2;

                // Bias shift: adds linear term in I₁
                if delta_shift != 0.0 {
                    exp_arg += 6.0 * i1_inv * delta_shift / sig2;
                }

                if exp_arg < -200.0 { continue; }

                let vandermonde = g1 * g2 * (g1 + g2);
                let fw = exp_arg.exp() * vandermonde * w3 * wg2 * wg1;

                let j = (1.0 - e1) * (1.0 - e2) * (1.0 - e3);
                norm += fw;

                // Accumulate CDF: binary search for insertion point
                // (thresholds are sorted, so we can stop early)
                for t in 0..n_thresh {
                    if j < j_thresholds[t] {
                        // All subsequent thresholds are larger, so j < them too
                        for t2 in t..n_thresh {
                            cdf[t2] += fw;
                        }
                        break;
                    }
                }
            }
        }
    }

    if norm > 0.0 {
        for c in cdf.iter_mut() {
            *c /= norm;
        }
    }
    cdf
}

/// PDF of the Zel'dovich Jacobian evaluated at `j_eval` points.
///
/// Uses kernel density estimation with a Gaussian kernel of width `bandwidth`.
/// Set bandwidth ~ σ/10 for smooth results that resolve the peak structure.
pub fn doroshkevich_pdf(
    sigma: f64,
    j_eval: &[f64],
    bandwidth: f64,
    n_gauss: usize,
    l_range: f64,
) -> Vec<f64> {
    doroshkevich_pdf_biased(sigma, 0.0, j_eval, bandwidth, n_gauss, l_range)
}

/// Biased PDF of the Zel'dovich Jacobian (for D-kNN around tracers).
/// Bias sign convention matches doroshkevich_cdf_biased: ⟨I₁⟩ = +b₁σ²,
/// tilt exp(+6b₁I₁). For b₁ > 0, enhances J < 1 (compressed volumes).
pub fn doroshkevich_pdf_biased(
    sigma: f64,
    b1: f64,
    j_eval: &[f64],
    bandwidth: f64,
    n_gauss: usize,
    l_range: f64,
) -> Vec<f64> {
    let n_eval = j_eval.len();
    if sigma <= 0.0 || n_eval == 0 {
        return vec![0.0; n_eval];
    }

    let (nodes, weights) = gauss_legendre(n_gauss);
    let sig2 = sigma * sigma;
    let delta_shift = b1 * sig2;
    let inv_bw = 1.0 / bandwidth;
    let gauss_norm = inv_bw / (2.0 * std::f64::consts::PI).sqrt();

    let e3_lo = -l_range * sigma;
    let e3_hi = l_range * sigma;
    let e3_scale = (e3_hi - e3_lo) / 2.0;
    let e3_mid = (e3_hi + e3_lo) / 2.0;

    let gap_hi = 2.0 * l_range * sigma;
    let gap_scale = gap_hi / 2.0;
    let gap_mid = gap_hi / 2.0;

    let mut pdf = vec![0.0; n_eval];
    let mut norm = 0.0;

    for i3 in 0..n_gauss {
        let e3 = e3_mid + e3_scale * nodes[i3];
        let w3 = e3_scale * weights[i3];

        for i2 in 0..n_gauss {
            let g2 = gap_mid + gap_scale * nodes[i2];
            if g2 <= 0.0 { continue; }
            let wg2 = gap_scale * weights[i2];
            let e2 = e3 + g2;

            for i1 in 0..n_gauss {
                let g1 = gap_mid + gap_scale * nodes[i1];
                if g1 <= 0.0 { continue; }
                let wg1 = gap_scale * weights[i1];
                let e1 = e2 + g1;

                let i1_inv = e1 + e2 + e3;
                let i2_inv = e1 * e2 + e1 * e3 + e2 * e3;

                let mut exp_arg = -3.0 * (i1_inv * i1_inv) / sig2
                                + 7.5 * i2_inv / sig2;

                if delta_shift != 0.0 {
                    exp_arg += 6.0 * i1_inv * delta_shift / sig2;
                }

                if exp_arg < -200.0 { continue; }

                let vandermonde = g1 * g2 * (g1 + g2);
                let fw = exp_arg.exp() * vandermonde * w3 * wg2 * wg1;

                let j = (1.0 - e1) * (1.0 - e2) * (1.0 - e3);
                norm += fw;

                // Gaussian kernel contributions at each evaluation point
                for (t, &je) in j_eval.iter().enumerate() {
                    let z = (je - j) * inv_bw;
                    if z.abs() < 5.0 {
                        pdf[t] += fw * (-0.5 * z * z).exp() * gauss_norm;
                    }
                }
            }
        }
    }

    if norm > 0.0 {
        for p in pdf.iter_mut() {
            *p /= norm;
        }
    }
    pdf
}

/// Find σ_eff such that Doroshkevich variance equals the target variance.
///
/// Uses binary search. The Doroshkevich variance is monotonically increasing in σ.
pub fn find_sigma_eff(target_variance: f64, n_gauss: usize) -> f64 {
    if target_variance <= 0.0 {
        return 0.0;
    }
    let mut lo = 0.001_f64;
    let mut hi = 5.0_f64;

    // Ensure bracket
    while doroshkevich_moments(hi, n_gauss, 6.0).variance < target_variance {
        hi *= 2.0;
        if hi > 50.0 { return hi; }
    }

    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        let var = doroshkevich_moments(mid, n_gauss, 6.0).variance;
        if var < target_variance {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Perturbative formula for comparison: σ²_Zel,pert = S(1+2S/15)².
/// This is the Taylor expansion of the Doroshkevich integral to O(S³).
#[inline]
pub fn sigma2_zel_perturbative(sigma2_lin: f64) -> f64 {
    sigma2_lin * (1.0 + 2.0 * sigma2_lin / 15.0).powi(2)
}

// ═══════════════════════════════════════════════════════════════════════════
// Gauss-Legendre quadrature nodes and weights
// ═══════════════════════════════════════════════════════════════════════════

/// Generate Gauss-Legendre nodes and weights on [-1,1].
pub(crate) fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
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
        // Recompute P_n(z) and P_{n-1}(z) at the converged z
        let mut p0 = 1.0;
        let mut p1 = z;
        for j in 2..=n {
            let p2 = ((2 * j - 1) as f64 * z * p1 - (j - 1) as f64 * p0) / j as f64;
            p0 = p1;
            p1 = p2;
        }
        // p1 = P_n(z), p0 = P_{n-1}(z)
        let dp = n as f64 * (z * p1 - p0) / (z * z - 1.0);
        let w = 2.0 / ((1.0 - z * z) * dp * dp);
        nodes[i] = -z;
        nodes[n - 1 - i] = z;
        weights[i] = w;
        weights[n - 1 - i] = w;
    }
    (nodes, weights)
}

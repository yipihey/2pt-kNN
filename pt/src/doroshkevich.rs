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
// Tilted Doroshkevich: incorporating PT corrections via exponential tilting
//
// We deform the Doroshkevich distribution by multiplying by exp(α I₁ + β I₂):
//
//     p_tilted(λ) ∝ p_D(λ) × exp(α I₁ + β I₂)
//
// where I₁ = λ₁ + λ₂ + λ₃ and I₂ = λ₁λ₂ + λ₁λ₃ + λ₂λ₃. The tilt parameters
// are chosen by 2D root-finding to match target moments (σ²_J and s₃) that
// include the perturbative corrections. Additionally, a linear bias b₁ for
// tracer-centred distributions contributes +6 b₁ I₁ (absorbed into α as
// α_biased = α + 6 b₁).
//
// For the mass-weighted PDF, the Doroshkevich weight is divided by J at each
// eigenvalue triple (volume elements weight each mass differently).
// ═══════════════════════════════════════════════════════════════════════════

/// Result of a single-pass tilted-Doroshkevich quadrature: moments,
/// volume-weighted PDF, and mass-weighted PDF on a user-specified J grid.
#[derive(Clone, Debug)]
pub struct TiltedPass {
    /// σ used (Doroshkevich scale).
    pub sigma: f64,
    /// α tilt parameter used.
    pub alpha: f64,
    /// β tilt parameter used.
    pub beta: f64,
    /// Linear bias used (enters as α_effective = α + 6 b₁).
    pub b1: f64,
    /// ⟨J⟩ under the tilted (and optionally mass-weighted) distribution.
    pub mean_j: f64,
    /// Variance Var(J) under the tilted volume-weighted distribution.
    pub variance: f64,
    /// μ₃ = ⟨(J-⟨J⟩)³⟩ under the tilted volume-weighted distribution.
    pub mu3: f64,
    /// Reduced skewness s₃ = μ₃ / Var²
    pub s3: f64,
    /// J grid used for PDFs.
    pub j_grid: Vec<f64>,
    /// p_V(J) — volume-weighted PDF (normalized).
    pub pdf_v: Vec<f64>,
    /// p_M(J) — mass-weighted PDF (normalized separately).
    pub pdf_m: Vec<f64>,
}

/// Helper: build a uniform J grid of `n_bins` centres on [j_min, j_max].
pub fn uniform_j_grid(j_min: f64, j_max: f64, n_bins: usize) -> Vec<f64> {
    let dj = (j_max - j_min) / n_bins as f64;
    (0..n_bins).map(|i| j_min + (i as f64 + 0.5) * dj).collect()
}

/// Tilted-Doroshkevich single-pass quadrature.
///
/// Evaluates, in one sweep through the 3D eigenvalue quadrature:
///  - Moments of (1-λ₁)(1-λ₂)(1-λ₃) = J under the volume-weighted tilted PDF.
///  - Histogram of J values (volume-weighted and mass-weighted) on `j_grid`.
///
/// # Parameters
/// * `sigma`  — Doroshkevich scale
/// * `alpha`  — tilt coefficient on I₁
/// * `beta`   — tilt coefficient on I₂
/// * `b1`     — linear bias (enters via α_effective = α + 6 b₁)
/// * `j_grid` — uniform grid of J bin centres (PDF values landed here)
/// * `n_gauss`, `l_range` — quadrature parameters
pub fn doroshkevich_tilted_pass(
    sigma: f64, alpha: f64, beta: f64, b1: f64,
    j_grid: &[f64], n_gauss: usize, l_range: f64,
) -> TiltedPass {
    let n_bins = j_grid.len();
    let empty_hist = || vec![0.0; n_bins];

    if sigma <= 0.0 {
        // Degenerate δ(J − 1); still fill the struct sensibly.
        return TiltedPass {
            sigma: 0.0, alpha, beta, b1,
            mean_j: 1.0, variance: 0.0, mu3: 0.0, s3: 0.0,
            j_grid: j_grid.to_vec(), pdf_v: empty_hist(), pdf_m: empty_hist(),
        };
    }

    // Infer bin width from grid assumption of uniform spacing.
    let dj = if n_bins >= 2 { j_grid[1] - j_grid[0] } else { 1.0 };
    let j_min_edge = j_grid[0] - 0.5 * dj;
    let inv_dj = 1.0 / dj;

    let (nodes, weights) = gauss_legendre(n_gauss);
    let sig2 = sigma * sigma;
    let alpha_eff = alpha + 6.0 * b1;

    let e3_lo = -l_range * sigma;
    let e3_hi = l_range * sigma;
    let e3_scale = (e3_hi - e3_lo) / 2.0;
    let e3_mid = (e3_hi + e3_lo) / 2.0;

    let gap_hi = 2.0 * l_range * sigma;
    let gap_scale = gap_hi / 2.0;
    let gap_mid = gap_hi / 2.0;

    // Accumulators.
    let mut z_v = 0.0_f64;    // partition function (volume-weighted)
    let mut z_m = 0.0_f64;    // partition function (mass-weighted)
    let mut s1_v = 0.0_f64;   // ⟨J⟩
    let mut s2_v = 0.0_f64;   // ⟨J²⟩
    let mut s3_v = 0.0_f64;   // ⟨J³⟩
    let mut pdf_v = empty_hist();
    let mut pdf_m = empty_hist();

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

                // Untilted Doroshkevich exponent + linear/quadratic tilt + bias.
                let exp_arg = -3.0 * i1_inv * i1_inv / sig2
                            + 7.5 * i2_inv / sig2
                            + alpha_eff * i1_inv
                            + beta * i2_inv;
                if exp_arg < -200.0 { continue; }

                let vandermonde = g1 * g2 * (g1 + g2);
                let fw = exp_arg.exp() * vandermonde * w3 * wg2 * wg1;

                let j = (1.0 - e1) * (1.0 - e2) * (1.0 - e3);

                z_v += fw;
                s1_v += fw * j;
                let jj = j * j;
                s2_v += fw * jj;
                s3_v += fw * jj * j;

                // Mass-weighted partition function: only finite contribution
                // when J > 0 (the mass-weighted density is ill-defined in
                // shell-crossed regions J ≤ 0; we clip them out).
                if j > 0.0 {
                    let fw_m = fw / j;
                    z_m += fw_m;

                    // Histogram both PDFs.
                    if j >= j_min_edge && j < j_min_edge + (n_bins as f64) * dj {
                        let bin = ((j - j_min_edge) * inv_dj) as usize;
                        if bin < n_bins {
                            pdf_v[bin] += fw;
                            pdf_m[bin] += fw_m;
                        }
                    }
                } else if j >= j_min_edge && j < j_min_edge + (n_bins as f64) * dj {
                    // Volume-weighted PDF still gets the J<0 contribution.
                    let bin = ((j - j_min_edge) * inv_dj) as usize;
                    if bin < n_bins {
                        pdf_v[bin] += fw;
                    }
                }
            }
        }
    }

    if z_v <= 0.0 {
        return TiltedPass {
            sigma, alpha, beta, b1,
            mean_j: 1.0, variance: 0.0, mu3: 0.0, s3: 0.0,
            j_grid: j_grid.to_vec(), pdf_v: empty_hist(), pdf_m: empty_hist(),
        };
    }

    let mean_j = s1_v / z_v;
    let m2 = s2_v / z_v;
    let m3 = s3_v / z_v;
    let variance = m2 - mean_j * mean_j;
    let mu3_val = m3 - 3.0 * m2 * mean_j + 2.0 * mean_j.powi(3);
    let s3_red = if variance > 1e-30 { mu3_val / (variance * variance) } else { 0.0 };

    // Normalize PDFs so ∫ p dJ = 1 (equivalent: Σ p_i dJ = 1 on uniform grid).
    let norm_v_inv = 1.0 / (z_v * dj);
    for p in pdf_v.iter_mut() { *p *= norm_v_inv; }
    if z_m > 0.0 {
        let norm_m_inv = 1.0 / (z_m * dj);
        for p in pdf_m.iter_mut() { *p *= norm_m_inv; }
    }

    TiltedPass {
        sigma, alpha, beta, b1,
        mean_j, variance, mu3: mu3_val, s3: s3_red,
        j_grid: j_grid.to_vec(), pdf_v, pdf_m,
    }
}

/// Lightweight moment-only version of the tilted pass (no histogram overhead).
/// Used inside the 2D moment-matching root-finder.
pub fn doroshkevich_tilted_moments(
    sigma: f64, alpha: f64, beta: f64, b1: f64, n_gauss: usize, l_range: f64,
) -> (f64, f64, f64, f64) {
    // Returns (mean_j, variance, mu3, s3_reduced).
    if sigma <= 0.0 { return (1.0, 0.0, 0.0, 0.0); }

    let (nodes, weights) = gauss_legendre(n_gauss);
    let sig2 = sigma * sigma;
    let alpha_eff = alpha + 6.0 * b1;

    let e3_lo = -l_range * sigma;
    let e3_hi = l_range * sigma;
    let e3_scale = (e3_hi - e3_lo) / 2.0;
    let e3_mid = (e3_hi + e3_lo) / 2.0;
    let gap_hi = 2.0 * l_range * sigma;
    let gap_scale = gap_hi / 2.0;
    let gap_mid = gap_hi / 2.0;

    let mut z = 0.0_f64;
    let mut s1 = 0.0_f64;
    let mut s2 = 0.0_f64;
    let mut s3 = 0.0_f64;

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
                let exp_arg = -3.0 * i1_inv * i1_inv / sig2
                            + 7.5 * i2_inv / sig2
                            + alpha_eff * i1_inv
                            + beta * i2_inv;
                if exp_arg < -200.0 { continue; }
                let vdm = g1 * g2 * (g1 + g2);
                let fw = exp_arg.exp() * vdm * w3 * wg2 * wg1;

                let j = (1.0 - e1) * (1.0 - e2) * (1.0 - e3);
                z += fw;
                s1 += fw * j;
                let jj = j * j;
                s2 += fw * jj;
                s3 += fw * jj * j;
            }
        }
    }

    if z <= 0.0 { return (1.0, 0.0, 0.0, 0.0); }
    let mean_j = s1 / z;
    let m2 = s2 / z;
    let m3 = s3 / z;
    let var = m2 - mean_j * mean_j;
    let mu3 = m3 - 3.0 * m2 * mean_j + 2.0 * mean_j.powi(3);
    let s3_red = if var > 1e-30 { mu3 / (var * var) } else { 0.0 };
    (mean_j, var, mu3, s3_red)
}

/// Find (α, β) such that the tilted Doroshkevich matches target variance and
/// reduced skewness s₃. 2D Newton with numerical Jacobian.
///
/// * `target_var` — desired Var(J)
/// * `target_s3`  — desired reduced skewness s₃ = μ₃/Var²
/// * Returns `(alpha, beta, achieved_var, achieved_s3)`.
///
/// Initial guess is (0, 0) — the untilted Doroshkevich. Converges in ~5-10
/// iterations when target moments are within ~50% of untilted values; may not
/// converge if targets are very far (e.g. σ-corrected but targets still 5x
/// different — unphysical).
pub fn fit_tilt_to_moments(
    sigma: f64, target_var: f64, target_s3: f64,
    n_gauss: usize, l_range: f64, max_iter: usize, tol: f64,
) -> (f64, f64, f64, f64) {
    let mut alpha = 0.0_f64;
    let mut beta = 0.0_f64;
    let eps_fd = 1e-4;

    for _ in 0..max_iter {
        let (_, v0, _, s30) = doroshkevich_tilted_moments(sigma, alpha, beta, 0.0, n_gauss, l_range);
        let r = [v0 - target_var, s30 - target_s3];
        // Normalised residual (so target scales are comparable).
        let r_norm = (r[0] / target_var.abs().max(1e-10)).hypot(r[1] / target_s3.abs().max(1e-10));
        if r_norm < tol {
            return (alpha, beta, v0, s30);
        }

        // Numerical Jacobian ∂(v, s3)/∂(α, β) via forward differences.
        let (_, v_a, _, s3_a) = doroshkevich_tilted_moments(sigma, alpha + eps_fd, beta, 0.0, n_gauss, l_range);
        let (_, v_b, _, s3_b) = doroshkevich_tilted_moments(sigma, alpha, beta + eps_fd, 0.0, n_gauss, l_range);
        let jac = [
            [(v_a - v0) / eps_fd, (v_b - v0) / eps_fd],
            [(s3_a - s30) / eps_fd, (s3_b - s30) / eps_fd],
        ];
        // Invert 2x2: [[a,b],[c,d]]^-1 = (1/det)[[d,-b],[-c,a]]
        let det = jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0];
        if det.abs() < 1e-16 {
            // Singular Jacobian; give up and return current guess.
            return (alpha, beta, v0, s30);
        }
        let inv_det = 1.0 / det;
        // Newton step: (α, β) -= J^-1 · r  (with damping for stability)
        let d_alpha = inv_det * ( jac[1][1] * r[0] - jac[0][1] * r[1]);
        let d_beta  = inv_det * (-jac[1][0] * r[0] + jac[0][0] * r[1]);
        let damping = 0.7;
        alpha -= damping * d_alpha;
        beta  -= damping * d_beta;
    }

    let (_, v_final, _, s3_final) =
        doroshkevich_tilted_moments(sigma, alpha, beta, 0.0, n_gauss, l_range);
    (alpha, beta, v_final, s3_final)
}

/// Cumulative trapezoid-rule integration of a PDF on a uniform grid.
/// Returns CDF[i] = ∫_{-∞}^{J_i} p(J) dJ, approximated by the cumulative
/// trapezoid from the PDF bin centres.
pub fn cdf_from_pdf(pdf: &[f64], j_grid: &[f64]) -> Vec<f64> {
    let n = pdf.len();
    assert_eq!(j_grid.len(), n);
    let mut cdf = vec![0.0; n];
    if n < 2 { return cdf; }
    let dj = j_grid[1] - j_grid[0];
    // Start cumulating from the first bin (assume p=0 below j_grid[0]-dj/2).
    cdf[0] = 0.5 * pdf[0] * dj;
    for i in 1..n {
        cdf[i] = cdf[i - 1] + 0.5 * (pdf[i - 1] + pdf[i]) * dj;
    }
    // Clamp to [0, 1] since both PDFs normalize to 1.
    for v in cdf.iter_mut() {
        if *v < 0.0 { *v = 0.0; }
        if *v > 1.0 { *v = 1.0; }
    }
    cdf
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

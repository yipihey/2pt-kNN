//! Lagrangian bias in V-language.
//!
//! The three-parameter Lagrangian bias expansion:
//!   δ_t^L = b₁ δ_L + (b₂/2)[δ_L² − σ²] + b_{s²}[s² − ⟨s²⟩]
//!
//! In V-variables:
//!   δ_L ≈ −3(V_iso^{1/3} − 1) + O(σ²)
//!   s² ≈ ½(V_iso − V)/V_iso^{2/3} + O(σ⁴)
//!
//! The bias parameters are separately measurable from DD/RD ratios,
//! factorial moments, and the sphere-vs-ellipsoid correction.

/// Lagrangian bias parameters.
#[derive(Debug, Clone)]
pub struct LagrangianBias {
    /// Linear bias b₁: response to mean stretch (isotropic compression/expansion)
    pub b1: f64,
    /// Quadratic bias b₂: quadratic response to mean stretch
    pub b2: f64,
    /// Tidal bias b_{s²}: response to tidal deficit V_iso − V
    pub bs2: f64,
    /// Linear variance σ² at the relevant mass scale
    pub sigma2: f64,
}

impl LagrangianBias {
    /// Unbiased tracers: b₁ = 1, b₂ = b_{s²} = 0.
    pub fn unbiased(sigma2: f64) -> Self {
        Self { b1: 1.0, b2: 0.0, bs2: 0.0, sigma2 }
    }

    /// Linear bias only.
    pub fn linear(b1: f64, sigma2: f64) -> Self {
        Self { b1, b2: 0.0, bs2: 0.0, sigma2 }
    }

    /// Full three-parameter bias.
    pub fn full(b1: f64, b2: f64, bs2: f64, sigma2: f64) -> Self {
        Self { b1, b2, bs2, sigma2 }
    }

    /// Conditional mean tracer overdensity ⟨δ_t | V⟩.
    ///
    /// At leading order in the bias expansion:
    ///   ⟨δ_t | V⟩ ≈ −3b₁(V^{1/3} − 1) + (9b₂/2)(V^{1/3} − 1)²
    ///               + (b_{s²}/2) × tidal_deficit_term
    pub fn conditional_delta(&self, v: f64) -> f64 {
        if v <= 0.0 {
            return 0.0;
        }
        let v13 = v.cbrt();
        let dv = v13 - 1.0;

        let linear = -3.0 * self.b1 * dv;
        let quadratic = 4.5 * self.b2 * (dv * dv - self.sigma2 / 9.0);

        // Tidal deficit: V_iso − V at leading order.
        // V_iso^{1/3} = mean stretch μ̄ = 1 + I₁/3
        // At leading order V_iso ≈ (1 + I₁/3)³ and V_iso − V ∝ S (tidal shear scalar)
        // For the conditional expectation ⟨s² | V⟩ we use the mean tidal shear
        // at fixed volume, which is ⟨S | V⟩ ≈ (4/15)σ² (independent of V at ZA).
        let tidal = 0.5 * self.bs2 * (4.0 / 15.0 * self.sigma2);

        linear + quadratic + tidal
    }

    /// Conditional second moment ⟨δ_t² | V⟩ for factorial moment deconvolution.
    pub fn conditional_delta2(&self, v: f64) -> f64 {
        let d = self.conditional_delta(v);
        // At leading order, ⟨δ_t²|V⟩ ≈ ⟨δ_t|V⟩² + Var(δ_t|V)
        // Var(δ_t|V) ≈ b₁² × conditional_variance_of_δ_L_given_V
        let cond_var = self.b1.powi(2) * self.sigma2 / 3.0;
        d * d + cond_var
    }

    /// Void-tail slope: d ln(P_DD/P_RD) / d(V^{1/3}).
    ///
    /// In the void tail (large V), this approaches −3b₁.
    pub fn void_tail_slope(&self) -> f64 {
        -3.0 * self.b1
    }

    /// Sphere-vs-ellipsoid correction to the kNN count.
    ///
    /// δ⟨k⟩_ellip = b_{s²} × ⟨(tidal at q₀) · (ellipsoidal window quadrupole)⟩
    /// This provides an independent handle on b_{s²}.
    pub fn ellipsoidal_correction(&self, sigma2: f64) -> f64 {
        self.bs2 * 4.0 / 15.0 * sigma2
    }

    /// Predicted DD/RD log-ratio as a function of V^{1/3}.
    ///
    /// Returns (V^{1/3}, log(P_DD/P_RD)) pairs for a grid of V values.
    pub fn dd_rd_log_ratio_curve(&self, v_min: f64, v_max: f64, n_points: usize) -> (Vec<f64>, Vec<f64>) {
        let dv = (v_max - v_min) / (n_points - 1) as f64;
        let mut v13_vals = Vec::with_capacity(n_points);
        let mut log_ratios = Vec::with_capacity(n_points);

        for i in 0..n_points {
            let v = v_min + i as f64 * dv;
            if v > 0.0 {
                let v13 = v.cbrt();
                let delta = self.conditional_delta(v);
                let log_ratio = (1.0 + delta).ln();
                v13_vals.push(v13);
                log_ratios.push(log_ratio);
            }
        }
        (v13_vals, log_ratios)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unbiased_conditional_delta_zero_at_v1() {
        let bias = LagrangianBias::unbiased(0.1);
        let delta = bias.conditional_delta(1.0);
        // At V=1: V^{1/3} = 1 → dv = 0 → linear term vanishes
        // Quadratic term: 4.5 × 0 × (0 - σ²/9) = -4.5 × σ²/9 × 0 = small
        // But b₂ = 0 for unbiased, so it's just the tidal term
        assert!(delta.abs() < 0.1, "⟨δ_t | V=1⟩ = {} should be small", delta);
    }

    #[test]
    fn test_void_tail_slope() {
        let bias = LagrangianBias::linear(2.0, 0.1);
        assert!((bias.void_tail_slope() - (-6.0)).abs() < 1e-15);
    }

    #[test]
    fn test_dd_rd_curve_monotone() {
        let bias = LagrangianBias::linear(1.5, 0.1);
        // Keep V range where the linear bias expansion is valid: |δ| < 1
        // For b₁=1.5: δ = -4.5(V^{1/3}-1), so need V^{1/3} < 1.22, V < 1.8
        let (_v13, lr) = bias.dd_rd_log_ratio_curve(0.5, 1.7, 50);
        let n = lr.len();
        assert!(lr[n - 1] < lr[0], "Log-ratio should decrease with V^{{1/3}} for b1 > 0");
    }
}

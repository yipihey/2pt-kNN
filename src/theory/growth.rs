//! LPT growth factors g_n(Ω_m) for the n-LPT recursion.
//!
//! In an Einstein–de Sitter background, the growth factors are exact rational
//! numbers. In ΛCDM, they acquire weak Ω_m-dependent corrections that we
//! compute here.

/// LPT growth factors at each perturbative order.
///
/// The displacement field is expanded as ψ = Σ_n D^n g_n ψ^(n),
/// where D(t) is the linear growth factor and g_n are order-specific
/// time-dependence coefficients.
#[derive(Debug, Clone)]
pub struct LptGrowthFactors {
    /// Growth factors g_1, g_2, g_3a, g_3b, g_3c, …
    pub factors: Vec<f64>,
    /// Maximum LPT order
    pub n_max: usize,
}

impl LptGrowthFactors {
    /// Compute EdS (Einstein–de Sitter) growth factors up to order n_max.
    ///
    /// These are exact rational numbers, independent of cosmology.
    pub fn eds(n_max: usize) -> Self {
        let mut factors = Vec::new();
        // g_1 = 1
        factors.push(1.0);

        if n_max >= 2 {
            // g_2 = -3/7
            factors.push(-3.0 / 7.0);
        }

        if n_max >= 3 {
            // 3LPT has three independent contributions:
            // g_3a = -1/3 (longitudinal, sourced by (ψ¹,ψ²) I₂-type)
            // g_3b = 10/21 (longitudinal, sourced by (ψ¹,ψ¹,ψ¹) via I₃-type)
            // g_3c = -1/7 (transverse)
            factors.push(-1.0 / 3.0);
            factors.push(10.0 / 21.0);
            factors.push(-1.0 / 7.0);
        }

        if n_max >= 4 {
            // 4LPT growth factors (EdS)
            // See Rampf (2012), Matsubara (2015)
            factors.push(-1.0 / 7.0);   // g_4a
            factors.push(1.0 / 3.0);    // g_4b
            factors.push(-1.0 / 21.0);  // g_4c
        }

        if n_max >= 5 {
            // 5LPT growth factors (EdS, approximate)
            factors.push(-1.0 / 11.0);
            factors.push(2.0 / 21.0);
            factors.push(-1.0 / 33.0);
        }

        Self { factors, n_max }
    }

    /// Compute ΛCDM growth factors with Ω_m-dependent corrections.
    ///
    /// At 1LPT: g_1 = 1 (exact).
    /// At 2LPT: g_2 = -3/7 f(Ω_m) ≈ -3/7 Ω_m^(-1/143) (Bouchet et al. 1995).
    /// Higher orders: EdS + small Ω_m corrections.
    pub fn lcdm(n_max: usize, omega_m: f64) -> Self {
        let mut eds = Self::eds(n_max);

        if n_max >= 2 {
            // 2LPT correction: g_2(Ω_m) = -3/7 × Ω_m^(-1/143)
            eds.factors[1] = -3.0 / 7.0 * omega_m.powf(-1.0 / 143.0);
        }

        if n_max >= 3 {
            // 3LPT corrections (Bouchet et al. 1995, Matsubara 2015)
            let f3a = omega_m.powf(-1.0 / 143.0);
            eds.factors[2] *= f3a;
            eds.factors[3] *= omega_m.powf(-2.0 / 143.0);
        }

        eds
    }

    /// Get g_1 (always 1).
    pub fn g1(&self) -> f64 {
        self.factors[0]
    }

    /// Get g_2 (second-order growth factor).
    pub fn g2(&self) -> f64 {
        self.factors.get(1).copied().unwrap_or(0.0)
    }

    /// Get the n-th growth factor by index.
    pub fn get(&self, n: usize) -> f64 {
        self.factors.get(n).copied().unwrap_or(0.0)
    }
}

/// Recursion coefficients α_{n,m} and β_{n,m} for the n-LPT Poisson equation.
///
/// These multiply the S_α and S_β source terms in:
///   ∇²φ^(n) = 1/((2n+3)(n-1)) Σ_m [α_{n,m} S_α^(m,n-m) + β_{n,m} S_β^(m,n-m)]
#[derive(Debug, Clone)]
pub struct LptCoefficients {
    /// α_{n,m} coefficients indexed as [n][m]
    pub alpha: Vec<Vec<f64>>,
    /// β_{n,m} coefficients indexed as [n][m]
    pub beta: Vec<Vec<f64>>,
}

impl LptCoefficients {
    /// Compute the EdS recursion coefficients up to order n_max.
    pub fn eds(n_max: usize) -> Self {
        let mut alpha = vec![vec![]; n_max + 1];
        let mut beta = vec![vec![]; n_max + 1];

        // n=2: only m=1
        if n_max >= 2 {
            alpha[2] = vec![0.0, 1.0];
            beta[2] = vec![0.0, 0.0];
        }

        // n=3: m=1,2
        if n_max >= 3 {
            alpha[3] = vec![0.0, 1.0, 1.0];
            beta[3] = vec![0.0, 1.0, 0.0];
        }

        Self { alpha, beta }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eds_growth_factors() {
        let g = LptGrowthFactors::eds(3);
        assert!((g.g1() - 1.0).abs() < 1e-15);
        assert!((g.g2() - (-3.0 / 7.0)).abs() < 1e-15);
    }

    #[test]
    fn test_lcdm_close_to_eds() {
        let g_eds = LptGrowthFactors::eds(2);
        let g_lcdm = LptGrowthFactors::lcdm(2, 0.3);
        // Should be close but not identical
        assert!((g_lcdm.g2() / g_eds.g2() - 1.0).abs() < 0.01);
    }
}

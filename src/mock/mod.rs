//! CoxMock: isotropic line-point Cox process generator.
//!
//! Following the Euclid 2PCF-GC paper (de la Torre et al. 2025), the CoxMock
//! suite uses an isotropic line-point Cox process where lines of given length
//! are randomly placed in a periodic cube and points are randomly scattered
//! on those lines. The resulting ξ(r) is a damped power law with index −2,
//! analytically known.
//!
//! # Analytic two-point correlation function
//!
//! For N_L lines of length ℓ in a cube of volume V, with one point per line
//! (N_p = N_L), the pair probability has an excess from pairs on the same line.
//! However in the CoxMock convention, N_p points are distributed across N_L
//! lines (N_p / N_L points per line on average).
//!
//! The analytic ξ(r) for the line-point Cox process is derived from the
//! pair distance PDF on a line segment: f(d) = (2/ℓ)(1 − d/ℓ) for d ∈ [0, ℓ].
//!
//! With random line assignment (each of N_p points picks a uniformly
//! random line), each ordered pair has probability 1/N_L of sharing a
//! line.  The expected same-line ordered pairs are N_p(N_p−1)/N_L.
//! Each same-line pair has separation drawn from f(d).  The excess in
//! a spherical shell 4πr²dr, divided by the Poisson expectation
//! n̄·4πr²dr, gives:
//!
//!   ξ(r) = (N_p − 1) / (N_L · n̄ · 2π · r² · ℓ) · (1 − r/ℓ)
//!        ≈ V / (N_L · 2π · r² · ℓ) · (1 − r/ℓ)     for 0 < r < ℓ
//!   ξ(r) = 0                                            for r ≥ ℓ
//!
//! This is an r⁻² · (1 − r/ℓ) profile — so DD(r) ∝ r² ξ(r) ∝ (1 − r/ℓ),
//! which is nearly constant at small r and falls linearly to zero at r = ℓ.

use rand::Rng;
use rand::SeedableRng;

/// Parameters for the CoxMock generator.
#[derive(Debug, Clone)]
pub struct CoxMockParams {
    /// Side length of the periodic cube
    pub box_size: f64,
    /// Number of random lines
    pub n_lines: usize,
    /// Length of each line
    pub line_length: f64,
    /// Total number of points to scatter on lines
    pub n_points: usize,
}

impl CoxMockParams {
    /// Euclid-like parameters (scaled down for quick validation).
    pub fn euclid_small() -> Self {
        Self {
            box_size: 1000.0,
            n_lines: 10_000,
            line_length: 400.0,
            n_points: 100_000,
        }
    }

    /// Minimal test parameters.
    pub fn tiny() -> Self {
        Self {
            box_size: 500.0,
            n_lines: 5_000,
            line_length: 200.0,
            n_points: 50_000,
        }
    }

    /// Fast validation parameters: moderate ξ, good DD/DR overlap.
    ///
    /// N_p = 10_000 on N_L = 1_000 lines of length 200 in a 500³ box.
    /// m = 10, ξ(r) = 0.5625/r · (1 − r/200) — peaks at ξ(5) ≈ 0.11.
    /// n̄ = 8×10⁻⁵ → r₁ ≈ 14 Mpc, same-line spacing ≈ 20 Mpc.
    /// The 1st NN is usually NOT a same-line pair, so DD and DR kNN
    /// distances overlap well: the finite-k truncation bias is small.
    /// k=32 probes to r ≈ 46 Mpc where ξ ≈ 0.01.
    pub fn validation() -> Self {
        Self {
            box_size: 500.0,
            n_lines: 1_000,
            line_length: 200.0,
            n_points: 10_000,
        }
    }

    /// Characteristic k-NN distance for the k-th neighbor at this density.
    /// r_k ≈ (k / (n̄ · 4π/3))^{1/3}
    pub fn r_char_k(&self, k: usize) -> f64 {
        (k as f64 / (self.nbar() * 4.0 / 3.0 * std::f64::consts::PI)).cbrt()
    }

    /// Poisson field for bias testing: m=1, no clustering, ξ=0.
    pub fn poisson() -> Self {
        Self {
            box_size: 500.0,
            n_lines: 10_000,
            line_length: 200.0,
            n_points: 10_000,
        }
    }

    /// Mean number density n̄ = N_p / V
    pub fn nbar(&self) -> f64 {
        self.n_points as f64 / self.volume()
    }

    /// Volume of the box
    pub fn volume(&self) -> f64 {
        self.box_size.powi(3)
    }

    /// Mean number of points per line
    pub fn points_per_line(&self) -> f64 {
        self.n_points as f64 / self.n_lines as f64
    }

    /// Analytic ξ(r) for this Cox process.
    ///
    /// ξ(r) = (N_p − 1) / (N_L · n̄ · 2π · r² · ℓ) · (1 − r/ℓ)
    ///       ≈ V / (N_L · 2π · r² · ℓ) · (1 − r/ℓ)       for 0 < r < ℓ
    /// ξ(r) = 0                                               for r ≥ ℓ
    ///
    /// Each point picks a random line (Multinomial assignment), so the
    /// expected number of same-line ordered pairs is N_p(N_p−1)/N_L
    /// (each ordered pair has probability 1/N_L of sharing a line).
    /// Their separation PDF is f(d) = (2/ℓ)(1−d/ℓ).  Dividing the
    /// excess in shell 4πr²dr by the Poisson expectation gives the
    /// formula above.  Note: for deterministic m-per-line assignment,
    /// the prefactor would be (m−1) instead of m = N_p/N_L.
    pub fn xi_analytic(&self, r: f64) -> f64 {
        if r <= 0.0 || r >= self.line_length {
            return 0.0;
        }
        let np = self.n_points as f64;
        let nl = self.n_lines as f64;
        let nbar = self.nbar();
        // (N_p - 1) / (N_L · n̄ · 2π · r² · ℓ) · (1 − r/ℓ)
        let prefactor = (np - 1.0) / (nl * nbar * 2.0 * std::f64::consts::PI * self.line_length);
        prefactor * (1.0 / (r * r)) * (1.0 - r / self.line_length)
    }
}

/// A generated CoxMock realization.
pub struct CoxMock {
    /// The 3D positions of the points
    pub positions: Vec<[f64; 3]>,
    /// The parameters used to generate this mock
    pub params: CoxMockParams,
}

impl CoxMock {
    /// Generate a CoxMock realization with the given parameters and RNG seed.
    pub fn generate(params: &CoxMockParams, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut positions = Vec::with_capacity(params.n_points);

        // Pre-generate all line endpoints and directions
        let lines: Vec<([f64; 3], [f64; 3])> = (0..params.n_lines)
            .map(|_| {
                // Random endpoint inside the box
                let endpoint: [f64; 3] = [
                    rng.gen::<f64>() * params.box_size,
                    rng.gen::<f64>() * params.box_size,
                    rng.gen::<f64>() * params.box_size,
                ];
                // Random direction on the unit sphere (uniform via cos⁻¹)
                let theta = (1.0 - 2.0 * rng.gen::<f64>()).acos();
                let phi = rng.gen::<f64>() * 2.0 * std::f64::consts::PI;
                let dir: [f64; 3] = [
                    theta.sin() * phi.cos(),
                    theta.sin() * phi.sin(),
                    theta.cos(),
                ];
                (endpoint, dir)
            })
            .collect();

        // Place points on randomly chosen lines
        for _ in 0..params.n_points {
            let line_idx = rng.gen_range(0..params.n_lines);
            let (ref endpoint, ref dir) = lines[line_idx];
            let t = rng.gen::<f64>() * params.line_length;

            let mut point = [
                endpoint[0] + t * dir[0],
                endpoint[1] + t * dir[1],
                endpoint[2] + t * dir[2],
            ];

            // Periodic boundary conditions: wrap into [0, box_size)
            for coord in point.iter_mut() {
                *coord = coord.rem_euclid(params.box_size);
            }

            positions.push(point);
        }

        CoxMock {
            positions,
            params: params.clone(),
        }
    }

    /// Generate a uniform random catalog (for DR and RR terms).
    pub fn generate_randoms(n_random: usize, box_size: f64, seed: u64) -> Vec<[f64; 3]> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..n_random)
            .map(|_| {
                [
                    rng.gen::<f64>() * box_size,
                    rng.gen::<f64>() * box_size,
                    rng.gen::<f64>() * box_size,
                ]
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coxmock_generates_correct_count() {
        let params = CoxMockParams::tiny();
        let mock = CoxMock::generate(&params, 42);
        assert_eq!(mock.positions.len(), params.n_points);
    }

    #[test]
    fn test_positions_in_box() {
        let params = CoxMockParams::tiny();
        let mock = CoxMock::generate(&params, 42);
        for pos in &mock.positions {
            for &c in pos {
                assert!(c >= 0.0 && c < params.box_size);
            }
        }
    }

    #[test]
    fn test_xi_analytic_boundary() {
        let params = CoxMockParams::tiny();
        assert!(params.xi_analytic(0.0) == 0.0);
        assert!(params.xi_analytic(params.line_length) == 0.0);
        assert!(params.xi_analytic(params.line_length * 1.1) == 0.0);
        assert!(params.xi_analytic(1.0) > 0.0);
    }
}

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
//! The analytic ξ(r) for the line-point Cox process is:
//!
//!   ξ(r) = (V / (N_L · ℓ)) · (1/r) · (1 − r/ℓ)   for r < ℓ
//!   ξ(r) = 0                                          for r ≥ ℓ
//!
//! This is an r⁻¹ · (1 − r/ℓ) profile — a damped "power law" with effective
//! index −1 in ξ(r), corresponding to −2 in the pair count since
//! DD(r) ∝ r² ξ(r) ∝ r · (1 − r/ℓ).
//!
//! For the general case of m = N_p / N_L points per line, the self-pair
//! contribution scales as m(m−1), giving:
//!
//!   ξ(r) = m(m−1) · V / (N_p² · ℓ) · (1/r) · (1 − r/ℓ)

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
    /// ξ(r) = m(m−1) · V / (N_p² · ℓ) · (1/r) · (1 − r/ℓ)   for r < ℓ
    /// ξ(r) = 0                                                   for r ≥ ℓ
    ///
    /// where m = N_p / N_L is the mean points per line.
    pub fn xi_analytic(&self, r: f64) -> f64 {
        if r <= 0.0 || r >= self.line_length {
            return 0.0;
        }
        let m = self.points_per_line();
        let np = self.n_points as f64;
        let prefactor = m * (m - 1.0) * self.volume() / (np * np * self.line_length);
        prefactor * (1.0 / r) * (1.0 - r / self.line_length)
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
                // Random direction on the unit sphere
                let theta = rng.gen::<f64>() * std::f64::consts::PI;
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

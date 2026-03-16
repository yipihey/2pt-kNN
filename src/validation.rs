//! Reusable CoxMock validation pipeline.
//!
//! Extracts the core validation logic so it can be called from the CLI binary,
//! from WASM entry points, or from tests — with either CPU or GPU backends.

use crate::estimator::{KnnDistances, KnnDistributions, LandySzalayKnn, linear_bins};
use crate::ladder::DilutionLadder;
use crate::mock::{CoxMock, CoxMockParams};
#[cfg(not(target_arch = "wasm32"))]
use crate::tree::PointTree;
use serde::{Deserialize, Serialize};

/// Configuration for a CoxMock validation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Number of independent mock realizations
    pub n_mocks: usize,
    /// Maximum kNN count
    pub k_max: usize,
    /// Number of radial bins
    pub n_bins: usize,
    /// Minimum separation
    pub r_min: f64,
    /// Maximum separation
    pub r_max: f64,
    /// Random/data ratio for DR term
    pub random_ratio: usize,
    /// CoxMock parameters
    pub params: CoxMockParams,
    /// Maximum dilution ladder level
    pub max_dilution_level: usize,
    /// Box size for periodic boundary conditions (None = non-periodic)
    pub box_size: Option<f64>,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            n_mocks: 10,
            k_max: 8,
            n_bins: 40,
            r_min: 5.0,
            r_max: 250.0,
            random_ratio: 5,
            params: CoxMockParams::euclid_small(),
            max_dilution_level: 2,
            box_size: None,
        }
    }
}

/// Full results of a CoxMock validation run.
#[derive(Debug, Clone, Serialize)]
pub struct ValidationResult {
    /// Bin centers
    pub r_centers: Vec<f64>,
    /// Analytic ξ(r) at bin centers
    pub xi_analytic: Vec<f64>,
    /// Mean estimated ξ(r) across mocks
    pub mean_xi: Vec<f64>,
    /// Standard deviation of ξ(r) across mocks
    pub std_xi: Vec<f64>,
    /// Standard error of the mean
    pub stderr_xi: Vec<f64>,
    /// χ² statistic
    pub chi2: f64,
    /// χ²/dof
    pub chi2_per_dof: f64,
    /// Number of mocks
    pub n_mocks: usize,

    // --- Full kNN outputs (unique to kNN, not available from pair counting) ---

    /// Empirical kNN-CDFs: cdf_k[k-1][bin_idx] for the last mock
    pub knn_cdfs: Vec<Vec<f64>>,
    /// ξ̂ per dilution level (last mock): dilution_xi[level][bin_idx]
    pub dilution_xi: Vec<Vec<f64>>,
    /// Variance of ξ̂ across subsamples at each dilution level
    pub dilution_variance: Vec<Vec<f64>>,
    /// Characteristic scale per dilution level
    pub dilution_r_char: Vec<f64>,
}

/// Trait for kNN distance backends (CPU or GPU).
///
/// Both the CPU tree-based path and the GPU brute-force path implement this,
/// allowing the validation pipeline to be backend-agnostic.
pub trait KnnBackend {
    fn query_distances(
        &self,
        data: &[[f64; 3]],
        queries: &[[f64; 3]],
        k_max: usize,
    ) -> KnnDistributions;

    /// Query with periodic boundary conditions. Default: delegates to non-periodic.
    fn query_distances_periodic(
        &self,
        data: &[[f64; 3]],
        queries: &[[f64; 3]],
        k_max: usize,
        _box_size: f64,
    ) -> KnnDistributions {
        self.query_distances(data, queries, k_max)
    }
}

/// CPU backend using bosque KD-tree (native only — not WASM-compatible).
#[cfg(not(target_arch = "wasm32"))]
pub struct CpuKnnBackend;

#[cfg(not(target_arch = "wasm32"))]
impl KnnBackend for CpuKnnBackend {
    fn query_distances(
        &self,
        data: &[[f64; 3]],
        queries: &[[f64; 3]],
        k_max: usize,
    ) -> KnnDistributions {
        let data_tree = PointTree::build(data.to_vec());
        let estimator = LandySzalayKnn::new(k_max);
        estimator.query_distances(&data_tree, queries)
    }

    fn query_distances_periodic(
        &self,
        data: &[[f64; 3]],
        queries: &[[f64; 3]],
        k_max: usize,
        box_size: f64,
    ) -> KnnDistributions {
        let data_tree = PointTree::build(data.to_vec());
        let estimator = LandySzalayKnn::new(k_max);
        estimator.query_distances_periodic(&data_tree, queries, box_size)
    }
}

/// Brute-force CPU kNN backend. No tree, no threads — WASM-compatible.
/// O(n·m·k) for n data points, m queries, k neighbors.
/// Fast enough for browser-scale datasets (≤100k points).
pub struct BruteForceKnnBackend;

impl KnnBackend for BruteForceKnnBackend {
    fn query_distances(
        &self,
        data: &[[f64; 3]],
        queries: &[[f64; 3]],
        k_max: usize,
    ) -> KnnDistributions {
        let k = k_max.min(data.len());
        let per_query: Vec<KnnDistances> = queries
            .iter()
            .map(|q| {
                // Compute all squared distances
                let mut dists: Vec<f64> = data
                    .iter()
                    .map(|d| {
                        let dx = q[0] - d[0];
                        let dy = q[1] - d[1];
                        let dz = q[2] - d[2];
                        dx * dx + dy * dy + dz * dz
                    })
                    .collect();

                // Partial sort to get k smallest
                dists.select_nth_unstable_by(k - 1, |a, b| {
                    a.partial_cmp(b).unwrap()
                });
                dists.truncate(k);
                dists.sort_by(|a, b| a.partial_cmp(b).unwrap());

                KnnDistances {
                    distances: dists.iter().map(|d| d.sqrt()).collect(),
                }
            })
            .collect();

        KnnDistributions { per_query, k_max }
    }
}

/// Run the full CoxMock validation pipeline.
///
/// This function is backend-agnostic: pass `CpuKnnBackend` for CPU,
/// or a GPU backend for WebGPU acceleration. The statistical pipeline
/// (pair-count density, ξ estimation, dilution ladder, kNN-CDFs)
/// is identical regardless of backend.
pub fn run_validation(config: &ValidationConfig, backend: &dyn KnnBackend) -> ValidationResult {
    let r_edges = linear_bins(config.r_min, config.r_max, config.n_bins);
    let r_centers: Vec<f64> = r_edges.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect();
    let xi_analytic: Vec<f64> = r_centers.iter().map(|&r| config.params.xi_analytic(r)).collect();

    let n_random = config.params.n_points * config.random_ratio;
    let mut all_xi: Vec<Vec<f64>> = Vec::with_capacity(config.n_mocks);

    // Store last mock's detailed outputs for kNN-CDF and dilution plots
    let mut last_knn_cdfs: Vec<Vec<f64>> = Vec::new();
    let mut last_dilution_xi: Vec<Vec<f64>> = Vec::new();
    let mut last_dilution_variance: Vec<Vec<f64>> = Vec::new();

    for mock_idx in 0..config.n_mocks {
        let seed_data = (mock_idx * 2) as u64;
        let seed_rand = (mock_idx * 2 + 1) as u64 + 1_000_000;

        // Generate data and randoms
        let mock = CoxMock::generate(&config.params, seed_data);
        let randoms = CoxMock::generate_randoms(n_random, config.params.box_size, seed_rand);

        // kNN queries via the chosen backend (periodic if configured)
        let (dd_dists, dr_dists) = if let Some(box_size) = config.box_size {
            (
                backend.query_distances_periodic(&mock.positions, &mock.positions, config.k_max, box_size),
                backend.query_distances_periodic(&mock.positions, &randoms, config.k_max, box_size),
            )
        } else {
            (
                backend.query_distances(&mock.positions, &mock.positions, config.k_max),
                backend.query_distances(&mock.positions, &randoms, config.k_max),
            )
        };

        // Pair-count densities and ξ estimation (shared pipeline)
        let dd = LandySzalayKnn::pair_count_density(&dd_dists, &r_edges);
        let dr = LandySzalayKnn::pair_count_density(&dr_dists, &r_edges);
        let xi_est = LandySzalayKnn::estimate_xi_dp(&dd, &dr);

        all_xi.push(xi_est.xi.clone());

        // On last mock, compute the kNN-CDFs and dilution ladder
        if mock_idx == config.n_mocks - 1 {
            // kNN-CDFs at bin centers for k = 1..k_max
            last_knn_cdfs = (1..=config.k_max)
                .map(|k| LandySzalayKnn::empirical_cdf(&dd_dists, k, &r_centers))
                .collect();

            // Dilution ladder
            let ladder = DilutionLadder::build(
                mock.positions.len(),
                config.max_dilution_level,
                seed_data + 999,
            );

            for level in &ladder.levels {
                let mut level_xi_estimates: Vec<Vec<f64>> = Vec::new();

                for subsample_indices in &level.subsamples {
                    let subsample_positions: Vec<[f64; 3]> = subsample_indices
                        .iter()
                        .map(|&i| mock.positions[i])
                        .collect();

                    let n_sub_random = subsample_positions.len() * config.random_ratio;
                    let sub_randoms = CoxMock::generate_randoms(
                        n_sub_random,
                        config.params.box_size,
                        seed_data + level.level as u64 * 1000,
                    );

                    let (sub_dd, sub_dr) = if let Some(box_size) = config.box_size {
                        (
                            backend.query_distances_periodic(&subsample_positions, &subsample_positions, config.k_max, box_size),
                            backend.query_distances_periodic(&subsample_positions, &sub_randoms, config.k_max, box_size),
                        )
                    } else {
                        (
                            backend.query_distances(&subsample_positions, &subsample_positions, config.k_max),
                            backend.query_distances(&subsample_positions, &sub_randoms, config.k_max),
                        )
                    };

                    let sub_dd_density =
                        LandySzalayKnn::pair_count_density(&sub_dd, &r_edges);
                    let sub_dr_density =
                        LandySzalayKnn::pair_count_density(&sub_dr, &r_edges);
                    let sub_xi = LandySzalayKnn::estimate_xi_dp(&sub_dd_density, &sub_dr_density);

                    level_xi_estimates.push(sub_xi.xi);
                }

                // Mean ξ across subsamples at this level
                let n_sub = level_xi_estimates.len() as f64;
                let mean_sub: Vec<f64> = (0..config.n_bins)
                    .map(|i| {
                        level_xi_estimates.iter().map(|xi| xi[i]).sum::<f64>() / n_sub
                    })
                    .collect();

                // Variance across subsamples
                let var_sub: Vec<f64> = (0..config.n_bins)
                    .map(|i| {
                        let m = mean_sub[i];
                        level_xi_estimates
                            .iter()
                            .map(|xi| (xi[i] - m).powi(2))
                            .sum::<f64>()
                            / (n_sub - 1.0).max(1.0)
                    })
                    .collect();

                last_dilution_xi.push(mean_sub);
                last_dilution_variance.push(var_sub);
            }
        }
    }

    // Compute mean, std, stderr across mocks
    let n = config.n_mocks as f64;
    let mean_xi: Vec<f64> = (0..config.n_bins)
        .map(|i| all_xi.iter().map(|xi| xi[i]).sum::<f64>() / n)
        .collect();
    let std_xi: Vec<f64> = (0..config.n_bins)
        .map(|i| {
            let m = mean_xi[i];
            let var = all_xi.iter().map(|xi| (xi[i] - m).powi(2)).sum::<f64>() / (n - 1.0);
            var.sqrt()
        })
        .collect();
    let stderr_xi: Vec<f64> = std_xi.iter().map(|s| s / n.sqrt()).collect();

    // χ²
    let chi2: f64 = (0..config.n_bins)
        .map(|i| {
            if stderr_xi[i] > 0.0 {
                ((mean_xi[i] - xi_analytic[i]) / stderr_xi[i]).powi(2)
            } else {
                0.0
            }
        })
        .sum();

    // Dilution characteristic scales
    let dilution_r_char: Vec<f64> = (0..=config.max_dilution_level)
        .map(|l| {
            DilutionLadder::r_char(config.k_max, 8usize.pow(l as u32), config.params.nbar())
        })
        .collect();

    ValidationResult {
        r_centers,
        xi_analytic,
        mean_xi,
        std_xi,
        stderr_xi,
        chi2,
        chi2_per_dof: chi2 / config.n_bins as f64,
        n_mocks: config.n_mocks,
        knn_cdfs: last_knn_cdfs,
        dilution_xi: last_dilution_xi,
        dilution_variance: last_dilution_variance,
        dilution_r_char,
    }
}

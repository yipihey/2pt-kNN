//! Reusable CoxMock validation pipeline.
//!
//! Extracts the core validation logic so it can be called from the CLI binary,
//! from WASM entry points, or from tests — using the bosque KD-tree backend.
//!
//! The pipeline is split into `run_single_mock()` and `aggregate_mocks()` so
//! that WASM callers can yield to the JS event loop between mocks for progress.

use crate::estimator::{
    cdf_k_values, cdf_r_grid, exclude_self_pairs, KnnCdfs, KnnDistributions,
    LandySzalayKnn, linear_bins,
};
use crate::ladder::{DilutionLadder, KnnCdfSummary};
use crate::mock::{CoxMock, CoxMockParams};
use crate::tree::PointTree;
use serde::{Deserialize, Serialize};

/// Configuration for a CoxMock validation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    pub n_mocks: usize,
    pub k_max: usize,
    pub n_bins: usize,
    pub r_min: f64,
    pub r_max: f64,
    pub random_ratio: usize,
    pub params: CoxMockParams,
    pub max_dilution_level: usize,
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

/// Per-mock output from a single validation mock.
pub struct MockResult {
    /// Estimated ξ(r) for this mock
    pub xi: Vec<f64>,
    /// kNN-CDFs from the RR catalog (for all k=1..k_max)
    pub rr_cdfs: KnnCdfs,
    /// kNN-CDFs from the DD catalog (for all k=1..k_max)
    pub dd_cdfs: KnnCdfs,
}

/// Full results of a CoxMock validation run.
#[derive(Debug, Clone, Serialize)]
pub struct ValidationResult {
    pub r_centers: Vec<f64>,
    pub xi_analytic: Vec<f64>,
    pub mean_xi: Vec<f64>,
    pub std_xi: Vec<f64>,
    pub stderr_xi: Vec<f64>,
    pub chi2: f64,
    pub chi2_per_dof: f64,
    pub n_mocks: usize,

    // --- kNN-CDF summary across mocks (mean ± std) ---

    /// Dense r-grid used for CDF evaluation
    pub cdf_r_grid: Vec<f64>,
    /// k-values used for CDFs (powers of 2)
    pub cdf_k_values: Vec<usize>,
    /// RR CDF mean across mocks: [k_idx][r_idx]
    pub cdf_rr_mean: Vec<Vec<f64>>,
    /// RR CDF std across mocks: [k_idx][r_idx]
    pub cdf_rr_std: Vec<Vec<f64>>,
    /// DD CDF mean across mocks: [k_idx][r_idx]
    pub cdf_dd_mean: Vec<Vec<f64>>,
    /// DD CDF std across mocks: [k_idx][r_idx]
    pub cdf_dd_std: Vec<Vec<f64>>,

    // --- Legacy per-bin CDFs (for backward compat) ---

    /// Empirical kNN-CDFs at bin centers: cdf_k[k-1][bin_idx] (last mock)
    pub knn_cdfs: Vec<Vec<f64>>,

    // --- Dilution ladder ---

    pub dilution_xi: Vec<Vec<f64>>,
    pub dilution_variance: Vec<Vec<f64>>,
    pub dilution_r_char: Vec<f64>,
}

fn query_knn(
    tree: &PointTree,
    queries: &[[f64; 3]],
    k_max: usize,
    box_size: Option<f64>,
    is_self_query: bool,
) -> KnnDistributions {
    let k_query = if is_self_query { k_max + 1 } else { k_max };
    let estimator = LandySzalayKnn::new(k_query);
    let dists = if let Some(bs) = box_size {
        estimator.query_distances_periodic(tree, queries, bs)
    } else {
        estimator.query_distances(tree, queries)
    };
    if is_self_query {
        exclude_self_pairs(dists, k_max)
    } else {
        dists
    }
}

/// Run a single mock and return per-mock results.
///
/// This is the inner loop body, extracted so WASM callers can yield
/// to the JS event loop between mocks for progress reporting.
pub fn run_single_mock(config: &ValidationConfig, mock_idx: usize) -> MockResult {
    let r_edges = linear_bins(config.r_min, config.r_max, config.n_bins);

    let n_random = config.params.n_points * config.random_ratio;
    let n_data = config.params.n_points;
    let n_rr = n_data;

    let seed_data = (mock_idx * 2) as u64;
    let seed_rand = (mock_idx * 2 + 1) as u64 + 1_000_000;

    let mock = CoxMock::generate(&config.params, seed_data);
    let randoms = CoxMock::generate_randoms(n_random, config.params.box_size, seed_rand);

    let data_tree = PointTree::build(mock.positions.clone());

    let dd_dists = query_knn(
        &data_tree,
        &mock.positions,
        config.k_max,
        config.box_size,
        true,
    );
    let dr_dists = query_knn(
        &data_tree,
        &randoms,
        config.k_max,
        config.box_size,
        false,
    );

    let rr_points: Vec<[f64; 3]> = randoms[..n_rr].to_vec();
    let rr_tree = PointTree::build(rr_points.clone());
    let rr_dists = query_knn(&rr_tree, &rr_points, config.k_max, config.box_size, true);

    // ξ(r) via Landy-Szalay
    let dd = LandySzalayKnn::pair_count_density(&dd_dists, &r_edges);
    let dr = LandySzalayKnn::pair_count_density(&dr_dists, &r_edges);
    let rr = LandySzalayKnn::pair_count_density(&rr_dists, &r_edges);
    let xi_est = LandySzalayKnn::estimate_xi_ls(&dd, &dr, &rr, n_data, n_rr);

    // CDFs on a dense r-grid for both DD and RR
    let k_vals = cdf_k_values(config.k_max);
    let cdf_r = cdf_r_grid(config.r_min, config.r_max, 100);
    let rr_cdfs = LandySzalayKnn::empirical_cdfs(&rr_dists, &k_vals, &cdf_r);
    let dd_cdfs = LandySzalayKnn::empirical_cdfs(&dd_dists, &k_vals, &cdf_r);

    MockResult {
        xi: xi_est.xi,
        rr_cdfs,
        dd_cdfs,
    }
}

/// Aggregate per-mock results into final statistics.
pub fn aggregate_mocks(config: &ValidationConfig, mocks: &[MockResult]) -> ValidationResult {
    let r_edges = linear_bins(config.r_min, config.r_max, config.n_bins);
    let r_centers: Vec<f64> = r_edges.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect();
    let xi_analytic: Vec<f64> = r_centers
        .iter()
        .map(|&r| config.params.xi_analytic(r))
        .collect();

    let n = mocks.len() as f64;

    // ξ(r) statistics
    let mean_xi: Vec<f64> = (0..config.n_bins)
        .map(|i| mocks.iter().map(|m| m.xi[i]).sum::<f64>() / n)
        .collect();
    let std_xi: Vec<f64> = (0..config.n_bins)
        .map(|i| {
            let m = mean_xi[i];
            let var =
                mocks.iter().map(|mr| (mr.xi[i] - m).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
            var.sqrt()
        })
        .collect();
    let stderr_xi: Vec<f64> = std_xi.iter().map(|s| s / n.sqrt()).collect();

    let chi2: f64 = (0..config.n_bins)
        .map(|i| {
            if stderr_xi[i] > 0.0 {
                ((mean_xi[i] - xi_analytic[i]) / stderr_xi[i]).powi(2)
            } else {
                0.0
            }
        })
        .sum();

    // CDF statistics across mocks
    let rr_cdfs: Vec<&KnnCdfs> = mocks.iter().map(|m| &m.rr_cdfs).collect();
    let dd_cdfs: Vec<&KnnCdfs> = mocks.iter().map(|m| &m.dd_cdfs).collect();
    let rr_summary = average_cdfs_refs(&rr_cdfs);
    let dd_summary = average_cdfs_refs(&dd_cdfs);

    // Legacy per-bin CDFs: interpolate from summary onto bin centers
    let last_knn_cdfs = (1..=config.k_max)
        .map(|k| {
            let cdf_r = &rr_summary.r_values;
            let ki = rr_summary
                .k_values
                .iter()
                .position(|&kv| kv == k);
            r_centers
                .iter()
                .map(|&r| {
                    if let Some(ki) = ki {
                        let idx = cdf_r
                            .iter()
                            .position(|&rv| rv >= r)
                            .unwrap_or(cdf_r.len() - 1);
                        rr_summary.cdf_mean[ki][idx]
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect::<Vec<Vec<f64>>>();

    // Dilution ladder (last mock only)
    let mut dilution_xi = Vec::new();
    let mut dilution_variance = Vec::new();
    if let Some(last) = mocks.last() {
        let last_idx = mocks.len() - 1;
        let seed_data = (last_idx * 2) as u64;
        let mock = CoxMock::generate(&config.params, seed_data);

        let ladder = DilutionLadder::build(
            mock.positions.len(),
            config.max_dilution_level,
            seed_data + 999,
        );

        for level in &ladder.levels {
            let mut level_xi_estimates: Vec<Vec<f64>> = Vec::new();

            for subsample_indices in &level.subsamples {
                let sub_pos: Vec<[f64; 3]> = subsample_indices
                    .iter()
                    .map(|&i| mock.positions[i])
                    .collect();

                let n_sub_rand = sub_pos.len() * config.random_ratio;
                let sub_randoms = CoxMock::generate_randoms(
                    n_sub_rand,
                    config.params.box_size,
                    seed_data + level.level as u64 * 1000,
                );
                let n_sub_rr = sub_pos.len();
                let sub_rr: Vec<[f64; 3]> = sub_randoms[..n_sub_rr].to_vec();

                let sub_data_tree = PointTree::build(sub_pos.clone());
                let sub_dd =
                    query_knn(&sub_data_tree, &sub_pos, config.k_max, config.box_size, true);
                let sub_dr = query_knn(
                    &sub_data_tree,
                    &sub_randoms,
                    config.k_max,
                    config.box_size,
                    false,
                );

                let sub_rr_tree = PointTree::build(sub_rr.clone());
                let sub_rr_dists =
                    query_knn(&sub_rr_tree, &sub_rr, config.k_max, config.box_size, true);

                let sub_dd_d = LandySzalayKnn::pair_count_density(&sub_dd, &r_edges);
                let sub_dr_d = LandySzalayKnn::pair_count_density(&sub_dr, &r_edges);
                let sub_rr_d = LandySzalayKnn::pair_count_density(&sub_rr_dists, &r_edges);
                let sub_xi = LandySzalayKnn::estimate_xi_ls(
                    &sub_dd_d,
                    &sub_dr_d,
                    &sub_rr_d,
                    sub_pos.len(),
                    n_sub_rr,
                );

                level_xi_estimates.push(sub_xi.xi);
            }

            let n_sub = level_xi_estimates.len() as f64;
            let mean_sub: Vec<f64> = (0..config.n_bins)
                .map(|i| level_xi_estimates.iter().map(|xi| xi[i]).sum::<f64>() / n_sub)
                .collect();
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

            dilution_xi.push(mean_sub);
            dilution_variance.push(var_sub);
        }
        let _ = last; // suppress unused warning
    }

    let dilution_r_char: Vec<f64> = (0..=config.max_dilution_level)
        .map(|l| DilutionLadder::r_char(config.k_max, 8usize.pow(l as u32), config.params.nbar()))
        .collect();

    ValidationResult {
        r_centers,
        xi_analytic,
        mean_xi,
        std_xi,
        stderr_xi,
        chi2,
        chi2_per_dof: chi2 / config.n_bins as f64,
        n_mocks: mocks.len(),
        cdf_r_grid: rr_summary.r_values.clone(),
        cdf_k_values: rr_summary.k_values.clone(),
        cdf_rr_mean: rr_summary.cdf_mean,
        cdf_rr_std: rr_summary.cdf_std,
        cdf_dd_mean: dd_summary.cdf_mean,
        cdf_dd_std: dd_summary.cdf_std,
        knn_cdfs: last_knn_cdfs,
        dilution_xi,
        dilution_variance,
        dilution_r_char,
    }
}

/// Like `average_cdfs` but takes references (avoids cloning per-mock CDFs).
fn average_cdfs_refs(cdf_subs: &[&KnnCdfs]) -> KnnCdfSummary {
    assert!(!cdf_subs.is_empty());
    let first = cdf_subs[0];
    let n_k = first.k_values.len();
    let n_r = first.r_values.len();
    let n_s = cdf_subs.len();
    let n_sf = n_s as f64;

    let mut cdf_mean = vec![vec![0.0; n_r]; n_k];
    let mut cdf_std = vec![vec![0.0; n_r]; n_k];

    for ki in 0..n_k {
        for ri in 0..n_r {
            let sum: f64 = cdf_subs.iter().map(|c| c.cdf_values[ki][ri]).sum();
            cdf_mean[ki][ri] = sum / n_sf;
        }
        if n_s > 1 {
            for ri in 0..n_r {
                let m = cdf_mean[ki][ri];
                let var: f64 = cdf_subs
                    .iter()
                    .map(|c| (c.cdf_values[ki][ri] - m).powi(2))
                    .sum::<f64>()
                    / (n_sf - 1.0);
                cdf_std[ki][ri] = var.sqrt();
            }
        }
    }

    KnnCdfSummary {
        r_values: first.r_values.clone(),
        k_values: first.k_values.clone(),
        cdf_mean,
        cdf_std,
        n_subsamples: n_s,
    }
}

/// Convenience wrapper: run all mocks and aggregate.
pub fn run_validation(config: &ValidationConfig) -> ValidationResult {
    let mocks: Vec<MockResult> = (0..config.n_mocks)
        .map(|i| run_single_mock(config, i))
        .collect();
    aggregate_mocks(config, &mocks)
}

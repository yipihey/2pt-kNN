//! WebAssembly entry points for the twopoint pipeline.
//!
//! Exposes the CoxMock validation + plotting pipeline to JavaScript via
//! wasm-bindgen. Supports both CPU and GPU (WebGPU) backends.

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

use crate::mock::CoxMockParams;
use crate::plotting;
use crate::validation::{BruteForceKnnBackend, ValidationConfig, ValidationResult, run_validation};

/// Configuration passed from JavaScript.
#[derive(Deserialize)]
pub struct WasmConfig {
    pub n_mocks: usize,
    pub n_points: usize,
    pub n_lines: usize,
    pub line_length: f64,
    pub box_size: f64,
    pub k_max: usize,
    pub n_bins: usize,
    pub r_min: f64,
    pub r_max: f64,
    pub random_ratio: usize,
    pub max_dilution_level: usize,
    pub use_gpu: bool,
}

/// Result returned to JavaScript.
#[derive(Serialize)]
pub struct WasmResult {
    /// SVG: ξ(r) vs analytic with error bars
    pub svg_xi: String,
    /// SVG: r²ξ(r) comparison
    pub svg_r2xi: String,
    /// SVG: kNN-CDF (unique to kNN — pair counting can't produce this)
    pub svg_cdf: String,
    /// SVG: Dilution ladder variance (model-free, unique to kNN)
    pub svg_dilution: String,
    /// TSV data for download
    pub tsv: String,
    /// Summary statistics as JSON
    pub stats_json: String,
}

#[wasm_bindgen]
pub async fn run_validation_wasm(config_js: JsValue) -> Result<JsValue, JsError> {
    // Parse config from JavaScript
    let config: WasmConfig =
        serde_wasm_bindgen::from_value(config_js).map_err(|e| JsError::new(&e.to_string()))?;

    let params = CoxMockParams {
        box_size: config.box_size,
        n_lines: config.n_lines,
        line_length: config.line_length,
        n_points: config.n_points,
    };

    let val_config = ValidationConfig {
        n_mocks: config.n_mocks,
        k_max: config.k_max,
        n_bins: config.n_bins,
        r_min: config.r_min,
        r_max: config.r_max,
        random_ratio: config.random_ratio,
        params,
        max_dilution_level: config.max_dilution_level,
        box_size: None,
    };

    // Run validation with the appropriate backend
    let result: ValidationResult;

    #[cfg(feature = "gpu")]
    {
        if config.use_gpu {
            if let Some(gpu) = crate::gpu::GpuKnn::new().await {
                result = run_validation_gpu(&val_config, &gpu).await;
            } else {
                // GPU not available, fall back to CPU
                let backend = BruteForceKnnBackend;
                result = run_validation(&val_config, &backend);
            }
        } else {
            let backend = BruteForceKnnBackend;
            result = run_validation(&val_config, &backend);
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        let backend = BruteForceKnnBackend;
        result = run_validation(&val_config, &backend);
    }

    // Generate SVG plots
    let svg_xi = plotting::render_xi_plot(&result);
    let svg_r2xi = plotting::render_r2xi_plot(&result);
    let svg_cdf = plotting::render_cdf_plot(&result);
    let svg_dilution = plotting::render_dilution_plot(&result);

    // Generate TSV
    let mut tsv = String::from("# r\txi_analytic\txi_mean\txi_std\txi_stderr\n");
    for i in 0..result.r_centers.len() {
        tsv.push_str(&format!(
            "{:.2}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\n",
            result.r_centers[i],
            result.xi_analytic[i],
            result.mean_xi[i],
            result.std_xi[i],
            result.stderr_xi[i]
        ));
    }

    // Summary stats JSON
    let stats = serde_json::json!({
        "chi2": result.chi2,
        "chi2_per_dof": result.chi2_per_dof,
        "n_mocks": result.n_mocks,
        "n_bins": result.r_centers.len(),
        "dilution_r_char": result.dilution_r_char,
    });

    let wasm_result = WasmResult {
        svg_xi,
        svg_r2xi,
        svg_cdf,
        svg_dilution,
        tsv,
        stats_json: stats.to_string(),
    };

    serde_wasm_bindgen::to_value(&wasm_result).map_err(|e| JsError::new(&e.to_string()))
}

/// GPU validation path — runs kNN queries asynchronously on WebGPU.
#[cfg(feature = "gpu")]
async fn run_validation_gpu(
    config: &ValidationConfig,
    gpu: &crate::gpu::GpuKnn,
) -> ValidationResult {
    // For the GPU path, we implement the same logic as run_validation()
    // but call gpu.query_distances().await instead of the sync backend.
    // This is necessary because WASM can't block on futures.
    use crate::estimator::{LandySzalayKnn, linear_bins};
    use crate::ladder::DilutionLadder;
    use crate::mock::CoxMock;

    let r_edges = linear_bins(config.r_min, config.r_max, config.n_bins);
    let r_centers: Vec<f64> = r_edges.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect();
    let xi_analytic: Vec<f64> = r_centers.iter().map(|&r| config.params.xi_analytic(r)).collect();

    let n_random = config.params.n_points * config.random_ratio;
    let mut all_xi: Vec<Vec<f64>> = Vec::with_capacity(config.n_mocks);
    let mut last_knn_cdfs = Vec::new();
    let mut last_dilution_xi = Vec::new();
    let mut last_dilution_variance = Vec::new();

    for mock_idx in 0..config.n_mocks {
        let seed_data = (mock_idx * 2) as u64;
        let seed_rand = (mock_idx * 2 + 1) as u64 + 1_000_000;

        let mock = CoxMock::generate(&config.params, seed_data);
        let randoms = CoxMock::generate_randoms(n_random, config.params.box_size, seed_rand);

        // GPU kNN queries
        let dd_dists = gpu
            .query_distances(&mock.positions, &mock.positions, config.k_max)
            .await;
        let dr_dists = gpu
            .query_distances(&mock.positions, &randoms, config.k_max)
            .await;

        let dd = LandySzalayKnn::pair_count_density(&dd_dists, &r_edges);
        let dr = LandySzalayKnn::pair_count_density(&dr_dists, &r_edges);
        let xi_est = LandySzalayKnn::estimate_xi_dp(&dd, &dr);

        all_xi.push(xi_est.xi.clone());

        if mock_idx == config.n_mocks - 1 {
            last_knn_cdfs = (1..=config.k_max)
                .map(|k| LandySzalayKnn::empirical_cdf(&dd_dists, k, &r_centers))
                .collect();

            let ladder = DilutionLadder::build(
                mock.positions.len(),
                config.max_dilution_level,
                seed_data + 999,
            );

            for level in &ladder.levels {
                let mut level_xi_estimates = Vec::new();
                for subsample_indices in &level.subsamples {
                    let sub_pos: Vec<[f64; 3]> =
                        subsample_indices.iter().map(|&i| mock.positions[i]).collect();
                    let n_sub_rand = sub_pos.len() * config.random_ratio;
                    let sub_rand = CoxMock::generate_randoms(
                        n_sub_rand,
                        config.params.box_size,
                        seed_data + level.level as u64 * 1000,
                    );

                    let sub_dd = gpu.query_distances(&sub_pos, &sub_pos, config.k_max).await;
                    let sub_dr = gpu.query_distances(&sub_pos, &sub_rand, config.k_max).await;

                    let sub_dd_d = LandySzalayKnn::pair_count_density(&sub_dd, &r_edges);
                    let sub_dr_d = LandySzalayKnn::pair_count_density(&sub_dr, &r_edges);
                    let sub_xi = LandySzalayKnn::estimate_xi_dp(&sub_dd_d, &sub_dr_d);
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

                last_dilution_xi.push(mean_sub);
                last_dilution_variance.push(var_sub);
            }
        }
    }

    let n = config.n_mocks as f64;
    let mean_xi: Vec<f64> = (0..config.n_bins)
        .map(|i| all_xi.iter().map(|xi| xi[i]).sum::<f64>() / n)
        .collect();
    let std_xi: Vec<f64> = (0..config.n_bins)
        .map(|i| {
            let m = mean_xi[i];
            (all_xi.iter().map(|xi| (xi[i] - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
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

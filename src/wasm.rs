//! WebAssembly entry points for the twopoint pipeline.
//!
//! Exposes the CoxMock validation pipeline to JavaScript via wasm-bindgen.
//! Returns raw data arrays — the web frontend renders plots client-side
//! for instant interactivity.
//!
//! The mock loop runs in WASM with async yields between mocks so the
//! browser can update the progress bar.

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

use crate::mock::CoxMockParams;
use crate::validation::{ValidationConfig, run_single_mock, aggregate_mocks};

fn log(msg: &str) {
    web_sys::console::log_1(&JsValue::from_str(msg));
}

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

/// Result returned to JavaScript — raw data arrays for client-side plotting.
#[derive(Serialize)]
pub struct WasmResult {
    pub r_centers: Vec<f64>,
    pub xi_analytic: Vec<f64>,
    pub mean_xi: Vec<f64>,
    pub std_xi: Vec<f64>,
    pub stderr_xi: Vec<f64>,
    /// kNN-CDFs (legacy, per-bin): cdfs[k][r_idx]
    pub knn_cdfs: Vec<Vec<f64>>,
    /// kNN-PDFs (legacy, per-bin): pdfs[k][r_idx]
    pub knn_pdfs: Vec<Vec<f64>>,
    /// Dense r-grid for CDF/PDF evaluation
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
    pub dilution_xi: Vec<Vec<f64>>,
    pub dilution_stderr: Vec<Vec<f64>>,
    pub dilution_r_char: Vec<f64>,
    pub chi2: f64,
    pub chi2_per_dof: f64,
    pub n_mocks: usize,
}

/// Yield to the JS event loop so the browser can repaint (progress bar).
async fn yield_to_js() {
    let promise = js_sys::Promise::resolve(&JsValue::NULL);
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}

#[wasm_bindgen]
pub async fn run_validation_wasm(
    config_js: JsValue,
    on_progress: Option<js_sys::Function>,
) -> Result<JsValue, JsError> {
    let config: WasmConfig =
        serde_wasm_bindgen::from_value(config_js).map_err(|e| JsError::new(&e.to_string()))?;

    log(&format!(
        "[wasm] Starting: {} points, {} mocks, k_max={}",
        config.n_points, config.n_mocks, config.k_max
    ));

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
        box_size: Some(config.box_size),
    };

    // Run mocks one at a time, yielding between each for progress updates
    let mut mock_results = Vec::with_capacity(config.n_mocks);
    for mock_idx in 0..config.n_mocks {
        let mr = run_single_mock(&val_config, mock_idx);
        mock_results.push(mr);

        // Report progress to JS
        if let Some(ref cb) = on_progress {
            let _ = cb.call2(
                &JsValue::NULL,
                &JsValue::from((mock_idx + 1) as f64),
                &JsValue::from(config.n_mocks as f64),
            );
        }
        // Yield to event loop so browser can repaint
        yield_to_js().await;
    }

    log("[wasm] Aggregating results...");
    let result = aggregate_mocks(&val_config, &mock_results);
    log(&format!(
        "[wasm] Done. chi2/dof = {:.2}",
        result.chi2_per_dof
    ));

    // Compute PDFs (dCDF/dr) from the dense CDF grid
    let _cdf_r = &result.cdf_r_grid;

    // Legacy per-bin PDFs
    let r_legacy = &result.r_centers;
    let nr_legacy = r_legacy.len();
    let knn_pdfs: Vec<Vec<f64>> = result
        .knn_cdfs
        .iter()
        .map(|cdf| {
            let mut pdf = vec![0.0; nr_legacy];
            if nr_legacy >= 3 {
                pdf[0] = (cdf[1] - cdf[0]) / (r_legacy[1] - r_legacy[0]);
                for i in 1..nr_legacy - 1 {
                    pdf[i] =
                        (cdf[i + 1] - cdf[i - 1]) / (r_legacy[i + 1] - r_legacy[i - 1]);
                }
                pdf[nr_legacy - 1] = (cdf[nr_legacy - 1] - cdf[nr_legacy - 2])
                    / (r_legacy[nr_legacy - 1] - r_legacy[nr_legacy - 2]);
            }
            pdf
        })
        .collect();

    let dilution_stderr: Vec<Vec<f64>> = result
        .dilution_variance
        .iter()
        .map(|var| var.iter().map(|v| v.sqrt()).collect())
        .collect();

    let wasm_result = WasmResult {
        r_centers: result.r_centers,
        xi_analytic: result.xi_analytic,
        mean_xi: result.mean_xi,
        std_xi: result.std_xi,
        stderr_xi: result.stderr_xi,
        knn_cdfs: result.knn_cdfs,
        knn_pdfs,
        cdf_r_grid: result.cdf_r_grid,
        cdf_k_values: result.cdf_k_values,
        cdf_rr_mean: result.cdf_rr_mean,
        cdf_rr_std: result.cdf_rr_std,
        cdf_dd_mean: result.cdf_dd_mean,
        cdf_dd_std: result.cdf_dd_std,
        dilution_xi: result.dilution_xi,
        dilution_stderr,
        dilution_r_char: result.dilution_r_char,
        chi2: result.chi2,
        chi2_per_dof: result.chi2_per_dof,
        n_mocks: result.n_mocks,
    };

    serde_wasm_bindgen::to_value(&wasm_result).map_err(|e| JsError::new(&e.to_string()))
}

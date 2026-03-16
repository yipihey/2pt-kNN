//! Bridge to Corrfunc via a Python subprocess.
//!
//! Writes positions as raw little-endian f64 bytes, passes a JSON config
//! to `scripts/corrfunc_xi.py`, reads back the JSON result. Results are
//! cached on disk so Corrfunc only runs once per unique configuration.
//!
//! Both kNN and Corrfunc now use the full Landy-Szalay estimator with
//! explicit DD, DR, RR pair counts — no analytic RR shortcuts.

use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

/// Result of a Corrfunc xi(r) computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrfuncResult {
    pub r_avg: Vec<f64>,
    pub xi: Vec<f64>,
    pub npairs_dd: Vec<u64>,
    pub npairs_dr: Vec<u64>,
    pub npairs_rr: Vec<u64>,
    pub wall_time_secs: f64,
}

/// Errors from the Corrfunc bridge.
#[derive(Debug, thiserror::Error)]
pub enum CorrfuncError {
    #[error("python3 not found on PATH")]
    PythonNotFound,
    #[error("Corrfunc not installed (pip install Corrfunc)")]
    CorrfuncNotInstalled,
    #[error("Corrfunc process failed (exit {code}): {stderr}")]
    ProcessFailed { stderr: String, code: i32 },
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON parse error: {0}")]
    Parse(#[from] serde_json::Error),
}

/// JSON config written for the Python script.
#[derive(Serialize)]
struct ScriptConfig {
    data_file: String,
    n_data: usize,
    randoms_file: String,
    n_randoms: usize,
    box_size: f64,
    r_edges: Vec<f64>,
    output_file: String,
    nthreads: usize,
}

/// Manages Corrfunc computation and caching.
pub struct CorrfuncRunner {
    cache_dir: PathBuf,
    python: String,
}

/// Find a Python interpreter that has Corrfunc installed.
///
/// If `hint` is provided and works, use it. Otherwise try `python3`, then
/// common Homebrew versioned paths (`python3.11`, `python3.12`, ...).
pub fn find_python(hint: Option<&str>) -> Result<String, CorrfuncError> {
    let candidates: Vec<String> = {
        let mut v: Vec<String> = Vec::new();
        if let Some(h) = hint {
            v.push(h.to_string());
        }
        v.push("python3".into());
        // Homebrew versioned interpreters (most likely to have Corrfunc)
        for minor in (9..=14).rev() {
            v.push(format!("python3.{}", minor));
            v.push(format!("/opt/homebrew/opt/python@3.{}/bin/python3.{}", minor, minor));
            v.push(format!("/usr/local/opt/python@3.{}/bin/python3.{}", minor, minor));
        }
        v
    };

    for py in &candidates {
        let ok = Command::new(py)
            .args(["-c", "import Corrfunc"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        if let Ok(status) = ok {
            if status.success() {
                return Ok(py.clone());
            }
        }
    }

    Err(CorrfuncError::CorrfuncNotInstalled)
}

/// Write positions as raw little-endian f64 bytes to a file.
fn write_positions(path: &Path, positions: &[[f64; 3]]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    for p in positions {
        for &coord in p {
            f.write_all(&coord.to_le_bytes())?;
        }
    }
    Ok(())
}

impl CorrfuncRunner {
    /// Create a new runner. Discovers a Python with Corrfunc installed.
    ///
    /// `python_hint` is an optional user-supplied interpreter path (e.g.
    /// from `--python`); if it has Corrfunc we use it directly.
    pub fn new(output_dir: &Path, python_hint: Option<&str>) -> Result<Self, CorrfuncError> {
        let cache_dir = output_dir.join(".corrfunc_cache");
        std::fs::create_dir_all(&cache_dir)?;
        let python = find_python(python_hint)?;
        Ok(Self { cache_dir, python })
    }

    /// The Python interpreter path that will be used.
    pub fn python(&self) -> &str {
        &self.python
    }

    /// Build a deterministic cache key from mock parameters.
    pub fn cache_key(
        preset: &str,
        seed: u64,
        r_min: f64,
        r_max: f64,
        n_bins: usize,
    ) -> String {
        format!(
            "{}_{}_{}_{}_{}",
            preset, seed, r_min, r_max, n_bins
        )
    }

    /// Compute xi(r) via Corrfunc Landy-Szalay (DD, DR, RR), using cache when available.
    pub fn compute_xi(
        &self,
        data_positions: &[[f64; 3]],
        random_positions: &[[f64; 3]],
        box_size: f64,
        r_edges: &[f64],
        nthreads: usize,
        cache_key: &str,
    ) -> Result<CorrfuncResult, CorrfuncError> {
        // Check cache
        let cache_path = self.cache_dir.join(format!("{}.json", cache_key));
        if cache_path.exists() {
            let data = std::fs::read_to_string(&cache_path)?;
            let result: CorrfuncResult = serde_json::from_str(&data)?;
            return Ok(result);
        }

        // Write data positions
        let data_path = self.cache_dir.join(format!("{}_data.bin", cache_key));
        write_positions(&data_path, data_positions)?;

        // Write random positions
        let rand_path = self.cache_dir.join(format!("{}_rand.bin", cache_key));
        write_positions(&rand_path, random_positions)?;

        // Write config JSON
        let result_path = self.cache_dir.join(format!("{}_result.json", cache_key));
        let config = ScriptConfig {
            data_file: data_path.to_string_lossy().into_owned(),
            n_data: data_positions.len(),
            randoms_file: rand_path.to_string_lossy().into_owned(),
            n_randoms: random_positions.len(),
            box_size,
            r_edges: r_edges.to_vec(),
            output_file: result_path.to_string_lossy().into_owned(),
            nthreads,
        };
        let config_path = self.cache_dir.join(format!("{}_config.json", cache_key));
        let config_json = serde_json::to_string(&config)?;
        std::fs::write(&config_path, &config_json)?;

        // Locate the Python script
        let script_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("scripts")
            .join("corrfunc_xi.py");

        // Run the script
        let _start = Instant::now();
        let output = Command::new(&self.python)
            .arg(&script_path)
            .arg(&config_path)
            .output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
            let code = output.status.code().unwrap_or(-1);

            if stderr.contains("No module named") && stderr.contains("Corrfunc") {
                return Err(CorrfuncError::CorrfuncNotInstalled);
            }
            return Err(CorrfuncError::ProcessFailed { stderr, code });
        }

        // Read result
        let result_data = std::fs::read_to_string(&result_path)?;
        let result: CorrfuncResult = serde_json::from_str(&result_data)?;

        // Cache the result
        std::fs::copy(&result_path, &cache_path)?;

        // Clean up temporary files
        std::fs::remove_file(&data_path).ok();
        std::fs::remove_file(&rand_path).ok();
        std::fs::remove_file(&config_path).ok();
        std::fs::remove_file(&result_path).ok();

        Ok(result)
    }
}

//! Head-to-head benchmark: Morton & LCA estimators vs Corrfunc.
//!
//! Generates CoxMock data at several sizes, runs each estimator, and
//! prints a wall-clock comparison table plus accuracy vs analytic ξ.
//!
//! Usage:
//!   cargo run --release --bin bench-vs-corrfunc [--python python3]

use clap::Parser;
use std::time::Instant;

use twopoint::corrfunc::{CorrfuncRunner, CorrfuncError};
use twopoint::lca_tree::{LcaTree, randoms as lca_randoms, estimator as lca_estimator};
use twopoint::mock::{CoxMock, CoxMockParams};
use twopoint::morton::MortonConfig;
use twopoint::xi_morton::{self, MortonXiConfig};
use twopoint::xi_morton::cell_pairs;

#[derive(Parser, Debug)]
#[command(name = "bench-vs-corrfunc")]
#[command(about = "Benchmark Morton/LCA estimators vs Corrfunc pair counting")]
struct Args {
    /// Python interpreter (must have Corrfunc installed)
    #[arg(long)]
    python: Option<String>,

    /// Number of threads for Corrfunc
    #[arg(long, default_value_t = 4)]
    nthreads: usize,

    /// Sizes to benchmark (comma-separated N_data values)
    #[arg(long, default_value = "10000,50000,200000,500000,1000000")]
    sizes: String,

    /// Number of radial bins for Corrfunc and LCA
    #[arg(long, default_value_t = 15)]
    n_bins: usize,

    /// Skip Corrfunc (just compare Morton vs LCA)
    #[arg(long)]
    no_corrfunc: bool,
}

/// Single-size benchmark result.
struct BenchResult {
    n_data: usize,
    n_rand: usize,
    morton_secs: f64,
    lca_secs: f64,
    corrfunc_secs: Option<f64>,
    /// RMS fractional error vs analytic ξ for each estimator.
    morton_rms_err: f64,
    lca_rms_err: f64,
    corrfunc_rms_err: Option<f64>,
}

fn main() {
    let args = Args::parse();
    let sizes: Vec<usize> = args
        .sizes
        .split(',')
        .map(|s| s.trim().parse().expect("bad size"))
        .collect();

    // Try to set up Corrfunc runner.
    let corrfunc_runner = if args.no_corrfunc {
        None
    } else {
        let tmp = std::env::temp_dir().join("bench_vs_corrfunc");
        std::fs::create_dir_all(&tmp).ok();
        match CorrfuncRunner::new(&tmp, args.python.as_deref()) {
            Ok(r) => {
                eprintln!("Corrfunc: using {}", r.python());
                Some(r)
            }
            Err(CorrfuncError::CorrfuncNotInstalled) | Err(CorrfuncError::PythonNotFound) => {
                eprintln!("WARNING: Corrfunc not available — will skip Corrfunc columns");
                None
            }
            Err(e) => {
                eprintln!("WARNING: Corrfunc init failed ({e}) — skipping");
                None
            }
        }
    };

    let mut results = Vec::new();

    for &n_data in &sizes {
        eprintln!("\n=== N_data = {} ===", n_data);
        let result = run_benchmark(n_data, &args, corrfunc_runner.as_ref());
        results.push(result);
    }

    // Print summary table.
    println!();
    println!("╔═══════════╤═══════════╤══════════════════════╤══════════════════════╤══════════════════════╗");
    println!("║  N_data   │  N_rand   │     Morton (grid)    │    LCA (tree)        │   Corrfunc (pairs)   ║");
    println!("║           │           │  time(s)  RMS err    │  time(s)  RMS err    │  time(s)  RMS err    ║");
    println!("╠═══════════╪═══════════╪══════════════════════╪══════════════════════╪══════════════════════╣");
    for r in &results {
        let cf_time = r.corrfunc_secs.map_or("   ---".to_string(), |t| format!("{:7.3}", t));
        let cf_err = r.corrfunc_rms_err.map_or("   ---".to_string(), |e| format!("{:7.4}", e));
        println!(
            "║ {:>9} │ {:>9} │ {:7.3}   {:7.4}    │ {:7.3}   {:7.4}    │ {}   {}    ║",
            r.n_data,
            r.n_rand,
            r.morton_secs,
            r.morton_rms_err,
            r.lca_secs,
            r.lca_rms_err,
            cf_time,
            cf_err,
        );
    }
    println!("╚═══════════╧═══════════╧══════════════════════╧══════════════════════╧══════════════════════╝");

    // Scaling summary.
    println!("\nScaling summary (time relative to smallest N):");
    if let Some(first) = results.first() {
        let n0 = first.n_data as f64;
        let m0 = first.morton_secs;
        let l0 = first.lca_secs;
        let c0 = first.corrfunc_secs;

        for r in &results {
            let n_ratio = r.n_data as f64 / n0;
            let m_ratio = r.morton_secs / m0;
            let l_ratio = r.lca_secs / l0;
            let c_ratio = match (r.corrfunc_secs, c0) {
                (Some(t), Some(t0)) => format!("{:6.1}x", t / t0),
                _ => "   ---".to_string(),
            };
            println!(
                "  N={:>9}  ({:5.1}x):  Morton {:6.1}x   LCA {:6.1}x   Corrfunc {}",
                r.n_data, n_ratio, m_ratio, l_ratio, c_ratio,
            );
        }
    }

    // Speedup summary.
    println!("\nSpeedup (Corrfunc / our estimator):");
    for r in &results {
        if let Some(cf) = r.corrfunc_secs {
            println!(
                "  N={:>9}:  Morton {:5.1}x faster   LCA {:5.1}x faster",
                r.n_data,
                cf / r.morton_secs,
                cf / r.lca_secs,
            );
        }
    }
}

fn run_benchmark(
    n_data: usize,
    args: &Args,
    corrfunc: Option<&CorrfuncRunner>,
) -> BenchResult {
    let box_size = 1000.0;
    let line_length = 400.0;
    let n_lines = n_data / 10; // keep clustering strength roughly constant
    let n_rand = n_data * 3;   // 3x randoms

    let params = CoxMockParams {
        box_size,
        n_lines: n_lines.max(100),
        line_length,
        n_points: n_data,
    };

    let seed = 42u64;
    eprintln!("  Generating mock: {} data, {} rand ...", n_data, n_rand);
    let mock = CoxMock::generate(&params, seed);
    let data = &mock.positions;
    let randoms = CoxMock::generate_randoms(n_rand, box_size, seed + 1000);

    // Bin edges for Corrfunc and LCA (same bins for fair comparison).
    let r_min = 5.0;
    let r_max = 200.0;
    let bin_edges = cell_pairs::log_bin_edges(r_min, r_max, args.n_bins);

    // -----------------------------------------------------------------------
    // Morton estimator
    // -----------------------------------------------------------------------
    eprintln!("  Running Morton estimator ...");
    let l_max = MortonXiConfig::auto_l_max(box_size, n_data, 0.5);
    let morton_config = MortonXiConfig {
        morton_config: MortonConfig::new(box_size, true),
        l_min: 1,
        l_max,
        n_offsets: 8,
        seed: 42,
    };

    let t0 = Instant::now();
    let morton_est = xi_morton::estimate_xi_with_offsets(data, &randoms, &morton_config);
    let morton_secs = t0.elapsed().as_secs_f64();
    eprintln!("    Morton: {:.3}s, {} bins", morton_secs, morton_est.r.len());

    // -----------------------------------------------------------------------
    // LCA estimator
    // -----------------------------------------------------------------------
    eprintln!("  Running LCA estimator ...");
    let t0 = Instant::now();
    let data_weights = vec![1.0; n_data];
    let rand_weights = vec![1.0; n_rand];
    let mut tree = LcaTree::build(data, &data_weights, Some(32), Some(box_size));
    lca_randoms::insert_randoms(&mut tree, &randoms, &rand_weights);

    let lca_config = lca_estimator::LcaEstimatorConfig {
        bin_edges: bin_edges.clone(),
        mc_samples: 512,
        box_size: Some(box_size),
        seed: 12345,
    };
    let lca_est = lca_estimator::estimate_xi(&tree, &lca_config);
    let lca_secs = t0.elapsed().as_secs_f64();
    eprintln!("    LCA: {:.3}s, {} bins", lca_secs, lca_est.r.len());

    // -----------------------------------------------------------------------
    // Corrfunc
    // -----------------------------------------------------------------------
    let (corrfunc_secs, corrfunc_xi_at_r) = if let Some(cf) = corrfunc {
        eprintln!("  Running Corrfunc ...");
        let cache_key = format!("bench_{}", n_data);
        let t0 = Instant::now();
        match cf.compute_xi(data, &randoms, box_size, &bin_edges, args.nthreads, &cache_key) {
            Ok(result) => {
                let secs = t0.elapsed().as_secs_f64();
                eprintln!("    Corrfunc: {:.3}s", secs);
                let xi_at_r: Vec<(f64, f64)> = result
                    .r_avg
                    .iter()
                    .zip(result.xi.iter())
                    .map(|(&r, &x)| (r, x))
                    .collect();
                (Some(secs), Some(xi_at_r))
            }
            Err(e) => {
                eprintln!("    Corrfunc failed: {e}");
                (None, None)
            }
        }
    } else {
        (None, None)
    };

    // -----------------------------------------------------------------------
    // Accuracy: RMS fractional error vs analytic ξ
    // -----------------------------------------------------------------------
    let morton_rms_err = rms_error(&morton_est.r, &morton_est.xi, &params);
    let lca_rms_err = rms_error(&lca_est.r, &lca_est.xi, &params);
    let corrfunc_rms_err = corrfunc_xi_at_r.as_ref().map(|xr| {
        let (r, xi): (Vec<f64>, Vec<f64>) = xr.iter().cloned().unzip();
        rms_error(&r, &xi, &params)
    });

    // Print per-bin comparison.
    eprintln!("  Per-bin comparison (analytic | Morton | LCA | Corrfunc):");
    for i in 0..lca_est.r.len().min(args.n_bins) {
        let r = lca_est.r[i];
        let xi_true = params.xi_analytic(r);
        let xi_lca = lca_est.xi[i];
        let xi_cf = corrfunc_xi_at_r
            .as_ref()
            .and_then(|v| v.get(i))
            .map(|(_r, x)| *x);

        // Find nearest Morton bin.
        let xi_morton = nearest_value(&morton_est.r, &morton_est.xi, r);

        let cf_str = xi_cf.map_or("     ---".to_string(), |x| format!("{:+8.5}", x));
        eprintln!(
            "    r={:6.1}:  true={:+8.5}  morton={:+8.5}  lca={:+8.5}  cf={}",
            r, xi_true, xi_morton, xi_lca, cf_str,
        );
    }

    BenchResult {
        n_data,
        n_rand,
        morton_secs,
        lca_secs,
        corrfunc_secs,
        morton_rms_err,
        lca_rms_err,
        corrfunc_rms_err,
    }
}

/// RMS fractional error: sqrt(mean((xi_est - xi_true)^2 / xi_true^2))
/// Only includes bins where xi_true > threshold to avoid noise domination.
fn rms_error(r: &[f64], xi: &[f64], params: &CoxMockParams) -> f64 {
    let threshold = 0.001; // skip bins with nearly zero true ξ
    let mut sum_sq = 0.0;
    let mut count = 0usize;
    for (ri, xi_est) in r.iter().zip(xi.iter()) {
        let xi_true = params.xi_analytic(*ri);
        if xi_true.abs() > threshold {
            let frac = (xi_est - xi_true) / xi_true;
            sum_sq += frac * frac;
            count += 1;
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        (sum_sq / count as f64).sqrt()
    }
}

/// Find the ξ value at the r_target closest to `target`.
fn nearest_value(r: &[f64], xi: &[f64], target: f64) -> f64 {
    r.iter()
        .zip(xi.iter())
        .min_by(|(a, _), (b, _)| {
            let da = (*a - target).abs();
            let db = (*b - target).abs();
            da.partial_cmp(&db).unwrap()
        })
        .map(|(_, x)| *x)
        .unwrap_or(0.0)
}

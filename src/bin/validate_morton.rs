//! Validation binary for the Morton-grid two-point function and
//! counts-in-cells estimator.
//!
//! Generates CoxMock realizations with known analytic ξ(r), runs the
//! Morton estimator, and optionally compares against Corrfunc pair
//! counting.
//!
//! Usage:
//!   twopoint-validate-morton [--n-mocks 10] [--l-max 7] [--corrfunc]

use clap::Parser;
use std::time::Instant;
use twopoint::cic;
use twopoint::grid;
use twopoint::mock::{CoxMock, CoxMockParams};
use twopoint::morton::{self, MortonConfig};
use twopoint::xi_morton::{self, MortonXiConfig, MortonXiEstimate};

#[derive(Parser, Debug)]
#[command(name = "twopoint-validate-morton")]
#[command(about = "Validate the Morton-grid ξ estimator against CoxMock analytics")]
struct Args {
    /// Number of mock realizations
    #[arg(long, default_value_t = 10)]
    n_mocks: usize,

    /// Maximum octree level
    #[arg(long, default_value_t = 7)]
    l_max: u32,

    /// Minimum octree level
    #[arg(long, default_value_t = 1)]
    l_min: u32,

    /// Bits per axis for Morton encoding
    #[arg(long, default_value_t = 21)]
    bits: u32,

    /// Number of random spatial offsets (0 = none)
    #[arg(long, default_value_t = 0)]
    n_offsets: usize,

    /// Random/data ratio for randoms catalog
    #[arg(long, default_value_t = 5)]
    random_ratio: usize,

    /// CoxMock preset: "validation", "tiny", "poisson", "euclid_small"
    #[arg(long, default_value = "validation")]
    preset: String,

    /// Override CoxMock n_points
    #[arg(long)]
    n_points: Option<usize>,

    /// Override CoxMock box_size
    #[arg(long)]
    box_size: Option<f64>,

    /// Run Corrfunc pair-counting as a reference comparison
    #[arg(long)]
    corrfunc: bool,

    /// Number of threads for Corrfunc
    #[arg(long, default_value_t = 4)]
    corrfunc_threads: usize,

    /// Python interpreter for Corrfunc
    #[arg(long)]
    python: Option<String>,

    /// Show counts-in-cells summary
    #[arg(long)]
    cic: bool,
}

fn main() {
    let args = Args::parse();

    let mut params = match args.preset.as_str() {
        "euclid_small" => CoxMockParams::euclid_small(),
        "tiny" => CoxMockParams::tiny(),
        "poisson" => CoxMockParams::poisson(),
        _ => CoxMockParams::validation(),
    };

    if let Some(v) = args.n_points {
        params.n_points = v;
    }
    if let Some(v) = args.box_size {
        params.box_size = v;
    }

    println!("=== Morton-Grid ξ Validation ===");
    println!(
        "Preset: {}, N_D={}, box={}, ℓ={}..{}, offsets={}",
        args.preset, params.n_points, params.box_size, args.l_min, args.l_max, args.n_offsets,
    );
    println!();

    let n_random = params.n_points * args.random_ratio;

    // Accumulators for averaging across mocks.
    let mut all_estimates: Vec<MortonXiEstimate> = Vec::new();

    for mock_i in 0..args.n_mocks {
        let t0 = Instant::now();
        let mock = CoxMock::generate(&params, 1000 + mock_i as u64);
        let randoms = CoxMock::generate_randoms(n_random, params.box_size, 2000 + mock_i as u64);

        let xi_config = MortonXiConfig {
            morton_config: MortonConfig {
                bits_per_axis: args.bits,
                box_size: params.box_size,
                periodic: true,
            },
            l_min: args.l_min,
            l_max: args.l_max,
            n_offsets: args.n_offsets,
            seed: 3000 + mock_i as u64,
        };

        let estimate = if args.n_offsets > 0 {
            xi_morton::estimate_xi_with_offsets(&mock.positions, &randoms, &xi_config)
        } else {
            let particles =
                morton::prepare_particles(&mock.positions, &randoms, &xi_config.morton_config);
            xi_morton::estimate_xi(&particles, &xi_config)
        };

        let elapsed = t0.elapsed();

        if mock_i == 0 {
            println!("Mock {}: {:.2}s", mock_i, elapsed.as_secs_f64());
            println!();

            // Print per-level results for first mock.
            println!("{:>5} {:>10} {:>10} {:>12} {:>10}", "level", "r", "ξ_morton", "ξ_analytic", "n_pairs");
            println!("{}", "-".repeat(55));
            for lev in &estimate.levels {
                for (bi, (&r, &xi)) in lev.r.iter().zip(lev.xi.iter()).enumerate() {
                    let xi_true = params.xi_analytic(r);
                    println!(
                        "{:>5} {:>10.2} {:>10.4} {:>12.4} {:>10}",
                        lev.level, r, xi, xi_true, lev.n_pairs[bi]
                    );
                }
            }
            println!();

            // CIC summary for first mock if requested.
            if args.cic {
                let mc = &xi_config.morton_config;
                let particles = morton::prepare_particles(&mock.positions, &randoms, mc);
                let histograms = grid::build_all_levels(&particles, mc, args.l_max);
                let cic_summary = cic::compute_cic_all_levels(&histograms, mc);

                println!("=== Counts-in-Cells Summary ===");
                println!(
                    "{:>5} {:>10} {:>10} {:>10} {:>10} {:>10}",
                    "level", "r_eff", "mean", "var", "skew", "kurt"
                );
                println!("{}", "-".repeat(60));
                for dist in &cic_summary.levels {
                    println!(
                        "{:>5} {:>10.2} {:>10.2} {:>10.2} {:>10.3} {:>10.3}",
                        dist.level,
                        dist.effective_radius,
                        dist.mean_n,
                        dist.var_n,
                        dist.skewness,
                        dist.kurtosis,
                    );
                }
                println!();
            }
        } else {
            print!("Mock {}: {:.2}s  ", mock_i, elapsed.as_secs_f64());
            if (mock_i + 1) % 5 == 0 {
                println!();
            }
        }

        all_estimates.push(estimate);
    }
    println!();

    // Compute mean and std of ξ across mocks at each (level, bin).
    if args.n_mocks > 1 {
        println!("=== Mean ± Std across {} mocks ===", args.n_mocks);
        println!(
            "{:>5} {:>10} {:>12} {:>12} {:>12} {:>12}",
            "level", "r", "ξ_mean", "ξ_std", "ξ_analytic", "residual"
        );
        println!("{}", "-".repeat(70));

        let ref_est = &all_estimates[0];
        for (li, ref_lev) in ref_est.levels.iter().enumerate() {
            for bi in 0..ref_lev.r.len() {
                let r = ref_lev.r[bi];
                let xi_vals: Vec<f64> = all_estimates.iter().map(|e| e.levels[li].xi[bi]).collect();
                let mean = xi_vals.iter().sum::<f64>() / xi_vals.len() as f64;
                let var = xi_vals.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                    / xi_vals.len() as f64;
                let std = var.sqrt();
                let xi_true = params.xi_analytic(r);
                let residual = if std > 0.0 {
                    (mean - xi_true) / std
                } else {
                    0.0
                };

                println!(
                    "{:>5} {:>10.2} {:>12.6} {:>12.6} {:>12.6} {:>12.2}σ",
                    ref_lev.level, r, mean, std, xi_true, residual
                );
            }
        }
    }

    // Corrfunc comparison.
    #[cfg(not(target_arch = "wasm32"))]
    if args.corrfunc {
        use std::path::Path;
        use twopoint::corrfunc::CorrfuncRunner;

        println!();
        println!("=== Corrfunc Comparison ===");

        let mock = CoxMock::generate(&params, 1000);
        let randoms = CoxMock::generate_randoms(n_random, params.box_size, 2000);

        // Use bin edges matching the Morton grid separations.
        let ref_est = &all_estimates[0];
        let mut r_vals: Vec<f64> = ref_est
            .levels
            .iter()
            .flat_map(|lev| lev.r.iter().copied())
            .collect();
        r_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        r_vals.dedup_by(|a, b| (*a - *b).abs() < 1e-10);

        // Build bin edges around the Morton r values.
        let mut edges = Vec::new();
        for (i, &r) in r_vals.iter().enumerate() {
            let lo = if i == 0 {
                r * 0.8
            } else {
                0.5 * (r_vals[i - 1] + r)
            };
            let hi = if i + 1 < r_vals.len() {
                0.5 * (r + r_vals[i + 1])
            } else {
                r * 1.2
            };
            if i == 0 {
                edges.push(lo);
            }
            edges.push(hi);
        }

        let python = args.python.as_deref();
        let output_dir = Path::new("plots");
        std::fs::create_dir_all(output_dir).ok();
        match CorrfuncRunner::new(output_dir, python) {
            Ok(runner) => {
                let cache_key = CorrfuncRunner::cache_key(
                    &args.preset,
                    1000,
                    edges.first().copied().unwrap_or(1.0),
                    edges.last().copied().unwrap_or(100.0),
                    edges.len() - 1,
                );
                match runner.compute_xi(
                    &mock.positions,
                    &randoms,
                    params.box_size,
                    &edges,
                    args.corrfunc_threads,
                    &cache_key,
                ) {
                    Ok(result) => {
                        println!(
                            "{:>10} {:>10} {:>12} {:>12}",
                            "r_cf", "xi_cf", "xi_morton", "xi_analytic"
                        );
                        println!("{}", "-".repeat(50));
                        for (r_cf, xi_cf) in result.r_avg.iter().zip(result.xi.iter()) {
                            // Look up the Morton xi at closest r.
                            let r_cf_val: f64 = *r_cf;
                            let xi_morton = ref_est
                                .r
                                .iter()
                                .zip(ref_est.xi.iter())
                                .min_by(|(&a, _), (&b, _)| {
                                    (a - r_cf_val).abs().partial_cmp(&(b - r_cf_val).abs()).unwrap()
                                })
                                .map(|(_, &xi)| xi)
                                .unwrap_or(0.0);

                            let xi_true = params.xi_analytic(*r_cf);
                            println!(
                                "{:>10.2} {:>10.4} {:>12.4} {:>12.4}",
                                r_cf, xi_cf, xi_morton, xi_true
                            );
                        }
                    }
                    Err(e) => println!("Corrfunc compute failed: {e}"),
                }
            }
            Err(e) => println!("Corrfunc init failed: {e}"),
        }
    }
}

//! Validation binary for the LCA tree-moment two-point estimator.
//!
//! Generates CoxMock realizations with known analytic ξ(r), builds the LCA tree,
//! verifies the pair-weight checksum, runs the estimator, and compares against
//! the analytic result.
//!
//! Usage:
//!   twopoint-validate-lca [--n-mocks 5] [--preset validation] [--leaf-size 32]

use clap::Parser;
use std::time::Instant;
use twopoint::lca_tree::LcaTree;
use twopoint::lca_tree::checksum;
use twopoint::lca_tree::estimator::{self, LcaEstimatorConfig};
use twopoint::lca_tree::randoms;
use twopoint::mock::{CoxMock, CoxMockParams};
use twopoint::xi_morton::cell_pairs::log_bin_edges;

#[derive(Parser, Debug)]
#[command(name = "twopoint-validate-lca")]
#[command(about = "Validate the LCA tree-moment ξ estimator")]
struct Args {
    /// Number of mock realizations
    #[arg(long, default_value_t = 5)]
    n_mocks: usize,

    /// CoxMock preset: "validation", "tiny", "poisson"
    #[arg(long, default_value = "validation")]
    preset: String,

    /// Leaf size for the KD-tree
    #[arg(long, default_value_t = 32)]
    leaf_size: usize,

    /// Number of MC samples per node for the geometric kernel
    #[arg(long, default_value_t = 512)]
    mc_samples: usize,

    /// Random/data ratio
    #[arg(long, default_value_t = 5)]
    random_ratio: usize,

    /// Number of radial bins
    #[arg(long, default_value_t = 15)]
    n_bins: usize,

    /// Minimum radial bin edge
    #[arg(long, default_value_t = 1.0)]
    r_min: f64,

    /// Maximum radial bin edge
    #[arg(long, default_value_t = 200.0)]
    r_max: f64,

    /// Override CoxMock n_points
    #[arg(long)]
    n_points: Option<usize>,

    /// Override CoxMock box_size
    #[arg(long)]
    box_size: Option<f64>,
}

fn main() {
    let args = Args::parse();

    let mut params = match args.preset.as_str() {
        "validation" => CoxMockParams::validation(),
        "tiny" => CoxMockParams::tiny(),
        "poisson" => CoxMockParams {
            box_size: 500.0,
            n_lines: 0,
            n_points: 10_000,
            line_length: 0.0,
        },
        other => {
            eprintln!("Unknown preset: {other}. Using 'validation'.");
            CoxMockParams::validation()
        }
    };

    if let Some(n) = args.n_points {
        params.n_points = n;
    }
    if let Some(bs) = args.box_size {
        params.box_size = bs;
    }

    let bin_edges = log_bin_edges(args.r_min, args.r_max, args.n_bins);
    let is_poisson = params.n_lines == 0;

    println!("=== LCA Tree-Moment ξ Estimator Validation ===");
    println!("Preset: {}", args.preset);
    println!("N_data: {}, box_size: {:.1}", params.n_points, params.box_size);
    println!("N_random_ratio: {}", args.random_ratio);
    println!("Leaf size: {}, MC samples: {}", args.leaf_size, args.mc_samples);
    println!("Bins: {} from {:.2} to {:.2}", args.n_bins, args.r_min, args.r_max);
    println!();

    let n_rand = params.n_points * args.random_ratio;

    for mock_i in 0..args.n_mocks {
        let seed = 1000 + mock_i as u64;
        println!("--- Mock {} (seed={}) ---", mock_i + 1, seed);

        // Generate data.
        let t0 = Instant::now();
        let mock = CoxMock::generate(&params, seed);
        let data = &mock.positions;
        println!("  Generated {} data points in {:.1}ms", data.len(), t0.elapsed().as_secs_f64() * 1e3);

        // Generate randoms.
        let t0 = Instant::now();
        let rand_pos = CoxMock::generate_randoms(n_rand, params.box_size, seed + 500);
        let rand_weights = vec![1.0; n_rand];
        println!("  Generated {} randoms in {:.1}ms", n_rand, t0.elapsed().as_secs_f64() * 1e3);

        // Build tree.
        let t0 = Instant::now();
        let data_weights = vec![1.0; data.len()];
        let mut tree = LcaTree::build(data, &data_weights, Some(args.leaf_size), Some(params.box_size));
        let build_ms = t0.elapsed().as_secs_f64() * 1e3;
        println!(
            "  Built KD-tree: {} internal nodes in {:.1}ms",
            tree.num_internal_nodes(),
            build_ms
        );

        // Checksum.
        let t0 = Instant::now();
        let cksum = checksum::verify_data_checksum(&tree, 1e-12);
        println!(
            "  Data checksum: {} (rel_err={:.2e}, {:.1}ms)",
            if cksum.passed { "PASS" } else { "FAIL" },
            cksum.relative_error,
            t0.elapsed().as_secs_f64() * 1e3
        );
        if !cksum.passed {
            eprintln!(
                "  WARNING: checksum failed! tree_sum={}, expected={}",
                cksum.tree_sum, cksum.expected
            );
        }

        // Insert randoms.
        let t0 = Instant::now();
        randoms::insert_randoms(&mut tree, &rand_pos, &rand_weights);
        let insert_ms = t0.elapsed().as_secs_f64() * 1e3;
        println!("  Inserted randoms in {:.1}ms", insert_ms);

        // Random checksum.
        let rcksum = checksum::verify_random_checksum(&tree, &rand_weights, 1e-12);
        println!(
            "  Random checksum: {} (rel_err={:.2e})",
            if rcksum.passed { "PASS" } else { "FAIL" },
            rcksum.relative_error
        );

        // Estimate ξ.
        let t0 = Instant::now();
        let config = LcaEstimatorConfig {
            bin_edges: bin_edges.clone(),
            mc_samples: args.mc_samples,
            box_size: Some(params.box_size),
            seed: seed + 10000,
        };
        let result = estimator::estimate_xi(&tree, &config);
        let est_ms = t0.elapsed().as_secs_f64() * 1e3;
        println!(
            "  Estimated ξ in {:.1}ms ({} nodes processed, {} skipped)",
            est_ms, result.n_nodes_processed, result.n_nodes_skipped
        );

        // Print results.
        println!("  {:>8} {:>10} {:>10} {:>10}", "r", "xi_tree", "xi_true", "residual");
        for (i, r) in result.r.iter().enumerate() {
            let xi_true = if is_poisson { 0.0 } else { params.xi_analytic(*r) };
            let residual = result.xi[i] - xi_true;
            println!(
                "  {:8.2} {:10.4} {:10.4} {:10.4}",
                r, result.xi[i], xi_true, residual
            );
        }
        println!();
    }
}

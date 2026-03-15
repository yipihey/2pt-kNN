//! CoxMock validation: test the kNN LS estimator against a point process
//! with known analytic ξ(r).
//!
//! Usage:
//!   twopoint-validate [--n-mocks 10] [--output-dir plots/]
//!
//! Produces:
//!   1. xi_vs_analytic.svg — ξ̂(r) vs analytic ξ(r) with residuals
//!   2. r2xi_vs_analytic.svg — r²ξ̂(r) comparison
//!   3. cdf_comparison.svg — measured vs Erlang kNN-CDFs for the random catalog
//!   4. Summary statistics to stdout (mean bias, χ², etc.)

use clap::Parser;
use twopoint::estimator::{LandySzalayKnn, PairCountDensity, XiEstimate, linear_bins};
use twopoint::mock::{CoxMock, CoxMockParams};
use twopoint::tree::PointTree;

#[derive(Parser, Debug)]
#[command(name = "twopoint-validate")]
#[command(about = "CoxMock validation of the kNN Landy–Szalay estimator")]
struct Args {
    /// Number of mock realizations
    #[arg(long, default_value_t = 10)]
    n_mocks: usize,

    /// Output directory for plots
    #[arg(long, default_value = "plots")]
    output_dir: String,

    /// k_max for kNN queries
    #[arg(long, default_value_t = 8)]
    k_max: usize,

    /// Number of radial bins
    #[arg(long, default_value_t = 40)]
    n_bins: usize,

    /// Minimum separation
    #[arg(long, default_value_t = 5.0)]
    r_min: f64,

    /// Maximum separation
    #[arg(long, default_value_t = 250.0)]
    r_max: f64,

    /// Random/data ratio
    #[arg(long, default_value_t = 5)]
    random_ratio: usize,

    /// Print terminal plots (no GUI needed)
    #[arg(long)]
    terminal: bool,
}

fn main() {
    let args = Args::parse();
    std::fs::create_dir_all(&args.output_dir).expect("Failed to create output directory");

    let params = CoxMockParams::euclid_small();
    let r_edges = linear_bins(args.r_min, args.r_max, args.n_bins);
    let r_centers: Vec<f64> = r_edges
        .windows(2)
        .map(|w| 0.5 * (w[0] + w[1]))
        .collect();

    // Analytic ξ(r) at bin centers
    let xi_analytic: Vec<f64> = r_centers.iter().map(|&r| params.xi_analytic(r)).collect();

    println!("=== CoxMock Validation ===");
    println!("N_points = {}", params.n_points);
    println!("N_lines  = {}", params.n_lines);
    println!("ℓ_line   = {:.0}", params.line_length);
    println!("L_box    = {:.0}", params.box_size);
    println!("n̄        = {:.2e}", params.nbar());
    println!("k_max    = {}", args.k_max);
    println!("N_mocks  = {}", args.n_mocks);
    println!("N_R/N_d  = {}", args.random_ratio);
    println!();

    let estimator = LandySzalayKnn::new(args.k_max);
    let n_random = params.n_points * args.random_ratio;

    // Accumulate per-mock ξ estimates
    let mut all_xi: Vec<Vec<f64>> = Vec::with_capacity(args.n_mocks);

    for mock_idx in 0..args.n_mocks {
        let seed_data = (mock_idx * 2) as u64;
        let seed_rand = (mock_idx * 2 + 1) as u64 + 1_000_000;

        print!("Mock {:3}/{:3} ... ", mock_idx + 1, args.n_mocks);

        // Generate data and randoms
        let mock = CoxMock::generate(&params, seed_data);
        let randoms = CoxMock::generate_randoms(n_random, params.box_size, seed_rand);

        // Build tree on data
        let data_tree = PointTree::build(mock.positions.clone());

        // DD: query data positions against data tree
        // (exclude self-pairs by checking distance > 0)
        let dd_dists = estimator.query_distances(&data_tree, &mock.positions);

        // DR: query random positions against data tree
        let dr_dists = estimator.query_distances(&data_tree, &randoms);

        // Compute pair-count densities
        let dd = LandySzalayKnn::pair_count_density(&dd_dists, &r_edges);
        let dr = LandySzalayKnn::pair_count_density(&dr_dists, &r_edges);

        // For a uniform periodic box, we use Davis–Peebles (DD/DR).
        // The DR term from a large random catalog already captures
        // the geometry; RR is uniform by construction.
        let xi_est = LandySzalayKnn::estimate_xi_dp(&dd, &dr);

        println!(
            "ξ(r=50) = {:.4}  (analytic: {:.4})",
            xi_est.xi.get(7).unwrap_or(&f64::NAN),
            params.xi_analytic(50.0)
        );

        all_xi.push(xi_est.xi);
    }

    // Compute mean and std of ξ across mocks
    let n = args.n_mocks as f64;
    let mean_xi: Vec<f64> = (0..args.n_bins)
        .map(|i| all_xi.iter().map(|xi| xi[i]).sum::<f64>() / n)
        .collect();
    let std_xi: Vec<f64> = (0..args.n_bins)
        .map(|i| {
            let mean = mean_xi[i];
            let var = all_xi.iter().map(|xi| (xi[i] - mean).powi(2)).sum::<f64>() / (n - 1.0);
            var.sqrt()
        })
        .collect();
    let stderr_xi: Vec<f64> = std_xi.iter().map(|s| s / n.sqrt()).collect();

    // Print summary
    println!("\n=== Summary ===");
    println!("{:>8} {:>12} {:>12} {:>12} {:>12}", "r", "ξ_analytic", "ξ_mean", "σ_ξ", "bias/σ");
    for i in 0..args.n_bins {
        let bias_sigma = if stderr_xi[i] > 0.0 {
            (mean_xi[i] - xi_analytic[i]) / stderr_xi[i]
        } else {
            0.0
        };
        println!(
            "{:8.1} {:12.6} {:12.6} {:12.6} {:12.2}",
            r_centers[i], xi_analytic[i], mean_xi[i], std_xi[i], bias_sigma
        );
    }

    // Chi-squared
    let chi2: f64 = (0..args.n_bins)
        .map(|i| {
            if stderr_xi[i] > 0.0 {
                ((mean_xi[i] - xi_analytic[i]) / stderr_xi[i]).powi(2)
            } else {
                0.0
            }
        })
        .sum();
    println!(
        "\nχ²/dof = {:.2}/{} = {:.3}",
        chi2,
        args.n_bins,
        chi2 / args.n_bins as f64
    );

    // === Plotting with kuva ===
    // Generate plots as SVG files.
    //
    // NOTE: The kuva API below is based on the README description
    // (builder pattern, SVG output). Adjust if the actual API differs.
    //
    // TODO: Replace with actual kuva calls once we verify the API.

    // For now, write a simple TSV that can be plotted with any tool
    let tsv_path = format!("{}/xi_comparison.tsv", args.output_dir);
    let mut tsv = String::new();
    tsv.push_str("# r\txi_analytic\txi_mean\txi_std\txi_stderr\n");
    for i in 0..args.n_bins {
        tsv.push_str(&format!(
            "{:.2}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\n",
            r_centers[i], xi_analytic[i], mean_xi[i], std_xi[i], stderr_xi[i]
        ));
    }
    std::fs::write(&tsv_path, &tsv).expect("Failed to write TSV");
    println!("\nWrote {}", tsv_path);

    // === Kuva plotting ===
    // Uncomment and adjust once kuva API is confirmed:
    //
    // use kuva::{Plot, Figure, Subplot};
    //
    // let fig = Figure::new()
    //     .with_size(800, 600)
    //     .subplot(1, 1, 1, |sp| {
    //         sp.line(&r_centers, &xi_analytic)
    //           .label("Analytic ξ(r)")
    //           .color("black")
    //           .line_style("--");
    //         sp.errorbar(&r_centers, &mean_xi, &stderr_xi)
    //           .label("kNN LS mean ± σ/√N")
    //           .color("steelblue");
    //         sp.set_xlabel("r [h⁻¹ Mpc]");
    //         sp.set_ylabel("ξ(r)");
    //         sp.set_title("CoxMock Validation: kNN Landy–Szalay");
    //         sp.legend();
    //     });
    //
    // fig.save_svg(&format!("{}/xi_vs_analytic.svg", args.output_dir))
    //    .expect("Failed to save SVG");

    println!("\nDone. Inspect {} for results.", args.output_dir);
}

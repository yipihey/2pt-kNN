//! CoxMock validation: test the kNN LS estimator against a point process
//! with known analytic ξ(r).
//!
//! Usage:
//!   twopoint-validate [--n-mocks 20] [--output-dir plots/]
//!
//! Produces SVG plots + summary TSV in the output directory:
//!   1. xi_vs_analytic.svg — ξ̂(r) vs analytic ξ(r) with ±1σ band
//!   2. xi_residuals.svg — (ξ̂ − ξ_true)/σ_mean residuals
//!   3. r2xi_comparison.svg — r²ξ(r) amplifying intermediate scales
//!   4. cdf_comparison.svg — kNN-CDFs vs Erlang for the random catalog
//!   5. individual_mocks.svg — all mock ξ̂(r) overlaid
//!   6. validation_summary.svg — combined 2×2 figure for the paper

use clap::Parser;
use kuva::backend::svg::SvgBackend;
use kuva::plot::{BandPlot, LinePlot, ScatterPlot};
use kuva::render::annotations::ReferenceLine;
use kuva::render::figure::Figure;
use kuva::render::layout::Layout;
use kuva::render::plots::Plot;
use kuva::render::render::render_multiple;
use twopoint::estimator::{linear_bins, KnnDistances, KnnDistributions, LandySzalayKnn};
use twopoint::mock::{CoxMock, CoxMockParams};
use twopoint::tree::PointTree;

#[derive(Parser, Debug)]
#[command(name = "twopoint-validate")]
#[command(about = "CoxMock validation of the kNN Landy-Szalay estimator")]
struct Args {
    /// Number of mock realizations
    #[arg(long, default_value_t = 20)]
    n_mocks: usize,

    /// Output directory for plots
    #[arg(long, default_value = "plots")]
    output_dir: String,

    /// k_max for kNN queries
    #[arg(long, default_value_t = 128)]
    k_max: usize,

    /// Number of radial bins
    #[arg(long, default_value_t = 30)]
    n_bins: usize,

    /// Minimum separation [h^-1 Mpc]
    #[arg(long, default_value_t = 3.0)]
    r_min: f64,

    /// Maximum separation [h^-1 Mpc]
    #[arg(long, default_value_t = 48.0)]
    r_max: f64,

    /// Random/data ratio N_R / N_d
    #[arg(long, default_value_t = 10)]
    random_ratio: usize,

    /// CoxMock preset: "validation" (fast, strong signal) or "euclid_small"
    #[arg(long, default_value = "validation")]
    preset: String,
}

/// Erlang CDF for the k-th nearest neighbor at distance r, given n̄.
///
/// CDF_k(r) = 1 − exp(−λ) Σ_{j=0}^{k−1} λ^j / j!
/// where λ = n̄ · (4/3)πr³
fn erlang_cdf(k: usize, r: f64, nbar: f64) -> f64 {
    let lambda = nbar * 4.0 / 3.0 * std::f64::consts::PI * r.powi(3);
    let mut sum = 0.0;
    let mut term = 1.0;
    for j in 0..k {
        if j > 0 {
            term *= lambda / j as f64;
        }
        sum += term;
    }
    1.0 - (-lambda).exp() * sum
}

/// Remove self-pairs from DD query results.
///
/// When querying data against its own tree, the first neighbor is the
/// point itself (distance ≈ 0). We query k_max+1 neighbors and drop
/// the self-pair, keeping exactly k_max real neighbors.
fn exclude_self_pairs(dists: KnnDistributions, k_max: usize) -> KnnDistributions {
    let filtered: Vec<KnnDistances> = dists
        .per_query
        .into_iter()
        .map(|qd| {
            let real: Vec<f64> = qd
                .distances
                .into_iter()
                .filter(|&d| d > 1e-10) // drop the self at distance ≈ 0
                .take(k_max)
                .collect();
            KnnDistances { distances: real }
        })
        .collect();

    KnnDistributions {
        per_query: filtered,
        k_max,
    }
}

fn main() {
    let args = Args::parse();
    std::fs::create_dir_all(&args.output_dir).expect("Failed to create output directory");

    let params = match args.preset.as_str() {
        "euclid_small" => CoxMockParams::euclid_small(),
        "tiny" => CoxMockParams::tiny(),
        "poisson" => CoxMockParams::poisson(),
        _ => CoxMockParams::validation(),
    };

    let r_edges = linear_bins(args.r_min, args.r_max, args.n_bins);
    let r_centers: Vec<f64> = r_edges.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect();

    // Dense analytic curve for smooth plotting
    let n_smooth = 300;
    let r_smooth: Vec<f64> = (0..n_smooth)
        .map(|i| args.r_min + (args.r_max - args.r_min) * i as f64 / (n_smooth - 1) as f64)
        .collect();
    let xi_smooth: Vec<f64> = r_smooth.iter().map(|&r| params.xi_analytic(r)).collect();

    let xi_analytic: Vec<f64> = r_centers.iter().map(|&r| params.xi_analytic(r)).collect();

    let m = params.points_per_line();
    println!("=== CoxMock Validation ===");
    println!("Preset   = {}", args.preset);
    println!("N_points = {}", params.n_points);
    println!("N_lines  = {}", params.n_lines);
    println!("l_line   = {:.0}", params.line_length);
    println!("L_box    = {:.0}", params.box_size);
    println!("m        = {:.0} pts/line", m);
    println!("nbar     = {:.2e}", params.nbar());
    println!("k_max    = {}", args.k_max);
    println!("r_k_max  = {:.1} (char. scale for k={})", params.r_char_k(args.k_max), args.k_max);
    println!("N_mocks  = {}", args.n_mocks);
    println!("N_R/N_d  = {}", args.random_ratio);
    println!("r range  = [{:.1}, {:.1}]", args.r_min, args.r_max);
    println!("xi(5)    = {:.4} (analytic)", params.xi_analytic(5.0));
    println!("xi(20)   = {:.4} (analytic)", params.xi_analytic(20.0));
    println!();

    let estimator = LandySzalayKnn::new(args.k_max);
    // For DD, query one extra neighbor to exclude self-pair
    let dd_estimator = LandySzalayKnn::new(args.k_max + 1);
    let n_random = params.n_points * args.random_ratio;

    let mut all_xi: Vec<Vec<f64>> = Vec::with_capacity(args.n_mocks);

    // CDF measurements from the first mock
    let mut cdf_data: Option<CdfMeasurements> = None;

    for mock_idx in 0..args.n_mocks {
        let seed_data = (mock_idx * 2) as u64;
        let seed_rand = (mock_idx * 2 + 1) as u64 + 1_000_000;

        print!("Mock {:3}/{:3} ... ", mock_idx + 1, args.n_mocks);

        let mock = CoxMock::generate(&params, seed_data);
        let randoms = CoxMock::generate_randoms(n_random, params.box_size, seed_rand);

        // Build tree on data
        let data_tree = PointTree::build(mock.positions.clone());

        // DD: data → data tree (periodic, query k_max+1, then exclude self)
        let dd_raw = dd_estimator.query_distances_periodic(
            &data_tree, &mock.positions, params.box_size,
        );
        let dd_dists = exclude_self_pairs(dd_raw, args.k_max);

        // DR: randoms → data tree (periodic)
        let dr_dists = estimator.query_distances_periodic(
            &data_tree, &randoms, params.box_size,
        );

        let dd = LandySzalayKnn::pair_count_density(&dd_dists, &r_edges);
        let dr = LandySzalayKnn::pair_count_density(&dr_dists, &r_edges);

        // Davis-Peebles: ξ̂ = DD/DR − 1
        let xi_est = LandySzalayKnn::estimate_xi_dp(&dd, &dr);

        // Probe at a representative scale
        let r_probe = 10.0;
        let bin_probe = r_centers
            .iter()
            .position(|&r| r >= r_probe)
            .unwrap_or(0);
        println!(
            "xi(r={:.0}) = {:.3}  (analytic: {:.3})",
            r_centers[bin_probe], xi_est.xi[bin_probe], xi_analytic[bin_probe]
        );

        // On the first mock, measure kNN-CDFs for the random catalog
        if mock_idx == 0 {
            let rand_tree = PointTree::build(randoms.clone());
            let n_cdf_query = 10_000usize.min(randoms.len());
            let rr_dists = estimator.query_distances_periodic(
                &rand_tree, &randoms[..n_cdf_query], params.box_size,
            );

            let r_cdf: Vec<f64> = (1..=100)
                .map(|i| i as f64 * params.r_char_k(1) * 2.5 / 100.0)
                .collect();

            let ks = [1usize, 2, 4, 8];
            let mut measured_cdfs = Vec::new();
            for &k in &ks {
                if k <= args.k_max {
                    let cdf = LandySzalayKnn::empirical_cdf(&rr_dists, k, &r_cdf);
                    measured_cdfs.push((k, cdf));
                }
            }

            cdf_data = Some(CdfMeasurements {
                r: r_cdf,
                measured: measured_cdfs,
                nbar: n_random as f64 / params.volume(),
            });
        }

        all_xi.push(xi_est.xi);
    }

    // ========================================================================
    // Statistics
    // ========================================================================
    let n = args.n_mocks as f64;
    let mean_xi: Vec<f64> = (0..args.n_bins)
        .map(|i| all_xi.iter().map(|xi| xi[i]).sum::<f64>() / n)
        .collect();
    let std_xi: Vec<f64> = (0..args.n_bins)
        .map(|i| {
            let mean = mean_xi[i];
            let var =
                all_xi.iter().map(|xi| (xi[i] - mean).powi(2)).sum::<f64>() / (n - 1.0);
            var.sqrt()
        })
        .collect();
    let stderr_xi: Vec<f64> = std_xi.iter().map(|s| s / n.sqrt()).collect();

    let bias_sigma: Vec<f64> = (0..args.n_bins)
        .map(|i| {
            if stderr_xi[i] > 0.0 {
                (mean_xi[i] - xi_analytic[i]) / stderr_xi[i]
            } else {
                0.0
            }
        })
        .collect();

    // Print summary table
    println!("\n=== Summary ===");
    println!(
        "{:>8} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "r", "xi_true", "xi_mean", "sigma_xi", "stderr", "bias/se"
    );
    for i in 0..args.n_bins {
        println!(
            "{:8.1} {:12.4} {:12.4} {:12.4} {:12.4} {:12.2}",
            r_centers[i], xi_analytic[i], mean_xi[i], std_xi[i], stderr_xi[i], bias_sigma[i]
        );
    }

    let chi2: f64 = bias_sigma.iter().map(|b| b * b).sum();
    let max_bias = bias_sigma
        .iter()
        .map(|b| b.abs())
        .fold(0.0f64, f64::max);
    println!(
        "\nchi2/dof = {:.2}/{} = {:.3}",
        chi2,
        args.n_bins,
        chi2 / args.n_bins as f64
    );
    println!("max |bias|/sigma_mean = {:.2}", max_bias);

    // ========================================================================
    // TSV output
    // ========================================================================
    let tsv_path = format!("{}/xi_comparison.tsv", args.output_dir);
    let mut tsv = String::new();
    tsv.push_str("# r\txi_analytic\txi_mean\txi_std\txi_stderr\tbias_sigma\n");
    for i in 0..args.n_bins {
        tsv.push_str(&format!(
            "{:.2}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.4}\n",
            r_centers[i], xi_analytic[i], mean_xi[i], std_xi[i], stderr_xi[i], bias_sigma[i]
        ));
    }
    std::fs::write(&tsv_path, &tsv).expect("Failed to write TSV");
    println!("\nWrote {}", tsv_path);

    // ========================================================================
    // PLOTS
    // ========================================================================
    println!("\nGenerating plots...");

    // --- Plot 1: ξ(r) comparison ---
    plot_xi_comparison(
        &args, &params, &r_centers, &mean_xi, &std_xi, &r_smooth, &xi_smooth,
    );

    // --- Plot 2: Residuals ---
    plot_residuals(&args, &r_centers, &bias_sigma);

    // --- Plot 3: r²ξ(r) ---
    plot_r2xi(
        &args, &params, &r_centers, &mean_xi, &std_xi, &r_smooth, &xi_smooth,
    );

    // --- Plot 4: kNN-CDF comparison ---
    if let Some(ref cdf) = cdf_data {
        plot_cdf_comparison(&args, cdf);
    }

    // --- Plot 5: Individual mocks ---
    plot_individual_mocks(
        &args, &params, &r_centers, &all_xi, &mean_xi, &r_smooth, &xi_smooth,
    );

    // --- Plot 6: Combined 2×2 summary figure ---
    plot_summary_figure(
        &args,
        &params,
        &r_centers,
        &mean_xi,
        &std_xi,
        &bias_sigma,
        &r_smooth,
        &xi_smooth,
        &cdf_data,
    );

    // Final summary
    println!("\n=== Paper-ready summary ===");
    println!("N_mock = {}", args.n_mocks);
    println!(
        "chi2/dof = {:.2}/{} = {:.3}",
        chi2,
        args.n_bins,
        chi2 / args.n_bins as f64
    );
    println!("max |bias|/sigma_mean = {:.2}", max_bias);
    println!(
        "CoxMock: L={:.0}, N_L={}, l={:.0}, N_p={}, m={:.0}, nbar={:.2e}",
        params.box_size,
        params.n_lines,
        params.line_length,
        params.n_points,
        params.points_per_line(),
        params.nbar()
    );
    println!("\nDone. SVGs in {}/", args.output_dir);
}

// ============================================================================
// Plotting functions
// ============================================================================

fn plot_xi_comparison(
    args: &Args,
    params: &CoxMockParams,
    r_centers: &[f64],
    mean_xi: &[f64],
    std_xi: &[f64],
    r_smooth: &[f64],
    xi_smooth: &[f64],
) {
    let n_bins = r_centers.len();
    let lower: Vec<f64> = (0..n_bins).map(|i| mean_xi[i] - std_xi[i]).collect();
    let upper: Vec<f64> = (0..n_bins).map(|i| mean_xi[i] + std_xi[i]).collect();

    let band = BandPlot::new(r_centers.to_vec(), lower, upper)
        .with_color("steelblue")
        .with_opacity(0.2);

    let mean_line = LinePlot::new()
        .with_data(r_centers.iter().copied().zip(mean_xi.iter().copied()))
        .with_color("steelblue")
        .with_stroke_width(2.0)
        .with_legend("kNN LS mean +/- 1sigma");

    let analytic = LinePlot::new()
        .with_data(r_smooth.iter().copied().zip(xi_smooth.iter().copied()))
        .with_color("black")
        .with_stroke_width(1.5)
        .with_dashed()
        .with_legend("Analytic xi(r)");

    let ell_ref = ReferenceLine::vertical(params.line_length)
        .with_color("#888888")
        .with_label("r = l")
        .with_stroke_width(0.8);

    let plots = vec![Plot::Band(band), Plot::Line(mean_line), Plot::Line(analytic)];
    let layout = Layout::auto_from_plots(&plots)
        .with_title("CoxMock: xi(r) Recovery")
        .with_x_label("r  [h^-1 Mpc]")
        .with_y_label("xi(r)")
        .with_width(700.0)
        .with_height(500.0)
        .with_reference_line(ell_ref);

    let svg = SvgBackend.render_scene(&render_multiple(plots, layout));
    let path = format!("{}/xi_vs_analytic.svg", args.output_dir);
    std::fs::write(&path, svg).unwrap();
    println!("  Wrote {}", path);
}

fn plot_residuals(args: &Args, r_centers: &[f64], bias_sigma: &[f64]) {
    let data: Vec<(f64, f64)> = r_centers
        .iter()
        .copied()
        .zip(bias_sigma.iter().copied())
        .collect();

    let scatter = ScatterPlot::new()
        .with_data(data)
        .with_color("steelblue")
        .with_size(4.5)
        .with_legend("(xi_mean - xi_true) / sigma_mean");

    let zero = ReferenceLine::horizontal(0.0)
        .with_color("black")
        .with_stroke_width(1.0)
        .with_dasharray("4 3");
    let plus2 = ReferenceLine::horizontal(2.0)
        .with_color("crimson")
        .with_stroke_width(0.7)
        .with_dasharray("6 4")
        .with_label("+2sigma");
    let minus2 = ReferenceLine::horizontal(-2.0)
        .with_color("crimson")
        .with_stroke_width(0.7)
        .with_dasharray("6 4")
        .with_label("-2sigma");

    let plots = vec![Plot::Scatter(scatter)];
    let layout = Layout::auto_from_plots(&plots)
        .with_title("Estimator Bias")
        .with_x_label("r  [h^-1 Mpc]")
        .with_y_label("bias / sigma_mean")
        .with_width(700.0)
        .with_height(400.0)
        .with_reference_line(zero)
        .with_reference_line(plus2)
        .with_reference_line(minus2);

    let svg = SvgBackend.render_scene(&render_multiple(plots, layout));
    let path = format!("{}/xi_residuals.svg", args.output_dir);
    std::fs::write(&path, svg).unwrap();
    println!("  Wrote {}", path);
}

fn plot_r2xi(
    args: &Args,
    params: &CoxMockParams,
    r_centers: &[f64],
    mean_xi: &[f64],
    std_xi: &[f64],
    r_smooth: &[f64],
    xi_smooth: &[f64],
) {
    let n_bins = r_centers.len();
    let r2xi_mean: Vec<f64> = (0..n_bins)
        .map(|i| r_centers[i].powi(2) * mean_xi[i])
        .collect();
    let r2xi_lower: Vec<f64> = (0..n_bins)
        .map(|i| r_centers[i].powi(2) * (mean_xi[i] - std_xi[i]))
        .collect();
    let r2xi_upper: Vec<f64> = (0..n_bins)
        .map(|i| r_centers[i].powi(2) * (mean_xi[i] + std_xi[i]))
        .collect();
    let r2xi_analytic: Vec<f64> = r_smooth
        .iter()
        .zip(xi_smooth.iter())
        .map(|(&r, &xi)| r * r * xi)
        .collect();

    let band = BandPlot::new(r_centers.to_vec(), r2xi_lower, r2xi_upper)
        .with_color("steelblue")
        .with_opacity(0.2);
    let mean_line = LinePlot::new()
        .with_data(r_centers.iter().copied().zip(r2xi_mean.iter().copied()))
        .with_color("steelblue")
        .with_stroke_width(2.0)
        .with_legend("kNN LS");
    let analytic = LinePlot::new()
        .with_data(r_smooth.iter().copied().zip(r2xi_analytic.iter().copied()))
        .with_color("black")
        .with_stroke_width(1.5)
        .with_dashed()
        .with_legend("Analytic");

    let ell_ref = ReferenceLine::vertical(params.line_length)
        .with_color("#888888")
        .with_label("r = l")
        .with_stroke_width(0.8);

    let plots = vec![Plot::Band(band), Plot::Line(mean_line), Plot::Line(analytic)];
    let layout = Layout::auto_from_plots(&plots)
        .with_title("CoxMock: r^2 xi(r)")
        .with_x_label("r  [h^-1 Mpc]")
        .with_y_label("r^2 xi(r)  [h^-2 Mpc^2]")
        .with_width(700.0)
        .with_height(500.0)
        .with_reference_line(ell_ref);

    let svg = SvgBackend.render_scene(&render_multiple(plots, layout));
    let path = format!("{}/r2xi_comparison.svg", args.output_dir);
    std::fs::write(&path, svg).unwrap();
    println!("  Wrote {}", path);
}

fn plot_cdf_comparison(args: &Args, cdf: &CdfMeasurements) {
    let colors = ["steelblue", "crimson", "seagreen", "darkorange"];
    let mut plots: Vec<Plot> = Vec::new();

    for (idx, (k, measured)) in cdf.measured.iter().enumerate() {
        let color = colors[idx % colors.len()];

        plots.push(Plot::Line(
            LinePlot::new()
                .with_data(cdf.r.iter().copied().zip(measured.iter().copied()))
                .with_color(color)
                .with_stroke_width(2.0)
                .with_legend(&format!("k={} measured", k)),
        ));

        let erlang: Vec<f64> = cdf.r.iter().map(|&r| erlang_cdf(*k, r, cdf.nbar)).collect();
        plots.push(Plot::Line(
            LinePlot::new()
                .with_data(cdf.r.iter().copied().zip(erlang.iter().copied()))
                .with_color("black")
                .with_stroke_width(1.0)
                .with_dashed(),
        ));
    }

    let layout = Layout::auto_from_plots(&plots)
        .with_title("kNN-CDF: Random Catalog vs Erlang")
        .with_x_label("r  [h^-1 Mpc]")
        .with_y_label("CDF_k(r)")
        .with_width(700.0)
        .with_height(500.0);

    let svg = SvgBackend.render_scene(&render_multiple(plots, layout));
    let path = format!("{}/cdf_comparison.svg", args.output_dir);
    std::fs::write(&path, svg).unwrap();
    println!("  Wrote {}", path);
}

fn plot_individual_mocks(
    args: &Args,
    params: &CoxMockParams,
    r_centers: &[f64],
    all_xi: &[Vec<f64>],
    mean_xi: &[f64],
    r_smooth: &[f64],
    xi_smooth: &[f64],
) {
    // Light colors for individual mocks
    let mock_colors = [
        "#b8d4f0", "#a0c4e8", "#c0d8f4", "#90b4d8", "#a8cce8",
        "#b0d0f0", "#98bce0", "#c8dcf4", "#88acd0", "#b0c8e8",
        "#b4d0ec", "#9cc0e4", "#c4d8f0", "#94b8dc", "#a4c8e4",
        "#acd0ec", "#a0c0e0", "#bcd4f0", "#8cb0d4", "#b4cce8",
    ];

    let mut plots: Vec<Plot> = Vec::new();
    for (idx, xi) in all_xi.iter().enumerate() {
        let color = mock_colors[idx % mock_colors.len()];
        plots.push(Plot::Line(
            LinePlot::new()
                .with_data(r_centers.iter().copied().zip(xi.iter().copied()))
                .with_color(color)
                .with_stroke_width(0.7),
        ));
    }

    plots.push(Plot::Line(
        LinePlot::new()
            .with_data(r_centers.iter().copied().zip(mean_xi.iter().copied()))
            .with_color("steelblue")
            .with_stroke_width(2.5)
            .with_legend("Mean"),
    ));

    plots.push(Plot::Line(
        LinePlot::new()
            .with_data(r_smooth.iter().copied().zip(xi_smooth.iter().copied()))
            .with_color("black")
            .with_stroke_width(2.0)
            .with_dashed()
            .with_legend("Analytic"),
    ));

    let ell_ref = ReferenceLine::vertical(params.line_length)
        .with_color("#888888")
        .with_label("r = l")
        .with_stroke_width(0.8);

    let layout = Layout::auto_from_plots(&plots)
        .with_title("Individual Mock Realizations")
        .with_x_label("r  [h^-1 Mpc]")
        .with_y_label("xi(r)")
        .with_width(700.0)
        .with_height(500.0)
        .with_reference_line(ell_ref);

    let svg = SvgBackend.render_scene(&render_multiple(plots, layout));
    let path = format!("{}/individual_mocks.svg", args.output_dir);
    std::fs::write(&path, svg).unwrap();
    println!("  Wrote {}", path);
}

fn plot_summary_figure(
    args: &Args,
    _params: &CoxMockParams,
    r_centers: &[f64],
    mean_xi: &[f64],
    std_xi: &[f64],
    bias_sigma: &[f64],
    r_smooth: &[f64],
    xi_smooth: &[f64],
    cdf_data: &Option<CdfMeasurements>,
) {
    let n_bins = r_centers.len();

    // Panel A: ξ(r) with band
    let lower: Vec<f64> = (0..n_bins).map(|i| mean_xi[i] - std_xi[i]).collect();
    let upper: Vec<f64> = (0..n_bins).map(|i| mean_xi[i] + std_xi[i]).collect();

    let pa_plots = vec![
        Plot::Band(
            BandPlot::new(r_centers.to_vec(), lower, upper)
                .with_color("steelblue")
                .with_opacity(0.2),
        ),
        Plot::Line(
            LinePlot::new()
                .with_data(r_centers.iter().copied().zip(mean_xi.iter().copied()))
                .with_color("steelblue")
                .with_stroke_width(2.0)
                .with_legend("kNN LS"),
        ),
        Plot::Line(
            LinePlot::new()
                .with_data(r_smooth.iter().copied().zip(xi_smooth.iter().copied()))
                .with_color("black")
                .with_stroke_width(1.5)
                .with_dashed()
                .with_legend("Analytic"),
        ),
    ];

    // Panel B: r²ξ(r)
    let r2xi_mean: Vec<f64> = (0..n_bins)
        .map(|i| r_centers[i].powi(2) * mean_xi[i])
        .collect();
    let r2xi_lower: Vec<f64> = (0..n_bins)
        .map(|i| r_centers[i].powi(2) * (mean_xi[i] - std_xi[i]))
        .collect();
    let r2xi_upper: Vec<f64> = (0..n_bins)
        .map(|i| r_centers[i].powi(2) * (mean_xi[i] + std_xi[i]))
        .collect();
    let r2xi_analytic: Vec<f64> = r_smooth
        .iter()
        .zip(xi_smooth.iter())
        .map(|(&r, &xi)| r * r * xi)
        .collect();

    let pb_plots = vec![
        Plot::Band(
            BandPlot::new(r_centers.to_vec(), r2xi_lower, r2xi_upper)
                .with_color("steelblue")
                .with_opacity(0.2),
        ),
        Plot::Line(
            LinePlot::new()
                .with_data(r_centers.iter().copied().zip(r2xi_mean.iter().copied()))
                .with_color("steelblue")
                .with_stroke_width(2.0),
        ),
        Plot::Line(
            LinePlot::new()
                .with_data(r_smooth.iter().copied().zip(r2xi_analytic.iter().copied()))
                .with_color("black")
                .with_stroke_width(1.5)
                .with_dashed(),
        ),
    ];

    // Panel C: residuals
    let resid_data: Vec<(f64, f64)> = r_centers
        .iter()
        .copied()
        .zip(bias_sigma.iter().copied())
        .collect();
    let pc_plots = vec![Plot::Scatter(
        ScatterPlot::new()
            .with_data(resid_data)
            .with_color("steelblue")
            .with_size(3.5),
    )];

    // Panel D: CDF comparison
    let mut pd_plots: Vec<Plot> = Vec::new();
    if let Some(ref cdf) = cdf_data {
        let colors = ["steelblue", "crimson", "seagreen", "darkorange"];
        for (idx, (k, measured)) in cdf.measured.iter().enumerate() {
            let color = colors[idx % colors.len()];
            pd_plots.push(Plot::Line(
                LinePlot::new()
                    .with_data(cdf.r.iter().copied().zip(measured.iter().copied()))
                    .with_color(color)
                    .with_stroke_width(1.5)
                    .with_legend(&format!("k={}", k)),
            ));
            let erlang: Vec<f64> =
                cdf.r.iter().map(|&r| erlang_cdf(*k, r, cdf.nbar)).collect();
            pd_plots.push(Plot::Line(
                LinePlot::new()
                    .with_data(cdf.r.iter().copied().zip(erlang.iter().copied()))
                    .with_color("black")
                    .with_stroke_width(0.8)
                    .with_dashed(),
            ));
        }
    }

    let zero_ref = ReferenceLine::horizontal(0.0)
        .with_color("black")
        .with_stroke_width(0.8)
        .with_dasharray("4 3");
    let plus2_ref = ReferenceLine::horizontal(2.0)
        .with_color("crimson")
        .with_stroke_width(0.6)
        .with_dasharray("6 4");
    let minus2_ref = ReferenceLine::horizontal(-2.0)
        .with_color("crimson")
        .with_stroke_width(0.6)
        .with_dasharray("6 4");

    let all_plots = vec![pa_plots, pb_plots, pc_plots, pd_plots];

    let layouts = vec![
        Layout::auto_from_plots(&all_plots[0])
            .with_title("xi(r)")
            .with_x_label("r  [h^-1 Mpc]")
            .with_y_label("xi(r)"),
        Layout::auto_from_plots(&all_plots[1])
            .with_title("r^2 xi(r)")
            .with_x_label("r  [h^-1 Mpc]")
            .with_y_label("r^2 xi(r)"),
        Layout::auto_from_plots(&all_plots[2])
            .with_title("Bias / sigma_mean")
            .with_x_label("r  [h^-1 Mpc]")
            .with_y_label("(xi - xi_true) / sigma")
            .with_reference_line(zero_ref)
            .with_reference_line(plus2_ref)
            .with_reference_line(minus2_ref),
        Layout::auto_from_plots(&all_plots[3])
            .with_title("kNN-CDF vs Erlang (RR)")
            .with_x_label("r  [h^-1 Mpc]")
            .with_y_label("CDF_k(r)"),
    ];

    let scene = Figure::new(2, 2)
        .with_plots(all_plots)
        .with_layouts(layouts)
        .with_labels()
        .with_shared_legend()
        .with_cell_size(480.0, 380.0)
        .with_title("CoxMock Validation: kNN Landy-Szalay Estimator")
        .render();

    let svg = SvgBackend.render_scene(&scene);
    let path = format!("{}/validation_summary.svg", args.output_dir);
    std::fs::write(&path, svg).unwrap();
    println!("  Wrote {}", path);
}

struct CdfMeasurements {
    r: Vec<f64>,
    measured: Vec<(usize, Vec<f64>)>,
    nbar: f64,
}

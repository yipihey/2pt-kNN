//! CoxMock validation: test the kNN Landy-Szalay estimator against a point
//! process with known analytic ξ(r).
//!
//! Both kNN and Corrfunc use the full Landy-Szalay estimator with explicit
//! DD, DR, RR pair counts — no analytic RR shortcuts.
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
use std::time::Instant;
use twopoint::corrfunc::CorrfuncRunner;
use twopoint::estimator::{
    cdf_k_values, cdf_r_grid, exclude_self_pairs, linear_bins, LandySzalayKnn,
};
use twopoint::ladder::{
    average_cdfs, stitch_levels, DilutionLadder, KnnCdfSummary, LevelResult,
};
use twopoint::mock::{CoxMock, CoxMockParams};
use twopoint::plotting::{self, PlotConfig, PlotData, TypstPlotter};
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
    #[arg(long, default_value_t = 144.0)]
    r_max: f64,

    /// Random/data ratio N_R / N_d
    #[arg(long, default_value_t = 10)]
    random_ratio: usize,

    /// CoxMock preset: "validation" (fast, strong signal) or "euclid_small"
    #[arg(long, default_value = "validation")]
    preset: String,

    /// Run Corrfunc pair-counting as a reference comparison
    #[arg(long)]
    corrfunc: bool,

    /// Number of threads for Corrfunc
    #[arg(long, default_value_t = 4)]
    corrfunc_threads: usize,

    /// Python interpreter for Corrfunc (auto-detected if omitted)
    #[arg(long)]
    python: Option<String>,

    /// Enable dilution ladder for extended radial range
    #[arg(long)]
    dilution: bool,

    /// Maximum dilution level (R_max = 8^max_level)
    #[arg(long, default_value_t = 2)]
    max_dilution_level: usize,

    /// Number of radial bins per dilution level
    #[arg(long, default_value_t = 20)]
    bins_per_level: usize,

    /// Override CoxMock n_points
    #[arg(long)]
    n_points: Option<usize>,

    /// Override CoxMock box_size
    #[arg(long)]
    box_size: Option<f64>,

    /// Override CoxMock n_lines
    #[arg(long)]
    n_lines: Option<usize>,

    /// Override CoxMock line_length
    #[arg(long)]
    line_length: Option<f64>,

    /// Generate paper-quality 2-panel ξ(r) + ratio figure
    #[arg(long)]
    paper: bool,

    /// Launch interactive TUI plot explorer after batch generation
    #[cfg(feature = "interactive")]
    #[arg(long)]
    interactive: bool,
}

fn main() {
    let args = Args::parse();
    std::fs::create_dir_all(&args.output_dir).expect("Failed to create output directory");

    let mut params = match args.preset.as_str() {
        "euclid_small" => CoxMockParams::euclid_small(),
        "tiny" => CoxMockParams::tiny(),
        "poisson" => CoxMockParams::poisson(),
        _ => CoxMockParams::validation(),
    };

    // Apply CLI overrides to CoxMock parameters
    if let Some(v) = args.n_points {
        params.n_points = v;
    }
    if let Some(v) = args.box_size {
        params.box_size = v;
    }
    if let Some(v) = args.n_lines {
        params.n_lines = v;
    }
    if let Some(v) = args.line_length {
        params.line_length = v;
    }

    // Compute bin edges (per-level for dilution, single range otherwise)
    let dilution_level_edges: Option<Vec<Vec<f64>>> = if args.dilution {
        let nbar = params.nbar();
        let mut edges_per_level = Vec::new();
        for level in 0..=args.max_dilution_level {
            let edges = DilutionLadder::bin_edges_for_level(
                level,
                args.k_max,
                nbar,
                args.bins_per_level,
                args.r_min,
                params.box_size,
                params.n_points,
            );
            edges_per_level.push(edges);
        }
        Some(edges_per_level)
    } else {
        None
    };

    let r_max_eff;
    let r_edges;
    let r_centers;

    if args.dilution {
        let dle = dilution_level_edges.as_ref().unwrap();
        // Build composite edges (levels share boundary points)
        let mut composite = Vec::new();
        for (i, edges) in dle.iter().enumerate() {
            if i == 0 {
                composite.extend(edges);
            } else {
                composite.extend(&edges[1..]);
            }
        }
        r_max_eff = *composite.last().unwrap();
        r_centers = composite
            .windows(2)
            .map(|w| 0.5 * (w[0] + w[1]))
            .collect::<Vec<_>>();
        r_edges = composite;
    } else {
        // Clamp r_max to the kNN reach so both methods use identical bins
        let r_kmax = params.r_char_k(args.k_max);
        r_max_eff = if args.r_max > r_kmax {
            println!(
                "Note: r_max={:.1} exceeds kNN reach r_char(k={})={:.1}; clamping to {:.1}",
                args.r_max, args.k_max, r_kmax, r_kmax
            );
            r_kmax
        } else {
            args.r_max
        };
        r_edges = linear_bins(args.r_min, r_max_eff, args.n_bins);
        r_centers = r_edges
            .windows(2)
            .map(|w| 0.5 * (w[0] + w[1]))
            .collect::<Vec<_>>();
    }

    let n_bins = r_centers.len();

    // Level tags for composite data (which dilution level each bin came from)
    let composite_level_tags: Option<Vec<usize>> = if args.dilution {
        let dle = dilution_level_edges.as_ref().unwrap();
        let mut tags = Vec::new();
        for (level, edges) in dle.iter().enumerate() {
            tags.extend(std::iter::repeat(level).take(edges.len() - 1));
        }
        Some(tags)
    } else {
        None
    };

    // Dense analytic curve for smooth plotting
    let n_smooth = 300;
    let r_smooth_max = if args.dilution { r_max_eff } else { args.r_max };
    let r_smooth: Vec<f64> = (0..n_smooth)
        .map(|i| args.r_min + (r_smooth_max - args.r_min) * i as f64 / (n_smooth - 1) as f64)
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
    println!("RR mode  = diluted (subsample randoms to data density)");
    println!("r range  = [{:.1}, {:.1}]", args.r_min, r_max_eff);
    println!("n_bins   = {}", n_bins);
    println!("xi(5)    = {:.4} (analytic)", params.xi_analytic(5.0));
    println!("xi(20)   = {:.4} (analytic)", params.xi_analytic(20.0));
    if args.dilution {
        println!("\n--- Dilution Ladder ---");
        for level in 0..=args.max_dilution_level {
            let r_ell = 8usize.pow(level as u32);
            let n_sub = params.n_points / r_ell;
            let k_eff = if level == 0 {
                args.k_max
            } else {
                DilutionLadder::effective_k_max(args.k_max, params.n_points, r_ell)
            };
            let dle = dilution_level_edges.as_ref().unwrap();
            let lo = dle[level].first().unwrap();
            let hi = dle[level].last().unwrap();
            println!(
                "  Level {}: R={:>3}, n_sub={:>6}, k_eff={:>3}, r=[{:.1}, {:.1}], n_subsamples={}",
                level, r_ell, n_sub, k_eff, lo, hi, r_ell,
            );
        }
        println!("  Composite r range = [{:.1}, {:.1}]", args.r_min, r_max_eff);
        println!();
    }
    let rayon_threads = rayon::current_num_threads();
    println!("kNN threads (rayon) = {}", rayon_threads);
    if args.corrfunc {
        println!("Corrfunc threads    = {}", args.corrfunc_threads);
    }
    println!();

    let estimator = LandySzalayKnn::new(args.k_max);
    // For DD and RR (self-pairs), query one extra neighbor to exclude self-pair
    let dd_estimator = LandySzalayKnn::new(args.k_max + 1);
    let n_random = params.n_points * args.random_ratio;

    let mut all_xi: Vec<Vec<f64>> = Vec::with_capacity(args.n_mocks);
    let mut knn_times: Vec<f64> = Vec::with_capacity(args.n_mocks);

    // Corrfunc data (only populated when --corrfunc is active)
    let mut all_xi_corrfunc: Vec<Vec<f64>> = Vec::new();
    let mut corrfunc_times: Vec<f64> = Vec::new();
    let corrfunc_runner = if args.corrfunc {
        match CorrfuncRunner::new(
            std::path::Path::new(&args.output_dir),
            args.python.as_deref(),
        ) {
            Ok(runner) => {
                println!("Corrfunc python     = {}", runner.python());
                Some(runner)
            }
            Err(e) => {
                eprintln!("Warning: failed to initialize Corrfunc runner: {}", e);
                None
            }
        }
    } else {
        None
    };

    // CDF summary from level 0 RR (first mock, dilution path) or single measurement
    let mut cdf_rr_summary: Option<KnnCdfSummary> = None;
    let mut cdf_nbar: f64 = 0.0;

    for mock_idx in 0..args.n_mocks {
        let seed_data = (mock_idx * 2) as u64;
        let seed_rand = (mock_idx * 2 + 1) as u64 + 1_000_000;

        print!("Mock {:3}/{:3} ... ", mock_idx + 1, args.n_mocks);

        let mock = CoxMock::generate(&params, seed_data);
        let randoms = CoxMock::generate_randoms(n_random, params.box_size, seed_rand);

        // --- kNN block with timing ---
        let knn_start = Instant::now();

        let mock_xi = if args.dilution {
            // === Dilution path ===
            let dle = dilution_level_edges.as_ref().unwrap();
            let mut level_results: Vec<LevelResult> = Vec::new();

            // Level 0: full data
            {
                let l0_edges = &dle[0];
                let data_tree = PointTree::build(mock.positions.clone());

                let dd_raw = dd_estimator.query_distances_periodic(
                    &data_tree,
                    &mock.positions,
                    params.box_size,
                );
                let dd_dists = exclude_self_pairs(dd_raw, args.k_max);
                let dr_dists = estimator.query_distances_periodic(
                    &data_tree,
                    &randoms,
                    params.box_size,
                );

                // RR: subsample randoms to data density → k_max suffices
                let n_rr = params.n_points;
                let rr_points: Vec<[f64; 3]> = randoms[..n_rr].to_vec();
                let rr_tree = PointTree::build(rr_points.clone());
                let rr_raw = dd_estimator.query_distances_periodic(
                    &rr_tree,
                    &rr_points,
                    params.box_size,
                );
                let rr_dists = exclude_self_pairs(rr_raw, args.k_max);

                let dd = LandySzalayKnn::pair_count_density(&dd_dists, l0_edges);
                let dr = LandySzalayKnn::pair_count_density(&dr_dists, l0_edges);
                let rr = LandySzalayKnn::pair_count_density(&rr_dists, l0_edges);

                let xi_est = LandySzalayKnn::estimate_xi_ls(
                    &dd, &dr, &rr, params.n_points, n_rr,
                );

                // CDF computation for level 0 (single measurement, std=0)
                let k_vals = cdf_k_values(args.k_max);
                let cdf_r = cdf_r_grid(
                    *l0_edges.first().unwrap(),
                    *l0_edges.last().unwrap(),
                    150,
                );
                let cdf_dd_l0 = LandySzalayKnn::empirical_cdfs(&dd_dists, &k_vals, &cdf_r);
                let cdf_dr_l0 = LandySzalayKnn::empirical_cdfs(&dr_dists, &k_vals, &cdf_r);
                let cdf_rr_l0 = LandySzalayKnn::empirical_cdfs(&rr_dists, &k_vals, &cdf_r);

                let cdf_dd_sum = average_cdfs(&[cdf_dd_l0]);
                let cdf_dr_sum = average_cdfs(&[cdf_dr_l0]);
                let cdf_rr_sum = average_cdfs(&[cdf_rr_l0.clone()]);

                // Capture level 0 RR CDF for Erlang comparison on first mock
                if mock_idx == 0 {
                    cdf_rr_summary = Some(cdf_rr_sum.clone());
                    cdf_nbar = n_rr as f64 / params.volume();
                }

                let centers: Vec<f64> =
                    l0_edges.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect();
                level_results.push(LevelResult {
                    level: 0,
                    dilution_factor: 1,
                    r_centers: centers,
                    xi_mean: xi_est.xi,
                    xi_std: vec![0.0; l0_edges.len() - 1],
                    n_subsamples: 1,
                    cdf_dd: Some(cdf_dd_sum),
                    cdf_dr: Some(cdf_dr_sum),
                    cdf_rr: Some(cdf_rr_sum),
                });
            }

            // Levels 1..=max_level via DilutionLadder
            let data_ladder = DilutionLadder::build(
                params.n_points,
                args.max_dilution_level,
                seed_data + 2_000_000,
            );
            let rand_ladder = DilutionLadder::build(
                n_random,
                args.max_dilution_level,
                seed_rand + 2_000_000,
            );

            for level in 1..=args.max_dilution_level {
                if level >= data_ladder.levels.len() || level >= rand_ladder.levels.len() {
                    break;
                }
                if level >= dle.len() {
                    break;
                }

                let l_edges = &dle[level];
                let data_level = &data_ladder.levels[level];
                let rand_level = &rand_ladder.levels[level];
                let r_ell = data_level.dilution_factor;
                let k_eff =
                    DilutionLadder::effective_k_max(args.k_max, params.n_points, r_ell);
                if k_eff == 0 {
                    break;
                }

                let n_data_sub = params.n_points / r_ell;
                let n_rr_sub = n_data_sub; // dilute randoms to data density for RR
                let n_subs = data_level.subsamples.len();
                let n_bins_level = l_edges.len() - 1;

                let dd_sub_est = LandySzalayKnn::new(k_eff + 1);
                let dr_sub_est = LandySzalayKnn::new(k_eff);

                let k_vals = cdf_k_values(k_eff);
                let cdf_r = cdf_r_grid(
                    *l_edges.first().unwrap(),
                    *l_edges.last().unwrap(),
                    150,
                );

                let mut xi_subs: Vec<Vec<f64>> = Vec::with_capacity(n_subs);
                let mut cdf_dd_subs = Vec::with_capacity(n_subs);
                let mut cdf_dr_subs = Vec::with_capacity(n_subs);
                let mut cdf_rr_subs = Vec::with_capacity(n_subs);

                for sub_idx in 0..n_subs {
                    let data_sub: Vec<[f64; 3]> = data_level.subsamples[sub_idx]
                        .iter()
                        .map(|&i| mock.positions[i])
                        .collect();
                    let rand_sub: Vec<[f64; 3]> = rand_level.subsamples[sub_idx]
                        .iter()
                        .map(|&i| randoms[i])
                        .collect();

                    let data_sub_tree = PointTree::build(data_sub.clone());

                    // DD: data_sub → data_sub tree
                    let dd_raw = dd_sub_est.query_distances_periodic(
                        &data_sub_tree,
                        &data_sub,
                        params.box_size,
                    );
                    let dd_dists = exclude_self_pairs(dd_raw, k_eff);

                    // DR: full rand_sub → data_sub tree (more queries = less noise)
                    let dr_dists = dr_sub_est.query_distances_periodic(
                        &data_sub_tree,
                        &rand_sub,
                        params.box_size,
                    );

                    // RR: subsample rand_sub to data density → k_eff suffices
                    let rr_sub: Vec<[f64; 3]> = rand_sub[..n_rr_sub].to_vec();
                    let rr_sub_tree = PointTree::build(rr_sub.clone());
                    let rr_raw = dd_sub_est.query_distances_periodic(
                        &rr_sub_tree,
                        &rr_sub,
                        params.box_size,
                    );
                    let rr_dists = exclude_self_pairs(rr_raw, k_eff);

                    let dd = LandySzalayKnn::pair_count_density(&dd_dists, l_edges);
                    let dr = LandySzalayKnn::pair_count_density(&dr_dists, l_edges);
                    let rr = LandySzalayKnn::pair_count_density(&rr_dists, l_edges);

                    let xi_est = LandySzalayKnn::estimate_xi_ls(
                        &dd, &dr, &rr, n_data_sub, n_rr_sub,
                    );
                    xi_subs.push(xi_est.xi);

                    // CDFs from same distributions (no extra kNN queries)
                    cdf_dd_subs.push(LandySzalayKnn::empirical_cdfs(&dd_dists, &k_vals, &cdf_r));
                    cdf_dr_subs.push(LandySzalayKnn::empirical_cdfs(&dr_dists, &k_vals, &cdf_r));
                    cdf_rr_subs.push(LandySzalayKnn::empirical_cdfs(&rr_dists, &k_vals, &cdf_r));
                }

                // Average xi across subsamples
                let n_s = xi_subs.len() as f64;
                let xi_mean: Vec<f64> = (0..n_bins_level)
                    .map(|i| xi_subs.iter().map(|xi| xi[i]).sum::<f64>() / n_s)
                    .collect();
                let xi_std: Vec<f64> = (0..n_bins_level)
                    .map(|i| {
                        let m = xi_mean[i];
                        let var = xi_subs
                            .iter()
                            .map(|xi| (xi[i] - m).powi(2))
                            .sum::<f64>()
                            / (n_s - 1.0).max(1.0);
                        var.sqrt()
                    })
                    .collect();

                let centers: Vec<f64> =
                    l_edges.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect();
                level_results.push(LevelResult {
                    level,
                    dilution_factor: r_ell,
                    r_centers: centers,
                    xi_mean,
                    xi_std,
                    n_subsamples: n_subs,
                    cdf_dd: Some(average_cdfs(&cdf_dd_subs)),
                    cdf_dr: Some(average_cdfs(&cdf_dr_subs)),
                    cdf_rr: Some(average_cdfs(&cdf_rr_subs)),
                });
            }

            stitch_levels(&level_results).0.xi
        } else {
            // === Standard (non-dilution) path ===
            let data_tree = PointTree::build(mock.positions.clone());

            // DD: data → data tree (query k_max+1, then exclude self)
            let dd_raw = dd_estimator.query_distances_periodic(
                &data_tree,
                &mock.positions,
                params.box_size,
            );
            let dd_dists = exclude_self_pairs(dd_raw, args.k_max);

            // DR: randoms → data tree (full random catalog for low noise)
            let dr_dists = estimator.query_distances_periodic(
                &data_tree,
                &randoms,
                params.box_size,
            );

            // RR: subsample randoms to data density, then k_max suffices
            let n_rr = params.n_points;
            let rr_points: Vec<[f64; 3]> = randoms[..n_rr].to_vec();
            let rr_tree = PointTree::build(rr_points.clone());
            let rr_raw = dd_estimator.query_distances_periodic(
                &rr_tree,
                &rr_points,
                params.box_size,
            );
            let rr_dists = exclude_self_pairs(rr_raw, args.k_max);

            let dd = LandySzalayKnn::pair_count_density(&dd_dists, &r_edges);
            let dr = LandySzalayKnn::pair_count_density(&dr_dists, &r_edges);
            let rr = LandySzalayKnn::pair_count_density(&rr_dists, &r_edges);

            // n_rr matches n_data → rr_scale = 1.0 (correct for diluted RR)
            let xi_est = LandySzalayKnn::estimate_xi_ls(
                &dd, &dr, &rr, params.n_points, n_rr,
            );

            // CDF computation from same distributions (no extra kNN queries)
            if mock_idx == 0 {
                let k_vals = cdf_k_values(args.k_max);
                let cdf_r = cdf_r_grid(
                    *r_edges.first().unwrap(),
                    *r_edges.last().unwrap(),
                    150,
                );
                let cdf_rr_batch = LandySzalayKnn::empirical_cdfs(&rr_dists, &k_vals, &cdf_r);
                cdf_rr_summary = Some(average_cdfs(&[cdf_rr_batch]));
                cdf_nbar = n_rr as f64 / params.volume();
            }

            xi_est.xi
        };

        let knn_time = knn_start.elapsed().as_secs_f64();
        knn_times.push(knn_time);

        // Probe at a representative scale
        let r_probe = 10.0;
        let bin_probe = r_centers
            .iter()
            .position(|&r| r >= r_probe)
            .unwrap_or(0);

        // --- Corrfunc block ---
        if let Some(ref runner) = corrfunc_runner {
            let cache_key = CorrfuncRunner::cache_key(
                &args.preset,
                seed_data,
                args.r_min,
                r_max_eff,
                n_bins,
            );
            match runner.compute_xi(
                &mock.positions,
                &randoms,
                params.box_size,
                &r_edges,
                args.corrfunc_threads,
                &cache_key,
            ) {
                Ok(cf_result) => {
                    corrfunc_times.push(cf_result.wall_time_secs);
                    println!(
                        "xi_knn({:.0}) = {:.3}, xi_cf({:.0}) = {:.3}, knn: {:.2}s, cf: {:.2}s",
                        r_centers[bin_probe],
                        mock_xi[bin_probe],
                        r_centers[bin_probe],
                        cf_result.xi[bin_probe],
                        knn_time,
                        cf_result.wall_time_secs,
                    );
                    all_xi_corrfunc.push(cf_result.xi);
                }
                Err(e) => {
                    eprintln!(
                        "xi(r={:.0}) = {:.3}  (analytic: {:.3})  knn: {:.2}s  [Corrfunc error: {}]",
                        r_centers[bin_probe],
                        mock_xi[bin_probe],
                        xi_analytic[bin_probe],
                        knn_time,
                        e,
                    );
                }
            }
        } else {
            println!(
                "xi(r={:.0}) = {:.3}  (analytic: {:.3})  knn: {:.2}s",
                r_centers[bin_probe],
                mock_xi[bin_probe],
                xi_analytic[bin_probe],
                knn_time,
            );
        }

        all_xi.push(mock_xi);
    }

    // ========================================================================
    // Statistics
    // ========================================================================
    let n = args.n_mocks as f64;
    let mean_xi: Vec<f64> = (0..n_bins)
        .map(|i| all_xi.iter().map(|xi| xi[i]).sum::<f64>() / n)
        .collect();
    let std_xi: Vec<f64> = (0..n_bins)
        .map(|i| {
            let mean = mean_xi[i];
            let var =
                all_xi.iter().map(|xi| (xi[i] - mean).powi(2)).sum::<f64>() / (n - 1.0);
            var.sqrt()
        })
        .collect();
    let stderr_xi: Vec<f64> = std_xi.iter().map(|s| s / n.sqrt()).collect();

    let bias_sigma: Vec<f64> = (0..n_bins)
        .map(|i| {
            if stderr_xi[i] > 0.0 {
                (mean_xi[i] - xi_analytic[i]) / stderr_xi[i]
            } else {
                0.0
            }
        })
        .collect();

    // Corrfunc statistics (if available)
    let has_corrfunc = !all_xi_corrfunc.is_empty();
    let (mean_xi_corrfunc, std_xi_corrfunc) = if has_corrfunc {
        let n_cf = all_xi_corrfunc.len() as f64;
        let mean_cf: Vec<f64> = (0..n_bins)
            .map(|i| all_xi_corrfunc.iter().map(|xi| xi[i]).sum::<f64>() / n_cf)
            .collect();
        let std_cf: Vec<f64> = (0..n_bins)
            .map(|i| {
                let m = mean_cf[i];
                let var = all_xi_corrfunc
                    .iter()
                    .map(|xi| (xi[i] - m).powi(2))
                    .sum::<f64>()
                    / (n_cf - 1.0).max(1.0);
                var.sqrt()
            })
            .collect();
        (Some(mean_cf), Some(std_cf))
    } else {
        (None, None)
    };

    // Print summary table
    println!("\n=== Summary ===");
    if has_corrfunc {
        println!(
            "{:>8} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}",
            "r", "xi_true", "xi_knn", "sigma_knn", "bias/se", "xi_cf", "sigma_cf"
        );
        let mean_cf = mean_xi_corrfunc.as_ref().unwrap();
        let std_cf = std_xi_corrfunc.as_ref().unwrap();
        for i in 0..n_bins {
            println!(
                "{:8.1} {:12.4} {:12.4} {:12.4} {:12.2} {:12.4} {:12.4}",
                r_centers[i], xi_analytic[i], mean_xi[i], std_xi[i], bias_sigma[i],
                mean_cf[i], std_cf[i],
            );
        }
    } else {
        println!(
            "{:>8} {:>12} {:>12} {:>12} {:>12} {:>12}",
            "r", "xi_true", "xi_mean", "sigma_xi", "stderr", "bias/se"
        );
        for i in 0..n_bins {
            println!(
                "{:8.1} {:12.4} {:12.4} {:12.4} {:12.4} {:12.2}",
                r_centers[i], xi_analytic[i], mean_xi[i], std_xi[i], stderr_xi[i], bias_sigma[i]
            );
        }
    }

    let chi2: f64 = bias_sigma.iter().map(|b| b * b).sum();
    let max_bias = bias_sigma
        .iter()
        .map(|b| b.abs())
        .fold(0.0f64, f64::max);
    println!(
        "\nchi2/dof = {:.2}/{} = {:.3}",
        chi2,
        n_bins,
        chi2 / n_bins as f64
    );
    println!("max |bias|/sigma_mean = {:.2}", max_bias);

    if !knn_times.is_empty() {
        let avg_knn = knn_times.iter().sum::<f64>() / knn_times.len() as f64;
        println!(
            "avg kNN time: {:.3}s  ({} threads, rayon)",
            avg_knn, rayon_threads,
        );
    }
    if !corrfunc_times.is_empty() {
        let avg_cf = corrfunc_times.iter().sum::<f64>() / corrfunc_times.len() as f64;
        println!(
            "avg Corrfunc time: {:.3}s  ({} threads)",
            avg_cf, args.corrfunc_threads,
        );
    }

    // ========================================================================
    // TSV output
    // ========================================================================
    let tsv_path = format!("{}/xi_comparison.tsv", args.output_dir);
    let mut tsv = String::new();
    if has_corrfunc {
        tsv.push_str("# r\txi_analytic\txi_mean\txi_std\txi_stderr\tbias_sigma\txi_cf_mean\txi_cf_std\n");
        let mean_cf = mean_xi_corrfunc.as_ref().unwrap();
        let std_cf = std_xi_corrfunc.as_ref().unwrap();
        for i in 0..n_bins {
            tsv.push_str(&format!(
                "{:.2}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.4}\t{:.8}\t{:.8}\n",
                r_centers[i], xi_analytic[i], mean_xi[i], std_xi[i], stderr_xi[i],
                bias_sigma[i], mean_cf[i], std_cf[i],
            ));
        }
    } else {
        tsv.push_str("# r\txi_analytic\txi_mean\txi_std\txi_stderr\tbias_sigma\n");
        for i in 0..n_bins {
            tsv.push_str(&format!(
                "{:.2}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.4}\n",
                r_centers[i], xi_analytic[i], mean_xi[i], std_xi[i], stderr_xi[i], bias_sigma[i]
            ));
        }
    }
    std::fs::write(&tsv_path, &tsv).expect("Failed to write TSV");
    println!("\nWrote {}", tsv_path);

    // ========================================================================
    // Build shared PlotData
    // ========================================================================
    let plot_data = PlotData {
        r_centers: r_centers.clone(),
        r_smooth: r_smooth.clone(),
        xi_smooth: xi_smooth.clone(),
        mean_xi: mean_xi.clone(),
        std_xi: std_xi.clone(),
        bias_sigma: bias_sigma.clone(),
        all_xi: all_xi.clone(),
        line_length: params.line_length,
        cdf_rr_summary: cdf_rr_summary.clone(),
        cdf_nbar,
        corrfunc_mean_xi: mean_xi_corrfunc.clone(),
        corrfunc_std_xi: std_xi_corrfunc.clone(),
        corrfunc_all_xi: if has_corrfunc {
            Some(all_xi_corrfunc.clone())
        } else {
            None
        },
        knn_times: if !knn_times.is_empty() {
            Some(knn_times.clone())
        } else {
            None
        },
        corrfunc_times: if !corrfunc_times.is_empty() {
            Some(corrfunc_times.clone())
        } else {
            None
        },
        level_tags: composite_level_tags.clone(),
    };

    // ========================================================================
    // PLOTS (typst + lilaq)
    // ========================================================================
    println!("\nGenerating plots...");
    let plotter = TypstPlotter::new();
    let config = PlotConfig::default();

    // Wide config for summary
    let summary_config = PlotConfig {
        width_cm: 32.0,
        height_cm: 22.0,
        ..PlotConfig::default()
    };

    // 1. ξ(r) comparison
    let pdf = plotter.render_pdf(&plotting::render_xi_comparison(&plot_data, &config));
    let path = format!("{}/xi_vs_analytic.pdf", args.output_dir);
    std::fs::write(&path, &pdf).unwrap();
    println!("  Wrote {}", path);

    // 2. Residuals
    let pdf = plotter.render_pdf(&plotting::render_residuals(&plot_data, &config));
    let path = format!("{}/xi_residuals.pdf", args.output_dir);
    std::fs::write(&path, &pdf).unwrap();
    println!("  Wrote {}", path);

    // 3. r²ξ(r)
    let pdf = plotter.render_pdf(&plotting::render_r2xi(&plot_data, &config));
    let path = format!("{}/r2xi_comparison.pdf", args.output_dir);
    std::fs::write(&path, &pdf).unwrap();
    println!("  Wrote {}", path);

    // 4. CDF comparison
    let pdf = plotter.render_pdf(&plotting::render_cdf_comparison(&plot_data, &config));
    let path = format!("{}/cdf_comparison.pdf", args.output_dir);
    std::fs::write(&path, &pdf).unwrap();
    println!("  Wrote {}", path);

    // 5. Individual mocks
    let pdf = plotter.render_pdf(&plotting::render_individual_mocks(&plot_data, &config));
    let path = format!("{}/individual_mocks.pdf", args.output_dir);
    std::fs::write(&path, &pdf).unwrap();
    println!("  Wrote {}", path);

    // 6. Summary 2×2 figure
    let pdf = plotter.render_pdf(&plotting::render_summary_figure(&plot_data, &summary_config));
    let path = format!("{}/validation_summary.pdf", args.output_dir);
    std::fs::write(&path, &pdf).unwrap();
    println!("  Wrote {}", path);

    // --- Corrfunc-specific plots ---
    if has_corrfunc {
        let pdf = plotter.render_pdf(&plotting::render_xi_corrfunc_overlay(&plot_data, &config));
        let path = format!("{}/xi_corrfunc_overlay.pdf", args.output_dir);
        std::fs::write(&path, &pdf).unwrap();
        println!("  Wrote {}", path);

        let pdf = plotter.render_pdf(&plotting::render_xi_ratio(&plot_data, &config));
        let path = format!("{}/xi_ratio.pdf", args.output_dir);
        std::fs::write(&path, &pdf).unwrap();
        println!("  Wrote {}", path);

        let pdf = plotter.render_pdf(&plotting::render_timing(&plot_data, &config));
        let path = format!("{}/timing_comparison.pdf", args.output_dir);
        std::fs::write(&path, &pdf).unwrap();
        println!("  Wrote {}", path);
    }

    // --- Paper figure ---
    if args.paper {
        let pdf = plotter.render_pdf(&plotting::render_paper_xi_figure(&plot_data, args.n_mocks));
        let path = format!("{}/paper_xi_comparison.pdf", args.output_dir);
        std::fs::write(&path, &pdf).unwrap();
        println!("  Wrote {}", path);
    }

    // --- Dilution-specific plots ---
    if args.dilution {
        let pdf = plotter.render_pdf(&plotting::render_dilution_levels(&plot_data, &config));
        let path = format!("{}/dilution_levels.pdf", args.output_dir);
        std::fs::write(&path, &pdf).unwrap();
        println!("  Wrote {}", path);
    }

    // Interactive TUI explorer
    #[cfg(feature = "interactive")]
    if args.interactive {
        if let Err(e) = twopoint::explorer::run_explorer(&plot_data, &args.output_dir) {
            eprintln!("Explorer error: {}", e);
        }
    }

    // Final summary
    println!("\n=== Paper-ready summary ===");
    println!("N_mock = {}", args.n_mocks);
    println!(
        "chi2/dof = {:.2}/{} = {:.3}",
        chi2,
        n_bins,
        chi2 / n_bins as f64
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
    println!(
        "kNN: {} threads (rayon/bosque){}",
        rayon_threads,
        if args.corrfunc {
            format!(", Corrfunc: {} threads", args.corrfunc_threads)
        } else {
            String::new()
        },
    );
    println!("\nDone. SVGs in {}/", args.output_dir);
}

//! Per-point density profile smoke test.
//!
//! Runs 20 CoxMock realizations, computes per-point profiles using
//! both the cumulative (ξ̄) and differential (ξ via Savitzky-Golay)
//! estimators, then renders comparison PDFs showing mean ± σ
//! against the analytic ξ(r).
//!
//! Usage:
//!   cargo run --release --example test_per_point

use std::time::Instant;

use twopoint::estimator::{cdf_r_grid, savgol};
use twopoint::ladder::DilutionLadder;
use twopoint::mock::{CoxMock, CoxMockParams};
use twopoint::plotting::TypstPlotter;
use twopoint::{PerPointProfiles, RrMode};

fn fmt_array(values: &[f64]) -> String {
    let items: Vec<String> = values.iter().map(|v| format!("{:.8}", v)).collect();
    format!("({})", items.join(", "))
}

fn main() {
    let n_mocks = 20;
    let k_max = 32;
    let max_dilution_level = 1;
    let random_ratio = 5;
    let sg_half_window = 5;
    let sg_poly_order = 3;

    let params = CoxMockParams::validation();
    let box_size = params.box_size;
    let nbar = params.nbar();
    let n_random = params.n_points * random_ratio;

    // r-grid for evaluation
    let r_lo = 3.0;
    let r_hi = DilutionLadder::r_char_knn(
        DilutionLadder::effective_k_max(k_max, params.n_points, 8),
        8,
        nbar,
    )
    .min(box_size / 2.0);
    let n_grid = 200;
    let r_grid = cdf_r_grid(r_lo, r_hi, n_grid);

    println!("=== Per-Point Density Profile Test ===");
    println!("N_mocks  = {}", n_mocks);
    println!("N_points = {}", params.n_points);
    println!("N_random = {}", n_random);
    println!("k_max    = {}", k_max);
    println!("box_size = {:.0}", box_size);
    println!("nbar     = {:.2e}", nbar);
    println!("r range  = [{:.1}, {:.1}]", r_lo, r_hi);
    println!("SG window= {} (full width {})", sg_half_window, 2 * sg_half_window + 1);
    println!("SG poly  = {}", sg_poly_order);
    println!();

    // Analytic ξ(r) and ξ̄(<r)
    let xi_analytic: Vec<f64> = r_grid.iter().map(|&r| params.xi_analytic(r)).collect();
    let xi_bar_analytic: Vec<f64> = r_grid
        .iter()
        .map(|&r| {
            if r <= 0.0 {
                return 0.0;
            }
            let ell = params.line_length;
            let np = params.n_points as f64;
            let nl = params.n_lines as f64;
            let nbar_val = params.nbar();
            let c = (np - 1.0)
                / (nl * nbar_val * 2.0 * std::f64::consts::PI * ell);
            if r >= ell {
                // ξ̄(<r) = (3/r³) ∫₀ˡ ξ(s)s² ds  (integral saturates at ℓ)
                // ∫₀ˡ C/s²·(1−s/ℓ)·s² ds = C[ℓ − ℓ/2] = Cℓ/2
                3.0 * c * ell / (2.0 * r * r * r) * r.powi(3).recip() * r.powi(3)
                    // simplify: 3Cℓ/(2r³)
            } else {
                // ξ̄(<r) = (3/r³) ∫₀ʳ C(1−s/ℓ) ds = 3C/r²·(1 − r/(2ℓ))
                3.0 * c / (r * r) * (1.0 - r / (2.0 * ell))
            }
        })
        .collect();

    // Per-mock: collect mean ξ̄ and mean ξ_diff per grid point
    let mut all_xi_bar: Vec<Vec<f64>> = Vec::with_capacity(n_mocks);
    let mut all_xi_diff: Vec<Vec<f64>> = Vec::with_capacity(n_mocks);

    let t0 = Instant::now();
    for mock_idx in 0..n_mocks {
        let t_mock = Instant::now();
        let seed_data = (mock_idx * 2) as u64;
        let seed_rand = (mock_idx * 2 + 1) as u64 + 1_000_000;

        let mock = CoxMock::generate(&params, seed_data);
        let randoms = CoxMock::generate_randoms(n_random, box_size, seed_rand);

        let ladder = DilutionLadder::build(
            mock.positions.len(),
            max_dilution_level,
            seed_data + 999,
        );

        let profiles = PerPointProfiles::compute(
            &mock.positions,
            &randoms,
            &ladder,
            k_max,
            box_size,
        );

        // Cumulative (volume-averaged) ξ̄
        let mean_bar = profiles.mean_xi(&r_grid, &RrMode::Analytic);
        all_xi_bar.push(mean_bar.xi);

        // Differential ξ via SG
        let mean_diff =
            profiles.mean_xi_differential(&r_grid, &RrMode::Analytic, sg_half_window, sg_poly_order);
        all_xi_diff.push(mean_diff.xi);

        let dt = t_mock.elapsed().as_secs_f64();
        let xi_probe = all_xi_diff.last().unwrap();
        let r_probe_idx = r_grid.iter().position(|&r| r >= 20.0).unwrap_or(0);
        println!(
            "Mock {:2}/{}: xi_diff(r=20) = {:.4}  (analytic: {:.4})  {:.2}s",
            mock_idx + 1,
            n_mocks,
            xi_probe[r_probe_idx],
            xi_analytic[r_probe_idx],
            dt,
        );
    }
    let total = t0.elapsed().as_secs_f64();
    println!("\nTotal time: {:.1}s ({:.2}s/mock)", total, total / n_mocks as f64);

    // ========================================================================
    // Statistics
    // ========================================================================
    let n = n_mocks as f64;

    let mean_xi_bar: Vec<f64> = (0..n_grid)
        .map(|i| all_xi_bar.iter().map(|xi| xi[i]).sum::<f64>() / n)
        .collect();
    let std_xi_bar: Vec<f64> = (0..n_grid)
        .map(|i| {
            let m = mean_xi_bar[i];
            let var = all_xi_bar
                .iter()
                .map(|xi| (xi[i] - m).powi(2))
                .sum::<f64>()
                / (n - 1.0);
            var.sqrt()
        })
        .collect();
    let stderr_xi_bar: Vec<f64> = std_xi_bar.iter().map(|s| s / n.sqrt()).collect();

    let mean_xi_diff: Vec<f64> = (0..n_grid)
        .map(|i| all_xi_diff.iter().map(|xi| xi[i]).sum::<f64>() / n)
        .collect();
    let std_xi_diff: Vec<f64> = (0..n_grid)
        .map(|i| {
            let m = mean_xi_diff[i];
            let var = all_xi_diff
                .iter()
                .map(|xi| (xi[i] - m).powi(2))
                .sum::<f64>()
                / (n - 1.0);
            var.sqrt()
        })
        .collect();
    let stderr_xi_diff: Vec<f64> = std_xi_diff.iter().map(|s| s / n.sqrt()).collect();

    // SG-smooth the analytic ξ̄ → ξ recovery (shows how well the inversion works)
    let h = r_grid[1] - r_grid[0];
    let (xi_bar_smooth, dxi_bar_dr) =
        savgol::sg_smooth_diff(&xi_bar_analytic, h, sg_half_window, sg_poly_order);
    let xi_from_analytic_bar: Vec<f64> = r_grid
        .iter()
        .enumerate()
        .map(|(i, &r)| xi_bar_smooth[i] + (r / 3.0) * dxi_bar_dr[i])
        .collect();

    // Print summary at a few scales
    println!("\n=== Summary ===");
    println!(
        "{:>8} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "r", "xi_true", "xi_bar", "±stderr", "xi_diff", "±stderr"
    );
    for &r_probe in &[5.0, 10.0, 20.0, 40.0, 60.0, 80.0] {
        if let Some(idx) = r_grid.iter().position(|&r| r >= r_probe) {
            println!(
                "{:8.1} {:12.4} {:12.4} {:12.4} {:12.4} {:12.4}",
                r_grid[idx],
                xi_analytic[idx],
                mean_xi_bar[idx],
                stderr_xi_bar[idx],
                mean_xi_diff[idx],
                stderr_xi_diff[idx],
            );
        }
    }

    // ========================================================================
    // Plots
    // ========================================================================
    let output_dir = "plots/per_point";
    std::fs::create_dir_all(output_dir).unwrap();
    let plotter = TypstPlotter::new();

    // --- Plot 1: ξ̄(<r) cumulative estimator vs analytic ---
    {
        let r = fmt_array(&r_grid);
        let xi_a = fmt_array(&xi_bar_analytic);
        let xi_m = fmt_array(&mean_xi_bar);
        let xi_lo: Vec<f64> = mean_xi_bar
            .iter()
            .zip(stderr_xi_bar.iter())
            .map(|(&m, &s)| m - s)
            .collect();
        let xi_hi: Vec<f64> = mean_xi_bar
            .iter()
            .zip(stderr_xi_bar.iter())
            .map(|(&m, &s)| m + s)
            .collect();
        let xi_sd_lo: Vec<f64> = mean_xi_bar
            .iter()
            .zip(std_xi_bar.iter())
            .map(|(&m, &s)| m - s)
            .collect();
        let xi_sd_hi: Vec<f64> = mean_xi_bar
            .iter()
            .zip(std_xi_bar.iter())
            .map(|(&m, &s)| m + s)
            .collect();
        let lo = fmt_array(&xi_lo);
        let hi = fmt_array(&xi_hi);
        let sd_lo = fmt_array(&xi_sd_lo);
        let sd_hi = fmt_array(&xi_sd_hi);

        let src = format!(
            r##"#set page(width: 18cm, height: 12cm, margin: 5mm)
#set text(font: "New Computer Modern")
#import "@preview/lilaq:0.6.0" as lq

#let r = {r}
#let xi-analytic = {xi_a}
#let xi-mean = {xi_m}
#let lo = {lo}
#let hi = {hi}
#let sd-lo = {sd_lo}
#let sd-hi = {sd_hi}

#lq.diagram(
  title: [Per-point cumulative $overline(xi)(<r)$: {n_mocks} CoxMock realizations],
  xlabel: [$r$ #h(0.3em) $[$h$""^(-1)$ Mpc$]$],
  ylabel: [$overline(xi)(<r)$],
  xscale: lq.scale.symlog(threshold: 0.01),
  yscale: lq.scale.symlog(threshold: 0.001),
  lq.fill-between(r, sd-lo, y2: sd-hi,
    fill: blue.lighten(85%), stroke: none,
    label: [$plus.minus 1 sigma$ (mock scatter)]),
  lq.fill-between(r, lo, y2: hi,
    fill: blue.lighten(60%), stroke: none,
    label: [$plus.minus sigma / sqrt({n_mocks})$]),
  lq.plot(r, xi-analytic,
    label: [Analytic $overline(xi)(<r)$],
    stroke: (dash: "dashed", paint: black, thickness: 1.5pt)),
  lq.plot(r, xi-mean,
    label: [kNN per-point mean],
    stroke: (paint: blue, thickness: 1.5pt)),
)
"##,
            r = r,
            xi_a = xi_a,
            xi_m = xi_m,
            lo = lo,
            hi = hi,
            sd_lo = sd_lo,
            sd_hi = sd_hi,
            n_mocks = n_mocks,
        );
        let pdf = plotter.render_pdf(&src);
        let path = format!("{}/xi_bar_cumulative.pdf", output_dir);
        std::fs::write(&path, &pdf).unwrap();
        println!("\nWrote {}", path);
    }

    // --- Plot 2: ξ(r) differential estimator vs analytic ---
    {
        let r = fmt_array(&r_grid);
        let xi_a = fmt_array(&xi_analytic);
        let xi_m = fmt_array(&mean_xi_diff);
        let xi_lo: Vec<f64> = mean_xi_diff
            .iter()
            .zip(stderr_xi_diff.iter())
            .map(|(&m, &s)| m - s)
            .collect();
        let xi_hi: Vec<f64> = mean_xi_diff
            .iter()
            .zip(stderr_xi_diff.iter())
            .map(|(&m, &s)| m + s)
            .collect();
        let xi_sd_lo: Vec<f64> = mean_xi_diff
            .iter()
            .zip(std_xi_diff.iter())
            .map(|(&m, &s)| m - s)
            .collect();
        let xi_sd_hi: Vec<f64> = mean_xi_diff
            .iter()
            .zip(std_xi_diff.iter())
            .map(|(&m, &s)| m + s)
            .collect();
        let lo = fmt_array(&xi_lo);
        let hi = fmt_array(&xi_hi);
        let sd_lo = fmt_array(&xi_sd_lo);
        let sd_hi = fmt_array(&xi_sd_hi);
        let xi_inv = fmt_array(&xi_from_analytic_bar);

        let src = format!(
            r##"#set page(width: 18cm, height: 12cm, margin: 5mm)
#set text(font: "New Computer Modern")
#import "@preview/lilaq:0.6.0" as lq

#let r = {r}
#let xi-analytic = {xi_a}
#let xi-mean = {xi_m}
#let lo = {lo}
#let hi = {hi}
#let sd-lo = {sd_lo}
#let sd-hi = {sd_hi}
#let xi-inv = {xi_inv}

#lq.diagram(
  title: [Per-point differential $xi(r)$ via SG inversion: {n_mocks} mocks],
  xlabel: [$r$ #h(0.3em) $[$h$""^(-1)$ Mpc$]$],
  ylabel: [$xi(r)$],
  xscale: lq.scale.symlog(threshold: 0.01),
  yscale: lq.scale.symlog(threshold: 0.001),
  lq.fill-between(r, sd-lo, y2: sd-hi,
    fill: red.lighten(85%), stroke: none,
    label: [$plus.minus 1 sigma$ (mock scatter)]),
  lq.fill-between(r, lo, y2: hi,
    fill: red.lighten(60%), stroke: none,
    label: [$plus.minus sigma / sqrt({n_mocks})$]),
  lq.plot(r, xi-analytic,
    label: [Analytic $xi(r)$],
    stroke: (dash: "dashed", paint: black, thickness: 1.5pt)),
  lq.plot(r, xi-inv,
    label: [SG inversion of analytic $overline(xi)$],
    stroke: (dash: "dotted", paint: gray, thickness: 1.2pt)),
  lq.plot(r, xi-mean,
    label: [kNN per-point mean (SG)],
    stroke: (paint: red, thickness: 1.5pt)),
)
"##,
            r = r,
            xi_a = xi_a,
            xi_m = xi_m,
            lo = lo,
            hi = hi,
            sd_lo = sd_lo,
            sd_hi = sd_hi,
            xi_inv = xi_inv,
            n_mocks = n_mocks,
        );
        let pdf = plotter.render_pdf(&src);
        let path = format!("{}/xi_differential_sg.pdf", output_dir);
        std::fs::write(&path, &pdf).unwrap();
        println!("Wrote {}", path);
    }

    // --- Plot 3: side-by-side comparison on same axes ---
    {
        let r = fmt_array(&r_grid);
        let xi_a = fmt_array(&xi_analytic);
        let xi_bar_a = fmt_array(&xi_bar_analytic);
        let xi_bar_m = fmt_array(&mean_xi_bar);
        let xi_diff_m = fmt_array(&mean_xi_diff);
        let xi_bar_se = fmt_array(&stderr_xi_bar);
        let xi_diff_se = fmt_array(&stderr_xi_diff);

        let src = format!(
            r##"#set page(width: 18cm, height: 12cm, margin: 5mm)
#set text(font: "New Computer Modern")
#import "@preview/lilaq:0.6.0" as lq

#let r = {r}
#let xi-analytic = {xi_a}
#let xi-bar-analytic = {xi_bar_a}
#let xi-bar-mean = {xi_bar_m}
#let xi-diff-mean = {xi_diff_m}
#let xi-bar-se = {xi_bar_se}
#let xi-diff-se = {xi_diff_se}

#lq.diagram(
  title: [Cumulative $overline(xi)$ vs Differential $xi$: {n_mocks} CoxMock mocks],
  xlabel: [$r$ #h(0.3em) $[$h$""^(-1)$ Mpc$]$],
  ylabel: [$xi$],
  xscale: lq.scale.symlog(threshold: 0.01),
  yscale: lq.scale.symlog(threshold: 0.001),
  lq.plot(r, xi-analytic,
    label: [Analytic $xi(r)$],
    stroke: (dash: "dashed", paint: black, thickness: 1.5pt)),
  lq.plot(r, xi-bar-analytic,
    label: [Analytic $overline(xi)(<r)$],
    stroke: (dash: "dotted", paint: gray, thickness: 1.2pt)),
  lq.plot(r, xi-bar-mean,
    yerr: xi-bar-se,
    label: [Per-point $overline(xi)$ mean],
    stroke: (paint: blue, thickness: 1.5pt)),
  lq.plot(r, xi-diff-mean,
    yerr: xi-diff-se,
    label: [Per-point $xi$ (SG) mean],
    stroke: (paint: red, thickness: 1.5pt)),
)
"##,
            r = r,
            xi_a = xi_a,
            xi_bar_a = xi_bar_a,
            xi_bar_m = xi_bar_m,
            xi_diff_m = xi_diff_m,
            xi_bar_se = xi_bar_se,
            xi_diff_se = xi_diff_se,
            n_mocks = n_mocks,
        );
        let pdf = plotter.render_pdf(&src);
        let path = format!("{}/xi_comparison.pdf", output_dir);
        std::fs::write(&path, &pdf).unwrap();
        println!("Wrote {}", path);
    }

    // --- Plot 4: r²ξ(r) comparison (amplifies intermediate scales) ---
    {
        let r2_xi_analytic: Vec<f64> = r_grid
            .iter()
            .zip(xi_analytic.iter())
            .map(|(&r, &xi)| r * r * xi)
            .collect();
        let r2_xi_diff: Vec<f64> = r_grid
            .iter()
            .zip(mean_xi_diff.iter())
            .map(|(&r, &xi)| r * r * xi)
            .collect();
        let r2_stderr: Vec<f64> = r_grid
            .iter()
            .zip(stderr_xi_diff.iter())
            .map(|(&r, &se)| r * r * se)
            .collect();
        let r2_std: Vec<f64> = r_grid
            .iter()
            .zip(std_xi_diff.iter())
            .map(|(&r, &s)| r * r * s)
            .collect();
        let r2_lo: Vec<f64> = r2_xi_diff
            .iter()
            .zip(r2_std.iter())
            .map(|(&m, &s)| m - s)
            .collect();
        let r2_hi: Vec<f64> = r2_xi_diff
            .iter()
            .zip(r2_std.iter())
            .map(|(&m, &s)| m + s)
            .collect();

        let r = fmt_array(&r_grid);
        let r2xi_a = fmt_array(&r2_xi_analytic);
        let r2xi_m = fmt_array(&r2_xi_diff);
        let r2xi_se = fmt_array(&r2_stderr);
        let sd_lo = fmt_array(&r2_lo);
        let sd_hi = fmt_array(&r2_hi);

        let src = format!(
            r##"#set page(width: 18cm, height: 12cm, margin: 5mm)
#set text(font: "New Computer Modern")
#import "@preview/lilaq:0.6.0" as lq

#let r = {r}
#let r2xi-analytic = {r2xi_a}
#let r2xi-mean = {r2xi_m}
#let r2xi-stderr = {r2xi_se}
#let sd-lo = {sd_lo}
#let sd-hi = {sd_hi}

#lq.diagram(
  title: [$r^2 xi(r)$ — differential estimator ({n_mocks} mocks)],
  xlabel: [$r$ #h(0.3em) $[$h$""^(-1)$ Mpc$]$],
  ylabel: [$r^2 xi(r)$],
  lq.fill-between(r, sd-lo, y2: sd-hi,
    fill: red.lighten(85%), stroke: none,
    label: [$plus.minus 1 sigma$]),
  lq.plot(r, r2xi-analytic,
    label: [Analytic],
    stroke: (dash: "dashed", paint: black, thickness: 1.5pt)),
  lq.plot(r, r2xi-mean,
    yerr: r2xi-stderr,
    label: [kNN per-point mean $plus.minus$ stderr],
    stroke: (paint: red, thickness: 1.5pt)),
)
"##,
            r = r,
            r2xi_a = r2xi_a,
            r2xi_m = r2xi_m,
            r2xi_se = r2xi_se,
            sd_lo = sd_lo,
            sd_hi = sd_hi,
            n_mocks = n_mocks,
        );
        let pdf = plotter.render_pdf(&src);
        let path = format!("{}/r2xi_differential.pdf", output_dir);
        std::fs::write(&path, &pdf).unwrap();
        println!("Wrote {}", path);
    }

    // --- Plot 5: individual mock ξ(r) curves ---
    {
        let r = fmt_array(&r_grid);
        let xi_a = fmt_array(&xi_analytic);
        let mut mock_plots = String::new();
        let colors = [
            "blue", "red", "green", "purple", "orange", "teal", "maroon",
            "navy", "olive", "gray",
        ];
        for (mock_idx, xi) in all_xi_diff.iter().enumerate() {
            let xi_arr = fmt_array(xi);
            let color = colors[mock_idx % colors.len()];
            mock_plots.push_str(&format!(
                "  lq.plot(r, {xi_arr}, stroke: (paint: {color}.lighten(50%), thickness: 0.5pt)),\n",
            ));
        }

        let src = format!(
            r##"#set page(width: 18cm, height: 12cm, margin: 5mm)
#set text(font: "New Computer Modern")
#import "@preview/lilaq:0.6.0" as lq

#let r = {r}
#let xi-analytic = {xi_a}

#lq.diagram(
  title: [Individual mock $xi(r)$ — differential estimator],
  xlabel: [$r$ #h(0.3em) $[$h$""^(-1)$ Mpc$]$],
  ylabel: [$xi(r)$],
  xscale: lq.scale.symlog(threshold: 0.01),
  yscale: lq.scale.symlog(threshold: 0.001),
{mock_plots}  lq.plot(r, xi-analytic,
    label: [Analytic $xi(r)$],
    stroke: (paint: black, thickness: 2pt, dash: "dashed")),
)
"##,
            r = r,
            xi_a = xi_a,
            mock_plots = mock_plots,
        );
        let pdf = plotter.render_pdf(&src);
        let path = format!("{}/individual_mocks.pdf", output_dir);
        std::fs::write(&path, &pdf).unwrap();
        println!("Wrote {}", path);
    }

    // --- Plot 6: kNN-PDF  dN/dr  (pair-count density) ---
    //
    // The cumulative neighbor count is N(<r) = n̄_D · V(r) · (1 + ξ̄(<r)).
    // Its derivative gives the pair-count density (the kNN-PDF):
    //   dN/dr = n̄_D · 4πr² · (1 + ξ(r))
    //
    // We compute this per-mock from the ξ̄ data we already have,
    // via SG differentiation of N(<r).
    {
        let four_pi = 4.0 * std::f64::consts::PI;

        // Per-mock: reconstruct N(<r) from ξ̄, then SG-differentiate
        let mut all_dndr: Vec<Vec<f64>> = Vec::with_capacity(n_mocks);
        for xi_bar in &all_xi_bar {
            let n_of_r: Vec<f64> = r_grid
                .iter()
                .zip(xi_bar.iter())
                .map(|(&r, &xb)| nbar * twopoint::estimator::volume(r) * (1.0 + xb))
                .collect();
            let (_, dn_dr) = savgol::sg_smooth_diff(&n_of_r, h, sg_half_window, sg_poly_order);
            all_dndr.push(dn_dr);
        }

        let mean_dndr: Vec<f64> = (0..n_grid)
            .map(|i| all_dndr.iter().map(|d| d[i]).sum::<f64>() / n)
            .collect();
        let std_dndr: Vec<f64> = (0..n_grid)
            .map(|i| {
                let m = mean_dndr[i];
                let var = all_dndr
                    .iter()
                    .map(|d| (d[i] - m).powi(2))
                    .sum::<f64>()
                    / (n - 1.0);
                var.sqrt()
            })
            .collect();

        // Analytic dN/dr = n̄ · 4πr² · (1 + ξ(r))
        let dndr_analytic: Vec<f64> = r_grid
            .iter()
            .zip(xi_analytic.iter())
            .map(|(&r, &xi)| nbar * four_pi * r * r * (1.0 + xi))
            .collect();

        // Poisson baseline: n̄ · 4πr²
        let dndr_poisson: Vec<f64> = r_grid
            .iter()
            .map(|&r| nbar * four_pi * r * r)
            .collect();

        let sd_lo: Vec<f64> = mean_dndr.iter().zip(std_dndr.iter()).map(|(&m, &s)| m - s).collect();
        let sd_hi: Vec<f64> = mean_dndr.iter().zip(std_dndr.iter()).map(|(&m, &s)| m + s).collect();

        let r = fmt_array(&r_grid);
        let src = format!(
            r##"#set page(width: 18cm, height: 12cm, margin: 5mm)
#set text(font: "New Computer Modern")
#import "@preview/lilaq:0.6.0" as lq

#let r = {r}
#let dndr-analytic = {dndr_a}
#let dndr-poisson = {dndr_p}
#let dndr-mean = {dndr_m}
#let sd-lo = {sd_lo}
#let sd-hi = {sd_hi}

#lq.diagram(
  title: [kNN-PDF: pair-count density $d N / d r$ ({n_mocks} mocks)],
  xlabel: [$r$ #h(0.3em) $[$h$""^(-1)$ Mpc$]$],
  ylabel: [$d angle.l N angle.r slash d r$],
  lq.fill-between(r, sd-lo, y2: sd-hi,
    fill: blue.lighten(85%), stroke: none,
    label: [$plus.minus 1 sigma$]),
  lq.plot(r, dndr-poisson,
    label: [Poisson: $overline(n) dot 4 pi r^2$],
    stroke: (dash: "dotted", paint: gray, thickness: 1.2pt)),
  lq.plot(r, dndr-analytic,
    label: [Analytic: $overline(n) dot 4 pi r^2 (1 + xi)$],
    stroke: (dash: "dashed", paint: black, thickness: 1.5pt)),
  lq.plot(r, dndr-mean,
    label: [kNN SG-smoothed mean],
    stroke: (paint: blue, thickness: 1.5pt)),
)
"##,
            r = r,
            dndr_a = fmt_array(&dndr_analytic),
            dndr_p = fmt_array(&dndr_poisson),
            dndr_m = fmt_array(&mean_dndr),
            sd_lo = fmt_array(&sd_lo),
            sd_hi = fmt_array(&sd_hi),
            n_mocks = n_mocks,
        );
        let pdf = plotter.render_pdf(&src);
        let path = format!("{}/knn_pdf_dndr.pdf", output_dir);
        std::fs::write(&path, &pdf).unwrap();
        println!("Wrote {}", path);
    }

    // --- Plot 7: kNN-PDF in volume space  dN/dV / n̄  = 1 + ξ(r) ---
    //
    // The volume-space density removes the 4πr² geometric factor,
    // leaving a quantity that is flat (= 1) for Poisson and elevated
    // by 1 + ξ(r) for a clustered field.
    {
        let four_pi = 4.0 * std::f64::consts::PI;

        // dN/dV = (dN/dr) / (4πr²), normalized by n̄ → 1 + ξ(r)
        let mut all_rho_norm: Vec<Vec<f64>> = Vec::with_capacity(n_mocks);
        for xi_bar in &all_xi_bar {
            let n_of_r: Vec<f64> = r_grid
                .iter()
                .zip(xi_bar.iter())
                .map(|(&r, &xb)| nbar * twopoint::estimator::volume(r) * (1.0 + xb))
                .collect();
            let (_, dn_dr) = savgol::sg_smooth_diff(&n_of_r, h, sg_half_window, sg_poly_order);
            let rho_norm: Vec<f64> = r_grid
                .iter()
                .zip(dn_dr.iter())
                .map(|(&r, &d)| {
                    let shell = nbar * four_pi * r * r;
                    if shell > 0.0 { d / shell } else { 0.0 }
                })
                .collect();
            all_rho_norm.push(rho_norm);
        }

        let mean_rho: Vec<f64> = (0..n_grid)
            .map(|i| all_rho_norm.iter().map(|rn| rn[i]).sum::<f64>() / n)
            .collect();
        let std_rho: Vec<f64> = (0..n_grid)
            .map(|i| {
                let m = mean_rho[i];
                let var = all_rho_norm
                    .iter()
                    .map(|rn| (rn[i] - m).powi(2))
                    .sum::<f64>()
                    / (n - 1.0);
                var.sqrt()
            })
            .collect();

        // Analytic: 1 + ξ(r)
        let one_plus_xi: Vec<f64> = xi_analytic.iter().map(|&xi| 1.0 + xi).collect();

        let sd_lo: Vec<f64> = mean_rho.iter().zip(std_rho.iter()).map(|(&m, &s)| m - s).collect();
        let sd_hi: Vec<f64> = mean_rho.iter().zip(std_rho.iter()).map(|(&m, &s)| m + s).collect();

        let r = fmt_array(&r_grid);
        let src = format!(
            r##"#set page(width: 18cm, height: 12cm, margin: 5mm)
#set text(font: "New Computer Modern")
#import "@preview/lilaq:0.6.0" as lq

#let r = {r}
#let one-plus-xi = {opxi}
#let rho-mean = {rho_m}
#let sd-lo = {sd_lo}
#let sd-hi = {sd_hi}

#lq.diagram(
  title: [Volume-space density $(d N slash d V) slash overline(n)$ ({n_mocks} mocks)],
  xlabel: [$r$ #h(0.3em) $[$h$""^(-1)$ Mpc$]$],
  ylabel: [$(d N slash d V) slash overline(n) = 1 + xi(r)$],
  xscale: lq.scale.symlog(threshold: 0.01),
  yscale: lq.scale.symlog(threshold: 0.01),
  lq.fill-between(r, sd-lo, y2: sd-hi,
    fill: green.lighten(85%), stroke: none,
    label: [$plus.minus 1 sigma$]),
  lq.plot(r, {ones},
    label: [Poisson ($= 1$)],
    stroke: (dash: "dotted", paint: gray, thickness: 1pt)),
  lq.plot(r, one-plus-xi,
    label: [Analytic $1 + xi(r)$],
    stroke: (dash: "dashed", paint: black, thickness: 1.5pt)),
  lq.plot(r, rho-mean,
    label: [kNN SG-smoothed mean],
    stroke: (paint: green.darken(20%), thickness: 1.5pt)),
)
"##,
            r = r,
            opxi = fmt_array(&one_plus_xi),
            rho_m = fmt_array(&mean_rho),
            sd_lo = fmt_array(&sd_lo),
            sd_hi = fmt_array(&sd_hi),
            ones = fmt_array(&vec![1.0; n_grid]),
            n_mocks = n_mocks,
        );
        let pdf = plotter.render_pdf(&src);
        let path = format!("{}/knn_pdf_volume_density.pdf", output_dir);
        std::fs::write(&path, &pdf).unwrap();
        println!("Wrote {}", path);
    }

    println!("\nDone! Plots in {}/", output_dir);
}

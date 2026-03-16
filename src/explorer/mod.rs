//! Interactive plot explorer for twopoint-validate.
//!
//! Serves plots via a tiny localhost HTTP server and opens them in the default
//! browser. The browser captures keyboard shortcuts and sends them back;
//! the terminal also accepts keys as a fallback.

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::terminal;
use kuva::backend::svg::SvgBackend;
use kuva::plot::{BandPlot, LinePlot, ScatterPlot};
use kuva::render::annotations::ReferenceLine;
use kuva::render::figure::Figure;
use kuva::render::layout::Layout;
use kuva::render::plots::Plot;
use kuva::render::render::render_multiple;
use std::io::{Read as _, Write};
use std::net::TcpListener;
use std::process::Command;
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

/// Erlang CDF for the k-th nearest neighbor at distance r, given number density.
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

/// Erlang PDF (derivative of CDF w.r.t. r) for the k-th nearest neighbor.
///
/// f_k(r) = nbar · 4π r² · exp(−λ) · λ^(k−1) / (k−1)!
/// where λ = nbar · (4π/3) · r³.
fn erlang_pdf(k: usize, r: f64, nbar: f64) -> f64 {
    let lambda = nbar * 4.0 / 3.0 * std::f64::consts::PI * r.powi(3);
    let dlambda_dr = nbar * 4.0 * std::f64::consts::PI * r.powi(2);
    if k == 0 || r <= 0.0 {
        return 0.0;
    }
    // log(λ^(k-1) / (k-1)!) = (k-1)*ln(λ) - ln((k-1)!)
    let log_term = (k as f64 - 1.0) * lambda.ln()
        - (1..k).map(|j| (j as f64).ln()).sum::<f64>();
    dlambda_dr * (-lambda + log_term).exp()
}

/// All computed arrays needed for plotting, populated from validate.rs after the mock loop.
pub struct PlotData {
    pub r_centers: Vec<f64>,
    pub r_smooth: Vec<f64>,
    pub xi_smooth: Vec<f64>,
    pub mean_xi: Vec<f64>,
    pub std_xi: Vec<f64>,
    pub bias_sigma: Vec<f64>,
    pub all_xi: Vec<Vec<f64>>,
    pub line_length: f64,
    pub cdf_rr_summary: Option<crate::ladder::KnnCdfSummary>,
    pub cdf_nbar: f64,
    // Corrfunc reference data (all Optional so explorer works without it)
    pub corrfunc_mean_xi: Option<Vec<f64>>,
    pub corrfunc_std_xi: Option<Vec<f64>>,
    pub corrfunc_all_xi: Option<Vec<Vec<f64>>>,
    pub knn_times: Option<Vec<f64>>,
    pub corrfunc_times: Option<Vec<f64>>,
    pub level_tags: Option<Vec<usize>>,
}

/// Which plot is currently displayed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlotKind {
    XiComparison,
    Residuals,
    R2Xi,
    CdfComparison,
    PeakedCdf,
    IndividualMocks,
    XiRatio,
    TimingComparison,
    Summary,
}

impl PlotKind {
    const ALL: [PlotKind; 9] = [
        PlotKind::XiComparison,
        PlotKind::Residuals,
        PlotKind::R2Xi,
        PlotKind::CdfComparison,
        PlotKind::PeakedCdf,
        PlotKind::IndividualMocks,
        PlotKind::XiRatio,
        PlotKind::TimingComparison,
        PlotKind::Summary,
    ];

    /// The 8 non-summary panels used in the summary figure.
    const PANELS: [PlotKind; 8] = [
        PlotKind::XiComparison,
        PlotKind::Residuals,
        PlotKind::R2Xi,
        PlotKind::CdfComparison,
        PlotKind::PeakedCdf,
        PlotKind::IndividualMocks,
        PlotKind::XiRatio,
        PlotKind::TimingComparison,
    ];

    fn index(self) -> usize {
        match self {
            PlotKind::XiComparison => 0,
            PlotKind::Residuals => 1,
            PlotKind::R2Xi => 2,
            PlotKind::CdfComparison => 3,
            PlotKind::PeakedCdf => 4,
            PlotKind::IndividualMocks => 5,
            PlotKind::XiRatio => 6,
            PlotKind::TimingComparison => 7,
            PlotKind::Summary => 8,
        }
    }

    fn label(self) -> &'static str {
        match self {
            PlotKind::XiComparison => "xi",
            PlotKind::Residuals => "resid",
            PlotKind::R2Xi => "r2xi",
            PlotKind::CdfComparison => "CDF",
            PlotKind::PeakedCdf => "PDF",
            PlotKind::IndividualMocks => "mocks",
            PlotKind::XiRatio => "ratio",
            PlotKind::TimingComparison => "timing",
            PlotKind::Summary => "summary",
        }
    }

    fn filename(self) -> &'static str {
        match self {
            PlotKind::XiComparison => "xi_vs_analytic.svg",
            PlotKind::Residuals => "xi_residuals.svg",
            PlotKind::R2Xi => "r2xi_comparison.svg",
            PlotKind::CdfComparison => "cdf_comparison.svg",
            PlotKind::PeakedCdf => "knn_pdf.svg",
            PlotKind::IndividualMocks => "individual_mocks.svg",
            PlotKind::XiRatio => "xi_ratio.svg",
            PlotKind::TimingComparison => "timing_comparison.svg",
            PlotKind::Summary => "validation_summary.svg",
        }
    }

    /// Default (log_x, log_y) for plots with positive-definite data.
    fn default_log(self) -> (bool, bool) {
        match self {
            PlotKind::XiComparison => (true, true),
            PlotKind::Residuals => (false, false),
            PlotKind::R2Xi => (true, true),
            PlotKind::CdfComparison => (false, false),
            PlotKind::PeakedCdf => (true, true),
            PlotKind::IndividualMocks => (true, true),
            PlotKind::XiRatio => (true, false),
            PlotKind::TimingComparison => (false, false),
            PlotKind::Summary => (false, false),
        }
    }
}

// ---------------------------------------------------------------------------
// Per-plot view state
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct PlotViewState {
    log_x: bool,
    log_y: bool,
    x_range: Option<(f64, f64)>,
    y_range: Option<(f64, f64)>,
}

/// Mutable state for the explorer.
struct ExplorerState {
    plot_kind: PlotKind,
    /// Per-plot view settings (one per PlotKind, indexed by PlotKind::index()).
    views: [PlotViewState; 9],
    status_message: String,
    show_help: bool,
    quit: bool,
}

impl ExplorerState {
    fn new() -> Self {
        let views = PlotKind::ALL.map(|kind| {
            let (log_x, log_y) = kind.default_log();
            PlotViewState {
                log_x,
                log_y,
                x_range: None,
                y_range: None,
            }
        });
        Self {
            plot_kind: PlotKind::XiComparison,
            views,
            status_message: String::new(),
            show_help: false,
            quit: false,
        }
    }

    fn view(&self) -> &PlotViewState {
        &self.views[self.plot_kind.index()]
    }

    fn view_mut(&mut self) -> &mut PlotViewState {
        let idx = self.plot_kind.index();
        &mut self.views[idx]
    }

    /// Reset the current plot's view to its default log scales, auto ranges.
    fn reset_current_view(&mut self) {
        let (log_x, log_y) = self.plot_kind.default_log();
        let v = self.view_mut();
        v.log_x = log_x;
        v.log_y = log_y;
        v.x_range = None;
        v.y_range = None;
    }

    /// Reset all views to defaults (used on summary screen).
    fn reset_all_views(&mut self) {
        for kind in &PlotKind::ALL {
            let (log_x, log_y) = kind.default_log();
            self.views[kind.index()] = PlotViewState {
                log_x,
                log_y,
                x_range: None,
                y_range: None,
            };
        }
    }
}

// ---------------------------------------------------------------------------
// Plot builders (return raw plots + layout, no dimensions applied)
// ---------------------------------------------------------------------------

/// Build raw (plots, layout) for a non-summary plot kind.
fn build_raw(kind: PlotKind, data: &PlotData) -> (Vec<Plot>, Layout) {
    match kind {
        PlotKind::XiComparison => build_xi_comparison(data),
        PlotKind::Residuals => build_residuals(data),
        PlotKind::R2Xi => build_r2xi(data),
        PlotKind::CdfComparison => build_cdf_comparison(data),
        PlotKind::PeakedCdf => build_peaked_cdf(data),
        PlotKind::IndividualMocks => build_individual_mocks(data),
        PlotKind::XiRatio => build_xi_ratio(data),
        PlotKind::TimingComparison => build_timing_comparison(data),
        PlotKind::Summary => build_xi_comparison(data), // fallback, not used directly
    }
}

/// Apply a PlotViewState (log scales, range overrides) to a layout.
fn apply_view(layout: &mut Layout, view: &PlotViewState) {
    if view.log_x {
        layout.log_x = true;
    }
    if view.log_y {
        layout.log_y = true;
    }
    if let Some((lo, hi)) = view.x_range {
        layout.x_range = (lo, hi);
        layout.x_axis_min = Some(lo);
        layout.x_axis_max = Some(hi);
    }
    if let Some((lo, hi)) = view.y_range {
        layout.y_range = (lo, hi);
        layout.y_axis_min = Some(lo);
        layout.y_axis_max = Some(hi);
    }
}

fn build_xi_comparison(data: &PlotData) -> (Vec<Plot>, Layout) {
    let n = data.r_centers.len();
    let level_colors = ["steelblue", "seagreen", "darkorange", "crimson"];

    let mut plots: Vec<Plot> = Vec::new();

    if let Some(ref tags) = data.level_tags {
        // Per-level bands and lines
        let max_level = *tags.iter().max().unwrap_or(&0);
        for level in 0..=max_level {
            let color = level_colors[level % level_colors.len()];
            let indices: Vec<usize> = (0..n).filter(|&i| tags[i] == level).collect();
            if indices.is_empty() {
                continue;
            }
            let r_l: Vec<f64> = indices.iter().map(|&i| data.r_centers[i]).collect();
            let lo: Vec<f64> = indices
                .iter()
                .map(|&i| data.mean_xi[i] - data.std_xi[i])
                .collect();
            let hi: Vec<f64> = indices
                .iter()
                .map(|&i| data.mean_xi[i] + data.std_xi[i])
                .collect();
            let xi_l: Vec<f64> = indices.iter().map(|&i| data.mean_xi[i]).collect();
            plots.push(Plot::Band(
                BandPlot::new(r_l.clone(), lo, hi)
                    .with_color(color)
                    .with_opacity(0.15),
            ));
            plots.push(Plot::Line(
                LinePlot::new()
                    .with_data(r_l.iter().copied().zip(xi_l.iter().copied()))
                    .with_color(color)
                    .with_stroke_width(2.0)
                    .with_legend(&format!("Level {}", level)),
            ));
        }
    } else {
        let lower: Vec<f64> = (0..n).map(|i| data.mean_xi[i] - data.std_xi[i]).collect();
        let upper: Vec<f64> = (0..n).map(|i| data.mean_xi[i] + data.std_xi[i]).collect();
        plots.push(Plot::Band(
            BandPlot::new(data.r_centers.clone(), lower, upper)
                .with_color("steelblue")
                .with_opacity(0.2),
        ));
        plots.push(Plot::Line(
            LinePlot::new()
                .with_data(
                    data.r_centers
                        .iter()
                        .copied()
                        .zip(data.mean_xi.iter().copied()),
                )
                .with_color("steelblue")
                .with_stroke_width(2.0)
                .with_legend("kNN LS mean +/- 1sigma"),
        ));
    }

    let analytic = LinePlot::new()
        .with_data(
            data.r_smooth
                .iter()
                .copied()
                .zip(data.xi_smooth.iter().copied()),
        )
        .with_color("black")
        .with_stroke_width(1.5)
        .with_dashed()
        .with_legend("Analytic xi(r)");

    let ell_ref = ReferenceLine::vertical(data.line_length)
        .with_color("#888888")
        .with_label("r = l")
        .with_stroke_width(0.8);

    plots.push(Plot::Line(analytic));

    // Overlay Corrfunc as error bars at bin centers
    if let (Some(cf_mean), Some(cf_std)) = (&data.corrfunc_mean_xi, &data.corrfunc_std_xi) {
        let pts: Vec<(f64, f64)> = data
            .r_centers
            .iter()
            .copied()
            .zip(cf_mean.iter().copied())
            .collect();
        plots.push(Plot::Scatter(
            ScatterPlot::new()
                .with_data(pts)
                .with_y_err(cf_std.iter().copied())
                .with_color("crimson")
                .with_size(4.0)
                .with_legend("Corrfunc +/- 1sigma"),
        ));
    }

    let layout = Layout::auto_from_plots(&plots)
        .with_title("CoxMock: xi(r) Recovery")
        .with_x_label("r  [h^-1 Mpc]")
        .with_y_label("xi(r)")
        .with_reference_line(ell_ref);

    (plots, layout)
}

fn build_residuals(data: &PlotData) -> (Vec<Plot>, Layout) {
    let pts: Vec<(f64, f64)> = data
        .r_centers
        .iter()
        .copied()
        .zip(data.bias_sigma.iter().copied())
        .collect();

    let scatter = ScatterPlot::new()
        .with_data(pts)
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
        .with_reference_line(zero)
        .with_reference_line(plus2)
        .with_reference_line(minus2);

    (plots, layout)
}

fn build_r2xi(data: &PlotData) -> (Vec<Plot>, Layout) {
    let n = data.r_centers.len();
    let level_colors = ["steelblue", "seagreen", "darkorange", "crimson"];

    let r2xi_analytic: Vec<f64> = data
        .r_smooth
        .iter()
        .zip(data.xi_smooth.iter())
        .map(|(&r, &xi)| r * r * xi)
        .collect();

    let mut plots: Vec<Plot> = Vec::new();

    if let Some(ref tags) = data.level_tags {
        let max_level = *tags.iter().max().unwrap_or(&0);
        for level in 0..=max_level {
            let color = level_colors[level % level_colors.len()];
            let indices: Vec<usize> = (0..n).filter(|&i| tags[i] == level).collect();
            if indices.is_empty() {
                continue;
            }
            let r_l: Vec<f64> = indices.iter().map(|&i| data.r_centers[i]).collect();
            let lo: Vec<f64> = indices
                .iter()
                .map(|&i| data.r_centers[i].powi(2) * (data.mean_xi[i] - data.std_xi[i]))
                .collect();
            let hi: Vec<f64> = indices
                .iter()
                .map(|&i| data.r_centers[i].powi(2) * (data.mean_xi[i] + data.std_xi[i]))
                .collect();
            let r2xi_l: Vec<f64> = indices
                .iter()
                .map(|&i| data.r_centers[i].powi(2) * data.mean_xi[i])
                .collect();
            plots.push(Plot::Band(
                BandPlot::new(r_l.clone(), lo, hi)
                    .with_color(color)
                    .with_opacity(0.15),
            ));
            plots.push(Plot::Line(
                LinePlot::new()
                    .with_data(r_l.iter().copied().zip(r2xi_l.iter().copied()))
                    .with_color(color)
                    .with_stroke_width(2.0)
                    .with_legend(&format!("Level {}", level)),
            ));
        }
    } else {
        let r2xi_mean: Vec<f64> = (0..n)
            .map(|i| data.r_centers[i].powi(2) * data.mean_xi[i])
            .collect();
        let r2xi_lower: Vec<f64> = (0..n)
            .map(|i| data.r_centers[i].powi(2) * (data.mean_xi[i] - data.std_xi[i]))
            .collect();
        let r2xi_upper: Vec<f64> = (0..n)
            .map(|i| data.r_centers[i].powi(2) * (data.mean_xi[i] + data.std_xi[i]))
            .collect();
        plots.push(Plot::Band(
            BandPlot::new(data.r_centers.clone(), r2xi_lower, r2xi_upper)
                .with_color("steelblue")
                .with_opacity(0.2),
        ));
        plots.push(Plot::Line(
            LinePlot::new()
                .with_data(
                    data.r_centers
                        .iter()
                        .copied()
                        .zip(r2xi_mean.iter().copied()),
                )
                .with_color("steelblue")
                .with_stroke_width(2.0)
                .with_legend("kNN LS"),
        ));
    }

    let analytic = LinePlot::new()
        .with_data(
            data.r_smooth
                .iter()
                .copied()
                .zip(r2xi_analytic.iter().copied()),
        )
        .with_color("black")
        .with_stroke_width(1.5)
        .with_dashed()
        .with_legend("Analytic");

    let ell_ref = ReferenceLine::vertical(data.line_length)
        .with_color("#888888")
        .with_label("r = l")
        .with_stroke_width(0.8);

    plots.push(Plot::Line(analytic));

    // Overlay Corrfunc r²ξ as error bars
    if let (Some(cf_mean), Some(cf_std)) = (&data.corrfunc_mean_xi, &data.corrfunc_std_xi) {
        let r2cf_mean: Vec<f64> = (0..n)
            .map(|i| data.r_centers[i].powi(2) * cf_mean[i])
            .collect();
        let r2cf_err: Vec<f64> = (0..n)
            .map(|i| data.r_centers[i].powi(2) * cf_std[i])
            .collect();
        let pts: Vec<(f64, f64)> = data
            .r_centers
            .iter()
            .copied()
            .zip(r2cf_mean.iter().copied())
            .collect();
        plots.push(Plot::Scatter(
            ScatterPlot::new()
                .with_data(pts)
                .with_y_err(r2cf_err.iter().copied())
                .with_color("crimson")
                .with_size(4.0)
                .with_legend("Corrfunc"),
        ));
    }

    let layout = Layout::auto_from_plots(&plots)
        .with_title("CoxMock: r^2 xi(r)")
        .with_x_label("r  [h^-1 Mpc]")
        .with_y_label("r^2 xi(r)  [h^-2 Mpc^2]")
        .with_reference_line(ell_ref);

    (plots, layout)
}

fn build_cdf_comparison(data: &PlotData) -> (Vec<Plot>, Layout) {
    let colors = ["steelblue", "crimson", "seagreen", "darkorange"];
    let mut plots: Vec<Plot> = Vec::new();

    if let Some(ref cdf) = data.cdf_rr_summary {
        for (idx, &k) in cdf.k_values.iter().enumerate() {
            let color = colors[idx % colors.len()];
            plots.push(Plot::Line(
                LinePlot::new()
                    .with_data(
                        cdf.r_values
                            .iter()
                            .copied()
                            .zip(cdf.cdf_mean[idx].iter().copied()),
                    )
                    .with_color(color)
                    .with_stroke_width(2.0)
                    .with_legend(&format!("k={} measured", k)),
            ));
            let erlang: Vec<f64> = cdf
                .r_values
                .iter()
                .map(|&r| erlang_cdf(k, r, data.cdf_nbar))
                .collect();
            plots.push(Plot::Line(
                LinePlot::new()
                    .with_data(cdf.r_values.iter().copied().zip(erlang.iter().copied()))
                    .with_color("black")
                    .with_stroke_width(1.0)
                    .with_dashed(),
            ));
        }
    }

    let layout = Layout::auto_from_plots(&plots)
        .with_title("kNN-CDF: Random Catalog vs Erlang")
        .with_x_label("r  [h^-1 Mpc]")
        .with_y_label("CDF_k(r)");

    (plots, layout)
}

/// Peaked CDF (PDF): numerical dCDF/dr for measured data and analytic Erlang PDF.
fn build_peaked_cdf(data: &PlotData) -> (Vec<Plot>, Layout) {
    let colors = ["steelblue", "crimson", "seagreen", "darkorange"];
    let mut plots: Vec<Plot> = Vec::new();

    if let Some(ref cdf) = data.cdf_rr_summary {
        let r = &cdf.r_values;
        let nr = r.len();

        for (idx, &k) in cdf.k_values.iter().enumerate() {
            let color = colors[idx % colors.len()];
            let measured = &cdf.cdf_mean[idx];

            // Numerical derivative via central differences
            let mut pdf = vec![0.0; nr];
            if nr >= 3 {
                pdf[0] = (measured[1] - measured[0]) / (r[1] - r[0]);
                for i in 1..nr - 1 {
                    pdf[i] = (measured[i + 1] - measured[i - 1]) / (r[i + 1] - r[i - 1]);
                }
                pdf[nr - 1] = (measured[nr - 1] - measured[nr - 2]) / (r[nr - 1] - r[nr - 2]);
            }

            plots.push(Plot::Line(
                LinePlot::new()
                    .with_data(r.iter().copied().zip(pdf.iter().copied()))
                    .with_color(color)
                    .with_stroke_width(2.0)
                    .with_legend(&format!("k={} measured", k)),
            ));

            // Analytic Erlang PDF
            let analytic: Vec<f64> =
                r.iter().map(|&ri| erlang_pdf(k, ri, data.cdf_nbar)).collect();
            plots.push(Plot::Line(
                LinePlot::new()
                    .with_data(r.iter().copied().zip(analytic.iter().copied()))
                    .with_color("black")
                    .with_stroke_width(1.0)
                    .with_dashed(),
            ));
        }
    }

    let layout = Layout::auto_from_plots(&plots)
        .with_title("kNN-PDF: dCDF/dr (peaked CDF)")
        .with_x_label("r  [h^-1 Mpc]")
        .with_y_label("dCDF_k/dr");

    (plots, layout)
}

fn build_individual_mocks(data: &PlotData) -> (Vec<Plot>, Layout) {
    let mock_colors = [
        "#b8d4f0", "#a0c4e8", "#c0d8f4", "#90b4d8", "#a8cce8", "#b0d0f0", "#98bce0", "#c8dcf4",
        "#88acd0", "#b0c8e8", "#b4d0ec", "#9cc0e4", "#c4d8f0", "#94b8dc", "#a4c8e4", "#acd0ec",
        "#a0c0e0", "#bcd4f0", "#8cb0d4", "#b4cce8",
    ];

    let mut plots: Vec<Plot> = Vec::new();
    for (idx, xi) in data.all_xi.iter().enumerate() {
        let color = mock_colors[idx % mock_colors.len()];
        plots.push(Plot::Line(
            LinePlot::new()
                .with_data(data.r_centers.iter().copied().zip(xi.iter().copied()))
                .with_color(color)
                .with_stroke_width(0.7),
        ));
    }
    plots.push(Plot::Line(
        LinePlot::new()
            .with_data(
                data.r_centers
                    .iter()
                    .copied()
                    .zip(data.mean_xi.iter().copied()),
            )
            .with_color("steelblue")
            .with_stroke_width(2.5)
            .with_legend("Mean"),
    ));
    plots.push(Plot::Line(
        LinePlot::new()
            .with_data(
                data.r_smooth
                    .iter()
                    .copied()
                    .zip(data.xi_smooth.iter().copied()),
            )
            .with_color("black")
            .with_stroke_width(2.0)
            .with_dashed()
            .with_legend("Analytic"),
    ));

    let ell_ref = ReferenceLine::vertical(data.line_length)
        .with_color("#888888")
        .with_label("r = l")
        .with_stroke_width(0.8);

    let layout = Layout::auto_from_plots(&plots)
        .with_title("Individual Mock Realizations")
        .with_x_label("r  [h^-1 Mpc]")
        .with_y_label("xi(r)")
        .with_reference_line(ell_ref);

    (plots, layout)
}

/// xi_kNN / xi_Corrfunc ratio per bin.
fn build_xi_ratio(data: &PlotData) -> (Vec<Plot>, Layout) {
    let mut plots: Vec<Plot> = Vec::new();

    if let Some(cf_mean) = &data.corrfunc_mean_xi {
        let ratio: Vec<f64> = data
            .mean_xi
            .iter()
            .zip(cf_mean.iter())
            .map(|(&knn, &cf)| if cf.abs() > 1e-15 { knn / cf } else { 1.0 })
            .collect();
        let pts: Vec<(f64, f64)> = data
            .r_centers
            .iter()
            .copied()
            .zip(ratio.iter().copied())
            .collect();
        plots.push(Plot::Scatter(
            ScatterPlot::new()
                .with_data(pts)
                .with_color("steelblue")
                .with_size(4.5)
                .with_legend("xi_kNN / xi_Corrfunc"),
        ));
    }

    let one = ReferenceLine::horizontal(1.0)
        .with_color("black")
        .with_stroke_width(1.0)
        .with_dasharray("4 3");
    let plus2p = ReferenceLine::horizontal(1.02)
        .with_color("crimson")
        .with_stroke_width(0.7)
        .with_dasharray("6 4")
        .with_label("+2%");
    let minus2p = ReferenceLine::horizontal(0.98)
        .with_color("crimson")
        .with_stroke_width(0.7)
        .with_dasharray("6 4")
        .with_label("-2%");

    let layout = Layout::auto_from_plots(&plots)
        .with_title("xi Ratio: kNN / Corrfunc")
        .with_x_label("r  [h^-1 Mpc]")
        .with_y_label("xi_kNN / xi_Corrfunc")
        .with_reference_line(one)
        .with_reference_line(plus2p)
        .with_reference_line(minus2p);

    (plots, layout)
}

/// Wall-clock timing comparison: kNN vs Corrfunc per mock.
fn build_timing_comparison(data: &PlotData) -> (Vec<Plot>, Layout) {
    let mut plots: Vec<Plot> = Vec::new();

    if let Some(knn_t) = &data.knn_times {
        let pts: Vec<(f64, f64)> = knn_t
            .iter()
            .enumerate()
            .map(|(i, &t)| (i as f64, t))
            .collect();
        plots.push(Plot::Scatter(
            ScatterPlot::new()
                .with_data(pts)
                .with_color("steelblue")
                .with_size(5.0)
                .with_legend("kNN"),
        ));
    }

    if let Some(cf_t) = &data.corrfunc_times {
        let pts: Vec<(f64, f64)> = cf_t
            .iter()
            .enumerate()
            .map(|(i, &t)| (i as f64, t))
            .collect();
        plots.push(Plot::Scatter(
            ScatterPlot::new()
                .with_data(pts)
                .with_color("crimson")
                .with_size(5.0)
                .with_legend("Corrfunc"),
        ));
    }

    let layout = Layout::auto_from_plots(&plots)
        .with_title("Wall-Clock Time per Mock")
        .with_x_label("Mock index")
        .with_y_label("Time [s]");

    (plots, layout)
}

// ---------------------------------------------------------------------------
// SVG rendering (single plot or summary figure)
// ---------------------------------------------------------------------------

/// Render SVG for a specific plot kind, reading view state from `state.views`.
fn render_svg_for(
    state: &ExplorerState,
    kind: PlotKind,
    data: &PlotData,
    width: f64,
    height: f64,
) -> String {
    if kind == PlotKind::Summary {
        // 4×2 multi-panel figure using per-plot view state
        let mut all_plots = Vec::new();
        let mut all_layouts = Vec::new();
        for panel in &PlotKind::PANELS {
            let (plots, mut layout) = build_raw(*panel, data);
            apply_view(&mut layout, &state.views[panel.index()]);
            all_plots.push(plots);
            all_layouts.push(layout);
        }
        let scene = Figure::new(4, 2)
            .with_plots(all_plots)
            .with_layouts(all_layouts)
            .with_labels()
            .with_figure_size(width, height)
            .render();
        SvgBackend.render_scene(&scene)
    } else {
        let (plots, mut layout) = build_raw(kind, data);
        layout.width = Some(width);
        layout.height = Some(height);
        apply_view(&mut layout, &state.views[kind.index()]);
        SvgBackend.render_scene(&render_multiple(plots, layout))
    }
}

/// Render the currently selected plot to SVG.
fn render_svg(state: &ExplorerState, data: &PlotData) -> String {
    let (w, h) = if state.plot_kind == PlotKind::Summary {
        (1200.0, 900.0)
    } else {
        (700.0, 500.0)
    };
    render_svg_for(state, state.plot_kind, data, w, h)
}

// ---------------------------------------------------------------------------
// Status bars
// ---------------------------------------------------------------------------

/// Format the status bar (2 lines) for the bottom of the terminal.
fn format_status_bar(state: &ExplorerState, cols: usize) -> String {
    let mut tabs = String::new();
    for kind in &PlotKind::ALL {
        let idx = kind.index() + 1;
        if *kind == state.plot_kind {
            tabs.push_str(&format!("[{}:{}]", idx, kind.label()));
        } else {
            tabs.push_str(&format!(" {}:{} ", idx, kind.label()));
        }
    }

    let info = if !state.status_message.is_empty() {
        state.status_message.clone()
    } else if state.plot_kind == PlotKind::Summary {
        "Summary (per-plot settings) | s:save r:reset-all ?:help q:quit".to_string()
    } else {
        let v = state.view();
        let x_scale = if v.log_x { "log" } else { "linear" };
        let y_scale = if v.log_y { "log" } else { "linear" };
        let x_range = match v.x_range {
            Some((lo, hi)) => format!("[{:.1}, {:.1}]", lo, hi),
            None => "[auto]".to_string(),
        };
        let y_range = match v.y_range {
            Some((lo, hi)) => format!("[{:.1}, {:.1}]", lo, hi),
            None => "[auto]".to_string(),
        };
        format!(
            "X: {} {}  Y: {} {}  | s:save r:reset ?:help q:quit",
            x_scale, x_range, y_scale, y_range,
        )
    };

    let line1 = if tabs.len() > cols {
        &tabs[..cols]
    } else {
        &tabs
    };
    let line2 = if info.len() > cols {
        &info[..cols]
    } else {
        &info
    };
    format!("{}\r\n{}", line1, line2)
}

// ---------------------------------------------------------------------------
// Key handling
// ---------------------------------------------------------------------------

/// Handle a key event, mutating the explorer state.
fn handle_key(key: KeyEvent, state: &mut ExplorerState, data: &PlotData, output_dir: &str) {
    state.status_message.clear();
    let is_summary = state.plot_kind == PlotKind::Summary;

    match key.code {
        KeyCode::Char('q') | KeyCode::Esc => {
            state.quit = true;
        }
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            state.quit = true;
        }
        KeyCode::Char('?') => {
            state.show_help = !state.show_help;
        }
        // Plot selection by number
        KeyCode::Char(c @ '1'..='9') => {
            let idx = (c as usize) - ('1' as usize);
            if idx < PlotKind::ALL.len() {
                state.plot_kind = PlotKind::ALL[idx];
            }
        }
        // Next/previous plot
        KeyCode::Char('n') => {
            let idx = (state.plot_kind.index() + 1) % 9;
            state.plot_kind = PlotKind::ALL[idx];
        }
        KeyCode::Char('p') => {
            let idx = (state.plot_kind.index() + 8) % 9;
            state.plot_kind = PlotKind::ALL[idx];
        }
        // Toggle log scales (no-op on summary)
        KeyCode::Char('x') if !is_summary => {
            state.view_mut().log_x = !state.view().log_x;
        }
        KeyCode::Char('y') if !is_summary => {
            state.view_mut().log_y = !state.view().log_y;
        }
        // Zoom in/out (no-op on summary)
        KeyCode::Char('+') | KeyCode::Char('=') if !is_summary => {
            zoom(state, data, 0.8);
        }
        KeyCode::Char('-') if !is_summary => {
            zoom(state, data, 1.25);
        }
        // Pan (no-op on summary)
        KeyCode::Left if !is_summary => pan(state, data, -0.1, 0.0),
        KeyCode::Right if !is_summary => pan(state, data, 0.1, 0.0),
        KeyCode::Up if !is_summary => pan(state, data, 0.0, 0.1),
        KeyCode::Down if !is_summary => pan(state, data, 0.0, -0.1),
        // Reset
        KeyCode::Char('r') => {
            if is_summary {
                state.reset_all_views();
                state.status_message = "All views reset".to_string();
            } else {
                state.reset_current_view();
                state.status_message = "View reset".to_string();
            }
        }
        // Save current plot
        KeyCode::Char('s') => {
            state.status_message = save_svg(state, data, output_dir);
        }
        // Save all plots
        KeyCode::Char('S') => {
            state.status_message = save_all_svgs(state, data, output_dir);
        }
        _ => {}
    }
}

/// Get the effective axis ranges for the current plot.
fn effective_ranges(state: &ExplorerState, data: &PlotData) -> ((f64, f64), (f64, f64)) {
    let (_, layout) = build_raw(state.plot_kind, data);
    let v = state.view();
    let x = v.x_range.unwrap_or(layout.x_range);
    let y = v.y_range.unwrap_or(layout.y_range);
    (x, y)
}

/// Zoom by scaling both axis ranges around their centers.
fn zoom(state: &mut ExplorerState, data: &PlotData, factor: f64) {
    let (xr, yr) = effective_ranges(state, data);

    let x_center = (xr.0 + xr.1) / 2.0;
    let x_half = (xr.1 - xr.0) / 2.0 * factor;
    let y_center = (yr.0 + yr.1) / 2.0;
    let y_half = (yr.1 - yr.0) / 2.0 * factor;

    let v = state.view_mut();
    v.x_range = Some((x_center - x_half, x_center + x_half));
    v.y_range = Some((y_center - y_half, y_center + y_half));
}

/// Pan the view by a fraction of the current span.
fn pan(state: &mut ExplorerState, data: &PlotData, dx_frac: f64, dy_frac: f64) {
    let (xr, yr) = effective_ranges(state, data);

    let dx = (xr.1 - xr.0) * dx_frac;
    let dy = (yr.1 - yr.0) * dy_frac;

    let v = state.view_mut();
    v.x_range = Some((xr.0 + dx, xr.1 + dx));
    v.y_range = Some((yr.0 + dy, yr.1 + dy));
}

/// Save the current view as a publication-quality SVG.
fn save_svg(state: &ExplorerState, data: &PlotData, output_dir: &str) -> String {
    let (w, h) = if state.plot_kind == PlotKind::Summary {
        (1200.0, 900.0)
    } else {
        (700.0, 500.0)
    };
    let svg = render_svg_for(state, state.plot_kind, data, w, h);
    let filename = state.plot_kind.filename();
    let path = format!("{}/{}", output_dir, filename);
    match std::fs::write(&path, svg) {
        Ok(_) => format!("Saved {}", path),
        Err(e) => format!("Failed to save {}: {}", path, e),
    }
}

/// Save all plots with their per-plot view settings.
fn save_all_svgs(state: &ExplorerState, data: &PlotData, output_dir: &str) -> String {
    let mut saved = 0;
    for kind in &PlotKind::ALL {
        let (w, h) = if *kind == PlotKind::Summary {
            (1200.0, 900.0)
        } else {
            (700.0, 500.0)
        };
        let svg = render_svg_for(state, *kind, data, w, h);
        let path = format!("{}/{}", output_dir, kind.filename());
        if std::fs::write(&path, &svg).is_ok() {
            saved += 1;
        }
    }
    format!("Saved {} SVGs to {}/", saved, output_dir)
}

// ---------------------------------------------------------------------------
// HTTP server + browser UI
// ---------------------------------------------------------------------------

/// Pre-rendered frame data shared between main thread and HTTP server thread.
struct FrameData {
    svg: String,
    status_html: String,
    show_help: bool,
    title: String,
    quit: bool,
}

/// Render the current state into shared frame data.
fn update_frame(state: &ExplorerState, data: &PlotData, frame: &Mutex<FrameData>) {
    let svg = render_svg(state, data);
    let status_html = format_status_bar_html(state);

    let mut f = frame.lock().unwrap();
    f.svg = svg;
    f.status_html = status_html;
    f.show_help = state.show_help;
    f.title = state.plot_kind.label().to_string();
    f.quit = state.quit;
}

/// Format the status bar as HTML for the browser.
fn format_status_bar_html(state: &ExplorerState) -> String {
    let mut tabs = String::new();
    for kind in &PlotKind::ALL {
        let idx = kind.index() + 1;
        if *kind == state.plot_kind {
            tabs.push_str(&format!(
                r#"<span class="active">{}: {}</span>"#,
                idx,
                kind.label()
            ));
        } else {
            tabs.push_str(&format!(r#"<span>{}: {}</span>"#, idx, kind.label()));
        }
    }

    let info = if !state.status_message.is_empty() {
        format!(
            "<em>{}</em>",
            state
                .status_message
                .replace('<', "&lt;")
                .replace('>', "&gt;")
        )
    } else if state.plot_kind == PlotKind::Summary {
        "Summary (per-plot settings) &nbsp;\u{2502}&nbsp; \
         <kbd>s</kbd>save <kbd>r</kbd>reset all <kbd>?</kbd>help <kbd>q</kbd>quit"
            .to_string()
    } else {
        let v = state.view();
        let x_scale = if v.log_x { "log" } else { "lin" };
        let y_scale = if v.log_y { "log" } else { "lin" };
        let x_range = match v.x_range {
            Some((lo, hi)) => format!("[{:.1},{:.1}]", lo, hi),
            None => "auto".into(),
        };
        let y_range = match v.y_range {
            Some((lo, hi)) => format!("[{:.1},{:.1}]", lo, hi),
            None => "auto".into(),
        };
        format!(
            "X:{} {} &nbsp; Y:{} {} &nbsp;\u{2502}&nbsp; \
             <kbd>s</kbd>save <kbd>r</kbd>reset <kbd>?</kbd>help <kbd>q</kbd>quit",
            x_scale, x_range, y_scale, y_range,
        )
    };

    format!(
        r#"<div class="tabs">{}</div><div class="info">{}</div>"#,
        tabs, info
    )
}

/// Escape a string for embedding in a JSON string value.
fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + s.len() / 8);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = std::fmt::Write::write_fmt(
                    &mut out,
                    format_args!("\\u{:04x}", c as u32),
                );
            }
            c => out.push(c),
        }
    }
    out
}

/// Map a browser `KeyboardEvent.key` string to a crossterm `KeyEvent`.
fn parse_browser_key(key: &str) -> Option<KeyEvent> {
    let code = match key {
        "ArrowLeft" => KeyCode::Left,
        "ArrowRight" => KeyCode::Right,
        "ArrowUp" => KeyCode::Up,
        "ArrowDown" => KeyCode::Down,
        "Escape" => KeyCode::Esc,
        "+" => KeyCode::Char('+'),
        "=" => KeyCode::Char('='),
        "-" => KeyCode::Char('-'),
        s if s.len() == 1 => KeyCode::Char(s.chars().next().unwrap()),
        _ => return None,
    };
    Some(KeyEvent::new(code, KeyModifiers::empty()))
}

/// Handle a single HTTP request on a TCP stream.
fn handle_http(
    mut stream: std::net::TcpStream,
    frame: &Mutex<FrameData>,
    key_tx: &mpsc::Sender<String>,
) {
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .ok();
    let mut buf = vec![0u8; 4096];
    let n = match stream.read(&mut buf) {
        Ok(n) if n > 0 => n,
        _ => return,
    };
    let request = String::from_utf8_lossy(&buf[..n]);

    let first_line = request.lines().next().unwrap_or("");
    let parts: Vec<&str> = first_line.split_whitespace().collect();
    let (method, path) = match parts.as_slice() {
        [m, p, ..] => (*m, *p),
        _ => return,
    };

    match (method, path) {
        ("GET", "/") => {
            let body = EXPLORER_HTML;
            let header = format!(
                "HTTP/1.1 200 OK\r\n\
                 Content-Type: text/html; charset=utf-8\r\n\
                 Content-Length: {}\r\n\r\n",
                body.len()
            );
            stream.write_all(header.as_bytes()).ok();
            stream.write_all(body.as_bytes()).ok();
        }
        ("GET", "/frame") => {
            let f = frame.lock().unwrap();
            let json = format!(
                r#"{{"svg":"{}","status_html":"{}","show_help":{},"title":"{}","quit":{}}}"#,
                escape_json(&f.svg),
                escape_json(&f.status_html),
                f.show_help,
                escape_json(&f.title),
                f.quit,
            );
            let header = format!(
                "HTTP/1.1 200 OK\r\n\
                 Content-Type: application/json\r\n\
                 Cache-Control: no-cache\r\n\
                 Content-Length: {}\r\n\r\n",
                json.len()
            );
            stream.write_all(header.as_bytes()).ok();
            stream.write_all(json.as_bytes()).ok();
        }
        ("POST", "/key") => {
            if let Some(idx) = request.find("\r\n\r\n") {
                let body = request[idx + 4..].trim();
                if !body.is_empty() {
                    key_tx.send(body.to_string()).ok();
                }
            }
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
                .ok();
        }
        _ => {
            stream
                .write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
                .ok();
        }
    }
}

/// Static HTML page served to the browser.
const EXPLORER_HTML: &str = r##"<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>2pt-kNN Explorer</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    font-family: 'SF Mono','Menlo','Consolas',monospace;
    background:#1e1e2e; color:#cdd6f4;
    height:100vh; display:flex; flex-direction:column;
  }
  #plot {
    flex:1; display:flex; align-items:center; justify-content:center;
    padding:12px; background:#fff; min-height:0; overflow:hidden;
  }
  #plot svg { max-width:100%; max-height:100%; }
  #help {
    display:none; background:#1e1e2e; padding:10px 16px;
    font-size:13px; line-height:1.8; border-top:1px solid #313244;
  }
  #help.visible { display:block; }
  #help kbd {
    background:#313244; padding:1px 6px; border-radius:3px;
    font-family:inherit; color:#89b4fa;
  }
  #bar {
    background:#1e1e2e; padding:8px 16px; font-size:13px;
    border-top:1px solid #313244;
  }
  .tabs span { margin-right:10px; color:#6c7086; cursor:pointer; }
  .tabs .active {
    color:#89b4fa; font-weight:bold;
    border:1px solid #89b4fa; border-radius:3px; padding:1px 6px;
  }
  .info { color:#a6adc8; margin-top:4px; }
  .info kbd {
    background:#313244; padding:0 4px; border-radius:2px;
    font-family:inherit; color:#89b4fa;
  }
  .info em { color:#f9e2af; font-style:normal; }
  #closed {
    display:none; flex:1; align-items:center; justify-content:center;
    color:#6c7086; font-size:16px;
  }
</style>
</head><body>
  <div id="plot"></div>
  <div id="help">
    <div><kbd>1</kbd>–<kbd>9</kbd> jump to plot &nbsp;
         <kbd>n</kbd>/<kbd>p</kbd> next/prev &nbsp;
         <kbd>x</kbd>/<kbd>y</kbd> toggle log scale</div>
    <div><kbd>+</kbd>/<kbd>−</kbd> zoom &nbsp;
         <kbd>←</kbd><kbd>→</kbd><kbd>↑</kbd><kbd>↓</kbd> pan &nbsp;
         <kbd>r</kbd> reset view &nbsp;
         <kbd>s</kbd> save SVG &nbsp;
         <kbd>S</kbd> save all &nbsp;
         <kbd>?</kbd> toggle help</div>
  </div>
  <div id="bar"></div>
  <div id="closed">Explorer session ended</div>
  <script>
    let active = true;
    async function poll() {
      if (!active) return;
      try {
        const r = await fetch('/frame');
        if (!r.ok) { setTimeout(poll, 500); return; }
        const d = await r.json();
        if (d.quit) {
          document.getElementById('plot').style.display = 'none';
          document.getElementById('help').style.display = 'none';
          document.getElementById('bar').style.display = 'none';
          document.getElementById('closed').style.display = 'flex';
          active = false;
          return;
        }
        document.getElementById('plot').innerHTML = d.svg;
        document.getElementById('bar').innerHTML = d.status_html;
        document.getElementById('help').className = d.show_help ? 'visible' : '';
        document.title = d.title + ' \u2014 2pt-kNN Explorer';
      } catch(e) {
        document.getElementById('plot').style.display = 'none';
        document.getElementById('help').style.display = 'none';
        document.getElementById('bar').style.display = 'none';
        document.getElementById('closed').style.display = 'flex';
        active = false;
        return;
      }
      setTimeout(poll, 150);
    }
    poll();
    document.addEventListener('keydown', async (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey || !active) return;
      e.preventDefault();
      try { await fetch('/key', { method:'POST', body:e.key }); } catch(e) {}
    });
  </script>
</body>
</html>"##;

/// Main entry point: run the interactive plot explorer.
///
/// Starts a tiny HTTP server on localhost, opens the browser, and enters
/// an event loop that accepts keys from both the terminal and the browser.
pub fn run_explorer(data: &PlotData, output_dir: &str) -> std::io::Result<()> {
    let mut state = ExplorerState::new();
    let mut stdout = std::io::stdout();

    // Bind to a random available port
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let port = listener.local_addr()?.port();

    // Shared frame data (read by HTTP thread, written by main thread)
    let frame = Arc::new(Mutex::new(FrameData {
        svg: String::new(),
        status_html: String::new(),
        show_help: false,
        title: String::new(),
        quit: false,
    }));

    // Channel: browser key events → main thread
    let (key_tx, key_rx) = mpsc::channel::<String>();

    // Initial render
    update_frame(&state, data, &frame);

    // Spawn HTTP server thread
    let frame_http = Arc::clone(&frame);
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            handle_http(stream, &frame_http, &key_tx);
        }
    });

    // Open browser
    Command::new("open")
        .arg(format!("http://127.0.0.1:{}", port))
        .spawn()
        .ok();

    // Terminal raw mode for keypress capture
    terminal::enable_raw_mode()?;
    let cols = terminal::size().map(|(c, _)| c as usize).unwrap_or(80);
    write!(
        stdout,
        "\r\x1b[J2pt-kNN explorer \u{2192} http://127.0.0.1:{}\r\n",
        port
    )?;
    write!(stdout, "{}", format_status_bar(&state, cols))?;
    stdout.flush()?;

    loop {
        // Poll terminal events with short timeout so we can also drain browser keys
        if crossterm::event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                handle_key(key, &mut state, data, output_dir);
                update_frame(&state, data, &frame);
                write!(stdout, "\r\x1b[J{}", format_status_bar(&state, cols)).ok();
                stdout.flush().ok();
                if state.quit {
                    break;
                }
            }
        }

        // Drain browser key events
        let mut changed = false;
        while let Ok(key_str) = key_rx.try_recv() {
            if let Some(key) = parse_browser_key(&key_str) {
                handle_key(key, &mut state, data, output_dir);
                changed = true;
            }
        }
        if changed {
            update_frame(&state, data, &frame);
            write!(stdout, "\r\x1b[J{}", format_status_bar(&state, cols)).ok();
            stdout.flush().ok();
        }
        if state.quit {
            break;
        }
    }

    // Signal quit to any polling browser
    frame.lock().unwrap().quit = true;

    // Restore terminal
    terminal::disable_raw_mode()?;
    write!(stdout, "\r\n")?;
    stdout.flush()?;

    Ok(())
}

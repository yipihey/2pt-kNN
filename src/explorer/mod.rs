//! Interactive plot explorer for twopoint-validate.
//!
//! Serves plots via a tiny localhost HTTP server and opens them in the default
//! browser. The browser captures keyboard shortcuts and sends them back;
//! the terminal also accepts keys as a fallback.

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::terminal;
use std::io::{Read as _, Write};
use std::net::TcpListener;
use std::process::Command;
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use crate::plotting::{self, PlotConfig, PlotData, TypstPlotter};

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

    /// Render function for this plot kind.
    fn render_source(self, data: &PlotData, config: &PlotConfig) -> String {
        match self {
            PlotKind::XiComparison => plotting::render_xi_comparison(data, config),
            PlotKind::Residuals => plotting::render_residuals(data, config),
            PlotKind::R2Xi => plotting::render_r2xi(data, config),
            PlotKind::CdfComparison => plotting::render_cdf_comparison(data, config),
            PlotKind::PeakedCdf => plotting::render_peaked_cdf(data, config),
            PlotKind::IndividualMocks => plotting::render_individual_mocks(data, config),
            PlotKind::XiRatio => plotting::render_xi_ratio(data, config),
            PlotKind::TimingComparison => plotting::render_timing(data, config),
            PlotKind::Summary => plotting::render_xi_comparison(data, config), // fallback
        }
    }

    /// Compute natural data ranges for zoom/pan support.
    fn data_ranges(self, data: &PlotData) -> ((f64, f64), (f64, f64)) {
        match self {
            PlotKind::XiComparison | PlotKind::IndividualMocks => {
                let xr = data_extent(&data.r_centers);
                let ymin = data.mean_xi.iter().zip(data.std_xi.iter())
                    .map(|(&m, &s)| m - s)
                    .fold(f64::INFINITY, f64::min);
                let ymax = data.mean_xi.iter().zip(data.std_xi.iter())
                    .map(|(&m, &s)| m + s)
                    .fold(f64::NEG_INFINITY, f64::max);
                (xr, (ymin, ymax))
            }
            PlotKind::Residuals => {
                let xr = data_extent(&data.r_centers);
                let yr = data_extent(&data.bias_sigma);
                (xr, (yr.0.min(-3.0), yr.1.max(3.0)))
            }
            PlotKind::R2Xi => {
                let xr = data_extent(&data.r_centers);
                let n = data.r_centers.len();
                let vals: Vec<f64> = (0..n).map(|i| data.r_centers[i].powi(2) * data.mean_xi[i]).collect();
                let yr = data_extent(&vals);
                (xr, yr)
            }
            PlotKind::CdfComparison | PlotKind::PeakedCdf => {
                if let Some(ref cdf) = data.cdf_rr_summary {
                    let xr = data_extent(&cdf.r_values);
                    (xr, (0.0, 1.0))
                } else {
                    ((1.0, 100.0), (0.0, 1.0))
                }
            }
            PlotKind::XiRatio => {
                let xr = data_extent(&data.r_centers);
                (xr, (0.95, 1.05))
            }
            PlotKind::TimingComparison => {
                let n = data.knn_times.as_ref().map(|t| t.len()).unwrap_or(1);
                let all_times: Vec<f64> = data.knn_times.iter()
                    .flatten()
                    .chain(data.corrfunc_times.iter().flatten())
                    .copied()
                    .collect();
                let yr = if all_times.is_empty() { (0.0, 1.0) } else { data_extent(&all_times) };
                ((0.0, n as f64), yr)
            }
            PlotKind::Summary => {
                let xr = data_extent(&data.r_centers);
                let yr = data_extent(&data.mean_xi);
                (xr, yr)
            }
        }
    }
}

fn data_extent(vals: &[f64]) -> (f64, f64) {
    let lo = vals.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if (hi - lo).abs() < 1e-15 {
        (lo - 1.0, hi + 1.0)
    } else {
        let pad = (hi - lo) * 0.05;
        (lo - pad, hi + pad)
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
    views: [PlotViewState; 9],
    plotter: TypstPlotter,
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
            plotter: TypstPlotter::new(),
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

    fn reset_current_view(&mut self) {
        let (log_x, log_y) = self.plot_kind.default_log();
        let v = self.view_mut();
        v.log_x = log_x;
        v.log_y = log_y;
        v.x_range = None;
        v.y_range = None;
    }

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
// SVG rendering via typst + lilaq
// ---------------------------------------------------------------------------

fn config_from_view(view: &PlotViewState, width_cm: f64, height_cm: f64) -> PlotConfig {
    PlotConfig {
        width_cm,
        height_cm,
        log_x: view.log_x,
        log_y: view.log_y,
        x_range: view.x_range,
        y_range: view.y_range,
    }
}

fn render_svg_for(
    state: &ExplorerState,
    kind: PlotKind,
    data: &PlotData,
    width_cm: f64,
    height_cm: f64,
) -> String {
    if kind == PlotKind::Summary {
        let mut panel_configs: [PlotConfig; 8] = std::array::from_fn(|i| {
            let panel = PlotKind::PANELS[i];
            config_from_view(&state.views[panel.index()], 8.5, 6.0)
        });
        // Each panel gets its own config from the per-plot view state
        let _ = &mut panel_configs; // suppress unused warning
        let source = plotting::render_explorer_summary(data, &panel_configs);
        state.plotter.render(&source)
    } else {
        let config = config_from_view(&state.views[kind.index()], width_cm, height_cm);
        let source = kind.render_source(data, &config);
        state.plotter.render(&source)
    }
}

fn render_svg(state: &ExplorerState, data: &PlotData) -> String {
    let (w, h) = if state.plot_kind == PlotKind::Summary {
        (36.0, 26.0)
    } else {
        (18.0, 12.0)
    };
    render_svg_for(state, state.plot_kind, data, w, h)
}

// ---------------------------------------------------------------------------
// Status bars
// ---------------------------------------------------------------------------

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
        KeyCode::Char(c @ '1'..='9') => {
            let idx = (c as usize) - ('1' as usize);
            if idx < PlotKind::ALL.len() {
                state.plot_kind = PlotKind::ALL[idx];
            }
        }
        KeyCode::Char('n') => {
            let idx = (state.plot_kind.index() + 1) % 9;
            state.plot_kind = PlotKind::ALL[idx];
        }
        KeyCode::Char('p') => {
            let idx = (state.plot_kind.index() + 8) % 9;
            state.plot_kind = PlotKind::ALL[idx];
        }
        KeyCode::Char('x') if !is_summary => {
            state.view_mut().log_x = !state.view().log_x;
        }
        KeyCode::Char('y') if !is_summary => {
            state.view_mut().log_y = !state.view().log_y;
        }
        KeyCode::Char('+') | KeyCode::Char('=') if !is_summary => {
            zoom(state, data, 0.8);
        }
        KeyCode::Char('-') if !is_summary => {
            zoom(state, data, 1.25);
        }
        KeyCode::Left if !is_summary => pan(state, data, -0.1, 0.0),
        KeyCode::Right if !is_summary => pan(state, data, 0.1, 0.0),
        KeyCode::Up if !is_summary => pan(state, data, 0.0, 0.1),
        KeyCode::Down if !is_summary => pan(state, data, 0.0, -0.1),
        KeyCode::Char('r') => {
            if is_summary {
                state.reset_all_views();
                state.status_message = "All views reset".to_string();
            } else {
                state.reset_current_view();
                state.status_message = "View reset".to_string();
            }
        }
        KeyCode::Char('s') => {
            state.status_message = save_svg(state, data, output_dir);
        }
        KeyCode::Char('S') => {
            state.status_message = save_all_svgs(state, data, output_dir);
        }
        _ => {}
    }
}

fn effective_ranges(state: &ExplorerState, data: &PlotData) -> ((f64, f64), (f64, f64)) {
    let (default_x, default_y) = state.plot_kind.data_ranges(data);
    let v = state.view();
    let x = v.x_range.unwrap_or(default_x);
    let y = v.y_range.unwrap_or(default_y);
    (x, y)
}

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

fn pan(state: &mut ExplorerState, data: &PlotData, dx_frac: f64, dy_frac: f64) {
    let (xr, yr) = effective_ranges(state, data);

    let dx = (xr.1 - xr.0) * dx_frac;
    let dy = (yr.1 - yr.0) * dy_frac;

    let v = state.view_mut();
    v.x_range = Some((xr.0 + dx, xr.1 + dx));
    v.y_range = Some((yr.0 + dy, yr.1 + dy));
}

fn save_svg(state: &ExplorerState, data: &PlotData, output_dir: &str) -> String {
    let (w, h) = if state.plot_kind == PlotKind::Summary {
        (36.0, 26.0)
    } else {
        (18.0, 12.0)
    };
    let svg = render_svg_for(state, state.plot_kind, data, w, h);
    let filename = state.plot_kind.filename();
    let path = format!("{}/{}", output_dir, filename);
    match std::fs::write(&path, svg) {
        Ok(_) => format!("Saved {}", path),
        Err(e) => format!("Failed to save {}: {}", path, e),
    }
}

fn save_all_svgs(state: &ExplorerState, data: &PlotData, output_dir: &str) -> String {
    let mut saved = 0;
    for kind in &PlotKind::ALL {
        let (w, h) = if *kind == PlotKind::Summary {
            (36.0, 26.0)
        } else {
            (18.0, 12.0)
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

struct FrameData {
    svg: String,
    status_html: String,
    show_help: bool,
    title: String,
    quit: bool,
}

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
pub fn run_explorer(data: &PlotData, output_dir: &str) -> std::io::Result<()> {
    let mut state = ExplorerState::new();
    let mut stdout = std::io::stdout();

    let listener = TcpListener::bind("127.0.0.1:0")?;
    let port = listener.local_addr()?.port();

    let frame = Arc::new(Mutex::new(FrameData {
        svg: String::new(),
        status_html: String::new(),
        show_help: false,
        title: String::new(),
        quit: false,
    }));

    let (key_tx, key_rx) = mpsc::channel::<String>();

    update_frame(&state, data, &frame);

    let frame_http = Arc::clone(&frame);
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            handle_http(stream, &frame_http, &key_tx);
        }
    });

    Command::new("open")
        .arg(format!("http://127.0.0.1:{}", port))
        .spawn()
        .ok();

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

    frame.lock().unwrap().quit = true;

    terminal::disable_raw_mode()?;
    write!(stdout, "\r\n")?;
    stdout.flush()?;

    Ok(())
}

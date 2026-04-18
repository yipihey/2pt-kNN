//! Theory-prediction plots for V-cumulants and PDF.
//!
//! Generates SVG plots comparing recursion-based and action-based EFT
//! predictions, V-cumulants vs σ², EFT-order convergence, and the
//! saddle-point/Edgeworth-reconstructed V PDF.

use crate::theory::spectral::SpectralParams;
use crate::theory::growth::LptGrowthFactors;
use crate::theory::cumulants::VolumeCumulants;
use crate::theory::volume_pdf::VolumePdf;
use crate::theory::eft::EftParams;
use crate::theory::action::action_lpt::ActionLpt;
use crate::plotting::{fmt_array, compile_to_svg};

fn make_sp(sigma2: f64, gamma: f64) -> SpectralParams {
    SpectralParams {
        mass: 1e12, radius: 10.0, sigma2, gamma, gamma_n: vec![0.0],
    }
}

/// Plot comparing recursion-based vs action-based κ_m predictions
/// as a function of σ² (extended to 1.0 to see breakdown), for a fixed LPT order.
pub fn render_formulation_comparison(lpt_order: usize) -> String {
    let sigma2_grid: Vec<f64> = (1..=200).map(|i| i as f64 * 0.005).collect(); // 0.005 to 1.0

    let mut k2_rec = Vec::new();
    let mut k3_rec = Vec::new();
    let mut k4_rec = Vec::new();
    let mut k2_act = Vec::new();
    let mut k3_act = Vec::new();
    let mut k4_act = Vec::new();

    let gf = LptGrowthFactors::eds(lpt_order.max(2));
    let action_lpt = ActionLpt::new(1.0, lpt_order.max(2));

    for &s2 in &sigma2_grid {
        let sp = make_sp(s2, 1.0);

        let rec = match lpt_order {
            1 => VolumeCumulants::za(&sp),
            2 => VolumeCumulants::two_lpt(&sp, &gf),
            _ => VolumeCumulants::three_lpt(&sp, &gf),
        };
        let act = action_lpt.cumulants(&sp, lpt_order);

        k2_rec.push(rec.kappa2);
        k3_rec.push(rec.kappa3);
        k4_rec.push(rec.kappa4);
        k2_act.push(act.kappa[2]);
        k3_act.push(act.kappa[3]);
        k4_act.push(act.kappa[4]);
    }

    let source = format!(
        r##"#set page(width: 18cm, height: 12cm, margin: 8mm)
#set text(font: "New Computer Modern")
#import "@preview/lilaq:0.6.0" as lq
#let steelblue = rgb("#4682B4")
#let crimson = rgb("#DC143C")
#let seagreen = rgb("#2E8B57")
#let darkorange = rgb("#FF8C00")

#let s2 = {s2}
#let k2-rec = {k2_rec}
#let k3-rec = {k3_rec}
#let k4-rec = {k4_rec}
#let k2-act = {k2_act}
#let k3-act = {k3_act}
#let k4-act = {k4_act}

#lq.diagram(
  title: [Cumulants of V at {label}: recursion vs action formulation],
  xlabel: [$sigma^2$],
  ylabel: [$kappa_m (V)$],
  xlim: (0, 1.0),
  lq.plot(s2, k2-rec, stroke: (paint: steelblue, thickness: 2pt),
    label: [$kappa_2$ (recursion)]),
  lq.plot(s2, k2-act, stroke: (paint: steelblue, dash: "dashed", thickness: 1pt),
    mark: "o", mark-size: 2pt, label: [$kappa_2$ (action)]),
  lq.plot(s2, k3-rec, stroke: (paint: crimson, thickness: 2pt),
    label: [$kappa_3$ (recursion)]),
  lq.plot(s2, k3-act, stroke: (paint: crimson, dash: "dashed", thickness: 1pt),
    mark: "o", mark-size: 2pt, label: [$kappa_3$ (action)]),
  lq.plot(s2, k4-rec, stroke: (paint: seagreen, thickness: 2pt),
    label: [$kappa_4$ (recursion)]),
  lq.plot(s2, k4-act, stroke: (paint: seagreen, dash: "dashed", thickness: 1pt),
    mark: "o", mark-size: 2pt, label: [$kappa_4$ (action)]),
  lq.hlines(0, stroke: (dash: "dotted", paint: gray, thickness: 0.7pt)),
)
"##,
        s2 = fmt_array(&sigma2_grid),
        k2_rec = fmt_array(&k2_rec),
        k3_rec = fmt_array(&k3_rec),
        k4_rec = fmt_array(&k4_rec),
        k2_act = fmt_array(&k2_act),
        k3_act = fmt_array(&k3_act),
        k4_act = fmt_array(&k4_act),
        label = match lpt_order { 1 => "ZA", 2 => "2LPT", _ => "3LPT" },
    );

    compile_to_svg(&source)
}

/// Plot V-cumulants κ_m as a function of LPT order, at fixed σ².
pub fn render_cumulants_vs_order(sigma2_values: &[f64]) -> String {
    let orders: Vec<f64> = vec![1.0, 2.0, 3.0];
    let gf = LptGrowthFactors::eds(3);

    let mut all_k2 = Vec::new();
    let mut all_s3 = Vec::new();

    for &s2 in sigma2_values {
        let sp = make_sp(s2, 1.0);
        let za = VolumeCumulants::za(&sp);
        let two = VolumeCumulants::two_lpt(&sp, &gf);
        let three = VolumeCumulants::three_lpt(&sp, &gf);

        all_k2.push(vec![za.kappa2, two.kappa2, three.kappa2]);
        all_s3.push(vec![za.s3, two.s3, three.s3]);
    }

    let mut series_k2 = String::new();
    let mut series_s3 = String::new();
    let colors = ["steelblue", "crimson", "seagreen", "darkorange"];

    for (i, &s2) in sigma2_values.iter().enumerate() {
        let color = colors[i % colors.len()];
        series_k2.push_str(&format!(
            "  lq.plot(orders, {},\n    stroke: (paint: {}, thickness: 2pt),\n    mark: \"o\", mark-size: 4pt,\n    label: [$sigma^2 = {:.2}$]),\n",
            fmt_array(&all_k2[i]), color, s2,
        ));
        series_s3.push_str(&format!(
            "  lq.plot(orders, {},\n    stroke: (paint: {}, thickness: 2pt),\n    mark: \"o\", mark-size: 4pt,\n    label: [$sigma^2 = {:.2}$]),\n",
            fmt_array(&all_s3[i]), color, s2,
        ));
    }

    let source = format!(
        r##"#set page(width: 22cm, height: 11cm, margin: 8mm)
#set text(font: "New Computer Modern")
#import "@preview/lilaq:0.6.0" as lq
#let steelblue = rgb("#4682B4")
#let crimson = rgb("#DC143C")
#let seagreen = rgb("#2E8B57")
#let darkorange = rgb("#FF8C00")

#let orders = {orders}

#grid(columns: 2, gutter: 5mm,
  lq.diagram(
    width: 9.5cm, height: 9cm,
    title: [$kappa_2(V)$ vs LPT order],
    xlabel: [LPT order $N$],
    ylabel: [$kappa_2$],
    xaxis: (ticks: ((1, [ZA]), (2, [2LPT]), (3, [3LPT]))),
{series_k2}  ),
  lq.diagram(
    width: 9.5cm, height: 9cm,
    title: [Normalized skewness $S_3 = kappa_3 slash kappa_2^2$],
    xlabel: [LPT order $N$],
    ylabel: [$S_3$],
    xaxis: (ticks: ((1, [ZA]), (2, [2LPT]), (3, [3LPT]))),
{series_s3}  ),
)
"##,
        orders = fmt_array(&orders),
        series_k2 = series_k2,
        series_s3 = series_s3,
    );

    compile_to_svg(&source)
}

/// Plot the V probability density p(V) at several σ² values using the
/// saddle-point approximation (guaranteed positive, no oscillations).
pub fn render_v_pdf(sigma2_values: &[f64], lpt_order: usize) -> String {
    let v_grid: Vec<f64> = (-10..=200).map(|i| i as f64 * 0.02).collect(); // -0.2 to 4.0
    let gf = LptGrowthFactors::eds(lpt_order.max(2));

    let mut series = String::new();
    let colors = ["steelblue", "crimson", "seagreen", "darkorange"];

    for (i, &s2) in sigma2_values.iter().enumerate() {
        let sp = make_sp(s2, 1.0);
        let cumulants = match lpt_order {
            1 => VolumeCumulants::za(&sp),
            2 => VolumeCumulants::two_lpt(&sp, &gf),
            _ => VolumeCumulants::three_lpt(&sp, &gf),
        };
        let pdf = VolumePdf::new(cumulants);
        let p_vals: Vec<f64> = v_grid.iter().map(|&v| pdf.evaluate_saddlepoint(v)).collect();

        let color = colors[i % colors.len()];
        series.push_str(&format!(
            "  lq.plot(v, {},\n    stroke: (paint: {}, thickness: 2pt),\n    label: [$sigma^2 = {:.2}$]),\n",
            fmt_array(&p_vals), color, s2,
        ));
    }

    let source = format!(
        r##"#set page(width: 18cm, height: 11cm, margin: 8mm)
#set text(font: "New Computer Modern")
#import "@preview/lilaq:0.6.0" as lq
#let steelblue = rgb("#4682B4")
#let crimson = rgb("#DC143C")
#let seagreen = rgb("#2E8B57")
#let darkorange = rgb("#FF8C00")

#let v = {v_grid}

#lq.diagram(
  title: [$p(V)$ from saddle-point approximation ({label})],
  xlabel: [$V = det(I + G)$],
  ylabel: [$p(V)$],
  xlim: (0, 3.5),
  lq.vlines(1, stroke: (dash: "dotted", paint: gray, thickness: 0.7pt)),
  lq.vlines(0, stroke: (dash: "dotted", paint: black, thickness: 0.7pt)),
{series})
"##,
        v_grid = fmt_array(&v_grid),
        series = series,
        label = match lpt_order { 1 => "ZA", 2 => "2LPT", _ => "3LPT" },
    );

    compile_to_svg(&source)
}

/// Edgeworth convergence: show p(V) at orders 0–8 alongside the saddle-point
/// reference, and plot residuals from the 8th-order Edgeworth for orders 0–7.
///
/// The orders correspond to successive terms in the expansion of
/// exp(λ₃t³ + λ₄t⁴):
///   0: Gaussian; 1:+He₃; 2:+He₄; 3:+He₆; 4:+He₇; 5:+He₈; 6:+He₉;
///   7:+He₁₀; 8:+He₁₁,He₁₂.
pub fn render_edgeworth_convergence(sigma2: f64, lpt_order: usize) -> String {
    let v_grid: Vec<f64> = (-10..=200).map(|i| i as f64 * 0.02).collect();
    let gf = LptGrowthFactors::eds(lpt_order.max(2));

    let sp = make_sp(sigma2, 1.0);
    let cumulants = match lpt_order {
        1 => VolumeCumulants::za(&sp),
        2 => VolumeCumulants::two_lpt(&sp, &gf),
        _ => VolumeCumulants::three_lpt(&sp, &gf),
    };
    let pdf = VolumePdf::new(cumulants);

    // Reference: saddle-point and 8th-order Edgeworth
    let p_sp: Vec<f64> = v_grid.iter().map(|&v| pdf.evaluate_saddlepoint(v)).collect();
    let p8: Vec<f64> = v_grid.iter().map(|&v| pdf.evaluate_at_order(v, 8)).collect();

    // All 9 orders (0–8)
    let mut p_orders: Vec<Vec<f64>> = Vec::new();
    for n in 0..=8 {
        let p: Vec<f64> = v_grid.iter().map(|&v| pdf.evaluate_at_order(v, n)).collect();
        p_orders.push(p);
    }

    // Build PDF series string for orders 0–8
    let mut pdf_series = String::new();
    let colors = [
        "rgb(\"#1f77b4\")", "rgb(\"#ff7f0e\")", "rgb(\"#2ca02c\")",
        "rgb(\"#d62728\")", "rgb(\"#9467bd\")", "rgb(\"#8c564b\")",
        "rgb(\"#e377c2\")", "rgb(\"#7f7f7f\")", "rgb(\"#17becf\")",
    ];
    for n in 0..=8 {
        pdf_series.push_str(&format!(
            "    lq.plot(v, {}, stroke: (paint: {}, thickness: 1.2pt), label: [order {}]),\n",
            fmt_array(&p_orders[n]), colors[n], n,
        ));
    }
    // Overlay the saddle-point reference
    pdf_series.push_str(&format!(
        "    lq.plot(v, {}, stroke: (paint: black, dash: \"dashed\", thickness: 2pt), label: [saddle-point]),\n",
        fmt_array(&p_sp),
    ));

    // Residuals from 8th order: (p_N - p_8) for N = 0..7
    let mut resid_series = String::new();
    for n in 0..=7 {
        let resid: Vec<f64> = p_orders[n].iter().zip(p8.iter())
            .map(|(&pn, &p8v)| pn - p8v)
            .collect();
        resid_series.push_str(&format!(
            "    lq.plot(v, {}, stroke: (paint: {}, thickness: 1.2pt), label: [order {}]),\n",
            fmt_array(&resid), colors[n], n,
        ));
    }
    // Also residual of saddle-point vs order 8
    let resid_sp: Vec<f64> = p_sp.iter().zip(p8.iter())
        .map(|(&psp, &p8v)| psp - p8v)
        .collect();
    resid_series.push_str(&format!(
        "    lq.plot(v, {}, stroke: (paint: black, dash: \"dashed\", thickness: 2pt), label: [saddle-point]),\n",
        fmt_array(&resid_sp),
    ));

    let label = match lpt_order { 1 => "ZA", 2 => "2LPT", _ => "3LPT" };

    let source = format!(
        r##"#set page(width: 26cm, height: 11cm, margin: 8mm)
#set text(font: "New Computer Modern")
#import "@preview/lilaq:0.6.0" as lq

#let v = {v_grid}

#grid(columns: 2, gutter: 5mm,
  lq.diagram(
    width: 11.5cm, height: 9cm,
    title: [$p(V)$ Edgeworth convergence at $sigma^2 = {sigma2:.2}$ ({label})],
    xlabel: [$V$],
    ylabel: [$p(V)$],
    xlim: (0, 3.5),
    lq.vlines(1, stroke: (dash: "dotted", paint: gray, thickness: 0.7pt)),
    lq.vlines(0, stroke: (dash: "dotted", paint: black, thickness: 0.7pt)),
{pdf_series}  ),
  lq.diagram(
    width: 11.5cm, height: 9cm,
    title: [Residual $p_N - p_8$ (Edgeworth $N = 0..7$ vs 8th order)],
    xlabel: [$V$],
    ylabel: [$p_N(V) - p_8(V)$],
    xlim: (0, 3.5),
    lq.hlines(0, stroke: (dash: "dotted", paint: black, thickness: 0.7pt)),
{resid_series}  ),
)
"##,
        v_grid = fmt_array(&v_grid),
        pdf_series = pdf_series,
        resid_series = resid_series,
        sigma2 = sigma2,
        label = label,
    );

    compile_to_svg(&source)
}

/// Plot of EFT order convergence: cumulants with c₁ varied.
pub fn render_eft_order_convergence(sigma2: f64) -> String {
    let c1_grid: Vec<f64> = (0..=50).map(|i| i as f64 * 0.04 - 1.0).collect(); // -1 to 1

    let sp = make_sp(sigma2, 1.0);
    let gf = LptGrowthFactors::eds(2);

    let mut k2_vals = Vec::new();
    let mut k3_vals = Vec::new();
    let mut k4_vals = Vec::new();

    for &c1 in &c1_grid {
        let eft = EftParams::trace_only(c1, 2.0);
        let c = VolumeCumulants::compute(&sp, &gf, 2, Some(&eft));
        k2_vals.push(c.kappa2);
        k3_vals.push(c.kappa3);
        k4_vals.push(c.kappa4);
    }

    let source = format!(
        r##"#set page(width: 18cm, height: 11cm, margin: 8mm)
#set text(font: "New Computer Modern")
#import "@preview/lilaq:0.6.0" as lq
#let steelblue = rgb("#4682B4")
#let crimson = rgb("#DC143C")
#let seagreen = rgb("#2E8B57")

#let c1 = {c1}
#let k2 = {k2}
#let k3 = {k3}
#let k4 = {k4}

#lq.diagram(
  title: [V-cumulants vs EFT counterterm $c_1$ at $sigma^2 = {sigma2:.2}$, $R_* = 2$ h$""^(-1)$ Mpc],
  xlabel: [$c_1$ (longitudinal counterterm)],
  ylabel: [$kappa_m (V)$],
  lq.plot(c1, k2, stroke: (paint: steelblue, thickness: 2pt),
    label: [$kappa_2$]),
  lq.plot(c1, k3, stroke: (paint: crimson, thickness: 2pt),
    label: [$kappa_3$]),
  lq.plot(c1, k4, stroke: (paint: seagreen, thickness: 2pt),
    label: [$kappa_4$]),
  lq.vlines(0, stroke: (dash: "dotted", paint: black, thickness: 0.7pt)),
)
"##,
        c1 = fmt_array(&c1_grid),
        k2 = fmt_array(&k2_vals),
        k3 = fmt_array(&k3_vals),
        k4 = fmt_array(&k4_vals),
        sigma2 = sigma2,
    );

    compile_to_svg(&source)
}

/// Multi-panel summary: σ² range extended to 1.0 to see where each
/// expansion breaks down. PDF panel uses saddle-point (no oscillations).
pub fn render_summary(sigma2_values: &[f64]) -> String {
    let v_grid: Vec<f64> = (-10..=200).map(|i| i as f64 * 0.02).collect();
    // Extended σ² range to 1.0 so breakdown is visible
    let s2_dense: Vec<f64> = (1..=200).map(|i| i as f64 * 0.005).collect();
    let gf = LptGrowthFactors::eds(3);

    // Panel 1: PDF at different σ² using saddle-point
    let mut pdf_series = String::new();
    let colors = ["steelblue", "crimson", "seagreen", "darkorange"];
    for (i, &s2) in sigma2_values.iter().enumerate() {
        let sp = make_sp(s2, 1.0);
        let pdf = VolumePdf::new(VolumeCumulants::three_lpt(&sp, &gf));
        let p: Vec<f64> = v_grid.iter().map(|&v| pdf.evaluate_saddlepoint(v)).collect();
        let c = colors[i % colors.len()];
        pdf_series.push_str(&format!(
            "  lq.plot(v, {}, stroke: (paint: {}, thickness: 1.5pt), label: [$sigma^2={:.2}$]),\n",
            fmt_array(&p), c, s2,
        ));
    }

    // Panel 2: κ₂ vs σ² for each LPT order
    let mut k2_za = Vec::new();
    let mut k2_2lpt = Vec::new();
    let mut k2_3lpt = Vec::new();
    let mut s3_za = Vec::new();
    let mut s3_2lpt = Vec::new();
    let mut s3_3lpt = Vec::new();
    for &s2 in &s2_dense {
        let sp = make_sp(s2, 1.0);
        let za = VolumeCumulants::za(&sp);
        let two = VolumeCumulants::two_lpt(&sp, &gf);
        let three = VolumeCumulants::three_lpt(&sp, &gf);
        k2_za.push(za.kappa2);
        k2_2lpt.push(two.kappa2);
        k2_3lpt.push(three.kappa2);
        s3_za.push(za.s3);
        s3_2lpt.push(two.s3);
        s3_3lpt.push(three.s3);
    }

    // Panel 4: Edgeworth convergence residuals at σ²=0.3 (showing orders 0..7 vs 8)
    let sigma2_probe = 0.3;
    let sp_probe = make_sp(sigma2_probe, 1.0);
    let cumulants_probe = VolumeCumulants::three_lpt(&sp_probe, &gf);
    let pdf_probe = VolumePdf::new(cumulants_probe);
    let p8: Vec<f64> = v_grid.iter().map(|&v| pdf_probe.evaluate_at_order(v, 8)).collect();

    let orders_to_show = [0usize, 1, 2, 3, 4, 5, 6, 7];
    let mut resid_series = String::new();
    let resid_colors = [
        "rgb(\"#1f77b4\")", "rgb(\"#ff7f0e\")", "rgb(\"#2ca02c\")",
        "rgb(\"#d62728\")", "rgb(\"#9467bd\")", "rgb(\"#8c564b\")",
        "rgb(\"#e377c2\")", "rgb(\"#7f7f7f\")",
    ];
    for (idx, &n) in orders_to_show.iter().enumerate() {
        let p_n: Vec<f64> = v_grid.iter().map(|&v| pdf_probe.evaluate_at_order(v, n)).collect();
        let resid: Vec<f64> = p_n.iter().zip(p8.iter())
            .map(|(&pn, &p8v)| pn - p8v)
            .collect();
        resid_series.push_str(&format!(
            "    lq.plot(v, {}, stroke: (paint: {}, thickness: 1.2pt), label: [$N={}$]),\n",
            fmt_array(&resid), resid_colors[idx], n,
        ));
    }
    let p_sp_probe: Vec<f64> = v_grid.iter().map(|&v| pdf_probe.evaluate_saddlepoint(v)).collect();
    let resid_sp: Vec<f64> = p_sp_probe.iter().zip(p8.iter())
        .map(|(&psp, &p8v)| psp - p8v)
        .collect();
    resid_series.push_str(&format!(
        "    lq.plot(v, {}, stroke: (paint: black, dash: \"dashed\", thickness: 1.5pt), label: [saddle-pt]),\n",
        fmt_array(&resid_sp),
    ));

    let source = format!(
        r##"#set page(width: 28cm, height: 20cm, margin: 8mm)
#set text(font: "New Computer Modern")
#import "@preview/lilaq:0.6.0" as lq
#let steelblue = rgb("#4682B4")
#let crimson = rgb("#DC143C")
#let seagreen = rgb("#2E8B57")
#let darkorange = rgb("#FF8C00")

#let v = {v_grid}
#let s2 = {s2_dense}

#grid(columns: 2, gutter: 5mm,
  // Panel 1: V PDF via saddle-point (positive-definite)
  lq.diagram(
    width: 13cm, height: 9cm,
    title: [$p(V)$ at 3LPT (saddle-point) for varying $sigma^2$],
    xlabel: [$V = det(I + G)$],
    ylabel: [$p(V)$],
    xlim: (-0.1, 3.0),
    lq.vlines(1, stroke: (dash: "dotted", paint: gray)),
    lq.vlines(0, stroke: (dash: "dashed", paint: black)),
{pdf_series}  ),

  // Panel 2: kappa_2 vs sigma^2 at each LPT order (extended to sigma^2=1)
  lq.diagram(
    width: 13cm, height: 9cm,
    title: [$kappa_2(V)$ convergence with LPT order (extended range)],
    xlabel: [$sigma^2$],
    ylabel: [$kappa_2$],
    xlim: (0, 1.0),
    lq.plot(s2, {k2_za}, stroke: (paint: steelblue, thickness: 2pt), label: [ZA]),
    lq.plot(s2, {k2_2lpt}, stroke: (paint: crimson, thickness: 2pt), label: [2LPT]),
    lq.plot(s2, {k2_3lpt}, stroke: (paint: seagreen, thickness: 2pt), label: [3LPT]),
  ),

  // Panel 3: S_3 vs sigma^2 at each LPT order
  lq.diagram(
    width: 13cm, height: 9cm,
    title: [Normalized skewness $S_3 = kappa_3 slash kappa_2^2$ (extended range)],
    xlabel: [$sigma^2$],
    ylabel: [$S_3$],
    xlim: (0, 1.0),
    lq.hlines(2, stroke: (dash: "dotted", paint: gray, thickness: 0.7pt)),
    lq.plot(s2, {s3_za}, stroke: (paint: steelblue, thickness: 2pt), label: [ZA]),
    lq.plot(s2, {s3_2lpt}, stroke: (paint: crimson, thickness: 2pt), label: [2LPT]),
    lq.plot(s2, {s3_3lpt}, stroke: (paint: seagreen, thickness: 2pt), label: [3LPT]),
  ),

  // Panel 4: Edgeworth convergence residuals at sigma^2=0.3
  lq.diagram(
    width: 13cm, height: 9cm,
    title: [Edgeworth residual $p_N(V) - p_8(V)$ at $sigma^2 = {sigma2_probe:.2}$ (3LPT)],
    xlabel: [$V$],
    ylabel: [$p_N(V) - p_8(V)$],
    xlim: (0, 3.0),
    lq.hlines(0, stroke: (dash: "dotted", paint: black, thickness: 0.7pt)),
{resid_series}  ),
)
"##,
        v_grid = fmt_array(&v_grid),
        s2_dense = fmt_array(&s2_dense),
        pdf_series = pdf_series,
        k2_za = fmt_array(&k2_za),
        k2_2lpt = fmt_array(&k2_2lpt),
        k2_3lpt = fmt_array(&k2_3lpt),
        s3_za = fmt_array(&s3_za),
        s3_2lpt = fmt_array(&s3_2lpt),
        s3_3lpt = fmt_array(&s3_3lpt),
        resid_series = resid_series,
        sigma2_probe = sigma2_probe,
    );

    compile_to_svg(&source)
}

//! Quick test of symlog rendering.
use twopoint::plotting::TypstPlotter;

fn try_compile(plotter: &TypstPlotter, name: &str, source: &str) {
    eprintln!("Testing {}...", name);
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| plotter.render_pdf(source))) {
        Ok(pdf) => {
            let path = format!("/tmp/{}.pdf", name);
            std::fs::write(&path, &pdf).unwrap();
            eprintln!("  OK ({} bytes) -> {}", pdf.len(), path);
        }
        Err(_) => {
            eprintln!("  FAILED");
        }
    }
}

fn main() {
    let plotter = TypstPlotter::new();

    // Test 1: symlog basic
    try_compile(&plotter, "sym1_basic", r##"#set page(width: 16cm, height: 11cm, margin: 5mm)
#set text(font: "New Computer Modern")
#import "@preview/lilaq:0.6.0" as lq
#lq.diagram(
  xscale: "symlog", yscale: "symlog",
  lq.plot((1,2,3,10,100), (0.1, 1, 5, 50, 200)),
)
"##);

    // Test 2: symlog with negative values
    try_compile(&plotter, "sym2_negatives", r##"#set page(width: 16cm, height: 11cm, margin: 5mm)
#set text(font: "New Computer Modern")
#import "@preview/lilaq:0.6.0" as lq
#lq.diagram(
  xscale: "symlog", yscale: "symlog",
  lq.plot((1,2,3,10,100), (-5, 1, 5, 50, -10)),
)
"##);

    // Test 3: symlog with data range 3-45 (like validate)
    try_compile(&plotter, "sym3_range", r##"#set page(width: 16cm, height: 11cm, margin: 5mm)
#set text(font: "New Computer Modern")
#import "@preview/lilaq:0.6.0" as lq
#lq.diagram(
  xscale: "symlog", yscale: "symlog",
  lq.plot((3.7, 10.8, 20.8, 30.8, 45.0), (7.2, 0.7, 0.17, 0.09, -0.16)),
)
"##);

    // Test 4: symlog with custom threshold
    try_compile(&plotter, "sym4_threshold", r##"#set page(width: 16cm, height: 11cm, margin: 5mm)
#set text(font: "New Computer Modern")
#import "@preview/lilaq:0.6.0" as lq
#lq.diagram(
  xscale: lq.scale.symlog(threshold: 0.01), yscale: lq.scale.symlog(threshold: 0.01),
  lq.plot((3.7, 10.8, 20.8, 30.8, 45.0), (7.2, 0.7, 0.17, 0.09, -0.16)),
)
"##);

    // Test 5: symlog x, linear y
    try_compile(&plotter, "sym5_mixed", r##"#set page(width: 16cm, height: 11cm, margin: 5mm)
#set text(font: "New Computer Modern")
#import "@preview/lilaq:0.6.0" as lq
#lq.diagram(
  xscale: "symlog", yscale: "linear",
  lq.plot((3.7, 10.8, 20.8, 30.8, 45.0), (7.2, 0.7, 0.17, 0.09, -0.16)),
)
"##);

    // Test 6: symlog x, symlog y, all positive
    try_compile(&plotter, "sym6_allpos", r##"#set page(width: 16cm, height: 11cm, margin: 5mm)
#set text(font: "New Computer Modern")
#import "@preview/lilaq:0.6.0" as lq
#lq.diagram(
  xscale: "symlog", yscale: "symlog",
  lq.plot((3.7, 10.8, 20.8, 30.8, 45.0), (7.2, 0.7, 0.17, 0.09, 0.04)),
)
"##);

    // Test 7: symlog with fill-between
    try_compile(&plotter, "sym7_fill", r##"#set page(width: 16cm, height: 11cm, margin: 5mm)
#set text(font: "New Computer Modern")
#import "@preview/lilaq:0.6.0" as lq
#lq.diagram(
  xscale: "symlog", yscale: "symlog",
  lq.fill-between((3.7, 10.8, 20.8, 30.8, 45.0), (6.0, 0.5, 0.1, 0.05, -0.2), y2: (8.0, 0.9, 0.25, 0.13, 0.1),
    fill: blue.lighten(80%),
    stroke: none),
  lq.plot((3.7, 10.8, 20.8, 30.8, 45.0), (7.2, 0.7, 0.17, 0.09, -0.05)),
)
"##);

    // Test 8: 30-element arrays with symlog (like actual data)
    let r30: Vec<String> = (0..30).map(|i| format!("{:.4}", 3.7 + i as f64 * 1.42)).collect();
    let v30: Vec<String> = (0..30).map(|i| format!("{:.4}", 7.0 * (-0.15 * i as f64).exp())).collect();
    let lo30: Vec<String> = (0..30).map(|i| format!("{:.4}", 6.5 * (-0.15 * i as f64).exp())).collect();
    let hi30: Vec<String> = (0..30).map(|i| format!("{:.4}", 7.5 * (-0.15 * i as f64).exp())).collect();
    let src8 = format!(r##"#set page(width: 16cm, height: 11cm, margin: 5mm)
#set text(font: "New Computer Modern")
#import "@preview/lilaq:0.6.0" as lq
#let r = ({r})
#let mean-xi = ({v})
#let lower = ({lo})
#let upper = ({hi})
#lq.diagram(
  xscale: "symlog", yscale: "symlog",
  lq.fill-between(r, lower, y2: upper,
    fill: blue.lighten(80%),
    stroke: none),
  lq.plot(r, mean-xi,
    stroke: (paint: blue, thickness: 2pt)),
)
"##, r=r30.join(", "), v=v30.join(", "), lo=lo30.join(", "), hi=hi30.join(", "));
    try_compile(&plotter, "sym8_big", &src8);

    eprintln!("\nDone!");
}

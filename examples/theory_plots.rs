//! Generate theory-prediction SVG plots: cumulants, V PDF, formulation comparison.
//!
//! Run with: cargo run --example theory_plots --features typst-plots --release
//!
//! Outputs SVGs to /tmp/theory_*.svg

use twopoint::theory_plots;

fn main() {
    let out_dir = std::env::args().nth(1).unwrap_or_else(|| "/tmp".to_string());

    let sigma2_grid = vec![0.05, 0.1, 0.2, 0.3];

    eprintln!("Rendering V PDF (multi-σ², 3LPT)...");
    let svg = theory_plots::render_v_pdf(&sigma2_grid, 3);
    let path = format!("{}/theory_v_pdf_3lpt.svg", out_dir);
    std::fs::write(&path, &svg).unwrap();
    eprintln!("  → {} ({} bytes)", path, svg.len());

    eprintln!("Rendering V PDF at ZA...");
    let svg = theory_plots::render_v_pdf(&sigma2_grid, 1);
    let path = format!("{}/theory_v_pdf_za.svg", out_dir);
    std::fs::write(&path, &svg).unwrap();
    eprintln!("  → {} ({} bytes)", path, svg.len());

    eprintln!("Rendering cumulants vs LPT order...");
    let svg = theory_plots::render_cumulants_vs_order(&sigma2_grid);
    let path = format!("{}/theory_cumulants_vs_order.svg", out_dir);
    std::fs::write(&path, &svg).unwrap();
    eprintln!("  → {} ({} bytes)", path, svg.len());

    eprintln!("Rendering recursion vs action at ZA...");
    let svg = theory_plots::render_formulation_comparison(1);
    let path = format!("{}/theory_formulation_za.svg", out_dir);
    std::fs::write(&path, &svg).unwrap();
    eprintln!("  → {} ({} bytes)", path, svg.len());

    eprintln!("Rendering recursion vs action at 2LPT...");
    let svg = theory_plots::render_formulation_comparison(2);
    let path = format!("{}/theory_formulation_2lpt.svg", out_dir);
    std::fs::write(&path, &svg).unwrap();
    eprintln!("  → {} ({} bytes)", path, svg.len());

    eprintln!("Rendering recursion vs action at 3LPT...");
    let svg = theory_plots::render_formulation_comparison(3);
    let path = format!("{}/theory_formulation_3lpt.svg", out_dir);
    std::fs::write(&path, &svg).unwrap();
    eprintln!("  → {} ({} bytes)", path, svg.len());

    eprintln!("Rendering EFT order convergence...");
    let svg = theory_plots::render_eft_order_convergence(0.2);
    let path = format!("{}/theory_eft_convergence.svg", out_dir);
    std::fs::write(&path, &svg).unwrap();
    eprintln!("  → {} ({} bytes)", path, svg.len());

    eprintln!("Rendering 4-panel summary...");
    let svg = theory_plots::render_summary(&sigma2_grid);
    let path = format!("{}/theory_summary.svg", out_dir);
    std::fs::write(&path, &svg).unwrap();
    eprintln!("  → {} ({} bytes)", path, svg.len());

    eprintln!("\nAll plots written to {}/", out_dir);
}

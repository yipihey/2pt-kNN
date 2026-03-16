//! Quick test of just the plotting pipeline — no validation needed.
use twopoint::validation::ValidationResult;
use twopoint::plotting;

fn main() {
    // Fake validation result with 5 bins
    let result = ValidationResult {
        r_centers: vec![10.0, 30.0, 60.0, 100.0, 150.0],
        xi_analytic: vec![0.5, 0.2, 0.08, 0.03, 0.01],
        mean_xi: vec![0.48, 0.22, 0.07, 0.035, 0.012],
        std_xi: vec![0.05, 0.03, 0.02, 0.01, 0.005],
        stderr_xi: vec![0.02, 0.01, 0.008, 0.004, 0.002],
        chi2: 4.5,
        chi2_per_dof: 1.1,
        n_mocks: 5,
        knn_cdfs: vec![
            vec![0.1, 0.3, 0.5, 0.7, 0.9],
            vec![0.05, 0.2, 0.4, 0.6, 0.85],
        ],
        dilution_xi: vec![
            vec![0.48, 0.22, 0.07, 0.035, 0.012],
        ],
        dilution_variance: vec![
            vec![0.0025, 0.0009, 0.0004, 0.0001, 0.000025],
        ],
        dilution_r_char: vec![50.0],
    };

    eprintln!("Rendering xi plot...");
    let svg = plotting::render_xi_plot(&result);
    std::fs::write("/tmp/test_xi.svg", &svg).unwrap();
    eprintln!("Wrote /tmp/test_xi.svg ({} bytes)", svg.len());

    eprintln!("Rendering CDF plot...");
    let svg = plotting::render_cdf_plot(&result);
    std::fs::write("/tmp/test_cdf.svg", &svg).unwrap();
    eprintln!("Wrote /tmp/test_cdf.svg ({} bytes)", svg.len());

    eprintln!("Rendering r2xi plot...");
    let svg = plotting::render_r2xi_plot(&result);
    std::fs::write("/tmp/test_r2xi.svg", &svg).unwrap();
    eprintln!("Wrote /tmp/test_r2xi.svg ({} bytes)", svg.len());

    eprintln!("Rendering dilution plot...");
    let svg = plotting::render_dilution_plot(&result);
    std::fs::write("/tmp/test_dilution.svg", &svg).unwrap();
    eprintln!("Wrote /tmp/test_dilution.svg ({} bytes)", svg.len());
}

use twopoint::validation::{ValidationConfig, BruteForceKnnBackend, run_validation};
use twopoint::plotting;
use twopoint::mock::CoxMockParams;

fn main() {
    let config = ValidationConfig {
        n_mocks: 2,
        k_max: 4,
        n_bins: 10,
        r_min: 10.0,
        r_max: 200.0,
        random_ratio: 2,
        params: CoxMockParams::euclid_small(),
        max_dilution_level: 1,
        box_size: None,
    };

    let backend = BruteForceKnnBackend;
    eprintln!("Running validation...");
    let result = run_validation(&config, &backend);
    eprintln!("Done. chi2/dof = {:.3}", result.chi2_per_dof);

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

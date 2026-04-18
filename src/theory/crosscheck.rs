//! Cross-validation between recursion-based and action-based formulations.
//!
//! Both approaches must give identical V-cumulants at each LPT order.
//! This module provides systematic tests and benchmarks.

#[cfg(test)]
mod tests {
    use crate::theory::spectral::SpectralParams;
    use crate::theory::growth::LptGrowthFactors;
    use crate::theory::cumulants::VolumeCumulants;
    use crate::theory::eft::EftParams;
    use crate::theory::action::action_lpt::ActionLpt;
    use crate::theory::action::action_eft::ActionEft;
    use crate::theory::action::diagrams::{DiagramEngine, complexity_comparison};

    fn make_sp(sigma2: f64, gamma: f64) -> SpectralParams {
        SpectralParams {
            mass: 1e12, radius: 10.0, sigma2, gamma, gamma_n: vec![0.0],
        }
    }

    // ---- ZA cross-checks ----

    #[test]
    fn crosscheck_za_kappa2() {
        for &s2 in &[0.01, 0.05, 0.1, 0.3, 0.5] {
            let sp = make_sp(s2, 1.0);

            let recursion = VolumeCumulants::za(&sp);
            let action = ActionLpt::new(1.0, 1).cumulants_za(&sp);
            let diagrams = DiagramEngine::new(1).compute_cumulants(&sp);

            let tol = 1e-12 * recursion.kappa2.abs().max(1e-15);
            assert!((recursion.kappa2 - action.kappa[2]).abs() < tol,
                    "ZA κ₂ mismatch at σ²={}: rec={}, act={}", s2, recursion.kappa2, action.kappa[2]);
            assert!((recursion.kappa2 - diagrams.kappa[2]).abs() < tol,
                    "ZA κ₂ mismatch at σ²={}: rec={}, diag={}", s2, recursion.kappa2, diagrams.kappa[2]);
        }
    }

    #[test]
    fn crosscheck_za_kappa3() {
        for &s2 in &[0.01, 0.1, 0.3] {
            let sp = make_sp(s2, 1.0);

            let recursion = VolumeCumulants::za(&sp);
            let action = ActionLpt::new(1.0, 1).cumulants_za(&sp);

            let tol = 1e-12 * recursion.kappa3.abs().max(1e-15);
            assert!((recursion.kappa3 - action.kappa[3]).abs() < tol,
                    "ZA κ₃ mismatch at σ²={}: rec={}, act={}", s2, recursion.kappa3, action.kappa[3]);
        }
    }

    #[test]
    fn crosscheck_za_kappa4() {
        for &s2 in &[0.01, 0.1, 0.3] {
            let sp = make_sp(s2, 1.0);

            let recursion = VolumeCumulants::za(&sp);
            let action = ActionLpt::new(1.0, 1).cumulants_za(&sp);

            let tol = 1e-12 * recursion.kappa4.abs().max(1e-15);
            assert!((recursion.kappa4 - action.kappa[4]).abs() < tol,
                    "ZA κ₄ mismatch at σ²={}: rec={}, act={}", s2, recursion.kappa4, action.kappa[4]);
        }
    }

    // ---- 2LPT cross-checks ----

    #[test]
    fn crosscheck_2lpt_eds() {
        let sp = make_sp(0.3, 1.0);
        let gf = LptGrowthFactors::eds(2);
        let recursion = VolumeCumulants::two_lpt(&sp, &gf);
        let action = ActionLpt::new(1.0, 2).cumulants_2lpt(&sp); // EdS

        let tol = 1e-12;
        assert!((recursion.kappa2 - action.kappa[2]).abs() < tol,
                "2LPT κ₂ EdS: rec={}, act={}", recursion.kappa2, action.kappa[2]);
        assert!((recursion.kappa3 - action.kappa[3]).abs() < tol,
                "2LPT κ₃ EdS: rec={}, act={}", recursion.kappa3, action.kappa[3]);
    }

    #[test]
    fn crosscheck_2lpt_lcdm() {
        let sp = make_sp(0.2, 1.2);
        let omega_m = 0.3111;
        let gf = LptGrowthFactors::lcdm(2, omega_m);
        let recursion = VolumeCumulants::two_lpt(&sp, &gf);
        let action = ActionLpt::new(omega_m, 2).cumulants_2lpt(&sp);

        let tol = 1e-12;
        assert!((recursion.kappa2 - action.kappa[2]).abs() < tol,
                "2LPT κ₂ ΛCDM: rec={}, act={}", recursion.kappa2, action.kappa[2]);
    }

    // ---- 3LPT cross-checks ----

    #[test]
    fn crosscheck_3lpt_eds() {
        let sp = make_sp(0.2, 1.0);
        let gf = LptGrowthFactors::eds(3);
        let recursion = VolumeCumulants::three_lpt(&sp, &gf);
        let action = ActionLpt::new(1.0, 3).cumulants_3lpt(&sp);

        let tol = 1e-10;
        assert!((recursion.kappa2 - action.kappa[2]).abs() < tol,
                "3LPT κ₂ EdS: rec={}, act={}", recursion.kappa2, action.kappa[2]);
        assert!((recursion.kappa3 - action.kappa[3]).abs() < tol,
                "3LPT κ₃ EdS: rec={}, act={}", recursion.kappa3, action.kappa[3]);
    }

    // ---- EFT cross-checks ----

    #[test]
    fn crosscheck_eft_variance_correction() {
        let sp = make_sp(0.3, 1.0);

        // Recursion EFT
        let eft_rec = EftParams::trace_only(1.5, 2.0);
        let (dk2_rec, _, _) = eft_rec.cumulant_corrections(&sp);

        // Action EFT: same physics, different parameterization
        let eft_act = ActionEft {
            c_long: 1.5,
            r_star: 2.0,
            ..ActionEft::default()
        };
        let mut c = ActionLpt::new(1.0, 1).cumulants_za(&sp);
        let k2_before = c.kappa[2];
        eft_act.apply_corrections(&mut c, &sp);
        let dk2_act = c.kappa[2] - k2_before;

        assert!((dk2_rec - dk2_act).abs() < 1e-12,
                "EFT δκ₂: rec={}, act={}", dk2_rec, dk2_act);
    }

    // ---- Normalized cumulant cross-checks ----

    #[test]
    fn crosscheck_s3_recursion_vs_action() {
        for &s2 in &[0.01, 0.05, 0.1, 0.3] {
            let sp = make_sp(s2, 1.0);

            let rec = VolumeCumulants::za(&sp);
            let act = ActionLpt::new(1.0, 1).cumulants_za(&sp);

            let tol = 1e-10;
            assert!((rec.s3 - act.s3()).abs() < tol,
                    "S₃ rec vs act at σ²={}: {} vs {}", s2, rec.s3, act.s3());
        }
    }

    #[test]
    fn crosscheck_s4_recursion_vs_action() {
        for &s2 in &[0.01, 0.05, 0.1] {
            let sp = make_sp(s2, 1.0);

            let rec = VolumeCumulants::za(&sp);
            let act = ActionLpt::new(1.0, 1).cumulants_za(&sp);

            let tol = 1e-10;
            assert!((rec.s4 - act.s4()).abs() < tol,
                    "S₄ rec vs act at σ²={}: {} vs {}", s2, rec.s4, act.s4());
        }
    }

    // ---- Computational efficiency ----

    #[test]
    fn benchmark_complexity_scaling() {
        for order in 1..=5 {
            let (rec, act, diag) = complexity_comparison(order);
            eprintln!("Order {}: recursion={}, action={}, diagrams={}, speedup={}×",
                     order, rec, act, diag, rec as f64 / act as f64);
        }
    }

    #[test]
    fn benchmark_wall_time() {
        use std::time::Instant;

        let sp = make_sp(0.2, 1.0);
        let n_iter = 10000;

        // Recursion timing
        let start = Instant::now();
        for _ in 0..n_iter {
            let gf = LptGrowthFactors::eds(2);
            let _ = VolumeCumulants::two_lpt(&sp, &gf);
        }
        let t_rec = start.elapsed();

        // Action timing
        let start = Instant::now();
        let lpt = ActionLpt::new(1.0, 2);
        for _ in 0..n_iter {
            let _ = lpt.cumulants_2lpt(&sp);
        }
        let t_act = start.elapsed();

        // Diagram timing
        let start = Instant::now();
        let engine = DiagramEngine::new(2);
        for _ in 0..n_iter {
            let _ = engine.compute_cumulants(&sp);
        }
        let t_diag = start.elapsed();

        eprintln!("2LPT wall time ({} iterations):", n_iter);
        eprintln!("  Recursion:   {:?}", t_rec);
        eprintln!("  Action:      {:?}", t_act);
        eprintln!("  Diagrammatic:{:?}", t_diag);
        eprintln!("  Action/Rec speedup: {:.2}×", t_rec.as_nanos() as f64 / t_act.as_nanos() as f64);
    }
}

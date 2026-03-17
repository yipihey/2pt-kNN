//! Phase 3: Density–gravity consistency tests.
//!
//! At each node v on galaxy i's ancestry path, compare the measured density
//! inertia tensor eigenvalues against the prediction from the gravity-kernel
//! Doroshkevich eigenvalues.
//!
//! At 1LPT, the predicted density inertia eigenvalues are:
//!   Q_ii^pred = (L²/12)(1 − D₊λᵢ)² + variance from sub-cell structure
//!
//! The LPT convergence diagnostic χ²(v) tracks where the perturbative
//! expansion breaks down as a function of tree depth (scale).

use super::eigen::{Sym3x3, sym3x3_eigenvalues};
use super::tidal::WebType;

/// LPT order for predictions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LptOrder {
    Lpt1,
    Lpt2,
    Lpt3,
}

/// Per-node LPT convergence diagnostic.
#[derive(Debug, Clone)]
pub struct LptDiagnostic {
    /// Tree depth of this node.
    pub depth: usize,
    /// Characteristic scale at this depth.
    pub scale: f64,
    /// χ² of density inertia vs 1LPT prediction.
    pub chi2_1lpt: f64,
    /// χ² of density inertia vs 2LPT prediction.
    pub chi2_2lpt: f64,
    /// Web type classification at this scale.
    pub web_type: WebType,
    /// Measured inertia eigenvalues (descending).
    pub inertia_eig_measured: [f64; 3],
    /// 1LPT predicted inertia eigenvalues (descending).
    pub inertia_eig_1lpt: [f64; 3],
    /// 2LPT predicted inertia eigenvalues (descending).
    pub inertia_eig_2lpt: [f64; 3],
}

/// Predict density inertia eigenvalues from tidal eigenvalues at 1LPT.
///
/// Q_ii^pred = (L²/12)(1 − D₊λᵢ)² + Δσ²·L²/180
///
/// The first term is the Zel'dovich deformation of a uniform cell.
/// The second term is the variance from sub-cell density fluctuations
/// (white noise contribution to inertia at one level finer).
///
/// # Arguments
/// - `tidal_eig`: Doroshkevich eigenvalues [λ₁, λ₂, λ₃] (descending)
/// - `cell_size`: characteristic size L of the cell at this level
/// - `d_plus`: linear growth factor D₊
/// - `delta_s`: variance increment ΔS at this level
pub fn predict_inertia_1lpt(
    tidal_eig: &[f64; 3],
    cell_size: f64,
    d_plus: f64,
    delta_s: f64,
) -> [f64; 3] {
    let l2_12 = cell_size * cell_size / 12.0;
    let sub_cell_var = delta_s * cell_size * cell_size / 180.0;
    [
        l2_12 * (1.0 - d_plus * tidal_eig[0]).powi(2) + sub_cell_var,
        l2_12 * (1.0 - d_plus * tidal_eig[1]).powi(2) + sub_cell_var,
        l2_12 * (1.0 - d_plus * tidal_eig[2]).powi(2) + sub_cell_var,
    ]
}

/// Predict density inertia eigenvalues from tidal eigenvalues at 2LPT.
///
/// Includes the ε = -3/7 second-order correction:
/// Q_ii^pred = (L²/12)(1 − D₊λᵢ − ε D₊²(λᵢ² − λⱼλₖ))² + sub-cell
pub fn predict_inertia_2lpt(
    tidal_eig: &[f64; 3],
    cell_size: f64,
    d_plus: f64,
    delta_s: f64,
) -> [f64; 3] {
    let l2_12 = cell_size * cell_size / 12.0;
    let sub_cell_var = delta_s * cell_size * cell_size / 180.0;
    let eps = -3.0 / 7.0;
    let d2 = d_plus * d_plus;
    let [l1, l2, l3] = *tidal_eig;

    let deform = |li: f64, lj: f64, lk: f64| -> f64 {
        let correction = eps * d2 * (li * li - lj * lk);
        (1.0 - d_plus * li - correction).powi(2)
    };

    [
        l2_12 * deform(l1, l2, l3) + sub_cell_var,
        l2_12 * deform(l2, l1, l3) + sub_cell_var,
        l2_12 * deform(l3, l1, l2) + sub_cell_var,
    ]
}

/// Compute χ² between measured and predicted inertia eigenvalues.
///
/// χ² = Σᵢ (Q_meas_i − Q_pred_i)² / Var(Q_i)
///
/// The variance is estimated from the Doroshkevich distribution:
/// Var(Q_i) ≈ σ_Q² = (L⁴/144) · 2·ΔS · (1 − D₊λᵢ)²
/// (leading-order variance from the stochastic increment).
pub fn chi2_inertia(
    measured: &[f64; 3],
    predicted: &[f64; 3],
    cell_size: f64,
    d_plus: f64,
    tidal_eig: &[f64; 3],
    delta_s: f64,
) -> f64 {
    let l4_144 = cell_size.powi(4) / 144.0;
    let mut chi2 = 0.0;

    for i in 0..3 {
        let deform = (1.0 - d_plus * tidal_eig[i]).abs().max(0.01); // floor to avoid div/0
        let var_q = l4_144 * 2.0 * delta_s * deform * deform;
        if var_q > 1e-30 {
            let diff = measured[i] - predicted[i];
            chi2 += diff * diff / var_q;
        }
    }
    chi2
}

/// Compute LPT diagnostic for a single node given measured and tidal data.
pub fn compute_lpt_diagnostic(
    inertia_measured: &Sym3x3,
    tidal_eig: &[f64; 3],
    depth: usize,
    scale: f64,
    d_plus: f64,
    delta_s: f64,
) -> LptDiagnostic {
    let meas_eig = sym3x3_eigenvalues(*inertia_measured);
    let pred_1lpt = predict_inertia_1lpt(tidal_eig, scale, d_plus, delta_s);
    let pred_2lpt = predict_inertia_2lpt(tidal_eig, scale, d_plus, delta_s);

    let chi2_1 = chi2_inertia(&meas_eig, &pred_1lpt, scale, d_plus, tidal_eig, delta_s);
    let chi2_2 = chi2_inertia(&meas_eig, &pred_2lpt, scale, d_plus, tidal_eig, delta_s);

    let web_type = WebType::classify_1lpt(tidal_eig, d_plus);

    LptDiagnostic {
        depth,
        scale,
        chi2_1lpt: chi2_1,
        chi2_2lpt: chi2_2,
        web_type,
        inertia_eig_measured: meas_eig,
        inertia_eig_1lpt: pred_1lpt,
        inertia_eig_2lpt: pred_2lpt,
    }
}

/// Summary statistics of LPT diagnostics across all nodes at a given depth.
#[derive(Debug, Clone)]
pub struct LptDepthSummary {
    /// Tree depth.
    pub depth: usize,
    /// Characteristic scale.
    pub scale: f64,
    /// Mean χ² at 1LPT.
    pub mean_chi2_1lpt: f64,
    /// Mean χ² at 2LPT.
    pub mean_chi2_2lpt: f64,
    /// Fraction of nodes where 2LPT improves over 1LPT.
    pub frac_2lpt_better: f64,
    /// Number of nodes at this depth.
    pub n_nodes: usize,
    /// Web type fractions: [void, pancake, filament, halo].
    pub web_fractions: [f64; 4],
}

/// Aggregate LPT diagnostics by depth.
pub fn summarize_by_depth(diagnostics: &[LptDiagnostic]) -> Vec<LptDepthSummary> {
    if diagnostics.is_empty() {
        return vec![];
    }

    let max_depth = diagnostics.iter().map(|d| d.depth).max().unwrap_or(0);
    let mut summaries = Vec::new();

    for depth in 0..=max_depth {
        let at_depth: Vec<&LptDiagnostic> = diagnostics.iter()
            .filter(|d| d.depth == depth)
            .collect();

        if at_depth.is_empty() {
            continue;
        }

        let n = at_depth.len();
        let mean_chi2_1 = at_depth.iter().map(|d| d.chi2_1lpt).sum::<f64>() / n as f64;
        let mean_chi2_2 = at_depth.iter().map(|d| d.chi2_2lpt).sum::<f64>() / n as f64;
        let n_2lpt_better = at_depth.iter().filter(|d| d.chi2_2lpt < d.chi2_1lpt).count();

        let mut web_counts = [0usize; 4];
        for d in &at_depth {
            match d.web_type {
                WebType::Void => web_counts[0] += 1,
                WebType::Pancake => web_counts[1] += 1,
                WebType::Filament => web_counts[2] += 1,
                WebType::Halo => web_counts[3] += 1,
            }
        }

        summaries.push(LptDepthSummary {
            depth,
            scale: at_depth[0].scale,
            mean_chi2_1lpt: mean_chi2_1,
            mean_chi2_2lpt: mean_chi2_2,
            frac_2lpt_better: n_2lpt_better as f64 / n as f64,
            n_nodes: n,
            web_fractions: [
                web_counts[0] as f64 / n as f64,
                web_counts[1] as f64 / n as f64,
                web_counts[2] as f64 / n as f64,
                web_counts[3] as f64 / n as f64,
            ],
        });
    }

    summaries
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predict_inertia_no_deformation() {
        // λ = 0 → no deformation → inertia = L²/12 + sub-cell
        let pred = predict_inertia_1lpt(&[0.0, 0.0, 0.0], 10.0, 1.0, 0.0);
        let expected = 100.0 / 12.0;
        for &p in &pred {
            assert!((p - expected).abs() < 1e-10, "got {p}, expected {expected}");
        }
    }

    #[test]
    fn predict_inertia_full_collapse() {
        // D₊λ = 1 → fully collapsed → inertia = sub-cell only
        let pred = predict_inertia_1lpt(&[1.0, 1.0, 1.0], 10.0, 1.0, 0.1);
        let sub_cell = 0.1 * 100.0 / 180.0;
        for &p in &pred {
            assert!((p - sub_cell).abs() < 1e-10);
        }
    }

    #[test]
    fn chi2_zero_when_perfect() {
        let eig = [0.5, 0.3, 0.1];
        let pred = predict_inertia_1lpt(&eig, 10.0, 1.0, 0.1);
        let chi2 = chi2_inertia(&pred, &pred, 10.0, 1.0, &eig, 0.1);
        assert!(chi2 < 1e-20, "chi2 = {chi2}");
    }

    #[test]
    fn _2lpt_correction_nonzero() {
        let eig = [0.8, 0.5, 0.3];
        let pred_1 = predict_inertia_1lpt(&eig, 10.0, 1.0, 0.0);
        let pred_2 = predict_inertia_2lpt(&eig, 10.0, 1.0, 0.0);
        // 2LPT should differ from 1LPT
        let diff: f64 = (0..3).map(|i| (pred_1[i] - pred_2[i]).abs()).sum();
        assert!(diff > 1e-6, "1LPT and 2LPT predictions are identical");
    }

    #[test]
    fn diagnostic_struct() {
        let inertia = [5.0, 3.0, 1.0, 0.0, 0.0, 0.0]; // diagonal
        let tidal_eig = [0.3, 0.1, -0.1];
        let diag = compute_lpt_diagnostic(&inertia, &tidal_eig, 3, 50.0, 1.0, 0.1);
        assert_eq!(diag.depth, 3);
        assert!((diag.scale - 50.0).abs() < 1e-14);
        assert!(diag.chi2_1lpt >= 0.0);
        assert!(diag.chi2_2lpt >= 0.0);
    }

    #[test]
    fn summarize_by_depth_basic() {
        let diagnostics = vec![
            LptDiagnostic {
                depth: 0, scale: 100.0, chi2_1lpt: 2.0, chi2_2lpt: 1.5,
                web_type: WebType::Void,
                inertia_eig_measured: [5.0, 3.0, 1.0],
                inertia_eig_1lpt: [4.0, 3.0, 1.5],
                inertia_eig_2lpt: [4.5, 3.0, 1.2],
            },
            LptDiagnostic {
                depth: 0, scale: 100.0, chi2_1lpt: 3.0, chi2_2lpt: 2.0,
                web_type: WebType::Pancake,
                inertia_eig_measured: [6.0, 4.0, 2.0],
                inertia_eig_1lpt: [5.0, 4.0, 2.5],
                inertia_eig_2lpt: [5.5, 4.0, 2.2],
            },
        ];
        let summaries = summarize_by_depth(&diagnostics);
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].n_nodes, 2);
        assert!((summaries[0].mean_chi2_1lpt - 2.5).abs() < 1e-10);
        assert!((summaries[0].frac_2lpt_better - 1.0).abs() < 1e-10); // both 2LPT better
    }
}

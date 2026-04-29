//! Mark-weight construction.
//!
//! The Rust core consumes pre-computed effective per-galaxy weights:
//!
//! ```text
//! w_eff_i = w_i * phi(m_i)
//! ```
//!
//! This module provides convenience builders for the four common mark modes
//! (none, multiplicative numeric, threshold, quantile bin). High-level
//! construction logic lives in Python; these helpers are exposed primarily
//! for use from Rust binaries / tests.

#[derive(Clone, Debug)]
pub enum MarkMode<'a> {
    /// All weights are 1 (or the user-supplied baseline).
    None,
    /// Multiplicative numeric weight phi(m_i).
    Multiplicative(&'a [f64]),
    /// Threshold: w_eff_i = w_i if m_i > cut else 0.
    Threshold { marks: &'a [f64], cut: f64 },
    /// Quantile bin: w_eff_i = w_i if rank(m_i) in [q_lo, q_hi] else 0,
    /// where rank is the empirical rank in [0, 1].
    QuantileBin {
        marks: &'a [f64],
        q_lo: f64,
        q_hi: f64,
    },
}

/// Apply the requested mark mode to a baseline weight vector and return a
/// fresh effective-weight vector. Lengths must match.
pub fn apply_mark(baseline: &[f64], mode: &MarkMode<'_>) -> Vec<f64> {
    match mode {
        MarkMode::None => baseline.to_vec(),
        MarkMode::Multiplicative(phi) => {
            assert_eq!(baseline.len(), phi.len(), "mark length mismatch");
            baseline.iter().zip(phi.iter()).map(|(a, b)| a * b).collect()
        }
        MarkMode::Threshold { marks, cut } => {
            assert_eq!(baseline.len(), marks.len(), "mark length mismatch");
            baseline
                .iter()
                .zip(marks.iter())
                .map(|(b, m)| if *m > *cut { *b } else { 0.0 })
                .collect()
        }
        MarkMode::QuantileBin { marks, q_lo, q_hi } => {
            assert_eq!(baseline.len(), marks.len(), "mark length mismatch");
            let mut order: Vec<u32> = (0..marks.len() as u32).collect();
            order.sort_unstable_by(|&a, &b| {
                marks[a as usize]
                    .partial_cmp(&marks[b as usize])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let n = marks.len();
            let lo_i = ((*q_lo).clamp(0.0, 1.0) * n as f64).floor() as usize;
            let hi_i = ((*q_hi).clamp(0.0, 1.0) * n as f64).ceil() as usize;
            let mut keep = vec![false; n];
            for &j in &order[lo_i..hi_i.min(n)] {
                keep[j as usize] = true;
            }
            baseline
                .iter()
                .zip(keep.iter())
                .map(|(b, &k)| if k { *b } else { 0.0 })
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn none_returns_baseline() {
        let b = vec![1.0, 2.0, 3.0];
        let got = apply_mark(&b, &MarkMode::None);
        assert_eq!(got, b);
    }

    #[test]
    fn multiplicative_multiplies() {
        let b = vec![1.0, 2.0, 3.0];
        let m = vec![0.5, 1.0, 2.0];
        let got = apply_mark(&b, &MarkMode::Multiplicative(&m));
        for (g, w) in got.iter().zip([0.5, 2.0, 6.0].iter()) {
            assert_abs_diff_eq!(*g, *w, epsilon = 1e-12);
        }
    }

    #[test]
    fn threshold_zeroes_below() {
        let b = vec![1.0; 5];
        let m = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let got = apply_mark(&b, &MarkMode::Threshold { marks: &m, cut: 1.5 });
        assert_eq!(got, vec![0.0, 0.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn quantile_bin_selects_interval() {
        let b = vec![1.0; 10];
        let m: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let got = apply_mark(
            &b,
            &MarkMode::QuantileBin {
                marks: &m,
                q_lo: 0.5,
                q_hi: 0.8,
            },
        );
        // top 50% to 80% by mark value -> indices 5..8.
        let want = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
        assert_eq!(got, want);
    }
}

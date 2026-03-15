//! Dilution ladder for multi-scale ξ(r) estimation.
//!
//! The dilution ladder extends kNN measurements to large scales by creating
//! a geometric sequence of random, volume-filling subsamples. At each level ℓ,
//! the data are randomly partitioned into R_ℓ = 8^ℓ disjoint subsets.
//!
//! The overlap between adjacent levels provides consistency checks and
//! the scatter across subsamples gives empirical variance estimates.

// TODO: re-enable when ladder implementation uses XiEstimate
// use crate::estimator::XiEstimate;
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// A single dilution level.
#[derive(Debug)]
pub struct DilutionLevel {
    /// Dilution factor R_ℓ = 8^ℓ
    pub dilution_factor: usize,
    /// Level index ℓ
    pub level: usize,
    /// Indices into the original data array for each subsample
    pub subsamples: Vec<Vec<usize>>,
}

/// The full dilution ladder.
pub struct DilutionLadder {
    pub levels: Vec<DilutionLevel>,
}

impl DilutionLadder {
    /// Build a dilution ladder from n_data points up to the given max level.
    ///
    /// Level 0: full data (1 subset)
    /// Level ℓ: 8^ℓ disjoint, volume-filling subsets of n_data / 8^ℓ points each
    pub fn build(n_data: usize, max_level: usize, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut indices: Vec<usize> = (0..n_data).collect();
        indices.shuffle(&mut rng);

        let mut levels = Vec::with_capacity(max_level + 1);

        for level in 0..=max_level {
            let r_ell = 8usize.pow(level as u32);
            if r_ell > n_data {
                break;
            }
            let subset_size = n_data / r_ell;
            if subset_size == 0 {
                break;
            }

            let subsamples: Vec<Vec<usize>> = (0..r_ell)
                .map(|m| {
                    let start = m * subset_size;
                    let end = start + subset_size;
                    indices[start..end].to_vec()
                })
                .collect();

            levels.push(DilutionLevel {
                dilution_factor: r_ell,
                level,
                subsamples,
            });
        }

        DilutionLadder { levels }
    }

    /// Characteristic scale probed at level ℓ with k-th neighbor.
    /// r_char(k, R_ℓ) ~ (k · R_ℓ / n̄)^{1/3}
    pub fn r_char(k: usize, dilution_factor: usize, nbar: f64) -> f64 {
        (k as f64 * dilution_factor as f64 / nbar).cbrt()
    }
}

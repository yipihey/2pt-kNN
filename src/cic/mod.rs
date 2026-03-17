//! Counts-in-cells statistics from Morton octree grid histograms.
//!
//! Computes the count distribution P(N=k), cumulative P(N≥k), and
//! moments (mean, variance, skewness, kurtosis) of the data counts
//! in unmasked cells at each octree level.

use crate::grid::CellHistogram;
use crate::morton::MortonConfig;

/// Counts-in-cells distribution at one octree level.
#[derive(Debug, Clone)]
pub struct CicDistribution {
    pub level: u32,
    /// Physical volume of each cell at this level.
    pub cell_volume: f64,
    /// Effective sphere radius: r = (3V / 4π)^{1/3}.
    pub effective_radius: f64,
    /// P(N = k) for k = 0, 1, 2, ...
    pub p_n: Vec<f64>,
    /// Cumulative P(N ≥ k) for k = 0, 1, 2, ...
    pub p_geq: Vec<f64>,
    /// Mean count in unmasked cells.
    pub mean_n: f64,
    /// Variance of counts in unmasked cells.
    pub var_n: f64,
    /// Skewness (3rd standardized moment).
    pub skewness: f64,
    /// Excess kurtosis (4th standardized moment − 3).
    pub kurtosis: f64,
}

/// Summary of counts-in-cells across all octree levels.
#[derive(Debug, Clone)]
pub struct CicSummary {
    pub levels: Vec<CicDistribution>,
}

/// Compute the counts-in-cells distribution at one level.
///
/// Only cells covered by the random catalog (n_R > 0) are counted
/// as "unmasked". Empty cells outside the histogram but inside the
/// total grid are not included (they are not covered by randoms and
/// thus outside the survey footprint for a survey geometry; for a
/// periodic box the randoms should cover all occupied cells).
pub fn compute_cic(hist: &CellHistogram, config: &MortonConfig) -> CicDistribution {
    let cell_side = config.box_size / (1u64 << hist.level) as f64;
    let cell_volume = cell_side.powi(3);
    let effective_radius = (3.0 * cell_volume / (4.0 * std::f64::consts::PI)).cbrt();

    // Collect data counts in unmasked cells.
    let mut counts: Vec<u32> = Vec::new();
    for (i, &nr) in hist.n_random.iter().enumerate() {
        if nr > 0 {
            counts.push(hist.n_data[i]);
        }
    }

    let n_cells = counts.len();
    if n_cells == 0 {
        return CicDistribution {
            level: hist.level,
            cell_volume,
            effective_radius,
            p_n: vec![1.0],
            p_geq: vec![1.0],
            mean_n: 0.0,
            var_n: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
        };
    }

    let max_count = *counts.iter().max().unwrap() as usize;

    // Build P(N = k).
    let mut freq = vec![0usize; max_count + 1];
    for &c in &counts {
        freq[c as usize] += 1;
    }
    let p_n: Vec<f64> = freq.iter().map(|&f| f as f64 / n_cells as f64).collect();

    // Build P(N ≥ k).
    let mut p_geq = vec![0.0; max_count + 2];
    p_geq[max_count + 1] = 0.0;
    for k in (0..=max_count).rev() {
        p_geq[k] = p_geq[k + 1] + p_n[k];
    }
    p_geq.truncate(max_count + 1);

    // Moments.
    let mean_n = counts.iter().map(|&c| c as f64).sum::<f64>() / n_cells as f64;
    let var_n = counts
        .iter()
        .map(|&c| {
            let d = c as f64 - mean_n;
            d * d
        })
        .sum::<f64>()
        / n_cells as f64;

    let std_n = var_n.sqrt();
    let (skewness, kurtosis) = if std_n > 0.0 {
        let m3 = counts
            .iter()
            .map(|&c| {
                let d = (c as f64 - mean_n) / std_n;
                d * d * d
            })
            .sum::<f64>()
            / n_cells as f64;
        let m4 = counts
            .iter()
            .map(|&c| {
                let d = (c as f64 - mean_n) / std_n;
                d * d * d * d
            })
            .sum::<f64>()
            / n_cells as f64;
        (m3, m4 - 3.0)
    } else {
        (0.0, 0.0)
    };

    CicDistribution {
        level: hist.level,
        cell_volume,
        effective_radius,
        p_n,
        p_geq,
        mean_n,
        var_n,
        skewness,
        kurtosis,
    }
}

/// Compute counts-in-cells for all levels.
pub fn compute_cic_all_levels(
    histograms: &[CellHistogram],
    config: &MortonConfig,
) -> CicSummary {
    let levels = histograms
        .iter()
        .map(|hist| compute_cic(hist, config))
        .collect();
    CicSummary { levels }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid;
    use crate::morton::{self, MortonConfig};
    use rand::SeedableRng;
    use rand::Rng;

    #[test]
    fn cic_poisson_mean() {
        // Uniform random data in a periodic box.
        let box_size = 100.0;
        let n_data = 10_000;
        let n_random = 50_000;
        let config = MortonConfig::new(box_size, true);

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let data: Vec<[f64; 3]> = (0..n_data)
            .map(|_| {
                [
                    rng.gen::<f64>() * box_size,
                    rng.gen::<f64>() * box_size,
                    rng.gen::<f64>() * box_size,
                ]
            })
            .collect();
        let randoms: Vec<[f64; 3]> = (0..n_random)
            .map(|_| {
                [
                    rng.gen::<f64>() * box_size,
                    rng.gen::<f64>() * box_size,
                    rng.gen::<f64>() * box_size,
                ]
            })
            .collect();

        let particles = morton::prepare_particles(&data, &randoms, &config);
        let hist = grid::build_cell_histogram(&particles, 3, config.bits_per_axis);
        let cic = compute_cic(&hist, &config);

        // Mean count should be roughly n_data / n_cells_covered.
        // At level 3, 8^3 = 512 cells. With 10k data particles, mean ≈ 19.5.
        assert!(cic.mean_n > 5.0, "mean too low: {}", cic.mean_n);
        assert!(cic.mean_n < 100.0, "mean too high: {}", cic.mean_n);

        // P(N=0) + P(N=1) + ... should sum to 1.
        let total_p: f64 = cic.p_n.iter().sum();
        assert!(
            (total_p - 1.0).abs() < 1e-10,
            "P(N) doesn't sum to 1: {}",
            total_p
        );

        // P(N >= 0) should be 1.
        assert!((cic.p_geq[0] - 1.0).abs() < 1e-10);

        // For Poisson, variance ≈ mean. With clustering, var > mean.
        // For uniform random, var should be close to mean.
        let ratio = cic.var_n / cic.mean_n;
        assert!(
            ratio > 0.5 && ratio < 2.0,
            "var/mean = {:.2}, expected ~1 for Poisson",
            ratio
        );
    }

    #[test]
    fn cic_effective_radius() {
        let config = MortonConfig::new(100.0, true);
        let cell_side: f64 = 100.0 / 8.0; // level 3
        let expected_r = (3.0 * cell_side.powi(3) / (4.0 * std::f64::consts::PI)).cbrt();

        // Create minimal histogram just to test radius calculation.
        let hist = CellHistogram {
            level: 3,
            cell_indices: vec![0],
            w_data: vec![1.0],
            w_random: vec![1.0],
            n_data: vec![1],
            n_random: vec![1],
        };
        let cic = compute_cic(&hist, &config);
        assert!(
            (cic.effective_radius - expected_r).abs() < 1e-10,
            "effective_radius = {}, expected {}",
            cic.effective_radius,
            expected_r
        );
    }
}

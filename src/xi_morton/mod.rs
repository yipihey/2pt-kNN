//! Two-point correlation function estimation via Morton-grid neighbor lags.
//!
//! At each octree level ℓ, the estimator computes ξ̂ from cell-pair
//! correlations at the 13 unique nearest-neighbor lag vectors, giving
//! 3 separation bins per level (face, edge, corner). Multi-level
//! analysis yields logarithmically spaced radial bins.

pub mod cell_pairs;

use crate::grid;
use crate::grid::OverdensityField;
use crate::morton::{self, MortonConfig, MortonParticle, NEIGHBOR_LAGS};

// TODO: rayon parallelism over cells within each lag
// #[cfg(feature = "parallel")]
// use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Kahan compensated summation
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct KahanAccumulator {
    sum: f64,
    comp: f64,
}

impl KahanAccumulator {
    fn new() -> Self {
        Self {
            sum: 0.0,
            comp: 0.0,
        }
    }

    #[inline]
    fn add(&mut self, val: f64) {
        let y = val - self.comp;
        let t = self.sum + y;
        self.comp = (t - self.sum) - y;
        self.sum = t;
    }

    fn value(&self) -> f64 {
        self.sum
    }

    fn merge(&mut self, other: &Self) {
        self.add(other.sum);
        // Also carry over remaining compensation
        self.add(-other.comp);
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// ξ̂ result at one octree level, with separate values for each lag.
#[derive(Debug, Clone)]
pub struct MortonXiLevel {
    pub level: u32,
    /// Physical separation for each lag (13 values).
    pub r: Vec<f64>,
    /// cos(θ) to the z-axis for each lag (for future anisotropic binning).
    pub mu: Vec<f64>,
    /// Estimated ξ at each lag.
    pub xi: Vec<f64>,
    /// Number of valid cell pairs contributing to each lag.
    pub n_pairs: Vec<u64>,
}

/// Binned ξ̂ result at one level: 3 bins (face, edge, corner).
#[derive(Debug, Clone)]
pub struct MortonXiBinned {
    pub level: u32,
    /// Physical separation for each bin (3 values: h, h√2, h√3).
    pub r: Vec<f64>,
    /// Estimated ξ at each bin (averaged over lags at same |r|).
    pub xi: Vec<f64>,
    /// Total number of valid cell pairs per bin.
    pub n_pairs: Vec<u64>,
}

/// Combined multi-resolution ξ̂ result.
#[derive(Debug, Clone)]
pub struct MortonXiEstimate {
    /// Per-level binned results (grid-based LS).
    pub levels: Vec<MortonXiBinned>,
    /// Fine-scale ξ from direct pair counting (if hybrid mode used).
    pub cell_pair_xi: Option<cell_pairs::CellPairXi>,
    /// Merged (r, ξ) sorted by r.
    pub r: Vec<f64>,
    pub xi: Vec<f64>,
}

/// Configuration for the Morton ξ estimator.
#[derive(Debug, Clone)]
pub struct MortonXiConfig {
    pub morton_config: MortonConfig,
    /// Minimum octree level (coarsest grid).
    pub l_min: u32,
    /// Maximum octree level (finest grid).
    pub l_max: u32,
    /// Number of random spatial offsets for sub-cell interpolation (0 = none).
    pub n_offsets: usize,
    /// Random seed for offset generation.
    pub seed: u64,
}

impl MortonXiConfig {
    /// Reasonable defaults for a periodic box.
    pub fn new(box_size: f64, l_max: u32) -> Self {
        Self {
            morton_config: MortonConfig::new(box_size, true),
            l_min: 1,
            l_max,
            n_offsets: 8,
            seed: 42,
        }
    }

    /// Compute the maximum useful octree level given a particle count.
    ///
    /// Returns the finest level where the mean number of data particles
    /// per cell is at least `min_per_cell`. The count-based Landy-Szalay
    /// estimator works well down to ~0.2 particles/cell (use min_per_cell ≈ 1
    /// for a conservative choice, or ≈ 0.2 to push the resolution limit).
    pub fn auto_l_max(_box_size: f64, n_data: usize, min_per_cell: f64) -> u32 {
        let threshold = (n_data as f64 / min_per_cell) as u64;
        let mut l = 1u32;
        loop {
            // n_cells at level l+1 = 8^(l+1) — exact integer power of 8.
            let n_cells_next = 1u64 << (3 * (l + 1));
            if n_cells_next > threshold {
                return l;
            }
            l += 1;
            if l >= 20 {
                return l;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Core estimator
// ---------------------------------------------------------------------------

/// Compute ξ̂ at a single octree level from the overdensity field.
pub fn compute_xi_level(field: &OverdensityField, config: &MortonConfig) -> MortonXiLevel {
    let h = field.cell_size;
    let n_lags = NEIGHBOR_LAGS.len();

    let mut r_vals = Vec::with_capacity(n_lags);
    let mut mu_vals = Vec::with_capacity(n_lags);
    let mut xi_vals = Vec::with_capacity(n_lags);
    let mut n_pairs_vals = Vec::with_capacity(n_lags);

    for &(dx, dy, dz) in &NEIGHBOR_LAGS {
        let dist = h * ((dx * dx + dy * dy + dz * dz) as f64).sqrt();
        let mu = if dist > 0.0 {
            (dz as f64 * h) / dist
        } else {
            0.0
        };
        r_vals.push(dist);
        mu_vals.push(mu);

        let (xi, np) = correlate_lag(field, dx, dy, dz, config);
        xi_vals.push(xi);
        n_pairs_vals.push(np);
    }

    MortonXiLevel {
        level: field.level,
        r: r_vals,
        mu: mu_vals,
        xi: xi_vals,
        n_pairs: n_pairs_vals,
    }
}

/// Compute the correlation for a single lag vector via count-based
/// Landy-Szalay.
///
/// The inner loop accumulates integer-valued products DD, DR, RR
/// (exact in f64 for unit weights up to 2^53). The final formula
/// uses total counts N_D, N_R to avoid intermediate α divisions:
///
///   ξ = (DD·N_R² − 2·DR·N_D·N_R + RR·N_D²) / (RR·N_D²)
///
/// Only one float division at the end; the numerator stays in
/// integer-scaled arithmetic as long as possible.
fn correlate_lag(
    field: &OverdensityField,
    dx: i32,
    dy: i32,
    dz: i32,
    config: &MortonConfig,
) -> (f64, u64) {
    let mut dd = KahanAccumulator::new();
    let mut dr = KahanAccumulator::new();
    let mut rr = KahanAccumulator::new();
    let mut n_pairs = 0u64;

    for (i, &ci) in field.cell_indices.iter().enumerate() {
        if !field.valid[i] {
            continue;
        }

        if let Some(nbr_ci) = morton::neighbor_cell(ci, dx, dy, dz, field.level, config.periodic) {
            if let Some(&j) = field.cell_map.get(&nbr_ci) {
                if field.valid[j] {
                    dd.add(field.w_data[i] * field.w_data[j]);
                    dr.add(field.w_data[i] * field.w_random[j]
                         + field.w_random[i] * field.w_data[j]);
                    rr.add(field.w_random[i] * field.w_random[j]);
                    n_pairs += 1;
                }
            }
        }
    }

    let rr_val = rr.value();
    let nd = field.total_wd;
    let nr = field.total_wr;
    let xi = if rr_val > 0.0 {
        let denom = rr_val * nd * nd;
        // DR is already symmetrized (D_i·R_j + R_i·D_j), so no factor of 2.
        (dd.value() * nr * nr - dr.value() * nd * nr + rr_val * nd * nd) / denom
    } else {
        0.0
    };

    (xi, n_pairs)
}

/// Bin the 13 per-lag ξ values into 3 separation bins (face, edge, corner).
pub fn bin_xi_level(level_result: &MortonXiLevel) -> MortonXiBinned {
    let h = if !level_result.r.is_empty() {
        // Face lags have distance h; extract it.
        level_result.r[0]
    } else {
        0.0
    };

    // Group: face (indices 0-2), edge (3-8), corner (9-12)
    let groups: &[&[usize]] = &[&[0, 1, 2], &[3, 4, 5, 6, 7, 8], &[9, 10, 11, 12]];
    let multipliers = [1.0, std::f64::consts::SQRT_2, 3.0_f64.sqrt()];

    let mut r = Vec::with_capacity(3);
    let mut xi = Vec::with_capacity(3);
    let mut n_pairs = Vec::with_capacity(3);

    for (g, &mult) in groups.iter().zip(multipliers.iter()) {
        r.push(h * mult);

        let total_np: u64 = g.iter().map(|&i| level_result.n_pairs[i]).sum();
        if total_np > 0 {
            // Weighted average by number of pairs.
            let sum_xi: f64 = g
                .iter()
                .map(|&i| level_result.xi[i] * level_result.n_pairs[i] as f64)
                .sum();
            xi.push(sum_xi / total_np as f64);
        } else {
            xi.push(0.0);
        }
        n_pairs.push(total_np);
    }

    MortonXiBinned {
        level: level_result.level,
        r,
        xi,
        n_pairs,
    }
}

/// Run the full multi-level Morton ξ estimator.
///
/// Takes pre-sorted particles (from `morton::prepare_particles`).
pub fn estimate_xi(
    sorted_particles: &[MortonParticle],
    config: &MortonXiConfig,
) -> MortonXiEstimate {
    let mc = &config.morton_config;

    let mut levels = Vec::new();

    for level in config.l_min..=config.l_max {
        let hist = grid::build_cell_histogram(sorted_particles, level, mc.bits_per_axis);
        let field = grid::compute_overdensity(&hist, mc);
        let xi_level = compute_xi_level(&field, mc);
        let binned = bin_xi_level(&xi_level);
        levels.push(binned);
    }

    // Merge into a single (r, ξ) array sorted by r.
    let mut merged: Vec<(f64, f64)> = levels
        .iter()
        .flat_map(|lev| lev.r.iter().copied().zip(lev.xi.iter().copied()))
        .filter(|(r, _)| *r > 0.0)
        .collect();
    merged.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let r = merged.iter().map(|(r, _)| *r).collect();
    let xi = merged.iter().map(|(_, x)| *x).collect();

    MortonXiEstimate { levels, cell_pair_xi: None, r, xi }
}

/// Run the Morton ξ estimator with random spatial offsets for sub-cell
/// interpolation. Averages over `n_offsets` runs with different shifts.
pub fn estimate_xi_with_offsets(
    data: &[[f64; 3]],
    randoms: &[[f64; 3]],
    config: &MortonXiConfig,
) -> MortonXiEstimate {
    use rand::SeedableRng;
    use rand::Rng;

    if config.n_offsets == 0 {
        let particles = morton::prepare_particles(data, randoms, &config.morton_config);
        return estimate_xi(&particles, config);
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);
    let box_size = config.morton_config.box_size;
    let finest_cell = box_size / (1u64 << config.l_max) as f64;

    let mut all_estimates: Vec<MortonXiEstimate> = Vec::with_capacity(config.n_offsets);

    for _ in 0..config.n_offsets {
        let offset = [
            rng.gen::<f64>() * finest_cell,
            rng.gen::<f64>() * finest_cell,
            rng.gen::<f64>() * finest_cell,
        ];

        let shifted_data: Vec<[f64; 3]> = data
            .iter()
            .map(|p| wrap_position(p, &offset, box_size))
            .collect();
        let shifted_randoms: Vec<[f64; 3]> = randoms
            .iter()
            .map(|p| wrap_position(p, &offset, box_size))
            .collect();

        let particles =
            morton::prepare_particles(&shifted_data, &shifted_randoms, &config.morton_config);
        all_estimates.push(estimate_xi(&particles, config));
    }

    // Average across offsets: merge by matching (level, bin_index).
    average_estimates(&all_estimates)
}

/// Hybrid estimator: grid-based LS at coarse levels, direct pair counting
/// at fine scales below the grid resolution limit.
///
/// `l_grid_max`: finest level for grid-based LS (auto-detected if 0).
/// `l_pair`: octree level at which to group cells for pair counting.
///   Typically `l_grid_max + 1` or `l_grid_max + 2`.
/// `n_pair_bins`: number of logarithmic radial bins for fine-scale ξ.
/// `r_pair_min`: minimum separation for pair counting (e.g. 1.0 Mpc/h).
pub fn estimate_xi_hybrid(
    data: &[[f64; 3]],
    randoms: &[[f64; 3]],
    config: &MortonXiConfig,
    l_pair: u32,
    n_pair_bins: usize,
    r_pair_min: f64,
) -> MortonXiEstimate {
    // 1. Grid-based LS at coarse levels (with offsets).
    let mut grid_est = if config.n_offsets > 0 {
        estimate_xi_with_offsets(data, randoms, config)
    } else {
        let particles = morton::prepare_particles(data, randoms, &config.morton_config);
        estimate_xi(&particles, config)
    };

    // 2. Direct pair counting at fine scale.
    let mc = &config.morton_config;
    let _cell_size = mc.box_size / (1u64 << l_pair) as f64;
    // r_max for pairs: the diagonal of a neighbor cell, h * sqrt(3).
    // Actually we want to cover at least to where the grid LS starts.
    // The coarsest grid LS bin is at h_grid * 1.0 (face neighbor).
    // So r_pair_max should overlap with the finest grid level.
    let h_grid = mc.box_size / (1u64 << config.l_max) as f64;
    let r_pair_max = h_grid * 2.0_f64.sqrt(); // overlap into edge-neighbor scale

    if r_pair_max <= r_pair_min {
        // No room for pair counting — return grid-only result.
        return grid_est;
    }

    let r_edges = cell_pairs::log_bin_edges(r_pair_min, r_pair_max, n_pair_bins);
    let particles = morton::prepare_particles(data, randoms, mc);
    let counts = cell_pairs::count_cell_pairs(
        &particles,
        data,
        randoms,
        l_pair,
        &r_edges,
        mc,
    );
    let fine_xi = cell_pairs::landy_szalay(&counts);

    // 3. Merge: grid levels + fine-scale pair counting.
    // The fine-scale bins go into the merged (r, xi) array.
    let mut merged: Vec<(f64, f64)> = grid_est
        .levels
        .iter()
        .flat_map(|lev| lev.r.iter().copied().zip(lev.xi.iter().copied()))
        .filter(|(r, _)| *r > 0.0)
        .collect();

    for (r, xi) in fine_xi.r.iter().zip(fine_xi.xi.iter()) {
        merged.push((*r, *xi));
    }

    merged.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    grid_est.r = merged.iter().map(|(r, _)| *r).collect();
    grid_est.xi = merged.iter().map(|(_, x)| *x).collect();
    grid_est.cell_pair_xi = Some(fine_xi);

    grid_est
}

/// Shift a position by an offset and wrap periodically.
fn wrap_position(pos: &[f64; 3], offset: &[f64; 3], box_size: f64) -> [f64; 3] {
    [
        (pos[0] + offset[0]).rem_euclid(box_size),
        (pos[1] + offset[1]).rem_euclid(box_size),
        (pos[2] + offset[2]).rem_euclid(box_size),
    ]
}

/// Average multiple MortonXiEstimates (from different offsets).
fn average_estimates(estimates: &[MortonXiEstimate]) -> MortonXiEstimate {
    if estimates.is_empty() {
        return MortonXiEstimate {
            levels: Vec::new(),
            cell_pair_xi: None,
            r: Vec::new(),
            xi: Vec::new(),
        };
    }

    let n = estimates.len() as f64;
    let n_levels = estimates[0].levels.len();

    let mut levels = Vec::with_capacity(n_levels);
    for li in 0..n_levels {
        let ref_level = &estimates[0].levels[li];
        let n_bins = ref_level.r.len();
        let mut avg_xi = vec![0.0; n_bins];
        let mut total_pairs = vec![0u64; n_bins];

        for est in estimates {
            for bi in 0..n_bins {
                avg_xi[bi] += est.levels[li].xi[bi];
                total_pairs[bi] += est.levels[li].n_pairs[bi];
            }
        }

        for bi in 0..n_bins {
            avg_xi[bi] /= n;
        }

        levels.push(MortonXiBinned {
            level: ref_level.level,
            r: ref_level.r.clone(),
            xi: avg_xi,
            n_pairs: total_pairs,
        });
    }

    // Rebuild merged arrays.
    let mut merged: Vec<(f64, f64)> = levels
        .iter()
        .flat_map(|lev| lev.r.iter().copied().zip(lev.xi.iter().copied()))
        .filter(|(r, _)| *r > 0.0)
        .collect();
    merged.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let r = merged.iter().map(|(r, _)| *r).collect();
    let xi = merged.iter().map(|(_, x)| *x).collect();

    MortonXiEstimate { levels, cell_pair_xi: None, r, xi }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morton;
    use rand::SeedableRng;
    use rand::Rng;

    /// Generate uniform random points in a periodic box.
    fn uniform_random(n: usize, box_size: f64, seed: u64) -> Vec<[f64; 3]> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                [
                    rng.gen::<f64>() * box_size,
                    rng.gen::<f64>() * box_size,
                    rng.gen::<f64>() * box_size,
                ]
            })
            .collect()
    }

    #[test]
    fn poisson_xi_near_zero() {
        // Uniform random catalog → ξ should be consistent with 0.
        let box_size = 100.0;
        let n = 50_000;
        let data = uniform_random(n, box_size, 1);
        let randoms = uniform_random(n * 5, box_size, 2);

        let config = MortonXiConfig::new(box_size, 5);
        let particles = morton::prepare_particles(&data, &randoms, &config.morton_config);
        let result = estimate_xi(&particles, &config);

        // ξ should be close to 0 for all separations.
        for (r, xi) in result.r.iter().zip(result.xi.iter()) {
            assert!(
                xi.abs() < 0.1,
                "ξ({:.1}) = {:.4}, expected ~0 for Poisson",
                r,
                xi
            );
        }
    }

    #[test]
    fn xi_level_structure() {
        let box_size = 100.0;
        let data = uniform_random(10_000, box_size, 10);
        let randoms = uniform_random(50_000, box_size, 20);

        let config = MortonXiConfig::new(box_size, 3);
        let particles = morton::prepare_particles(&data, &randoms, &config.morton_config);
        let result = estimate_xi(&particles, &config);

        // Should have 3 levels (1, 2, 3), each with 3 bins.
        assert_eq!(result.levels.len(), 3);
        for lev in &result.levels {
            assert_eq!(lev.r.len(), 3);
            assert_eq!(lev.xi.len(), 3);
            // All pairs should be non-zero (cells exist at each level).
            for &np in &lev.n_pairs {
                assert!(np > 0, "expected non-zero pairs at level {}", lev.level);
            }
        }

        // Merged r should be sorted.
        for w in result.r.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }
}

//! Cell histograms and overdensity fields on the Morton octree grid.
//!
//! Given Morton-sorted particles, builds per-cell weight accumulators at
//! any octree level in O(N) time by scanning contiguous runs.

use std::collections::HashMap;

use crate::morton::{self, CatalogFlag, MortonConfig, MortonParticle};

/// Accumulated counts per cell at a given octree level.
#[derive(Debug, Clone)]
pub struct CellHistogram {
    pub level: u32,
    /// Cell Morton indices (sorted).
    pub cell_indices: Vec<u64>,
    /// Sum of data-catalog weights per cell.
    pub w_data: Vec<f64>,
    /// Sum of random-catalog weights per cell.
    pub w_random: Vec<f64>,
    /// Count of data particles per cell.
    pub n_data: Vec<u32>,
    /// Count of random particles per cell.
    pub n_random: Vec<u32>,
}

/// Overdensity field δ_c at one octree level.
#[derive(Debug, Clone)]
pub struct OverdensityField {
    pub level: u32,
    /// Cell Morton indices (sorted).
    pub cell_indices: Vec<u64>,
    /// δ_c = n_D / (α · n_R) - 1 for each cell.
    pub delta: Vec<f64>,
    /// True if cell is valid (n_R > 0 ↔ inside survey footprint).
    pub valid: Vec<bool>,
    /// Data-catalog weight per cell.
    pub w_data: Vec<f64>,
    /// Random-catalog weight per cell (for pair weighting).
    pub w_random: Vec<f64>,
    /// Global normalization α = N_D / N_R.
    pub alpha: f64,
    /// Total data count (integer, for exact LS arithmetic).
    pub total_wd: f64,
    /// Total random count (integer, for exact LS arithmetic).
    pub total_wr: f64,
    /// Physical side length of a cell at this level.
    pub cell_size: f64,
    /// Fast lookup: cell Morton index → position in the arrays.
    pub cell_map: HashMap<u64, usize>,
}

/// Build a cell histogram from Morton-sorted particles at a given level.
///
/// Cost: O(N) — single scan of the sorted array.
pub fn build_cell_histogram(
    sorted_particles: &[MortonParticle],
    level: u32,
    bits_per_axis: u32,
) -> CellHistogram {
    let mut cell_indices = Vec::new();
    let mut w_data = Vec::new();
    let mut w_random = Vec::new();
    let mut n_data_vec = Vec::new();
    let mut n_random_vec = Vec::new();

    if sorted_particles.is_empty() {
        return CellHistogram {
            level,
            cell_indices,
            w_data,
            w_random,
            n_data: n_data_vec,
            n_random: n_random_vec,
        };
    }

    let mut current_cell = morton::cell_index(sorted_particles[0].code, level, bits_per_axis);
    let mut wd = 0.0f64;
    let mut wr = 0.0f64;
    let mut nd = 0u32;
    let mut nr = 0u32;

    for p in sorted_particles {
        let ci = morton::cell_index(p.code, level, bits_per_axis);
        if ci != current_cell {
            // Flush previous cell.
            cell_indices.push(current_cell);
            w_data.push(wd);
            w_random.push(wr);
            n_data_vec.push(nd);
            n_random_vec.push(nr);
            current_cell = ci;
            wd = 0.0;
            wr = 0.0;
            nd = 0;
            nr = 0;
        }
        match p.catalog {
            CatalogFlag::Data => {
                wd += p.weight;
                nd += 1;
            }
            CatalogFlag::Random => {
                wr += p.weight;
                nr += 1;
            }
        }
    }
    // Flush last cell.
    cell_indices.push(current_cell);
    w_data.push(wd);
    w_random.push(wr);
    n_data_vec.push(nd);
    n_random_vec.push(nr);

    CellHistogram {
        level,
        cell_indices,
        w_data,
        w_random,
        n_data: n_data_vec,
        n_random: n_random_vec,
    }
}

/// Compute the overdensity field from a cell histogram.
///
/// For unweighted catalogs (unit weights), δ_c = n_D / (α · n_R) - 1,
/// where α = Σ w_D / Σ w_R = N_D / N_R.
///
/// Cells with n_R = 0 are masked (outside the survey footprint).
pub fn compute_overdensity(hist: &CellHistogram, config: &MortonConfig) -> OverdensityField {
    let total_wd: f64 = hist.w_data.iter().sum();
    let total_wr: f64 = hist.w_random.iter().sum();
    let alpha = if total_wr > 0.0 {
        total_wd / total_wr
    } else {
        1.0
    };

    let n = hist.cell_indices.len();
    let mut delta = Vec::with_capacity(n);
    let mut valid = Vec::with_capacity(n);
    let mut w_data = Vec::with_capacity(n);
    let mut w_random = Vec::with_capacity(n);
    let mut cell_map = HashMap::with_capacity(n);

    for (i, &ci) in hist.cell_indices.iter().enumerate() {
        cell_map.insert(ci, i);
        let wd = hist.w_data[i];
        let wr = hist.w_random[i];
        w_data.push(wd);
        w_random.push(wr);
        if wr > 0.0 {
            delta.push(wd / (alpha * wr) - 1.0);
            valid.push(true);
        } else {
            delta.push(0.0);
            valid.push(false);
        }
    }

    let cell_size = config.box_size / (1u64 << hist.level) as f64;

    OverdensityField {
        level: hist.level,
        cell_indices: hist.cell_indices.clone(),
        delta,
        valid,
        w_data,
        w_random,
        alpha,
        total_wd,
        total_wr,
        cell_size,
        cell_map,
    }
}

/// Build cell histograms for all levels from 1 to l_max.
pub fn build_all_levels(
    sorted_particles: &[MortonParticle],
    config: &MortonConfig,
    l_max: u32,
) -> Vec<CellHistogram> {
    (1..=l_max)
        .map(|level| build_cell_histogram(sorted_particles, level, config.bits_per_axis))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morton::{encode_morton_64, MortonConfig, MortonParticle, CatalogFlag};

    fn make_particle(ix: u32, iy: u32, iz: u32, cat: CatalogFlag) -> MortonParticle {
        MortonParticle {
            code: encode_morton_64(ix, iy, iz),
            weight: 1.0,
            catalog: cat,
            orig_index: 0,
        }
    }

    #[test]
    fn histogram_basic() {
        let config = MortonConfig::new(8.0, true);
        // Place data and random particles at known grid locations.
        // At level 1, the grid is 2×2×2. Coords 0-1048575 map to cells 0,1.
        // Mid-point = 1048576 (2^20).
        let mut particles = vec![
            make_particle(0, 0, 0, CatalogFlag::Data),
            make_particle(0, 0, 0, CatalogFlag::Random),
            make_particle(0, 0, 0, CatalogFlag::Random),
            make_particle(1 << 20, 0, 0, CatalogFlag::Data),
            make_particle(1 << 20, 0, 0, CatalogFlag::Data),
            make_particle(1 << 20, 0, 0, CatalogFlag::Random),
        ];
        morton::sort_particles(&mut particles);

        let hist = build_cell_histogram(&particles, 1, config.bits_per_axis);
        // Should have 2 cells.
        assert_eq!(hist.cell_indices.len(), 2);

        // Cell (0,0,0): 1 data, 2 random
        // Cell (1,0,0): 2 data, 1 random
        let total_d: f64 = hist.w_data.iter().sum();
        let total_r: f64 = hist.w_random.iter().sum();
        assert_eq!(total_d, 3.0);
        assert_eq!(total_r, 3.0);
    }

    #[test]
    fn overdensity_uniform() {
        let config = MortonConfig::new(8.0, true);
        // Uniform: each cell has same data/random ratio → δ = 0.
        let mut particles = vec![
            make_particle(0, 0, 0, CatalogFlag::Data),
            make_particle(0, 0, 0, CatalogFlag::Random),
            make_particle(1 << 20, 0, 0, CatalogFlag::Data),
            make_particle(1 << 20, 0, 0, CatalogFlag::Random),
        ];
        morton::sort_particles(&mut particles);

        let hist = build_cell_histogram(&particles, 1, config.bits_per_axis);
        let field = compute_overdensity(&hist, &config);

        for (i, &v) in field.valid.iter().enumerate() {
            if v {
                assert!(
                    field.delta[i].abs() < 1e-12,
                    "expected δ=0 for uniform, got {}",
                    field.delta[i]
                );
            }
        }
    }
}

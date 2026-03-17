//! Direct pair counting between particles in neighboring Morton cells.
//!
//! At the finest octree levels where the grid-based LS estimator loses
//! resolution, this module enumerates all particle pairs between a cell
//! and its 13 half-space neighbors (plus intra-cell with i<j), bins by
//! Euclidean distance, and accumulates DD/DR/RR counts for Landy-Szalay.
//!
//! Cost: O(N × 26 × ⟨n_per_cell⟩) where ⟨n_per_cell⟩ ~ n̄ h³.
//! For the finest usable level with ~1 particle/cell this is O(N).

use std::collections::HashMap;

use crate::morton::{self, CatalogFlag, MortonConfig, MortonParticle};

/// Self (0,0,0) plus the 13 half-space neighbors.
/// Using half-space lags avoids double-counting inter-cell pairs:
/// for each inter-cell pair (A, B) with A ≠ B, exactly one of the cells
/// sees the other via a half-space lag.
/// Intra-cell pairs use i < j.
const HALF_SPACE_LAGS: [(i32, i32, i32); 14] = [
    // self (intra-cell pairs with i < j)
    (0, 0, 0),
    // face (3)
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    // edge (6)
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
    (1, -1, 0),
    (1, 0, -1),
    (0, 1, -1),
    // corner (4)
    (1, 1, 1),
    (1, -1, 1),
    (1, 1, -1),
    (1, -1, -1),
];

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Pair-count result from direct cell-pair enumeration.
#[derive(Debug, Clone)]
pub struct CellPairCounts {
    /// Radial bin edges (n_bins + 1 values).
    pub r_edges: Vec<f64>,
    /// Bin centers.
    pub r_centers: Vec<f64>,
    /// Weighted data-data pair counts per bin.
    pub dd: Vec<f64>,
    /// Weighted data-random pair counts per bin.
    pub dr: Vec<f64>,
    /// Weighted random-random pair counts per bin.
    pub rr: Vec<f64>,
    /// Total number of data particles.
    pub n_data: usize,
    /// Total number of random particles.
    pub n_random: usize,
}

/// ξ estimate from direct cell-pair counting.
#[derive(Debug, Clone)]
pub struct CellPairXi {
    /// Bin centers.
    pub r: Vec<f64>,
    /// Landy-Szalay ξ estimate.
    pub xi: Vec<f64>,
    /// DD counts (for diagnostics).
    pub dd: Vec<f64>,
    /// RR counts (for diagnostics).
    pub rr: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Cell occupancy map
// ---------------------------------------------------------------------------

/// A span of particles in the Morton-sorted array belonging to one cell.
#[derive(Clone, Copy, Debug)]
struct CellSpan {
    start: usize,
    end: usize, // exclusive
}

/// Build a map from cell Morton index → particle span in the sorted array.
fn build_cell_spans(
    sorted_particles: &[MortonParticle],
    level: u32,
    bits_per_axis: u32,
) -> HashMap<u64, CellSpan> {
    let mut spans = HashMap::new();
    if sorted_particles.is_empty() {
        return spans;
    }

    let mut current_cell = morton::cell_index(sorted_particles[0].code, level, bits_per_axis);
    let mut start = 0usize;

    for (i, p) in sorted_particles.iter().enumerate() {
        let ci = morton::cell_index(p.code, level, bits_per_axis);
        if ci != current_cell {
            spans.insert(current_cell, CellSpan { start, end: i });
            current_cell = ci;
            start = i;
        }
    }
    spans.insert(current_cell, CellSpan { start, end: sorted_particles.len() });
    spans
}

// ---------------------------------------------------------------------------
// Distance computation with periodic minimum image
// ---------------------------------------------------------------------------

#[inline]
fn periodic_dist_sq(a: &[f64; 3], b: &[f64; 3], box_size: f64, periodic: bool) -> f64 {
    let mut r2 = 0.0;
    for k in 0..3 {
        let mut d = a[k] - b[k];
        if periodic {
            if d > 0.5 * box_size {
                d -= box_size;
            } else if d < -0.5 * box_size {
                d += box_size;
            }
        }
        r2 += d * d;
    }
    r2
}

// ---------------------------------------------------------------------------
// Core pair counter
// ---------------------------------------------------------------------------

/// Count pairs between particles in neighboring Morton cells.
///
/// `sorted_particles` must be Morton-sorted (from `prepare_particles`).
/// `data_positions` and `random_positions` are the original position arrays
/// (indexed by `MortonParticle::orig_index`).
///
/// `level`: octree level at which to group particles into cells.
/// `r_edges`: radial bin edges for the histogram.
pub fn count_cell_pairs(
    sorted_particles: &[MortonParticle],
    data_positions: &[[f64; 3]],
    random_positions: &[[f64; 3]],
    level: u32,
    r_edges: &[f64],
    config: &MortonConfig,
) -> CellPairCounts {
    let n_bins = r_edges.len() - 1;
    let mut dd = vec![0.0f64; n_bins];
    let mut dr = vec![0.0f64; n_bins];
    let mut rr = vec![0.0f64; n_bins];

    // Precompute squared bin edges for fast comparison.
    let r2_edges: Vec<f64> = r_edges.iter().map(|&r| r * r).collect();

    let spans = build_cell_spans(sorted_particles, level, config.bits_per_axis);
    let cell_indices: Vec<u64> = spans.keys().copied().collect();

    let mut n_data = 0usize;
    let mut n_random = 0usize;
    for p in sorted_particles {
        match p.catalog {
            CatalogFlag::Data => n_data += 1,
            CatalogFlag::Random => n_random += 1,
        }
    }

    // For each occupied cell, iterate over self + 13 half-space neighbors.
    // Using half-space lags ensures every inter-cell pair is counted once.
    // Intra-cell pairs use i < j.
    for &ci in &cell_indices {
        let span_a = spans[&ci];

        for &(dx, dy, dz) in &HALF_SPACE_LAGS {
            let is_self = dx == 0 && dy == 0 && dz == 0;
            let nbr_ci = if is_self {
                Some(ci)
            } else {
                morton::neighbor_cell(ci, dx, dy, dz, level, config.periodic)
            };

            let nbr_ci = match nbr_ci {
                Some(c) => c,
                None => continue,
            };

            let span_b = match spans.get(&nbr_ci) {
                Some(s) => *s,
                None => continue,
            };

            // Enumerate all pairs between span_a and span_b.
            // For self-cell, use i < j to avoid double counting.
            // For inter-cell (half-space), enumerate all (i, j) pairs.
            for ia in span_a.start..span_a.end {
                let pa = &sorted_particles[ia];
                let pos_a = get_position(pa, data_positions, random_positions);

                let jstart = if is_self { ia + 1 } else { span_b.start };
                for ib in jstart..span_b.end {
                    let pb = &sorted_particles[ib];
                    let pos_b = get_position(pb, data_positions, random_positions);

                    let r2 = periodic_dist_sq(&pos_a, &pos_b, config.box_size, config.periodic);

                    // Binary search into bins.
                    if r2 < r2_edges[0] || r2 >= r2_edges[n_bins] {
                        continue;
                    }
                    let bin = match r2_edges.binary_search_by(|e| {
                        e.partial_cmp(&r2).unwrap()
                    }) {
                        Ok(i) => i.min(n_bins - 1),
                        Err(i) => {
                            if i == 0 { continue; }
                            (i - 1).min(n_bins - 1)
                        }
                    };

                    let w = pa.weight * pb.weight;
                    match (pa.catalog, pb.catalog) {
                        (CatalogFlag::Data, CatalogFlag::Data) => dd[bin] += w,
                        (CatalogFlag::Data, CatalogFlag::Random)
                        | (CatalogFlag::Random, CatalogFlag::Data) => dr[bin] += w,
                        (CatalogFlag::Random, CatalogFlag::Random) => rr[bin] += w,
                    }
                }
            }
        }
    }

    // Bin centers.
    let r_centers: Vec<f64> = (0..n_bins)
        .map(|i| 0.5 * (r_edges[i] + r_edges[i + 1]))
        .collect();

    CellPairCounts {
        r_edges: r_edges.to_vec(),
        r_centers,
        dd,
        dr,
        rr,
        n_data,
        n_random,
    }
}

#[inline]
fn get_position(
    p: &MortonParticle,
    data_positions: &[[f64; 3]],
    random_positions: &[[f64; 3]],
) -> [f64; 3] {
    match p.catalog {
        CatalogFlag::Data => data_positions[p.orig_index as usize],
        CatalogFlag::Random => random_positions[p.orig_index as usize],
    }
}

/// Compute Landy-Szalay ξ from cell-pair counts.
pub fn landy_szalay(counts: &CellPairCounts) -> CellPairXi {
    let nd = counts.n_data as f64;
    let nr = counts.n_random as f64;
    let n_bins = counts.dd.len();
    let mut xi = vec![0.0; n_bins];

    for i in 0..n_bins {
        let dd_val = counts.dd[i];
        let dr_val = counts.dr[i];
        let rr_val = counts.rr[i];

        if rr_val > 0.0 && nd > 0.0 && nr > 0.0 {
            // Standard Landy-Szalay:
            //   ξ = (DD·nr² − 2·DR·nd·nr + RR·nd²) / (RR·nd²)
            // But DD/DR/RR are raw weighted counts (not normalized).
            // Normalize: DD_norm = DD / (nd*(nd-1)/2), etc.
            let dd_norm = dd_val / (nd * (nd - 1.0) / 2.0);
            let dr_norm = dr_val / (nd * nr);
            let rr_norm = rr_val / (nr * (nr - 1.0) / 2.0);
            if rr_norm > 0.0 {
                xi[i] = (dd_norm - 2.0 * dr_norm + rr_norm) / rr_norm;
            }
        }
    }

    CellPairXi {
        r: counts.r_centers.clone(),
        xi,
        dd: counts.dd.clone(),
        rr: counts.rr.clone(),
    }
}

// ---------------------------------------------------------------------------
// Convenience: logarithmic bin edges
// ---------------------------------------------------------------------------

/// Generate logarithmically spaced bin edges from r_min to r_max.
pub fn log_bin_edges(r_min: f64, r_max: f64, n_bins: usize) -> Vec<f64> {
    let log_min = r_min.ln();
    let log_max = r_max.ln();
    (0..=n_bins)
        .map(|i| (log_min + (log_max - log_min) * i as f64 / n_bins as f64).exp())
        .collect()
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
    fn cell_pair_poisson_null() {
        let box_size = 100.0;
        let n_data = 5_000;
        let n_random = 25_000;
        let data = uniform_random(n_data, box_size, 42);
        let randoms = uniform_random(n_random, box_size, 43);

        let config = MortonConfig::new(box_size, true);
        let particles = morton::prepare_particles(&data, &randoms, &config);

        let level = 3; // 8^3 = 512 cells → ~10 particles/cell
        let r_edges = log_bin_edges(5.0, 60.0, 10);

        let counts = count_cell_pairs(&particles, &data, &randoms, level, &r_edges, &config);
        let result = landy_szalay(&counts);

        // Poisson → ξ ≈ 0 (within sample variance for N=5k).
        for (r, xi) in result.r.iter().zip(result.xi.iter()) {
            assert!(
                xi.abs() < 0.25,
                "ξ({:.1}) = {:.4}, expected ~0 for Poisson",
                r,
                xi
            );
        }
    }

    #[test]
    fn pair_count_consistency() {
        // Total DD + DR + RR should account for all pairs within r_max.
        let box_size = 50.0;
        let data = uniform_random(200, box_size, 10);
        let randoms = uniform_random(400, box_size, 11);
        let config = MortonConfig::new(box_size, true);
        let particles = morton::prepare_particles(&data, &randoms, &config);

        let level = 2; // 64 cells
        let r_edges = log_bin_edges(1.0, 40.0, 20);
        let counts = count_cell_pairs(&particles, &data, &randoms, level, &r_edges, &config);

        // All counts should be non-negative.
        for i in 0..counts.dd.len() {
            assert!(counts.dd[i] >= 0.0);
            assert!(counts.dr[i] >= 0.0);
            assert!(counts.rr[i] >= 0.0);
        }

        // DD total should be ≤ n_data*(n_data-1)/2.
        let dd_total: f64 = counts.dd.iter().sum();
        let max_dd = data.len() as f64 * (data.len() as f64 - 1.0) / 2.0;
        assert!(dd_total <= max_dd + 1.0, "DD total {dd_total} exceeds max {max_dd}");
    }
}

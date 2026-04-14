//! Two-pass compute engine for per-point density profiles.
//!
//! Pass 1 (DD): for each dilution level, build sub-trees from subsamples,
//! query all subsample points against their own tree, then scale counts
//! by the dilution factor and merge across levels.
//!
//! Pass 2 (DR): build one random tree, then query each data point into it
//! with a per-point k_dr tailored to match the DD radial reach.

use crate::estimator::{
    exclude_self_pairs, volume, CumulativeProfile, LandySzalayKnn, PerPointProfiles, PointProfile,
};
use crate::ladder::DilutionLadder;
use crate::tree::PointTree;

impl PerPointProfiles {
    /// Compute per-point DD and DR profiles across dilution levels.
    ///
    /// - `data`: the full data catalog (N_d points).
    /// - `randoms`: the full random catalog (N_r points).
    /// - `ladder`: pre-built dilution ladder over the data.
    /// - `k_max`: maximum neighbors at level 0.
    /// - `box_size`: periodic box side length.
    pub fn compute(
        data: &[[f64; 3]],
        randoms: &[[f64; 3]],
        ladder: &DilutionLadder,
        k_max: usize,
        box_size: f64,
    ) -> Self {
        let n_data = data.len();
        let n_random = randoms.len();
        let vol = box_size * box_size * box_size;
        let nbar_data = n_data as f64 / vol;
        let nbar_random = n_random as f64 / vol;

        // Per-point DD: accumulate CumulativeProfiles from each dilution level.
        // dd_levels[point_index] = Vec of per-level CumulativeProfiles
        let mut dd_levels: Vec<Vec<CumulativeProfile>> = vec![Vec::new(); n_data];

        // --- Pass 1: DD across dilution levels ---
        for level in &ladder.levels {
            let dilution_factor = level.dilution_factor;
            let k_eff = DilutionLadder::effective_k_max(k_max, n_data, dilution_factor);
            if k_eff == 0 {
                continue;
            }

            for subsample_indices in &level.subsamples {
                let sub_pos: Vec<[f64; 3]> =
                    subsample_indices.iter().map(|&i| data[i]).collect();
                let sub_tree = PointTree::build(sub_pos.clone());

                // Query k_eff+1 to exclude self-pair
                let estimator = LandySzalayKnn::new(k_eff + 1);
                let raw_dists =
                    estimator.query_distances_periodic(&sub_tree, &sub_pos, box_size);
                let dists = exclude_self_pairs(raw_dists, k_eff);

                // Map results back to original point indices
                for (local_idx, qd) in dists.per_query.iter().enumerate() {
                    let global_idx = subsample_indices[local_idx];
                    let cp = CumulativeProfile::from_knn(&qd.distances, dilution_factor);
                    dd_levels[global_idx].push(cp);
                }
            }
        }

        // Merge DD across levels for each point
        let dd_merged: Vec<CumulativeProfile> = dd_levels
            .into_iter()
            .map(|levels| CumulativeProfile::merge(&levels))
            .collect();

        // --- Pass 2: DR (data → random tree) ---
        let random_tree = PointTree::build(randoms.to_vec());

        let dr_profiles: Vec<CumulativeProfile> = dd_merged
            .iter()
            .enumerate()
            .map(|(i, dd)| {
                // Determine r_max from DD profile, then set k_dr to cover that volume
                let r_max = dd.radii.last().copied().unwrap_or(0.0);
                if r_max <= 0.0 {
                    return CumulativeProfile {
                        radii: vec![],
                        counts: vec![],
                    };
                }
                let v_max = volume(r_max);
                let k_dr_raw = (nbar_random * v_max).ceil() as usize;
                let k_dr = k_dr_raw.max(1).min(n_random / 4);

                let estimator = LandySzalayKnn::new(k_dr);
                let query = &[data[i]];
                let dists = estimator.query_distances_periodic(&random_tree, query, box_size);
                CumulativeProfile::from_knn(&dists.per_query[0].distances, 1)
            })
            .collect();

        // Assemble PointProfiles
        let profiles: Vec<PointProfile> = dd_merged
            .into_iter()
            .zip(dr_profiles.into_iter())
            .enumerate()
            .map(|(i, (dd, dr))| PointProfile {
                point_index: i,
                dd,
                dr,
            })
            .collect();

        PerPointProfiles {
            profiles,
            nbar_data,
            nbar_random,
            box_size,
        }
    }
}

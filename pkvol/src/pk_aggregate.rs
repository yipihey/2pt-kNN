//! Streaming F_k / P_k aggregation across queries with rayon parallelism.
//!
//! Per-query memory is bounded: we never materialize all per-query K matrices.
//! Each thread keeps a private F-accumulator of shape `[n_kp1, n_u, n_int]`
//! where `n_kp1 = n_k + 1` (we accumulate `1[K_q >= k]` for `k_values` and
//! one extra `k_max + 1` so that `P_k = F_k - F_{k+1}` is computable for the
//! largest requested k). Final reduction is a simple sum of arrays.
//!
//! See `lib.rs::measure_pk_core` for the top-level driver.

use rayon::prelude::*;

use crate::angular_search::AngularTree;
use crate::ecdf2d::{ecdf2d_histogram, ecdf2d_sweep, EcdfBackend};
use crate::haversine::theta_from_dot;
use crate::shell_counts::shell_counts;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AngularVar {
    /// x_i = theta_i in radians.
    Theta,
    /// x_i = theta_i^2 (steradian-area proxy in the small-angle limit).
    Theta2,
    /// x_i = 1 - cos(theta_i)  (proportional to differential solid angle).
    Omega,
}

#[derive(Clone, Debug)]
pub struct ShellSpec {
    pub z_edges: Vec<f64>,
    /// (l, r) index pairs into z_edges; usize::MAX for `l` means "from -inf".
    pub intervals: Vec<(usize, usize)>,
}

#[derive(Clone, Debug)]
pub struct MeasureConfig {
    pub theta_max: f64,
    pub angular_variable: AngularVar,
    pub backend: EcdfBackend,
    pub exclude_self: bool,
    /// Tolerance (in radians) for self-matching detection when `exclude_self` is true.
    pub self_match_tol: f64,
}

impl Default for MeasureConfig {
    fn default() -> Self {
        MeasureConfig {
            theta_max: 0.05,
            angular_variable: AngularVar::Theta,
            backend: EcdfBackend::Sweep,
            exclude_self: false,
            self_match_tol: 1e-9,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PkOutput {
    pub n_k: usize,
    pub n_u: usize,
    pub n_intervals: usize,
    /// F: shape [n_k, n_u, n_intervals], row-major.
    pub f: Vec<f64>,
    /// P: shape [n_k, n_u, n_intervals], row-major.
    pub p: Vec<f64>,
    /// Total query weight used in normalization.
    pub total_weight: f64,
    /// Number of queries that contributed (== n_query in this version).
    pub n_query_used: usize,
    /// Diagnostic: mean local candidate count per query.
    pub mean_candidates: f64,
    /// Mean K_q[u_max, z_total] across queries (count summary).
    pub mean_total_count: f64,
}

/// Convert a haversine angle to the chosen angular variable.
#[inline]
fn x_from_theta(theta: f64, av: AngularVar) -> f64 {
    match av {
        AngularVar::Theta => theta,
        AngularVar::Theta2 => theta * theta,
        AngularVar::Omega => 1.0 - theta.cos(),
    }
}

/// Lower-edge transform corresponding to `theta_max`. Used to know which
/// candidates are within u_edges.last(). The angular tree already returned
/// only candidates with theta <= theta_max.
#[inline]
pub fn x_max_for(av: AngularVar, theta_max: f64) -> f64 {
    x_from_theta(theta_max, av)
}

/// Top-level streaming measurement. `gal_*` arrays describe the count catalog,
/// `query_*` arrays describe the query catalog. `u_edges` are in the angular
/// variable units (caller must transform if needed). `k_values` must be sorted
/// ascending and >= 0 (typically integer-valued).
#[allow(clippy::too_many_arguments)]
pub fn measure_pk(
    tree: &AngularTree,
    gal_z: &[f64],
    gal_w: &[f64],
    query_ra: &[f64],
    query_dec: &[f64],
    query_w: Option<&[f64]>,
    u_edges: &[f64],
    shell: &ShellSpec,
    k_values: &[f64],
    cfg: &MeasureConfig,
) -> PkOutput {
    let n_q = query_ra.len();
    assert_eq!(query_dec.len(), n_q);
    if let Some(qw) = query_w {
        assert_eq!(qw.len(), n_q);
    }
    assert_eq!(gal_z.len(), gal_w.len());
    assert_eq!(gal_z.len(), tree.len());

    let n_u = u_edges.len();
    let n_z = shell.z_edges.len();
    let n_int = shell.intervals.len();
    let n_k = k_values.len();

    if n_q == 0 || n_u == 0 || n_int == 0 || n_k == 0 {
        return PkOutput {
            n_k,
            n_u,
            n_intervals: n_int,
            f: vec![0.0; n_k * n_u * n_int],
            p: vec![0.0; n_k * n_u * n_int],
            total_weight: 0.0,
            n_query_used: 0,
            mean_candidates: 0.0,
            mean_total_count: 0.0,
        };
    }

    // We need 1[K >= k] for k in `k_values` plus k = k_max + 1 to produce P_k.
    let kp1 = n_k + 1;
    let mut k_values_ext = Vec::with_capacity(kp1);
    k_values_ext.extend_from_slice(k_values);
    let last_k = *k_values.last().unwrap();
    // Use next integer if the user gave integer-valued thresholds; otherwise +1.
    let extra = (last_k + 1.0).floor().max(last_k + 1.0);
    k_values_ext.push(extra);

    // Precompute for the angular variable transform: convert query candidate
    // theta -> x. We need u_edges sorted ascending (caller assumed).
    let u_edges_sorted: Vec<f64> = {
        let mut v = u_edges.to_vec();
        // sanity: if not sorted, sort
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v
    };
    let z_edges_sorted: Vec<f64> = {
        let mut v = shell.z_edges.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v
    };

    // Per-query weights default to 1.
    let qw_default: Vec<f64> = vec![1.0; n_q];
    let qw: &[f64] = query_w.unwrap_or(&qw_default);

    // Total query weight.
    let total_w: f64 = qw.iter().sum();

    let cfg = cfg.clone();
    let av = cfg.angular_variable;
    let backend = cfg.backend;
    let theta_max = cfg.theta_max;
    let exclude_self = cfg.exclude_self;
    let self_tol = cfg.self_match_tol;

    type Accum = (Vec<f64>, f64, u64, f64);
    // (F-accum [kp1 * n_u * n_int], cand_sum, total_count_sum, mean_total_count_sum)

    let acc_init = || -> Accum {
        (vec![0.0; kp1 * n_u * n_int], 0.0, 0, 0.0)
    };

    let result: Accum = (0..n_q)
        .into_par_iter()
        .fold(acc_init, |mut acc, q_idx| {
            let ra_q = query_ra[q_idx];
            let dec_q = query_dec[q_idx];
            let w_q = qw[q_idx];

            let mut cand: Vec<usize> = Vec::with_capacity(64);
            tree.query_radius(ra_q, dec_q, theta_max, &mut cand);

            // Build local (x, y, w) arrays.
            let mut xs = Vec::with_capacity(cand.len());
            let mut ys = Vec::with_capacity(cand.len());
            let mut ws = Vec::with_capacity(cand.len());
            let q_unit = crate::haversine::radec_to_unit(ra_q, dec_q);
            for &i in &cand {
                let theta = theta_from_dot(&q_unit, tree.unit(i));
                if exclude_self && theta < self_tol {
                    continue;
                }
                xs.push(x_from_theta(theta, av));
                ys.push(gal_z[i]);
                ws.push(gal_w[i]);
            }
            acc.1 += xs.len() as f64;

            // Compute K(u_a, z_b) for this query.
            let mut k_2d = vec![0.0; n_u * n_z];
            match backend {
                EcdfBackend::Sweep => {
                    ecdf2d_sweep(&xs, &ys, &ws, &u_edges_sorted, &z_edges_sorted, &mut k_2d);
                }
                EcdfBackend::Histogram => {
                    ecdf2d_histogram(&xs, &ys, &ws, &u_edges_sorted, &z_edges_sorted, &mut k_2d);
                }
            }

            // Diagnostic: K at largest u and largest z.
            if n_u > 0 && n_z > 0 {
                acc.2 += 1;
                acc.3 += k_2d[(n_u - 1) * n_z + (n_z - 1)];
            }

            // Shell counts.
            let mut k_shell = vec![0.0; n_u * n_int];
            shell_counts(&k_2d, n_u, n_z, &shell.intervals, &mut k_shell);

            // Accumulate 1[K_shell >= k] * w_q for each k in k_values_ext.
            for j in 0..kp1 {
                let kj = k_values_ext[j];
                let plane = j * n_u * n_int;
                for a in 0..n_u {
                    for ell in 0..n_int {
                        let val = k_shell[a * n_int + ell];
                        if val >= kj {
                            acc.0[plane + a * n_int + ell] += w_q;
                        }
                    }
                }
            }
            acc
        })
        .reduce(acc_init, |mut a, b| {
            for (x, y) in a.0.iter_mut().zip(b.0.iter()) {
                *x += *y;
            }
            a.1 += b.1;
            a.2 += b.2;
            a.3 += b.3;
            a
        });

    let (f_sum, cand_sum, n_used, total_count_sum) = result;

    // Normalize: F_k = f_sum / total_w.
    let inv_w = if total_w > 0.0 { 1.0 / total_w } else { 0.0 };
    let mut f = vec![0.0; n_k * n_u * n_int];
    let mut p = vec![0.0; n_k * n_u * n_int];
    for j in 0..n_k {
        let plane_in = j * n_u * n_int;
        let plane_out = j * n_u * n_int;
        for ai in 0..(n_u * n_int) {
            f[plane_out + ai] = f_sum[plane_in + ai] * inv_w;
        }
    }
    // P_k = F_k - F_{k+1}; for the last requested k, use F_{k+1} from the extra slot.
    for j in 0..n_k {
        let plane_out = j * n_u * n_int;
        let next_plane = (j + 1) * n_u * n_int;
        for ai in 0..(n_u * n_int) {
            p[plane_out + ai] = f[plane_out + ai] - (f_sum[next_plane + ai] * inv_w);
        }
    }

    PkOutput {
        n_k,
        n_u,
        n_intervals: n_int,
        f,
        p,
        total_weight: total_w,
        n_query_used: n_q,
        mean_candidates: if n_q > 0 { cand_sum / n_q as f64 } else { 0.0 },
        mean_total_count: if n_used > 0 {
            total_count_sum / n_used as f64
        } else {
            0.0
        },
    }
}

/// Compute Lambda_q(u, z_interval): the weighted count of *catalog* points
/// (e.g. randoms) in each aperture/shell, *for each query individually*.
/// Returned shape is `[n_query, n_u, n_intervals]` row-major. This is a
/// memory-heavier O(n_q * n_u * n_int) output, intended for edge-fraction
/// computations. Use with care for very large n_q.
#[allow(clippy::too_many_arguments)]
pub fn lambda_per_query(
    tree: &AngularTree,
    gal_z: &[f64],
    gal_w: &[f64],
    query_ra: &[f64],
    query_dec: &[f64],
    u_edges: &[f64],
    shell: &ShellSpec,
    cfg: &MeasureConfig,
) -> Vec<f64> {
    let n_q = query_ra.len();
    assert_eq!(query_dec.len(), n_q);
    assert_eq!(gal_z.len(), gal_w.len());
    assert_eq!(gal_z.len(), tree.len());

    let n_u = u_edges.len();
    let n_z = shell.z_edges.len();
    let n_int = shell.intervals.len();
    let mut out = vec![0.0; n_q * n_u * n_int];

    let cfg = cfg.clone();
    let av = cfg.angular_variable;
    let backend = cfg.backend;
    let theta_max = cfg.theta_max;
    let exclude_self = cfg.exclude_self;
    let self_tol = cfg.self_match_tol;

    let u_edges_sorted: Vec<f64> = {
        let mut v = u_edges.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v
    };
    let z_edges_sorted: Vec<f64> = {
        let mut v = shell.z_edges.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v
    };

    out.par_chunks_mut(n_u * n_int)
        .enumerate()
        .for_each(|(q_idx, slot)| {
            let ra_q = query_ra[q_idx];
            let dec_q = query_dec[q_idx];

            let mut cand = Vec::with_capacity(64);
            tree.query_radius(ra_q, dec_q, theta_max, &mut cand);

            let q_unit = crate::haversine::radec_to_unit(ra_q, dec_q);
            let mut xs = Vec::with_capacity(cand.len());
            let mut ys = Vec::with_capacity(cand.len());
            let mut ws = Vec::with_capacity(cand.len());
            for &i in &cand {
                let theta = theta_from_dot(&q_unit, tree.unit(i));
                if exclude_self && theta < self_tol {
                    continue;
                }
                xs.push(x_from_theta(theta, av));
                ys.push(gal_z[i]);
                ws.push(gal_w[i]);
            }

            let mut k_2d = vec![0.0; n_u * n_z];
            match backend {
                EcdfBackend::Sweep => {
                    ecdf2d_sweep(&xs, &ys, &ws, &u_edges_sorted, &z_edges_sorted, &mut k_2d);
                }
                EcdfBackend::Histogram => {
                    ecdf2d_histogram(&xs, &ys, &ws, &u_edges_sorted, &z_edges_sorted, &mut k_2d);
                }
            }
            shell_counts(&k_2d, n_u, n_z, &shell.intervals, slot);
        });

    out
}

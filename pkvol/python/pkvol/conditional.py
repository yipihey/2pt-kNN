"""Conditional, median-centered P_k measurement using random catalogs.

The random catalog is the central object: it absorbs the survey window
function and selection effects through Lambda_q, the random-expected count
in each aperture. The data catalog only contributes the count statistic
N_q. Per-query coordinates (one (s, eta) per query x aperture cell):

    s_q  = log Lambda_q                                           (effective volume)
    eta_q = log( F_ref(z_q) * sqrt(Delta_Omega) / Delta_z )       (dimensionless aspect ratio)
    Lambda_q = alpha * (random count in aperture)                  (alpha = data/random weights)
    F_ref(z) = z * sqrt(1+z)                                       (default; user-supplied OK)

Median centering per (k, query_type X in {R, D}, z-bin of query):

    s_tilde_q  = s_q  - s_med(k, X, z_bin)
    eta_tilde_q = eta_q - eta_med(k, X, z_bin)

Target statistic (binned in centered coords):

    F_k^X(s_tilde_bin, eta_tilde_bin, z_bin) = P( N_q >= k | s_tilde, eta_tilde, z )
    P_k^X = F_k^X - F_{k+1}^X

Default log base is log10.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np

from ._api import _as_f64, apply_marks, measure_lambda


_ARR = np.ndarray


def default_F_ref(z: _ARR) -> _ARR:
    """Default redshift reference: F_ref(z) = z * sqrt(1+z)."""
    z = np.asarray(z, dtype=np.float64)
    return z * np.sqrt(1.0 + z)


def _log_fn(log_base: str) -> Callable[[_ARR], _ARR]:
    if log_base == "10":
        return np.log10
    if log_base == "e":
        return np.log
    raise ValueError(f"log_base must be '10' or 'e'; got {log_base!r}")


def _angular_var_from_theta(theta_edges_rad: _ARR, angular_variable: str) -> _ARR:
    """Map theta (radians) to the engine's angular variable units."""
    if angular_variable == "theta":
        return theta_edges_rad
    if angular_variable == "theta2":
        return theta_edges_rad * theta_edges_rad
    if angular_variable == "omega":
        return 1.0 - np.cos(theta_edges_rad)
    raise ValueError(
        f"angular_variable must be one of 'theta', 'theta2', 'omega'; got {angular_variable!r}"
    )


def _delta_omega_steradian(theta_edges_rad: _ARR) -> _ARR:
    """Solid angle of the polar cap at each theta (sr): 2*pi*(1 - cos theta).

    The 2*pi factor cancels under median centering; we keep it for
    dimensional clarity.
    """
    return 2.0 * np.pi * (1.0 - np.cos(theta_edges_rad))


def _adjacent_intervals(n_z: int) -> Sequence[Tuple[int, int]]:
    return [(i, i + 1) for i in range(n_z - 1)]


def _delta_z_per_shell(z_edges: _ARR, intervals: Sequence[Tuple[int, int]]) -> _ARR:
    """Per-shell Delta_z = z_edges[r] - z_edges[l]; l = -1 => lower bound = 0."""
    out = np.empty(len(intervals), dtype=np.float64)
    for i, (l, r) in enumerate(intervals):
        z_lo = 0.0 if l < 0 else float(z_edges[l])
        z_hi = float(z_edges[r])
        out[i] = z_hi - z_lo
    return out


def _shell_contains(z_edges: _ARR, intervals: Sequence[Tuple[int, int]], z_q: _ARR) -> _ARR:
    """Boolean array [n_q, n_int]: True where z_q is in shell (z_lo, z_hi]."""
    z_q = np.asarray(z_q, dtype=np.float64)
    out = np.zeros((z_q.size, len(intervals)), dtype=bool)
    for i, (l, r) in enumerate(intervals):
        z_lo = -np.inf if l < 0 else float(z_edges[l])
        z_hi = float(z_edges[r])
        out[:, i] = (z_q > z_lo) & (z_q <= z_hi)
    return out


def measure_pk_conditional(
    *,
    # data catalog
    ra_gal: _ARR,
    dec_gal: _ARR,
    z_gal: _ARR,
    weights_gal: Optional[_ARR] = None,
    marks_gal: Optional[_ARR] = None,
    mark_mode_gal: str = "none",
    mark_cut_gal: Optional[float] = None,
    mark_quantile_gal: Optional[Tuple[float, float]] = None,
    # random catalog
    ra_random: _ARR,
    dec_random: _ARR,
    z_random: _ARR,
    weights_random: Optional[_ARR] = None,
    marks_random: Optional[_ARR] = None,
    mark_mode_random: str = "none",
    mark_cut_random: Optional[float] = None,
    mark_quantile_random: Optional[Tuple[float, float]] = None,
    # queries
    ra_query: _ARR,
    dec_query: _ARR,
    z_query: _ARR,
    query_type: str,                   # "R" or "D"
    weights_query: Optional[_ARR] = None,
    query_indices: Optional[_ARR] = None,   # required for D + exclude_self
    exclude_self: Optional[bool] = None,    # default True for D, False for R
    # aperture grid
    theta_edges: _ARR,                 # radians (raw angular separation)
    z_edges: _ARR,                     # redshift thresholds for shells
    z_intervals: Optional[Sequence[Tuple[int, int]]] = None,
    angular_variable: str = "omega",
    theta_max: Optional[float] = None,
    backend: str = "ecdf",
    # k axis
    k_max: int = 8,
    # centered binning
    s_tilde_edges: _ARR,
    eta_tilde_edges: _ARR,
    z_query_edges: _ARR,               # binning of query z; medians taken per bin
    # anisotropy axis
    F_ref: Optional[Callable[[_ARR], _ARR]] = None,
    log_base: str = "10",
    # numerical
    n_threads: Optional[int] = None,
    return_diagnostics: bool = True,
) -> dict:
    """Conditional, median-centered P_k from a random catalog.

    Returns a dict with arrays shaped ``[k, s_tilde, eta_tilde, z_bin]`` for
    ``P``, ``F``, ``denominator``, plus per-(k, z_bin) ``s_med`` and
    ``eta_med`` so absolute coordinates remain recoverable. See module
    docstring for the math.
    """
    # ---------- 1. Input validation and prep ----------
    if query_type not in ("R", "D"):
        raise ValueError(f"query_type must be 'R' or 'D'; got {query_type!r}")
    if exclude_self is None:
        exclude_self = (query_type == "D")
    if F_ref is None:
        F_ref = default_F_ref
    log = _log_fn(log_base)

    ra_g = _as_f64(ra_gal, "ra_gal")
    dec_g = _as_f64(dec_gal, "dec_gal")
    z_g = _as_f64(z_gal, "z_gal")
    ra_r = _as_f64(ra_random, "ra_random")
    dec_r = _as_f64(dec_random, "dec_random")
    z_r = _as_f64(z_random, "z_random")
    ra_q = _as_f64(ra_query, "ra_query")
    dec_q = _as_f64(dec_query, "dec_query")
    z_q = _as_f64(z_query, "z_query")
    if not (ra_g.shape == dec_g.shape == z_g.shape):
        raise ValueError("ra_gal, dec_gal, z_gal must have equal length")
    if not (ra_r.shape == dec_r.shape == z_r.shape):
        raise ValueError("ra_random, dec_random, z_random must have equal length")
    if not (ra_q.shape == dec_q.shape == z_q.shape):
        raise ValueError("ra_query, dec_query, z_query must have equal length")

    th_e = _as_f64(theta_edges, "theta_edges")
    z_e = _as_f64(z_edges, "z_edges")
    s_e = _as_f64(s_tilde_edges, "s_tilde_edges")
    e_e = _as_f64(eta_tilde_edges, "eta_tilde_edges")
    zq_e = _as_f64(z_query_edges, "z_query_edges")
    if th_e.size < 1:
        raise ValueError("theta_edges must be non-empty")
    if z_e.size < 1:
        raise ValueError("z_edges must be non-empty")
    if s_e.size < 2 or e_e.size < 2 or zq_e.size < 2:
        raise ValueError("s_tilde_edges, eta_tilde_edges, z_query_edges need >= 2 entries")

    if theta_max is None:
        theta_max = float(th_e.max()) + 1e-9

    intervals = list(z_intervals) if z_intervals is not None else _adjacent_intervals(z_e.size)
    if len(intervals) == 0:
        raise ValueError("no shell intervals (need len(z_edges) >= 2 or explicit z_intervals)")

    # ---------- 2. Mark-applied effective weights and alpha ----------
    wg_eff = apply_marks(
        weights_gal,
        marks_gal,
        mode=mark_mode_gal,
        cut=mark_cut_gal,
        quantile=mark_quantile_gal,
        n_galaxies=ra_g.size,
    )
    wr_eff = apply_marks(
        weights_random,
        marks_random,
        mode=mark_mode_random,
        cut=mark_cut_random,
        quantile=mark_quantile_random,
        n_galaxies=ra_r.size,
    )
    sum_wg = float(wg_eff.sum())
    sum_wr = float(wr_eff.sum())
    if sum_wr <= 0.0:
        raise ValueError("random catalog has zero total (effective) weight")
    if sum_wg <= 0.0:
        raise ValueError("data catalog has zero total (effective) weight")
    alpha = sum_wg / sum_wr

    if weights_query is None:
        wq = np.ones(ra_q.size, dtype=np.float64)
    else:
        wq = _as_f64(weights_query, "weights_query")
        if wq.shape != ra_q.shape:
            raise ValueError("weights_query length must equal ra_query length")

    # ---------- 3. Aperture geometry ----------
    delta_omega = _delta_omega_steradian(th_e)             # (n_u,)
    delta_z = _delta_z_per_shell(z_e, intervals)           # (n_int,)
    if np.any(delta_omega <= 0.0):
        raise ValueError("Delta Omega must be positive: pass theta_edges in radians, > 0")
    if np.any(delta_z <= 0.0):
        raise ValueError("Delta z must be positive across all shells; check z_edges/z_intervals")

    # ---------- 4. Per-query Lambda and N via measure_lambda ----------
    z_int_arr = np.asarray(intervals, dtype=np.int64)

    Lambda_raw = measure_lambda(
        ra_r, dec_r, z_r, ra_q, dec_q,
        th_e, z_e, theta_max,
        weights_random=wr_eff,
        z_intervals=z_int_arr,
        angular_variable=angular_variable,
        backend=backend,
        n_threads=n_threads,
    )                                                       # (n_q, n_u, n_int)
    N = measure_lambda(
        ra_g, dec_g, z_g, ra_q, dec_q,
        th_e, z_e, theta_max,
        weights_random=wg_eff,
        z_intervals=z_int_arr,
        angular_variable=angular_variable,
        backend=backend,
        n_threads=n_threads,
    )

    Lambda = alpha * Lambda_raw

    # ---------- 5. Self-exclusion for D-type queries ----------
    if exclude_self and query_type == "D":
        if query_indices is None:
            raise ValueError(
                "query_type='D' with exclude_self=True requires query_indices "
                "(indices of each query into the data catalog so the self contribution can be removed)."
            )
        qi = np.asarray(query_indices, dtype=np.int64)
        if qi.shape != ra_q.shape:
            raise ValueError("query_indices must have the same length as ra_query")
        if (qi < 0).any() or (qi >= ra_g.size).any():
            raise ValueError("query_indices out of range for the data catalog")
        self_w = wg_eff[qi]                                 # (n_q,)
        self_z = z_g[qi]
        in_shell = _shell_contains(z_e, intervals, self_z)  # (n_q, n_int)
        # theta_self = 0 is inside every angular cell with u_edge >= 0.
        # Subtract self contribution from N[q, :, b] for shells containing self_z.
        # N shape: (n_q, n_u, n_int). Broadcast subtraction.
        sub = self_w[:, None] * in_shell.astype(np.float64)  # (n_q, n_int)
        N = np.maximum(N - sub[:, None, :], 0.0)             # numerical safety

    # ---------- 6. Per-cell s, eta ----------
    sqrt_omega = np.sqrt(delta_omega)                       # (n_u,)
    Fz = np.asarray(F_ref(z_q), dtype=np.float64)           # (n_q,)
    if Fz.shape != z_q.shape:
        raise ValueError("F_ref(z_query) must return an array of the same shape as z_query")

    # eta[q, a, b] = log( Fz[q] * sqrt_omega[a] / delta_z[b] )
    eta_qa = log(Fz[:, None] * sqrt_omega[None, :])         # (n_q, n_u)
    eta = eta_qa[:, :, None] - log(delta_z)[None, None, :]  # (n_q, n_u, n_int)

    # s[q, a, b] = log Lambda[q, a, b]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = log(np.where(Lambda > 0.0, Lambda, np.nan))     # (n_q, n_u, n_int)

    valid_se = np.isfinite(s) & np.isfinite(eta)            # (n_q, n_u, n_int)

    # ---------- 7. z-bin assignment for queries ----------
    n_zq = zq_e.size - 1
    z_bin_q = np.digitize(z_q, zq_e, right=False) - 1       # (n_q,)
    z_bin_q = np.where((z_q >= zq_e[0]) & (z_q < zq_e[-1]), z_bin_q, -1)

    # ---------- 8. Per-(k, z_bin) medians ----------
    n_k_internal = k_max + 1     # need F_{k_max+1} to compute P_{k_max}
    s_med = np.full((n_k_internal, n_zq), np.nan, dtype=np.float64)
    eta_med = np.full((n_k_internal, n_zq), np.nan, dtype=np.float64)

    # Broadcast z_bin_q to per-cell shape for masking in pass 1.
    z_bin_cell = np.broadcast_to(z_bin_q[:, None, None], s.shape)

    for k_idx in range(n_k_internal):
        k_val = k_idx + 1
        for zb in range(n_zq):
            mask = valid_se & (N >= k_val) & (z_bin_cell == zb)
            if mask.any():
                s_med[k_idx, zb] = float(np.median(s[mask]))
                eta_med[k_idx, zb] = float(np.median(eta[mask]))

    # ---------- 9. Centered binning (pass 2) ----------
    n_s = s_e.size - 1
    n_e = e_e.size - 1
    F_num = np.zeros((n_k_internal, n_s, n_e, n_zq), dtype=np.float64)
    denom = np.zeros((n_k_internal, n_s, n_e, n_zq), dtype=np.float64)

    if return_diagnostics:
        diag_lambda_sum = np.zeros_like(F_num)
        diag_N_sum = np.zeros_like(F_num)
        diag_eta_sum = np.zeros_like(F_num)
        diag_count = np.zeros_like(F_num)
        used_per_kbin = np.zeros((n_k_internal, n_zq), dtype=np.float64)
        total_per_zbin = np.zeros(n_zq, dtype=np.float64)
        n_q_total = float(ra_q.size) * float(th_e.size) * float(len(intervals))
        # total_per_zbin[zb] = total weight of cells with z_q in zb (regardless of validity)
        for zb in range(n_zq):
            mask_zb = (z_bin_cell == zb)
            total_per_zbin[zb] = float((wq[:, None, None] * mask_zb).sum())

    # Per-cell flat arrays once.
    wq_cell = np.broadcast_to(wq[:, None, None], s.shape)
    Lambda_cell = Lambda
    N_cell = N
    eta_cell = eta

    for k_idx in range(n_k_internal):
        k_val = k_idx + 1
        smk = s_med[k_idx]                    # (n_zq,)
        emk = eta_med[k_idx]
        # Skip cells whose z_bin has no median.
        med_valid_zb = np.isfinite(smk) & np.isfinite(emk)

        for zb in range(n_zq):
            if not med_valid_zb[zb]:
                continue
            mask_zb = valid_se & (z_bin_cell == zb)
            if not mask_zb.any():
                continue
            s_tilde = s[mask_zb] - smk[zb]
            eta_tilde = eta[mask_zb] - emk[zb]
            w_cell = wq_cell[mask_zb]
            n_cell = N_cell[mask_zb]
            l_cell = Lambda_cell[mask_zb]
            etav_cell = eta_cell[mask_zb]

            # Bin in (s_tilde, eta_tilde).
            d2, _, _ = np.histogram2d(
                s_tilde, eta_tilde,
                bins=(s_e, e_e),
                weights=w_cell,
            )
            denom[k_idx, :, :, zb] = d2

            kk_mask = n_cell >= k_val
            if kk_mask.any():
                f2, _, _ = np.histogram2d(
                    s_tilde[kk_mask], eta_tilde[kk_mask],
                    bins=(s_e, e_e),
                    weights=w_cell[kk_mask],
                )
                F_num[k_idx, :, :, zb] = f2

            if return_diagnostics:
                used_per_kbin[k_idx, zb] = float(d2.sum())
                # Mean diagnostics: numerator = weighted sum, denominator = unweighted count.
                cnt2, _, _ = np.histogram2d(s_tilde, eta_tilde, bins=(s_e, e_e))
                lam2, _, _ = np.histogram2d(
                    s_tilde, eta_tilde, bins=(s_e, e_e), weights=l_cell
                )
                n2, _, _ = np.histogram2d(
                    s_tilde, eta_tilde, bins=(s_e, e_e), weights=n_cell
                )
                eta2, _, _ = np.histogram2d(
                    s_tilde, eta_tilde, bins=(s_e, e_e), weights=etav_cell
                )
                diag_count[k_idx, :, :, zb] = cnt2
                diag_lambda_sum[k_idx, :, :, zb] = lam2
                diag_N_sum[k_idx, :, :, zb] = n2
                diag_eta_sum[k_idx, :, :, zb] = eta2

    # ---------- 10. F, P ----------
    with np.errstate(divide="ignore", invalid="ignore"):
        F_full = np.where(denom > 0.0, F_num / denom, np.nan)
    # Output P, F, denominator at k = 1..k_max only.
    F_out = F_full[:k_max]
    P_out = F_full[:k_max] - F_full[1:k_max + 1]
    denom_out = denom[:k_max]
    s_med_out = s_med[:k_max]
    eta_med_out = eta_med[:k_max]
    k_values = np.arange(1, k_max + 1, dtype=np.int64)

    out = {
        "P": P_out,
        "F": F_out,
        "denominator": denom_out,
        "s_med": s_med_out,
        "eta_med": eta_med_out,
        "s_tilde_edges": s_e,
        "eta_tilde_edges": e_e,
        "z_query_edges": zq_e,
        "k_values": k_values,
        "alpha": alpha,
        "query_type": query_type,
        "log_base": log_base,
        "delta_omega_steradian": delta_omega,
        "delta_z": delta_z,
        "theta_edges": th_e,
        "z_edges": z_e,
        "z_intervals": np.asarray(intervals, dtype=np.int64),
        "angular_variable": angular_variable,
    }

    if return_diagnostics:
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_lambda = np.where(diag_count > 0, diag_lambda_sum / diag_count, np.nan)
            mean_N = np.where(diag_count > 0, diag_N_sum / diag_count, np.nan)
            mean_eta = np.where(diag_count > 0, diag_eta_sum / diag_count, np.nan)
            frac_used = np.where(
                total_per_zbin[None, :] > 0,
                used_per_kbin / total_per_zbin[None, :],
                0.0,
            )
        out["diagnostics"] = {
            "fraction_used": frac_used[:k_max],
            "mean_lambda": mean_lambda[:k_max],
            "mean_N": mean_N[:k_max],
            "mean_eta": mean_eta[:k_max],
            "cell_count": diag_count[:k_max],
            "n_query": int(ra_q.size),
            "n_data": int(ra_g.size),
            "n_random": int(ra_r.size),
        }

    return out


def measure_pk_R(
    *,
    ra_gal: _ARR,
    dec_gal: _ARR,
    z_gal: _ARR,
    ra_random: _ARR,
    dec_random: _ARR,
    z_random: _ARR,
    n_query_subsample: Optional[int] = None,
    subsample_seed: int = 0,
    **kwargs,
) -> dict:
    """Conditional P_k with R-style queries (drawn from the random catalog).

    By default uses every random as a query. Pass ``n_query_subsample`` for
    a deterministic subsample.
    """
    n_r = int(np.asarray(ra_random).shape[0])
    if n_query_subsample is not None and n_query_subsample < n_r:
        from ._api import subsample_randoms
        idx = subsample_randoms(n_r, n_query_subsample, seed=subsample_seed)
        ra_q = np.asarray(ra_random)[idx]
        dec_q = np.asarray(dec_random)[idx]
        z_q = np.asarray(z_random)[idx]
    else:
        ra_q = np.asarray(ra_random)
        dec_q = np.asarray(dec_random)
        z_q = np.asarray(z_random)
    return measure_pk_conditional(
        ra_gal=ra_gal, dec_gal=dec_gal, z_gal=z_gal,
        ra_random=ra_random, dec_random=dec_random, z_random=z_random,
        ra_query=ra_q, dec_query=dec_q, z_query=z_q,
        query_type="R",
        exclude_self=False,
        **kwargs,
    )


def measure_pk_D(
    *,
    ra_gal: _ARR,
    dec_gal: _ARR,
    z_gal: _ARR,
    ra_random: _ARR,
    dec_random: _ARR,
    z_random: _ARR,
    n_query_subsample: Optional[int] = None,
    subsample_seed: int = 0,
    exclude_self: bool = True,
    **kwargs,
) -> dict:
    """Conditional P_k with D-style queries (drawn from the data catalog).

    Tracks the data-catalog index of each query so self-exclusion can be
    applied to N_q.
    """
    n_g = int(np.asarray(ra_gal).shape[0])
    if n_query_subsample is not None and n_query_subsample < n_g:
        from ._api import subsample_randoms
        idx = subsample_randoms(n_g, n_query_subsample, seed=subsample_seed)
    else:
        idx = np.arange(n_g, dtype=np.int64)
    ra_q = np.asarray(ra_gal)[idx]
    dec_q = np.asarray(dec_gal)[idx]
    z_q = np.asarray(z_gal)[idx]
    return measure_pk_conditional(
        ra_gal=ra_gal, dec_gal=dec_gal, z_gal=z_gal,
        ra_random=ra_random, dec_random=dec_random, z_random=z_random,
        ra_query=ra_q, dec_query=dec_q, z_query=z_q,
        query_type="D",
        exclude_self=exclude_self,
        query_indices=idx,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _need_mpl():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
        return plt
    except ImportError as e:
        raise ImportError("plotting helpers require matplotlib") from e


def _midpoints(edges: _ARR) -> _ARR:
    return 0.5 * (edges[:-1] + edges[1:])


def _marginalize(arr: _ARR, axis: int) -> _ARR:
    """Sum over an axis, treating NaN as 0 (for marginalising probabilities
    weighted by their denominator). Caller should pass weighted sums, not means.
    """
    return np.nansum(arr, axis=axis)


def plot_marginal_s(result: dict, *, ax=None, k_indices: Optional[Sequence[int]] = None,
                    z_bin: int = 0, center: bool = True):
    """P_k(s_tilde) marginalised over eta_tilde at one z_bin."""
    plt = _need_mpl()
    if ax is None:
        _, ax = plt.subplots()
    P = result["P"]                  # [k, s, e, z]
    denom = result["denominator"]
    # Recover joint sum F_num then marginalise: P(s) = sum_e F_num / sum_e denom.
    F_num = result["F"] * denom
    # Build P at k by computing F_k(s) - F_{k+1}(s); approximate via P*denom marginalisation.
    # Simpler: marginalise denom and (F-F_next)*denom separately; here we marginalise
    # F_num directly (per k) and rely on the per-bin denominator.
    s_e = result["s_tilde_edges"]
    s_med = result["s_med"]
    n_k = P.shape[0]
    if k_indices is None:
        k_indices = list(range(n_k))
    s_mid = _midpoints(s_e)
    for k_idx in k_indices:
        F_marg_num = _marginalize(F_num[k_idx, :, :, z_bin], axis=1)
        d_marg = _marginalize(denom[k_idx, :, :, z_bin], axis=1)
        # Next-k for P
        if k_idx + 1 < denom.shape[0]:
            F_next_num = _marginalize(
                result["F"][k_idx + 1, :, :, z_bin] * denom[k_idx + 1, :, :, z_bin], axis=1
            )
            d_next = _marginalize(denom[k_idx + 1, :, :, z_bin], axis=1)
        else:
            F_next_num = np.zeros_like(F_marg_num)
            d_next = np.zeros_like(d_marg)
        with np.errstate(divide="ignore", invalid="ignore"):
            F_marg = np.where(d_marg > 0, F_marg_num / d_marg, np.nan)
            F_next_marg = np.where(d_next > 0, F_next_num / d_next, np.nan)
        P_marg = F_marg - F_next_marg
        x = s_mid if center else s_mid + s_med[k_idx, z_bin]
        ax.plot(x, P_marg, marker="o", ms=3, label=f"k={result['k_values'][k_idx]}")
    ax.set_xlabel(r"$\tilde s_k$" if center else r"$s$")
    ax.set_ylabel(r"$P_k$ (marg. over $\tilde\eta$)")
    ax.set_title(f"{result['query_type']}-queries, z_bin={z_bin}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def plot_marginal_eta(result: dict, *, ax=None, k_indices: Optional[Sequence[int]] = None,
                      z_bin: int = 0, center: bool = True):
    """P_k(eta_tilde) marginalised over s_tilde at one z_bin."""
    plt = _need_mpl()
    if ax is None:
        _, ax = plt.subplots()
    P = result["P"]
    denom = result["denominator"]
    F_num = result["F"] * denom
    e_e = result["eta_tilde_edges"]
    eta_med = result["eta_med"]
    n_k = P.shape[0]
    if k_indices is None:
        k_indices = list(range(n_k))
    e_mid = _midpoints(e_e)
    for k_idx in k_indices:
        F_marg_num = _marginalize(F_num[k_idx, :, :, z_bin], axis=0)
        d_marg = _marginalize(denom[k_idx, :, :, z_bin], axis=0)
        if k_idx + 1 < denom.shape[0]:
            F_next_num = _marginalize(
                result["F"][k_idx + 1, :, :, z_bin] * denom[k_idx + 1, :, :, z_bin], axis=0
            )
            d_next = _marginalize(denom[k_idx + 1, :, :, z_bin], axis=0)
        else:
            F_next_num = np.zeros_like(F_marg_num)
            d_next = np.zeros_like(d_marg)
        with np.errstate(divide="ignore", invalid="ignore"):
            F_marg = np.where(d_marg > 0, F_marg_num / d_marg, np.nan)
            F_next_marg = np.where(d_next > 0, F_next_num / d_next, np.nan)
        P_marg = F_marg - F_next_marg
        x = e_mid if center else e_mid + eta_med[k_idx, z_bin]
        ax.plot(x, P_marg, marker="o", ms=3, label=f"k={result['k_values'][k_idx]}")
    ax.set_xlabel(r"$\tilde\eta_k$" if center else r"$\eta$")
    ax.set_ylabel(r"$P_k$ (marg. over $\tilde s$)")
    ax.set_title(f"{result['query_type']}-queries, z_bin={z_bin}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def plot_heatmap(result: dict, *, k_index: int, z_bin: int = 0, ax=None,
                 center: bool = True, cmap: str = "viridis"):
    """2-D heatmap P_k(s_tilde, eta_tilde) at fixed (k, z_bin)."""
    plt = _need_mpl()
    if ax is None:
        _, ax = plt.subplots()
    P = result["P"][k_index, :, :, z_bin]                     # (n_s, n_e)
    s_e = result["s_tilde_edges"]
    e_e = result["eta_tilde_edges"]
    if not center:
        s_e = s_e + result["s_med"][k_index, z_bin]
        e_e = e_e + result["eta_med"][k_index, z_bin]
    pcm = ax.pcolormesh(s_e, e_e, P.T, cmap=cmap, shading="auto")
    plt.colorbar(pcm, ax=ax, label=f"P_k (k={result['k_values'][k_index]})")
    ax.set_xlabel(r"$\tilde s_k$" if center else r"$s$")
    ax.set_ylabel(r"$\tilde\eta_k$" if center else r"$\eta$")
    ax.set_title(f"{result['query_type']}-queries, z_bin={z_bin}")
    return ax


def plot_dr_contrast(result_D: dict, result_R: dict, *, k_index: int, z_bin: int = 0,
                     ax=None, cmap: str = "RdBu_r"):
    """Heatmap of Delta P_k = P_k^D - P_k^R."""
    plt = _need_mpl()
    if ax is None:
        _, ax = plt.subplots()
    PD = result_D["P"][k_index, :, :, z_bin]
    PR = result_R["P"][k_index, :, :, z_bin]
    delta = PD - PR
    s_e = result_D["s_tilde_edges"]
    e_e = result_D["eta_tilde_edges"]
    vmax = float(np.nanmax(np.abs(delta))) if np.isfinite(delta).any() else 1.0
    pcm = ax.pcolormesh(s_e, e_e, delta.T, cmap=cmap, shading="auto",
                        vmin=-vmax, vmax=vmax)
    plt.colorbar(pcm, ax=ax, label=f"P_k^D - P_k^R (k={result_D['k_values'][k_index]})")
    ax.set_xlabel(r"$\tilde s_k$")
    ax.set_ylabel(r"$\tilde\eta_k$")
    ax.set_title(f"D - R contrast, z_bin={z_bin}")
    return ax


def plot_median_trends(result: dict, *, ax=None):
    """s_med(k) and eta_med(k) for each z_bin (one panel per quantity)."""
    plt = _need_mpl()
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    s_med = result["s_med"]
    eta_med = result["eta_med"]
    k_values = result["k_values"]
    n_zq = s_med.shape[1]
    for zb in range(n_zq):
        ax[0].plot(k_values, s_med[:, zb], marker="o", label=f"z_bin {zb}")
        ax[1].plot(k_values, eta_med[:, zb], marker="o", label=f"z_bin {zb}")
    ax[0].set_xlabel("k"); ax[0].set_ylabel(r"$s_\mathrm{med}(k)$"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("k"); ax[1].set_ylabel(r"$\eta_\mathrm{med}(k)$"); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
    ax[0].set_title(f"{result['query_type']}-queries")
    return ax


def plot_mark_response(result_marked: dict, result_unmarked: dict, *, k_index: int = 0,
                       z_bin: int = 0, ax=None):
    """Compare s_med, eta_med trends with vs without marks (Delta s, Delta eta)."""
    plt = _need_mpl()
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    k_values = result_marked["k_values"]
    ds = result_marked["s_med"][:, z_bin] - result_unmarked["s_med"][:, z_bin]
    de = result_marked["eta_med"][:, z_bin] - result_unmarked["eta_med"][:, z_bin]
    ax[0].plot(k_values, ds, marker="o")
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_xlabel("k"); ax[0].set_ylabel(r"$\Delta s_\mathrm{med}$"); ax[0].grid(alpha=0.3)
    ax[1].plot(k_values, de, marker="o")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xlabel("k"); ax[1].set_ylabel(r"$\Delta\eta_\mathrm{med}$"); ax[1].grid(alpha=0.3)
    return ax

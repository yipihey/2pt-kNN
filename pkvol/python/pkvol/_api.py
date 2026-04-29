"""High-level Python API on top of the Rust core (``pkvol._pkvol``)."""

from __future__ import annotations

from typing import Optional, Tuple, Union, Sequence

import numpy as np

from . import _pkvol


_ARR = np.ndarray


def _as_f64(x, name: str) -> np.ndarray:
    a = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {a.shape}")
    return a


def apply_marks(
    weights: Optional[_ARR],
    marks: Optional[_ARR],
    *,
    mode: str = "none",
    cut: Optional[float] = None,
    quantile: Optional[Tuple[float, float]] = None,
    n_galaxies: Optional[int] = None,
) -> np.ndarray:
    """Build effective per-galaxy weights from a mark mode.

    Parameters
    ----------
    weights
        Baseline observational weights; if ``None`` defaults to all ones
        (``n_galaxies`` must then be supplied).
    marks
        Mark values; required for all modes except ``"none"``.
    mode
        One of ``"none"``, ``"multiplicative"``, ``"threshold"``, ``"quantile"``.
    cut
        Threshold value (mode ``"threshold"`` only).
    quantile
        ``(q_lo, q_hi)`` rank interval in [0,1] (mode ``"quantile"`` only).
    n_galaxies
        Number of galaxies, used only when ``weights`` is ``None``.

    Returns
    -------
    np.ndarray of float64
    """
    if weights is None:
        if n_galaxies is None and marks is not None:
            n_galaxies = int(np.asarray(marks).shape[0])
        if n_galaxies is None:
            raise ValueError("either weights or n_galaxies must be provided")
        weights = np.ones(int(n_galaxies), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if mode == "none":
        return weights.copy()

    if marks is None:
        raise ValueError(f"mode '{mode}' requires marks")
    marks_arr = np.asarray(marks, dtype=np.float64)
    if marks_arr.shape != weights.shape:
        raise ValueError(
            f"marks shape {marks_arr.shape} != weights shape {weights.shape}"
        )

    if mode == "multiplicative":
        return weights * marks_arr
    if mode == "threshold":
        if cut is None:
            raise ValueError("threshold mode requires `cut`")
        return np.where(marks_arr > cut, weights, 0.0)
    if mode == "quantile":
        if quantile is None:
            raise ValueError("quantile mode requires `quantile`")
        q_lo, q_hi = quantile
        if not (0.0 <= q_lo <= q_hi <= 1.0):
            raise ValueError("quantile must satisfy 0 <= q_lo <= q_hi <= 1")
        n = marks_arr.size
        order = np.argsort(marks_arr, kind="stable")
        lo_i = int(np.floor(q_lo * n))
        hi_i = int(np.ceil(q_hi * n))
        keep = np.zeros(n, dtype=bool)
        keep[order[lo_i:hi_i]] = True
        return np.where(keep, weights, 0.0)
    raise ValueError(f"unknown mark mode: {mode!r}")


def subsample_randoms(
    n: int,
    k: int,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Reproducibly draw ``min(k, n)`` distinct sorted indices in ``[0, n)``."""
    return np.asarray(_pkvol.subsample_indices_py(int(n), int(k), int(seed)))


def measure_pk(
    ra_gal: _ARR,
    dec_gal: _ARR,
    z_gal: _ARR,
    weights_gal: Optional[_ARR],
    ra_query: _ARR,
    dec_query: _ARR,
    theta_edges: _ARR,
    z_edges: _ARR,
    k_values: _ARR,
    theta_max: float,
    *,
    z_query: Optional[_ARR] = None,
    query_weights: Optional[_ARR] = None,
    marks: Optional[_ARR] = None,
    mark_mode: str = "none",
    mark_cut: Optional[float] = None,
    mark_quantile: Optional[Tuple[float, float]] = None,
    z_intervals: Optional[Sequence[Tuple[int, int]]] = None,
    exclude_self: bool = False,
    angular_variable: str = "theta",
    backend: str = "ecdf",
    n_threads: Optional[int] = None,
    self_match_tol: float = 1e-9,
) -> dict:
    """Measure exact-count F_k and P_k.

    See ``README`` for scientific definitions. ``z_query`` is currently unused
    by the engine (queries live in observable angular space only) but accepted
    for API symmetry.

    Parameters
    ----------
    ra_gal, dec_gal, z_gal
        Galaxy catalog (radians, radians, dimensionless redshift).
    weights_gal
        Per-galaxy observational weight; ``None`` means unit weights.
    ra_query, dec_query
        Query positions (radians).
    theta_edges
        Angular thresholds (radians) at which K_q is evaluated.
    z_edges
        Redshift thresholds at which K_q is evaluated.
    k_values
        Counts at which P_k is evaluated. Must be sorted ascending.
    theta_max
        Maximum angular search radius (radians).
    query_weights
        Per-query weights for the F_k average; ``None`` means uniform.
    marks
        Optional galaxy marks combined into effective weights via :func:`apply_marks`.
    mark_mode, mark_cut, mark_quantile
        Mark configuration; see :func:`apply_marks`.
    z_intervals
        ``(l, r)`` index pairs into ``z_edges``; ``l = -1`` denotes
        "from -inf". Default: adjacent pairs.
    exclude_self
        Drop the central galaxy when a query coincides with one (within
        ``self_match_tol`` radians).
    angular_variable
        ``"theta"``, ``"theta2"``, or ``"omega"``. Sets the per-galaxy x_i.
    backend
        ``"ecdf"`` (sweep + Fenwick, default) or ``"histogram"``.
    n_threads
        Optional rayon thread count override.

    Returns
    -------
    dict with keys ``F``, ``P`` (shape ``[n_k, n_theta, n_intervals]``),
    ``k_values``, ``theta_edges``, ``z_edges``, ``z_intervals``,
    ``total_weight``, ``n_query_used``, ``mean_candidates``,
    ``mean_total_count``, ``query_valid_fraction``, ``angular_variable``,
    ``backend``.
    """
    del z_query  # accepted for API symmetry but not used in observable angular search
    ra_g = _as_f64(ra_gal, "ra_gal")
    dec_g = _as_f64(dec_gal, "dec_gal")
    z_g = _as_f64(z_gal, "z_gal")

    if weights_gal is None and marks is None:
        wg_eff = None
    else:
        baseline = (
            _as_f64(weights_gal, "weights_gal")
            if weights_gal is not None
            else np.ones(ra_g.size, dtype=np.float64)
        )
        wg_eff = apply_marks(
            baseline,
            marks,
            mode=mark_mode,
            cut=mark_cut,
            quantile=mark_quantile,
        ) if marks is not None or mark_mode != "none" else baseline

    ra_q = _as_f64(ra_query, "ra_query")
    dec_q = _as_f64(dec_query, "dec_query")
    qw = _as_f64(query_weights, "query_weights") if query_weights is not None else None

    th_e = _as_f64(theta_edges, "theta_edges")
    z_e = _as_f64(z_edges, "z_edges")
    k_v = _as_f64(k_values, "k_values")
    if not np.all(np.diff(k_v) > 0):
        raise ValueError("k_values must be strictly increasing")

    if z_intervals is not None:
        zi_arr = np.asarray(z_intervals, dtype=np.int64)
        if zi_arr.ndim != 2 or zi_arr.shape[1] != 2:
            raise ValueError("z_intervals must have shape (n_intervals, 2)")
    else:
        zi_arr = None

    return _pkvol.measure_pk_py(
        ra_g,
        dec_g,
        z_g,
        wg_eff,
        ra_q,
        dec_q,
        th_e,
        z_e,
        k_v,
        float(theta_max),
        zi_arr,
        qw,
        bool(exclude_self),
        float(self_match_tol),
        str(angular_variable),
        str(backend),
        None if n_threads is None else int(n_threads),
    )


def measure_random_query(
    ra_gal: _ARR,
    dec_gal: _ARR,
    z_gal: _ARR,
    ra_random: _ARR,
    dec_random: _ARR,
    theta_edges: _ARR,
    z_edges: _ARR,
    k_values: _ARR,
    theta_max: float,
    *,
    n_random_subsample: Optional[int] = None,
    random_subsample_seed: int = 0,
    weights_gal: Optional[_ARR] = None,
    **kwargs,
) -> dict:
    """Convenience wrapper: queries are a (subsampled) random catalog."""
    n_r = ra_random.shape[0]
    if n_random_subsample is not None and n_random_subsample < n_r:
        idx = subsample_randoms(n_r, n_random_subsample, seed=random_subsample_seed)
        ra_q = np.asarray(ra_random)[idx]
        dec_q = np.asarray(dec_random)[idx]
    else:
        ra_q = np.asarray(ra_random)
        dec_q = np.asarray(dec_random)
    return measure_pk(
        ra_gal,
        dec_gal,
        z_gal,
        weights_gal,
        ra_q,
        dec_q,
        theta_edges,
        z_edges,
        k_values,
        theta_max,
        exclude_self=False,
        **kwargs,
    )


def measure_data_query(
    ra_gal: _ARR,
    dec_gal: _ARR,
    z_gal: _ARR,
    theta_edges: _ARR,
    z_edges: _ARR,
    k_values: _ARR,
    theta_max: float,
    *,
    weights_gal: Optional[_ARR] = None,
    n_subsample: Optional[int] = None,
    subsample_seed: int = 0,
    exclude_self: bool = True,
    **kwargs,
) -> dict:
    """Convenience wrapper: queries are a (subsampled) galaxy catalog itself."""
    n = np.asarray(ra_gal).shape[0]
    if n_subsample is not None and n_subsample < n:
        idx = subsample_randoms(n, n_subsample, seed=subsample_seed)
        ra_q = np.asarray(ra_gal)[idx]
        dec_q = np.asarray(dec_gal)[idx]
    else:
        ra_q = np.asarray(ra_gal)
        dec_q = np.asarray(dec_gal)
    return measure_pk(
        ra_gal,
        dec_gal,
        z_gal,
        weights_gal,
        ra_q,
        dec_q,
        theta_edges,
        z_edges,
        k_values,
        theta_max,
        exclude_self=exclude_self,
        **kwargs,
    )


def measure_lambda(
    ra_random: _ARR,
    dec_random: _ARR,
    z_random: _ARR,
    ra_query: _ARR,
    dec_query: _ARR,
    theta_edges: _ARR,
    z_edges: _ARR,
    theta_max: float,
    *,
    weights_random: Optional[_ARR] = None,
    z_intervals: Optional[Sequence[Tuple[int, int]]] = None,
    angular_variable: str = "theta",
    backend: str = "ecdf",
    n_threads: Optional[int] = None,
) -> np.ndarray:
    """Per-query Lambda_q(u, z_interval): the weighted random count.

    Returned array shape: ``[n_query, n_theta, n_intervals]``.
    Use this to derive edge/completeness fractions via :func:`apply_edge_cut`.
    """
    ra_g = _as_f64(ra_random, "ra_random")
    dec_g = _as_f64(dec_random, "dec_random")
    z_g = _as_f64(z_random, "z_random")
    w = _as_f64(weights_random, "weights_random") if weights_random is not None else None
    ra_q = _as_f64(ra_query, "ra_query")
    dec_q = _as_f64(dec_query, "dec_query")
    th_e = _as_f64(theta_edges, "theta_edges")
    z_e = _as_f64(z_edges, "z_edges")
    if z_intervals is not None:
        zi_arr = np.asarray(z_intervals, dtype=np.int64)
    else:
        zi_arr = None
    return np.asarray(_pkvol.lambda_per_query_py(
        ra_g,
        dec_g,
        z_g,
        w,
        ra_q,
        dec_q,
        th_e,
        z_e,
        float(theta_max),
        zi_arr,
        str(angular_variable),
        str(backend),
        None if n_threads is None else int(n_threads),
    ))


def apply_edge_cut(
    lambda_q: np.ndarray,
    expected_lambda: Union[np.ndarray, float],
    f_min: float,
) -> np.ndarray:
    """Boolean mask of valid (query, theta, interval) triples.

    Returns an array of shape ``lambda_q.shape`` where True means the local
    completeness fraction ``Lambda_q / expected_lambda`` exceeds ``f_min``.
    """
    expected = np.asarray(expected_lambda, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(expected > 0.0, lambda_q / expected, 0.0)
    return ratio > float(f_min)

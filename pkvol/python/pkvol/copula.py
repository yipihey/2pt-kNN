"""Copula compression of P_k via shifted Legendre polynomials.

For each k, query type, redshift bin, mark we receive a 2D table
``P_k[u_a, ell]`` (ell indexes redshift shells). We treat each row/column as
empirical marginals and compute:

- angular marginal ``P_{k, Omega}(u) = sum_ell P_k(u, ell)``,
- radial marginal  ``P_{k, z}(ell)  = sum_a P_k(a, ell)``,
- the empirical copula ``C(U, V)`` via cumulative ranks,
- residual ``Delta C = C - U V``,
- low-mode shifted-Legendre coefficients ``a_mn``.

The default modes are ``(m, n) in {(1,1), (1,2), (2,1), (2,2), (1,3), (3,1)}``.

This is implemented in pure NumPy because it operates on small grids
``(n_theta, n_intervals)``. A Rust port is feasible later.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


DEFAULT_MODES: Sequence[Tuple[int, int]] = (
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
    (1, 3),
    (3, 1),
)


def shifted_legendre(n: int, x: np.ndarray) -> np.ndarray:
    """Shifted Legendre polynomial P_n(2x - 1), n >= 0, x in [0, 1].

    Returns an array the same shape as `x`. Normalization: orthonormal w.r.t.
    Lebesgue measure on [0, 1] -> we multiply by ``sqrt(2n + 1)``.
    """
    t = 2.0 * np.asarray(x, dtype=np.float64) - 1.0
    if n == 0:
        p = np.ones_like(t)
    elif n == 1:
        p = t
    else:
        p_prev = np.ones_like(t)
        p_curr = t
        for k in range(1, n):
            p_next = ((2.0 * k + 1.0) * t * p_curr - k * p_prev) / (k + 1.0)
            p_prev = p_curr
            p_curr = p_next
        p = p_curr
    return np.sqrt(2.0 * n + 1.0) * p


def legendre_basis_2d(
    nodes_u: np.ndarray,
    nodes_v: np.ndarray,
    modes: Sequence[Tuple[int, int]] = DEFAULT_MODES,
) -> np.ndarray:
    """Pre-computed 2D orthonormal Legendre basis on a `(u, v)` grid.

    Returns an array of shape ``(len(modes), len(nodes_u), len(nodes_v))``.
    """
    out = np.empty((len(modes), nodes_u.size, nodes_v.size), dtype=np.float64)
    for i, (m, n) in enumerate(modes):
        out[i] = np.outer(shifted_legendre(m, nodes_u), shifted_legendre(n, nodes_v))
    return out


def _empirical_marginal_cdf(values: np.ndarray) -> np.ndarray:
    """Cumulative-rank transform of a non-negative weight vector to [0, 1]."""
    arr = np.asarray(values, dtype=np.float64)
    total = arr.sum()
    if total <= 0.0:
        return np.linspace(0.0, 1.0, arr.size + 1)[1:]
    cum = np.cumsum(arr) / total
    return cum


def compute_copula_summary(
    p_table: np.ndarray,
    *,
    modes: Sequence[Tuple[int, int]] = DEFAULT_MODES,
) -> dict:
    """Copula compression of a single 2D `P` slice.

    Parameters
    ----------
    p_table
        ``P_k(u_a, ell)`` slice, non-negative, shape ``(n_u, n_ell)``.
    modes
        Iterable of ``(m, n)`` pairs (degrees in u and v respectively).

    Returns
    -------
    dict with keys

        - ``angular_marginal``: P_{k, u}, shape (n_u,)
        - ``radial_marginal``: P_{k, ell}, shape (n_ell,)
        - ``U``, ``V``: empirical-rank node positions in [0, 1]
        - ``copula``: empirical copula table C(U, V)
        - ``delta``: C - U V (residual)
        - ``coeffs``: dict mapping (m, n) -> Legendre coefficient
        - ``modes``: list of modes
    """
    p = np.asarray(p_table, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError("p_table must be 2D")
    n_u, n_v = p.shape

    angular = p.sum(axis=1)
    radial = p.sum(axis=0)
    u_nodes = _empirical_marginal_cdf(angular)
    v_nodes = _empirical_marginal_cdf(radial)
    total = p.sum()
    if total <= 0.0:
        copula = np.zeros_like(p)
    else:
        # Joint CDF on the lattice: cumulative along axis 0 then axis 1, normalized.
        cum_u = np.cumsum(p, axis=0)
        cum_uv = np.cumsum(cum_u, axis=1) / total
        copula = cum_uv

    uv = np.outer(u_nodes, v_nodes)
    delta = copula - uv

    # Compute Legendre coefficients via discrete inner products with weights
    # equal to (du, dv) where du, dv are the differences of the rank-CDF nodes.
    du = np.diff(np.r_[0.0, u_nodes])
    dv = np.diff(np.r_[0.0, v_nodes])
    weights = np.outer(du, dv)

    basis = legendre_basis_2d(u_nodes, v_nodes, modes)
    coeffs = {}
    for i, mn in enumerate(modes):
        coeffs[mn] = float((basis[i] * delta * weights).sum())

    return {
        "angular_marginal": angular,
        "radial_marginal": radial,
        "U": u_nodes,
        "V": v_nodes,
        "copula": copula,
        "delta": delta,
        "coeffs": coeffs,
        "modes": list(modes),
    }


def coefficients_for_pk(
    p_volume: np.ndarray,
    *,
    modes: Sequence[Tuple[int, int]] = DEFAULT_MODES,
) -> dict:
    """Apply :func:`compute_copula_summary` to each k-slice of a P_k cube.

    Parameters
    ----------
    p_volume
        Array of shape ``(n_k, n_u, n_ell)``.

    Returns
    -------
    dict with keys
        - ``coeffs``: ndarray shape ``(n_k, n_modes)``
        - ``modes``: list[(m, n)]
        - ``per_k``: list of full :func:`compute_copula_summary` outputs.
    """
    arr = np.asarray(p_volume, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError("p_volume must be 3D (n_k, n_u, n_ell)")
    n_k = arr.shape[0]
    coeffs = np.zeros((n_k, len(modes)), dtype=np.float64)
    per_k = []
    for j in range(n_k):
        s = compute_copula_summary(arr[j], modes=modes)
        per_k.append(s)
        for i, mn in enumerate(modes):
            coeffs[j, i] = s["coeffs"][mn]
    return {"coeffs": coeffs, "modes": list(modes), "per_k": per_k}

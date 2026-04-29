"""Clustered toy: P_k for a clumpy catalog vs Poisson.

Builds a Cox-style cluster catalog (parents drawn uniformly, children
poisson-clustered around each parent on the sphere) and shows that the
high-k tail of P_k is enhanced relative to a uniform Poisson catalog of the
same total count.
"""

from __future__ import annotations

import numpy as np

import pkvol


def cox_catalog(rng, n_parent: int, mean_children: float, sigma: float, L: float):
    """Tiny tangent-plane Cox process on a square patch of side L."""
    parent_ra = rng.uniform(0, L, n_parent)
    parent_dec = rng.uniform(-L / 2, L / 2, n_parent)
    n_children = rng.poisson(mean_children, n_parent)
    ra = []
    dec = []
    for i in range(n_parent):
        if n_children[i] == 0:
            continue
        ra.append(parent_ra[i] + rng.normal(0, sigma, n_children[i]))
        dec.append(parent_dec[i] + rng.normal(0, sigma, n_children[i]))
    ra = np.concatenate(ra) if ra else np.empty(0)
    dec = np.concatenate(dec) if dec else np.empty(0)
    inside = (ra > 0) & (ra < L) & (dec > -L / 2) & (dec < L / 2)
    return ra[inside], dec[inside]


def main() -> None:
    rng = np.random.default_rng(0)
    L = 0.4

    ra_c, dec_c = cox_catalog(rng, n_parent=200, mean_children=40, sigma=0.01, L=L)
    n_gal = ra_c.size
    z_c = rng.uniform(0.0, 1.0, n_gal)
    print(f"clustered: N_gal = {n_gal}")

    # Matched-density Poisson catalog.
    ra_p = rng.uniform(0, L, n_gal)
    dec_p = rng.uniform(-L / 2, L / 2, n_gal)
    z_p = rng.uniform(0.0, 1.0, n_gal)

    # Random query points.
    n_query = 5_000
    margin = 0.05
    ra_q = rng.uniform(margin, L - margin, n_query)
    dec_q = rng.uniform(-L / 2 + margin, L / 2 - margin, n_query)

    theta_max = 0.005
    theta_edges = np.array([theta_max])
    z_edges = np.array([1.0])
    z_intervals = np.array([[-1, 0]])
    k_values = np.arange(1, 11, dtype=float)

    print("\nP_k(clustered) vs P_k(poisson):")
    print(f"{'k':>3} {'cluster':>10} {'poisson':>10} {'ratio':>8}")
    for cat_name, ra, dec, zc in [("clustered", ra_c, dec_c, z_c), ("poisson", ra_p, dec_p, z_p)]:
        out = pkvol.measure_pk(
            ra, dec, zc, None, ra_q, dec_q, theta_edges, z_edges, k_values,
            theta_max=theta_max + 1e-9, z_intervals=z_intervals,
        )
        if cat_name == "clustered":
            P_clust = out["P"][:, 0, 0]
            mean_clust = out["mean_total_count"]
        else:
            P_pois = out["P"][:, 0, 0]
            mean_pois = out["mean_total_count"]
    print(f"  mean K  cluster={mean_clust:.2f}  poisson={mean_pois:.2f}")
    for kj, pc, pp in zip(k_values, P_clust, P_pois):
        ratio = pc / max(pp, 1e-10)
        print(f"{int(kj):>3} {pc:>10.4f} {pp:>10.4f} {ratio:>8.2f}")


if __name__ == "__main__":
    main()

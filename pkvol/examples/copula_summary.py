"""Compute copula summary coefficients for a measured P_k cube."""

from __future__ import annotations

import numpy as np

import pkvol
from pkvol.copula import coefficients_for_pk, DEFAULT_MODES


def main() -> None:
    rng = np.random.default_rng(42)
    L = 0.5
    n_gal = 12_000
    ra = rng.uniform(0, L, n_gal)
    dec = rng.uniform(-L / 2, L / 2, n_gal)
    z = rng.uniform(0.0, 1.0, n_gal)

    n_query = 4_000
    margin = 0.05
    ra_q = rng.uniform(margin, L - margin, n_query)
    dec_q = rng.uniform(-L / 2 + margin, L / 2 - margin, n_query)

    theta_edges = np.array([0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05])
    z_edges = np.linspace(0.0, 1.0, 11)
    k_values = np.array([1.0, 2.0, 4.0, 8.0])

    out = pkvol.measure_pk(
        ra, dec, z, None, ra_q, dec_q, theta_edges, z_edges, k_values,
        theta_max=0.06,
    )
    print("F shape:", out["F"].shape)
    print("P shape:", out["P"].shape)

    summary = coefficients_for_pk(out["P"])
    print("\nLegendre coefficients (k x mode):")
    header = "  k  " + "  ".join(f"({m},{n})" for m, n in summary["modes"])
    print(header)
    for j, k in enumerate(k_values):
        row = " ".join(f"{c:+.3e}" for c in summary["coeffs"][j])
        print(f"k={int(k):2d}  {row}")


if __name__ == "__main__":
    main()

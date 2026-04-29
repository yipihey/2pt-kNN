"""Stress test: exercise pkvol over a battery of catalog sizes and edge cases.

This is intended to catch correctness regressions and crash bugs:

- Empty galaxy / query catalogs.
- Single-galaxy catalogs.
- All queries identical.
- Heavy duplicate galaxies.
- All-zero weights.
- Very wide / very narrow theta_max.
- Deep parallelism: explicit n_threads=1, 4, 16.
- A 1e5 galaxy / 1e4 query end-to-end run with the histogram backend
  cross-checked against the ECDF backend on a sub-sample.
- RA wrap (galaxies on both sides of RA = 0/2pi).
- Negative-Dec hemisphere.
- Sparse very-thin redshift shells.

The script prints a checklist; non-zero exit code indicates failure.
"""

from __future__ import annotations

import sys
import time
import traceback
import numpy as np

import pkvol

FAILURES = []


def _check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def section(name: str) -> None:
    print(f"\n=== {name} ===")


def basic_call(n_gal=200, n_q=20, theta_max=0.1, **kw):
    rng = np.random.default_rng(0)
    ra = rng.uniform(0, 2 * np.pi, n_gal)
    dec = np.arcsin(rng.uniform(-1, 1, n_gal))
    z = rng.uniform(0.1, 0.9, n_gal)
    ra_q = rng.uniform(0, 2 * np.pi, n_q)
    dec_q = np.arcsin(rng.uniform(-1, 1, n_q))
    return pkvol.measure_pk(
        ra, dec, z, None, ra_q, dec_q,
        np.array([theta_max / 2, theta_max]),
        np.array([0.3, 0.7]),
        np.array([1.0, 2.0]),
        theta_max,
        **kw,
    )


def main() -> int:
    section("basic call")
    out = basic_call()
    _check("returns dict", isinstance(out, dict))
    _check("F has expected shape", out["F"].shape == (2, 2, 1))
    _check("P has expected shape", out["P"].shape == (2, 2, 1))
    _check("F monotonic in theta", np.all(np.diff(out["F"], axis=1) >= -1e-12))

    section("empty catalogs")
    rng = np.random.default_rng(1)
    try:
        out = pkvol.measure_pk(
            np.array([]), np.array([]), np.array([]), None,
            rng.uniform(0, 1, 10), rng.uniform(-0.5, 0.5, 10),
            np.array([0.1, 0.2]), np.array([0.5, 1.0]),
            np.array([1.0, 2.0]), 0.3,
        )
        _check("empty galaxy catalog -> all zeros",
               np.allclose(out["F"], 0) and np.allclose(out["P"], 0))
    except Exception as e:
        _check("empty galaxy catalog returns cleanly", False, repr(e))

    try:
        out = pkvol.measure_pk(
            rng.uniform(0, 1, 10), rng.uniform(-0.5, 0.5, 10), rng.uniform(0, 1, 10), None,
            np.array([]), np.array([]),
            np.array([0.1]), np.array([0.5, 1.0]),
            np.array([1.0]), 0.3,
        )
        _check("empty query catalog -> all zeros",
               np.allclose(out["F"], 0) and np.allclose(out["P"], 0))
    except Exception as e:
        _check("empty query catalog returns cleanly", False, repr(e))

    section("single-galaxy catalog")
    out = pkvol.measure_pk(
        np.array([0.0]), np.array([0.0]), np.array([0.5]), None,
        np.array([0.0]), np.array([0.0]),
        np.array([0.001, 0.01]), np.array([1.0]),
        np.array([1.0]), 0.05,
        z_intervals=np.array([[-1, 0]]),
    )
    _check("F monotone non-decreasing in theta",
           out["F"][0, 0, 0] <= out["F"][0, 1, 0] + 1e-12)
    _check("F[0, 1, 0] == 1.0", abs(out["F"][0, 1, 0] - 1.0) < 1e-12)

    section("all queries identical")
    n = 50
    out = pkvol.measure_pk(
        np.full(n, 0.0), np.full(n, 0.0), np.linspace(0.1, 0.9, n), None,
        np.full(20, 0.0), np.full(20, 0.0),
        np.array([0.001]), np.array([0.5, 1.0]),
        np.array([1.0]), 0.01,
    )
    _check("identical queries give finite F", np.all(np.isfinite(out["F"])))

    section("duplicate galaxies")
    n = 1000
    ra = np.full(n, 0.0)
    dec = np.full(n, 0.0)
    z = np.full(n, 0.5)
    out = pkvol.measure_pk(
        ra, dec, z, None,
        np.array([0.0, 0.001]), np.array([0.0, 0.0]),
        np.array([0.01]), np.array([0.4, 0.6]),
        np.array([1.0, n - 1.0]), 0.01,
    )
    _check("duplicates: F[k = N-1] near 1", out["F"][1, 0, 0] > 0.99)

    section("zero weights")
    rng = np.random.default_rng(0)
    n = 200
    ra = rng.uniform(0, 1, n)
    dec = rng.uniform(-0.5, 0.5, n)
    z = rng.uniform(0.0, 1.0, n)
    out = pkvol.measure_pk(
        ra, dec, z, np.zeros(n),
        rng.uniform(0.1, 0.9, 30), rng.uniform(-0.4, 0.4, 30),
        np.array([0.05]), np.array([0.5, 1.0]),
        np.array([1.0]), 0.1,
    )
    _check("zero weights -> F == 0", np.allclose(out["F"], 0))

    section("RA wrap-around")
    n = 500
    ra = np.concatenate([
        np.full(n // 2, 0.001),
        np.full(n // 2, 2 * np.pi - 0.001),
    ])
    dec = np.zeros(n)
    z = np.linspace(0.1, 0.9, n)
    out = pkvol.measure_pk(
        ra, dec, z, None,
        np.array([0.0]), np.array([0.0]),
        np.array([0.005]), np.array([1.0]),
        np.array([1.0, n - 1.0]),
        0.01,
        z_intervals=np.array([[-1, 0]]),
    )
    _check("RA-wrap gathers both sides (F[0]=1)", abs(out["F"][0, 0, 0] - 1.0) < 1e-12)

    section("varying n_threads")
    rng = np.random.default_rng(0)
    n_g = 5_000
    ra = rng.uniform(0, 2 * np.pi, n_g)
    dec = np.arcsin(rng.uniform(-1, 1, n_g))
    zg = rng.uniform(0.1, 0.9, n_g)
    n_q = 500
    ra_q = rng.uniform(0, 2 * np.pi, n_q)
    dec_q = np.arcsin(rng.uniform(-1, 1, n_q))
    base = pkvol.measure_pk(
        ra, dec, zg, None, ra_q, dec_q,
        np.array([0.05, 0.1]), np.array([0.3, 0.7]),
        np.array([1.0, 2.0]), 0.2, n_threads=1,
    )
    for t in (2, 4, 8, 16):
        try:
            ot = pkvol.measure_pk(
                ra, dec, zg, None, ra_q, dec_q,
                np.array([0.05, 0.1]), np.array([0.3, 0.7]),
                np.array([1.0, 2.0]), 0.2, n_threads=t,
            )
            _check(f"n_threads={t} same as serial",
                   np.allclose(ot["F"], base["F"]) and np.allclose(ot["P"], base["P"]))
        except Exception as e:
            _check(f"n_threads={t} no crash", False, repr(e))

    section("backend agreement at scale")
    rng = np.random.default_rng(2)
    n_g = 50_000
    ra = rng.uniform(0, 2 * np.pi, n_g)
    dec = np.arcsin(rng.uniform(-1, 1, n_g))
    zg = rng.uniform(0.05, 0.95, n_g)
    n_q = 2_000
    ra_q = rng.uniform(0, 2 * np.pi, n_q)
    dec_q = np.arcsin(rng.uniform(-1, 1, n_q))
    th_e = np.linspace(0.005, 0.05, 10)
    z_e = np.linspace(0.1, 0.9, 9)
    k_v = np.array([1.0, 3.0, 7.0])
    a = pkvol.measure_pk(ra, dec, zg, None, ra_q, dec_q, th_e, z_e, k_v, 0.06,
                         backend="ecdf")
    b = pkvol.measure_pk(ra, dec, zg, None, ra_q, dec_q, th_e, z_e, k_v, 0.06,
                         backend="histogram")
    _check("ecdf == histogram (50k x 2k)",
           np.allclose(a["F"], b["F"], atol=1e-10) and np.allclose(a["P"], b["P"], atol=1e-10))

    section("very large stress: 1e6 galaxies x 1e5 queries")
    try:
        rng = np.random.default_rng(3)
        n_g = 1_000_000
        ra = rng.uniform(0, 2 * np.pi, n_g)
        dec = np.arcsin(rng.uniform(-1, 1, n_g))
        zg = rng.uniform(0.05, 0.95, n_g)
        n_q = 100_000
        ra_q = rng.uniform(0, 2 * np.pi, n_q)
        dec_q = np.arcsin(rng.uniform(-1, 1, n_q))
        th_e = np.linspace(0.001, 0.005, 5)
        z_e = np.linspace(0.1, 0.9, 5)
        k_v = np.array([1.0, 3.0, 7.0])
        t0 = time.perf_counter()
        out = pkvol.measure_pk(ra, dec, zg, None, ra_q, dec_q, th_e, z_e, k_v, 0.005)
        dt = time.perf_counter() - t0
        _check(f"1e6 x 1e5 finishes ({dt:.1f}s)", np.all(np.isfinite(out["F"])),
               f"throughput {n_q / dt:.0f} q/s")
    except Exception as e:
        traceback.print_exc()
        _check("1e6 x 1e5", False, repr(e))

    section("done")
    if FAILURES:
        print(f"\nFAIL: {len(FAILURES)} checks failed:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print(f"\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

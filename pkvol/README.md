# pkvol

**Metric-agnostic exact-count P_k galaxy statistics in observable space.**

`pkvol` measures the exact-count angular-redshift kNN-style statistic

```
P_k^X(u; z_-, z_+) = F_k^X(u; z_-, z_+) - F_{k+1}^X(u; z_-, z_+)
```

from a galaxy catalog `D` and a query catalog `Q` using only observable
coordinates (RA, Dec, redshift, weights, marks). No comoving distance
conversion is performed.

The kernel is

```
K_q(u, z) = sum_i w_i 1[ DeltaOmega(q, i) <= u ] 1[ z_i <= z ]
```

with shell counts

```
K_q(u; z_-, z_+) = K_q(u, z_+) - K_q(u, z_-)
```

and CDF

```
F_k^X(u; z_-, z_+) = P_{q in Q_X}[ K_q(u; z_-, z_+) >= k ].
```

`X = R` uses random-catalog query points (volume-weighted); `X = D` uses
galaxy query points (galaxy-weighted, with optional self-exclusion).

## Installation

The package is a Rust core with PyO3 bindings, built via maturin.

```bash
cd pkvol
maturin build --release
pip install dist/pkvol-*.whl
# or, in a virtualenv:
maturin develop --release
```

For pure-Rust use add `pkvol = { path = "pkvol" }` to your `Cargo.toml`.

## Quickstart

```python
import numpy as np
import pkvol

ra_gal  = ...  # (N_gal,) radians
dec_gal = ...  # (N_gal,) radians
z_gal   = ...  # (N_gal,)

ra_q  = ...    # (N_query,)
dec_q = ...    # (N_query,)

theta_edges = np.linspace(0.005, 0.05, 10)  # radians
z_edges     = np.linspace(0.1, 0.9, 9)
k_values    = np.arange(1, 8, dtype=float)

out = pkvol.measure_pk(
    ra_gal, dec_gal, z_gal, None,
    ra_q, dec_q,
    theta_edges, z_edges, k_values,
    theta_max=0.06,
    angular_variable="theta",  # or "theta2", "omega"
    backend="ecdf",            # or "histogram"
)
out["F"]  # shape (n_k, n_theta, n_intervals)
out["P"]  # shape (n_k, n_theta, n_intervals)
```

For random-query / data-query versions:

```python
out_R = pkvol.measure_random_query(ra_gal, dec_gal, z_gal,
                                   ra_random, dec_random,
                                   theta_edges, z_edges, k_values,
                                   theta_max=0.06,
                                   n_random_subsample=20000,
                                   random_subsample_seed=42)

out_D = pkvol.measure_data_query(ra_gal, dec_gal, z_gal,
                                 theta_edges, z_edges, k_values,
                                 theta_max=0.06,
                                 n_subsample=20000,
                                 subsample_seed=42,
                                 exclude_self=True)
```

## Marks

```python
# multiplicative numeric weights
out = pkvol.measure_pk(..., marks=phi, mark_mode="multiplicative")

# threshold mark m > cut
out = pkvol.measure_pk(..., marks=mass, mark_mode="threshold", mark_cut=1e10)

# top quartile by mark
out = pkvol.measure_pk(..., marks=color, mark_mode="quantile",
                       mark_quantile=(0.75, 1.0))
```

## Edge / completeness fractions

```python
lambda_q = pkvol.measure_lambda(ra_random, dec_random, z_random,
                                ra_q, dec_q, theta_edges, z_edges,
                                theta_max=0.06)
expected = lambda_q.mean(axis=0)
mask = pkvol.apply_edge_cut(lambda_q, expected, f_min=0.8)
```

## Copula compression

```python
from pkvol.copula import coefficients_for_pk
summary = coefficients_for_pk(out["P"])
summary["coeffs"]  # shape (n_k, len(modes))
summary["modes"]   # default: [(1,1),(1,2),(2,1),(2,2),(1,3),(3,1)]
```

## Algorithms

- **Angular search**: KD-tree on 3D unit vectors with chord (Euclidean)
  radius queries, exactly equivalent to a haversine angular ball.
  RA periodicity is handled implicitly by the unit-vector mapping.
- **Local 2D ECDF (default)**: sweep-line over angular threshold u with a
  Fenwick tree on z buckets; cost `O(m log n_z + n_u n_z)` per query.
- **Aggregation**: per-query streaming reduction in rayon — never stores
  the per-query K_q matrix beyond a thread-local buffer.

## Layout

```
pkvol/
  Cargo.toml         pyproject.toml
  src/
    lib.rs           angular_search.rs   ecdf2d.rs
    haversine.rs     pk_aggregate.rs     shell_counts.rs
    marks.rs         randoms.rs          py_bindings.rs
  python/pkvol/      _api.py copula.py sedist.py
  tests/             integration_pk.rs
  python_tests/      test_basic.py test_marks_randoms.py test_copula_sedist.py
  examples/          poisson_null_test.py clustered_demo.py
                     mark_perturbation.py copula_summary.py
                     random_vs_data_query.py
  benchmarks/        bench_pk.py stress_test.py
  docs/              DESIGN.md
```

## Tests, examples, benchmarks

```bash
# Rust core tests
cd pkvol
cargo test --no-default-features

# Python tests
maturin develop --release   # or build + install the wheel
python -m pytest python_tests -v

# Examples
python examples/poisson_null_test.py
python examples/clustered_demo.py
python examples/mark_perturbation.py
python examples/random_vs_data_query.py
python examples/copula_summary.py

# Benchmarks (1e6 gal x 1e5 queries in ~0.5s on a recent CPU)
python benchmarks/bench_pk.py --full
python benchmarks/stress_test.py
```

## License

MIT.

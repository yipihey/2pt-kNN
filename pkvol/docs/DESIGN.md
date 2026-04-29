# pkvol Design Document

## Goals

`pkvol` implements the *exact-count* angular-redshift kNN-CDF statistic
`P_k^X(u; z_-, z_+) = F_k^X - F_{k+1}^X` directly in observable space (RA,
Dec, redshift). The package is intentionally metric-agnostic: it does not
convert to comoving distances or assume a cosmological model.

The library is a Rust core with PyO3 bindings packaged via maturin. The
core is parallelized over query points with rayon and uses streaming
aggregation so that intermediate per-query K-matrices never need to be
materialized for the whole catalog.

## Core algorithm

```text
for each query q in parallel:
    candidates = AngularTree.query_radius(q, theta_max)
    extract local arrays (x_i, y_i, w_i)   # x = theta | theta^2 | 1-cos(theta)
    K[u_a, z_b] = sweep+Fenwick(xs, ys, ws, u_edges, z_edges)
    K_shell[u_a, ell] = K[u_a, z_r(ell)] - K[u_a, z_l(ell)]
    for each k in k_values + (k_max + 1):
        F_acc[k, u_a, ell] += w_q * 1[K_shell[u_a, ell] >= k]
F = F_acc[k_values] / total_query_weight
P = F - F_acc[k_values + 1] / total_query_weight
```

## Modules

| module             | purpose                                                          |
|--------------------|------------------------------------------------------------------|
| `haversine.rs`     | great-circle distance, RA/Dec ↔ unit vector, chord helpers       |
| `angular_search.rs`| KD-tree on 3D unit vectors with squared-chord radius queries     |
| `ecdf2d.rs`        | sweep-line + Fenwick + histogram backends                        |
| `shell_counts.rs`  | K(u, z_+) - K(u, z_-) finite differences                         |
| `pk_aggregate.rs`  | rayon-parallel streaming reduction; F_k, P_k assembly            |
| `marks.rs`         | effective-weight construction (none/mult/threshold/quantile)     |
| `randoms.rs`       | reproducible deterministic subsampling                           |
| `py_bindings.rs`   | PyO3 entry points (gated by the `python` feature)                |

Each Rust module is self-contained and unit-tested. The Python layer
(`python/pkvol/_api.py`, `copula.py`, `sedist.py`) provides the
numpy-friendly public API and the copula compression / SEDist-style
compressed CDF objects.

## Angular search

The KD-tree is built over 3D unit vectors (`u = (cos d cos r, cos d sin r,
sin d)`) and uses the bijection `chord(theta) = 2 sin(theta / 2)` to map
an angular radius into a Euclidean radius in 3-space. RA wrap-around is
intrinsic to the unit-vector mapping.

The implementation is a small, recursive median-split tree with
axis-aligned bounding boxes for pruning:

- `build` selects the longest-extent axis at each node and uses
  `select_nth_unstable_by` for an O(N log N) build.
- `query_radius` recurses with bounding-box squared-distance pruning,
  reporting all original-index points within the chord radius.

This keeps us free of `bosque` / `kiddo` / `kd-tree` dependency churn and
fits in ~150 LOC. For high-cardinality cases the constant factor is
competitive with established crates because the tree is contiguous and
contains no hash-map indirections.

## 2D weighted ECDF

For each query we extract `m` local points `(x_i, y_i, w_i)` and want

```text
K(u_a, z_b) = sum_i w_i * 1[x_i <= u_a] * 1[y_i <= z_b]
```

at every `(u_a, z_b)`.

Algorithm (`ecdf2d_sweep`):

1. Sort points ascending in `x` (index permutation, points untouched).
2. Bucket each `y_i` to the smallest index `b` such that `z_edges[b] >=
   y_i` (`partition_point`). Drop points with `y_i > z_edges.last()`.
3. Sweep `u_a` ascending; for each `u_a`, advance the pointer through the
   sorted-by-x array, adding `w_i` to the Fenwick tree at bucket `b_i`.
4. For each `b`, read `prefix(b + 1)` from the Fenwick tree → `K(u_a, z_b)`.

Cost: `O(m log n_z + n_u n_z log n_z)` per query. The `n_u n_z log n_z`
term dominates only when `n_z` is large; for typical galaxy-survey grids
(`n_u ~ 20`, `n_z ~ 50`) this is < 1 µs per query in the inner loop.

A naive 2D-histogram backend (`ecdf2d_histogram`) is included for
cross-validation and for the rare regime where the histogram cost is
favorable.

## Streaming aggregation

`measure_pk` accumulates `1[K_shell >= k]` into a per-thread buffer of
shape `(n_k + 1, n_u, n_int)` (the extra `+1` carries `F_{k_max + 1}` so
that `P_k = F_k - F_{k+1}` is computable for the largest requested k).
After all queries, the per-thread accumulators are summed (one
allocation per thread) and divided by the total query weight.

This keeps memory bounded at `n_threads * (n_k + 1) * n_u * n_int * 8
bytes`, independent of `n_query`.

## Marks

The Rust core consumes a single per-galaxy "effective weight" array
`w_eff_i = w_i * phi(m_i)`. The Python layer (`apply_marks`) builds this
vector from the four supported mark modes:

- `none`: identity.
- `multiplicative`: `w_eff = w * phi(m)`.
- `threshold`: `w_eff = w` if `m > cut` else `0`.
- `quantile`: `w_eff = w` for galaxies with `marks` rank in `[q_lo, q_hi]`,
  else `0`.

This keeps the core small while supporting easy comparison of P_k across
mark configurations from Python.

## Query types

- `measure_pk`: low-level — caller supplies query positions.
- `measure_random_query`: queries are a (subsampled) random catalog.
- `measure_data_query`: queries are a (subsampled) galaxy catalog with
  optional self-exclusion.
- `measure_lambda`: per-query `Lambda_q(u, z_interval)` selection counts
  used by `apply_edge_cut` for completeness masking.

Self-exclusion is performed within the Rust core: when a query is co-
located (within `self_match_tol` radians) with a galaxy, that galaxy is
dropped from the local `(x, y, w)` array.

## Reproducibility

`subsample_randoms(n, k, seed)` uses a `ChaCha8Rng` seeded by `seed` and
returns sorted indices. Identical `(n, k, seed)` triples always yield the
same indices. The Rust unit tests assert this.

## Copula compression

Implemented in pure NumPy in `python/pkvol/copula.py`. For each
`P_k(u, ell)` slice:

1. Marginals `P_{k, u} = sum_ell P_k(u, ell)` and
   `P_{k, ell} = sum_u P_k(u, ell)`.
2. Empirical-rank node positions `U_a = CDF_u(a)`, `V_l = CDF_ell(l)`.
3. Empirical copula `C(U_a, V_l) = (cumsum_2d P) / total`.
4. Residual `Delta C = C - U V`.
5. Low-mode shifted Legendre coefficients `a_mn` via discrete inner
   products with weights `du dv` (rank-CDF gaps).

Default modes: `(1,1), (1,2), (2,1), (2,2), (1,3), (3,1)`.

## SEDist (compressed CDF)

`pkvol.CompressedCdf1D` and `pkvol.CompressedCdf2D` provide light-weight
piecewise-linear / bilinear-interpolation wrappers with
`from_samples` constructors and `to_dict` / `from_dict` serialization.
These are intended to receive the rich-output marginals and joint CDFs
produced by larger pipelines (e.g. SEDist-style compression).

## Performance

On a stock CPU, with default settings:

| N_gal   | N_query | theta_max | mean K | time (ECDF) |
|---------|---------|-----------|--------|-------------|
| 50 k    | 5 k     | 0.01 rad  | ~ 60   | 0.07 s      |
| 100 k   | 10 k    | 0.005 rad | ~ 30   | 0.07 s      |
| 1 M     | 50 k    | 0.001 rad | ~ 12   | 0.5 s       |
| 1 M     | 100 k   | 0.005 rad | varies | ~ 0.4 s     |

These numbers come from `benchmarks/bench_pk.py` and
`benchmarks/stress_test.py`.

## Testing matrix

- 18 Rust unit tests — haversine math, KD-tree vs brute force, ECDF
  sweep vs brute force, marks, randoms, shell counts.
- 4 Rust integration tests — end-to-end measure_pk vs brute force,
  ECDF vs histogram backend, self-exclusion, Poisson-mean sanity.
- 19 Python pytests — measure_pk vs brute force, backends agree, angular
  variables agree, haversine periodicity, query-radius round-trip,
  self-exclusion, P telescoping, mark modes, lambda_per_query vs brute
  force, copula independence vs dependence, compressed CDF round-trips.
- A standalone stress-test script exercising empty catalogs, single-
  galaxy catalogs, identical queries, duplicate galaxies, zero weights,
  RA wrap, varying thread counts, and a 1e6-galaxy / 1e5-query end-to-end
  run.

## Future work

- Native Rust copula module (port of `pkvol/python/pkvol/copula.py`).
- Optional bosque integration for very large catalogs.
- Persistent SEDist serialization compatible with the upstream
  https://github.com/yipihey/SEDist project.
- Multiple parallel mark-weight columns in a single core call (current
  Python wrapper does this by calling the core multiple times).

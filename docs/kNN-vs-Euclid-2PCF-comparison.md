# kNN Ladder vs. Euclid 2PCF-GC: Detailed Comparison

**Abel et al. (2026) vs. Euclid Collaboration: de la Torre et al. (2025, A&A 700, A78)**

---

## 1. What the Euclid pipeline actually does

The Euclid 2PCF-GC is a production-grade, heavily validated C++11 code for estimating ξ(r⊥, r∥), ξ(r, μ), ξ_ℓ(r), and w⊥(r⊥) from ~20–30 million Hα emitters at 0.9 < z < 1.8. Key design facts:

- **Three independent pair-counting backends**: linked-list (chained mesh), kd-tree (dual-tree, sliding-midpoint splits, leaf size 100), and octree (hashed Morton codes, dual-tree). All three are exact — no approximation at the pair-counting level. Having three provides cross-validation.

- **Landy–Szalay estimator** with DD, DR, RR pair counts (Eq. 2 of the paper), plus cross-correlation variants (Eq. 9) and the "modified" estimator for BAO reconstruction (Eq. 10). Also supports the two-random-catalog case for systematic subtraction.

- **Random split technique** (Keihänen et al. 2019): splits the M = N_R/N_d = 50× random catalog into N_S = 50 sub-catalogs, each ≈ N_d in size. RR is computed as the average over sub-catalog auto-pairs. Reduces wall-clock RR cost by ~10× for 5 × 10⁶ data points.

- **Binning**: linear or log₁₀ bins in r; the anisotropic measurement uses 200 μ-bins × 40 r-bins (or similar). Multipoles via Riemann sums over ξ(r, μ).

- **Line-of-sight conventions**: midpoint (default), bisector, endpoint — all implemented and tested.

- **Parallelization**: OpenMP shared-memory. Scales well to 32 threads; SDCs can provide up to 128.

- **Validation**: CoxMock (isotropic Cox process with known analytic ξ) for accuracy; ELM Pinocchio mocks for runtime; Flagship N-body for end-to-end. Passed six maturity gates within the SGS.

- **Runtime at Euclid scale** (from Fig. 3): for N_d = 5 × 10⁶ on 32 threads with the random-split option, the full multipole measurement takes ~7 CPU-hours (~30 min wall-clock for a single z-bin at DR1 density). Without random split: ~4 days. The octree is marginally fastest at large N_d; linked-list wins at N_d < 5 × 10⁵.

- **Scaling**: linked-list goes as N_d² exactly; tree-based methods as N_d^1.96 for N_d > 10⁵.

- **Compared to Corrfunc**: 2PCF-GC is 3–5× slower than Corrfunc v2.5.3 at N_d ~ 10⁶ (Fig. 4), attributed to Corrfunc's AVX/SSE vectorization and cell-pair optimizations. The Euclid code prioritizes portability and SGS integration over raw speed.

- **Output**: ξ(r) and the raw pair counts (DD, DR, RR). Nothing else. All downstream products — covariance matrices, non-Gaussian diagnostics, counts-in-cells, density-split statistics — require separate computations, separate codes, and separate computational budgets.


## 2. The central distinction: one query, many statistics

The key framing for the paper is not "we compute ξ(r) faster" — it is that **the same kNN tree queries that yield ξ(r) via the Landy–Szalay estimator simultaneously deliver an entire suite of clustering statistics**, all from a single computational pass:

| Statistic | From the kNN queries | Euclid pipeline |
|-----------|---------------------|-----------------|
| ξ(r) via Landy–Szalay | D^DD·D^RR/[D^DR]² pair-count densities | ✓ (primary output) |
| Var[ξ̂(r)] | Scatter across dilution subsamples | Requires 1000+ mock runs |
| kNN-CDFs (all k) | Direct output of the tree query | Not computed |
| Counts-in-cells P(N\|V) | From kNN-CDF differences (Eq. 4) | Separate code/computation |
| Void probability function | P₀ = 1 − CDF₁(r) | Separate code/computation |
| σ²_V(R) (two-point) | From measured ξ̂ via sphere-overlap kernel | From measured ξ̂ (same formula) |
| σ²_NL(R) (full nonlinear) | Var[N(<R)] across query points, model-free | Separate code/computation |
| α_SN(V) scale-dependent | From Δ_k residuals (Eq. 49) | Single global parameter from jackknife |
| Density-split clustering | Post-stratify by octant density label | Separate code/computation |
| Excursion-set trajectories | Nested octant density sequence | Not available |
| σ²_{1/V}[k] → σ(M) | Var of Lagrangian density at fixed k | Not available |
| Non-Poissonian residuals Δ_k | CDF_measured − CDF_Poisson\|ξ | Not available |
| Factorial cumulants C_j(V) | From weighted sums of Δ_k | Not available |

Every row below the first is free — no additional tree builds, no additional queries, no additional computational cost beyond what was already spent to measure ξ(r). The Euclid pipeline would need separate processing functions, separate mock suites, and separate computational allocations for each of these. This is the argument: not that any single operation is dramatically faster, but that a single set of O(N log N) tree queries — computing the Landy–Szalay estimator via kNN pair-count densities — replaces an entire ecosystem of separate measurements.


## 3. Honest complexity comparison

Both approaches use tree-accelerated algorithms. The complexity comparison must reflect this.

### 3a. Tree-accelerated pair counting (Euclid)

The Euclid dual-tree traversal has cost O(N log N) to navigate the tree and locate relevant leaf pairs, plus O(N_shell) actual pair evaluations at the leaves for each separation bin. The total per-query cost is O(log N + N_shell), so the full DR cost is:

    DR_Euclid = O(N_R × (log N_d + N_shell(r)))

At small r, N_shell ~ O(1) and the tree traversal dominates: effectively O(N_R log N_d). At large r (say 200 h⁻¹ Mpc, 1 Mpc bins), N_shell ~ n̄ × 4πr²Δr ~ 10⁴, and the shell processing dominates. Tree acceleration eliminates the cost of *finding* the shell, but you still process every pair in it.

### 3b. kNN query (our approach)

Each query point finds its k_max = 8 nearest neighbors via a single tree descent: cost O(log N_d) per query, independent of separation. The total DR cost is:

    DR_kNN = O(N_R × log N_d) per dilution level

The kNN cost per query point does not grow with r. The dilution ladder shifts which scales are probed at each level, but the per-query cost remains O(log N_d).

### 3c. RR term

For realistic surveys with completeness weights, the RR term is **not** analytic — it requires a random-to-random kNN query:

    RR_kNN = O(N_R × log N_R)

The analytic Erlang result applies only to unweighted uniform/periodic boxes (simulation volumes). For Euclid-like data, both approaches require a tree computation for RR. The kNN version scales as O(N_R log N_R); the Euclid split-random approach costs roughly M × O(N_d × (log N_d + N_shell)). The kNN approach wins particularly at large r where N_shell dominates the split-random cost, but it is not zero cost.

### 3d. The honest comparison table

| Operation | Euclid (tree-accelerated LS) | kNN ladder |
|-----------|------------------------------|------------|
| DD | O(N_d × (log N_d + N_shell)) | O(N_d log N_d) per level |
| DR | O(N_R × (log N_d + N_shell)) | O(N_R log N_d) per level |
| RR (unweighted periodic) | O(N_R²) or split-random | Analytic (Erlang) |
| RR (weighted survey) | Split-random: M × O(N_d (log N_d + N_shell)) | O(N_R log N_R) |

The distinction is that the kNN per-query cost is O(log N) independent of separation (fixed k = 8), while the pair-count per-query cost is O(log N + N_shell(r)), where N_shell grows as r² at large separations. At small r where N_shell ~ O(1), both methods have essentially the same scaling. The kNN advantage opens up at large r — which is also where the dilution ladder operates and where cosmologically interesting scales (BAO, linear bias, growth rate) live.

### 3e. What you get for the cost

Even if the wall-clock speedup on ξ(r) alone were modest at small separations (and it may be substantial at large r — benchmarks needed), the kNN computation simultaneously delivers the entire statistical suite listed in Section 2. The Euclid pipeline's ~7 CPU-hours buys you ξ(r) and pair counts. Our computation — using the same Landy–Szalay estimator, realized through kNN pair-count densities — buys you ξ(r), its variance, the kNN-CDFs, counts-in-cells, the VPF, σ²_NL(R), σ(M) for the Press–Schechter mass function, α_SN(V), density-split clustering, and excursion-set trajectories — all at once, from the same tree queries.


## 4. Where Euclid's approach is competitive or superior

### 4a. The 2D (r⊥, r∥) measurement with fine binning

Euclid measures ξ(r⊥, r∥) on a 200 × 40 grid simultaneously. Standard pair counting bins each pair into (r⊥, r∥) at the time of the distance computation — a single pass. The kNN approach requires either:

- The λ-metric trick (Sec. 8 of the paper), which stretches the LOS to capture pairs with small r⊥ but large π. This works, but each λ value requires a separate query pass, and the optimal λ depends on π_max/r_p,max.
- Large k_max to capture enough neighbors per query point to populate 200 μ-bins.

For the specific case of Euclid's high-resolution 2D grid, conventional pair counting with cylindrical shells may remain competitive, particularly with Corrfunc-type vectorized codes.

### 4b. Fine angular (μ) binning for multipoles

The Euclid pipeline computes ξ(r, μ) in 200 μ-bins and integrates for multipoles. Each pair contributes to exactly one (r, μ) cell. The kNN Legendre-weighted approach works well for ξ_ℓ(s), but each k-th neighbor contributes a single μ value. With k_max = 8 neighbors per query point, only 8 (s, μ) samples are obtained per query. For 200 μ-bins, the effective sampling is sparse unless many dilution levels contribute.

**Resolution**: the Legendre weighting accumulates the multipole moment continuously (no μ-binning needed — v_j = L_ℓ(μ_qj) per neighbor), and the sum over k ensures each pair at separation s is counted once. The μ-binning concern is a non-issue for multipoles computed via direct Legendre weighting. It would only matter if one wanted the full 2D ξ(r, μ) map.

### 4c. Production maturity and validation depth

The Euclid code has:

- Three independent pair-counting implementations cross-checked against each other
- Six maturity gates within the ESA SGS framework
- Validation on CoxMock (analytic truth), ELM (Pinocchio), and Flagship (full N-body)
- Demonstrated Corrfunc comparison (Fig. 4)
- Known, characterized accuracy relative to the Euclid 10% (of statistical error) requirement
- CI/CD pipeline (CODEEN), version control, distributed deployment across 9 SDCs

Our pipeline is at the formalism + notes stage. The validation section (Sec. 10) is a plan, not results. The code/README.md lists planned modules. This is the most important gap to close.

### 4d. Cross-correlation and modified estimators

Euclid implements the full cross-correlation estimator (Eq. 9: D₁D₂ − D₁R₂ − R₁D₂ + R₁R₂) and the modified estimator for BAO reconstruction (Eq. 10: DD − 2DS + SS, with an auxiliary "shifted" catalog S). Our paper focuses on the auto-correlation case. The kNN cross-correlation extension exists in the formalism (Banerjee & Abel 2021b), but the modified estimator and the two-random-catalog variants need explicit development.


## 5. What we learn from this paper

### 5a. The random-split technique as a benchmark target

Keihänen et al. (2019) showed that splitting the M = 50 random catalog into 50 sub-catalogs and averaging the RR sub-counts reduces runtime by >10× with no bias or variance penalty. Our paper should compare against this split-random cost — not naive O(N_R²). With the split, the RR baseline is M tree-accelerated auto-correlations of N_d-sized sub-catalogs:

    RR_split ~ M × O(N_d × (log N_d + N_shell))

Our O(N_R log N_R) kNN query still wins (particularly at large r where N_shell dominates), but the margin is smaller than vs. unsplit RR. The paper should be explicit about this.

### 5b. The LC (Linear Construction) trick for covariance

Keihänen et al. (2022) showed that the covariance of the LS estimator can be decomposed by powers of 1/M (where M = N_R/N_d). By computing the covariance at M = 1 and M = 2, one can linearly construct the covariance for arbitrary M — a 14× speedup for M = 50 with 2 Mpc/h bins. This is specific to mock-based covariance estimation.

Our kNN framework has a natural analog that is worth stating explicitly: the DR variance is analytically known — it is the binomial variance of the empirical CDF (Eq. in Sec. 6.2 of the paper), Var[CDF^DR_k(r)] = CDF(1−CDF)/N_R. The M-dependence of the estimator variance is therefore predicted without needing the LC extrapolation at all. This is a genuine advantage: where LC requires two separate mock measurements to characterize the M-dependence, the kNN framework predicts it from the functional form of the CDF variance. The paper should state this clearly as a point of comparison.

Our dilution-ladder variance is complementary to both LC and mock-based covariance. It does **not** replace mock-based covariance estimation — mocks remain essential for pipeline validation, model-dependent covariance at different parameter points, and systematic error characterization. Whatever per-catalog speedup the kNN method provides applies equally to the mock runs. The dilution-ladder variance is an independent, data-driven cross-check, not a substitute.

### 5c. Accuracy requirements are stringent: 10% of statistical error

Euclid formally requires that systematic errors in the 2PCF estimate (from finite randoms, estimator bias, etc.) be less than 10% of the statistical uncertainty. This means M = 50 is not optional — it's driven by a formal ESA requirement. Our paper should state clearly what accuracy the kNN estimator achieves relative to this benchmark, i.e., at what k_max and N_R/N_d does the kNN estimator satisfy ≤10% systematic-to-statistical ratio.

### 5d. CoxMock-type validation with known analytic truth

The CoxMock suite (isotropic line-point Cox process) provides catalogs with a known analytic ξ(r) — a damped power-law with index −2. This is a cleaner test than comparing to Corrfunc on N-body mocks (where both methods have their own noise). Consider adding a CoxMock validation to our Test 1, or designing an equivalent test with a known generating process.

### 5e. Pair line-of-sight definitions matter

The Euclid paper carefully distinguishes midpoint, bisector, and endpoint definitions of the pair LOS (Eq. 15, Fig. 1). Our anisotropic section (Sec. 8) uses the plane-parallel approximation with a note about wide-angle corrections via the pair-specific LOS. For a paper targeting Euclid/DESI-level surveys, explicitly implementing and testing the midpoint vs. bisector definitions would strengthen the case for drop-in replacement.

### 5f. Kerscher et al. (2022) — quasi-Monte Carlo and DRshell estimator

Also worth reading: Kerscher et al. (2022) developed improved-accuracy estimators using quasi-Monte Carlo methods for the RR term and a "DRshell" estimator for edge corrections. These represent additional optimizations in the pair-counting ecosystem that our paper should be aware of. The quasi-Monte Carlo RR approach achieves faster convergence than Poisson randoms, potentially reducing the required N_R/N_d ratio — which is relevant to our claim about reduced dependence on massive random catalogs (Sec. 6d).

### 5g. Two distinct σ² quantities — be precise

The paper currently defines two different variance quantities that must not be conflated:

- **σ²_V(r₀)** from the generating function (Eq. in Sec. 7.1): this is the volume-averaged two-point function, computed from the measured ξ̂(r) via the sphere-overlap weight W(r|r₀). It enters the Poisson|ξ prediction and depends on the generating-function truncation at n = 2.
- **σ²_NL(R₀)** from the kNN counts in spherical tophats (Eq. in Sec. 9): this is the full nonlinear density variance, measured model-free from Var[N(<R₀)] across all query points with Poisson shot noise subtracted. No Gaussianity assumption, no generating function truncation.

Both are available from the same tree query, but they are conceptually distinct. σ²_V is a two-point quantity (derived from ξ); σ²_NL is the all-orders answer. Their difference measures the contribution of connected N-point functions with N ≥ 3 to the variance. The paper should be precise about this distinction — it's a concrete example of the "more than two-point information" that the kNN framework provides.


## 6. What the Euclid pipeline could learn from us

### 6a. A unified measurement framework

The deepest difference is architectural. The Euclid SGS has separate processing functions for the 2PCF, the 3PCF (Veropalumbo et al., in prep.), the power spectrum and bispectrum (Salvalaggio et al., in prep.), and presumably separate codes for counts-in-cells, void statistics, and covariance estimation. Each requires its own tree build, its own pair/tuple counting, its own computational allocation.

The kNN framework unifies these through two complementary structures built from the same data:

**The dilution ladder** uses random, volume-filling partitions of the data. Each dilution level needs its own tree build on a subset of N_d/R_ℓ points that span the full survey volume. These yield ξ(r) at successively larger scales, with overlap consistency checks and empirical variance from the scatter across subsamples.

**The spatial octant decomposition** uses the in-place KD-tree structure of the full-data tree. After 3ℓ levels of splitting, the array contains 8^ℓ contiguous blocks that tile the volume into approximate octants. This provides density labels, the nested variance hierarchy, density-split clustering, and excursion-set trajectories via post-stratification of the global query results. Crucially, the ξ measurement uses the global tree — neighbors are found across octant boundaries. The octant labels are metadata for stratification, not search constraints, so there is no boundary bias.

These two structures are separate and complementary: the ladder gives ξ(r) and Var[ξ̂] at all scales; the spatial decomposition gives environmental diagnostics and additional statistical products. Together with the raw kNN query output (distance distributions from every query point to its k nearest neighbors), they provide the full suite of statistics listed in Section 2. The tree query is the expensive step; everything else is analytic manipulation of the query output.

### 6b. Scale-dependent α_SN(V) from kNN residuals

The Δ_k residuals between the measured kNN-CDFs and the Poisson|ξ prediction yield the excess variance δσ²_N(V) at every sphere radius, providing α_SN(V) as a measured, scale-dependent function. This replaces the single-parameter α_SN fit in RascalC — the least constrained ingredient in the current semi-analytical covariance pipeline — with a direct, data-driven measurement. If validated, this could improve Euclid's covariance accuracy for both BAO and full-shape analyses.

### 6c. Continuous ξ(r) without bin choice

Eliminates a nuisance parameter in the measurement. Particularly valuable at the BAO scale, where the bin width affects the sharpness of the acoustic feature.

### 6d. Reduced dependence on massive random catalogs

If the kNN estimator can achieve Euclid's accuracy requirement with smaller N_R/N_d (because the DR term converges faster in the kNN framework), the entire pipeline becomes lighter. The random catalog generation, storage, and I/O are significant costs at Euclid scale.


## 7. Concrete recommendations for the paper

1. **Lead with the unified-output story.** The paper should not be framed as "we compute ξ(r) faster." It should be framed as: "The Landy–Szalay estimator, realized through kNN pair-count densities, simultaneously delivers ξ(r), its variance, kNN-CDFs, counts-in-cells, VPF, σ²_NL(R), σ(M), α_SN(V), density-split clustering, and excursion-set diagnostics from a single set of tree queries. Standard pipelines require separate computations for each."

2. **Fix the complexity comparison.** Both approaches are tree-accelerated. The honest distinction is O(log N) per query (kNN, independent of r) vs. O(log N + N_shell(r)) per query (pair counting, growing as r² at large separations). Present the corrected table (Section 3d above). Do not claim O(N²) vs. O(N log N).

3. **Be explicit that RR is O(N_R log N_R) for weighted surveys**, not zero. The Erlang analytic result is restricted to unweighted periodic boxes. For Euclid-like data with completeness weights, both approaches require a tree computation for RR; the kNN version scales better (particularly at large r) but is not free.

4. **Benchmark at Euclid scale**: N_d = 5 × 10⁶, N_R = 2.5 × 10⁸. Compare wall-clock time against Corrfunc and (ideally) the published Euclid runtimes in Fig. 3 of the A&A paper. Report the total cost for the full suite (ξ + variance + diagnostics), not just ξ alone — this is where the amortized advantage is largest.

5. **Add a CoxMock-like known-truth validation test** in addition to the Poisson and N-body tests already planned.

6. **Implement and test the cross-correlation estimator** (D₁D₂/D₁R₂ ratio, kNN analog). Required for multi-tracer analyses and BAO reconstruction.

7. **Explicitly state the accuracy vs. k_max and N_R/N_d** for the Euclid 10% requirement. Show convergence plots.

8. **Acknowledge the Euclid pipeline's maturity** and frame the kNN approach as a unified measurement engine that produces the same ξ(r) plus a suite of additional statistics, compatible with the existing downstream infrastructure (covariance models, window corrections, parameter estimation).

9. **The α_SN(V) story is the scientific headline.** The computational story is important but secondary. What makes this paper more than a speed improvement is that the same computation that gives ξ(r) also directly measures the non-Poissonian and non-Gaussian structure that enters the covariance — the hardest ingredient to model analytically.

10. **Frame mock runs correctly.** The kNN approach does not eliminate the need for mock catalogs. Mocks are essential for pipeline validation, model-dependent covariance, and systematic error characterization. The kNN speedup per catalog applies equally to mock runs. The dilution-ladder variance and the α_SN(V) diagnostic are complementary, data-driven cross-checks — not replacements for mock-based covariance. The paper should say this explicitly to avoid overselling.

11. **Double-check the factorial cumulant derivation.** The corrected identity (from session notes) is δ⟨N^(2)⟩ = 2 Σ_{k≥2} (k−1) Δ_k, with weight (k−1) per Δ_k reflecting the binomial coefficient structure — not the simpler 2 Σ_{k≥2} Δ_k. This is currently correct in sec_nonpoisson.tex (Eq. 43) but should be verified against independent calculation. Getting this wrong would invalidate the α_SN(V) extraction.

12. **Distinguish σ²_V from σ²_NL in the text.** The paper defines both σ²_V(r₀) (two-point, from ξ̂ via sphere-overlap kernel W(r|r₀)) and σ²_NL(R₀) (full nonlinear, from Var[N(<R₀)] with shot-noise subtraction). Both come from the same tree query, but they are conceptually distinct. Their difference measures connected N ≥ 3 contributions. This distinction should be highlighted as a concrete example of beyond-two-point information.

13. **Promote the kNN Landy–Szalay estimator to the primary equation.** The current paper presents the DD/DR ratio (Eq. 14, Davis–Peebles) as the boxed primary estimator and relegates the full LS form in kNN language to a parenthetical paragraph: "If one wishes to retain the full LS correction..." (Eq. 15, D^DD · D^RR / [D^DR]²). This must be inverted. The community is deeply committed to LS — it is the standard estimator in DESI, Euclid, and every major survey pipeline. Presenting Davis–Peebles as the primary estimator and LS as an optional add-on will create immediate resistance, regardless of the formal equivalence in the N_R → ∞ limit. The restructuring should be:

    - **Eq. 15 (LS in kNN language) becomes the primary, boxed estimator.** Write it in both the Hamilton form and the standard LS form:

          1 + ξ̂_LS^kNN(r) = D^DD(r) · D^RR(r) / [D^DR(r)]²     (Hamilton)

          ξ̂_LS^kNN(r) = [D^DD(r) − 2D^DR(r) + D^RR(r)] / D^RR(r)  (Landy–Szalay)

      where D^DD, D^DR, D^RR are the summed kNN pair-count densities reconstructed from the DD, DR, RR kNN-CDFs respectively.

    - **State that all results in the paper use this LS estimator.** Every statistic — ξ(r), ξ_ℓ(s), w_p(r_p), and the Δ_k residuals — should be presented as computed via the kNN realization of LS.

    - **Note the DD/DR simplification as a limit.** For N_R ≫ N_d, the RR term can be computed analytically (Erlang for uniform boxes) or from a smaller random catalog, and the estimator reduces to the DD/DR ratio (Davis–Peebles). This is a computational optimization for large N_R, not the definition of the estimator.

    - **For weighted surveys, the RR kNN query is required anyway** (O(N_R log N_R)), so there is no additional cost to using the full LS form instead of Davis–Peebles. The RR term is already being computed; using it in the estimator improves the variance at no extra expense.

    This framing makes the paper a drop-in replacement for existing LS pipelines rather than a proposal to switch estimators.

14. **Add σ_{1/V}[k]: the Press–Schechter mass function from kNN distances.** This is a new statistical product that belongs in the unified-output table (Section 2) and deserves its own subsection in the paper, likely in Sec. 9 (Implementation) or as an addition to Sec. 7 (Non-Poissonian diagnostics).

    The idea: at a given dilution level, each query point's k-th nearest neighbor distance r_k defines a sphere of volume V_k = (4/3)πr_k³. The number of data points inside that sphere is exactly k (by construction). The "Lagrangian density" in that sphere is

        δ_L(k) = k / (n̄ V_k) − 1

    The variance of this quantity across all query points, at fixed k, defines

        σ²_{1/V}[k] ≡ Var_q[k / (n̄ V_k(q))]

    This is a direct, model-free measurement of the density variance smoothed at the mass scale M = ρ̄ V_k. Under linear bias (b), the galaxy σ² maps to the matter σ² via σ²_matter = σ²_galaxy / b², and the resulting σ(M) is exactly the quantity that enters the Press–Schechter mass function and its extensions (Sheth–Tormen, excursion set).

    Key properties:
    - Available at every k from 1 to k_max, and at every dilution level, giving σ(M) as a continuous function of M spanning the full range from individual halos to cluster scales.
    - No smoothing kernel choice — the sphere radius is defined by the data (the k-th neighbor distance), not imposed externally. Different k values probe different effective smoothing scales.
    - The derivative d ln σ²/d ln M, which enters the mass function, is well-determined because σ²(M) is densely sampled.
    - Combined with the excursion-set trajectories from the octant hierarchy (Sec. 9), this provides both ingredients of the excursion-set mass function — σ(M) and the barrier-crossing statistics — measured directly from the galaxy distribution.

    This connects the kNN framework to the halo mass function literature and gives the paper reach beyond the correlation-function community. It should appear in the statistics table and in the paper body.


## Summary table

| Aspect | Euclid 2PCF-GC | kNN Ladder |
|--------|----------------|------------|
| Estimator | Landy–Szalay (exact) | Landy–Szalay via kNN pair-count densities (D^DD·D^RR/[D^DR]²) |
| DD cost | O(N_d (log N_d + N_shell)) | O(N_d log N_d) per level |
| DR cost | O(N_R (log N_d + N_shell)) | O(N_R log N_d) per level |
| RR (unweighted periodic) | O(N_R²) or split-random | Analytic (Erlang) |
| RR (weighted survey) | Split-random: M × O(N_d (log N_d + N_shell)) | O(N_R log N_R) |
| Cost distinction | Per-query: O(log N + N_shell(r)); grows with r | Per-query: O(log N); independent of r |
| Binning | Required (linear or log) | Continuous; post-hoc if desired |
| Anisotropic | ξ(r⊥,r∥), ξ(r,μ), ξ_ℓ(r) | ξ_ℓ(s) via Legendre weighting; w_p via λ-metric |
| Cross-correlation | Full (Eq. 9) + modified (Eq. 10) | Auto only (cross-corr. formalism exists) |
| LOS conventions | Midpoint, bisector, endpoint | Plane-parallel + wide-angle note |
| **Output from one computation** | **ξ(r) + pair counts** | **ξ(r) + Var[ξ̂] + kNN-CDFs + CiC + VPF + σ²_V + σ²_NL + σ(M) + α_SN(V) + DSC + Δ_k + C_j(V) + excursion sets** |
| Covariance | External (mocks, LC trick, RascalC) | Dilution-ladder variance + scale-dependent α_SN(V) |
| Parallelization | OpenMP, 32–128 threads | Embarrassingly parallel (per-query) |
| Maturity | Production (6 gates, 3 backends, CI/CD) | Formalism + planned code |
| Runtime (N_d = 5×10⁶, 32 thr) | ~7 CPU-hr (ξ only, with split-random) | TBD — but this buys the full suite |

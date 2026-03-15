# Session Notes for Next Session

## Current State
- Paper v6: 33 pages, 11 section files, compiles cleanly
- Working notes on Euclid comparison in notes_euclid_comparison.tex
- GitHub repo structure prepared in /home/claude/2pt-from-kNN/ (needs updating with latest paper)

## Outstanding Corrections Needed

### 1. Computational Cost Table (sec_discussion.tex, §10.4)
Current table still has issues:
- RR row should show O(N_R · N_shell) with tree acceleration, NOT O(N_R^2)
- Euclid uses split-random trick: RR cost ~ M × O(N_d · N_shell), comparable to DD
- Need to read full text of Euclid 2501.16555 for actual wall-clock numbers
- Need to read full text of Keihänen et al. 2022 (2205.11852) for LC method details

### 2. Linear Construction (LC) Trick — What We Understand
- LC decomposes covariance of LS estimator by powers of 1/M (M = N_R/N_d)
- Compute covariance at M=1 and M=2, construct arbitrary M by linear combination
- 14× speedup for M=50 with 2 Mpc/h bins
- Specific to mock-based covariance estimation
- Our kNN analog: DR variance is analytically known (binomial CDF variance),
  so the M-dependence is predicted without needing LC extrapolation
- This is a genuine advantage worth stating clearly

### 3. Mock Runs — Correct Framing
- Mock catalogs (EZmocks, PINOCCHIO, Abacus) are essential for:
  - Pipeline validation
  - Model-dependent covariance at different parameter points
  - Systematic error characterization
- kNN speedup per catalog applies to mock runs equally
- Dilution-ladder variance is complementary diagnostic, NOT replacement
- Already partially corrected in v6 but should be reviewed

### 4. RR with Weights
- Already corrected in v6: "analytic only for uniform unweighted boxes"
- For weighted surveys: O(N_R log N_R) via kNN query
- But should compare to Euclid's actual RR cost with split randoms + tree accel

## Key Papers to Read in Full Next Session
1. **Euclid 2501.16555** (de la Torre et al. 2025, A&A 700, A78)
   - 2PCF estimation methodology and software
   - kd-tree, octree, linked-list implementations
   - Wall-clock performance numbers
   - OpenMP parallelization details
   - Validation on mock catalogs

2. **Keihänen et al. 2022** (2205.11852, A&A 666, A129)
   - Linear Construction method derivation
   - How covariance decomposes by M
   - Exact formulas for the M=1, M=2 construction
   - Speedup factors and accuracy validation

3. **Kerscher et al. 2022** (A&A 2022)
   - Improved accuracy estimators for pair counts
   - Quasi-Monte Carlo methods for RR
   - DRshell estimator for edge corrections

## Decisions Made During This Session

### Architecture
- Dilution ladder uses random, volume-filling partitions (NOT spatial octants)
- Each dilution level needs its own tree build
- Spatial octant decomposition of full tree is SEPARATE — provides density labels,
  nested variance, DSC, excursion set trajectories via post-stratification
- Boundary correction: xi measurement uses global tree; octant labels are metadata

### Anisotropic Measurements
- Single tree build serves all lambda values (query-time metric parameter)
- bosque tree topology unchanged by z → z/λ (rank-order invariant)
- Ellipsoidal generating function: V = (4/3)π r³ λ, W → W_λ
- Multipoles via Legendre value-weighting, survival corrections cancel

### Scale-Dilution Matching
- No special variance treatment needed at any level
- Small scales: few subsamples but huge number of independent spatial volumes
- Large scales: many subsamples (4096 at level 4) with ample degrees of freedom

### Nonlinear σ²(R)
- From kNN counts in spherical tophats: model-free, no Gaussianity assumption
- Distinct from σ_V² extracted via Eqs 20-21 (which assumes generating function truncation)
- Both available from same tree query

### Factorial Cumulant Identity (corrected)
- δ⟨N^(2)⟩ = 2 Σ_{k≥2} (k-1) Δ_k  [NOT 2 Σ_{k≥2} Δ_k]
- Weight (k-1) per Δ_k reflects binomial coefficient structure
- Derivation via identity: Σ_{k≥1} C(k-1, j-1) CDF_kNN = ⟨N^(j)⟩/j!

## File Inventory
- main.tex — master file with bibliography (34 references)
- sec_introduction.tex — 33 lines
- sec_background.tex — 70 lines
- sec_estimator.tex — 113 lines
- sec_weights.tex — 110 lines
- sec_ladder.tex — 82 lines (corrected: random volume-filling partitions)
- sec_errors.tex — 87 lines (corrected: RR cost, mock framing)
- sec_nonpoisson.tex — 232 lines (corrected: factorial cumulant derivation)
- sec_anisotropic.tex — 94 lines (NEW: lambda metric, ellipsoidal genfunc)
- sec_implementation.tex — 90 lines (NEW: tree reuse, spatial decomposition, DSC)
- sec_validation.tex — 102 lines (6 tests, figure plan)
- sec_discussion.tex — 69 lines (needs: cost table revision)
- notes_euclid_comparison.tex — 118 lines (working notes, not compiled)

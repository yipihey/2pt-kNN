#!/usr/bin/env node
// Programmatic test for the WASM pipeline.
//
// Usage:
//   # Rebuild + test (from project root):
//   wasm-pack build --target nodejs --out-dir pkg-node --no-default-features --features=wasm && node test_wasm.mjs
//
//   # Quick re-run (no rebuild):
//   node test_wasm.mjs
//
//   # Verbose output:
//   node test_wasm.mjs --verbose

import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const { run_validation_wasm } = require('./pkg-node/twopoint.js');

const verbose = process.argv.includes('--verbose');

// "Quick" preset: matches web frontend defaults
// CoxMock validation physics: box=500, line=200, m=10
const config = {
    n_mocks: 3,
    n_points: 5000,
    n_lines: 500,
    line_length: 200.0,
    box_size: 500.0,
    k_max: 16,
    n_bins: 15,
    r_min: 3.0,
    r_max: 37.0,   // 0.8 × r_char(k=16, nbar=4e-5) ≈ 36.6
    random_ratio: 5,
    max_dilution_level: 1,
    use_gpu: false,
};

let failures = 0;

function assert(cond, msg) {
    if (!cond) {
        console.error(`  FAIL: ${msg}`);
        failures++;
    } else if (verbose) {
        console.log(`  ok: ${msg}`);
    }
}

function assertClose(a, b, tol, msg) {
    const diff = Math.abs(a - b);
    if (diff > tol) {
        console.error(`  FAIL: ${msg} — got ${a}, expected ${b} (diff=${diff.toExponential(2)}, tol=${tol})`);
        failures++;
    } else if (verbose) {
        console.log(`  ok: ${msg} (${a})`);
    }
}

async function main() {
    console.log('=== WASM Pipeline Test ===');
    console.log(`Config: n_points=${config.n_points}, n_mocks=${config.n_mocks}, k_max=${config.k_max}`);

    const t0 = performance.now();
    const d = await run_validation_wasm(config);
    const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
    console.log(`Completed in ${elapsed}s\n`);

    // --- Structure checks ---
    console.log('1. Structure checks');
    assert(Array.isArray(d.r_centers), 'r_centers is array');
    assert(d.r_centers.length === config.n_bins, `r_centers has ${config.n_bins} bins (got ${d.r_centers.length})`);
    assert(d.xi_analytic.length === config.n_bins, 'xi_analytic length matches n_bins');
    assert(d.mean_xi.length === config.n_bins, 'mean_xi length matches n_bins');
    assert(d.std_xi.length === config.n_bins, 'std_xi length matches n_bins');
    assert(d.stderr_xi.length === config.n_bins, 'stderr_xi length matches n_bins');
    assert(typeof d.chi2 === 'number' && !isNaN(d.chi2), 'chi2 is a valid number');
    assert(typeof d.chi2_per_dof === 'number' && !isNaN(d.chi2_per_dof), 'chi2_per_dof is a valid number');
    assert(d.n_mocks === config.n_mocks, `n_mocks matches (${d.n_mocks})`);

    // --- CDF checks ---
    console.log('2. CDF checks');
    assert(Array.isArray(d.knn_cdfs), 'knn_cdfs is array');
    assert(d.knn_cdfs.length === config.k_max, `knn_cdfs has k_max=${config.k_max} entries (got ${d.knn_cdfs.length})`);
    assert(Array.isArray(d.knn_pdfs), 'knn_pdfs is array');
    assert(d.knn_pdfs.length === config.k_max, `knn_pdfs has k_max entries (got ${d.knn_pdfs.length})`);
    // CDFs should be monotonically non-decreasing
    for (let k = 0; k < d.knn_cdfs.length; k++) {
        const cdf = d.knn_cdfs[k];
        let mono = true;
        for (let i = 1; i < cdf.length; i++) {
            if (cdf[i] < cdf[i - 1] - 1e-12) { mono = false; break; }
        }
        assert(mono, `CDF k=${k + 1} is monotonically non-decreasing`);
    }

    // --- Dilution checks ---
    console.log('3. Dilution checks');
    assert(Array.isArray(d.dilution_xi), 'dilution_xi is array');
    assert(d.dilution_xi.length >= 1, `dilution_xi has ≥1 level (got ${d.dilution_xi.length})`);
    assert(Array.isArray(d.dilution_stderr), 'dilution_stderr is array');
    assert(d.dilution_r_char.length >= 1, 'dilution_r_char has entries');

    // --- Physics sanity checks ---
    console.log('4. Physics sanity');
    // r_centers should be sorted ascending
    let sorted = true;
    for (let i = 1; i < d.r_centers.length; i++) {
        if (d.r_centers[i] <= d.r_centers[i - 1]) { sorted = false; break; }
    }
    assert(sorted, 'r_centers are sorted ascending');

    // xi_analytic should be positive at small r (CoxMock has clustering)
    assert(d.xi_analytic[0] > 0, `xi_analytic[0] > 0 (got ${d.xi_analytic[0].toFixed(4)})`);

    // mean_xi should roughly track xi_analytic (not wildly off)
    // Check at the smallest bin where signal is strongest
    const ratio = d.mean_xi[0] / d.xi_analytic[0];
    assert(ratio > 0.1 && ratio < 10, `mean_xi[0]/xi_analytic[0] within order of magnitude (ratio=${ratio.toFixed(3)})`);

    // std_xi should be non-negative
    const allNonNeg = d.std_xi.every(v => v >= 0);
    assert(allNonNeg, 'std_xi all non-negative');

    // chi2_per_dof should be finite and positive
    assert(d.chi2_per_dof > 0, `chi2_per_dof > 0 (got ${d.chi2_per_dof.toFixed(2)})`);
    assert(isFinite(d.chi2_per_dof), 'chi2_per_dof is finite');

    // --- Verbose output ---
    if (verbose) {
        console.log('\n5. Detailed output');
        console.log(`   chi2/dof = ${d.chi2_per_dof.toFixed(3)}`);
        console.log(`   r_centers: [${d.r_centers[0].toFixed(1)}, ..., ${d.r_centers[d.r_centers.length - 1].toFixed(1)}]`);
        console.log(`   xi_analytic range: [${Math.min(...d.xi_analytic).toFixed(4)}, ${Math.max(...d.xi_analytic).toFixed(4)}]`);
        console.log(`   mean_xi range: [${Math.min(...d.mean_xi).toFixed(4)}, ${Math.max(...d.mean_xi).toFixed(4)}]`);
        console.log(`   knn_cdfs: ${d.knn_cdfs.length} × ${d.knn_cdfs[0]?.length} grid`);
        console.log(`   dilution levels: ${d.dilution_xi.length}`);
        console.log(`   dilution_r_char: [${d.dilution_r_char.map(v => v.toFixed(1)).join(', ')}]`);

        console.log('\n   r        xi_true    xi_est     stderr');
        for (let i = 0; i < d.r_centers.length; i++) {
            console.log(`   ${d.r_centers[i].toFixed(1).padStart(6)}  ${d.xi_analytic[i].toFixed(4).padStart(10)}  ${d.mean_xi[i].toFixed(4).padStart(10)}  ${d.stderr_xi[i].toFixed(4).padStart(10)}`);
        }
    }

    // --- Summary ---
    console.log(`\n${failures === 0 ? 'ALL PASSED' : `${failures} FAILURE(S)`}`);
    process.exit(failures === 0 ? 0 : 1);
}

main().catch(err => {
    console.error('Fatal error:', err);
    process.exit(2);
});

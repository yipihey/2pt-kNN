#!/usr/bin/env node
// Final timing sweep with r_max properly capped below r_char(k_max).
// Rule: r_max ≤ 0.8 × r_k to avoid edge effects.

import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const { run_validation_wasm } = require('./pkg-node/twopoint.js');

function r_char(k, nbar) {
    return (k / (nbar * 4 * Math.PI / 3)) ** (1/3);
}

function makeConfig(np, nm, km, nb, rr, box, line_len) {
    const nbar = np / (box ** 3);
    const rk = r_char(km, nbar);
    const r_max = Math.min(rk * 0.8, line_len * 0.5, box / 2);
    return {
        config: {
            n_mocks: nm, n_points: np,
            n_lines: Math.round(np / 10),
            line_length: line_len, box_size: box,
            k_max: km, n_bins: nb,
            r_min: 3.0, r_max,
            random_ratio: rr,
            max_dilution_level: 1,
            use_gpu: false,
        },
        rk, r_max, nbar,
    };
}

const sweeps = [
    // [np, nm, km, nb, rr, box, line, label]
    // --- Quick candidates (target: <1s node, <2s browser) ---
    [ 3000, 3,  8, 15, 3, 500, 200, "3k k8"],
    [ 5000, 3,  8, 15, 3, 500, 200, "5k k8 rr3"],
    [ 5000, 3,  8, 15, 5, 500, 200, "5k k8 rr5"],
    [ 5000, 3, 16, 15, 5, 500, 200, "5k k16"],
    [ 5000, 5, 16, 15, 5, 500, 200, "5k k16 5m"],

    // --- Medium candidates (target: <3s node, <6s browser) ---
    [10000, 3, 16, 20, 5, 500, 200, "10k k16 3m"],
    [10000, 5, 16, 20, 5, 500, 200, "10k k16 5m"],
    [10000, 5, 32, 20, 5, 500, 200, "10k k32 5m"],
    [10000, 5, 32, 25, 10, 500, 200, "10k k32 rr10"],

    // --- Full candidates (target: <8s node, <15s browser) ---
    [10000, 10, 32, 25, 5, 500, 200, "10k k32 10m"],
    [10000, 10, 64, 25, 5, 500, 200, "10k k64 10m"],
    [20000, 5, 32, 25, 5, 500, 200, "20k k32 5m"],
    [20000, 5, 64, 25, 5, 500, 200, "20k k64 5m"],
    [20000, 10, 32, 25, 5, 500, 200, "20k k32 10m"],
];

async function bench(np, nm, km, nb, rr, box, line, label) {
    const { config, rk, r_max } = makeConfig(np, nm, km, nb, rr, box, line);

    const t0 = performance.now();
    const d = await run_validation_wasm(config);
    const elapsed = (performance.now() - t0) / 1000;

    // Quality: mean frac error at bins where xi_analytic > 0.01
    let sumFE = 0, cntFE = 0;
    for (let i = 0; i < d.r_centers.length; i++) {
        if (d.xi_analytic[i] > 0.01) {
            sumFE += Math.abs((d.mean_xi[i] - d.xi_analytic[i]) / d.xi_analytic[i]);
            cntFE++;
        }
    }
    const meanFE = cntFE > 0 ? sumFE / cntFE * 100 : NaN;

    return { label, np, nm, km, nb, rr, elapsed, chi2dof: d.chi2_per_dof, meanFE, rk, r_max };
}

async function main() {
    console.log('=== WASM Timing Sweep (r_max capped at 0.8·r_k) ===\n');
    console.log(
        `${'label'.padEnd(16)} ${'N'.padStart(6)} ${'m'.padStart(3)} ${'k'.padStart(3)} ${'R/D'.padStart(3)} ` +
        `${'r_k'.padStart(5)} ${'r_max'.padStart(5)} ` +
        `${'time'.padStart(6)} ${'chi2/d'.padStart(8)} ${'frac%'.padStart(7)}`
    );
    console.log('-'.repeat(80));

    for (const [np, nm, km, nb, rr, box, line, label] of sweeps) {
        const r = await bench(np, nm, km, nb, rr, box, line, label);
        console.log(
            `${r.label.padEnd(16)} ${String(r.np).padStart(6)} ${String(r.nm).padStart(3)} ${String(r.km).padStart(3)} ${String(r.rr).padStart(3)} ` +
            `${r.rk.toFixed(0).padStart(5)} ${r.r_max.toFixed(0).padStart(5)} ` +
            `${r.elapsed.toFixed(2).padStart(6)} ${r.chi2dof.toFixed(1).padStart(8)} ` +
            `${(isNaN(r.meanFE) ? 'N/A' : r.meanFE.toFixed(1) + '%').padStart(7)}`
        );
    }

    console.log('\nBrowser ≈ 1.5-2× slower. Quick <2s, Medium <6s, Full <15s.');
}

main().catch(err => { console.error(err); process.exit(1); });

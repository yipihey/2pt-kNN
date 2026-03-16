// twopoint — kNN 2PCF Explorer
//
// Browser application that loads the twopoint WASM module and provides
// an interactive dashboard for CoxMock validation with the kNN Landy-Szalay
// estimator. Supports both CPU and WebGPU backends.

import init, { run_validation_wasm } from './pkg/twopoint.js';

let wasmReady = false;
let hasGpu = false;
let lastResult = null;

// --- Initialization ---

async function initialize() {
    try {
        await init();
        wasmReady = true;
        console.log('WASM module loaded');
    } catch (e) {
        console.error('Failed to load WASM:', e);
        document.getElementById('gpu-badge').textContent = 'WASM Error';
        document.getElementById('gpu-badge').className = 'badge badge-cpu';
        return;
    }

    // Detect WebGPU
    if (navigator.gpu) {
        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (adapter) {
                hasGpu = true;
                const info = await adapter.requestAdapterInfo();
                const name = info.device || info.description || 'Unknown GPU';
                document.getElementById('gpu-badge').textContent = `GPU: ${name}`;
                document.getElementById('gpu-badge').className = 'badge badge-gpu';
            }
        } catch (e) {
            console.warn('WebGPU detection failed:', e);
        }
    }

    if (!hasGpu) {
        document.getElementById('gpu-badge').textContent = 'CPU only';
        document.getElementById('gpu-badge').className = 'badge badge-cpu';
        document.getElementById('use-gpu').checked = false;
        document.getElementById('use-gpu').disabled = true;
    }
}

// --- Slider bindings ---

function bindSlider(id) {
    const slider = document.getElementById(id);
    const display = document.getElementById(id + '-val');
    if (slider && display) {
        slider.addEventListener('input', () => {
            display.textContent = slider.value;
        });
    }
}

// --- Preset configurations ---

const presets = {
    tiny: {
        n_points: 50000, n_lines: 5000, box_size: 500,
        line_length: 200, n_mocks: 10, k_max: 8, n_bins: 40,
        r_min: 5, r_max: 250, random_ratio: 5,
    },
    euclid: {
        n_points: 100000, n_lines: 10000, box_size: 1000,
        line_length: 400, n_mocks: 10, k_max: 8, n_bins: 40,
        r_min: 5, r_max: 250, random_ratio: 5,
    },
    large: {
        n_points: 500000, n_lines: 50000, box_size: 2000,
        line_length: 800, n_mocks: 5, k_max: 8, n_bins: 60,
        r_min: 5, r_max: 400, random_ratio: 3,
    },
};

function applyPreset(name) {
    const p = presets[name];
    if (!p) return;

    const setSlider = (id, val) => {
        const el = document.getElementById(id);
        if (el) {
            el.value = val;
            const display = document.getElementById(id + '-val');
            if (display) display.textContent = val;
        }
    };

    setSlider('n-points', p.n_points);
    setSlider('n-lines', p.n_lines);
    setSlider('n-mocks', p.n_mocks);
    setSlider('k-max', p.k_max);
    setSlider('n-bins', p.n_bins);
    setSlider('r-min', p.r_min);
    setSlider('r-max', p.r_max);
    setSlider('random-ratio', p.random_ratio);

    // Update active preset button
    document.querySelectorAll('.preset-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById('preset-' + name)?.classList.add('active');
}

// --- Tab switching ---

function setupTabs() {
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.plot-content').forEach(p => p.classList.remove('active'));

            tab.classList.add('active');
            const target = document.getElementById('plot-' + tab.dataset.tab);
            if (target) target.classList.add('active');
        });
    });
}

// --- Run validation ---

async function runValidation() {
    if (!wasmReady) {
        alert('WASM module not yet loaded');
        return;
    }

    const btn = document.getElementById('run-btn');
    btn.disabled = true;
    btn.textContent = 'Running...';

    const progressContainer = document.getElementById('progress-container');
    progressContainer.classList.remove('hidden');
    document.getElementById('progress-fill').style.width = '10%';
    document.getElementById('progress-text').textContent = 'Generating mocks...';

    const config = {
        n_mocks: parseInt(document.getElementById('n-mocks').value),
        n_points: parseInt(document.getElementById('n-points').value),
        n_lines: parseInt(document.getElementById('n-lines').value),
        line_length: parseFloat(document.getElementById('r-max').value) * 1.6, // scale with r_max
        box_size: parseFloat(document.getElementById('r-max').value) * 4, // ~4x r_max
        k_max: parseInt(document.getElementById('k-max').value),
        n_bins: parseInt(document.getElementById('n-bins').value),
        r_min: parseFloat(document.getElementById('r-min').value),
        r_max: parseFloat(document.getElementById('r-max').value),
        random_ratio: parseInt(document.getElementById('random-ratio').value),
        max_dilution_level: 2,
        use_gpu: hasGpu && document.getElementById('use-gpu').checked,
    };

    const startTime = performance.now();

    try {
        document.getElementById('progress-fill').style.width = '30%';
        document.getElementById('progress-text').textContent = 'Running kNN queries...';

        const result = await run_validation_wasm(config);
        const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);

        lastResult = result;

        document.getElementById('progress-fill').style.width = '100%';
        document.getElementById('progress-text').textContent = `Done in ${elapsed}s`;

        // Update plots
        updatePlot('plot-xi', result.svg_xi);
        updatePlot('plot-r2xi', result.svg_r2xi);
        updatePlot('plot-cdf', result.svg_cdf);
        updatePlot('plot-dilution', result.svg_dilution);

        // Update stats
        const stats = JSON.parse(result.stats_json);
        document.getElementById('stat-chi2').textContent = `${stats.chi2.toFixed(2)} / ${stats.n_bins} = ${stats.chi2_per_dof.toFixed(3)}`;
        document.getElementById('stat-nmocks').textContent = stats.n_mocks;
        document.getElementById('stat-nbins').textContent = stats.n_bins;
        document.getElementById('stat-backend').textContent = config.use_gpu ? 'WebGPU' : 'CPU (WASM)';
        document.getElementById('stat-time').textContent = `${elapsed}s`;

        // Update dilution info
        if (stats.dilution_r_char && stats.dilution_r_char.length > 0) {
            const dilDiv = document.getElementById('dilution-info');
            dilDiv.innerHTML = stats.dilution_r_char.map((rc, l) =>
                `<div style="font-family: var(--font-mono); font-size: 12px; margin: 2px 0;">
                    Level ${l}: R<sub>l</sub>=${Math.pow(8, l)}, r<sub>char</sub>=${rc.toFixed(1)}
                </div>`
            ).join('');
        }

        // Update data table
        updateDataTable(result.tsv);

        // Enable downloads
        document.getElementById('download-tsv').disabled = false;
        document.getElementById('download-svg').disabled = false;

    } catch (e) {
        console.error('Validation failed:', e);
        document.getElementById('progress-text').textContent = `Error: ${e.message || e}`;
        document.getElementById('progress-fill').style.width = '0%';
    }

    btn.disabled = false;
    btn.textContent = 'Run Validation';
}

function updatePlot(containerId, svg) {
    const el = document.getElementById(containerId);
    if (el && svg) {
        el.innerHTML = svg;
    }
}

function updateDataTable(tsv) {
    const tbody = document.getElementById('data-table-body');
    if (!tbody || !tsv) return;

    tbody.innerHTML = '';
    const lines = tsv.trim().split('\n').filter(l => !l.startsWith('#'));
    for (const line of lines) {
        const cols = line.split('\t');
        if (cols.length >= 4) {
            const tr = document.createElement('tr');
            tr.innerHTML = cols.slice(0, 4).map(v => `<td>${parseFloat(v).toFixed(4)}</td>`).join('');
            tbody.appendChild(tr);
        }
    }
}

// --- Downloads ---

function downloadFile(content, filename, mime) {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

// --- Setup ---

document.addEventListener('DOMContentLoaded', () => {
    // Bind sliders
    ['n-points', 'n-lines', 'n-mocks', 'k-max', 'n-bins', 'r-min', 'r-max', 'random-ratio']
        .forEach(bindSlider);

    // Presets
    document.getElementById('preset-tiny').addEventListener('click', () => applyPreset('tiny'));
    document.getElementById('preset-euclid').addEventListener('click', () => applyPreset('euclid'));
    document.getElementById('preset-large').addEventListener('click', () => applyPreset('large'));

    // Tabs
    setupTabs();

    // Run button
    document.getElementById('run-btn').addEventListener('click', runValidation);

    // Downloads
    document.getElementById('download-tsv').addEventListener('click', () => {
        if (lastResult?.tsv) downloadFile(lastResult.tsv, 'xi_comparison.tsv', 'text/tab-separated-values');
    });
    document.getElementById('download-svg').addEventListener('click', () => {
        if (lastResult?.svg_xi) downloadFile(lastResult.svg_xi, 'xi_vs_analytic.svg', 'image/svg+xml');
    });

    // Initialize WASM
    initialize();
});

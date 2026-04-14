// twopoint — kNN 2PCF Explorer
//
// Browser app for CoxMock validation with the kNN Landy-Szalay estimator.
// WASM returns raw data arrays; plots are rendered client-side as SVG.

import init, { run_validation_wasm } from './pkg/twopoint.js';

let wasmReady = false;
let hasGpu = false;
let lastResult = null;
let lastConfig = null;

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
        } catch (e) { console.warn('WebGPU detection failed:', e); }
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
            updateTimeEstimate();
        });
    }
}

// --- CoxMock physics (fixed: validation preset) ---
// box=500, line_length=200, m=N_p/N_L=10
// xi(r) = prefactor/r² · (1 − r/200), non-zero for r < 200

const BOX_SIZE = 500;
const LINE_LENGTH = 200;

// kNN reach: r_char(k) = (k / (nbar × 4π/3))^(1/3)
function knnReach(k, nPoints) {
    const nbar = nPoints / (BOX_SIZE ** 3);
    return (k / (nbar * 4 * Math.PI / 3)) ** (1/3);
}

// --- Presets (tuned for KD-tree pipeline with periodic BCs) ---

const presets = {
    quick:  { n_points: 5000,  n_mocks: 3,  k_max: 16, n_bins: 15, r_min: 3, random_ratio: 5 },
    medium: { n_points: 10000, n_mocks: 5,  k_max: 32, n_bins: 20, r_min: 3, random_ratio: 5 },
    full:   { n_points: 10000, n_mocks: 10, k_max: 32, n_bins: 25, r_min: 3, random_ratio: 5 },
};

function applyPreset(name) {
    const p = presets[name];
    if (!p) return;
    const set = (id, val) => {
        const el = document.getElementById(id);
        if (el) { el.value = val; const d = document.getElementById(id + '-val'); if (d) d.textContent = val; }
    };
    const n_lines = Math.round(p.n_points / 10);
    const r_max = Math.floor(knnReach(p.k_max, p.n_points) * 0.8);
    set('n-points', p.n_points); set('n-lines', n_lines); set('n-mocks', p.n_mocks);
    set('k-max', p.k_max); set('n-bins', p.n_bins); set('r-min', p.r_min);
    set('r-max', r_max); set('random-ratio', p.random_ratio);
    document.querySelectorAll('.preset-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById('preset-' + name)?.classList.add('active');
    updateTimeEstimate();
}

function estimateSeconds() {
    const n = parseInt(document.getElementById('n-points').value) || 5000;
    const mocks = parseInt(document.getElementById('n-mocks').value) || 3;
    const k = parseInt(document.getElementById('k-max').value) || 16;
    const ratio = parseInt(document.getElementById('random-ratio').value) || 5;
    const queriesPerMock = n * (2 + ratio);
    const opsPerMock = queriesPerMock * Math.log2(n) * k * 1.3;
    return opsPerMock * mocks / 50_000_000;
}

function updateTimeEstimate() {
    const el = document.getElementById('time-estimate');
    if (!el) return;
    const secs = estimateSeconds();
    if (secs < 5) { el.textContent = `~${Math.ceil(secs)}s`; el.className = 'time-estimate fast'; }
    else if (secs < 30) { el.textContent = `~${Math.ceil(secs)}s`; el.className = 'time-estimate medium'; }
    else if (secs < 120) { el.textContent = `~${Math.ceil(secs/10)*10}s (slow)`; el.className = 'time-estimate slow'; }
    else { el.textContent = `~${Math.ceil(secs/60)}min — reduce N`; el.className = 'time-estimate slow'; }
}

// =========================================================================
// Analytic kNN distributions (Poisson/Erlang reference)
// =========================================================================

function logFactorial(n) {
    let s = 0;
    for (let i = 2; i <= n; i++) s += Math.log(i);
    return s;
}

// CDF_k(r) = P(Poisson(λ) >= k) = 1 - Σ_{j=0}^{k-1} e^{-λ} λ^j / j!
// where λ(r) = nbar × 4π/3 × r³
function erlangCdf(r, k, nbar) {
    const lam = nbar * (4 / 3) * Math.PI * r * r * r;
    let sum = 0, term = Math.exp(-lam);
    for (let j = 0; j < k; j++) {
        if (!isFinite(term)) break;
        sum += term;
        term *= lam / (j + 1);
    }
    return 1 - sum;
}

// PDF_k(r) = dCDF_k/dr = e^{-λ} λ^{k-1} / (k-1)! × 4π nbar r²
function erlangPdf(r, k, nbar) {
    if (r <= 0 || k < 1) return 0;
    const lam = nbar * (4 / 3) * Math.PI * r * r * r;
    const dlam_dr = 4 * Math.PI * nbar * r * r;
    const logP = -lam + (k - 1) * Math.log(lam) - logFactorial(k - 1);
    return Math.exp(logP) * dlam_dr;
}

// Powers of 2 up to kMax: [1, 2, 4, 8, ...]
function cdfKValues(kMax) {
    const ks = [];
    let k = 1;
    while (k <= kMax) { ks.push(k); k *= 2; }
    return ks;
}

// Dense r grid for smooth analytic curves
function denseGrid(rMin, rMax, n = 200) {
    const dr = (rMax - rMin) / (n - 1);
    return Array.from({ length: n }, (_, i) => rMin + i * dr);
}

// =========================================================================
// Axis scale transforms
// =========================================================================

const SYMLOG_C = 0.01; // linear threshold for symlog

function scaleTransform(v, mode) {
    if (mode === 'log') return v > 0 ? Math.log10(v) : NaN;
    if (mode === 'symlog') {
        if (Math.abs(v) <= SYMLOG_C) return v / SYMLOG_C;
        return Math.sign(v) * (1 + Math.log10(Math.abs(v) / SYMLOG_C));
    }
    return v; // linear
}

function scaleTicks(vmin, vmax, mode, n = 6) {
    if (mode === 'log') {
        const lo = Math.floor(vmin), hi = Math.ceil(vmax);
        const ticks = [];
        for (let e = lo; e <= hi; e++) ticks.push({ val: e, label: `10${superscript(e)}` });
        // If too few ticks, add halves
        if (ticks.length < 3) {
            for (let e = lo; e < hi; e++) ticks.push({ val: e + Math.log10(3), label: '' });
        }
        return ticks.filter(t => t.val >= vmin && t.val <= vmax);
    }
    if (mode === 'symlog') {
        const ticks = [];
        // Linear region
        ticks.push({ val: 0, label: '0' });
        // Log region: powers of 10
        for (let e = 0; e <= 6; e++) {
            const v = 1 + e; // symlog(10^e × C) = 1 + e
            if (v <= vmax) ticks.push({ val: v, label: niceNum(SYMLOG_C * Math.pow(10, e)) });
            if (-v >= vmin) ticks.push({ val: -v, label: niceNum(-SYMLOG_C * Math.pow(10, e)) });
        }
        return ticks.filter(t => t.val >= vmin - 0.01 && t.val <= vmax + 0.01);
    }
    // Linear ticks
    const ticks = [];
    for (let i = 0; i <= n; i++) {
        const v = vmin + (vmax - vmin) * i / n;
        ticks.push({ val: v, label: niceNum(v) });
    }
    return ticks;
}

function superscript(e) {
    const map = { '0': '\u2070', '1': '\xB9', '2': '\xB2', '3': '\xB3', '4': '\u2074',
                  '5': '\u2075', '6': '\u2076', '7': '\u2077', '8': '\u2078', '9': '\u2079', '-': '\u207B' };
    return String(e).split('').map(c => map[c] || c).join('');
}

// =========================================================================
// Axis scale state per plot tab
// =========================================================================

const plotScales = {
    xi:       { x: 'linear', y: 'linear' },
    r2xi:     { x: 'linear', y: 'linear' },
    cdf:      { x: 'linear', y: 'linear' },
    pdf:      { x: 'linear', y: 'linear' },
    dilution: { x: 'linear', y: 'linear' },
};

function activeTab() {
    const tab = document.querySelector('.tab.active');
    return tab ? tab.dataset.tab : 'xi';
}

function setScale(axis, mode) {
    const tab = activeTab();
    plotScales[tab][axis] = mode;
    rerenderPlot(tab);
    updateScaleButtons();
}

function updateScaleButtons() {
    const tab = activeTab();
    const s = plotScales[tab];
    document.querySelectorAll('.scale-btn').forEach(btn => {
        const active = (btn.dataset.axis === 'x' && btn.dataset.mode === s.x) ||
                       (btn.dataset.axis === 'y' && btn.dataset.mode === s.y);
        btn.classList.toggle('active', active);
    });
}

// --- Tab switching ---

function setupTabs() {
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.plot-content').forEach(p => p.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById('plot-' + tab.dataset.tab)?.classList.add('active');
            updateScaleButtons();
        });
    });
}

// =========================================================================
// Client-side SVG plot renderer
// =========================================================================

const COLORS = ['#4682B4', '#DC143C', '#2E8B57', '#FF8C00', '#8B008B', '#B8860B', '#006400', '#8B0000'];
const PAD = { top: 40, right: 20, bottom: 50, left: 70 };

function svgPlot({ title, xlabel, ylabel, series, width = 700, height = 450, hlines = [], xscale = 'linear', yscale = 'linear' }) {
    const w = width - PAD.left - PAD.right;
    const h = height - PAD.top - PAD.bottom;

    // Transform all data points
    const txSeries = series.map(s => {
        const pts = [];
        for (let i = 0; i < s.x.length; i++) {
            const tx = scaleTransform(s.x[i], xscale);
            const ty = scaleTransform(s.y[i], yscale);
            const terr = s.yerr ? Math.abs(scaleTransform(s.y[i] + s.yerr[i], yscale) - ty) : 0;
            const terrLo = s.yerr ? Math.abs(ty - scaleTransform(s.y[i] - s.yerr[i], yscale)) : 0;
            if (isFinite(tx) && isFinite(ty)) pts.push({ tx, ty, errHi: terr, errLo: terrLo, origX: s.x[i] });
        }
        return { ...s, pts };
    });

    // Compute transformed extent
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    for (const s of txSeries) {
        for (const p of s.pts) {
            xmin = Math.min(xmin, p.tx); xmax = Math.max(xmax, p.tx);
            ymin = Math.min(ymin, p.ty - p.errLo); ymax = Math.max(ymax, p.ty + p.errHi);
        }
    }
    for (const hl of hlines) {
        const th = scaleTransform(hl, yscale);
        if (isFinite(th)) { ymin = Math.min(ymin, th); ymax = Math.max(ymax, th); }
    }
    if (!isFinite(xmin)) { xmin = 0; xmax = 1; }
    if (!isFinite(ymin)) { ymin = 0; ymax = 1; }
    const ypad = (ymax - ymin) * 0.08 || 0.5;
    ymin -= ypad; ymax += ypad;
    const xpad = (xmax - xmin) * 0.02 || 0.5;
    xmin -= xpad; xmax += xpad;

    const sx = x => PAD.left + (x - xmin) / (xmax - xmin) * w;
    const sy = y => PAD.top + (1 - (y - ymin) / (ymax - ymin)) * h;

    let svg = `<svg viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg" style="font-family:'Helvetica Neue',sans-serif;background:#fff">`;

    // Grid lines from ticks
    const xticks = scaleTicks(xmin, xmax, xscale);
    const yticks = scaleTicks(ymin, ymax, yscale);
    svg += `<g stroke="#e0e0e0" stroke-width="0.5">`;
    for (const t of yticks) svg += `<line x1="${PAD.left}" y1="${sy(t.val)}" x2="${width - PAD.right}" y2="${sy(t.val)}"/>`;
    for (const t of xticks) svg += `<line x1="${sx(t.val)}" y1="${PAD.top}" x2="${sx(t.val)}" y2="${PAD.top + h}"/>`;
    svg += `</g>`;

    // Horizontal reference lines
    for (const hl of hlines) {
        const th = scaleTransform(hl, yscale);
        if (isFinite(th)) svg += `<line x1="${PAD.left}" y1="${sy(th)}" x2="${width-PAD.right}" y2="${sy(th)}" stroke="#999" stroke-width="1" stroke-dasharray="6 4"/>`;
    }

    // Axes box
    svg += `<rect x="${PAD.left}" y="${PAD.top}" width="${w}" height="${h}" fill="none" stroke="#333" stroke-width="1"/>`;

    // Tick labels
    svg += `<g font-size="11" fill="#333">`;
    for (const t of xticks) {
        if (t.label) svg += `<text x="${sx(t.val)}" y="${PAD.top + h + 18}" text-anchor="middle">${t.label}</text>`;
    }
    for (const t of yticks) {
        if (t.label) svg += `<text x="${PAD.left - 8}" y="${sy(t.val) + 4}" text-anchor="end">${t.label}</text>`;
    }
    svg += `</g>`;

    // Series
    for (const s of txSeries) {
        const color = s.color || '#4682B4';
        const pts = s.pts;
        if (pts.length === 0) continue;

        // Error band
        if (s.yerr && pts.length >= 2) {
            let d = `M ${sx(pts[0].tx)} ${sy(pts[0].ty + pts[0].errHi)}`;
            for (let i = 1; i < pts.length; i++) d += ` L ${sx(pts[i].tx)} ${sy(pts[i].ty + pts[i].errHi)}`;
            for (let i = pts.length - 1; i >= 0; i--) d += ` L ${sx(pts[i].tx)} ${sy(pts[i].ty - pts[i].errLo)}`;
            d += ' Z';
            svg += `<path d="${d}" fill="${color}" fill-opacity="0.15" stroke="none"/>`;
        }
        // Line
        if (s.type !== 'scatter') {
            let d = '';
            for (const p of pts) {
                const px = sx(p.tx), py = sy(p.ty);
                d += (d === '' ? 'M' : 'L') + ` ${px} ${py}`;
            }
            const dash = s.dash ? ' stroke-dasharray="8 5"' : '';
            svg += `<path d="${d}" fill="none" stroke="${color}" stroke-width="${s.width || 2}"${dash}/>`;
        }
        // Dots
        if (s.type === 'scatter' || s.dots) {
            for (const p of pts) svg += `<circle cx="${sx(p.tx)}" cy="${sy(p.ty)}" r="3" fill="${color}"/>`;
        }
    }

    // Legend
    const legendSeries = series.filter(s => s.label);
    if (legendSeries.length > 0) {
        const lx = PAD.left + 12, ly = PAD.top + 12;
        svg += `<g font-size="11" fill="#333">`;
        legendSeries.forEach((s, i) => {
            const y = ly + i * 16;
            if (s.dash) {
                svg += `<line x1="${lx}" y1="${y}" x2="${lx+20}" y2="${y}" stroke="${s.color}" stroke-width="2" stroke-dasharray="6 3"/>`;
            } else if (s.type === 'scatter') {
                svg += `<circle cx="${lx+10}" cy="${y}" r="3" fill="${s.color}"/>`;
            } else {
                svg += `<line x1="${lx}" y1="${y}" x2="${lx+20}" y2="${y}" stroke="${s.color}" stroke-width="2"/>`;
            }
            svg += `<text x="${lx+26}" y="${y+4}">${s.label}</text>`;
        });
        svg += `</g>`;
    }

    // Title and axis labels
    const scaleLabel = (mode) => mode === 'linear' ? '' : ` [${mode}]`;
    svg += `<text x="${width/2}" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">${title}</text>`;
    svg += `<text x="${width/2}" y="${height-6}" text-anchor="middle" font-size="12" fill="#555">${xlabel}${scaleLabel(xscale)}</text>`;
    svg += `<text x="14" y="${height/2}" text-anchor="middle" font-size="12" fill="#555" transform="rotate(-90 14 ${height/2})">${ylabel}${scaleLabel(yscale)}</text>`;

    svg += `</svg>`;
    return svg;
}

function niceNum(v) {
    if (Math.abs(v) < 0.001 && v !== 0) return v.toExponential(1);
    if (Math.abs(v) >= 10000) return v.toExponential(1);
    if (Number.isInteger(v)) return v.toString();
    return v.toPrecision(3);
}

// =========================================================================
// Plot generators from WASM result data
// =========================================================================

function plotXi(d, xscale = 'linear', yscale = 'linear') {
    return svgPlot({
        title: 'CoxMock: \u03BE(r) Recovery (Landy-Szalay)',
        xlabel: 'r [h\u207B\xB9 Mpc]', ylabel: '\u03BE(r)',
        xscale, yscale,
        series: [
            { x: d.r_centers, y: d.mean_xi, yerr: d.std_xi, color: COLORS[0], label: 'kNN LS mean \xB1 1\u03C3' },
            { x: d.r_centers, y: d.xi_analytic, color: '#000', dash: true, label: 'Analytic \u03BE(r)', width: 1.5 },
        ]
    });
}

function plotR2Xi(d, xscale = 'linear', yscale = 'linear') {
    const r2xi_mean = d.r_centers.map((r, i) => r * r * d.mean_xi[i]);
    const r2xi_std = d.r_centers.map((r, i) => r * r * d.std_xi[i]);
    const r2xi_analytic = d.r_centers.map((r, i) => r * r * d.xi_analytic[i]);
    return svgPlot({
        title: 'r\xB2\u03BE(r) Comparison',
        xlabel: 'r [h\u207B\xB9 Mpc]', ylabel: 'r\xB2 \u03BE(r)',
        xscale, yscale,
        series: [
            { x: d.r_centers, y: r2xi_mean, yerr: r2xi_std, color: COLORS[0], label: 'kNN LS' },
            { x: d.r_centers, y: r2xi_analytic, color: '#000', dash: true, label: 'Analytic', width: 1.5 },
        ]
    });
}

function plotCdf(d, config, xscale = 'linear', yscale = 'linear') {
    const nbar = config.n_points / (BOX_SIZE ** 3);
    const rGrid = d.cdf_r_grid || [];
    const kVals = d.cdf_k_values || cdfKValues(config.k_max);
    const rDense = denseGrid(d.r_centers[0], d.r_centers[d.r_centers.length - 1], 200);

    const series = [];
    kVals.forEach((k, ki) => {
        const color = COLORS[ki % COLORS.length];
        // RR empirical CDF with ±1σ band (from multi-mock accumulation)
        if (d.cdf_rr_mean && ki < d.cdf_rr_mean.length) {
            series.push({
                x: rGrid, y: d.cdf_rr_mean[ki],
                yerr: d.cdf_rr_std[ki],
                color, label: `RR k=${k}`, width: 1.5,
            });
        }
        // DD empirical CDF with ±1σ band
        if (d.cdf_dd_mean && ki < d.cdf_dd_mean.length) {
            series.push({
                x: rGrid, y: d.cdf_dd_mean[ki],
                yerr: d.cdf_dd_std[ki],
                color, label: `DD k=${k}`, width: 1.5, dash: false,
                // Use a slightly different rendering: dots on DD
                dots: true,
            });
        }
        // Analytic Erlang (dashed)
        const analyticCdf = rDense.map(r => erlangCdf(r, k, nbar));
        series.push({ x: rDense, y: analyticCdf, color: '#000', dash: true, width: 0.8 });
    });

    return svgPlot({
        title: 'kNN-CDF: DD (dots) & RR (line) vs Poisson (dashed)',
        xlabel: 'r [h\u207B\xB9 Mpc]', ylabel: 'CDF_k(r)',
        xscale, yscale, series
    });
}

function plotPdf(d, config, xscale = 'linear', yscale = 'linear') {
    const nbar = config.n_points / (BOX_SIZE ** 3);
    const rGrid = d.cdf_r_grid || [];
    const kVals = d.cdf_k_values || cdfKValues(config.k_max);
    const rDense = denseGrid(d.r_centers[0], d.r_centers[d.r_centers.length - 1], 200);

    // Compute PDFs from CDF summary via numerical differentiation
    function cdfToPdf(cdfMean, cdfStd, r) {
        const n = r.length;
        const pdfMean = new Array(n).fill(0);
        const pdfStd = new Array(n).fill(0);
        if (n >= 3) {
            for (let i = 1; i < n - 1; i++) {
                const dr = r[i + 1] - r[i - 1];
                pdfMean[i] = (cdfMean[i + 1] - cdfMean[i - 1]) / dr;
                // Propagate std through derivative (approx)
                pdfStd[i] = Math.sqrt(cdfStd[i + 1] ** 2 + cdfStd[i - 1] ** 2) / dr;
            }
            pdfMean[0] = (cdfMean[1] - cdfMean[0]) / (r[1] - r[0]);
            pdfStd[0] = Math.sqrt(cdfStd[1] ** 2 + cdfStd[0] ** 2) / (r[1] - r[0]);
            pdfMean[n - 1] = (cdfMean[n - 1] - cdfMean[n - 2]) / (r[n - 1] - r[n - 2]);
            pdfStd[n - 1] = Math.sqrt(cdfStd[n - 1] ** 2 + cdfStd[n - 2] ** 2) / (r[n - 1] - r[n - 2]);
        }
        return { pdfMean, pdfStd };
    }

    const series = [];
    kVals.forEach((k, ki) => {
        const color = COLORS[ki % COLORS.length];
        // RR empirical PDF with ±1σ band
        if (d.cdf_rr_mean && ki < d.cdf_rr_mean.length) {
            const { pdfMean, pdfStd } = cdfToPdf(d.cdf_rr_mean[ki], d.cdf_rr_std[ki], rGrid);
            series.push({
                x: rGrid, y: pdfMean, yerr: pdfStd,
                color, label: `RR k=${k}`, width: 1.5,
            });
        }
        // DD empirical PDF with ±1σ band
        if (d.cdf_dd_mean && ki < d.cdf_dd_mean.length) {
            const { pdfMean, pdfStd } = cdfToPdf(d.cdf_dd_mean[ki], d.cdf_dd_std[ki], rGrid);
            series.push({
                x: rGrid, y: pdfMean, yerr: pdfStd,
                color, label: `DD k=${k}`, width: 1.5, dots: true,
            });
        }
        // Analytic Erlang PDF (dashed)
        const analyticPdf = rDense.map(r => erlangPdf(r, k, nbar));
        series.push({ x: rDense, y: analyticPdf, color: '#000', dash: true, width: 0.8 });
    });

    return svgPlot({
        title: 'kNN-PDF: DD (dots) & RR (line) vs Erlang (dashed)',
        xlabel: 'r [h\u207B\xB9 Mpc]', ylabel: 'dCDF_k/dr',
        xscale, yscale, series
    });
}

function plotDilution(d, xscale = 'linear', yscale = 'linear') {
    const series = d.dilution_xi.map((xi, level) => ({
        x: d.r_centers, y: xi,
        yerr: d.dilution_stderr[level],
        color: COLORS[level % COLORS.length],
        label: `Level ${level} (r_char=${d.dilution_r_char[level]?.toFixed(0) || '?'})`
    }));
    series.push({ x: d.r_centers, y: d.xi_analytic, color: '#000', dash: true, label: 'Analytic', width: 1.5 });
    return svgPlot({
        title: 'Dilution Ladder: \u03BE(r) \xB1 \u03C3',
        xlabel: 'r [h\u207B\xB9 Mpc]', ylabel: '\u03BE(r)',
        xscale, yscale, series
    });
}

// =========================================================================
// Re-render active plot with current scale settings
// =========================================================================

function rerenderPlot(tab) {
    if (!lastResult || !lastConfig) return;
    const d = lastResult, c = lastConfig;
    const s = plotScales[tab];
    const el = document.getElementById('plot-' + tab);
    if (!el) return;

    const renderers = {
        xi:       () => plotXi(d, s.x, s.y),
        r2xi:     () => plotR2Xi(d, s.x, s.y),
        cdf:      () => plotCdf(d, c, s.x, s.y),
        pdf:      () => plotPdf(d, c, s.x, s.y),
        dilution: () => plotDilution(d, s.x, s.y),
    };

    if (renderers[tab]) el.innerHTML = renderers[tab]();
}

function renderAllPlots() {
    if (!lastResult || !lastConfig) return;
    for (const tab of Object.keys(plotScales)) rerenderPlot(tab);
}

// =========================================================================
// Run validation
// =========================================================================

function yieldToUI() { return new Promise(resolve => setTimeout(resolve, 0)); }

async function runValidation() {
    if (!wasmReady) { alert('WASM module not yet loaded'); return; }

    const btn = document.getElementById('run-btn');
    btn.disabled = true;
    btn.textContent = 'Running...';

    const progressContainer = document.getElementById('progress-container');
    progressContainer.classList.remove('hidden');
    document.getElementById('progress-fill').style.width = '10%';
    const est = Math.max(1, Math.ceil(estimateSeconds()));
    document.getElementById('progress-text').textContent = `Computing (~${est}s)...`;

    const nPoints = parseInt(document.getElementById('n-points').value);
    const kMax = parseInt(document.getElementById('k-max').value);
    const rMaxSlider = parseFloat(document.getElementById('r-max').value);
    const rk = knnReach(kMax, nPoints);
    const rMaxCapped = Math.min(rMaxSlider, rk * 0.8);

    const config = {
        n_mocks: parseInt(document.getElementById('n-mocks').value),
        n_points: nPoints,
        n_lines: parseInt(document.getElementById('n-lines').value),
        line_length: LINE_LENGTH,
        box_size: BOX_SIZE,
        k_max: kMax,
        n_bins: parseInt(document.getElementById('n-bins').value),
        r_min: parseFloat(document.getElementById('r-min').value),
        r_max: rMaxCapped,
        random_ratio: parseInt(document.getElementById('random-ratio').value),
        max_dilution_level: 2,
        use_gpu: hasGpu && document.getElementById('use-gpu').checked,
    };

    await yieldToUI();
    const startTime = performance.now();

    try {
        const onProgress = (done, total) => {
            const pct = 10 + (done / total) * 70;
            document.getElementById('progress-fill').style.width = pct + '%';
            document.getElementById('progress-text').textContent = `Mock ${done}/${total}...`;
        };
        const d = await run_validation_wasm(config, onProgress);
        const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
        lastResult = d;
        lastConfig = config;

        document.getElementById('progress-fill').style.width = '80%';
        document.getElementById('progress-text').textContent = 'Rendering plots...';
        await yieldToUI();

        renderAllPlots();

        document.getElementById('progress-fill').style.width = '100%';
        document.getElementById('progress-text').textContent = `Done in ${elapsed}s`;

        // Stats
        document.getElementById('stat-chi2').textContent = `${d.chi2.toFixed(2)} / ${d.r_centers.length} = ${d.chi2_per_dof.toFixed(3)}`;
        document.getElementById('stat-nmocks').textContent = d.n_mocks;
        document.getElementById('stat-nbins').textContent = d.r_centers.length;
        document.getElementById('stat-backend').textContent = config.use_gpu ? 'WebGPU' : 'CPU (WASM)';
        document.getElementById('stat-time').textContent = `${elapsed}s`;

        if (d.dilution_r_char?.length > 0) {
            document.getElementById('dilution-info').innerHTML = d.dilution_r_char.map((rc, l) =>
                `<div style="font-family:var(--font-mono);font-size:12px;margin:2px 0">Level ${l}: R<sub>l</sub>=${Math.pow(8,l)}, r<sub>char</sub>=${rc.toFixed(1)}</div>`
            ).join('');
        }

        // Data table
        const tbody = document.getElementById('data-table-body');
        tbody.innerHTML = '';
        for (let i = 0; i < d.r_centers.length; i++) {
            const tr = document.createElement('tr');
            tr.innerHTML = `<td>${d.r_centers[i].toFixed(2)}</td><td>${d.xi_analytic[i].toFixed(4)}</td><td>${d.mean_xi[i].toFixed(4)}</td><td>${d.std_xi[i].toFixed(4)}</td>`;
            tbody.appendChild(tr);
        }

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

// --- Downloads ---

function downloadFile(content, filename, mime) {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename; a.click();
    URL.revokeObjectURL(url);
}

function makeTsv(d) {
    let tsv = '# r\txi_analytic\txi_mean\txi_std\txi_stderr\n';
    for (let i = 0; i < d.r_centers.length; i++) {
        tsv += `${d.r_centers[i].toFixed(2)}\t${d.xi_analytic[i].toFixed(8)}\t${d.mean_xi[i].toFixed(8)}\t${d.std_xi[i].toFixed(8)}\t${d.stderr_xi[i].toFixed(8)}\n`;
    }
    return tsv;
}

// --- Setup ---

document.addEventListener('DOMContentLoaded', () => {
    ['n-points','n-lines','n-mocks','k-max','n-bins','r-min','r-max','random-ratio'].forEach(bindSlider);
    document.getElementById('preset-quick').addEventListener('click', () => applyPreset('quick'));
    document.getElementById('preset-medium').addEventListener('click', () => applyPreset('medium'));
    document.getElementById('preset-full').addEventListener('click', () => applyPreset('full'));
    setupTabs();
    document.getElementById('run-btn').addEventListener('click', runValidation);
    document.getElementById('download-tsv').addEventListener('click', () => {
        if (lastResult) downloadFile(makeTsv(lastResult), 'xi_comparison.tsv', 'text/tab-separated-values');
    });
    document.getElementById('download-svg').addEventListener('click', () => {
        if (lastResult) downloadFile(document.getElementById('plot-xi').innerHTML, 'xi_vs_analytic.svg', 'image/svg+xml');
    });

    // Scale toggle buttons
    document.querySelectorAll('.scale-btn').forEach(btn => {
        btn.addEventListener('click', () => setScale(btn.dataset.axis, btn.dataset.mode));
    });

    initialize();
    applyPreset('quick');
});

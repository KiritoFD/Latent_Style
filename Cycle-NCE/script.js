// Global State & Config
const vibrantPalette = [
    '#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', 
    '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85',
    '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#e377c2'
];

const standardPalette = [
    '#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b3', 
    '#937860', '#da8bc3', '#8c8c8c', '#ccb974', '#64b5cd'
];

let allGroups = new Set();
let enabledGroups = new Set();
let hasInitializedGroups = false;
let allRelatedWorks = new Set();
let enabledRelatedWorks = new Set();
let hasInitializedRelatedWorks = false;
let groupColorMap = new Map();

function getUiOptions() {
    return {
        showLines: document.getElementById('toggleLines')?.checked ?? true,
        isVibrant: document.getElementById('toggleVibrant')?.checked ?? true,
        hideDistill: document.getElementById('toggleHideDistill')?.checked ?? false,
        multiColorWithinGroup: document.getElementById('toggleMultiColorWithinGroup')?.checked ?? true
    };
}

function getSeriesName(group) {
    return group;
}

// Theme Management
function toggleTheme() {
    const html = document.documentElement;
    const currentTheme = html.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeToggleBtn();
    
    // Re-render plots to apply theme colors
    plotData();
}

function updateThemeToggleBtn() {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const btn = document.getElementById('themeToggleBtn');
    if (btn) btn.textContent = isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode';
}

function getCurrentThemeColors() {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    return {
        paper_bgcolor: isDark ? '#25262b' : '#ffffff',
        plot_bgcolor: isDark ? '#25262b' : '#ffffff',
        font_color: isDark ? '#e9ecef' : '#212529',
        grid_color: isDark ? '#373a40' : '#f1f3f5',
        related_works_marker: '#000000'
    };
}

const relatedWorksData = [
    { run_id: 'S2WAT (AAAI 2024)', clip_style: 0.6612154104808965, content_lpips: 0.5602266595166667, clip_content: 0.6941274672666667 },
    { run_id: 'SaMST（ACCV 2025）', clip_style: 0.7015254672368367, content_lpips: 0.5596446539666667, clip_content: 0.6903258735934894 },
    { run_id: 'StarGAN-v2', clip_style: 0.5850053535695089, content_lpips: 0.3471460113142112, clip_content: 0.8667943727225065 },
    { run_id: 'SDEdit str=0.35', clip_style: 0.661137411, content_lpips: 0.60866913 },
    { run_id: 'SDEdit str=0.40', clip_style: 0.661359812, content_lpips: 0.64001186 }
];

function asFloat(v) {
    const n = parseFloat(v);
    return Number.isNaN(n) ? null : n;
}

function oneMinus(v) {
    const n = asFloat(v);
    return n === null ? null : (1 - n);
}

function getNumericFromCandidates(row, keys) {
    for (const key of (keys || [])) {
        if (!key) continue;
        const n = asFloat(row[key]);
        if (n !== null) return n;
    }
    return null;
}

function getInvLpipsFromCandidates(row, candidates) {
    for (const c of (candidates || [])) {
        if (!c || !c.key) continue;
        const n = asFloat(row[c.key]);
        if (n === null) continue;
        return c.alreadyInverted ? n : (1 - n);
    }
    return null;
}

function parseCSV(text) {
    const lines = text.trim().split('\n').filter(Boolean);
    if (lines.length < 2) return [];
    const rawHeaders = lines[0].split(',').map(s => s.trim().replace(/^\uFEFF/, ''));
    const seen = new Map();
    const headers = rawHeaders.map((h, idx) => {
        const base = h || `col_${idx + 1}`;
        const n = seen.get(base) || 0;
        seen.set(base, n + 1);
        return n === 0 ? base : `${base}__dup${n + 1}`;
    });
    const out = [];

    for (let i = 1; i < lines.length; i++) {
        const parts = lines[i].split(',').map(s => s.trim());
        if (parts.length < headers.length) continue;
        const row = {};
        for (let j = 0; j < headers.length; j++) row[headers[j]] = parts[j];
        row.__rowIndex = i;
        out.push(row);
    }
    return out;
}

function findColumn(headers, candidates) {
    for (const c of candidates) {
        const hit = headers.find(h => h.trim().toLowerCase() === c);
        if (hit) return hit;
    }
    for (const c of candidates) {
        const hit = headers.find(h => h.trim().toLowerCase().includes(c));
        if (hit) return hit;
    }
    return null;
}

function findLpipsColumn(headers) {
    const inv = [
        '1-content_lpips', '1_content_lpips',
        'inv-content_lpips', 'inv_content_lpips',
        '1-all_content_lpips', '1_all_content_lpips',
        'inv-all_content_lpips', 'inv_all_content_lpips',
        '1-transfer_content_lpips', '1_transfer_content_lpips',
        'inv-transfer_content_lpips', 'inv_transfer_content_lpips',
        'inv-lpips', 'inv_lpips', '1-lpips', '1_lpips'
    ];
    const raw = ['content_lpips', 'lpips', 'all_content_lpips', 'transfer_content_lpips'];

    const invKey = findColumn(headers, inv);
    if (invKey) return { key: invKey, alreadyInverted: true };
    const rawKey = findColumn(headers, raw);
    if (rawKey) return { key: rawKey, alreadyInverted: false };
    return null;
}

function showError(msg) {
    document.getElementById('errorContainer').innerHTML = `<div class="error">${msg}</div>`;
}

function clearError() {
    document.getElementById('errorContainer').innerHTML = '';
}

function populateGroupCheckboxes(groups) {
    allGroups = new Set(Array.from(groups).sort());
    enabledGroups = new Set(Array.from(enabledGroups).filter(g => allGroups.has(g)));

    if (!hasInitializedGroups && enabledGroups.size === 0) {
        enabledGroups = new Set(allGroups);
        hasInitializedGroups = true;
    }

    const container = document.getElementById('groupCheckboxes');
    container.innerHTML = '';

    const { isVibrant } = getUiOptions();
    const activePalette = isVibrant ? vibrantPalette : standardPalette;

    const sorted = Array.from(allGroups).sort();
    
    groupColorMap.clear();
    let seriesColorIndex = new Map();
    let colorCounter = 0;
    
    sorted.forEach((group) => {
        const seriesName = getSeriesName(group);
        if (!seriesColorIndex.has(seriesName)) {
            seriesColorIndex.set(seriesName, colorCounter++);
        }
        const color = activePalette[seriesColorIndex.get(seriesName) % activePalette.length];
        groupColorMap.set(group, color);
    });

    sorted.forEach((group, idx) => {
        const color = groupColorMap.get(group);
        const row = document.createElement('label');
        row.style.display = 'flex';
        row.style.alignItems = 'center';
        row.style.gap = '8px';
        row.style.fontSize = '12px';
        row.style.cursor = 'pointer';
        row.className = 'plot-legend-item';

        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = enabledGroups.has(group);
        cb.onchange = () => {
            if (cb.checked) enabledGroups.add(group); else enabledGroups.delete(group);
            plotData();
        };

        const swatch = document.createElement('span');
        swatch.className = 'plot-legend-swatch';
        swatch.style.backgroundColor = color;
        swatch.style.borderRadius = '50%';

        const text = document.createElement('span');
        text.textContent = group;
        text.style.color = 'var(--text-primary)';

        row.appendChild(cb);
        row.appendChild(swatch);
        row.appendChild(text);
        container.appendChild(row);
    });

    document.getElementById('statGroupCount').textContent = String(sorted.length);
}

function populateRelatedWorksCheckboxes() {
    allRelatedWorks = new Set(relatedWorksData.map(rw => rw.run_id));
    enabledRelatedWorks = new Set(Array.from(enabledRelatedWorks).filter(name => allRelatedWorks.has(name)));

    if (!hasInitializedRelatedWorks && enabledRelatedWorks.size === 0) {
        enabledRelatedWorks = new Set(allRelatedWorks);
        hasInitializedRelatedWorks = true;
    }

    const container = document.getElementById('relatedWorksCheckboxes');
    if (!container) return;
    container.innerHTML = '';

    const colors = getCurrentThemeColors();
    const sorted = Array.from(allRelatedWorks).sort();
    sorted.forEach((name) => {
        const row = document.createElement('label');
        row.style.display = 'flex';
        row.style.alignItems = 'center';
        row.style.gap = '8px';
        row.style.fontSize = '12px';
        row.style.cursor = 'pointer';
        row.className = 'plot-legend-item';

        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = enabledRelatedWorks.has(name);
        cb.onchange = () => {
            if (cb.checked) enabledRelatedWorks.add(name); else enabledRelatedWorks.delete(name);
            plotData();
        };

        const swatch = document.createElement('span');
        swatch.className = 'plot-legend-swatch';
        swatch.innerHTML = '✖';
        swatch.style.color = colors.related_works_marker;
        swatch.style.backgroundColor = 'transparent';
        swatch.style.display = 'flex';
        swatch.style.alignItems = 'center';
        swatch.style.justifyContent = 'center';
        swatch.style.fontSize = '14px';

        const text = document.createElement('span');
        text.textContent = name;
        text.style.color = 'var(--text-primary)';

        row.appendChild(cb);
        row.appendChild(swatch);
        row.appendChild(text);
        container.appendChild(row);
    });
}

function selectAllGroups() {
    if (allGroups.size === 0) return;
    enabledGroups = new Set(allGroups);
    hasInitializedGroups = true;
    plotData();
}

function selectNoneGroups() {
    if (allGroups.size === 0) return;
    enabledGroups.clear();
    hasInitializedGroups = true;
    plotData();
}

function selectAllRelatedWorks() {
    if (allRelatedWorks.size === 0) return;
    enabledRelatedWorks = new Set(allRelatedWorks);
    hasInitializedRelatedWorks = true;
    plotData();
}

function selectNoneRelatedWorks() {
    if (allRelatedWorks.size === 0) return;
    enabledRelatedWorks.clear();
    hasInitializedRelatedWorks = true;
    plotData();
}

function buildGroupSeries(records, cols, metric) {
    const map = {};
    let expCount = 0;

    for (const r of records) {
        const x = getNumericFromCandidates(r, cols.clipStyleCandidates);
        const y = metric === 'clip_content'
            ? getNumericFromCandidates(r, cols.clipContentCandidates)
            : getInvLpipsFromCandidates(r, cols.lpipsCandidates);

        if (x === null || y === null) continue;

        const group = String(r[cols.expId] || '').trim();
        if (!group) continue;
        if (!enabledGroups.has(group)) continue;

        const epochRaw = r[cols.epoch];
        const epochNum = asFloat(epochRaw);
        const order = epochNum === null ? Number.MAX_SAFE_INTEGER : epochNum;
        const label = epochRaw ? `${group} (${epochRaw})` : group;

        if (!map[group]) map[group] = [];
        map[group].push({ x, y, label, order, row: r.__rowIndex });
        expCount += 1;
    }

    Object.keys(map).forEach(g => {
        map[g].sort((a, b) => {
            if (a.order !== b.order) return a.order - b.order;
            return a.row - b.row;
        });
    });

    return { map, expCount };
}

const colorUtils = {
    hexToHsl: function(hex) {
        let r = parseInt(hex.slice(1,3), 16) / 255;
        let g = parseInt(hex.slice(3,5), 16) / 255;
        let b = parseInt(hex.slice(5,7), 16) / 255;
        let max = Math.max(r, g, b), min = Math.min(r, g, b);
        let h, s, l = (max + min) / 2;
        if (max === min) { h = s = 0; } 
        else {
            let d = max - min;
            s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
            switch(max) {
                case r: h = (g - b) / d + (g < b ? 6 : 0); break;
                case g: h = (b - r) / d + 2; break;
                case b: h = (r - g) / d + 4; break;
            }
            h /= 6;
        }
        return [h, s, l];
    },
    hslToRgbString: function(h, s, l) {
        let r, g, b;
        if (s === 0) {
            r = g = b = l; 
        } else {
            const hue2rgb = (p, q, t) => {
                if (t < 0) t += 1;
                if (t > 1) t -= 1;
                if (t < 1/6) return p + (q - p) * 6 * t;
                if (t < 1/2) return q;
                if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
                return p;
            };
            const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
            const p = 2 * l - q;
            r = hue2rgb(p, q, h + 1/3);
            g = hue2rgb(p, q, h);
            b = hue2rgb(p, q, h - 1/3);
        }
        return `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;
    }
};

function buildMarkerColors(points, activePalette, baseColor, multiColorWithinGroup) {
    if (!multiColorWithinGroup || points.length <= 1) {
        return baseColor;
    }

    if (!baseColor.startsWith('#') || baseColor.length !== 7) {
        return points.map(() => baseColor);
    }

    const [h, s, l] = colorUtils.hexToHsl(baseColor);
    const n = points.length;
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    
    return points.map((_, idx) => {
        const t = idx / (n - 1); 
        
        // Shift hue by -45 degrees for earlier points to create a beautiful analogous color tail
        let h_new = h - (1 - t) * (45 / 360);
        if (h_new < 0) h_new += 1;
        
        // Blend lightness towards background for earlier points
        let targetL = isDark ? 0.2 : 0.9;
        let l_new = l + (1 - t) * (targetL - l) * 0.6;
        
        // Slightly desaturate earlier points
        let s_new = s * (0.4 + 0.6 * t);
        
        return colorUtils.hslToRgbString(h_new, s_new, l_new);
    });
}

function calcAutoRange(values, padRatio = 0.12) {
    const valid = values.filter(v => Number.isFinite(v));
    if (valid.length === 0) return [0, 1];
    let min = Math.min(...valid);
    let max = Math.max(...valid);
    if (min === max) {
        const d = Math.max(Math.abs(min) * 0.05, 0.03);
        min -= d;
        max += d;
    } else {
        const span = max - min;
        const pad = Math.max(span * padRatio, 0.015);
        min -= pad;
        max += pad;
    }
    return [min, max];
}

function getRangesFromTraces(traces) {
    const xs = [];
    const ys = [];
    traces.forEach(t => {
        (t.x || []).forEach(v => xs.push(v));
        (t.y || []).forEach(v => ys.push(v));
    });
    return {
        xRange: calcAutoRange(xs),
        yRange: calcAutoRange(ys)
    };
}

function resolveAxisRange(autoRange, minId, maxId) {
    const minVal = asFloat(document.getElementById(minId)?.value);
    const maxVal = asFloat(document.getElementById(maxId)?.value);

    let lo = autoRange[0];
    let hi = autoRange[1];

    if (minVal !== null) lo = minVal;
    if (maxVal !== null) hi = maxVal;

    if (lo >= hi) {
        if (maxVal === null && minVal !== null) hi = lo + 0.1;
        else if (minVal === null && maxVal !== null) lo = hi - 0.1;
        else return null;
    }
    return [lo, hi];
}

function resetAxisLimits() {
    const defaults = {
        'p1xMin': '0.35', 'p1xMax': '0.7', 'p1yMin': '0.55', 'p1yMax': '',
        'p2xMin': '0.35', 'p2xMax': '0.7', 'p2yMin': '0.55', 'p2yMax': ''
    };
    Object.keys(defaults).forEach(id => {
        const el = document.getElementById(id);
        if (el) el.value = defaults[id];
    });
    plotData();
}

function validateAxisInputs() {
    const checks = [
        ['p1xMin', 'p1xMax', 'Plot 1 X'],
        ['p1yMin', 'p1yMax', 'Plot 1 Y'],
        ['p2xMin', 'p2xMax', 'Plot 2 X'],
        ['p2yMin', 'p2yMax', 'Plot 2 Y']
    ];
    for (const [minId, maxId, label] of checks) {
        const minVal = asFloat(document.getElementById(minId)?.value);
        const maxVal = asFloat(document.getElementById(maxId)?.value);
        if (minVal !== null && maxVal !== null && !(minVal < maxVal)) {
            return `${label} limit invalid: min must be less than max.`;
        }
    }
    return null;
}

function buildTraces(groupMap, metric) {
    const traces = [];
    const groups = Object.keys(groupMap).sort();
    const colors = getCurrentThemeColors();
    const { showLines, isVibrant, multiColorWithinGroup } = getUiOptions();
    const activePalette = isVibrant ? vibrantPalette : standardPalette;

    groups.forEach((group) => {
        const pts = groupMap[group];
        const traceColor = groupColorMap.get(group) || activePalette[0];
        
        traces.push({
            x: pts.map(p => p.y),
            y: pts.map(p => p.x),
            customdata: pts.map(p => p.label),
            mode: showLines ? 'lines+markers' : 'markers',
            type: 'scatter',
            name: group,
            line: { width: 2, color: traceColor },
            marker: { 
                size: 8, 
                color: buildMarkerColors(pts, activePalette, traceColor, multiColorWithinGroup)
            },
            hovertemplate: metric === 'clip_content'
                ? '<b>%{customdata}</b><br>clip_content: %{x:.4f}<br>clip_style: %{y:.4f}<extra></extra>'
                : '<b>%{customdata}</b><br>1-content_lpips: %{x:.4f}<br>clip_style: %{y:.4f}<extra></extra>'
        });
    });

    const rwPts = relatedWorksData
        .filter(rw => enabledRelatedWorks.has(rw.run_id))
        .map((rw) => {
            const x = metric === 'clip_content' ? asFloat(rw.clip_content) : oneMinus(rw.content_lpips);
            const y = asFloat(rw.clip_style);
            return { x, y, label: rw.run_id };
        })
        .filter(p => p.x !== null && p.y !== null);

    if (rwPts.length > 0) {
        const rwTextPositions = ['top center', 'bottom center', 'top right', 'bottom right', 'top left', 'bottom left'];
        traces.push({
            x: rwPts.map(p => p.x),
            y: rwPts.map(p => p.y),
            customdata: rwPts.map(p => p.label),
            text: rwPts.map(p => p.label),
            textposition: rwPts.map((_, idx) => rwTextPositions[idx % rwTextPositions.length]),
            textfont: { color: colors.font_color, size: 10, weight: 600 },
            mode: 'markers+text',
            type: 'scatter',
            name: 'Related Works',
            marker: {
                symbol: 'x',
                color: colors.related_works_marker,
                size: 10,
                line: { width: 2, color: colors.related_works_marker }
            },
            hovertemplate: metric === 'clip_content'
                ? '<b>%{customdata}</b><br>clip_content: %{x:.4f}<br>clip_style: %{y:.4f}<extra></extra>'
                : '<b>%{customdata}</b><br>1-content_lpips: %{x:.4f}<br>clip_style: %{y:.4f}<extra></extra>'
        });
    }

    return traces;
}

function renderLegend(legendId, traces) {
    const el = document.getElementById(legendId);
    if (!el) return;
    if (!traces || traces.length === 0) {
        el.innerHTML = '<div class="plot-legend-title">Legend</div><div class="scroll-list"><div class="plot-legend-item">No visible traces</div></div>';
        return;
    }

    const rows = traces.map((t) => {
        if (t.name === 'Related Works') {
            return `
                <div class="plot-legend-item">
                    <span class="plot-legend-swatch" style="background:transparent; color:${t.marker.color}; font-size:14px; display:flex; align-items:center; justify-content:center;">✖</span>
                    <span style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${t.name || 'trace'}</span>
                </div>
            `;
        }
        const markerColor = t?.marker?.color || t?.line?.color || '#666';
        return `
            <div class="plot-legend-item">
                <span class="plot-legend-swatch" style="background:${markerColor};"></span>
                <span style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${t.name || 'trace'}</span>
            </div>
        `;
    }).join('');

    el.innerHTML = `<div class="plot-legend-title">Legend</div><div class="scroll-list" style="border:none;">${rows}</div>`;
}

function renderPlot(divId, legendId, traces, metric) {
    const autoRanges = getRangesFromTraces(traces);
    const colors = getCurrentThemeColors();
    const ids = metric === 'clip_content'
        ? { xMin: 'p1xMin', xMax: 'p1xMax', yMin: 'p1yMin', yMax: 'p1yMax' }
        : { xMin: 'p2xMin', xMax: 'p2xMax', yMin: 'p2yMin', yMax: 'p2yMax' };

    const xRange = resolveAxisRange(autoRanges.xRange, ids.xMin, ids.xMax) || autoRanges.xRange;
    const yRange = resolveAxisRange(autoRanges.yRange, ids.yMin, ids.yMax) || autoRanges.yRange;

    const layout = {
        title: {
            text: metric === 'clip_content' ? 'clip_content vs clip_style' : '1-content_lpips vs clip_style',
            font: { size: 14, color: colors.font_color, weight: 900 }
        },
        xaxis: {
            title: { text: metric === 'clip_content' ? 'clip_content' : '1-content_lpips', font: { size: 11, color: colors.font_color } },
            range: xRange,
            showgrid: true,
            gridcolor: colors.grid_color,
            tickfont: { color: colors.font_color, size: 10 }
        },
        yaxis: {
            title: { text: 'clip_style', font: { size: 11, color: colors.font_color } },
            range: yRange,
            showgrid: true,
            gridcolor: colors.grid_color,
            tickfont: { color: colors.font_color, size: 10 }
        },
        hovermode: 'closest',
        plot_bgcolor: colors.plot_bgcolor,
        paper_bgcolor: colors.paper_bgcolor,
        font: { color: colors.font_color, family: 'var(--font-mono)' },
        margin: { l: 60, r: 20, t: 50, b: 50 },
        showlegend: false
    };

    const config = {
        responsive: true,
        displaylogo: false,
        toImageButtonOptions: {
            format: 'png',
            filename: `scatter_${metric}`,
            height: 1000,
            width: 1500,
            scale: 2
        }
    };

    Plotly.newPlot(divId, traces, layout, config);
    renderLegend(legendId, traces);
}

function inferColumns(records) {
    const headers = Object.keys(records[0]);

    const expId = findColumn(headers, ['experiment_id']);
    const epoch = findColumn(headers, ['epoch']);
    const clipStyleUnified = findColumn(headers, ['clip_style']);
    const clipStyleAll = findColumn(headers, ['all_clip_style']);
    const clipStyleTransfer = findColumn(headers, ['transfer_clip_style']);
    const clipContentUnified = findColumn(headers, ['clip_content']);
    const clipContentAll = findColumn(headers, ['all_clip_content']);
    const clipContentTransfer = findColumn(headers, ['transfer_clip_content', 'clip_dir']);

    const lpipsUnifiedInv = findColumn(headers, [
        '1-content_lpips', '1_content_lpips',
        'inv-content_lpips', 'inv_content_lpips'
    ]);
    const lpipsUnifiedRaw = findColumn(headers, ['content_lpips', 'lpips']);
    const lpipsAllInv = findColumn(headers, [
        '1-all_content_lpips', '1_all_content_lpips',
        'inv-all_content_lpips', 'inv_all_content_lpips'
    ]);
    const lpipsAllRaw = findColumn(headers, ['all_content_lpips']);
    const lpipsTransferInv = findColumn(headers, [
        '1-transfer_content_lpips', '1_transfer_content_lpips',
        'inv-transfer_content_lpips', 'inv_transfer_content_lpips'
    ]);
    const lpipsTransferRaw = findColumn(headers, ['transfer_content_lpips']);

    const clipStyleCandidates = [clipStyleUnified, clipStyleAll, clipStyleTransfer].filter(Boolean);
    const clipContentCandidates = [clipContentUnified, clipContentAll, clipContentTransfer].filter(Boolean);
    const lpipsCandidates = [
        { key: lpipsUnifiedInv, alreadyInverted: true },
        { key: lpipsUnifiedRaw, alreadyInverted: false },
        { key: lpipsAllInv, alreadyInverted: true },
        { key: lpipsAllRaw, alreadyInverted: false },
        { key: lpipsTransferInv, alreadyInverted: true },
        { key: lpipsTransferRaw, alreadyInverted: false },
    ].filter(x => x.key);

    if (!expId) throw new Error('Missing experiment_id column');
    if (clipStyleCandidates.length === 0) throw new Error('Missing clip_style column');
    if (clipContentCandidates.length === 0) throw new Error('Missing clip_content column');
    if (lpipsCandidates.length === 0) throw new Error('Missing content_lpips column');

    return {
        expId,
        epoch: epoch || '__missing_epoch__',
        clipStyleCandidates,
        clipContentCandidates,
        lpipsCandidates
    };
}

function plotData() {
    clearError();
    const axisError = validateAxisInputs();
    if (axisError) {
        showError(axisError);
        return;
    }
    if (!hasInitializedRelatedWorks) populateRelatedWorksCheckboxes();

    const text = document.getElementById('csvInput').value.trim();
    if (!text) {
        renderRelatedOnly();
        document.getElementById('statExpCount').textContent = '0';
        document.getElementById('statRwCount').textContent = String(enabledRelatedWorks.size);
        return;
    }

    const records = parseCSV(text);
    if (records.length === 0) {
        showError('No valid rows in CSV');
        return;
    }

    let cols;
    try {
        cols = inferColumns(records);
    } catch (e) {
        showError(e.message);
        return;
    }

    const { hideDistill } = getUiOptions();
    const visibleRecords = hideDistill
        ? records.filter(r => !String(r[cols.expId] || '').toLowerCase().includes('distill'))
        : records;

    const groups = new Set(visibleRecords.map(r => String(r[cols.expId] || '').trim()).filter(Boolean));
    populateGroupCheckboxes(groups);

    const clipContentSeries = buildGroupSeries(visibleRecords, cols, 'clip_content');
    const invLpipsSeries = buildGroupSeries(visibleRecords, cols, 'inv_lpips');

    const clipContentTraces = buildTraces(clipContentSeries.map, 'clip_content');
    const invLpipsTraces = buildTraces(invLpipsSeries.map, 'inv_lpips');

    renderPlot('plotClipContent', 'legendClipContent', clipContentTraces, 'clip_content');
    renderPlot('plotInvLpips', 'legendInvLpips', invLpipsTraces, 'inv_lpips');

    document.getElementById('statExpCount').textContent = String(Math.max(clipContentSeries.expCount, invLpipsSeries.expCount));
    document.getElementById('statRwCount').textContent = String(enabledRelatedWorks.size);
}

function renderRelatedOnly() {
    renderPlot('plotClipContent', 'legendClipContent', buildTraces({}, 'clip_content'), 'clip_content');
    renderPlot('plotInvLpips', 'legendInvLpips', buildTraces({}, 'inv_lpips'), 'inv_lpips');
}

function clearData() {
    document.getElementById('csvInput').value = '';
    document.getElementById('groupCheckboxes').innerHTML = '';
    document.getElementById('relatedWorksCheckboxes').innerHTML = '';
    allGroups.clear();
    enabledGroups.clear();
    hasInitializedGroups = false;
    allRelatedWorks.clear();
    enabledRelatedWorks.clear();
    hasInitializedRelatedWorks = false;

    document.getElementById('statExpCount').textContent = '0';
    document.getElementById('statRwCount').textContent = String(relatedWorksData.length);
    document.getElementById('statGroupCount').textContent = '0';

    clearError();
    populateRelatedWorksCheckboxes();
    renderRelatedOnly();
}

function appendCSVText(newText) {
    const textarea = document.getElementById('csvInput');
    const existingText = textarea.value.trim();
    
    if (!existingText) {
        textarea.value = newText;
    } else {
        const existingLines = existingText.split('\n');
        const newLines = newText.trim().split('\n');
        
        if (existingLines.length > 0 && newLines.length > 0) {
            // Check if the first line of newText is the same as existing header
            if (existingLines[0].trim() === newLines[0].trim()) {
                newLines.shift(); // Remove header from new data
            }
        }
        
        if (newLines.length > 0) {
            textarea.value = existingText + '\n' + newLines.join('\n');
        }
    }
    plotData();
}

function handleFiles(files) {
    for (const file of files) {
        const reader = new FileReader();
        reader.onload = (e) => {
            appendCSVText(e.target.result);
        };
        reader.readAsText(file);
    }
}

function setupFileUpload() {
    const fileInput = document.getElementById('csvFileInput');
    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleFiles(e.target.files);
            }
            fileInput.value = ''; // Reset
        });
    }

    const dropZone = document.getElementById('csvDropZone');
    if (dropZone) {
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('drag-over');
        });
        dropZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
        });
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleFiles(e.dataTransfer.files);
            }
        });
    }
}

window.addEventListener('load', async () => {
    // Initialize theme
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeToggleBtn();

    populateRelatedWorksCheckboxes();
    renderRelatedOnly();
    setupFileUpload();
});

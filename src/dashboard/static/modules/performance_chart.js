import { escapeHtml } from './html_utils.js?v=1.0';

const SERIES_NAME = 'Gain since first trade';
const RANGE_MS = { '7': 7 * 24 * 3600 * 1000, '30': 30 * 24 * 3600 * 1000 };
const ENTRY_ACTIONS = new Set(['BUY', 'LONG', 'SELL', 'SHORT']);

let chart;
let points = [];
let appliedHeight = 0;
let activeRange = 'all';
let lastSignature = null;

function token(name, fallback) {
    if (typeof getComputedStyle !== 'function') return fallback;
    const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
}

function chartHeight() {
    return window.innerWidth <= 768 ? 260 : 320;
}

function formatPct(value, digits) {
    const rounded = Number(value.toFixed(digits));
    if (!rounded) return '0%';
    return `${rounded > 0 ? '+' : '-'}${Math.abs(rounded).toFixed(digits)}%`;
}

function formatMoney(value) {
    return new Intl.NumberFormat(navigator.language, {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value);
}

function formatStamp(value) {
    return new Intl.DateTimeFormat(navigator.language, { dateStyle: 'medium', timeStyle: 'short' })
        .format(new Date(value));
}

function formatTick(timestamp) {
    const date = new Date(timestamp);
    const day = new Intl.DateTimeFormat(navigator.language, { day: '2-digit', month: 'short' }).format(date);
    if (activeRange !== '7') return day;
    const time = new Intl.DateTimeFormat(navigator.language, { hour: '2-digit', minute: '2-digit' }).format(date);
    return `${day} ${time}`;
}

function buildPoints(history) {
    const first = Number((history[0] || {}).value);
    if (!Number.isFinite(first) || first === 0) return [];
    const built = [];
    let previous = 0;
    history.forEach((entry) => {
        const value = Number(entry.value);
        const time = new Date(entry.time).getTime();
        if (!Number.isFinite(value) || !Number.isFinite(time)) return;
        const gain = (value / first - 1) * 100;
        built.push({ x: time, y: gain, delta: gain - previous, extra: entry });
        previous = gain;
    });
    return built;
}

function axisBounds(list) {
    if (!list.length) return { min: -1, max: 1 };
    let lowest = 0;
    let highest = 0;
    list.forEach((point) => {
        lowest = Math.min(lowest, point.y);
        highest = Math.max(highest, point.y);
    });
    const padding = Math.max((highest - lowest) * 0.12, 0.05);
    return { min: lowest - padding, max: highest + padding };
}

function rangeBounds(list) {
    if (!list.length) return { min: undefined, max: undefined };
    const first = list[0].x;
    const last = list[list.length - 1].x;
    const span = RANGE_MS[activeRange];
    const start = span ? Math.max(first, last - span) : first;
    const padding = Math.max((last - start) * 0.02, 3600 * 1000);
    return { min: start - padding, max: last + padding };
}

function toAnnotation(point) {
    const delta = Number(point.delta);
    const color = delta > 0
        ? token('--accent-success', '#3fb950')
        : delta < 0
            ? token('--accent-danger', '#f85149')
            : token('--text-muted', '#97a3b3');
    return {
        x: point.x,
        y: point.y,
        marker: {
            size: 4,
            fillColor: color,
            strokeColor: token('--bg-dark-2', '#161b22'),
            strokeWidth: 1.5,
            shape: 'circle'
        }
    };
}

function toEntryAnnotation(point) {
    return {
        x: point.x,
        y: point.y,
        marker: {
            size: 4,
            fillColor: 'transparent',
            strokeColor: token('--text-muted', '#97a3b3'),
            strokeWidth: 1.5,
            shape: 'circle'
        }
    };
}

function tooltipMarkup(point) {
    if (!point) return '';
    const extra = point.extra || {};
    const rows = [];
    if (extra.action) {
        const action = String(extra.action);
        const direction = point.delta > 0 ? ' is-up' : point.delta < 0 ? ' is-down' : '';
        const price = Number(extra.price);
        rows.push(`<div class="chart-tip-row"><span class="chart-tip-action${direction}">${escapeHtml(action)}</span><span>${Number.isFinite(price) ? escapeHtml(formatMoney(price)) : ''}</span></div>`);
    }
    if (Math.abs(Number(point.delta) || 0) >= 0.005) {
        rows.push(`<div class="chart-tip-row"><span>Since previous</span><span>${escapeHtml(formatPct(point.delta, 2))}</span></div>`);
    }
    return `<div class="chart-tip">
        <div class="chart-tip-date">${escapeHtml(formatStamp(point.x))}</div>
        <div class="chart-tip-value">${escapeHtml(formatPct(point.y, 2))}</div>
        ${rows.join('')}
    </div>`;
}

function xAxisOptions() {
    return {
        type: 'datetime',
        labels: {
            datetimeUTC: false,
            formatter: (value, timestamp) => formatTick(timestamp ?? value),
            style: { colors: [token('--text-muted', '#97a3b3')], fontSize: '11px' },
            rotate: 0,
            hideOverlappingLabels: true
        },
        axisBorder: { show: false },
        axisTicks: { show: false },
        tooltip: { enabled: false },
        crosshairs: {
            show: true,
            stroke: { color: token('--border-glow', 'rgba(122,162,247,0.35)'), width: 1, dashArray: 3 }
        }
    };
}

function yAxisOptions() {
    return {
        tickAmount: 4,
        labels: {
            formatter: (value) => formatPct(value, 1),
            style: { colors: [token('--text-muted', '#97a3b3')], fontSize: '11px' }
        }
    };
}

function buildOptions() {
    return {
        series: [{ name: SERIES_NAME, data: [] }],
        chart: {
            type: 'area',
            height: chartHeight(),
            background: 'transparent',
            toolbar: { show: false },
            zoom: { enabled: true, type: 'x' },
            animations: { enabled: true, easing: 'easeinout', speed: 350 },
            fontFamily: 'Inter, system-ui, sans-serif',
            parentHeightOffset: 0
        },
        colors: [token('--accent-primary', '#7aa2f7')],
        dataLabels: { enabled: false },
        markers: { size: 0, strokeWidth: 0, hover: { size: 4 } },
        stroke: { curve: 'straight', width: 2 },
        fill: {
            type: 'gradient',
            gradient: { shadeIntensity: 0, opacityFrom: 0.2, opacityTo: 0.01, stops: [0, 100] }
        },
        theme: { mode: 'dark' },
        grid: {
            borderColor: token('--border-subtle', 'rgba(255,255,255,0.08)'),
            strokeDashArray: 4,
            xaxis: { lines: { show: false } },
            yaxis: { lines: { show: true } },
            padding: { top: 12, right: 14, bottom: 0, left: 6 }
        },
        xaxis: xAxisOptions(),
        yaxis: [yAxisOptions()],
        tooltip: {
            theme: 'dark',
            marker: { show: false },
            x: { show: false },
            custom: ({ dataPointIndex, w }) => tooltipMarkup(w.config.series[0].data[dataPointIndex])
        },
        annotations: { points: [] },
        noData: {
            text: 'No closed trades yet',
            style: { color: token('--text-muted', '#97a3b3'), fontSize: '13px' }
        }
    };
}

function updateRangeButtons() {
    document.querySelectorAll('.chart-range [data-range]').forEach((button) => {
        const active = button.dataset.range === activeRange;
        button.classList.toggle('is-active', active);
        button.setAttribute('aria-pressed', String(active));
    });
}

function applyRange() {
    if (!chart || !points.length) return;
    chart.updateOptions({
        xaxis: { ...xAxisOptions(), ...rangeBounds(points) },
        yaxis: [{ ...yAxisOptions(), ...axisBounds(points) }]
    }, false, false);
}

function bindRangeButtons() {
    document.querySelectorAll('.chart-range [data-range]').forEach((button) => {
        button.addEventListener('click', () => {
            activeRange = button.dataset.range;
            updateRangeButtons();
            applyRange();
        });
    });
}

function renderSeries(history) {
    points = buildPoints(history);
    if (!points.length) {
        chart.updateSeries([{ name: SERIES_NAME, data: [] }], false);
        return;
    }
    const actionOf = (point) => String((point.extra || {}).action || '').toUpperCase();
    const markers = points.filter((point) => actionOf(point).includes('CLOSE')).map(toAnnotation);
    const entries = points.filter((point) => ENTRY_ACTIONS.has(actionOf(point))).map(toEntryAnnotation);
    chart.updateSeries([{ name: SERIES_NAME, data: points }], false);
    chart.updateOptions({ annotations: { points: [...entries, ...markers] } }, false, false);
    applyRange();
}

function renderStats(stats) {
    const element = document.getElementById('stats-summary');
    if (!element || !stats) return;
    const pnl = Number(stats.total_pnl_pct) || 0;
    const win = Number(stats.win_rate) || 0;
    const items = [
        ['Trades', String(stats.total_trades || 0), ''],
        ['Win rate', `${win.toFixed(1)}%`, win >= 50 ? 'is-up' : 'is-down'],
        ['P&L', `${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}%`, pnl >= 0 ? 'is-up' : 'is-down'],
        ['Capital', formatMoney(Number(stats.current_capital) || 0), '']
    ];
    element.innerHTML = items
        .map(([label, value, modifier]) => `<div class="stat-strip-item"><span class="stat-strip-label">${label}</span><span class="stat-strip-value ${modifier}">${value}</span></div>`)
        .join('');
}

export function initPerformanceChart() {
    const container = document.getElementById('performance-chart');
    if (!container || typeof ApexCharts === 'undefined') return;
    appliedHeight = chartHeight();
    chart = new ApexCharts(container, buildOptions());
    chart.render();
    window.performanceChart = chart;
    bindRangeButtons();
    updateRangeButtons();
    window.addEventListener('resize', () => {
        const next = chartHeight();
        if (next === appliedHeight) return;
        appliedHeight = next;
        chart.updateOptions({ chart: { height: next } });
    });
}

export async function updatePerformanceData() {
    if (!chart) return;
    try {
        const [historyResponse, statsResponse] = await Promise.all([
            fetch('/api/performance/history'),
            fetch('/api/performance/stats')
        ]);
        const historyData = await historyResponse.json();
        const stats = await statsResponse.json();
        const history = Array.isArray(historyData.history) ? historyData.history : [];
        const last = history[history.length - 1] || {};
        const signature = history.length ? `${history.length}|${last.time}|${last.value}` : '0';
        if (signature !== lastSignature) {
            lastSignature = signature;
            renderSeries(history);
        }
        renderStats(stats);
    } catch (error) {
        console.error('Failed to update performance chart', error);
    }
}

export function resetPerformanceChartSize() {
    if (!chart) return;
    appliedHeight = chartHeight();
    chart.updateOptions({ chart: { height: appliedHeight, width: undefined } }, false, true);
}

window.refreshPerformanceAnnotations = updatePerformanceData;
window.resetPerformanceChartSize = resetPerformanceChartSize;

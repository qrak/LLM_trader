import { escapeHtml } from './html_utils.js?v=1.0';

const DECISION_ENDPOINT =
    '/api/brain/decision-summary?experience_limit=5&rule_limit=5&journal_limit=5&blocked_limit=5';

let lastError = null;

export async function initDecisionPanel() {
    window.updateDecisionPanel = updateDecisionPanel;
    await updateDecisionPanel();
}

export async function updateDecisionPanel() {
    const verdict = document.getElementById('decision-verdict');
    if (!verdict) return;
    try {
        const response = await fetch(DECISION_ENDPOINT);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        lastError = null;
        renderVerdict(data.synopsis_data || {});
        renderMeta(data);
        renderContext(data);
        renderFriction(data);
    } catch (error) {
        console.error('Failed to update the decision panel', error);
        lastError = error.message || String(error);
        verdict.innerHTML = `<span class="verdict-plain">Decision data unavailable: ${escapeHtml(lastError)}</span>`;
        const context = document.getElementById('decision-context');
        if (context) context.innerHTML = '';
    }
}

function positionLabel(synopsis) {
    if (!synopsis.has_position) return 'Flat, no open position';
    const direction = synopsis.direction ? String(synopsis.direction).toUpperCase() : 'OPEN';
    const symbol = synopsis.symbol ? ` ${synopsis.symbol}` : '';
    const entry = synopsis.entry_price ? ` @ ${synopsis.entry_price}` : '';
    return `${direction}${symbol}${entry}`;
}

function renderVerdict(synopsis) {
    const container = document.getElementById('decision-verdict');
    if (!container) return;
    const parts = [];
    parts.push(`<strong class="verdict-action">${escapeHtml(synopsis.action || '--')}</strong>`);
    if (synopsis.confidence !== null && synopsis.confidence !== undefined && synopsis.confidence !== '--') {
        parts.push(`<span>${escapeHtml(String(synopsis.confidence))}% confidence</span>`);
    }
    if (synopsis.trend && synopsis.trend !== '--') {
        parts.push(`<span class="verdict-plain">Trend ${escapeHtml(String(synopsis.trend))}</span>`);
    }
    parts.push(`<span class="verdict-plain">${escapeHtml(positionLabel(synopsis))}</span>`);
    container.innerHTML = DOMPurify.sanitize(parts.join(''));
}

function formatClock(iso) {
    if (!iso) return '';
    const parsed = new Date(iso);
    if (Number.isNaN(parsed.getTime())) return '';
    return new Intl.DateTimeFormat(navigator.language, { timeStyle: 'medium' }).format(parsed);
}

function formatDate(iso) {
    if (!iso) return '';
    const parsed = new Date(iso);
    if (Number.isNaN(parsed.getTime())) return '';
    return new Intl.DateTimeFormat(navigator.language, { dateStyle: 'short', timeStyle: 'short' }).format(parsed);
}

function renderMeta(data) {
    const meta = document.getElementById('decision-meta');
    if (!meta) return;
    const clock = formatClock(data.generated_at);
    meta.textContent = clock ? `Updated ${clock}` : '';
}

function formatMetric(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed.toFixed(1) : String(value);
}

function contextContent(data) {
    const synopsis = data.synopsis_data || {};
    const now = data.now || {};
    const tokens = String(synopsis.current_context || '')
        .split('+')
        .map((token) => token.trim())
        .filter(Boolean);
    if (!tokens.length) {
        return `<p class="chain-empty">No market context was recorded for this cycle.</p>`;
    }
    const chips = tokens.map((token) => `<span class="context-chip">${escapeHtml(token)}</span>`).join('');
    const metrics = [];
    if (now.adx !== null && now.adx !== undefined) metrics.push(`ADX ${escapeHtml(formatMetric(now.adx))}`);
    if (now.rsi !== null && now.rsi !== undefined) metrics.push(`RSI ${escapeHtml(formatMetric(now.rsi))}`);
    const metricLine = metrics.length
        ? `<p class="context-metrics">${metrics.join(' · ')}</p>`
        : '';
    return `<div class="context-chips">${chips}</div>${metricLine}`;
}

function renderContext(data) {
    const container = document.getElementById('decision-context');
    if (!container) return;
    container.innerHTML = DOMPurify.sanitize(contextContent(data));
}

function renderFriction(data) {
    const panel = document.getElementById('panel-friction');
    const content = document.getElementById('friction-content');
    const meta = document.getElementById('friction-meta');
    if (!panel || !content || !meta) return;
    const blocked = (data.memory && data.memory.blocked) || {};
    const items = blocked.items || [];
    const total = blocked.blocked_count ?? items.length;
    if (!total) {
        panel.hidden = true;
        content.innerHTML = '';
        meta.textContent = '';
        return;
    }
    panel.hidden = false;
    meta.textContent = `${total} recorded`;
    if (!items.length) {
        content.innerHTML = DOMPurify.sanitize(
            `<p class="chain-empty">Nothing blocked in the last seven days; the count above is all-time.</p>`
        );
        return;
    }
    const rows = items.map((item) => {
        const guard = String(item.guard_type || 'unknown guard').replace(/_/g, ' ');
        const rr = `${formatRatio(item.suggested_rr)} / ${formatRatio(item.required_rr)}`;
        return `
            <tr>
                <td>${escapeHtml(guard)}</td>
                <td>${escapeHtml(item.direction || '--')}</td>
                <td>${escapeHtml(item.confidence || '--')}</td>
                <td class="chain-num">${escapeHtml(rr)}</td>
                <td>${escapeHtml(formatDate(item.timestamp))}</td>
            </tr>
        `;
    }).join('');
    content.innerHTML = DOMPurify.sanitize(`
        <div class="vector-table compact">
            <table>
                <thead>
                    <tr><th scope="col">Guard</th><th scope="col">Direction</th><th scope="col">Confidence</th>
                        <th scope="col">R:R suggested / required</th><th scope="col">When</th></tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>
        </div>
    `);
}

function formatRatio(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed.toFixed(2) : '--';
}

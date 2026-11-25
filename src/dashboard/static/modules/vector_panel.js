import { escapeHtml } from './html_utils.js?v=1.0';

const RECALL_LIMIT = 5;
const STORE_LIMIT = 200;

const currentSort = {
    by: 'date',
    order: 'desc'
};

let showAll = false;

const CONTEXT_PILLS = [
    { label: 'BULLISH', cls: 'pill-bullish' },
    { label: 'BEARISH', cls: 'pill-bearish' },
    { label: 'STRONG_TREND', cls: 'pill-strong-trend' },
    { label: 'TRENDING', cls: 'pill-strong-trend' },
    { label: 'RANGING', cls: 'pill-low-vol' },
    { label: 'OVERSOLD', cls: 'pill-bullish' },
    { label: 'OVERBOUGHT', cls: 'pill-bearish' },
    { label: 'HIGH VOL', cls: 'pill-high-vol' },
    { label: 'LOW VOL', cls: 'pill-low-vol' },
];

export async function initVectorPanel() {
    window.updateVectorData = updateVectorData;
    window.vectorSort = async (field) => {
        showAll = true;
        if (currentSort.by === field) {
            currentSort.order = currentSort.order === 'desc' ? 'asc' : 'desc';
        } else {
            currentSort.by = field;
            currentSort.order = 'desc';
        }
        await updateVectorData();
    };

    await updateVectorData();
}

export async function updateVectorData() {
    const limit = showAll ? STORE_LIMIT : RECALL_LIMIT;
    const storeSort = currentSort.by === 'similarity' ? 'date' : currentSort.by;
    const sortBy = showAll ? storeSort : 'similarity';
    const order = showAll ? currentSort.order : 'desc';
    const scope = showAll ? '&scope=store' : '';
    try {
        const response = await fetch(
            `/api/brain/vectors?limit=${limit}&sort_by=${sortBy}&order=${order}${scope}`
        );
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        renderVectorPanel(await response.json());
    } catch (error) {
        console.error('Failed to fetch vector data', error);
        renderEmptyState();
    }
}

function toNumber(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
}

function formatDate(value, withTime = true) {
    if (!value) return '--';
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return '--';
    return new Intl.DateTimeFormat(navigator.language, withTime
        ? { dateStyle: 'short', timeStyle: 'short' }
        : { dateStyle: 'short' }).format(parsed);
}

function renderVectorPanel(data) {
    const container = document.getElementById('vector-content');
    if (!container) return;

    renderMeta(data);

    const activeElement = document.activeElement;
    const focusedToggle = activeElement && activeElement.classList.contains('vector-view-toggle');
    const focusedSortField = activeElement && activeElement.classList.contains('sortable-header')
        ? activeElement.getAttribute('data-sort')
        : null;

    container.innerHTML = DOMPurify.sanitize(`
        ${renderFreshness(data)}
        ${showAll ? '' : renderContext(data.current_context)}
        <div class="vector-table">${renderExperienceTable(data.experiences || [])}</div>
        <details class="memory-stats">
            <summary>Win rate by confidence, ADX, sentiment and volatility</summary>
            <div class="vector-stats">${renderStatsCards(data)}</div>
        </details>
    `);

    container.querySelectorAll('.sortable-header').forEach((header) => {
        const field = header.getAttribute('data-sort');
        if (!field) return;
        const handleSort = () => {
            if (window.vectorSort) window.vectorSort(field);
        };
        header.addEventListener('click', handleSort);
        header.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                handleSort();
            }
        });
    });

    const viewToggle = container.querySelector('.vector-view-toggle');
    if (viewToggle) {
        viewToggle.addEventListener('click', () => {
            showAll = !showAll;
            updateVectorData();
        });
    }

    container.querySelectorAll('.row-toggle').forEach((button) => {
        button.addEventListener('click', () => {
            const detail = document.getElementById(button.getAttribute('aria-controls'));
            if (!detail) return;
            const opening = detail.classList.contains('is-collapsed');
            detail.classList.toggle('is-collapsed', !opening);
            button.setAttribute('aria-expanded', String(opening));
        });
    });

    if (focusedToggle) {
        const restored = container.querySelector('.vector-view-toggle');
        if (restored) restored.focus();
    }

    if (focusedSortField) {
        const restored = container.querySelector(`.sortable-header[data-sort="${focusedSortField}"]`);
        if (restored) restored.focus();
    }
}

function renderMeta(data) {
    const meta = document.getElementById('memory-meta');
    if (!meta) return;
    const total = Number(data.experience_count) || 0;
    const shown = Array.isArray(data.experiences) ? data.experiences.length : 0;
    const mode = showAll ? 'shown' : 'recalled';
    const clock = new Intl.DateTimeFormat(navigator.language, { timeStyle: 'medium' }).format(new Date());
    meta.textContent = `${total} stored · ${shown} ${mode} · updated ${clock}`;
}

function renderFreshness(data) {
    const gate = data.evidence_gate;
    const gateHtml = gate
        ? `<span class="evidence-gate${gate.limited ? ' is-limited' : ''}" title="Recall counts as evidence only after ${escapeHtml(String(gate.min_trades))} closed trades; below that, matches are anecdotes.">
             ${gate.limited ? '⚠' : '✓'} Evidence ${escapeHtml(String(gate.trade_count))}/${escapeHtml(String(gate.min_trades))} trades
           </span>`
        : '';
    const total = Number(data.experience_count) || 0;
    let label = showAll ? 'Show only recalled' : `Show all ${total} stored`;
    if (!showAll && total > STORE_LIMIT) label = `Show newest ${STORE_LIMIT}`;
    const toggleHtml = total
        ? `<button type="button" class="btn btn-ghost btn-sm vector-view-toggle" aria-pressed="${showAll}">${escapeHtml(label)}</button>`
        : '';
    if (!gateHtml && !toggleHtml) return '';
    return `
        <div class="vector-freshness">
            ${gateHtml}
            ${toggleHtml}
        </div>
    `;
}

function renderContext(rawContext) {
    if (!rawContext) return '';
    return `
        <div class="context-indicator">
            <span>Ranked by how many entry conditions match the market context in Last decision.</span>
        </div>
    `;
}

function renderProgressRow(label, rawRate) {
    const rate = parseFloat(rawRate);
    if (Number.isNaN(rate)) {
        return `
            <div class="stat-row">
                <span>${escapeHtml(label)}</span>
                <span class="val">--</span>
            </div>
        `;
    }
    let colorClass = 'danger';
    if (rate >= 60) colorClass = 'success';
    else if (rate >= 40) colorClass = 'warning';

    return `
        <div class="stat-row">
            <span>${escapeHtml(label)}</span>
            <span class="val ${colorClass}">${Math.round(rate)}%</span>
        </div>
        <div class="metric-bar-container">
            <div class="metric-bar-fill ${colorClass}" style="width: ${Math.min(100, Math.max(0, rate))}%;"></div>
        </div>
    `;
}

function renderStatsCards(data) {
    const confidenceStats = data.confidence_stats || {};
    const adxStats = data.adx_stats || {};
    const factorStats = Array.isArray(data.factor_stats) ? data.factor_stats : Object.values(data.factor_stats || {});
    const factorWinRate = (keywords) => {
        const factor = factorStats.find(
            (entry) => keywords.some((keyword) => entry.factor_name && entry.factor_name.toUpperCase().includes(keyword))
        );
        return factor ? factor.win_rate : '--';
    };

    return `
        <div class="stat-card">
            <div class="stat-label">Win rate by confidence</div>
            ${renderProgressRow('High', confidenceStats.HIGH?.win_rate || '--')}
            ${renderProgressRow('Medium', confidenceStats.MEDIUM?.win_rate || '--')}
            ${renderProgressRow('Low', confidenceStats.LOW?.win_rate || '--')}
        </div>
        <div class="stat-card">
            <div class="stat-label">Win rate by ADX</div>
            ${renderProgressRow('High above 25', adxStats.HIGH?.win_rate || '--')}
            ${renderProgressRow('Medium 20-25', adxStats.MEDIUM?.win_rate || '--')}
            ${renderProgressRow('Low below 20', adxStats.LOW?.win_rate || '--')}
        </div>
        <div class="stat-card">
            <div class="stat-label">Win rate by sentiment</div>
            ${renderProgressRow('Bullish', factorWinRate(['BULLISH', 'GREED']))}
            ${renderProgressRow('Bearish', factorWinRate(['BEARISH', 'FEAR']))}
            ${renderProgressRow('Neutral', factorWinRate(['NEUTRAL']))}
        </div>
        <div class="stat-card">
            <div class="stat-label">Win rate by volatility</div>
            ${renderProgressRow('High volatility', factorWinRate(['HIGH VOLATILITY']))}
            ${renderProgressRow('Low volatility', factorWinRate(['LOW VOLATILITY']))}
        </div>
    `;
}

function parseSections(document) {
    const sections = Object.create(null);
    const allowedKeys = new Set(['Indicators', 'Structure', 'Confluences', 'Reasoning', 'Result', 'Post-trade']);
    const pattern = /\b(Indicators|Structure|Confluences|Reasoning|Result|Post-trade):\s*/g;
    const hits = [];
    let match;
    while ((match = pattern.exec(document)) !== null) {
        hits.push({ key: match[1], start: match.index, contentStart: match.index + match[0].length });
    }
    hits.forEach((hit, index) => {
        if (!allowedKeys.has(hit.key)) return;
        const end = index + 1 < hits.length ? hits[index + 1].start : document.length;
        sections[hit.key] = document.slice(hit.contentStart, end).trim();
    });
    sections['_header'] = hits.length ? document.slice(0, hits[0].start).trim() : document.trim();
    return sections;
}

function renderContextCell(document) {
    if (!document) return '--';
    const sections = parseSections(document);
    const headerText = (sections['_header'] || '').toUpperCase();
    const pills = CONTEXT_PILLS
        .filter((keyword) => headerText.includes(keyword.label))
        .map((keyword) => `<span class="context-pill ${keyword.cls}">${escapeHtml(keyword.label.replace('_', ' '))}</span>`)
        .join('');

    const metrics = [];
    const indicators = sections.Indicators || '';
    const adx = indicators.match(/ADX=(\d+\.?\d*)/);
    const rsi = indicators.match(/RSI=(\d+\.?\d*)/);
    const atr = indicators.match(/ATR=\$([\d]+)/);
    const structure = sections['Structure'] || '';
    const rr = structure.match(/RR=([\d.]+)/);
    if (adx) metrics.push(`ADX ${adx[1]}`);
    if (rsi) metrics.push(`RSI ${rsi[1]}`);
    if (atr) metrics.push(`ATR $${atr[1]}`);
    if (rr) metrics.push(`R:R ${rr[1]}`);

    const metricLine = metrics.length
        ? `<div class="context-metrics">${escapeHtml(metrics.join(' · '))}</div>`
        : '';
    return `<div class="context-pills">${pills}</div>${metricLine}`;
}

function sortHeader(field, label) {
    const activeSort = showAll ? currentSort : { by: 'recall', order: 'desc' };
    const active = activeSort.by === field;
    const ariaSort = active ? (activeSort.order === 'asc' ? 'ascending' : 'descending') : 'none';
    const mark = active
        ? `<span class="sort-mark" aria-hidden="true">${activeSort.order === 'asc' ? '▲' : '▼'}</span>`
        : '';
    return `<th scope="col" class="sortable-header${active ? ' is-sorted' : ''}" data-sort="${field}" tabindex="0" role="columnheader" aria-sort="${ariaSort}">${label}${mark}</th>`;
}

function renderDetailRow(experience, detailId, columnCount) {
    const meta = experience.metadata || {};
    const facts = [];
    if (meta.close_reason) facts.push(['Close reason', String(meta.close_reason)]);
    if (experience.match_factors) facts.push(['Match factors', String(experience.match_factors)]);
    if (meta.confidence) facts.push(['Confidence', String(meta.confidence)]);
    const factItems = facts.map(([label, value]) => `
        <div class="exp-fact">
            <span class="exp-fact-label">${escapeHtml(label)}</span>
            <span class="exp-fact-value">${escapeHtml(value)}</span>
        </div>
    `).join('');
    const document = experience.document
        ? `<details class="exp-document"><summary>Stored context</summary><p>${escapeHtml(experience.document)}</p></details>`
        : '';
    return `
        <tr class="exp-detail is-collapsed" id="${detailId}">
            <td colspan="${columnCount}">
                <div class="exp-facts">${factItems}</div>
                ${document}
            </td>
        </tr>
    `;
}

function renderExperienceTable(experiences) {
    if (!experiences || experiences.length === 0) {
        return `
            <div class="empty-state">
                <p class="empty-state-text">No experiences stored yet</p>
                <p class="empty-state-hint">A completed trade is recorded with its full context and outcome.</p>
            </div>
        `;
    }

    const showSimilarity = !showAll;
    const columnCount = showSimilarity ? 8 : 7;

    const rows = experiences.map((experience, index) => {
        const meta = experience.metadata || {};
        const detailId = `exp-detail-${index}`;
        const outcome = String(meta.outcome || '--').toUpperCase();
        const outcomeClass = outcome === 'WIN' ? 'win' : outcome === 'LOSS' ? 'loss' : '';
        const pnlValue = toNumber(meta.pnl_pct);
        const pnl = pnlValue === null ? '--' : `${pnlValue >= 0 ? '+' : ''}${pnlValue.toFixed(2)}%`;
        const pnlClass = pnlValue === null ? '' : (pnlValue >= 0 ? 'positive' : 'negative');
        const similarityValue = toNumber(experience.similarity);
        const similarity = similarityValue !== null && similarityValue > 0
            ? `${similarityValue.toFixed(1)}%`
            : '--';
        const direction = meta.direction
            ? `<span class="market-direction">${escapeHtml(meta.direction)}</span>`
            : '';
        return `
            <tr class="exp-row">
                <td class="exp-toggle-cell">
                    <button type="button" class="row-toggle" aria-expanded="false"
                        aria-controls="${detailId}" aria-label="Show details for ${escapeHtml(meta.symbol || 'this experience')}"></button>
                </td>
                <td class="exp-when" data-label="When">${escapeHtml(formatDate(meta.timestamp))}</td>
                <td class="exp-market" data-label="Market">${escapeHtml(meta.symbol || '--')}${direction}</td>
                <td class="context" data-label="Context">${renderContextCell(experience.document || '')}</td>
                <td class="${outcomeClass}" data-label="Outcome">${escapeHtml(outcome)}</td>
                <td class="${pnlClass}" data-label="P&amp;L">${escapeHtml(pnl)}</td>
                <td data-label="Confidence">${escapeHtml(meta.confidence || '--')}</td>
                ${showSimilarity ? `<td data-label="Match">${escapeHtml(similarity)}</td>` : ''}
            </tr>
            ${renderDetailRow(experience, detailId, columnCount)}
        `;
    }).join('');

    return `
        <table>
            <thead>
                <tr>
                    <th scope="col" class="exp-toggle-cell"><span class="sr-only">Details</span></th>
                    ${sortHeader('date', 'When')}
                    <th scope="col">Market</th>
                    <th scope="col">Context</th>
                    ${sortHeader('outcome', 'Outcome')}
                    ${sortHeader('pnl', 'P&amp;L')}
                    ${sortHeader('confidence', 'Confidence')}
                    ${showSimilarity ? '<th scope="col" aria-sort="descending" title="Share of entry conditions that match the current market">Match<span class="sort-mark" aria-hidden="true">▼</span></th>' : ''}
                </tr>
            </thead>
            <tbody>${rows}</tbody>
        </table>
    `;
}

function renderEmptyState() {
    const container = document.getElementById('vector-content');
    if (!container) return;
    const meta = document.getElementById('memory-meta');
    if (meta) meta.textContent = 'Memory unavailable';
    container.innerHTML = `
        <div class="empty-state">
            <p class="empty-state-text">Memory store not available</p>
            <p class="empty-state-hint">The vector store is reachable once the bot has recorded its first trade.</p>
        </div>
    `;
}

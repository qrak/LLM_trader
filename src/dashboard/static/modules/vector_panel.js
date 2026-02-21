// Sort state
let currentSort = {
    by: 'date',
    order: 'desc'
};

export async function initVectorPanel() {
    // Expose sort handler globally
    window.vectorSort = async (field) => {
        if (currentSort.by === field) {
            currentSort.order = currentSort.order === 'desc' ? 'asc' : 'desc';
        } else {
            currentSort.by = field;
            currentSort.order = 'desc'; // Default to desc for new field
        }
        await updateVectorData();
    };

    // Initial load
    await updateVectorData();
}

export async function updateVectorData() {
    try {
        const [vectorResponse, rulesResponse] = await Promise.all([
            fetch(`/api/brain/vectors?limit=50&sort_by=${currentSort.by}&order=${currentSort.order}`),
            fetch('/api/brain/rules')
        ]);
        const data = await vectorResponse.json();
        const rulesData = rulesResponse.ok ? await rulesResponse.json() : [];
        
        renderVectorPanel(data, rulesData);
    } catch (e) {
        console.error("Failed to fetch vector data", e);
        renderEmptyState();
    }
}

function renderVectorPanel(data, rulesData) {
    const container = document.getElementById('vector-content');
    if (!container) return;

    // Save focus state before re-render
    const activeElement = document.activeElement;
    let focusedSortField = null;
    if (activeElement && activeElement.classList.contains('sortable-header')) {
        focusedSortField = activeElement.getAttribute('data-sort');
    }

    // Build context indicator
    let readableContext = data.current_context || '';
    if (readableContext.includes('+')) {
        const parts = readableContext.split('+').map(p => p.trim());
        readableContext = `Searching for trades from similar markets: <span class="highlight">${parts.join(', ')}</span>`;
    }

    const contextHtml = data.current_context
        ? `<div class="context-indicator">
             <span class="context-value">${readableContext}</span>
           </div>`
        : '';

    // Build stats cards
    const statsHtml = renderStatsCards(data);
    
    // Build rules section
    const rulesHtml = renderSemanticRules(rulesData);

    // Build experience table
    const tableHtml = renderExperienceTable(data.experiences || []);

    container.innerHTML = `
        ${contextHtml}
        <div class="vector-stats">${statsHtml}</div>
        <div style="margin-bottom: 24px;">
            <h3 style="margin-top: 0; margin-bottom: 12px; color: var(--text-muted); font-size: 13px; text-transform: uppercase; letter-spacing: 1px;">Semantic Rules</h3>
            ${rulesHtml}
        </div>
        <div>
            <h3 style="margin-top: 0; margin-bottom: 12px; color: var(--text-muted); font-size: 13px; text-transform: uppercase; letter-spacing: 1px;">Experience History</h3>
            <div class="vector-table">${tableHtml}</div>
        </div>
    `;

    // Attach sort handlers
    container.querySelectorAll('.sortable-header').forEach(header => {
        const field = header.getAttribute('data-sort');
        if (field) {
            const handleSort = () => {
                if (window.vectorSort) window.vectorSort(field);
            };

            header.addEventListener('click', handleSort);

            // Add keyboard support (Enter/Space)
            header.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault(); // Prevent scroll on Space
                    handleSort();
                }
            });
        }
    });

    // Restore focus if it was on a sort header
    if (focusedSortField) {
        const newHeader = container.querySelector(`.sortable-header[data-sort="${focusedSortField}"]`);
        if (newHeader) newHeader.focus();
    }
}

function renderProgressRow(label, rawRate) {
    const rate = parseFloat(rawRate);
    if (isNaN(rate)) {
        return `
            <div class="stat-row">
                <span>${label}</span>
                <span class="val">--</span>
            </div>
        `;
    }
    
    let colorClass = 'danger';
    if (rate >= 60) colorClass = 'success';
    else if (rate >= 40) colorClass = 'warning';
    
    return `
        <div class="stat-row" style="margin-top: 10px;">
            <span>${label}</span>
            <span class="val" style="color: var(--accent-${colorClass});">${Math.round(rate)}%</span>
        </div>
        <div class="metric-bar-container">
            <div class="metric-bar-fill ${colorClass}" style="width: ${Math.min(100, Math.max(0, rate))}%;"></div>
        </div>
    `;
}

function renderStatsCards(data) {
    const confStats = data.confidence_stats || {};
    const adxStats = data.adx_stats || {};
    const factorStats = Array.isArray(data.factor_stats) ? data.factor_stats : Object.values(data.factor_stats || {});
    
    // Fallback getter for factor stats if keys are strictly strings
    const getFactorWR = (keywords) => {
        const factor = factorStats.find(f => keywords.some(k => f.factor_name && f.factor_name.toUpperCase().includes(k)));
        return factor ? factor.win_rate : '--';
    };

    return `
        <div class="stat-card">
            <div class="stat-label">Win Rate: Confidence</div>
            <div>
                ${renderProgressRow('HIGH', confStats.HIGH?.win_rate || '--')}
                ${renderProgressRow('MEDIUM', confStats.MEDIUM?.win_rate || '--')}
                ${renderProgressRow('LOW', confStats.LOW?.win_rate || '--')}
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Win Rate: ADX Range</div>
            <div>
                ${renderProgressRow('HIGH (>25)', adxStats.HIGH?.win_rate || '--')}
                ${renderProgressRow('MED (20-25)', adxStats.MEDIUM?.win_rate || '--')}
                ${renderProgressRow('LOW (<20)', adxStats.LOW?.win_rate || '--')}
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Win Rate: Sentiment</div>
            <div>
                ${renderProgressRow('BULLISH', getFactorWR(['BULLISH', 'GREED']))}
                ${renderProgressRow('BEARISH', getFactorWR(['BEARISH', 'FEAR']))}
                ${renderProgressRow('NEUTRAL', getFactorWR(['NEUTRAL']))}
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Win Rate: Volatility</div>
            <div>
                ${renderProgressRow('HIGH VOL', getFactorWR(['HIGH VOLATILITY']))}
                ${renderProgressRow('LOW VOL', getFactorWR(['LOW VOLATILITY']))}
            </div>
        </div>
    `;
}

function renderSemanticRules(rules) {
    if (!rules || rules.length === 0) {
        return `
            <div style="font-size: 13px; color: var(--text-muted); padding: 12px 16px; background: rgba(255,255,255,0.02); border: 1px dashed var(--border-subtle); border-radius: 6px;">
                Status: Gathering trade data to formulate rules... need 10+ wins matching a pattern.
            </div>
        `;
    }

    const cards = rules.map(rule => {
        const winRate = rule.win_rate !== undefined ? `${Math.round(rule.win_rate)}%` : '--';
        const srcTrades = rule.source_trades || '--';
        return `
            <div class="rule-card">
                <div class="rule-text">${escapeHtml(rule.rule_text)}</div>
                <div class="rule-meta">
                    <span>From ${srcTrades} Trades</span>
                    <span class="win-rate">${winRate} Win Rate</span>
                </div>
            </div>
        `;
    }).join('');

    return `<div class="rules-grid">${cards}</div>`;
}

function renderExperienceTable(experiences) {
    if (!experiences || experiences.length === 0) {
        return `
            <div class="empty-state">
                <p>No vector memories stored yet.</p>
                <p style="font-size: 0.8em; color: #8b949e;">
                    Experiences are recorded after completed trades with full metadata.
                </p>
            </div>
        `;
    }
    
function formatContextPills(text) {
    if (!text) return '--';
    
    // Basic heuristics to find contextual keywords and turn them into pills
    let html = escapeHtml(text);
    const keywords = [
        { label: 'HIGH VOLATILITY', class: 'pill-high-vol' },
        { label: 'LOW VOLATILITY', class: 'pill-low-vol' },
        { label: 'BULLISH', class: 'pill-bullish' },
        { label: 'BEARISH', class: 'pill-bearish' },
        { label: 'STRONG TREND', class: 'pill-strong-trend' },
        { label: 'WEAK TREND', class: 'pill-low-vol' },
        { label: 'OVERSOLD', class: 'pill-bullish' },
        { label: 'OVERBOUGHT', class: 'pill-bearish' }
    ];

    let replaced = false;
    let pillHtml = '';
    
    keywords.forEach(kw => {
        const regex = new RegExp(kw.label, 'gi');
        if (regex.test(text)) {
            pillHtml += `<span class="context-pill ${kw.class}">${kw.label}</span> `;
            replaced = true;
        }
    });

    if (replaced) {
        return pillHtml + `<div style="margin-top: 4px; opacity: 0.7; font-size: 0.9em;">${html.substring(0, 60)}${html.length > 60 ? '...' : ''}</div>`;
    }
    
    return html.substring(0, 80) + (html.length > 80 ? '...' : '');
}

    // UPDATE entries are now filtered at backend query level
    const rows = experiences.map(exp => {
        const meta = exp.metadata || {};
        const outcome = meta.outcome || '--';
        const outcomeClass = outcome === 'WIN' ? 'win' : outcome === 'LOSS' ? 'loss' : '';
        const pnl = meta.pnl_pct !== undefined ? `${meta.pnl_pct >= 0 ? '+' : ''}${meta.pnl_pct.toFixed(2)}%` : '--';
        const pnlClass = meta.pnl_pct >= 0 ? 'positive' : 'negative';
        const confidence = meta.confidence || '--';
        const direction = meta.direction || '--';
        const similarity = (exp.similarity !== undefined && exp.similarity > 0) ? `${exp.similarity.toFixed(1)}%` : '--';
        const timestamp = meta.timestamp ? new Intl.DateTimeFormat(navigator.language, { dateStyle: 'short', timeStyle: 'short' }).format(new Date(meta.timestamp)) : '--';
        
        const contextDisplay = formatContextPills(exp.document || '');
        
        return `
            <tr title="${escapeHtml(exp.document || '')}">
                <td style="font-family: 'JetBrains Mono', monospace; font-size: 0.9em;">${escapeHtml((exp.id || '').substring(0,8))}...</td>
                <td class="context">${contextDisplay}</td>
                <td class="${outcomeClass}">${escapeHtml(outcome)}</td>
                <td class="${pnlClass}">${pnl}</td>
                <td>${escapeHtml(confidence)}</td>
                <td>${escapeHtml(direction)}</td>
                <td>${similarity}</td>
                <td>${timestamp}</td>
            </tr>
        `;
    }).join('');
    
    const getSortIndicator = (field) => {
        if (currentSort.by !== field) return '<span style="opacity: 0.3">↕</span>';
        return currentSort.order === 'asc' ? '↑' : '↓';
    };

    const getAriaSort = (field) => {
        if (currentSort.by !== field) return 'none';
        return currentSort.order === 'asc' ? 'ascending' : 'descending';
    };
    
    return `
        <style>
            .sortable-header { cursor: pointer; user-select: none; }
            .sortable-header:hover { background-color: rgba(255, 255, 255, 0.05); }
            .sortable-header:focus-visible { outline: 2px solid var(--accent-primary); outline-offset: -2px; }
        </style>
        <table>
            <thead>
                <tr>
                    <th scope="col">ID</th>
                    <th scope="col">Context</th>
                    <th scope="col" class="sortable-header" data-sort="outcome" tabindex="0" role="columnheader" aria-sort="${getAriaSort('outcome')}">
                        Outcome <span aria-hidden="true">${getSortIndicator('outcome')}</span>
                    </th>
                    <th scope="col" class="sortable-header" data-sort="pnl" tabindex="0" role="columnheader" aria-sort="${getAriaSort('pnl')}">
                        P&L <span aria-hidden="true">${getSortIndicator('pnl')}</span>
                    </th>
                    <th scope="col" class="sortable-header" data-sort="confidence" tabindex="0" role="columnheader" aria-sort="${getAriaSort('confidence')}">
                        Confidence <span aria-hidden="true">${getSortIndicator('confidence')}</span>
                    </th>
                    <th scope="col" class="sortable-header" data-sort="direction" tabindex="0" role="columnheader" aria-sort="${getAriaSort('direction')}">
                        Direction <span aria-hidden="true">${getSortIndicator('direction')}</span>
                    </th>
                    <th scope="col" class="sortable-header" data-sort="similarity" tabindex="0" role="columnheader" aria-sort="${getAriaSort('similarity')}">
                        Similarity <span aria-hidden="true">${getSortIndicator('similarity')}</span>
                    </th>
                    <th scope="col" class="sortable-header" data-sort="date" tabindex="0" role="columnheader" aria-sort="${getAriaSort('date')}">
                        Date <span aria-hidden="true">${getSortIndicator('date')}</span>
                    </th>
                </tr>
            </thead>
            <tbody>
                ${rows}
            </tbody>
        </table>
    `;
}

function renderEmptyState() {
    const container = document.getElementById('vector-content');
    if (!container) return;
    
    container.innerHTML = `
        <div class="empty-state">
            <p>Vector memory service not available.</p>
            <p style="font-size: 0.8em; color: #8b949e;">
                Ensure ChromaDB is initialized and the bot has recorded trades.
            </p>
        </div>
    `;
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

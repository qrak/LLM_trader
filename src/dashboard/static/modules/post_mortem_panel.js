/**
 * Trade Journal panel module — Displays post-mortem analyses with FTS5 search.
 */

export async function initPostMortemPanel() {
    await updatePostMortemData();
    const searchInput = document.getElementById('pm-search-input');
    const searchBtn = document.getElementById('pm-search-btn');
    if (searchInput && searchBtn) {
        const doSearch = () => {
            updatePostMortemData(searchInput.value.trim());
        };
        searchBtn.addEventListener('click', doSearch);
        searchInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') doSearch();
        });
    }
}

export async function updatePostMortemData(query) {
    const container = document.getElementById('post-mortem-content');
    if (!container) return;
    try {
        let url = '/api/brain/post-mortems';
        if (query) {
            url += `?q=${encodeURIComponent(query)}`;
        }
        const response = await fetch(url);
        const data = await response.json();
        if (!data.post_mortems || data.post_mortems.length === 0) {
            container.innerHTML = DOMPurify.sanitize(`
                <div class="empty-state">
                    <div class="empty-state-icon" aria-hidden="true"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg></div>
                    <p class="empty-state-text">${query ? 'No post-mortems match your search' : 'No post-mortems yet'}</p>
                    <p style="font-size: 0.85em; margin-top: 8px; color: var(--text-muted);">
                        ${query ? 'Try a different search term.' : 'Post-mortems appear after positions are closed.'}
                    </p>
                </div>
            `);
            return;
        }
        container.innerHTML = DOMPurify.sanitize(renderPostMortems(data.post_mortems, query));
    } catch (e) {
        container.innerHTML = DOMPurify.sanitize(`
            <div class="empty-state">
                <div class="empty-state-icon" aria-hidden="true"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg></div>
                <p class="empty-state-text">Error loading post-mortems</p>
                <p style="font-size: 0.85em; margin-top: 8px; color: var(--accent-danger);">${escapeHtml(e.message)}</p>
            </div>
        `);
    }
}

function renderPostMortems(postMortems, query) {
    const verdictColors = {
        'overestimated_breakout': 'var(--accent-warning)',
        'good_exit': 'var(--accent-success)',
        'plan_followed': 'var(--accent-success)',
        'premature_entry': 'var(--accent-warning)',
        'held_too_long': 'var(--accent-danger)',
        'support_breakdown': 'var(--accent-danger)',
        'failed_breakout': 'var(--accent-warning)',
        'missed_stop': 'var(--accent-danger)',
    };

    return `
        <div class="pm-list">
            ${postMortems.map(pm => {
                const color = verdictColors[pm.verdict] || 'var(--text-secondary)';
                const pnlClass = pm.pnl_pct >= 0 ? 'pm-pnl-positive' : 'pm-pnl-negative';
                const pnlStr = pm.pnl_pct != null
                    ? `<span class="${pnlClass}">${pm.pnl_pct >= 0 ? '+' : ''}${pm.pnl_pct.toFixed(1)}%</span>`
                    : '';
                const dateStr = pm.created_at ? pm.created_at.slice(0, 10) : '';
                const highlighted = query ? highlightText(escapeHtml(pm.lesson_learned), query) : escapeHtml(pm.lesson_learned);
                const analysisHighlighted = query ? highlightText(escapeHtml(pm.llm_analysis || ''), query) : escapeHtml(pm.llm_analysis || '');

                return `
                    <div class="pm-card">
                        <div class="pm-header">
                            <span class="pm-verdict" style="color: ${color}">${escapeHtml(pm.verdict)}</span>
                            <span class="pm-symbol">${escapeHtml(pm.symbol)} ${escapeHtml(pm.direction || '')}</span>
                            ${pnlStr}
                            <span class="pm-date">${dateStr}</span>
                        </div>
                        <div class="pm-reason">
                            <strong>Close:</strong> ${escapeHtml(pm.close_reason || '—')}
                        </div>
                        <div class="pm-lesson">${highlighted}</div>
                        ${pm.llm_analysis ? `
                            <details class="pm-details">
                                <summary>Full Analysis</summary>
                                <div class="pm-analysis-body">${analysisHighlighted}</div>
                            </details>
                        ` : ''}
                    </div>
                `;
            }).join('')}
        </div>
    `;
}

function highlightText(text, query) {
    if (!query || typeof text !== 'string') return text || '';
    const qLower = String(query).toLowerCase();
    const tLower = text.toLowerCase();
    let idx = tLower.indexOf(qLower);
    if (idx === -1) return text;
    let result = '';
    let lastIdx = 0;
    while (idx !== -1) {
        result += text.substring(lastIdx, idx) + '<mark>' + text.substring(idx, idx + qLower.length) + '</mark>';
        lastIdx = idx + qLower.length;
        idx = tLower.indexOf(qLower, lastIdx);
    }
    result += text.substring(lastIdx);
    return result;
}

function escapeHtml(text) {
    if (!text) return '';
    return String(text)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

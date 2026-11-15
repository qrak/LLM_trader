
import { escapeHtml } from './html_utils.js?v=1.0';

const LESSONS_ENDPOINT = '/api/brain/post-mortems';
const RECENT_LIMIT = 5;
const ARCHIVE_LIMIT = 100;

let activeQuery = '';
let showAll = false;

export async function initLessonsPanel() {
    window.updateLessonsPanel = updateLessonsPanel;

    const searchInput = document.getElementById('pm-search-input');
    const searchBtn = document.getElementById('pm-search-btn');
    if (searchInput && searchBtn) {
        const doSearch = () => {
            activeQuery = searchInput.value.trim();
            updateLessonsPanel();
        };
        searchBtn.addEventListener('click', doSearch);
        searchInput.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') doSearch();
        });
    }

    const toggle = document.getElementById('lessons-view-toggle');
    if (toggle) {
        toggle.addEventListener('click', () => {
            showAll = !showAll;
            updateLessonsPanel();
        });
    }

    await updateLessonsPanel();
}

export async function updateLessonsPanel() {
    const container = document.getElementById('post-mortem-content');
    if (!container) return;
    const limit = showAll || activeQuery ? ARCHIVE_LIMIT : RECENT_LIMIT;
    syncToggle();
    try {
        let url = `${LESSONS_ENDPOINT}?limit=${limit}`;
        if (activeQuery) url += `&q=${encodeURIComponent(activeQuery)}`;
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        if (data.error) throw new Error(String(data.error));
        const items = Array.isArray(data.post_mortems) ? data.post_mortems : [];
        renderMeta(items.length);
        container.innerHTML = DOMPurify.sanitize(
            items.length ? renderPostMortems(items) : emptyStateHtml()
        );
    } catch (error) {
        console.error('Failed to update the lessons panel', error);
        renderMeta(null);
        container.innerHTML = DOMPurify.sanitize(errorStateHtml(error.message || String(error)));
    }
}

function syncToggle() {
    const toggle = document.getElementById('lessons-view-toggle');
    if (!toggle) return;
    toggle.textContent = showAll ? 'Show newest five' : 'Show all lessons';
    toggle.setAttribute('aria-pressed', String(showAll));
    toggle.disabled = Boolean(activeQuery);
}

function renderMeta(count) {
    const meta = document.getElementById('lessons-meta');
    if (!meta) return;
    if (count === null) {
        meta.textContent = 'Lessons unavailable';
    } else if (activeQuery) {
        meta.textContent = `${count} matching “${activeQuery}”`;
    } else {
        meta.textContent = showAll ? `${count} shown` : `${count} newest`;
    }
}

function emptyStateHtml() {
    return `
        <div class="empty-state">
            <div class="empty-state-icon" aria-hidden="true"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg></div>
            <p class="empty-state-text">${activeQuery ? 'No lessons match your search' : 'No lessons yet'}</p>
            <p class="empty-state-hint">${activeQuery ? 'Try a different search term.' : 'A lesson is written after a position closes.'}</p>
        </div>
    `;
}

function errorStateHtml(message) {
    return `
        <div class="empty-state">
            <div class="empty-state-icon" aria-hidden="true"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg></div>
            <p class="empty-state-text">Error loading lessons</p>
            <p class="empty-state-hint">${escapeHtml(message)}</p>
        </div>
    `;
}

function verdictTone(verdict) {
    if (verdict === 'good_exit' || verdict === 'plan_followed') return 'is-success';
    if (verdict === 'overestimated_breakout' || verdict === 'premature_entry' || verdict === 'failed_breakout') {
        return 'is-warning';
    }
    if (verdict === 'held_too_long' || verdict === 'support_breakdown' || verdict === 'missed_stop') {
        return 'is-danger';
    }
    return 'is-muted';
}

function renderPostMortems(postMortems) {
    return `
        <div class="pm-list">
            ${postMortems.map(pm => {
                const tone = verdictTone(pm.verdict);
                const pnlClass = pm.pnl_pct >= 0 ? 'pm-pnl-positive' : 'pm-pnl-negative';
                const pnlStr = pm.pnl_pct != null
                    ? `<span class="${pnlClass}">${pm.pnl_pct >= 0 ? '+' : ''}${pm.pnl_pct.toFixed(1)}%</span>`
                    : '';
                const dateStr = pm.created_at ? pm.created_at.slice(0, 10) : '';
                const highlighted = activeQuery ? highlightText(escapeHtml(pm.lesson_learned), activeQuery) : escapeHtml(pm.lesson_learned);
                const analysisHighlighted = activeQuery ? highlightText(escapeHtml(pm.llm_analysis || ''), activeQuery) : escapeHtml(pm.llm_analysis || '');

                return `
                    <div class="pm-card">
                        <div class="pm-header">
                            <span class="pm-verdict ${tone}">${escapeHtml(pm.verdict)}</span>
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

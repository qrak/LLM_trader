
import { escapeHtml } from './html_utils.js?v=1.0';
let lastPosition = null;

export function initPositionPanel() {
    const container = document.getElementById('position-content');
    if (!container) return;
    updatePositionData();
}

function formatDuration(seconds) {
    if (seconds < 0) seconds = 0;
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    if (days > 0) return `${days}d ${hours}h`;
    if (hours > 0) return `${hours}h ${minutes}m`;
    if (minutes > 0) return `${minutes}m`;
    return `${Math.floor(seconds)}s`;
}

function calculatePnL(entryPrice, currentPrice, direction) {
    if (!entryPrice || !currentPrice) return 0;
    const diff = direction === 'LONG'
        ? (currentPrice - entryPrice) / entryPrice
        : (entryPrice - currentPrice) / entryPrice;
    return diff * 100;
}

function toNumber(value, fallback = 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
}

const usdFormat = new Intl.NumberFormat(navigator.language, { style: 'currency', currency: 'USD' });

function formatUsd(value) {
    return usdFormat.format(value);
}

function gaugeAlign(left) {
    if (left <= 12) return 'start';
    if (left >= 88) return 'end';
    return 'center';
}

function gaugeScale(position) {
    const prices = [position.stop_loss, position.entry_price, position.take_profit].map(Number);
    if (!prices.every(Number.isFinite) || new Set(prices).size < 2) return null;
    const sign = position.direction === 'LONG' ? 1 : -1;
    const levels = [...new Set(prices)].sort((a, b) => (a - b) * sign);
    const step = 100 / (levels.length - 1);
    const low = Math.min(...prices);
    const high = Math.max(...prices);
    const leftOf = (rawPrice) => {
        const price = Math.min(high, Math.max(low, Number(rawPrice)));
        const i = levels.findIndex((level, k) => k < levels.length - 1 && (price - level) * (price - levels[k + 1]) <= 0);
        return (i + (price - levels[i]) / (levels[i + 1] - levels[i])) * step;
    };
    return { slotOf: (rawPrice) => levels.indexOf(Number(rawPrice)) * step, leftOf };
}

function formatRiskBadges(riskManagement, fallbackExitManagement) {
    const entryLabels = riskManagement?.at_entry_labels || {};
    const stopLossMode = entryLabels.stop_loss || formatExecutionPolicy(fallbackExitManagement, 'stop_loss');
    const takeProfitMode = entryLabels.take_profit || formatExecutionPolicy(fallbackExitManagement, 'take_profit');
    const changed = riskManagement?.policy_changed
        ? '<span class="risk-policy-changed" title="Current execution policy differs from the entry snapshot">policy changed</span>'
        : '';
    return `
        <div class="position-exit-management">
            <div><span class="label">SL Execution</span><span class="risk-badge sl">SL: ${escapeHtml(stopLossMode)}</span></div>
            <div><span class="label">TP Execution</span><span class="risk-badge tp">TP: ${escapeHtml(takeProfitMode)}</span></div>
            ${changed}
        </div>
    `;
}

function formatExecutionPolicy(policy, prefix) {
    if (!policy) return '--';
    const executionType = policy[`${prefix}_type`] || 'unknown';
    const checkInterval = policy[`${prefix}_check_interval`] || 'unknown';
    return `${executionType} / ${checkInterval}`;
}

export async function updatePositionData(currentPrice = null, fetchFresh = false) {
    const container = document.getElementById('position-content');
    if (!container) return;
    try {
        if (fetchFresh) {
            try {
                const priceResponse = await fetch('/api/brain/refresh-price');
                const priceData = await priceResponse.json();
                if (priceData.success && priceData.current_price) {
                    currentPrice = priceData.current_price;
                }
            } catch (e) {
                console.warn('Failed to refresh price:', e);
            }
        }
        const response = await fetch('/api/brain/position');
        const data = await response.json();
        const previousPosition = lastPosition;
        lastPosition = data;
        if (!data.has_position) {
            if (previousPosition && previousPosition.has_position) {
                document.dispatchEvent(new CustomEvent('trade-closed-detected'));
            }
            container.innerHTML = `
                <div class="no-position">
                    <span class="no-position-icon"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-6l-2 3h-4l-2-3H2"/><path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/></svg></span>
                    <span>No active position</span>
                </div>
            `;
            return;
        }
        const entryTime = new Date(data.entry_time);
        const timeInPosition = (Date.now() - entryTime.getTime()) / 1000;
        const priceToUse = currentPrice || data.current_price;
        const pnl = priceToUse ? calculatePnL(data.entry_price, priceToUse, data.direction) : null;
        const isProfit = pnl !== null && pnl >= 0;
        const directionClass = data.direction === 'LONG' ? 'long' : 'short';
        const exitManagement = data.exit_management_at_entry || data.exit_management || {};
        const lifecycle = data.lifecycle && data.lifecycle.available !== false ? data.lifecycle : null;
        const lifecyclePosition = lifecycle ? (lifecycle.position || {}) : {};
        const proof = lifecyclePosition.verification || null;
        const proofAgeSeconds = proof && proof.age_seconds !== null && proof.age_seconds !== undefined
            ? Number(proof.age_seconds)
            : NaN;
        const proofAge = Number.isFinite(proofAgeSeconds)
            ? ` · checked ${formatDuration(proofAgeSeconds)} ago`
            : (proof && proof.checked_at ? ` · checked ${escapeHtml(proof.checked_at)}` : '');
        const proofLine = proof
            ? `<div class="position-proof state-${escapeHtml(proof.severity || 'neutral')}" title="${escapeHtml(proof.detail || '')}">Execution proof: ${escapeHtml(proof.label || 'Unverified')}${proofAge}</div>`
            : '';
        const phaseNotice = ['entering', 'closing', 'divergence'].includes(lifecyclePosition.phase)
            ? `<div class="position-phase-notice state-${escapeHtml(lifecyclePosition.severity || 'warn')}">${escapeHtml(lifecyclePosition.label || '')}</div>`
            : '';
        const longIcon = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="icon-inline"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>`;
        const shortIcon = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="icon-inline"><polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/><polyline points="16 17 22 17 22 11"/></svg>`;
        const timeIcon = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="icon-inline"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>`;

        const scale = gaugeScale(data);
        const railValid = scale !== null;
        const gaugeLevels = [
            { role: 'sl', label: 'Bot SL', price: data.stop_loss },
            { role: 'entry', label: 'Entry', price: data.entry_price },
            { role: 'tp', label: 'TP', price: data.take_profit },
        ].map((mark) => ({ ...mark, left: railValid ? scale.slotOf(mark.price) : null }));
        const currentMark = priceToUse
            ? { label: 'Current', price: priceToUse, left: railValid ? scale.leftOf(priceToUse) : null }
            : null;
        const levelLabels = [...new Set(gaugeLevels.map((mark) => mark.left))]
            .map((left) => {
                const marks = gaugeLevels.filter((mark) => mark.left === left);
                return `<span class="gauge-label gauge-label--${gaugeAlign(left)}"><strong>${marks.map((mark) => mark.label).join(' = ')}</strong>${formatUsd(marks[0].price)}</span>`;
            })
            .join('');
        const currentLabel = railValid && currentMark
            ? `<span class="gauge-label gauge-current-label gauge-label--${gaugeAlign(currentMark.left)}" style="left: ${currentMark.left}%;"><strong>${currentMark.label}</strong>${formatUsd(currentMark.price)}</span>`
            : '';
        const gauge = railValid ? `
                <div class="position-gauge">
                    <div class="gauge-track">
                        ${gaugeLevels.map((mark) => `<span class="gauge-dot gauge-${mark.role}" style="left: ${mark.left}%;"></span>`).join('')}
                        ${currentMark ? `<span class="gauge-dot gauge-current" style="left: ${currentMark.left}%;"></span>` : ''}
                    </div>
                    <div class="gauge-labels${currentMark ? ' has-current' : ''}">
                        ${levelLabels}
                        ${currentLabel}
                    </div>
                </div>
                ` : `
                <div class="position-gauge">
                    <div class="gauge-levels">
                        ${gaugeLevels.map((mark) => `<span><strong>${mark.label}</strong>${formatUsd(mark.price)}</span>`).join('')}
                        ${currentMark ? `<span class="gauge-levels-current"><strong>${currentMark.label}</strong>${formatUsd(currentMark.price)}</span>` : ''}
                    </div>
                </div>
                `;

        const posHtml = `
            <div class="position-grid">
                <div class="position-badge ${directionClass}">
                    ${data.direction === 'LONG' ? longIcon : shortIcon} <span>${escapeHtml(data.direction)}</span>
                </div>
                <div class="position-symbol">${escapeHtml(data.symbol)}</div>
                <div class="position-stat">
                    <span class="label">Entry</span>
                    <span class="value">${formatUsd(data.entry_price)}</span>
                </div>
                <div class="position-stat">
                    <span class="label">Current</span>
                <span class="value" id="current-price">${priceToUse ? formatUsd(priceToUse) : '--'}</span>
                </div>
                <div class="position-pnl ${isProfit ? 'profit' : 'loss'}">
                    ${pnl !== null ? (pnl >= 0 ? '+' : '') + pnl.toFixed(2) + '%' : '--'}
                </div>
                ${formatRiskBadges(data.risk_management, exitManagement)}
                ${proofLine}
                ${phaseNotice}
                <div class="position-stat" title="Emergency venue trigger, not the bot's SL or a guaranteed fill price">
                    <span class="label">Venue backstop</span>
                    <span class="value">${Number.isFinite(data.venue_stop_price) ? formatUsd(data.venue_stop_price) : 'Unavailable'}</span>
                </div>
                <div class="position-meta">
                    <span title="Time in position" class="meta-item">${timeIcon} <span>${formatDuration(timeInPosition)}</span></span>
                    <span title="Reward-to-risk ratio at entry">Entry R:R ${toNumber(data.rr_ratio).toFixed(1)}</span>
                    <span title="Confidence">${escapeHtml(String(data.confidence))}</span>
                </div>
                
                <div class="position-indicators" style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.1);">
                    <div style="display: flex; justify-content: space-between; font-size: 0.9em; margin-bottom: 5px;">
                        <span title="ADX at Entry">ADX: ${data.adx_at_entry ? toNumber(data.adx_at_entry).toFixed(1) : '--'}</span>
                        <span title="RSI at Entry">RSI: ${data.rsi_at_entry ? toNumber(data.rsi_at_entry).toFixed(1) : '--'}</span>
                    </div>
                </div>

                ${data.confluence_factors && data.confluence_factors.length > 0 ? `
                <div class="confluence-factors" style="margin-top: 5px; font-size: 0.85em;">
                    <div style="opacity: 0.7; margin-bottom: 3px;">Confluence Factors:</div>
                    <ul style="list-style: none; padding: 0; margin: 0;">
                        ${data.confluence_factors.map(f => `
                            <li style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                                <span style="text-transform: capitalize;">${escapeHtml(f[0].replace(/_/g, ' '))}</span>
                                <span style="color: var(--accent-color);">${f[1]}%</span>
                            </li>
                        `).join('')}
                    </ul>
                </div>
                ` : ''}

                ${gauge}
            </div>
        `;
        container.innerHTML = window.DOMPurify ? DOMPurify.sanitize(posHtml) : posHtml;
    } catch (error) {
        container.innerHTML = '<div class="error">Error loading position data</div>';
    }
}

export function getLastPosition() {
    return lastPosition;
}

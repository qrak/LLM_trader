/**
 * Position panel module - Displays current open position with live P&L.
 */

let lastPosition = null;

/**
 * Initialize position panel.
 */
export function initPositionPanel() {
    const container = document.getElementById('position-content');
    if (!container) return;
    updatePositionData();
}

/**
 * Format time duration to human readable string (e.g., "2h 35m" or "3d 5h").
 */
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

/**
 * Calculate unrealized P&L percentage.
 */
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

function formatPercent(value, digits = 1) {
    return `${(toNumber(value) * 100).toFixed(digits)}%`;
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

/**
 * Update position panel with latest data.
 * @param {number|null} currentPrice - Optional price to use
 * @param {boolean} fetchFresh - If true, fetch fresh price from exchange
 */
export async function updatePositionData(currentPrice = null, fetchFresh = false) {
    const container = document.getElementById('position-content');
    if (!container) return;
    try {
        // Optionally fetch fresh price from exchange
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
        const longIcon = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="icon-inline"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>`;
        const shortIcon = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="icon-inline"><polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/><polyline points="16 17 22 17 22 11"/></svg>`;
        const timeIcon = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="icon-inline"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>`;

        container.innerHTML = `
            <div class="position-grid">
                <div class="position-badge ${directionClass}">
                    ${data.direction === 'LONG' ? longIcon : shortIcon} <span>${escapeHtml(data.direction)}</span>
                </div>
                <div class="position-symbol">${escapeHtml(data.symbol)}</div>
                <div class="position-stat">
                    <span class="label">Entry</span>
                    <span class="value">${new Intl.NumberFormat(navigator.language, { style: 'currency', currency: 'USD' }).format(data.entry_price)}</span>
                </div>
                <div class="position-stat">
                    <span class="label">Current</span>
                <span class="value" id="current-price">${priceToUse ? new Intl.NumberFormat(navigator.language, { style: 'currency', currency: 'USD' }).format(priceToUse) : '--'}</span>
                </div>
                <div class="position-pnl ${isProfit ? 'profit' : 'loss'}">
                    ${pnl !== null ? (pnl >= 0 ? '+' : '') + pnl.toFixed(2) + '%' : '--'}
                </div>
                <div class="position-sl-tp">
                    <div class="sl">SL: ${new Intl.NumberFormat(navigator.language, { style: 'currency', currency: 'USD' }).format(data.stop_loss)} <span class="pct">(-${formatPercent(data.sl_distance_pct)})</span></div>
                    <div class="tp">TP: ${new Intl.NumberFormat(navigator.language, { style: 'currency', currency: 'USD' }).format(data.take_profit)} <span class="pct">(+${formatPercent(data.tp_distance_pct)})</span></div>
                </div>
                ${formatRiskBadges(data.risk_management, exitManagement)}
                <div class="position-meta">
                    <span title="Time in position" class="meta-item">${timeIcon} <span>${formatDuration(timeInPosition)}</span></span>
                    <span title="Risk/Reward ratio">R:R ${toNumber(data.rr_ratio).toFixed(1)}</span>
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

                <div class="position-gauge">
                    <div class="gauge-track">
                        <div class="gauge-sl" style="left: 0;" title="Stop Loss: ${new Intl.NumberFormat(navigator.language, { style: 'currency', currency: 'USD' }).format(data.stop_loss)}"></div>
                        <div class="gauge-entry" style="left: ${calculateGaugePosition(data, data.entry_price)}%;" title="Entry: ${new Intl.NumberFormat(navigator.language, { style: 'currency', currency: 'USD' }).format(data.entry_price)}"></div>
                        <div class="gauge-tp" style="right: 0;" title="Take Profit: ${new Intl.NumberFormat(navigator.language, { style: 'currency', currency: 'USD' }).format(data.take_profit)}"></div>
                        ${priceToUse ? `<div class="gauge-current ${isProfit ? 'profit' : 'loss'}" style="left: ${calculateGaugePosition(data, priceToUse)}%;" title="Current: ${new Intl.NumberFormat(navigator.language, { style: 'currency', currency: 'USD' }).format(priceToUse)} (${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}%)"></div>` : ''}
                    </div>
                    <div class="gauge-labels">
                        <span title="Stop Loss: ${new Intl.NumberFormat(navigator.language, { style: 'currency', currency: 'USD' }).format(data.stop_loss)}">SL</span>
                        <span title="Entry: ${new Intl.NumberFormat(navigator.language, { style: 'currency', currency: 'USD' }).format(data.entry_price)}">Entry</span>
                        <span title="Take Profit: ${new Intl.NumberFormat(navigator.language, { style: 'currency', currency: 'USD' }).format(data.take_profit)}">TP</span>
                    </div>
                </div>
            </div>
        `;
    } catch (error) {
        container.innerHTML = '<div class="error">Error loading position data</div>';
    }
}

/**
 * Calculate gauge position (0-100) for current price between SL and TP.
 */
function calculateGaugePosition(position, currentPrice) {
    const sl = position.stop_loss;
    const tp = position.take_profit;
    const entry = position.entry_price;
    const direction = position.direction;

    // Handle invalid range
    if (tp === sl) return 50;

    // Fix entry at 50%
    if (currentPrice === entry) return 50;

    // The gauge logic depends on direction for SL/TP relationship
    // For LONG: SL < Entry < TP
    // For SHORT: TP < Entry < SL

    if (direction === 'LONG') {
        if (currentPrice < entry) {
            // Scale in [0, 50] range between SL and Entry
            const range = entry - sl;
            if (range <= 0) return 0;
            const pos = ((currentPrice - sl) / range) * 50;
            return Math.max(0, Math.min(50, pos));
        } else {
            // Scale in [50, 100] range between Entry and TP
            const range = tp - entry;
            if (range <= 0) return 100;
            const pos = 50 + ((currentPrice - entry) / range) * 50;
            return Math.max(50, Math.min(100, pos));
        }
    } else { // SHORT
        if (currentPrice > entry) {
            // Scale in [0, 50] range between SL and Entry (SL is higher in SHORT)
            const range = sl - entry;
            if (range <= 0) return 0;
            const pos = ((sl - currentPrice) / range) * 50;
            return Math.max(0, Math.min(50, pos));
        } else {
            // Scale in [50, 100] range between Entry and TP (TP is lower in SHORT)
            const range = entry - tp;
            if (range <= 0) return 100;
            const pos = 50 + ((entry - currentPrice) / range) * 50;
            return Math.max(50, Math.min(100, pos));
        }
    }
}

/**
 * Get last known position data.
 */
export function getLastPosition() {
    return lastPosition;
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

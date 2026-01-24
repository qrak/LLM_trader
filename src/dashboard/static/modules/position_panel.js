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
        lastPosition = data;
        if (!data.has_position) {
            container.innerHTML = `
                <div class="no-position">
                    <span class="no-position-icon">📭</span>
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
        container.innerHTML = `
            <div class="position-grid">
                <div class="position-badge ${directionClass}">
                    ${data.direction === 'LONG' ? '📈' : '📉'} ${data.direction}
                </div>
                <div class="position-symbol">${escapeHtml(data.symbol)}</div>
                <div class="position-stat">
                    <span class="label">Entry</span>
                    <span class="value">$${data.entry_price.toLocaleString()}</span>
                </div>
                <div class="position-stat">
                    <span class="label">Current</span>
                <span class="value" id="current-price">${priceToUse ? '$' + priceToUse.toLocaleString() : '--'}</span>
                </div>
                <div class="position-pnl ${isProfit ? 'profit' : 'loss'}">
                    ${pnl !== null ? (pnl >= 0 ? '+' : '') + pnl.toFixed(2) + '%' : '--'}
                </div>
                <div class="position-sl-tp">
                    <div class="sl">SL: $${data.stop_loss.toLocaleString()} <span class="pct">(-${data.sl_distance_pct.toFixed(1)}%)</span></div>
                    <div class="tp">TP: $${data.take_profit.toLocaleString()} <span class="pct">(+${data.tp_distance_pct.toFixed(1)}%)</span></div>
                </div>
                <div class="position-meta">
                    <span title="Time in position">⏱️ ${formatDuration(timeInPosition)}</span>
                    <span title="Risk/Reward ratio">R:R ${data.rr_ratio.toFixed(1)}</span>
                    <span title="Confidence">${escapeHtml(String(data.confidence))}</span>
                </div>
                
                <div class="position-indicators" style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.1);">
                    <div style="display: flex; justify-content: space-between; font-size: 0.9em; margin-bottom: 5px;">
                        <span title="ADX at Entry">ADX: ${data.adx_at_entry ? data.adx_at_entry.toFixed(1) : '--'}</span>
                        <span title="RSI at Entry">RSI: ${data.rsi_at_entry ? data.rsi_at_entry.toFixed(1) : '--'}</span>
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
                        <div class="gauge-sl" style="left: 0;" title="Stop Loss: $${data.stop_loss.toLocaleString()}"></div>
                        <div class="gauge-entry" style="left: ${calculateGaugePosition(data, data.entry_price)}%;" title="Entry: $${data.entry_price.toLocaleString()}"></div>
                        <div class="gauge-tp" style="right: 0;" title="Take Profit: $${data.take_profit.toLocaleString()}"></div>
                        ${priceToUse ? `<div class="gauge-current ${isProfit ? 'profit' : 'loss'}" style="left: ${calculateGaugePosition(data, priceToUse)}%;" title="Current: $${priceToUse.toLocaleString()} (${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}%)"></div>` : ''}
                    </div>
                    <div class="gauge-labels">
                        <span title="Stop Loss: $${data.stop_loss.toLocaleString()}">SL</span>
                        <span title="Entry: $${data.entry_price.toLocaleString()}">Entry</span>
                        <span title="Take Profit: $${data.take_profit.toLocaleString()}">TP</span>
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
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

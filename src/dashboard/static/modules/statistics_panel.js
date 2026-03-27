/**
 * Statistics panel module - Displays trading statistics with annotations.
 */

const STAT_ANNOTATIONS = {
    total_trades: "Number of completed trades (entries that have been closed). Higher volume indicates more active trading.",
    win_rate: "Percentage of trades that were profitable. Above 50% is generally good; above 60% is excellent.",
    total_pnl_pct: "Cumulative percentage change from initial capital. Shows overall performance since trading began.",
    total_pnl_quote: "Actual dollar amount gained or lost. Negative means net loss, positive means net profit.",
    sharpe_ratio: "Risk-adjusted return metric. Measures excess return per unit of risk. Above 1.0 is good, above 2.0 is excellent.",
    sortino_ratio: "Like Sharpe but only penalizes downside volatility. Higher is better. More forgiving of upside volatility.",
    profit_factor: "Gross profit divided by gross loss. Above 1.0 means profitable overall. Above 1.5 is considered good.",
    max_drawdown_pct: "Largest peak-to-trough decline during the trading period. Lower is better; indicates risk exposure.",
    avg_trade_pct: "Average return per trade. Shows typical trade performance.",
    best_trade_pct: "Your best single trade return. Shows upside potential.",
    worst_trade_pct: "Your worst single trade return. Shows maximum single-trade risk."
};

export async function initStatisticsPanel() {
    await updateStatisticsData();
}

export async function updateStatisticsData() {
    const container = document.getElementById('statistics-content');
    if (!container) return;
    try {
        const response = await fetch('/api/performance/stats');
        const stats = await response.json();
        if (!stats || Object.keys(stats).length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon" aria-hidden="true"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg></div>
                    <p class="empty-state-text">No statistics available yet</p>
                    <p style="font-size: 0.85em; margin-top: 8px; color: var(--text-muted);">
                        Start trading to see metrics.
                    </p>
                </div>
            `;
            return;
        }
        container.innerHTML = renderStatistics(stats);
    } catch (e) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon" aria-hidden="true"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg></div>
                <p class="empty-state-text">Error loading statistics</p>
                <p style="font-size: 0.85em; margin-top: 8px; color: var(--accent-danger);"></p>
            </div>
        `;
        container.querySelector('p:last-child').textContent = e.message;
    }
}

function renderStatistics(stats) {
    const cards = [
        {
            title: "Total Trades",
            value: `${stats.total_trades || 0}`,
            subValue: `${stats.winning_trades || 0}W / ${stats.losing_trades || 0}L`,
            annotation: STAT_ANNOTATIONS.total_trades,
            colorClass: "stat-neutral"
        },
        {
            title: "Win Rate",
            value: `${(stats.win_rate || 0).toFixed(1)}%`,
            annotation: STAT_ANNOTATIONS.win_rate,
            colorClass: (stats.win_rate || 0) >= 50 ? "stat-positive" : "stat-negative"
        },
        {
            title: "Total P&L",
            value: `${(stats.total_pnl_pct || 0) >= 0 ? '+' : ''}${(stats.total_pnl_pct || 0).toFixed(2)}%`,
            subValue: `$${(stats.total_pnl_quote || 0).toFixed(2)}`,
            annotation: STAT_ANNOTATIONS.total_pnl_pct,
            colorClass: (stats.total_pnl_quote || 0) >= 0 ? "stat-positive" : "stat-negative"
        },
        {
            title: "Current Capital",
            value: new Intl.NumberFormat(navigator.language, { style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(stats.current_capital || 0),
            subValue: `Started: ${new Intl.NumberFormat(navigator.language, { style: 'currency', currency: 'USD' }).format(stats.initial_capital || 0)}`,
            annotation: "Your current trading capital after all gains and losses.",
            colorClass: (stats.current_capital || 0) >= (stats.initial_capital || 0) ? "stat-positive" : "stat-negative"
        },
        {
            title: "Sharpe Ratio",
            value: (stats.sharpe_ratio || 0).toFixed(2),
            annotation: STAT_ANNOTATIONS.sharpe_ratio,
            colorClass: (stats.sharpe_ratio || 0) >= 1 ? "stat-positive" : ((stats.sharpe_ratio || 0) >= 0 ? "stat-neutral" : "stat-negative")
        },
        {
            title: "Sortino Ratio",
            value: (stats.sortino_ratio || 0).toFixed(2),
            annotation: STAT_ANNOTATIONS.sortino_ratio,
            colorClass: (stats.sortino_ratio || 0) >= 1 ? "stat-positive" : ((stats.sortino_ratio || 0) >= 0 ? "stat-neutral" : "stat-negative")
        },
        {
            title: "Profit Factor",
            value: (stats.profit_factor || 0).toFixed(2),
            annotation: STAT_ANNOTATIONS.profit_factor,
            colorClass: (stats.profit_factor || 0) >= 1.5 ? "stat-positive" : ((stats.profit_factor || 0) >= 1 ? "stat-neutral" : "stat-negative")
        },
        {
            title: "Max Drawdown",
            value: `${(stats.max_drawdown_pct || 0).toFixed(2)}%`,
            annotation: STAT_ANNOTATIONS.max_drawdown_pct,
            colorClass: "stat-negative"
        },
        {
            title: "Avg Trade",
            value: `${(stats.avg_trade_pct || 0) >= 0 ? '+' : ''}${(stats.avg_trade_pct || 0).toFixed(2)}%`,
            annotation: STAT_ANNOTATIONS.avg_trade_pct,
            colorClass: (stats.avg_trade_pct || 0) >= 0 ? "stat-positive" : "stat-negative"
        },
        {
            title: "Best Trade",
            value: `+${(stats.best_trade_pct || 0).toFixed(2)}%`,
            annotation: STAT_ANNOTATIONS.best_trade_pct,
            colorClass: "stat-positive"
        },
        {
            title: "Worst Trade",
            value: `${(stats.worst_trade_pct || 0).toFixed(2)}%`,
            annotation: STAT_ANNOTATIONS.worst_trade_pct,
            colorClass: "stat-negative"
        }
    ];
    const lastUpdated = stats.last_updated ? new Intl.DateTimeFormat(navigator.language, { dateStyle: 'short', timeStyle: 'short' }).format(new Date(stats.last_updated)) : 'Unknown';
    return `
        <div style="font-size: 11px; color: var(--text-dim); padding: 0 20px; margin-bottom: 10px;">
            Last updated: ${escapeHtml(lastUpdated)}
        </div>
        <div class="stats-grid">
            ${cards.map(card => `
                <div class="stat-card-lg">
                    <div class="stat-header">
                        <span class="stat-title">${escapeHtml(card.title)}</span>
                    </div>
                    <div class="stat-value-lg ${escapeHtml(card.colorClass)}">${escapeHtml(card.value)}</div>
                    ${card.subValue ? `<div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">${escapeHtml(card.subValue)}</div>` : ''}
                    <div class="stat-annotation">${escapeHtml(card.annotation)}</div>
                </div>
            `).join('')}
        </div>
    `;
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

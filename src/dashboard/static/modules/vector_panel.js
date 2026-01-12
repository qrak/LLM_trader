export async function initVectorPanel() {
    // Initial load
    await updateVectorData();
}

export async function updateVectorData() {
    try {
        const response = await fetch('/api/brain/vectors?limit=50');
        const data = await response.json();
        
        renderVectorPanel(data);
    } catch (e) {
        console.error("Failed to fetch vector data", e);
        renderEmptyState();
    }
}

function renderVectorPanel(data) {
    const container = document.getElementById('vector-content');
    if (!container) return;
    
    // Build stats cards
    const statsHtml = renderStatsCards(data);
    
    // Build experience table
    const tableHtml = renderExperienceTable(data.experiences || []);
    
    container.innerHTML = `
        <div class="vector-stats">${statsHtml}</div>
        <div class="vector-table">${tableHtml}</div>
    `;
}

function renderStatsCards(data) {
    const count = data.experience_count || 0;
    const ruleCount = data.rule_count || 0;
    const confStats = data.confidence_stats || {};
    const adxStats = data.adx_stats || {};
    
    // Extract win rates from confidence stats
    const highWR = confStats.HIGH?.win_rate?.toFixed(0) || '--';
    const medWR = confStats.MEDIUM?.win_rate?.toFixed(0) || '--';
    const lowWR = confStats.LOW?.win_rate?.toFixed(0) || '--';
    
    // Extract from ADX stats
    const adxLowWR = adxStats.LOW?.win_rate?.toFixed(0) || '--';
    const adxMedWR = adxStats.MEDIUM?.win_rate?.toFixed(0) || '--';
    const adxHighWR = adxStats.HIGH?.win_rate?.toFixed(0) || '--';
    
    return `
        <div class="stat-card">
            <div class="stat-label">Total Memories</div>
            <div class="stat-value">${count}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Semantic Rules</div>
            <div class="stat-value">${ruleCount}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Win Rate by Confidence</div>
            <div class="stat-breakdown">
                <span class="high">HIGH: ${highWR}%</span>
                <span class="med">MED: ${medWR}%</span>
                <span class="low">LOW: ${lowWR}%</span>
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Win Rate by ADX</div>
            <div class="stat-breakdown">
                <span class="high">HIGH: ${adxHighWR}%</span>
                <span class="med">MED: ${adxMedWR}%</span>
                <span class="low">LOW: ${adxLowWR}%</span>
            </div>
        </div>
    `;
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
    
    // UPDATE entries are now filtered at backend query level
    const rows = experiences.map(exp => {
        const meta = exp.metadata || {};
        const outcome = meta.outcome || '--';
        const outcomeClass = outcome === 'WIN' ? 'win' : outcome === 'LOSS' ? 'loss' : '';
        const pnl = meta.pnl_pct !== undefined ? `${meta.pnl_pct >= 0 ? '+' : ''}${meta.pnl_pct.toFixed(2)}%` : '--';
        const pnlClass = meta.pnl_pct >= 0 ? 'positive' : 'negative';
        const confidence = meta.confidence || '--';
        const direction = meta.direction || '--';
        const similarity = exp.similarity !== undefined ? `${exp.similarity}%` : '--';
        const timestamp = meta.timestamp ? new Date(meta.timestamp).toLocaleDateString() : '--';
        
        // Truncate document/context
        const context = (exp.document || '').substring(0, 60) + ((exp.document?.length > 60) ? '...' : '');
        
        return `
            <tr title="${exp.document || ''}">
                <td>${exp.id || '--'}</td>
                <td class="context">${context}</td>
                <td class="${outcomeClass}">${outcome}</td>
                <td class="${pnlClass}">${pnl}</td>
                <td>${confidence}</td>
                <td>${direction}</td>
                <td>${similarity}</td>
                <td>${timestamp}</td>
            </tr>
        `;
    }).join('');
    
    return `
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Context</th>
                    <th>Outcome</th>
                    <th>P&L</th>
                    <th>Confidence</th>
                    <th>Direction</th>
                    <th>Similarity</th>
                    <th>Date</th>
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

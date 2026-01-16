import { initPerformanceChart, updatePerformanceData } from './modules/performance_chart.js?v=3.1';
import { initSynapseNetwork, updateSynapses } from './modules/synapse_viewer.js?v=3.1';
import { updateLogs, updatePromptTab, updateResponseTab } from './modules/log_viewer.js?v=3.1';
import { updateVisuals } from './modules/visuals.js?v=3.3';
import { initVectorPanel, updateVectorData } from './modules/vector_panel.js?v=3.1';
import { initFullscreen } from './modules/fullscreen.js?v=3.2';
import { initWebSocket, startCountdownLoop } from './modules/websocket.js?v=3.1';
import { initPositionPanel, updatePositionData } from './modules/position_panel.js?v=3.1';
import { initUI } from './modules/ui.js?v=3.1';
import { initStatisticsPanel, updateStatisticsData } from './modules/statistics_panel.js?v=3.1';
import { initNewsPanel, updateNewsData } from './modules/news_panel.js?v=3.1';

const state = {
    isConnected: false,
    pollInterval: 10000,
    lastUpdateTime: null
};

function formatCost(cost) {
    if (cost === 0 || cost === null || cost === undefined) return '$0.00';
    if (cost < 0.0001) return `$${cost.toFixed(8)}`;
    if (cost < 0.01) return `$${cost.toFixed(6)}`;
    if (cost < 1) return `$${cost.toFixed(4)}`;
    return `$${cost.toFixed(2)}`;
}

async function fetchCosts() {
    try {
        const response = await fetch('/api/monitor/costs');
        const data = await response.json();
        updateCostDisplay(data);
    } catch (e) {
        console.error("Failed to fetch costs", e);
    }
}

function updateCostDisplay(data) {
    const costs = data.costs_by_provider || {};
    const total = data.total_session_cost || 0;
    const orCost = costs.openrouter || 0;
    const googleCost = costs.google || 0;
    
    // Update main total
    document.getElementById('overview-cost').textContent = formatCost(total);
    
    // Update tooltip or detailed view if we add one, for now just ensures these elements exist if we want to show them
    // The user requested removing LM Studio (free) from costs.
    // We already have specific IDs in index.html for specific providers if we want to show them in a detail view, 
    // but the request was "Overview with graph... Session Cost... I should have reset button". 
    // The reset button is in HTML. We just need to make sure the costs are accurate.
}

async function refreshCosts(btn) {
    if (btn) btn.classList.add('loading');
    try {
        await fetchCosts();
    } finally {
        if (btn) btn.classList.remove('loading');
    }
}

async function confirmResetCosts(btn) {
    if (confirm('Are you sure you want to reset all API costs to zero?')) {
        if (btn) btn.classList.add('loading');
        try {
            await resetCosts();
        } finally {
            if (btn) btn.classList.remove('loading');
        }
    }
}

async function resetCosts() {
    try {
        await fetch('/api/monitor/costs/reset', { method: 'POST' });
        await fetchCosts();
    } catch (e) {
        console.error("Failed to reset costs", e);
    }
}

async function fetchBrainStatus() {
    try {
        const response = await fetch('/api/brain/status');
        const data = await response.json();
        document.getElementById('connection-status').textContent = '🟢 Connected';
        document.getElementById('connection-status').style.color = '#238636';
        if (data.status) {
            document.getElementById('brain-state-indicator').textContent = data.status.toUpperCase();
        }
        document.getElementById('trend-val').textContent = data.trend || '--';
        document.getElementById('conf-val').textContent = data.confidence ? `${data.confidence}%` : '--';
        document.getElementById('action-val').textContent = data.action || '--';
        const trendEl = document.getElementById('trend-val');
        if (data.trend === 'BULLISH') {
            trendEl.style.color = '#238636';
        } else if (data.trend === 'BEARISH') {
            trendEl.style.color = '#f85149';
        } else {
            trendEl.style.color = '#8b949e';
        }
        const actionEl = document.getElementById('action-val');
        if (data.action === 'BUY') {
            actionEl.style.color = '#238636';
        } else if (data.action === 'SELL') {
            actionEl.style.color = '#f85149';
        } else {
            actionEl.style.color = '#58a6ff';
        }
        state.lastUpdateTime = new Date();
        updateLastUpdated();
    } catch (e) {
        document.getElementById('connection-status').textContent = '🔴 Disconnected';
        document.getElementById('connection-status').style.color = '#f85149';
    }
}

async function fetchRules() {
    try {
        const response = await fetch('/api/brain/rules');
        const rules = await response.json();
        const list = document.getElementById('rules-list');
        list.innerHTML = '';
        if (rules.length === 0) {
            const li = document.createElement('li');
            li.textContent = 'No rules learned yet. Trade more to build knowledge.';
            li.style.fontStyle = 'italic';
            list.appendChild(li);
            return;
        }
        rules.forEach(rule => {
            const li = document.createElement('li');
            li.textContent = rule.rule_text || rule.text || JSON.stringify(rule);
            list.appendChild(li);
        });
    } catch (e) {
        console.error("Failed to fetch rules", e);
    }
}

function updateLastUpdated() {
    const el = document.getElementById('last-updated');
    if (el && state.lastUpdateTime) {
        el.textContent = `Updated: ${state.lastUpdateTime.toLocaleTimeString()}`;
    }
}

function togglePanelMinimize(panelId) {
    const panel = document.getElementById(panelId);
    if (panel) {
        panel.classList.toggle('minimized');
        const btn = panel.querySelector('.toolbar-btn[title="Minimize"]');
        if (btn) {
            btn.textContent = panel.classList.contains('minimized') ? '+' : '−';
            btn.title = panel.classList.contains('minimized') ? 'Expand' : 'Minimize';
        }
    }
}

async function updateAll() {
    await fetchBrainStatus();
    await fetchRules();
    await fetchCosts();
    await updatePerformanceData();
    await updateSynapses();
    await updateLogs();
    await updateVisuals();
    await updateVectorData();
    await updatePositionData();
    await updateStatisticsData();
    await updateNewsData();
}

// Initialize application
function initApp() {
    console.log('Initializing Dashboard App...');
    
    // Make crucial functions global immediately
    window.updateAll = updateAll;
    window.updatePerformance = () => updatePerformanceData();
    window.updateVisuals = async (btn) => {
        if (btn) btn.classList.add('loading');
        try {
            await updateVisuals();
        } finally {
            if (btn) btn.classList.remove('loading');
        }
    };
    window.updateLogView = () => updateLogs();
    window.updateVectors = () => updateVectorData();
    window.updatePosition = () => updatePositionData(null, true);
    window.updatePromptTab = () => updatePromptTab();
    window.updateResponseTab = () => updateResponseTab();
    window.updateStatisticsPanel = () => updateStatisticsData();
    window.updateNewsPanel = () => updateNewsData();
    window.resetCosts = resetCosts;
    window.refreshCosts = refreshCosts;
    window.confirmResetCosts = confirmResetCosts;
    window.togglePanelMinimize = togglePanelMinimize;

    try {
        initPerformanceChart();
        initSynapseNetwork();
        initVectorPanel();
        initFullscreen();
        initPositionPanel();
        initStatisticsPanel();
        initNewsPanel();
        initWebSocket();
        initUI();
        startCountdownLoop();
        
        // Initial update
        updateAll();
        
        // Start polling
        setInterval(updateAll, state.pollInterval);

        // Listen for WS analysis complete
        document.addEventListener('analysis-complete', () => {
            console.log('Analysis complete, refreshing...');
            updateAll();
        });
        
        console.log('Dashboard App Initialized');
    } catch (e) {
        console.error('Error initializing dashboard:', e);
    }
}

// Run init when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
} else {
    initApp();
}


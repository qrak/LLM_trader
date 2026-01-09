import { initPerformanceChart, updatePerformanceData } from './modules/performance_chart.js';
import { initSynapseNetwork, updateSynapses } from './modules/synapse_viewer.js';
import { updateLogs } from './modules/log_viewer.js';
import { updateVisuals } from './modules/visuals.js';
import { initVectorPanel, updateVectorData } from './modules/vector_panel.js';
import { initFullscreen } from './modules/fullscreen.js';
import { initWebSocket, startCountdownLoop } from './modules/websocket.js';
import { initPositionPanel, updatePositionData } from './modules/position_panel.js';

const state = {
    isConnected: false,
    pollInterval: 10000,
    lastUpdateTime: null
};

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

async function updateAll() {
    await fetchBrainStatus();
    await fetchRules();
    await updatePerformanceData();
    await updateSynapses();
    await updateLogs();
    await updateVisuals();
    await updateVectorData();
    await updatePositionData();
}

document.addEventListener('DOMContentLoaded', () => {
    initPerformanceChart();
    initSynapseNetwork();
    initVectorPanel();
    initFullscreen();
    initPositionPanel();
    initWebSocket();
    startCountdownLoop();
    updateAll();
    setInterval(updateAll, state.pollInterval);
    window.updatePerformance = () => updatePerformanceData();
    window.updateVisuals = () => updateVisuals();
    window.updateLogView = () => updateLogs();
    window.updateVectors = () => updateVectorData();
    window.updatePosition = () => updatePositionData(null, true);
    document.addEventListener('analysis-complete', () => {
        console.log('Analysis complete, refreshing...');
        updateAll();
    });
});

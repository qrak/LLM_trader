import { initPerformanceChart, updatePerformanceData } from './modules/performance_chart.js?v=4.5';
import { initSynapseNetwork, updateSynapses } from './modules/synapse_viewer.js?v=4.5';
import { updateLogs, updatePromptTab, updateResponseTab } from './modules/log_viewer.js?v=4.5';
import { updateVisuals } from './modules/visuals.js?v=4.5';
import { initVectorPanel, updateVectorData } from './modules/vector_panel.js?v=4.5';
import { initFullscreen } from './modules/fullscreen.js?v=4.5';
import { initWebSocket, startCountdownLoop } from './modules/websocket.js?v=4.5';
import { initPositionPanel, updatePositionData } from './modules/position_panel.js?v=4.5';
import { initUI } from './modules/ui.js?v=4.5';
import { initStatisticsPanel, updateStatisticsData } from './modules/statistics_panel.js?v=4.5';
import { initNewsPanel, updateNewsData } from './modules/news_panel.js?v=4.5';

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







async function fetchBrainStatus() {
    try {
        const response = await fetch('/api/brain/status');
        const data = await response.json();
        const connStatus = document.getElementById('connection-status');
        const statusDot = document.querySelector('.status-dot');

        if (connStatus) {
            connStatus.textContent = 'Connected';
            connStatus.classList.remove('status-text', 'disconnected');
            connStatus.classList.add('status-text', 'connected');
            connStatus.style.color = ''; // Clear inline style if present
        }

        if (statusDot) {
            statusDot.classList.remove('disconnected');
            statusDot.classList.add('connected');
        }
        
        // Update Brain State Indicator
        // (Legacy indicator removed from UI, skipping update)

        // Direct update to Overview KPIs
        const trendEl = document.getElementById('overview-trend');
        if (trendEl) {
            trendEl.textContent = data.trend || '--';
            if (data.trend === 'BULLISH') {
                trendEl.className = 'value start-green';
            } else if (data.trend === 'BEARISH') {
                trendEl.className = 'value start-red';
            } else {
                trendEl.className = 'value'; // default color
            }
        }

        const confEl = document.getElementById('overview-conf');
        if (confEl) {
            confEl.textContent = data.confidence ? `${data.confidence}%` : '--%';
        }

        const actionEl = document.getElementById('overview-action');
        if (actionEl) {
            actionEl.textContent = data.action || 'WAITING';
            // Styling logic for action (sub-label text color usually muted, but user had color logic)
            // The previous logic colored the text. Let's keep it consistent if possible, 
            // but usually sub-labels are muted. The legacy code colored #action-val.
            // syncStatus copied text content, but NOT style. 
            // WAIT - syncStatus did NOT copy style from action-val to overview-action.
            // It only did: document.getElementById('overview-action').textContent = act;
            // So visible UI was NOT colored. I will stick to text content to match visible behavior.
        }

        state.lastUpdateTime = new Date();
        updateLastUpdated();
    } catch (e) {
        const connStatus = document.getElementById('connection-status');
        const statusDot = document.querySelector('.status-dot');

        if (connStatus) {
            connStatus.textContent = 'Disconnected';
            connStatus.classList.remove('status-text', 'connected');
            connStatus.classList.add('status-text', 'disconnected');
            connStatus.style.color = ''; // Clear inline style if present
        }

        if (statusDot) {
            statusDot.classList.remove('connected');
            statusDot.classList.add('disconnected');
        }
    }
}

async function fetchRules() {
    try {
        const response = await fetch('/api/brain/rules');
        const rules = await response.json();
        
        // Direct update to Rules Count KPI
        const countEl = document.getElementById('overview-rules-count');
        const hintEl = document.getElementById('overview-rules-hint');
        
        if (countEl) {
             const count = rules.length;
             countEl.textContent = count;
             
             if (hintEl) {
                 hintEl.style.display = count > 0 ? 'none' : 'block';
             }
        }
        
    } catch (e) {
        console.error("Failed to fetch rules", e);
    }
}

function updateLastUpdated() {
    const el = document.getElementById('last-updated');
    if (el && state.lastUpdateTime) {
        el.textContent = `Updated: ${new Intl.DateTimeFormat(navigator.language, { timeStyle: 'medium' }).format(state.lastUpdateTime)}`;
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

    window.togglePanelMinimize = togglePanelMinimize;

    // Mobile Menu Logic
    const toggleBtn = document.getElementById('mobile-menu-toggle');
    const closeBtn = document.getElementById('mobile-menu-close');
    const sidebar = document.getElementById('sidebar');
    
    function toggleMobileMenu() {
        sidebar.classList.toggle('mobile-open');
    }

    if (toggleBtn) toggleBtn.addEventListener('click', toggleMobileMenu);
    if (closeBtn) closeBtn.addEventListener('click', () => sidebar.classList.remove('mobile-open'));

    // Close on navigation
    document.querySelectorAll('.nav-item').forEach(link => {
        link.addEventListener('click', function() {
            // Update aria-selected for all tabs
            document.querySelectorAll('.nav-item').forEach(t => t.setAttribute('aria-selected', 'false'));
            // Set current to true
            this.setAttribute('aria-selected', 'true');

            if (window.innerWidth <= 768) {
                sidebar.classList.remove('mobile-open');
            }
        });
    });

    try {
        // Event listeners for static buttons
        const btnMinimize = document.getElementById('btn-minimize-visuals');
        if (btnMinimize) {
            btnMinimize.addEventListener('click', () => togglePanelMinimize('panel-visuals'));
        }

        const btnLightbox = document.getElementById('btn-lightbox-visuals');
        if (btnLightbox) {
            btnLightbox.addEventListener('click', () => {
                const img = document.getElementById('analysis-chart');
                if (img && img.src && window.openLightbox) window.openLightbox(img.src);
            });
        }

        const btnCopyPrompt = document.getElementById('btn-copy-prompt');
        if (btnCopyPrompt && window.copyPromptContent) {
            btnCopyPrompt.addEventListener('click', window.copyPromptContent);
        }

        const btnCopyResponse = document.getElementById('btn-copy-response');
        if (btnCopyResponse && window.copyResponseContent) {
            btnCopyResponse.addEventListener('click', window.copyResponseContent);
        }

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


/**
 * Fullscreen panel module - Opens panels in modal overlay.
 * Also handles panel visibility toggling.
 * 
 * IMPORTANT: For canvas-based charts (ApexCharts, vis-network), we MOVE the original
 * element to the fullscreen container instead of cloning, then move it back on close.
 */

let currentPanel = null;
let originalParent = null;
let originalNextSibling = null;

/**
 * Initialize fullscreen functionality.
 */
export function initFullscreen() {
    createModalContainer();
    attachPanelButtons();
}

/**
 * Create the modal container element.
 */
function createModalContainer() {
    if (document.getElementById('fullscreen-modal')) return;
    const modal = document.createElement('div');
    modal.id = 'fullscreen-modal';
    modal.className = 'fullscreen-modal';
    modal.innerHTML = `
        <div class="fullscreen-backdrop"></div>
        <div class="fullscreen-content">
            <div class="fullscreen-header">
                <h2 id="fullscreen-title"></h2>
                <button class="fullscreen-close" onclick="window.closeFullscreen()">✕</button>
            </div>
            <div id="fullscreen-body"></div>
        </div>
    `;
    document.body.appendChild(modal);
    modal.querySelector('.fullscreen-backdrop').addEventListener('click', closeFullscreen);
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeFullscreen();
    });
}

/**
 * Attach expand and hide buttons to all panels.
 */
function attachPanelButtons() {
    document.querySelectorAll('.panel').forEach(panel => {
        const header = panel.querySelector('.panel-header');
        if (!header) return;
        const panelId = panel.id || 'unknown';
        if (header.querySelector('.panel-controls')) return;
        const controls = document.createElement('div');
        controls.className = 'panel-controls';
        controls.innerHTML = `
            <button class="panel-btn hide-btn" onclick="window.togglePanel('${panelId}')" title="Hide panel">−</button>
            <button class="panel-btn expand-btn" onclick="window.openFullscreen('${panelId}')" title="Fullscreen">⛶</button>
        `;
        header.appendChild(controls);
    });
}

/**
 * Open a panel in fullscreen modal.
 */
export function openFullscreen(panelId) {
    const panel = document.getElementById(panelId);
    if (!panel) return;
    const modal = document.getElementById('fullscreen-modal');
    const titleEl = panel.querySelector('.panel-header h3');
    const title = titleEl ? titleEl.textContent : 'Panel';
    document.getElementById('fullscreen-title').textContent = title;
    const body = document.getElementById('fullscreen-body');
    body.innerHTML = '';
    const contentId = getContentId(panelId);
    const contentEl = document.getElementById(contentId);
    if (contentEl) {
        originalParent = contentEl.parentElement;
        originalNextSibling = contentEl.nextSibling;
        currentPanel = panelId;
        body.appendChild(contentEl);
        setTimeout(() => {
            resizeContent(panelId, contentEl);
        }, 50);
    }
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';
}

/**
 * Get the content element ID for a panel.
 */
function getContentId(panelId) {
    const mapping = {
        'panel-performance': 'performance-chart',
        'panel-synapses': 'synapse-network',
        'panel-thoughts': 'log-viewer',
        'panel-visuals': 'visual-container',
        'panel-position': 'position-content',
        'panel-vectors': 'vector-content'
    };
    return mapping[panelId] || null;
}

/**
 * Resize content after moving to fullscreen.
 */
function resizeContent(panelId, contentEl) {
    if (panelId === 'panel-performance') {
        const chart = window.performanceChart;
        if (chart && typeof chart.render === 'function') {
            contentEl.style.width = '100%';
            contentEl.style.height = '100%';
            const newHeight = contentEl.offsetHeight || 500;
            const newWidth = contentEl.offsetWidth || 800;
            chart.updateOptions({
                chart: {
                    height: newHeight,
                    width: newWidth
                }
            }, false, true);
            setTimeout(() => {
                chart.resetSeries(true, true);
            }, 100);
        }
    } else if (panelId === 'panel-synapses') {
        const network = window.synapseNetwork;
        if (network && typeof network.fit === 'function') {
            contentEl.style.width = '100%';
            contentEl.style.height = '100%';
            setTimeout(() => {
                network.setSize(contentEl.offsetWidth || 800, contentEl.offsetHeight || 500);
                network.redraw();
                network.fit({ animation: { duration: 300 } });
            }, 100);
        }
    }
}

/**
 * Close the fullscreen modal.
 */
export function closeFullscreen() {
    const modal = document.getElementById('fullscreen-modal');
    if (!modal.classList.contains('active')) return;
    const body = document.getElementById('fullscreen-body');
    const contentEl = body.firstElementChild;
    const panelId = currentPanel;
    if (contentEl && originalParent) {
        if (originalNextSibling) {
            originalParent.insertBefore(contentEl, originalNextSibling);
        } else {
            originalParent.appendChild(contentEl);
        }
        setTimeout(() => {
            restoreContent(panelId, contentEl);
        }, 50);
    }
    modal.classList.remove('active');
    document.body.style.overflow = '';
    currentPanel = null;
    originalParent = null;
    originalNextSibling = null;
}

/**
 * Restore content size after returning from fullscreen.
 */
function restoreContent(panelId, contentEl) {
    if (panelId === 'panel-performance') {
        const chart = window.performanceChart;
        if (chart) {
            contentEl.style.width = '';
            contentEl.style.height = '';
            setTimeout(() => {
                chart.updateOptions({
                    chart: {
                        height: 220,
                        width: undefined
                    }
                }, true, true);
                window.dispatchEvent(new Event('resize'));
            }, 150);
        }
    } else if (panelId === 'panel-synapses') {
        const network = window.synapseNetwork;
        if (network && typeof network.fit === 'function') {
            contentEl.style.width = '';
            contentEl.style.height = '';
            setTimeout(() => {
                network.setSize('100%', '100%');
                network.redraw();
                network.fit({ animation: false });
            }, 100);
        }
    }
}

/**
 * Toggle panel visibility.
 */
export function togglePanel(panelId) {
    const panel = document.getElementById(panelId);
    if (!panel) return;
    const isHidden = panel.classList.toggle('panel-hidden');
    const hideBtn = panel.querySelector('.hide-btn');
    if (hideBtn) {
        hideBtn.textContent = isHidden ? '+' : '−';
        hideBtn.title = isHidden ? 'Show panel' : 'Hide panel';
    }
}

window.openFullscreen = openFullscreen;
window.closeFullscreen = closeFullscreen;
window.togglePanel = togglePanel;

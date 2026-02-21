/**
 * Fullscreen panel module - Opens panels in modal overlay.
 * 
 * For canvas-based charts (ApexCharts, vis-network), we MOVE the original
 * element to the fullscreen container instead of cloning, then move it back on close.
 */

let currentPanel = null;
let originalParent = null;
let originalNextSibling = null;
let lastFocusedElement = null;

export function initFullscreen() {
    createModalContainer();
    attachPanelButtons();
}

function createModalContainer() {
    if (document.getElementById('fullscreen-modal')) return;
    const modal = document.createElement('div');
    modal.id = 'fullscreen-modal';
    modal.className = 'fullscreen-modal';
    modal.setAttribute('role', 'dialog');
    modal.setAttribute('aria-modal', 'true');
    modal.setAttribute('aria-labelledby', 'fullscreen-title');
    modal.innerHTML = `
        <div class="fullscreen-backdrop"></div>
        <div class="fullscreen-content">
            <div class="fullscreen-header">
                <h2 id="fullscreen-title"></h2>
                <button class="fullscreen-close" aria-label="Close fullscreen">✕</button>
            </div>
            <div id="fullscreen-body"></div>
        </div>
    `;
    document.body.appendChild(modal);
    modal.querySelector('.fullscreen-backdrop').addEventListener('click', closeFullscreen);
    modal.querySelector('.fullscreen-close').addEventListener('click', closeFullscreen);
    document.addEventListener('keydown', (e) => {
        if (!modal.classList.contains('active')) return;

        if (e.key === 'Escape') {
            closeFullscreen();
            return;
        }

        if (e.key === 'Tab') {
            const focusable = modal.querySelectorAll('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
            if (focusable.length === 0) return;

            const first = focusable[0];
            const last = focusable[focusable.length - 1];

            if (e.shiftKey) {
                if (document.activeElement === first) {
                    e.preventDefault();
                    last.focus();
                }
            } else {
                if (document.activeElement === last) {
                    e.preventDefault();
                    first.focus();
                }
            }
        }
    });
}

function attachPanelButtons() {
    document.querySelectorAll('.panel').forEach(panel => {
        const header = panel.querySelector('.panel-header');
        if (!header) return;
        const panelId = panel.id || 'unknown';
        if (panelId === 'unknown') return;
        if (panelId === 'panel-visuals') return;
        if (header.querySelector('.panel-controls') || header.querySelector('.panel-toolbar')) return;
        const controls = document.createElement('div');
        controls.className = 'panel-controls';

        const btn = document.createElement('button');
        btn.className = 'panel-btn expand-btn';
        btn.title = 'Fullscreen';
        btn.setAttribute('aria-label', 'Enter fullscreen');
        btn.textContent = '⛶';
        btn.addEventListener('click', () => openFullscreen(panelId));

        controls.appendChild(btn);
        header.appendChild(controls);
    });
}

export function openFullscreen(panelId) {
    lastFocusedElement = document.activeElement;
    const panel = document.getElementById(panelId);
    if (!panel) {
        console.error('Panel not found:', panelId);
        return;
    }
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
    } else {
        console.error('Content element not found for panel:', panelId, 'content ID:', contentId);
    }
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';

    // Trap focus and hide background
    const appContainer = document.getElementById('app-container');
    if (appContainer) appContainer.setAttribute('aria-hidden', 'true');

    const closeBtn = modal.querySelector('.fullscreen-close');
    if (closeBtn) closeBtn.focus();
}

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
                // NEW: Re-apply annotations at larger scale
                if (typeof window.refreshPerformanceAnnotations === 'function') {
                    window.refreshPerformanceAnnotations();
                }
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

export function closeFullscreen() {
    const modal = document.getElementById('fullscreen-modal');
    if (!modal || !modal.classList.contains('active')) return;
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

    const appContainer = document.getElementById('app-container');
    if (appContainer) appContainer.removeAttribute('aria-hidden');

    if (lastFocusedElement) {
        lastFocusedElement.focus();
        lastFocusedElement = null;
    }

    currentPanel = null;
    originalParent = null;
    originalNextSibling = null;
}

function restoreContent(panelId, contentEl) {
    if (panelId === 'panel-performance') {
        const chart = window.performanceChart;
        if (chart) {
            contentEl.style.width = '';
            contentEl.style.height = '';
            setTimeout(() => {
                chart.updateOptions({
                    chart: {
                        height: 450,
                        width: undefined
                    }
                }, true, true);
                window.dispatchEvent(new Event('resize'));
                // NEW: Restore annotations to original scale
                if (typeof window.refreshPerformanceAnnotations === 'function') {
                    window.refreshPerformanceAnnotations();
                }
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

window.openFullscreen = openFullscreen;
window.closeFullscreen = closeFullscreen;

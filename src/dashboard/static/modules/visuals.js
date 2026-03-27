export async function updateVisuals() {
    const container = document.getElementById('visual-container');
    const img = document.getElementById('analysis-chart');
    const noChartMsg = document.getElementById('no-chart-msg');
    
    try {
        // Try to fetch latest chart with cache buster
        const timestamp = Date.now();
        const response = await fetch(`/api/visuals/charts/latest?t=${timestamp}`);
        
        if (response.ok) {
            const data = await response.json();
            
            if (data.chart_base64) {
                img.src = `data:image/png;base64,${data.chart_base64}`;
                img.style.display = 'block';
                img.alt = 'Analysis chart';
                noChartMsg.style.display = 'none';
                
                // Add timestamp overlay
                updateChartTimestamp(data.timestamp);
                
                // Add click to enlarge
                img.onclick = () => openLightbox(img.src);
                img.style.cursor = 'pointer';
                img.title = 'Click to enlarge';
                img.setAttribute('role', 'button');
                img.setAttribute('tabindex', '0');
                img.setAttribute('aria-haspopup', 'dialog');
                img.setAttribute('aria-controls', 'lightbox-overlay');
                img.setAttribute('aria-label', 'Open full screen chart');
                img.onkeydown = (e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        openLightbox(img.src);
                    }
                };
            } else if (data.chart_url) {
                img.src = data.chart_url + `?t=${timestamp}`;
                img.style.display = 'block';
                img.alt = 'Analysis chart';
                noChartMsg.style.display = 'none';
                img.onclick = () => openLightbox(img.src);
                img.style.cursor = 'pointer';
                img.title = 'Click to enlarge';
                img.setAttribute('role', 'button');
                img.setAttribute('tabindex', '0');
                img.setAttribute('aria-haspopup', 'dialog');
                img.setAttribute('aria-controls', 'lightbox-overlay');
                img.setAttribute('aria-label', 'Open full screen chart');
                img.onkeydown = (e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        openLightbox(img.src);
                    }
                };
            } else {
                img.style.display = 'none';
                noChartMsg.textContent = 'No chart in memory - waiting for next analysis';
                noChartMsg.style.display = 'block';
            }
        } else {
            img.style.display = 'none';
            noChartMsg.textContent = 'Chart service unavailable';
            noChartMsg.style.display = 'block';
        }
    } catch (e) {
        console.error("Failed to update visuals", e);
        img.style.display = 'none';
        noChartMsg.textContent = 'Error loading chart';
        noChartMsg.style.display = 'block';
    }
}

function updateChartTimestamp(timestamp) {
    let overlay = document.getElementById('chart-timestamp');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'chart-timestamp';
        overlay.style.cssText = 'position: absolute; bottom: 5px; right: 5px; font-size: 9px; color: #8b949e; background: rgba(0,0,0,0.7); padding: 2px 5px; border-radius: 3px;';
        document.getElementById('visual-container').style.position = 'relative';
        document.getElementById('visual-container').appendChild(overlay);
    }
    
    if (timestamp) {
        overlay.textContent = new Intl.DateTimeFormat(navigator.language, { dateStyle: 'short', timeStyle: 'short' }).format(new Date(timestamp));
        overlay.style.display = 'block';
    } else {
        overlay.style.display = 'none';
    }
}

function openLightbox(src) {
    // 1. Capture previously focused element to restore later
    const lastFocusedElement = document.activeElement;

    let zoom = 1;
    let panX = 0, panY = 0;
    let panStart = { x: 0, y: 0 };

    const overlay = document.createElement('div');
    overlay.id = 'lightbox-overlay';

    // 2. Add Accessibility Attributes
    overlay.setAttribute('role', 'dialog');
    overlay.setAttribute('aria-modal', 'true');
    overlay.setAttribute('aria-label', 'Image View');

    // 3. Hide main content from screen readers
    const appContainer = document.getElementById('app-container');
    if (appContainer) appContainer.setAttribute('aria-hidden', 'true');

    // Toolbar (without measurement tool)
    const toolbar = document.createElement('div');
    toolbar.className = 'lightbox-toolbar';
    toolbar.innerHTML = `
        <button id="lb-zoom-in" title="Zoom In (+)" aria-label="Zoom In"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line><line x1="11" y1="8" x2="11" y2="14"></line><line x1="8" y1="11" x2="14" y2="11"></line></svg></button>
        <button id="lb-zoom-out" title="Zoom Out (-)" aria-label="Zoom Out"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line><line x1="8" y1="11" x2="14" y2="11"></line></svg></button>
        <button id="lb-reset" title="Reset View (0)" aria-label="Reset View"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"></path><path d="M3 3v5h5"></path></svg></button>
        <span class="zoom-display" id="lb-zoom-display">100%</span>
    `;

    // Close button
    const closeBtn = document.createElement('button');
    closeBtn.className = 'lightbox-close';
    closeBtn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>';
    closeBtn.title = 'Close (Esc)';
    closeBtn.setAttribute('aria-label', 'Close Image View');

    // Viewport
    const viewport = document.createElement('div');
    viewport.className = 'lightbox-viewport';

    const img = document.createElement('img');
    img.src = src;
    img.draggable = false;
    img.alt = "Enlarged chart view";

    viewport.appendChild(img);
    overlay.appendChild(toolbar);
    overlay.appendChild(closeBtn);
    overlay.appendChild(viewport);
    document.body.appendChild(overlay);

    // 4. Focus Management - Set initial focus
    closeBtn.focus();

    function updateTransform() {
        img.style.transform = `translate(${panX}px, ${panY}px) scale(${zoom})`;
        document.getElementById('lb-zoom-display').textContent = Math.round(zoom * 100) + '%';
    }

    function zoomIn() {
        zoom = Math.min(zoom * 1.25, 10);
        updateTransform();
    }

    function zoomOut() {
        zoom = Math.max(zoom / 1.25, 0.1);
        updateTransform();
    }

    function resetView() {
        zoom = 1;
        panX = 0;
        panY = 0;
        updateTransform();
    }

    // Event listeners
    document.getElementById('lb-zoom-in').onclick = zoomIn;
    document.getElementById('lb-zoom-out').onclick = zoomOut;
    document.getElementById('lb-reset').onclick = resetView;
    closeBtn.onclick = () => overlay.remove();

    // Mouse wheel zoom
    viewport.addEventListener('wheel', (e) => {
        e.preventDefault();
        if (e.deltaY < 0) zoomIn();
        else zoomOut();
    }, { passive: false });

    // Pan on drag (always enabled)
    viewport.addEventListener('mousedown', (e) => {
        if (e.target === img || e.target === viewport) {
            viewport.classList.add('panning');
            panStart = { x: e.clientX - panX, y: e.clientY - panY };
        }
    });

    viewport.addEventListener('mousemove', (e) => {
        if (viewport.classList.contains('panning')) {
            panX = e.clientX - panStart.x;
            panY = e.clientY - panStart.y;
            updateTransform();
        }
    });

    viewport.addEventListener('mouseup', () => {
        viewport.classList.remove('panning');
    });

    viewport.addEventListener('mouseleave', () => {
        viewport.classList.remove('panning');
    });

    // Keyboard shortcuts & Focus Trap
    function handleKeydown(e) {
        if (e.key === 'Escape') {
            overlay.remove();
            return;
        }

        // Focus Trap Logic
        if (e.key === 'Tab') {
            const focusableElements = overlay.querySelectorAll('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
            if (focusableElements.length === 0) return;

            const firstElement = focusableElements[0];
            const lastElement = focusableElements[focusableElements.length - 1];

            if (e.shiftKey) { // Shift + Tab
                if (document.activeElement === firstElement) {
                    e.preventDefault();
                    lastElement.focus();
                }
            } else { // Tab
                if (document.activeElement === lastElement) {
                    e.preventDefault();
                    firstElement.focus();
                }
            }
            return;
        }

        if (e.key === '+' || e.key === '=') zoomIn();
        else if (e.key === '-') zoomOut();
        else if (e.key === '0') resetView();
    }
    document.addEventListener('keydown', handleKeydown);

    // Cleanup function
    function cleanup() {
        document.removeEventListener('keydown', handleKeydown);
        if (appContainer) appContainer.removeAttribute('aria-hidden');
        if (lastFocusedElement) lastFocusedElement.focus();
    }

    // Remove overlay cleanup via MutationObserver
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            mutation.removedNodes.forEach((node) => {
                if (node === overlay) {
                    cleanup();
                    observer.disconnect();
                }
            });
        });
    });
    observer.observe(document.body, { childList: true });
}

window.openLightbox = openLightbox;


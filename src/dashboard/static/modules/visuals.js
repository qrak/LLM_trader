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
                noChartMsg.style.display = 'none';
                
                // Add timestamp overlay
                updateChartTimestamp(data.timestamp);
                
                // Add click to enlarge
                img.onclick = () => openLightbox(img.src);
                img.style.cursor = 'pointer';
                img.title = 'Click to enlarge';
            } else if (data.chart_url) {
                img.src = data.chart_url + `?t=${timestamp}`;
                img.style.display = 'block';
                noChartMsg.style.display = 'none';
                img.onclick = () => openLightbox(img.src);
                img.style.cursor = 'pointer';
                img.title = 'Click to enlarge';
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
        overlay.textContent = new Date(timestamp).toLocaleString();
        overlay.style.display = 'block';
    } else {
        overlay.style.display = 'none';
    }
}

function openLightbox(src) {
    let zoom = 1;
    let panX = 0, panY = 0;
    let panStart = { x: 0, y: 0 };

    const overlay = document.createElement('div');
    overlay.id = 'lightbox-overlay';

    // Toolbar (without measurement tool)
    const toolbar = document.createElement('div');
    toolbar.className = 'lightbox-toolbar';
    toolbar.innerHTML = `
        <button id="lb-zoom-in" title="Zoom In (+)">🔍+</button>
        <button id="lb-zoom-out" title="Zoom Out (-)">🔍−</button>
        <button id="lb-reset" title="Reset View (0)">↺</button>
        <span class="zoom-display" id="lb-zoom-display">100%</span>
    `;

    // Close button
    const closeBtn = document.createElement('button');
    closeBtn.className = 'lightbox-close';
    closeBtn.innerHTML = '✕';
    closeBtn.title = 'Close (Esc)';

    // Viewport
    const viewport = document.createElement('div');
    viewport.className = 'lightbox-viewport';

    const img = document.createElement('img');
    img.src = src;
    img.draggable = false;

    viewport.appendChild(img);
    overlay.appendChild(toolbar);
    overlay.appendChild(closeBtn);
    overlay.appendChild(viewport);
    document.body.appendChild(overlay);

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

    // Keyboard shortcuts
    function handleKeydown(e) {
        if (e.key === 'Escape') overlay.remove();
        else if (e.key === '+' || e.key === '=') zoomIn();
        else if (e.key === '-') zoomOut();
        else if (e.key === '0') resetView();
    }
    document.addEventListener('keydown', handleKeydown);

    overlay.addEventListener('remove', () => {
        document.removeEventListener('keydown', handleKeydown);
    });

    // Remove overlay cleanup
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            mutation.removedNodes.forEach((node) => {
                if (node === overlay) {
                    document.removeEventListener('keydown', handleKeydown);
                    observer.disconnect();
                }
            });
        });
    });
    observer.observe(document.body, { childList: true });
}

window.openLightbox = openLightbox;


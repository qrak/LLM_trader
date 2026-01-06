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
    // Create lightbox overlay
    const overlay = document.createElement('div');
    overlay.id = 'lightbox-overlay';
    overlay.style.cssText = `
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(0,0,0,0.9); z-index: 9999;
        display: flex; justify-content: center; align-items: center;
        cursor: pointer;
    `;
    
    const img = document.createElement('img');
    img.src = src;
    img.style.cssText = 'max-width: 95%; max-height: 95%; object-fit: contain; border-radius: 8px;';
    
    const closeBtn = document.createElement('div');
    closeBtn.textContent = '✕';
    closeBtn.style.cssText = `
        position: absolute; top: 20px; right: 30px;
        font-size: 30px; color: #fff; cursor: pointer;
    `;
    
    overlay.appendChild(img);
    overlay.appendChild(closeBtn);
    document.body.appendChild(overlay);
    
    // Close on click
    overlay.onclick = () => overlay.remove();
}

// Export for window access
window.openLightbox = openLightbox;

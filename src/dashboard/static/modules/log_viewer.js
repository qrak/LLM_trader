export async function updateLogs() {
    const type = document.getElementById('log-type').value;
    const endpoint = type === 'prompt' ? '/api/monitor/last_prompt' : '/api/monitor/last_response';
    const viewer = document.getElementById('log-viewer');
    
    try {
        const response = await fetch(endpoint);
        const data = await response.json();
        
        let content = '';
        if (type === 'prompt') {
            content = data.prompt || 'No prompt available';
        } else {
            content = data.response || 'No response available';
        }
        
        // Create formatted content with header and copy button
        const timestamp = data.timestamp ? new Date(data.timestamp).toLocaleString() : 'N/A';
        const source = data.source === 'disk' ? '💾 From disk' : '🧠 From memory';
        
        // Render markdown for response, plain text for prompt
        let formattedContent;
        if (type === 'response' && content && window.marked) {
            formattedContent = marked.parse(content);
        } else {
            // Escape HTML and preserve formatting
            formattedContent = `<pre style="white-space: pre-wrap; margin: 0;">${escapeHtml(content)}</pre>`;
        }
        
        viewer.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 1px solid #30363d;">
                <span style="font-size: 10px; color: #8b949e;">${source} | ${timestamp}</span>
                <button onclick="copyLogContent()" style="font-size: 10px; padding: 2px 8px;">📋 Copy</button>
            </div>
            <div id="log-content" style="overflow-y: auto; max-height: calc(100% - 40px);">
                ${formattedContent}
            </div>
        `;
        
        // Store raw content for copying
        viewer.dataset.rawContent = content;
        
    } catch (e) {
        viewer.textContent = "Error fetching logs: " + e.message;
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Global function for copy button
window.copyLogContent = function() {
    const viewer = document.getElementById('log-viewer');
    const content = viewer.dataset.rawContent || viewer.textContent;
    
    navigator.clipboard.writeText(content).then(() => {
        // Flash button to indicate success
        const btn = viewer.querySelector('button');
        if (btn) {
            const originalText = btn.textContent;
            btn.textContent = '✓ Copied!';
            btn.style.background = '#238636';
            setTimeout(() => {
                btn.textContent = originalText;
                btn.style.background = '';
            }, 1500);
        }
    }).catch(err => {
        console.error('Failed to copy: ', err);
    });
};

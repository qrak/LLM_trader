/**
 * Log viewer module - Handles prompt and response display.
 */

let cachedPrompt = null;
let cachedResponse = null;

export async function updateLogs() {
    // Legacy function - now we update dedicated tabs instead
    await updatePromptTab();
    await updateResponseTab();
}

export async function updatePromptTab() {
    const viewer = document.getElementById('prompt-viewer');
    const meta = document.getElementById('prompt-meta');
    if (!viewer) return;
    try {
        const response = await fetch('/api/monitor/last_prompt');
        const data = await response.json();
        const content = data.prompt || 'No prompt available';
        const timestamp = data.timestamp ? new Date(data.timestamp).toLocaleString() : 'N/A';
        const source = data.source === 'disk' ? '💾 From disk' : '🧠 From memory';
        if (meta) meta.textContent = `${source} | ${timestamp}`;
        // Render prompt with markdown and discord-content styling (same as response)
        if (content && window.marked) {
            viewer.innerHTML = marked.parse(content);
            viewer.classList.remove('prompt-content', 'code-block');
            viewer.classList.add('discord-content');
        } else {
            viewer.textContent = content;
        }
        cachedPrompt = content;
    } catch (e) {
        viewer.textContent = "Error fetching prompt: " + e.message;
    }
}

export async function updateResponseTab() {
    const viewer = document.getElementById('response-viewer');
    const meta = document.getElementById('response-meta');
    if (!viewer) return;
    try {
        const response = await fetch('/api/monitor/last_response');
        const data = await response.json();
        const content = data.response || 'No response available';
        const timestamp = data.timestamp ? new Date(data.timestamp).toLocaleString() : 'N/A';
        const source = data.source === 'disk' ? '💾 From disk' : '🧠 From memory';
        if (meta) meta.textContent = `${source} | ${timestamp}`;
        // Process content for Discord-style display
        if (content && window.marked) {
            let processed = content;
            // Strip JSON code blocks (```json ... ```) for cleaner display
            processed = processed.replace(/```json[\s\S]*?```/g, '');
            // Strip any remaining { "analysis": ... } JSON objects that aren't in code blocks
            processed = processed.replace(/\n*\{\s*"analysis"[\s\S]*?\n\}\s*$/g, '');
            // Highlight warning sections
            processed = processed.replace(/⚠️\s*([^.]+\.\s*)/g, '<div class="warning-banner">⚠️ $1</div>');
            // Clean up any double line breaks left by JSON removal
            processed = processed.replace(/\n{3,}/g, '\n\n');
            viewer.innerHTML = marked.parse(processed.trim());
        } else {
            viewer.innerHTML = `<pre style="white-space: pre-wrap; margin: 0;">${escapeHtml(content)}</pre>`;
        }
        cachedResponse = content;
    } catch (e) {
        viewer.innerHTML = "Error fetching response: " + e.message;
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

window.copyPromptContent = function() {
    if (!cachedPrompt) return;
    navigator.clipboard.writeText(cachedPrompt).then(() => {
        flashCopyButton('prompt');
    }).catch(err => console.error('Failed to copy:', err));
};

window.copyResponseContent = function() {
    if (!cachedResponse) return;
    navigator.clipboard.writeText(cachedResponse).then(() => {
        flashCopyButton('response');
    }).catch(err => console.error('Failed to copy:', err));
};

function flashCopyButton(type) {
    const viewer = document.getElementById(`${type}-viewer`);
    const panel = viewer?.closest('.panel');
    const btn = panel?.querySelector('button[onclick*="copy"]');
    if (btn) {
        const originalText = btn.textContent;
        btn.textContent = '✓ Copied!';
        btn.style.background = '#238636';
        setTimeout(() => {
            btn.textContent = originalText;
            btn.style.background = '';
        }, 1500);
    }
}

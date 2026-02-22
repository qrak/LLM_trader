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
        const timestamp = data.timestamp ? new Intl.DateTimeFormat(navigator.language, { dateStyle: 'short', timeStyle: 'short' }).format(new Date(data.timestamp)) : 'N/A';
        const source = data.source === 'disk' ? '💾 From disk' : '🧠 From memory';
        if (meta) meta.textContent = `${source} | ${timestamp}`;
        // Render prompt with markdown and discord-content styling (same as response)
        if (content && window.marked && window.DOMPurify) {
            viewer.innerHTML = DOMPurify.sanitize(marked.parse(content));
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
        const timestamp = data.timestamp ? new Intl.DateTimeFormat(navigator.language, { dateStyle: 'short', timeStyle: 'short' }).format(new Date(data.timestamp)) : 'N/A';
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
            // Sanitize output to prevent XSS (prompts/responses may contain unsanitized data)
            if (window.DOMPurify) {
                viewer.innerHTML = DOMPurify.sanitize(marked.parse(processed.trim()));
            } else {
                // Safe fallback if DOMPurify fails to load
                console.warn('DOMPurify not loaded. Rendering as safe text.');
                viewer.textContent = processed.trim();
            }
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
    // Buttons have IDs like btn-copy-prompt or btn-copy-response
    const btn = document.getElementById(`btn-copy-${type}`);
    if (btn) {
        const originalHTML = btn.innerHTML;
        const originalAria = btn.getAttribute('aria-label');

        // Use checkmark icon for feedback
        btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -2px; margin-right: 4px;"><polyline points="20 6 9 17 4 12"/></svg>Copied!`;
        btn.setAttribute('aria-label', 'Copied successfully');
        btn.style.background = '#238636';
        btn.disabled = true;

        setTimeout(() => {
            btn.innerHTML = originalHTML;
            if (originalAria) {
                btn.setAttribute('aria-label', originalAria);
            } else {
                btn.removeAttribute('aria-label');
            }
            btn.style.background = '';
            btn.disabled = false;
        }, 1500);
    }
}


import { escapeHtml } from './html_utils.js?v=1.0';
let cachedPrompt = null;
let cachedResponse = null;

export async function updateLogs() {
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
        if (content && window.marked && window.DOMPurify) {
            viewer.innerHTML = DOMPurify.sanitize(marked.parse(content));
            viewer.classList.remove('prompt-content', 'code-block');
            viewer.classList.add('console-content');
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
        if (content && window.marked) {
            let processed = content;
            processed = processed.replace(/```json[\s\S]*?```/g, '');
            processed = processed.replace(/\n*\{\s*"analysis"[\s\S]*?\n\}\s*$/g, '');
            processed = processed.replace(/⚠️\s*([^.]+\.\s*)/g, '<div class="warning-banner">⚠️ $1</div>');
            processed = processed.replace(/\n{3,}/g, '\n\n');
            if (window.DOMPurify) {
                viewer.innerHTML = DOMPurify.sanitize(marked.parse(processed.trim()));
            } else {
                console.warn('DOMPurify not loaded. Rendering as safe text.');
                viewer.textContent = processed.trim();
            }
        } else {
            if (window.DOMPurify) {
                viewer.innerHTML = DOMPurify.sanitize(`<pre style="white-space: pre-wrap; margin: 0;">${escapeHtml(content)}</pre>`);
            } else {
                viewer.textContent = content;
            }
        }
        cachedResponse = content;
    } catch (e) {
        viewer.textContent = "Error fetching response: " + e.message;
    }
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
    const btn = document.getElementById(`btn-copy-${type}`);
    if (btn) {
        const originalHTML = btn.innerHTML;
        const originalAria = btn.getAttribute('aria-label');

        btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>Copied!`;
        btn.setAttribute('aria-label', 'Copied successfully');
        btn.style.background = '#238636';
        btn.disabled = true;

        let announcer = document.getElementById('a11y-announcer');
        if (!announcer) {
            announcer = document.createElement('div');
            announcer.id = 'a11y-announcer';
            announcer.setAttribute('aria-live', 'polite');
            announcer.className = 'sr-only';
            announcer.style.cssText = 'position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0;';
            document.body.appendChild(announcer);
        }
        announcer.textContent = '';
        setTimeout(() => {
            announcer.textContent = 'Copied to clipboard';
        }, 50);

        setTimeout(() => {
            btn.innerHTML = DOMPurify.sanitize(originalHTML);
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

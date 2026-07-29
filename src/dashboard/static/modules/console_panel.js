/** Live Console panel — real-time log streaming with day-page history.
 *
 * Connects via WebSocket to /api/console/live for live updates.
 * Loads historical pages via REST /api/console/page/{n}.
 */

let ws = null;
let reconnectTimer = null;
let activePage = 'live';
let pagesLoaded = false;
let lineCount = 0;
const MAX_VISIBLE_LINES = 2000;

const container = () => document.getElementById('console-container');
const badge = () => document.getElementById('console-connection-badge');
const filter = () => document.getElementById('console-filter');
const autoscroll = () => document.getElementById('console-autoscroll');
const pageNav = () => document.getElementById('console-page-nav');

// ─── Connection ──────────────────────────────────────────────────────

function connect() {
    if (ws && ws.readyState === WebSocket.OPEN) return;

    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${protocol}//${location.host}/api/console/live`;

    try {
        ws = new WebSocket(url);
    } catch (e) {
        scheduleReconnect();
        return;
    }

    ws.onopen = () => {
        updateBadge('connected', 'Connected');
        if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
        if (!pagesLoaded) loadPages();
    };

    ws.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);
            if (msg.type === 'log') {
                appendLine(msg.line);
            }
        } catch (e) { /* ignore malformed */ }
    };

    ws.onclose = () => {
        updateBadge('disconnected', 'Disconnected');
        scheduleReconnect();
    };

    ws.onerror = () => {
        ws.close();
    };
}

function scheduleReconnect() {
    if (reconnectTimer) return;
    updateBadge('disconnected', 'Reconnecting…');
    reconnectTimer = setTimeout(() => {
        reconnectTimer = null;
        connect();
    }, 5000);
}

function updateBadge(cls, text) {
    const el = badge();
    if (!el) return;
    el.className = `lifecycle-badge ${cls}`;
    el.textContent = text;
}

// ─── Rendering ───────────────────────────────────────────────────────

function levelClass(line) {
    if (line.includes('[ERROR]') || line.includes('[CRITICAL]')) return 'lvl-error';
    if (line.includes('[WARNING]')) return 'lvl-warning';
    if (line.includes('[DEBUG]')) return 'lvl-debug';
    if (line.includes('[INFO]')) return 'lvl-info';
    return 'lvl-other';
}

function createLineElement(line) {
    const div = document.createElement('div');
    div.className = `console-line ${levelClass(line)}`;
    div.textContent = line;
    return div;
}

function appendLine(line) {
    if (activePage !== 'live') return;
    if (!filterLine(line)) return;

    const el = container();
    if (!el) return;

    const empty = el.querySelector('.empty-state');
    if (empty) empty.remove();

    el.appendChild(createLineElement(line));
    lineCount++;

    while (el.children.length > MAX_VISIBLE_LINES) {
        el.firstElementChild.remove();
    }

    if (autoscroll()?.checked) {
        el.scrollTop = el.scrollHeight;
    }
}

function filterLine(line) {
    const level = filter()?.value || 'all';
    if (level === 'all') return true;
    if (level === 'ERROR') return line.includes('[ERROR]') || line.includes('[CRITICAL]');
    if (level === 'WARNING') return line.includes('[WARNING]') || line.includes('[ERROR]') || line.includes('[CRITICAL]');
    if (level === 'INFO') return !line.includes('[DEBUG]');
    return true;
}

function applyFilter() {
    const el = container();
    if (!el) return;
    const lines = el.querySelectorAll('.console-line');
    const fv = filter()?.value || 'all';
    lines.forEach(ln => {
        const text = ln.textContent || '';
        if (fv === 'all') { ln.style.display = ''; return; }
        if (fv === 'ERROR') { ln.style.display = (text.includes('[ERROR]') || text.includes('[CRITICAL]')) ? '' : 'none'; return; }
        if (fv === 'WARNING') { ln.style.display = (text.includes('[WARNING]') || text.includes('[ERROR]') || text.includes('[CRITICAL]')) ? '' : 'none'; return; }
        if (fv === 'INFO') { ln.style.display = text.includes('[DEBUG]') ? 'none' : ''; return; }
    });
}

// ─── Page Navigation ─────────────────────────────────────────────────

async function loadPages() {
    try {
        const res = await fetch('/api/console/pages');
        const data = await res.json();
        renderPageButtons(data.pages || []);
        pagesLoaded = true;
    } catch (e) {
        console.error('Failed to load console pages', e);
    }
}

function renderPageButtons(pages) {
    const nav = pageNav();
    if (!nav) return;

    nav.querySelectorAll('.console-page-btn:not([data-page="live"])').forEach(b => b.remove());

    pages.forEach((p, i) => {
        if (i === 0) return;
        const btn = document.createElement('button');
        btn.className = 'toolbar-btn console-page-btn';
        btn.dataset.page = String(p.page);
        btn.title = p.date;
        btn.textContent = p.date.slice(5);
        btn.addEventListener('click', () => switchPage(p.page));
        nav.appendChild(btn);
    });
}

async function switchPage(page) {
    activePage = String(page);

    pageNav()?.querySelectorAll('.console-page-btn').forEach(b => {
        b.classList.toggle('active', String(b.dataset.page) === activePage);
    });

    const el = container();
    if (!el) return;
    el.innerHTML = '<div class="empty-state"><div class="spinner"></div><div class="empty-state-text">Loading…</div></div>';

    if (page === 'live') {
        connect();
        return;
    }

    if (ws) { ws.close(); ws = null; }

    try {
        const res = await fetch(`/api/console/page/${page}`);
        if (!res.ok) throw new Error('Page not found');
        const data = await res.json();
        el.innerHTML = '';
        data.lines.forEach(line => {
            if (filterLine(line)) {
                el.appendChild(createLineElement(line));
            }
        });
        el.scrollTop = el.scrollHeight;
    } catch (e) {
        el.innerHTML = `<div class="empty-state"><div class="empty-state-text">Failed to load: ${e.message}</div></div>`;
    }
}

// ─── Init ────────────────────────────────────────────────────────────

export function initConsolePanel() {
    connect();

    filter()?.addEventListener('change', () => {
        if (activePage === 'live') {
            applyFilter();
        } else {
            switchPage(activePage);
        }
    });

    document.getElementById('console-clear')?.addEventListener('click', () => {
        const el = container();
        if (el) el.innerHTML = '';
        lineCount = 0;
    });

    const observer = new MutationObserver(() => {
        const tab = document.getElementById('tab-console');
        if (tab && tab.classList.contains('active') && activePage === 'live') {
            connect();
        }
    });
    const tab = document.getElementById('tab-console');
    if (tab) {
        observer.observe(tab, { attributes: true, attributeFilter: ['class'] });
    }
}

export function updateConsoleData() {
    // No-op — console is push-based via WebSocket
}

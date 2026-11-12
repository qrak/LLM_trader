
import { escapeHtml } from './html_utils.js?v=1.0';

let ws = null;
let reconnectTimer = null;
let activePage = 'live';
let pagesLoaded = false;
let currentRange = null;
let rangeList = [];
let activeTerm = '';
const MAX_VISIBLE_LINES = 2000;

const container = () => document.getElementById('console-container');
const badge = () => document.getElementById('console-connection-badge');
const filter = () => document.getElementById('console-filter');
const search = () => document.getElementById('console-search');
const searchCount = () => document.getElementById('console-search-count');
const searchPrev = () => document.getElementById('console-search-prev');
const searchNext = () => document.getElementById('console-search-next');
const autoscroll = () => document.getElementById('console-autoscroll');
const pageNav = () => document.getElementById('console-page-nav');


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
        } catch (e) {  }
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
    if (!passesLevelFilter(line)) return;

    const el = container();
    if (!el) return;

    const empty = el.querySelector('.empty-state');
    if (empty) empty.remove();

    const element = createLineElement(line);
    el.appendChild(element);

    while (el.children.length > MAX_VISIBLE_LINES) {
        el.firstElementChild.remove();
    }

    const term = searchTerm();
    if (term) {
        addLineRanges(element, new RegExp(escapeRegExp(term), 'gi'));
        rangeList = rangeList.filter(range => range.startContainer.parentElement?.isConnected);
        updateFindState();
    } else if (autoscroll()?.checked) {
        el.scrollTop = el.scrollHeight;
    }
}

function passesLevelFilter(line) {
    const level = filter()?.value || 'all';
    if (level === 'all') return true;
    if (level === 'ERROR') return line.includes('[ERROR]') || line.includes('[CRITICAL]');
    if (level === 'WARNING') return line.includes('[WARNING]') || line.includes('[ERROR]') || line.includes('[CRITICAL]');
    if (level === 'INFO') return !line.includes('[DEBUG]');
    return true;
}

function escapeRegExp(text) {
    return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function searchTerm() {
    return (search()?.value || '').trim();
}

function textNodeOf(line) {
    const node = line.firstChild;
    return node && node.nodeType === Node.TEXT_NODE ? node : null;
}

function lineOf(range) {
    return range.startContainer.parentElement;
}

function addLineRanges(line, regex) {
    const node = textNodeOf(line);
    if (!node) return;
    for (const match of node.nodeValue.matchAll(regex)) {
        const range = document.createRange();
        range.setStart(node, match.index);
        range.setEnd(node, match.index + match[0].length);
        rangeList.push(range);
    }
}

function visibleRanges() {
    return rangeList.filter(range => {
        const line = lineOf(range);
        return line && line.isConnected && line.style.display !== 'none';
    });
}

function paintHighlights(visible) {
    if (!CSS.highlights || typeof Highlight !== 'function') return;
    const all = new Highlight();
    visible.forEach(range => {
        if (range !== currentRange) all.add(range);
    });
    CSS.highlights.set('console-find', all);
    const current = new Highlight();
    if (currentRange) current.add(currentRange);
    CSS.highlights.set('console-find-current', current);
}

function updateFindState() {
    const visible = visibleRanges();
    if (!visible.includes(currentRange)) currentRange = visible[0] || null;

    const term = searchTerm();
    const count = searchCount();
    if (count) {
        count.textContent = !term
            ? ''
            : (visible.length ? `${visible.indexOf(currentRange) + 1} / ${visible.length}` : 'No results');
    }

    const prev = searchPrev();
    const next = searchNext();
    if (prev) prev.disabled = !visible.length;
    if (next) next.disabled = !visible.length;
    paintHighlights(visible);
}

function refreshFind() {
    const wasActive = activeTerm;
    const term = searchTerm();
    activeTerm = term;
    rangeList = [];
    currentRange = null;

    const el = container();
    if (!el || !term) {
        updateFindState();
        if (!term && wasActive && activePage === 'live' && autoscroll()?.checked && el) {
            el.scrollTop = el.scrollHeight;
        }
        return;
    }

    const regex = new RegExp(escapeRegExp(term), 'gi');
    el.querySelectorAll('.console-line').forEach(line => {
        if (line.style.display !== 'none') addLineRanges(line, regex);
    });

    updateFindState();

    const line = currentRange ? lineOf(currentRange) : null;
    if (line) {
        const box = el.getBoundingClientRect();
        const rect = line.getBoundingClientRect();
        if (rect.top < box.top || rect.bottom > box.bottom) {
            line.scrollIntoView({ block: 'center' });
        }
    }
}

function stepMatch(delta) {
    const visible = visibleRanges();
    if (!visible.length) return;

    const index = currentRange ? visible.indexOf(currentRange) : -1;
    const next = index < 0
        ? (delta > 0 ? 0 : visible.length - 1)
        : (index + delta + visible.length) % visible.length;

    currentRange = visible[next];
    lineOf(currentRange).scrollIntoView({ block: 'center' });
    updateFindState();
}

function applyFilter() {
    const el = container();
    if (!el) return;
    el.querySelectorAll('.console-line').forEach(ln => {
        ln.style.display = passesLevelFilter(ln.textContent || '') ? '' : 'none';
    });
    refreshFind();
}


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
        data.lines.forEach(line => el.appendChild(createLineElement(line)));
        applyFilter();
        if (!searchTerm()) el.scrollTop = el.scrollHeight;
    } catch (e) {
        el.innerHTML = `<div class="empty-state"><div class="empty-state-text">Failed to load: ${escapeHtml(e.message)}</div></div>`;
        updateFindState();
    }
}


export function initConsolePanel() {
    connect();

    filter()?.addEventListener('change', applyFilter);
    search()?.addEventListener('input', refreshFind);
    search()?.addEventListener('search', refreshFind);
    search()?.addEventListener('keydown', (e) => {
        if (e.isComposing) return;
        if (e.key === 'Enter') {
            e.preventDefault();
            stepMatch(e.shiftKey ? -1 : 1);
        } else if (e.key === 'Escape') {
            e.preventDefault();
            if (search()) search().value = '';
            refreshFind();
        }
    });
    searchPrev()?.addEventListener('click', () => stepMatch(-1));
    searchNext()?.addEventListener('click', () => stepMatch(1));

    document.addEventListener('keydown', (e) => {
        const tab = document.getElementById('tab-console');
        if (!tab || !tab.classList.contains('active')) return;
        if ((e.ctrlKey || e.metaKey) && !e.altKey && !e.shiftKey && e.key.toLowerCase() === 'f') {
            e.preventDefault();
            search()?.focus();
            search()?.select();
            return;
        }
        if (e.key === 'F3') {
            e.preventDefault();
            stepMatch(e.shiftKey ? -1 : 1);
        }
    });

    document.getElementById('console-clear')?.addEventListener('click', () => {
        if (!window.confirm('Clear the console? Lines keep streaming in.')) return;
        const el = container();
        if (el) el.innerHTML = '';
        rangeList = [];
        currentRange = null;
        updateFindState();
    });

    updateFindState();

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

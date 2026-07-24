/**
 * Decision Pathways panel — multi-source hierarchical graph + synopsis + detail.
 * Hierarchical multi-source decision graph for Brain Activity.
 */

let network = null;
let nodesDS = null;
let edgesDS = null;
const nodePayloads = new Map();

const COLOR = {
    decision: { background: '#58a6ff', border: '#1f6feb' },
    context: { background: '#8b949e', border: '#6e7681' },
    position: { background: '#3fb950', border: '#238636' },
    position_flat: { background: '#6e7681', border: '#484f58' },
    position_short: { background: '#f85149', border: '#da3633' },
    memory_hub: { background: '#a371f7', border: '#8957e5' },
    rules_hub: { background: '#d2a8ff', border: '#a371f7' },
    journal_hub: { background: '#d4a72c', border: '#9e6a03' },
    blocked_hub: { background: '#f85149', border: '#da3633' },
    experience_win: { background: '#238636', border: '#2ea043' },
    experience_loss: { background: '#f85149', border: '#da3633' },
    experience: { background: '#58a6ff', border: '#1f6feb' },
    rule: { background: '#d2a8ff', border: '#a371f7' },
    journal: { background: '#f2cc60', border: '#d4a72c' },
    blocked: { background: '#ff7b72', border: '#f85149' },
    hub: { background: '#30363d', border: '#8b949e' },
};

function shapeFor(type) {
    switch (type) {
        case 'decision': return 'dot';
        case 'hub': return 'ellipse';
        case 'position': return 'box';
        case 'context': return 'box';
        case 'experience': return 'dot';
        case 'rule': return 'diamond';
        case 'journal': return 'box';
        case 'blocked': return 'triangle';
        default: return 'dot';
    }
}

function sizeFor(type) {
    if (type === 'decision') return 22;
    if (type === 'hub') return 18;
    if (type === 'experience') return 14;
    return 16;
}

function colorFor(type, group, data) {
    if (type === 'position') {
        if (!data || !data.has_position) return COLOR.position_flat;
        if (String(data.direction || '').toUpperCase() === 'SHORT') return COLOR.position_short;
        return COLOR.position;
    }
    if (group && COLOR[group]) return COLOR[group];
    if (type && COLOR[type]) return COLOR[type];
    return COLOR.hub;
}

function escapeHtml(text) {
    if (text === null || text === undefined) return '';
    return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function toVisNode(n) {
    const data = n.data || {};
    const rawLabel = n.label || n.id;
    // Truncate long labels to prevent overlap; full text on hover via title.
    // Leaf nodes (experience, journal, rule, blocked) get shorter labels since
    // they pack densely at the bottom level.
    const isLeaf = ['experience', 'journal', 'rule', 'blocked'].includes(n.type);
    const maxLen = isLeaf ? 18 : 28;
    const label = rawLabel.length > maxLen ? rawLabel.substring(0, maxLen - 1) + '…' : rawLabel;
    const level = n.level ?? 1;
    return {
        id: n.id,
        label: label,
        level: level,
        group: n.group || n.type,
        title: n.title || rawLabel || '',
        shape: shapeFor(n.type),
        size: sizeFor(n.type),
        color: colorFor(n.type, n.group, data),
        font: { size: 14, color: '#e6edf3', face: 'Inter, sans-serif', strokeWidth: 2, strokeColor: '#0d1117' },
        borderWidth: 2,
        margin: { top: 8, bottom: 8, left: 12, right: 12 },
        // Pre-compute positions from level to avoid relying on layout engine
        // for dynamically-added nodes (vis-network doesn't reposition them).
        x: 0,
        y: level * 160,
        raw: data,
        nodeType: n.type,
    };
}

function fitNetwork() {
    if (network && typeof network.fit === 'function') {
        network.fit({ animation: { duration: 300, easingFunction: 'easeInOutQuad' } });
    }
}

function zoomIn() {
    if (!network) return;
    const scale = network.getScale();
    network.moveTo({ scale: Math.min(scale * 1.3, 5.0) });
}

function zoomOut() {
    if (!network) return;
    const scale = network.getScale();
    network.moveTo({ scale: Math.max(scale / 1.3, 0.1) });
}

function setupZoomControls() {
    const zi = document.getElementById('zoom-in-btn');
    const zo = document.getElementById('zoom-out-btn');
    const zf = document.getElementById('zoom-fit-btn');
    if (zi) zi.addEventListener('click', zoomIn);
    if (zo) zo.addEventListener('click', zoomOut);
    if (zf) zf.addEventListener('click', fitNetwork);
}

function ensureNetwork() {
    const container = document.getElementById('decision-graph');
    if (!container) return null;

    // Never initialize vis-network when the container has zero dimensions
    // (e.g. Brain tab is hidden with display:none at page load). The canvas
    // will get 0×0 and fit() cannot recover. Defer until the tab is visible.
    const rect = container.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return null;

    if (typeof vis === 'undefined' || !vis.Network) {
        console.error('vis-network unavailable');
        return null;
    }
    if (network) return network;

    nodesDS = new vis.DataSet([]);
    edgesDS = new vis.DataSet([]);
    const options = {
        nodes: {
            font: { size: 14, color: '#e6edf3', face: 'Inter, sans-serif', strokeWidth: 2, strokeColor: '#0d1117' },
            borderWidth: 2,
            shadow: false,
            margin: 10,
        },
        edges: {
            color: { color: 'rgba(88,166,255,0.55)' },
            width: 1.5,
            smooth: { type: 'cubicBezier', forceDirection: 'vertical', roundness: 0.4 },
            arrows: { to: { enabled: true, scaleFactor: 0.45 } },
        },
        layout: {
            hierarchical: {
                enabled: true,
                direction: 'UD',
                sortMethod: 'directed',
                levelSeparation: 150,
                nodeSpacing: 180,
                treeSpacing: 200,
                blockShifting: true,
                edgeMinimization: true,
            },
        },
        physics: { enabled: false },
        interaction: {
            hover: true,
            tooltipDelay: 80,
            zoomView: true,
            dragView: true,
            selectConnectedEdges: true,
            keyboard: {
                enabled: true,
                bindToWindow: false,
                autoFocus: false,
                speed: { zoom: 0.3 },
            },
        },
    };
    network = new vis.Network(container, { nodes: nodesDS, edges: edgesDS }, options);
    window.decisionNetwork = network;
    window.fitDecisionNetwork = fitNetwork;
    window.decisionZoomIn = zoomIn;
    window.decisionZoomOut = zoomOut;
    setupZoomControls();

    network.on('selectNode', (params) => {
        const id = params.nodes && params.nodes[0];
        if (!id) return;
        const payload = nodePayloads.get(id);
        renderDetail(id, payload);
    });
    network.on('deselectNode', () => {
        renderDetailEmpty();
    });
    return network;
}

function renderDetailEmpty() {
    const detail = document.getElementById('decision-detail');
    if (!detail) return;
    detail.innerHTML = `
        <h4>Node detail</h4>
        <p class="decision-empty">Click a node to inspect source data.</p>
    `;
}

function kv(label, value) {
    if (value === null || value === undefined || value === '') return '';
    return `<div class="decision-kv"><span>${escapeHtml(label)}</span><span>${escapeHtml(String(value))}</span></div>`;
}

function renderDetail(id, payload) {
    const detail = document.getElementById('decision-detail');
    if (!detail) return;
    const type = payload?.nodeType || 'node';
    const data = payload?.raw || {};
    let body = '';

    if (type === 'decision') {
        body = [
            kv('Signal', data.action),
            kv('Confidence', data.confidence != null ? `${data.confidence}%` : null),
            kv('Trend', data.trend),
            kv('ADX', data.adx),
            kv('RSI', data.rsi),
            kv('Time', data.timestamp),
            data.reasoning_excerpt
                ? `<div class="decision-excerpt"><strong>Reasoning</strong><p>${escapeHtml(data.reasoning_excerpt)}</p></div>`
                : '',
        ].join('');
    } else if (type === 'position') {
        if (!data.has_position) {
            body = `<p class="decision-empty">No active position (flat).</p>`;
        } else {
            body = [
                kv('Direction', data.direction),
                kv('Symbol', data.symbol),
                kv('Entry', data.entry_price),
                kv('Current', data.current_price),
                kv('Stop loss', data.stop_loss),
                kv('Take profit', data.take_profit),
                kv('R:R', data.rr_ratio),
                kv('Confidence', data.confidence),
            ].join('');
        }
    } else if (type === 'experience') {
        body = [
            kv('Outcome', data.outcome),
            kv('Direction', data.direction),
            kv('P&L %', data.pnl_pct),
            kv('Similarity', data.similarity != null ? Number(data.similarity).toFixed(3) : null),
            kv('Confidence', data.confidence),
            kv('Symbol', data.symbol),
            data.document_excerpt
                ? `<div class="decision-excerpt"><strong>Context</strong><p>${escapeHtml(data.document_excerpt)}</p></div>`
                : '',
        ].join('');
    } else if (type === 'rule') {
        body = [
            kv('Type', data.rule_type),
            kv('Win rate', data.win_rate),
            kv('Score', data.final_score),
            kv('Adjustment', data.recommended_adjustment),
            data.rule_text
                ? `<div class="decision-excerpt"><strong>Rule</strong><p>${escapeHtml(data.rule_text)}</p></div>`
                : '',
        ].join('');
    } else if (type === 'journal') {
        body = [
            kv('Verdict', data.verdict),
            kv('Symbol', data.symbol),
            kv('Direction', data.direction),
            kv('P&L %', data.pnl_pct),
            kv('Close', data.close_reason),
            data.lesson_learned
                ? `<div class="decision-excerpt"><strong>Lesson</strong><p>${escapeHtml(data.lesson_learned)}</p></div>`
                : '',
        ].join('');
    } else if (type === 'context') {
        body = `<div class="decision-excerpt"><p>${escapeHtml(data.current_context || '')}</p></div>`;
    } else {
        body = Object.entries(data).slice(0, 12).map(([k, v]) => kv(k, typeof v === 'object' ? JSON.stringify(v) : v)).join('');
    }

    detail.innerHTML = DOMPurify.sanitize(`
        <h4>${escapeHtml(type)} · ${escapeHtml(id)}</h4>
        ${body || '<p class="decision-empty">No detail payload.</p>'}
    `);
}

function updateGraph(graph) {
    ensureNetwork();
    if (!nodesDS || !edgesDS || !network) return;
    nodePayloads.clear();
    const visNodes = (graph.nodes || []).map((n) => {
        const vn = toVisNode(n);
        nodePayloads.set(vn.id, { raw: vn.raw, nodeType: vn.nodeType });
        return vn;
    });
    const visEdges = (graph.edges || []).map((e) => ({
        id: e.id || `${e.from}->${e.to}`,
        from: e.from,
        to: e.to,
        arrows: 'to',
    }));
    nodesDS.clear();
    edgesDS.clear();
    nodesDS.add(visNodes);
    edgesDS.add(visEdges);
    // Spread same-level nodes horizontally — vis-network doesn't reposition
    // dynamically-added nodes with hierarchical layout + physics off.
    const levelNodes = {};
    visNodes.forEach(function (vn) {
        const lv = vn.level || 0;
        if (!levelNodes[lv]) levelNodes[lv] = [];
        levelNodes[lv].push(vn);
    });
    Object.keys(levelNodes).forEach(function (lv) {
        const group = levelNodes[lv];
        // Size row to fill available space with a generous minimum spacing
        // to prevent label overlap. Wider-than-container rows scroll naturally.
        const container = document.getElementById('decision-graph');
        const containerW = container ? container.getBoundingClientRect().width : 600;
        const minSpacing = 120;
        const desiredWidth = Math.max(containerW * 0.9, (group.length - 1) * minSpacing);
        const spacing = group.length > 1 ? desiredWidth / (group.length - 1) : 0;
        const totalWidth = (group.length - 1) * spacing;
        group.forEach(function (vn, i) {
            vn.x = i * spacing - totalWidth / 2;
        });
        nodesDS.update(group);
    });
    setTimeout(function () {
        network.fit({ animation: { duration: 200 } });
    }, 150);
}

function renderChrome(data) {
    const meta = document.getElementById('decision-meta');
    const exp = document.getElementById('experience-count');
    const generated = data.generated_at
        ? new Intl.DateTimeFormat(navigator.language, { timeStyle: 'medium' }).format(new Date(data.generated_at))
        : '';
    if (meta) {
        meta.textContent = `Memory ${data.counts?.experiences ?? 0} · Rules ${data.counts?.rules ?? 0} · Journal ${data.counts?.journal ?? 0} · Updated ${generated}`;
    }
    if (exp) exp.textContent = String(data.counts?.experiences ?? 0);

    const synopsis = document.getElementById('decision-synopsis');
    if (synopsis) {
        synopsis.innerHTML = DOMPurify.sanitize(`<p class="decision-synopsis-text">${escapeHtml(data.synopsis || 'No synopsis available.')}</p>`);
    }
}

function renderError(err) {
    const synopsis = document.getElementById('decision-synopsis');
    if (synopsis) {
        synopsis.innerHTML = DOMPurify.sanitize(`<p class="decision-empty">Failed to load decision pathways: ${escapeHtml(err.message || err)}</p>`);
    }
}

export async function initDecisionPathwaysPanel() {
    // Expose updateDecisionPathways early so tab-switch handlers can call it
    // even before the vis-network is created (deferred until tab is visible).
    window.updateDecisionPathways = updateDecisionPathways;
    ensureNetwork();
    renderDetailEmpty();
    await updateDecisionPathways();
}

export async function updateDecisionPathways() {
    const root = document.getElementById('decision-graph');
    if (!root) return;
    try {
        const res = await fetch(
            '/api/brain/decision-summary?experience_limit=5&rule_limit=5&journal_limit=5&blocked_limit=3'
        );
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        renderChrome(data);
        updateGraph(data.graph || { nodes: [], edges: [] });
    } catch (e) {
        console.error('Failed to update decision pathways', e);
        renderError(e);
    }
}

window.fitDecisionNetwork = fitNetwork;

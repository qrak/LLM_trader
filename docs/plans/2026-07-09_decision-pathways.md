# Synaptic Pathways → Decision Graph + Summary Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.
>
> **Mode:** PLAN ONLY. User will switch to a less-capable model and execute later. Do **not** implement until user says to execute. Written so a weaker model can implement without guessing.
>
> **Live reference:** https://semanticsignal.qrak.org/ → Brain Activity → Synaptic Pathways
> Current production problem: graph is ~20 unlabeled red dots + BUY/CLOSE/UPDATE legend only. Not readable. Does not surface memory, journal, position, or last decision.

**Goal:** Replace the useless trade-history action-node graph with a **new multi-source Decision Pathways graph** that is **visually readable** and merges **Vector Experience Database + Trade Journal + Active Market Position + Last Response**, plus a short server-built synopsis so the user immediately understands what is going on when opening Brain Activity.

**Architecture:**  
1. Backend `GET /api/brain/decision-summary` aggregates existing sources into: synopsis, section payloads, **and a graph model** (`nodes[]` + `edges[]`) with **labels, types, colors, tooltip text**.  
2. Frontend **keeps a graph** (vis-network retained — already loaded in dashboard; do not invent a new chart lib) but **deletes** `synapse_viewer.js` trade-history-only builder. New module builds labeled multi-type graph + detail/synopsis chrome.  
3. Old `/api/brain/memory` graph consumer is retired for this panel (endpoint may remain).

**Tech Stack:** FastAPI, `BrainRouter`, vanilla ES modules, **vis-network** (already on CDN), dashboard CSS tokens, pytest. No new frontend dependencies.

---

## Acceptance Criteria (Definition of Done)

When user opens **Brain Activity → Synaptic Pathways** (title may stay or become **Decision Pathways**):

1. **Graph is visible and readable** without hover:
   - Every node has a short **label** (not empty dots).
   - Node **color/shape encodes type** (decision / position / experience / rule / journal / blocked / context).
   - Legend explains types, not just BUY/CLOSE/UPDATE.
2. **Graph merges all required sources**, not only trade history actions:
   - Last decision / now (signal, confidence, trend)
   - Active position OR explicit flat marker
   - Similar vector experiences (top N)
   - Active semantic rules (top N)
   - Recent journal lessons (top N)
   - Optional blocked friction nodes
3. **Edges are logical**, e.g.:
   - `NOW → memory_context → experiences`
   - `NOW → rules` (constraints used advise decision)
   - `NOW → position` (if open) or `NOW → flat`
   - `experiences → journal` when same symbol/lesson linkage available, else `journal` free-floating under a Journal hub
   - No random sequential BUY→UPDATE spam chain as the main story
4. **On node click/select**, a detail panel (or title tooltip + side card) shows full fields for that node (P&L, lesson, rule_type, reasoning excerpt, similarity, SL/TP, etc.).
5. **Situation Synopsis** (2–4 sentences) sits above the graph, server-composed, no LLM call.
6. Old unlabeled trade-history-only graph code is **removed**.
7. Tests pass before production.

**Hard requirement from user:** Graph stays. UI is easy to read. Must be full summary of available decision-related tabs — not basic buy/sell dots.

**Non-goals / YAGNI:**
- Do not rewrite Memory Bank / Trade Journal / Last Response tabs.
- Do not add a second chart library (D3/ECharts/Cytoscape).
- Do not call LLM for synopsis or graph layout text.
- Do not re-introduce action-only trade_history chain as the primary visualization.

---

## Live Site Diagnosis (2026-07-09)

| Observation | Implication |
|-------------|-------------|
| Nodes are uniform red dots | Color must encode **type + outcome**, not a single default |
| No on-canvas labels | Labels ON (short) + tooltip title ON |
| Legend is only BUY/CLOSE/UPDATE | Legend must list Decision / Position / Experience / Rule / Journal / Blocked |
| ~20 “learned experiences” only from trade history actions | Wrong data root — graph must pull vectors + journal + position + last response |
| Canvas is sparse / aesthetic, not informational | Needed: hierarchical-ish layout or physics with **fixed hubs** so structure is scannable |

---

## Current Code Context (truth)

| Area | Path | Today |
|------|------|-------|
| Panel HTML | `src/dashboard/static/index.html` ~L253-264 | `#panel-synapses` + `#synapse-network` + BUY/CLOSE/UPDATE legend |
| Graph JS | `src/dashboard/static/modules/synapse_viewer.js` | `GET /api/brain/memory?limit=50` → vis nodes = trade actions only |
| Wired | `main.js` | `initSynapseNetwork`, `updateSynapses` in slow lane |
| UI tab | `ui.js` | `fitSynapseNetwork` on Brain tab |
| Fullscreen | `fullscreen.js` | `panel-synapses` → `synapse-network` |
| Backend memory | `brain.py` `get_vector_memory` | trade_history action list (basic) |
| Vectors | `get_vector_details` | experiences + `current_context` |
| Rules | `get_active_rules` | semantic rules |
| Position | `get_current_position` | open / flat |
| Journal | `get_post_mortems` | post-mortems |
| Last response | `monitor.py` `get_last_response` + `previous_response.json` | text_analysis + indicators |
| Brain status | `get_brain_status` | trend/action/confidence |
| Tests | `tests/test_dashboard_static_bindings.py`, `tests/test_dashboard_brain_router.py` | reassert IDs |

**Required merge sources:**
1. Vector Experience DB (experiences + rules + blocked)
2. Trade Journal (post-mortems)
3. Active Market Position
4. Last Response / now status

---

## Target UX

### Title
- Visible title: **Decision Pathways** (keep `id="panel-synapses"` for binding stability).
- Meta line: `Memory N · Rules N · Journal N · Updated HH:MM:SS`.

### Layout (top → bottom)

```
┌──────────────────────────────────────────────────────────────────┐
│ Decision Pathways                    meta counts + updated time   │
├───────────────────────────────────����──────────────────────────────┤
│ SYNOPSIS (server text, 2–4 sentences)                            │
├──────────────────────────────────────────────────────────────────┤
│ LEGEND: ● Decision  ■ Position  ◆ Experience  ▲ Rule  ● Journal  │
│         ⊕ Blocked   (colors listed in CSS)                       │
├──────────────────────────────┬───────────────────────────────────┤
│                              │ DETAIL CARD (selected node)       │
│   #decision-graph            │ type, key fields, excerpt         │
│   vis-network canvas         │ empty state: "Click a node"       │
│   (readable labels)          │                                   │
│                              │                                   │
└──────────────────────────────┴───────────────────────────────────┘
Visual Cortex panel remains below (unchanged).
```

Mobile (<768px): stack synopsis → graph (min-height 320px) → detail card full width.

### Graph model (must be easy to scan)

**Hub nodes (always present when data allows):**

| id | type | label example | shape | color |
|----|------|---------------|-------|-------|
| `hub_now` | decision | `HOLD 75%` | circle/dot large | cyan `#58a6ff` |
| `hub_context` | context | `BEARISH · Low ADX` (clip 28 chars) | box | `#8b949e` |
| `hub_position` | position | `FLAT` or `LONG BTC` | box | green if LONG, red if SHORT, gray if flat |
| `hub_memory` | hub | `Memory` | ellipse | `#a371f7` |
| `hub_rules` | hub | `Rules` | ellipse | `#d2a8ff` |
| `hub_journal` | hub | `Journal` | ellipse | `#f2cc60` |
| `hub_blocked` | hub | `Blocked` (omit if count=0) | ellipse | `#f85149` |

**Leaf nodes (capped):**

| type | source | count default | label |
|------|--------|---------------|-------|
| experience | vectors | ≤5 | `WIN +2.1%` / `LOSS -1.0%` + dir |
| rule | rules | ≤5 | first 24 chars of rule or type badge `ANTI` |
| journal | post-mortems | ≤5 | verdict short / symbol |
| blocked | blocked-trades | ≤3 | guard_type |

**Edges (fixed, logical — not sequential trade chain):**

```
hub_now → hub_context
hub_context → hub_memory
hub_now → hub_position
hub_now → hub_rules
hub_now → hub_journal
hub_now → hub_blocked          (if present)
hub_memory → experience_*
hub_rules → rule_*
hub_journal → journal_*
hub_blocked → blocked_*
```

Optional thin dashed edge `experience_* → journal_*` only when same symbol appears in metadata (do not force).

**Physics / layout for readability:**
- Use vis-network `hierarchical` layout **preferred** (direction `UD` or `LR`) with hubs on rows:
  - Level 0: `hub_now`
  - Level 1: context, position, memory/rules/journal/blocked hubs
  - Level 2: leaves
- If hierarchical is too rigid, alternative: physics + **fixed positions** for hubs (document coordinates in JS). Prefer hierarchical first; it is more readable for weaker implementers.
- After stabilize **once**, disable physics (`physics: false`) and call `fit()`.
- Font size ≥ 12, high-contrast labels (`#e6edf3`), node size larger for hubs.
- Canvas min-height: desktop 380px, mobile 320px.

**Interaction:**
- `network.on('selectNode', …)` → fill `#decision-detail` card from node `data` payload.
- Tooltip (`title`) shows multi-line summary always.
- Deselect → detail shows synopsis residual “Select a node for details.”

### Synopsis (deterministic string, server-side)

Same as previous plan version: compose from position + action/confidence/trend + context + top rule + top journal lesson + blocked count. No LLM.

---

## API Contract

### `GET /api/brain/decision-summary`

Query (optional):
- `experience_limit` default 5 (1–20)
- `rule_limit` default 5 (1–20)
- `journal_limit` default 5 (1–20)
- `blocked_limit` default 3 (1–20)

Response (all keys always present):

```json
{
  "generated_at": "2026-07-09T12:31:05+00:00",
  "synopsis": "No open position (flat). Latest model signal is HOLD at 75% confidence with market trend BEARISH. Memory is searching similar history under: BEARISH + Low ADX + …. Top active anti_pattern: …. Latest journal (premature_entry): …",
  "now": {
    "trend": "BEARISH",
    "action": "HOLD",
    "confidence": 75,
    "adx": 18.1,
    "rsi": 40.5,
    "exit_management": {},
    "brain_lifecycle": {"status": "idle", "message": "", "sequence": 0},
    "timestamp": "…"
  },
  "last_decision": {
    "source": "disk",
    "timestamp": "…",
    "signal": "HOLD",
    "confidence": 75,
    "reasoning_excerpt": "…",
    "text_available": true
  },
  "position": { "has_position": false, "current_price": 70000.0 },
  "memory": {
    "current_context": "…",
    "experience_count": 20,
    "rule_count": 3,
    "top_experiences": [
      {
        "id": "exp_…",
        "similarity": 0.91,
        "outcome": "WIN",
        "direction": "LONG",
        "pnl_pct": 2.1,
        "confidence": "HIGH",
        "document_excerpt": "…",
        "timestamp": "…",
        "symbol": "BTCUSDC"
      }
    ],
    "top_rules": [
      {
        "id": "rule_0",
        "rule_text": "…",
        "rule_type": "anti_pattern",
        "win_rate": 0.2,
        "final_score": 1.2,
        "recommended_adjustment": "…"
      }
    ],
    "blocked": { "blocked_count": 0, "items": [] },
    "confidence_stats": {}
  },
  "journal": {
    "count": 5,
    "items": [
      {
        "id": "pm_0",
        "verdict": "premature_entry",
        "symbol": "BTCUSDC",
        "direction": "LONG",
        "pnl_pct": -1.2,
        "close_reason": "stop_loss",
        "lesson_learned": "…",
        "created_at": "…"
      }
    ]
  },
  "counts": {
    "experiences": 20,
    "rules": 3,
    "journal": 12,
    "blocked": 0
  },
  "graph": {
    "nodes": [
      {
        "id": "hub_now",
        "type": "decision",
        "label": "HOLD 75%",
        "level": 0,
        "group": "decision",
        "title": "Latest signal HOLD · confidence 75% · trend BEARISH",
        "data": { "action": "HOLD", "confidence": 75, "trend": "BEARISH", "reasoning_excerpt": "…" }
      }
    ],
    "edges": [
      { "id": "e_now_ctx", "from": "hub_now", "to": "hub_context", "label": "" }
    ]
  }
}
```

**Server builds `graph.nodes/edges`.** Frontend must **not** invent node topology from raw arrays (only style/shape map by `type`). This keeps layout logic consistent and testable in pytest.

**Cache:** `decision_summary` TTL 15s. Invalidate in `dashboard_state.invalidate_brain_caches()` (add key + prefix).

**Errors:** never 500 user-facing empty; partial sections + empty `graph.nodes` only as last resort; always try at least `hub_now` if any status exists.

---

## Files

### Create
- `src/dashboard/static/modules/decision_pathways_panel.js` — graph + synopsis + detail
- `tests/test_decision_summary_api.py` — contract + graph topology tests

### Modify
- `src/dashboard/routers/brain.py` — route, aggregation, `_build_decision_synopsis`, `_build_decision_graph`
- `src/dashboard/dashboard_state.py` — invalidate `decision_summary`
- `src/dashboard/static/index.html` — panel chrome: synopsis, legend, graph canvas id, detail card; **keep** vis-network CDN
- `src/dashboard/static/main.js` — wire new module, remove old synapse imports
- `src/dashboard/static/modules/ui.js` — fit helper rename (`fitDecisionNetwork`)
- `src/dashboard/static/modules/fullscreen.js` — canvas id mapping
- `src/dashboard/static/css/components.css` + `panels.css` — layout/legend/detail styles
- `tests/test_dashboard_static_bindings.py`
- `CHANGELOG.md`, optional `docs/introduction.md`

### Delete
- `src/dashboard/static/modules/synapse_viewer.js` **after** all imports removed

### Leave
- Memory Bank / Trade Journal / Market Data / Last Response tabs as independent full views
- vis-network CDN script (still required)

---

## Graph builder rules (backend) — implement exactly

```python
def _short(text: str | None, n: int = 28) -> str:
    t = (text or "").strip()
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def _build_decision_graph(
    *,
    now: dict,
    last_decision: dict,
    position: dict,
    memory: dict,
    journal: dict,
) -> dict:
    nodes: list[dict] = []
    edges: list[dict] = []

    action = str(now.get("action") or last_decision.get("signal") or "--")
    conf = now.get("confidence") if now.get("confidence") not in (None, "--") else last_decision.get("confidence")
    conf_s = f"{conf}%" if conf not in (None, "--") else ""
    nodes.append({
        "id": "hub_now",
        "type": "decision",
        "label": f"{action} {conf_s}".strip(),
        "level": 0,
        "group": "decision",
        "title": f"Signal {action} | conf {conf_s} | trend {now.get('trend')}",
        "data": {
            "action": action,
            "confidence": conf,
            "trend": now.get("trend"),
            "adx": now.get("adx"),
            "rsi": now.get("rsi"),
            "reasoning_excerpt": last_decision.get("reasoning_excerpt"),
            "timestamp": last_decision.get("timestamp") or now.get("timestamp"),
        },
    })

    ctx = memory.get("current_context") or "No context"
    nodes.append({
        "id": "hub_context",
        "type": "context",
        "label": _short(ctx, 32),
        "level": 1,
        "group": "context",
        "title": ctx,
        "data": {"current_context": ctx},
    })
    edges.append({"id": "e_now_ctx", "from": "hub_now", "to": "hub_context"})

    if position.get("has_position"):
        pos_label = f"{position.get('direction')} {_short(str(position.get('symbol') or ''), 10)}"
        title = f"Open {position.get('direction')} {position.get('symbol')} entry={position.get('entry_price')}"
    else:
        pos_label = "FLAT"
        title = "No open position"
    nodes.append({
        "id": "hub_position",
        "type": "position",
        "label": pos_label,
        "level": 1,
        "group": "position",
        "title": title,
        "data": position,
    })
    edges.append({"id": "e_now_pos", "from": "hub_now", "to": "hub_position"})

    # Memory hub + experiences
    nodes.append({
        "id": "hub_memory",
        "type": "hub",
        "label": f"Memory ({memory.get('experience_count') or len(memory.get('top_experiences') or [])})",
        "level": 1,
        "group": "memory_hub",
        "title": "Similar vector experiences",
        "data": {"experience_count": memory.get("experience_count")},
    })
    edges.append({"id": "e_ctx_mem", "from": "hub_context", "to": "hub_memory"})
    for i, exp in enumerate(memory.get("top_experiences") or []):
        nid = f"exp_{i}"
        outcome = str(exp.get("outcome") or "?")
        pnl = exp.get("pnl_pct")
        pnl_s = f"{pnl:+.1f}%" if isinstance(pnl, (int, float)) else ""
        nodes.append({
            "id": nid,
            "type": "experience",
            "label": _short(f"{outcome} {pnl_s} {exp.get('direction') or ''}", 24),
            "level": 2,
            "group": "experience_win" if outcome.upper() == "WIN" else "experience_loss" if outcome.upper() == "LOSS" else "experience",
            "title": f"sim={exp.get('similarity')} | {exp.get('document_excerpt') or ''}",
            "data": exp,
        })
        edges.append({"id": f"e_mem_{i}", "from": "hub_memory", "to": nid})

    # Rules hub
    rules = memory.get("top_rules") or []
    nodes.append({
        "id": "hub_rules",
        "type": "hub",
        "label": f"Rules ({memory.get('rule_count') or len(rules)})",
        "level": 1,
        "group": "rules_hub",
        "title": "Active semantic rules",
        "data": {},
    })
    edges.append({"id": "e_now_rules", "from": "hub_now", "to": "hub_rules"})
    for i, rule in enumerate(rules):
        nid = f"rule_{i}"
        rtype = rule.get("rule_type") or "rule"
        nodes.append({
            "id": nid,
            "type": "rule",
            "label": _short(f"{rtype}: {rule.get('rule_text') or ''}", 28),
            "level": 2,
            "group": f"rule_{rtype}",
            "title": rule.get("rule_text") or "",
            "data": rule,
        })
        edges.append({"id": f"e_rule_{i}", "from": "hub_rules", "to": nid})

    # Journal hub
    items = journal.get("items") or []
    nodes.append({
        "id": "hub_journal",
        "type": "hub",
        "label": f"Journal ({journal.get('count') or len(items)})",
        "level": 1,
        "group": "journal_hub",
        "title": "Recent post-mortem lessons",
        "data": {},
    })
    edges.append({"id": "e_now_journal", "from": "hub_now", "to": "hub_journal"})
    for i, pm in enumerate(items):
        nid = f"pm_{i}"
        nodes.append({
            "id": nid,
            "type": "journal",
            "label": _short(f"{pm.get('verdict') or 'lesson'} {pm.get('symbol') or ''}", 28),
            "level": 2,
            "group": "journal",
            "title": pm.get("lesson_learned") or "",
            "data": pm,
        })
        edges.append({"id": f"e_pm_{i}", "from": "hub_journal", "to": nid})

    # Blocked hub (only if count > 0)
    blocked = (memory.get("blocked") or {})
    b_items = blocked.get("items") or []
    b_count = blocked.get("blocked_count") or len(b_items)
    if b_count:
        nodes.append({
            "id": "hub_blocked",
            "type": "hub",
            "label": f"Blocked ({b_count})",
            "level": 1,
            "group": "blocked_hub",
            "title": "System-rejected / friction events",
            "data": {"blocked_count": b_count},
        })
        edges.append({"id": "e_now_blocked", "from": "hub_now", "to": "hub_blocked"})
        for i, ev in enumerate(b_items):
            nid = f"blk_{i}"
            nodes.append({
                "id": nid,
                "type": "blocked",
                "label": _short(str(ev.get("guard_type") or ev.get("reason") or "blocked"), 24),
                "level": 2,
                "group": "blocked",
                "title": str(ev),
                "data": ev,
            })
            edges.append({"id": f"e_blk_{i}", "from": "hub_blocked", "to": nid})

    return {"nodes": nodes, "edges": edges}
```

Frontend maps `group`/`type` to vis colors/shapes **only**. Do not recompute topology.

---

## Frontend graph options (readable defaults)

```javascript
const options = {
  nodes: {
    font: { size: 13, color: '#e6edf3', face: 'Inter, sans-serif', strokeWidth: 2, strokeColor: '#0d1117' },
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
      levelSeparation: 110,
      nodeSpacing: 140,
      treeSpacing: 160,
      blockShifting: true,
      edgeMinimization: true,
    },
  },
  physics: { enabled: false }, // hierarchical already places nodes
  interaction: {
    hover: true,
    tooltipDelay: 80,
    zoomView: true,
    dragView: true,
    selectConnectedEdges: true,
  },
};

// group styling via network.setOptions groups or per-node color when converting API → vis.DataSet
```

Shape map:
- decision: `dot` size 22  
- hub: `ellipse`  
- position: `box`  
- experience: `dot` size 14 (green WIN / red LOSS)  
- rule: `diamond`  
- journal: `box`  
- blocked: `triangle`  
- context: `box`

Conversion helper:

```javascript
function toVisNode(n) {
  return {
    id: n.id,
    label: n.label,
    level: n.level,
    group: n.group,
    title: n.title,
    shape: shapeFor(n.type),
    color: colorFor(n.type, n.group, n.data),
    // stash payload
    raw: n.data,
    nodeType: n.type,
  };
}
```

---

## Bite-Sized Tasks

### Task 0: Baseline

```bash
cd /mnt/d/qrak/PythonScripts/LLM_trader_private
.venv/bin/pytest tests/test_dashboard_brain_router.py tests/test_dashboard_static_bindings.py tests/test_dashboard_server_cache.py -q
```

Record pass counts.

---

### Task 1: Failing backend tests

**Create:** `tests/test_decision_summary_api.py`

Must assert:
1. Response keys include `synopsis`, `now`, `last_decision`, `position`, `memory`, `journal`, `counts`, **`graph`**.
2. `graph.nodes` contains `hub_now`, `hub_position`, `hub_memory`, `hub_rules`, `hub_journal`.
3. Every node has non-empty `label` and `type`.
4. Edges connect hubs to leaves (at least one edge if journal items exist).
5. Flat position → a node label `FLAT` (or equivalent).
6. Missing vector_memory / post_mortem_repo → no crash; hubs still present where possible.
7. Synopsis is non-trivial string (>20 chars).

Include mock fixtures similar to the skeleton in conversation history (BrainRouter with MagicMocks). Adapt `Position` constructor with try/except TypeError → MagicMock attributes if needed.

Run:

```bash
.venv/bin/pytest tests/test_decision_summary_api.py -v
```

Expected FAIL until implementation.

Also add unit tests for pure helpers if exported:

```python
from src.dashboard.routers.brain import _build_decision_graph, _build_decision_synopsis
```

Export helpers at module level (not buried private if tests need them — module-level `_build_*` is fine).

---

### Task 2: Backend implementation

**Modify:** `brain.py`, `dashboard_state.py`

1. Register:

```python
self.router.add_api_route("/decision-summary", self.get_decision_summary, methods=["GET"])
```

2. Implement `_excerpt_text`, `_build_decision_synopsis`, `_build_decision_graph` (use code above).

3. Implement `get_decision_summary`:
   - cache key `"decision_summary"` TTL 15
   - reuse `get_current_position`, `get_active_rules`, post-mortems, vector details / experiences, blocked, brain status fields from previous_response
   - assemble sections + `graph = _build_decision_graph(...)`
   - return full dict

**Do not HTTP self-call.** Do not invent Chroma queries beyond existing helpers.

4. `invalidate_brain_caches` adds `"decision_summary"`.

5. Run tests → fix until green.

```bash
.venv/bin/pytest tests/test_decision_summary_api.py tests/test_dashboard_brain_router.py -q
```

Commit.

---

### Task 3: Failing static binding / no-old-graph tests

**Modify:** `tests/test_dashboard_static_bindings.py`

```python
def test_decision_pathways_panel_bindings():
    html = (DASHBOARD_STATIC / "index.html").read_text(encoding="utf-8")
    main_js = (DASHBOARD_STATIC / "main.js").read_text(encoding="utf-8")
    assert 'id="panel-synapses"' in html
    assert "Decision Pathways" in html or "Synaptic Pathways" in html  # prefer Decision Pathways
    assert 'id="decision-synopsis"' in html
    assert 'id="decision-graph"' in html  # NEW canvas container id (replace synapse-network)
    assert 'id="decision-detail"' in html
    assert 'id="decision-legend"' in html
    assert "decision_pathways_panel.js" in main_js
    assert "updateDecisionPathways" in main_js
    # Old module removed from app path
    assert "synapse_viewer.js" not in main_js
    assert "initSynapseNetwork" not in main_js
    # vis-network stays
    assert "vis-network" in html
```

Keep lifecycle/risk overview assertions.

---

### Task 4: HTML chrome

**Modify:** `index.html` panel block:

```html
<section id="panel-synapses" class="panel large-panel decision-pathways-panel">
  <div class="panel-header">
    <h3>Decision Pathways</h3>
    <div class="legend decision-meta-row">
      <span id="decision-meta" class="meta-info">Loading…</span>
      <span id="experience-count" class="meta-info sr-only">0</span>
    </div>
  </div>
  <div id="decision-synopsis" class="decision-synopsis" aria-live="polite">
    <span class="spinner spinner-sm"></span> Building synopsis…
  </div>
  <div id="decision-legend" class="decision-legend" aria-hidden="false">
    <span class="dl dl-decision"></span> Decision
    <span class="dl dl-position"></span> Position
    <span class="dl dl-exp"></span> Experience
    <span class="dl dl-rule"></span> Rule
    <span class="dl dl-journal"></span> Journal
    <span class="dl dl-blocked"></span> Blocked
  </div>
  <div class="decision-body">
    <div id="decision-graph" class="network-container decision-graph" role="img" aria-label="Decision pathways graph"></div>
    <aside id="decision-detail" class="decision-detail" aria-live="polite">
      <h4>Node detail</h4>
      <p class="decision-empty">Click a node to inspect source data.</p>
    </aside>
  </div>
</section>
```

- Remove old BUY/CLOSE/UPDATE-only legend.
- **Keep** vis-network script tag.
- Change canvas id from `synapse-network` → `decision-graph`.

---

### Task 5: New panel module + delete old graph

**Create:** `src/dashboard/static/modules/decision_pathways_panel.js`

Exports:
- `initDecisionPathwaysPanel()`
- `updateDecisionPathways()`
- set `window.fitDecisionNetwork` for ui/fullscreen

Behavior:
1. `fetch('/api/brain/decision-summary?...')`
2. Render synopsis + meta counts
3. Build vis.DataSet from `data.graph.nodes/edges` using shape/color maps
4. Create/rebuild network on first call; on updates use `nodes.clear(); nodes.add(...)` or `update` with stable ids
5. On selectNode → render detail card from `node.raw` / `nodeType`
6. Guard if `typeof vis === 'undefined'`

**main.js:**
```javascript
import { initDecisionPathwaysPanel, updateDecisionPathways } from './modules/decision_pathways_panel.js?v=1.0';
// remove synapse_viewer import
// replace updateSynapses() with await updateDecisionPathways()
// init: initDecisionPathwaysPanel()
```

**ui.js:** call `window.fitDecisionNetwork` on tab-brain instead of `fitSynapseNetwork`.

**fullscreen.js:** map `panel-synapses` → `decision-graph`; fit via `window.fitDecisionNetwork` / network.fit.

**Delete:** `synapse_viewer.js`

Grep required:

```bash
rg -n "synapse_viewer|initSynapseNetwork|updateSynapses|synapse-network|fitSynapseNetwork" src/dashboard tests
```

Zero hits outside CHANGELOG.

---

### Task 6: CSS readability

Add styles for:
- `.decision-synopsis`
- `.decision-legend` + color dots
- `.decision-body` grid `1.4fr 0.9fr` (stack on mobile)
- `.decision-graph` min-height 380px (320 mobile)
- `.decision-detail` card scrollable max-height

Do not leave graph as a left-aligned blob; full container width for canvas.

---

### Task 7: Tests green + cache route optional

```bash
.venv/bin/pytest tests/test_decision_summary_api.py tests/test_dashboard_static_bindings.py tests/test_dashboard_brain_router.py -v
```

If server-cache integration lists routes, add GET `/api/brain/decision-summary` smoke (optional).

Commit frontend + tests.

---

### Task 8: Pre-production verification (MANDATORY)

```bash
# Focused
.venv/bin/pytest tests/test_decision_summary_api.py tests/test_dashboard_static_bindings.py tests/test_dashboard_brain_router.py tests/test_dashboard_server_cache.py -q

# Dashboard / brain
.venv/bin/pytest tests/test_dashboard_*.py tests/test_brain_*.py -q

# Prefer full suite before prod
.venv/bin/pytest -q
```

**Manual checklist on local or staging (then prod https://semanticsignal.qrak.org/):**

- [ ] Brain Activity opens; graph visible with **readable labels**, not empty dots
- [ ] Synopsis readable above graph
- [ ] Legend matches colors
- [ ] Click HOLD/decision node → detail shows reasoning excerpt
- [ ] FLAT vs open position correct
- [ ] Memory leaves show WIN/LOSS + P&L
- [ ] Journal + Rules leaves appear when data exists (site currently has rules=3, experiences)
- [ ] Visual Cortex still works under panel
- [ ] Memory Bank + Trade Journal tabs unchanged
- [ ] Mobile densitiy OK; fit after tab switch
- [ ] Fullscreen still works
- [ ] No console errors (`vis is not defined`, failed imports)
- [ ] After deploy, hard-refresh / cache bust query params on JS

---

### Task 9: Docs

CHANGELOG + introduction.md: replace “vis.js BUY/SELL graph from trade history” with “Decision Pathways hierarchical multi-source graph (position, memory, rules, journal, last decision)”.

---

## Risks

| Risk | Mitigation |
|------|------------|
| Hierarchical overflow with many leaves | Hard caps 5/5/5/3 |
| Labels truncated unreadably | Short labels + full title tooltip + detail card |
| Weak model reintroduces action chain | Tests fail if only BUY/SELL nodes without hubs |
| vis CDN kept but logic wrong | Graph topology built server-side |
| Cache stale after close | invalidate_brain_caches includes decision_summary |
| Aesthetic-only regression | Acceptance requires labels on every node |

---

## Execution order for weaker model

1. Task 1–2 backend + graph fixture tests  
2. Task 3–5 frontend graph panel; delete old synapse viewer  
3. Task 6 CSS  
4. Task 7–8 tests + **manual graph readability check**  
5. Task 9 docs  

**Stop for user review** if focused pytest fails twice or graph still unlabeled on smoke.

---

## Done when

User opens Brain Activity on the dashboard and can **see and read** a multi-source pathways graph (not red unlabeled dots) that clearly links **last decision ↔ position ↔ memory experiences ↔ rules ↔ journal**, with synopsis and click-to-detail. Old implementation removed. Tests green before production use.

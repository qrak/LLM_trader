const FILTER_ALL = 'all';
const RELATIVE_UNITS = [
    ['year', 31536000],
    ['month', 2592000],
    ['day', 86400],
    ['hour', 3600],
    ['minute', 60]
];

const state = {
    events: [],
    filter: FILTER_ALL,
    highlight: '',
    updatedAt: null
};

export async function initActivityPanel() {
    bindFilterBar();
    await updateActivityFeed();
}

export async function updateActivityFeed() {
    const container = document.getElementById('github-activity-content');
    if (!container) return;
    try {
        const response = await fetch('/api/monitor/github_activity?limit=30');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        applyPayload(await response.json());
    } catch (error) {
        setStatus('Unavailable');
        renderMessage(container, 'GitHub activity unavailable', error.message || String(error));
    }
}

function applyPayload(data) {
    state.events = Array.isArray(data.events) ? data.events : [];
    state.highlight = data.highlight || '';
    state.updatedAt = data.fetched_at ? Date.parse(data.fetched_at) : null;
    const container = document.getElementById('github-activity-content');
    if (!container) return;
    renderFilters(data);
    renderStatus(data);
    if (data.error && state.events.length === 0) {
        renderMessage(container, 'No GitHub activity yet', data.error);
        return;
    }
    if (state.events.length === 0) {
        renderMessage(container, 'No public events', 'GitHub returned an empty event stream for this account.');
        return;
    }
    renderEvents(container, state.events, state.filter);
}

function renderFilters(data) {
    const bar = document.getElementById('github-activity-filters');
    if (!bar) return;
    const highlight = data.highlight || '';
    const options = [{ value: FILTER_ALL, label: 'All events', count: state.events.length }];
    if (highlight) {
        options.push({
            value: highlight,
            label: highlight,
            count: state.events.filter((event) => matchesFilter(event, highlight)).length
        });
    }
    bar.replaceChildren();
    options.forEach((option) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'gh-filter';
        button.dataset.filter = option.value;
        button.setAttribute('aria-pressed', String(state.filter === option.value));
        button.textContent = `${option.label} (${option.count})`;
        if (state.filter === option.value) button.classList.add('active');
        bar.appendChild(button);
    });
}

function bindFilterBar() {
    const bar = document.getElementById('github-activity-filters');
    if (!bar || bar.dataset.bound === 'true') return;
    bar.dataset.bound = 'true';
    bar.addEventListener('click', (event) => {
        const button = event.target.closest('.gh-filter');
        if (!button) return;
        state.filter = button.dataset.filter || FILTER_ALL;
        bar.querySelectorAll('.gh-filter').forEach((chip) => {
            const active = chip.dataset.filter === state.filter;
            chip.classList.toggle('active', active);
            chip.setAttribute('aria-pressed', String(active));
        });
        const container = document.getElementById('github-activity-content');
        if (container) renderEvents(container, state.events, state.filter);
    });
}

function matchesFilter(event, filter) {
    if (filter === FILTER_ALL) return true;
    const name = (event.repo || '').split('/').pop() || '';
    return name.toLowerCase() === filter.toLowerCase();
}

function renderEvents(container, events, filter) {
    const visible = events.filter((event) => matchesFilter(event, filter));
    container.replaceChildren();
    if (visible.length === 0) {
        renderMessage(container, 'Nothing matches this filter', 'Switch back to “All events” to see the whole stream.');
        return;
    }
    const list = document.createElement('div');
    list.className = 'gh-list';
    visible.forEach((event) => list.appendChild(buildRow(event)));
    container.appendChild(list);
}

function buildRow(event) {
    const row = document.createElement('article');
    row.className = 'gh-item';
    if (event.highlighted) row.classList.add('gh-item-highlighted');

    const badge = document.createElement('span');
    badge.className = `gh-badge gh-badge-${event.type || 'other'}`;
    badge.textContent = (event.label || event.type || 'event').toUpperCase();

    const body = document.createElement('div');
    body.className = 'gh-body';

    const headline = document.createElement('div');
    headline.className = 'gh-headline';
    const sentence = event.headline || '';
    const repoName = event.repo || '';
    const repoIndex = repoName ? sentence.lastIndexOf(repoName) : -1;
    if (repoIndex >= 0) {
        headline.appendChild(document.createTextNode(sentence.slice(0, repoIndex)));
        const repoLink = document.createElement('a');
        repoLink.href = event.repo_url || `https://github.com/${repoName}`;
        repoLink.target = '_blank';
        repoLink.rel = 'noopener noreferrer';
        repoLink.textContent = repoName;
        headline.appendChild(repoLink);
        headline.appendChild(document.createTextNode(sentence.slice(repoIndex + repoName.length)));
    } else {
        headline.appendChild(document.createTextNode(sentence));
    }
    body.appendChild(headline);

    if (event.detail) {
        const detail = document.createElement('div');
        detail.className = 'gh-detail';
        detail.textContent = event.detail;
        body.appendChild(detail);
    }

    const meta = document.createElement('div');
    meta.className = 'gh-meta';
    const parts = [];
    if (event.meta) parts.push(event.meta);
    if (event.created_at) parts.push(formatRelative(event.created_at));
    meta.textContent = parts.join(' · ');
    body.appendChild(meta);

    const open = document.createElement('a');
    open.className = 'gh-open';
    open.href = event.url || event.repo_url || '#';
    open.target = '_blank';
    open.rel = 'noopener noreferrer';
    open.setAttribute('aria-label', `Open ${event.repo} on GitHub`);
    open.textContent = '↗';

    row.appendChild(badge);
    row.appendChild(body);
    row.appendChild(open);
    return row;
}

function renderMessage(container, title, hint) {
    container.replaceChildren();
    const wrapper = document.createElement('div');
    wrapper.className = 'empty-state';
    const heading = document.createElement('div');
    heading.className = 'empty-state-text';
    heading.textContent = title;
    wrapper.appendChild(heading);
    if (hint) {
        const note = document.createElement('p');
        note.className = 'gh-empty-hint';
        note.textContent = hint;
        wrapper.appendChild(note);
    }
    container.appendChild(wrapper);
}

function renderStatus(data) {
    if (data.user) {
        setStatus(statusText(data));
    } else {
        setStatus('Not configured');
    }
}

function statusText(data) {
    const parts = [`@${data.user}`];
    if (state.updatedAt) parts.push(`updated ${formatRelative(new Date(state.updatedAt).toISOString())}`);
    if (data.stale) parts.push('stale copy');
    if (data.error) parts.push('GitHub API error');
    else parts.push('live from GitHub');
    return parts.join(' · ');
}

function setStatus(text) {
    const el = document.getElementById('github-activity-status');
    if (el) el.textContent = text;
}

function formatRelative(isoTimestamp) {
    const timestamp = Date.parse(isoTimestamp);
    if (Number.isNaN(timestamp)) return '';
    const seconds = Math.round((timestamp - Date.now()) / 1000);
    const formatter = new Intl.RelativeTimeFormat('en', { numeric: 'auto' });
    for (const [unit, size] of RELATIVE_UNITS) {
        if (Math.abs(seconds) >= size) {
            return formatter.format(Math.round(seconds / size), unit);
        }
    }
    return formatter.format(seconds, 'second');
}

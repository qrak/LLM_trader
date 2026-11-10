const state = {
    bound: false,
    local: false,
    docs: [],
    view: null,
    config: null,
    originals: new Map(),
    dirty: new Map(),
    keys: null,
    saving: false,
    settingIndex: 0,
    statusTimer: null,
    docsCache: new Map(),
};

document.addEventListener('DOMContentLoaded', init);

async function init() {
    bindNav();
    try {
        const status = await api('/api/settings/status');
        state.docs = Array.isArray(status.docs) ? status.docs : [];
        renderAccess(Boolean(status.local));
        renderDocsNav();
        route();
    } catch (error) {
        renderAccess(false);
        renderDocsNav();
        setView('docs');
        const content = document.getElementById('docs-content');
        content.replaceChildren(errorBlock(error.message, init));
        showStatus(`Could not load the settings API: ${error.message}`, 'error');
    }
}

function bindNav() {
    if (state.bound) {
        return;
    }
    state.bound = true;
    document.getElementById('nav-settings').addEventListener('click', () => {
        window.location.hash = 'settings';
    });
    document.getElementById('nav-keys').addEventListener('click', () => {
        window.location.hash = 'keys';
    });
    document.getElementById('nav-docs').addEventListener('click', () => {
        const slug = state.docs.length ? state.docs[0].slug : 'docs';
        window.location.hash = `docs/${slug}`;
    });
    document.getElementById('settings-search').addEventListener('input', filterSettings);
    document.getElementById('settings-save').addEventListener('click', saveSettings);
    document.getElementById('settings-discard').addEventListener('click', discardSettings);
    document.getElementById('keys-create').addEventListener('click', createKeys);
    window.addEventListener('hashchange', route);
    window.addEventListener('beforeunload', (event) => {
        if (state.dirty.size > 0) {
            event.preventDefault();
            event.returnValue = '';
        }
    });
}

async function api(path, options = {}) {
    const headers = {
        Accept: 'application/json',
        ...(options.body ? { 'Content-Type': 'application/json' } : {}),
        ...(options.headers || {}),
    };
    const response = await fetch(path, {
        ...options,
        headers,
        credentials: 'same-origin',
    });
    let payload = null;
    try {
        payload = await response.json();
    } catch (error) {
        payload = null;
    }
    if (!response.ok) {
        const problem = new Error(errorText(payload, response.status));
        problem.payload = payload;
        throw problem;
    }
    return payload;
}

function errorText(payload, status) {
    if (payload && typeof payload.error === 'string') {
        return payload.error;
    }
    if (payload && typeof payload.detail === 'string') {
        return payload.detail;
    }
    if (payload && Array.isArray(payload.detail)) {
        return payload.detail
            .map((item) => item && item.msg ? item.msg : '')
            .filter(Boolean)
            .join('; ') || `HTTP ${status}`;
    }
    return `HTTP ${status}`;
}

function node(tag, className, text) {
    const element = document.createElement(tag);
    if (className) {
        element.className = className;
    }
    if (text !== undefined) {
        element.textContent = text;
    }
    return element;
}

function stateKey(section, key) {
    return `${section}\u0000${key}`;
}

function truthy(value) {
    return ['true', 'yes', 'on', '1'].includes(String(value).trim().toLowerCase());
}

function renderAccess(local) {
    state.local = local;
    const pill = document.getElementById('settings-access');
    pill.hidden = false;
    pill.textContent = local ? 'LOCAL · settings enabled' : 'REMOTE · documentation only';
    pill.classList.toggle('local', local);
    document.getElementById('nav-settings').hidden = !local;
    document.getElementById('nav-keys').hidden = !local;
}

function route() {
    const hash = window.location.hash.replace(/^#/, '');
    let view = 'settings';
    let slug = null;
    if (hash.startsWith('docs/')) {
        view = 'docs';
        slug = hash.slice(5);
    } else if (hash === 'keys' && state.local) {
        view = 'keys';
    } else if (hash === 'settings' && state.local) {
        view = 'settings';
    } else if (!state.local) {
        view = 'docs';
    }
    setView(view);
    if (view === 'settings') {
        loadSettings();
    } else if (view === 'keys') {
        loadKeys();
    } else {
        loadDoc(slug || 'docs');
    }
}

function setView(view) {
    state.view = view;
    for (const name of ['settings', 'keys', 'docs']) {
        document.getElementById(`view-${name}`).hidden = name !== view;
        const button = document.getElementById(`nav-${name}`);
        button.classList.toggle('active', name === view);
        button.setAttribute('aria-selected', name === view ? 'true' : 'false');
    }
    document.getElementById('settings-sections-group').hidden = view !== 'settings';
    document.getElementById('docs-group').hidden = view !== 'docs';
    updateSavebar();
}

function showStatus(message, kind = 'info', actionLabel = '', action = null) {
    if (state.statusTimer) {
        window.clearTimeout(state.statusTimer);
        state.statusTimer = null;
    }
    const box = document.getElementById('settings-status');
    box.className = `settings-status settings-status-${kind}`;
    box.replaceChildren(node('span', '', message));
    if (actionLabel && action) {
        const button = node('button', 'settings-inline-btn', actionLabel);
        button.type = 'button';
        button.addEventListener('click', action);
        box.append(button);
    }
    box.hidden = false;
    if (kind === 'success') {
        state.statusTimer = window.setTimeout(() => {
            box.hidden = true;
            state.statusTimer = null;
        }, 8000);
    }
}

function errorBlock(message, retry) {
    const block = node('div', 'settings-error-block');
    block.append(node('p', 'settings-error-text', message));
    const button = node('button', 'settings-btn', 'Retry');
    button.type = 'button';
    button.addEventListener('click', retry);
    block.append(button);
    return block;
}

function markRowError(section, key, message) {
    const row = rowFor(section, key);
    if (!row) {
        return;
    }
    row.classList.add('settings-row-error');
    let error = row.querySelector('.settings-error-text');
    if (!error) {
        error = node('p', 'settings-error-text');
        row.append(error);
    }
    error.textContent = message;
}

function markPayloadError(error) {
    const payload = error && error.payload;
    if (!payload || !payload.section || !payload.key) {
        return;
    }
    markRowError(payload.section, payload.key, error.message);
}

function rowFor(section, key) {
    return Array.from(document.querySelectorAll('#settings-sections .settings-row'))
        .find((row) => row.dataset.section === section && row.dataset.key === key) || null;
}

async function loadSettings() {
    if (state.config && document.querySelector('#settings-sections .settings-section')) {
        return;
    }
    const container = document.getElementById('settings-sections');
    container.replaceChildren(node('p', 'settings-muted', 'Loading config.ini…'));
    try {
        const config = await api('/api/settings/config');
        state.config = config;
        renderSettings(config);
    } catch (error) {
        container.replaceChildren(errorBlock(error.message, () => {
            state.config = null;
            loadSettings();
        }));
    }
}

function renderSettings(config) {
    const container = document.getElementById('settings-sections');
    const sectionNav = document.getElementById('settings-section-nav');
    container.replaceChildren();
    sectionNav.replaceChildren();
    state.originals.clear();
    state.dirty.clear();
    state.settingIndex = 0;
    if (!config || !Array.isArray(config.sections) || config.sections.length === 0) {
        container.append(node('p', 'settings-muted', 'No settings found in config/config.ini.'));
        updateSavebar();
        return;
    }
    let total = 0;
    for (const section of config.sections) {
        const sectionElement = node('section', 'settings-section');
        sectionElement.id = `settings-section-${section.name}`;
        sectionElement.append(node('h2', '', section.title));
        const rows = node('div', 'settings-rows');
        for (const setting of section.keys) {
            rows.append(createRow(section, setting));
            total += 1;
        }
        sectionElement.append(rows);
        container.append(sectionElement);
        const navButton = node('button', 'settings-section-btn', `${section.title} (${section.keys.length})`);
        navButton.type = 'button';
        navButton.addEventListener('click', () => {
            const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
            sectionElement.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' });
        });
        sectionNav.append(navButton);
    }
    document.getElementById('settings-count').textContent = `${total} settings`;
    updateSavebar();
    filterSettings();
}

function createRow(section, setting) {
    const row = node('div', 'settings-row');
    row.dataset.section = section.name;
    row.dataset.key = setting.key;
    row.dataset.type = setting.type;
    const index = ++state.settingIndex;
    const inputId = `setting-${index}`;
    const helpId = `help-${index}`;
    const label = node('label', 'settings-label');
    label.htmlFor = inputId;
    label.append(node('code', '', setting.key));
    const badges = node('span', 'settings-badges');
    if (setting.required) {
        badges.append(node('span', 'settings-badge required', 'required'));
    }
    badges.append(node('span', 'settings-badge', 'restart'));
    label.append(badges);
    const control = node('div', 'settings-control');
    const input = createInput(setting, inputId, helpId);
    control.append(input);
    const help = node('div', 'settings-help');
    help.id = helpId;
    if (setting.help) {
        if (setting.help.length > 220) {
            const details = node('details', 'settings-help-more');
            details.append(node('summary', '', 'About this setting'));
            details.append(node('p', 'settings-help-text', setting.help));
            help.append(details);
        } else {
            help.append(node('p', 'settings-help-text', setting.help));
        }
    }
    if (setting.inline_help) {
        help.append(node('p', 'settings-inline-help', setting.inline_help));
    }
    if (Array.isArray(setting.options) && setting.options.length) {
        help.append(node('p', 'settings-inline-help', `Allowed: ${setting.options.join(', ')}`));
    }
    row.append(label, control, help);
    const original = setting.type === 'bool'
        ? (truthy(setting.value) ? 'true' : 'false')
        : setting.value;
    state.originals.set(stateKey(section.name, setting.key), {
        section: section.name,
        key: setting.key,
        type: setting.type,
        value: original,
    });
    input.dataset.original = original;
    input.addEventListener('input', () => markDirty(row, section, setting, input));
    input.addEventListener('change', () => markDirty(row, section, setting, input));
    return row;
}

function createInput(setting, inputId, helpId) {
    let input;
    if (setting.type === 'bool') {
        input = document.createElement('input');
        input.type = 'checkbox';
        input.checked = truthy(setting.value);
    } else if (setting.type === 'enum') {
        input = document.createElement('select');
        const options = Array.isArray(setting.options) ? setting.options : [];
        const values = options.includes(setting.value) ? options : [...options, setting.value];
        for (const option of values) {
            const element = document.createElement('option');
            element.value = option;
            element.textContent = options.includes(option) ? option : `${option} (current)`;
            input.append(element);
        }
        input.value = setting.value;
    } else {
        input = document.createElement('input');
        input.type = 'text';
        input.value = setting.value;
        input.spellcheck = false;
        input.autocomplete = 'off';
        if (setting.type === 'int') {
            input.inputMode = 'numeric';
        } else if (setting.type === 'float') {
            input.inputMode = 'decimal';
        }
    }
    input.id = inputId;
    input.setAttribute('aria-describedby', helpId);
    if (setting.required) {
        input.setAttribute('aria-required', 'true');
    }
    return input;
}

function markDirty(row, section, setting, input) {
    const key = stateKey(section.name, setting.key);
    const original = state.originals.get(key);
    if (!original) {
        return;
    }
    const current = input.type === 'checkbox'
        ? (input.checked ? 'true' : 'false')
        : input.value;
    row.classList.remove('settings-row-error');
    const error = row.querySelector('.settings-error-text');
    if (error) {
        error.remove();
    }
    if (current === original.value) {
        state.dirty.delete(key);
    } else {
        state.dirty.set(key, {
            section: section.name,
            key: setting.key,
            value: current,
        });
    }
    updateSavebar();
}

function updateSavebar() {
    const count = state.dirty.size;
    const bar = document.getElementById('settings-savebar');
    document.getElementById('settings-dirty').textContent =
        `${count} unsaved change${count === 1 ? '' : 's'}`;
    bar.hidden = state.view !== 'settings' || count === 0;
    document.getElementById('settings-save').disabled = count === 0 || state.saving;
    document.getElementById('settings-discard').disabled = count === 0 || state.saving;
}

async function saveSettings() {
    if (state.saving || state.dirty.size === 0 || !state.config) {
        return;
    }
    state.saving = true;
    updateSavebar();
    const updates = Array.from(state.dirty.values());
    try {
        const result = await api('/api/settings/config', {
            method: 'POST',
            body: JSON.stringify({ etag: state.config.etag, updates }),
        });
        state.config.etag = result.etag;
        applyUpdated(result.updated || []);
        state.dirty.clear();
        updateSavebar();
        showStatus('Saved. Restart the bot (Shift+R in the bot console) to apply.', 'success');
    } catch (error) {
        markPayloadError(error);
        if (error.message.toLowerCase().includes('changed on disk')) {
            showStatus(error.message, 'error', 'Reload', () => {
                state.config = null;
                loadSettings();
            });
        } else {
            showStatus(error.message, 'error');
        }
    } finally {
        state.saving = false;
        updateSavebar();
    }
}

function applyUpdated(updated) {
    if (!state.config) {
        return;
    }
    const changes = new Map(updated.map((item) => [stateKey(item.section, item.key), item]));
    for (const [id, item] of changes) {
        const original = state.originals.get(id);
        if (original) {
            original.value = item.value;
        }
        const row = rowFor(item.section, item.key);
        if (!row) {
            continue;
        }
        const input = row.querySelector('input, select');
        if (!input) {
            continue;
        }
        if (input.type === 'checkbox') {
            input.checked = item.value === 'true';
        } else {
            input.value = item.value;
        }
        input.dataset.original = item.value;
        row.classList.remove('settings-row-error');
        const error = row.querySelector('.settings-error-text');
        if (error) {
            error.remove();
        }
    }
    for (const section of state.config.sections) {
        for (const setting of section.keys) {
            const change = changes.get(stateKey(section.name, setting.key));
            if (change) {
                setting.value = change.value;
            }
        }
    }
}

function discardSettings() {
    if (state.dirty.size === 0) {
        return;
    }
    state.dirty.clear();
    resetRows();
    updateSavebar();
    showStatus('Changes discarded.', 'info');
}

function resetRows() {
    for (const row of document.querySelectorAll('#settings-sections .settings-row')) {
        const original = state.originals.get(stateKey(row.dataset.section, row.dataset.key));
        if (!original) {
            continue;
        }
        const input = row.querySelector('input, select');
        if (!input) {
            continue;
        }
        if (input.type === 'checkbox') {
            input.checked = original.value === 'true';
        } else {
            input.value = original.value;
        }
        input.dataset.original = original.value;
        row.classList.remove('settings-row-error');
        const error = row.querySelector('.settings-error-text');
        if (error) {
            error.remove();
        }
    }
}

function filterSettings() {
    const query = document.getElementById('settings-search').value.trim().toLowerCase();
    let visible = 0;
    for (const section of document.querySelectorAll('#settings-sections .settings-section')) {
        let sectionVisible = 0;
        for (const row of section.querySelectorAll('.settings-row')) {
            const show = !query || `${row.dataset.key} ${row.textContent}`.toLowerCase().includes(query);
            row.hidden = !show;
            if (show) {
                sectionVisible += 1;
            }
        }
        section.hidden = sectionVisible === 0;
        visible += sectionVisible;
    }
    document.getElementById('settings-count').textContent =
        `${visible} setting${visible === 1 ? '' : 's'}`;
}

async function loadKeys(force = false) {
    if (!force && state.keys && document.querySelector('#keys-list .keys-card')) {
        return;
    }
    const list = document.getElementById('keys-list');
    list.replaceChildren(node('p', 'settings-muted', 'Loading keys.env…'));
    try {
        const data = await api('/api/settings/keys');
        renderKeys(data);
    } catch (error) {
        list.replaceChildren(errorBlock(error.message, () => loadKeys(true)));
    }
}

function renderKeys(data) {
    state.keys = data;
    const list = document.getElementById('keys-list');
    const banner = document.getElementById('keys-banner');
    const create = document.getElementById('keys-create');
    list.replaceChildren();
    if (!data.exists) {
        banner.hidden = false;
        banner.textContent = 'keys.env does not exist yet. Create it from the template, then set the keys you need.';
        create.hidden = false;
    } else {
        banner.hidden = true;
        create.hidden = true;
    }
    if (!Array.isArray(data.keys) || data.keys.length === 0) {
        list.append(node('p', 'settings-muted', 'No keys found.'));
        return;
    }
    for (const entry of data.keys) {
        list.append(createKeyCard(entry, Boolean(data.exists)));
    }
}

function keyStateLabel(entry) {
    if (entry.source === 'both') {
        return 'set (OS env wins)';
    }
    if (entry.source === 'environment') {
        return entry.set ? 'set (OS env)' : 'unset (empty OS env)';
    }
    if (entry.source === 'file') {
        return 'set (keys.env)';
    }
    return 'not set';
}

function createKeyCard(entry, enabled) {
    const card = node('article', 'keys-card');
    card.dataset.key = entry.key;
    const head = node('div', 'keys-head');
    head.append(node('code', '', entry.key));
    const pill = node('span', 'settings-pill', keyStateLabel(entry));
    pill.classList.toggle('local', Boolean(entry.set));
    head.append(pill);
    card.append(head);
    if (entry.help) {
        card.append(node('p', 'keys-help', entry.help));
    }
    if (entry.inline_help) {
        card.append(node('p', 'settings-inline-help', entry.inline_help));
    }
    if (entry.source === 'environment' || entry.source === 'both' || entry.empty_override) {
        card.append(node('p', 'keys-warn',
            'The OS environment sets this key and wins over keys.env. Editing the file will not change what the bot reads until that environment variable is removed.'));
    }
    const actions = node('div', 'keys-actions');
    const input = document.createElement('input');
    input.type = 'password';
    input.autocomplete = 'new-password';
    input.spellcheck = false;
    input.placeholder = enabled ? 'Paste a new value' : 'Create keys.env first';
    input.setAttribute('aria-label', `New value for ${entry.key}`);
    input.disabled = !enabled;
    const setButton = node('button', 'settings-btn primary', 'Set');
    setButton.type = 'button';
    setButton.disabled = !enabled;
    const clearButton = node('button', 'settings-btn', 'Clear');
    clearButton.type = 'button';
    clearButton.disabled = !enabled;
    setButton.addEventListener('click', () => {
        const value = input.value.trim();
        if (!value) {
            showStatus('Enter a value first, or use Clear to unset the key.', 'error');
            input.focus();
            return;
        }
        saveKey(entry.key, value, input, card);
    });
    clearButton.addEventListener('click', () => {
        if (!window.confirm(`Clear ${entry.key}? The running bot keeps the old value until restart.`)) {
            return;
        }
        saveKey(entry.key, '', input, card);
    });
    actions.append(input, setButton, clearButton);
    card.append(actions);
    return card;
}

async function saveKey(key, value, input, card) {
    if (state.saving || !state.keys) {
        return;
    }
    state.saving = true;
    const buttons = Array.from(card.querySelectorAll('button'));
    for (const button of buttons) {
        button.disabled = true;
    }
    try {
        const result = await api('/api/settings/keys', {
            method: 'POST',
            body: JSON.stringify({ etag: state.keys.etag, key, value }),
        });
        state.keys.etag = result.etag;
        const entry = state.keys.keys.find((item) => item.key === key);
        if (entry) {
            entry.set = result.set;
            entry.source = result.source;
        }
        if (input) {
            input.value = '';
        }
        renderKeys(state.keys);
        showStatus(`Saved ${key}. Restart the bot to apply.`, 'success');
    } catch (error) {
        if (error.message.toLowerCase().includes('changed on disk')) {
            showStatus(error.message, 'error', 'Reload', () => loadKeys(true));
        } else {
            showStatus(error.message, 'error');
        }
    } finally {
        state.saving = false;
        for (const button of buttons) {
            button.disabled = false;
        }
    }
}

async function createKeys() {
    const button = document.getElementById('keys-create');
    button.disabled = true;
    try {
        const result = await api('/api/settings/keys/init', { method: 'POST' });
        showStatus(result.created ? 'keys.env created from the template.' : 'keys.env already exists.', 'success');
        await loadKeys(true);
    } catch (error) {
        showStatus(error.message, 'error');
    } finally {
        button.disabled = false;
    }
}

function renderDocsNav() {
    const nav = document.getElementById('docs-nav');
    nav.replaceChildren();
    for (const item of state.docs) {
        const button = node('button', 'settings-section-btn', item.title);
        button.type = 'button';
        button.dataset.doc = item.slug;
        button.addEventListener('click', () => {
            window.location.hash = `docs/${item.slug}`;
        });
        nav.append(button);
    }
}

async function loadDoc(slug) {
    const title = document.getElementById('docs-title');
    const content = document.getElementById('docs-content');
    const source = document.getElementById('docs-source');
    title.textContent = 'Documentation';
    source.hidden = true;
    content.replaceChildren(node('p', 'settings-muted', 'Loading…'));
    try {
        let documentData = state.docsCache.get(slug);
        if (!documentData) {
            documentData = await api(`/api/settings/docs/${encodeURIComponent(slug)}`);
            state.docsCache.set(slug, documentData);
        }
        title.textContent = documentData.title;
        source.href = documentData.source_url;
        source.hidden = false;
        if (!window.DOMPurify) {
            throw new Error('Documentation renderer unavailable (DOMPurify failed to load).');
        }
        content.innerHTML = window.DOMPurify.sanitize(documentData.html);
        for (const button of document.querySelectorAll('#docs-nav .settings-section-btn')) {
            button.classList.toggle('active', button.dataset.doc === slug);
        }
        const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        window.scrollTo({ top: 0, behavior: reduce ? 'auto' : 'smooth' });
    } catch (error) {
        content.replaceChildren(errorBlock(error.message, () => loadDoc(slug)));
    }
}


let ws = null;
let reconnectAttempts = 0;
let hasConnectedBefore = false;
const reconnectDelay = 3000;
const maxReconnectDelay = 30000;

export function initWebSocket() {
    connect();
}

function connect() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${location.host}/ws`);
    ws.onopen = () => {
        const isReconnect = hasConnectedBefore;
        hasConnectedBefore = true;
        reconnectAttempts = 0;
        updateConnectionStatus('connected');
        fetchInitialCountdown();
        if (isReconnect) {
            triggerRefreshAll();
        }
        console.log('WebSocket connected');
    };
    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleUpdate(data);
        } catch (e) {
            console.error('WebSocket message parse error:', e);
        }
    };
    ws.onclose = () => {
        updateConnectionStatus('disconnected');
        scheduleReconnect();
    };
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

function scheduleReconnect() {
    reconnectAttempts++;
    const delay = Math.min(reconnectDelay * reconnectAttempts, maxReconnectDelay);
    setTimeout(connect, delay);
}

function handleUpdate(data) {
    switch (data.type) {
        case 'countdown':
            updateCountdownFromWS(data);
            break;
        case 'position':
            updatePositionFromWS(data);
            break;
        case 'analysis_complete':
            triggerRefreshAll();
            break;
        case 'brain_rebuild_started':
        case 'brain_rebuild_completed':
        case 'brain_rebuild_failed':
            dispatchBrainLifecycle(data.data);
            break;
        case 'brain_state_updated':
            dispatchBrainStateUpdated(data.data);
            break;
        default:
            console.log('Unknown WS message type:', data.type);
    }
}

function updateCountdownFromWS(data) {
    if (data.next_check_utc) {
        window._nextCheckUTC = new Date(data.next_check_utc);
    }
}

function updatePositionFromWS(data) {
    const event = new CustomEvent('position-update', { detail: data.data });
    document.dispatchEvent(event);
}

function triggerRefreshAll() {
    const event = new CustomEvent('analysis-complete');
    document.dispatchEvent(event);
}

function dispatchBrainLifecycle(lifecycle) {
    const event = new CustomEvent('brain-lifecycle-update', { detail: lifecycle || {} });
    document.dispatchEvent(event);
}

function dispatchBrainStateUpdated(data) {
    const event = new CustomEvent('brain-state-updated', { detail: data || {} });
    document.dispatchEvent(event);
}

function updateConnectionStatus(status) {
    const indicator = document.getElementById('connection-status');
    const statusDot = document.querySelector('.status-dot');
    if (!indicator) return;

    if (status === 'connected') {
        indicator.textContent = 'Connected';
        indicator.classList.remove('status-text', 'disconnected');
        indicator.classList.add('status-text', 'connected');
        indicator.style.color = '';
        if (statusDot) {
            statusDot.classList.remove('disconnected');
            statusDot.classList.add('connected');
        }
    } else {
        indicator.textContent = 'Disconnected';
        indicator.classList.remove('status-text', 'connected');
        indicator.classList.add('status-text', 'disconnected');
        indicator.style.color = '';
        if (statusDot) {
            statusDot.classList.remove('connected');
            statusDot.classList.add('disconnected');
        }
    }
}

export function formatDuration(seconds) {
    if (seconds < 0) seconds = 0;
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    if (hours > 0) return `${hours}h ${minutes}m`;
    if (minutes > 0) return `${minutes}m ${secs}s`;
    return `${secs}s`;
}

export function startCountdownLoop() {
    fetchInitialCountdown();
    setInterval(updateCountdownDisplay, 1000);
}

async function fetchInitialCountdown() {
    try {
        const response = await fetch('/api/status/countdown');
        const data = await response.json();
        if (data.next_check_utc) {
            window._nextCheckUTC = new Date(data.next_check_utc);
        }
    } catch (e) {
        console.warn('Could not fetch initial countdown:', e);
    }
}

function updateCountdownDisplay() {
    const element = document.getElementById('next-analysis');
    if (!element) return;
    if (!window._nextCheckUTC) {
        element.textContent = '--:--';
        return;
    }
    const now = new Date();
    const remaining = (window._nextCheckUTC.getTime() - now.getTime()) / 1000;
    if (remaining <= 0) {
        element.textContent = 'soon...';
    } else {
        element.textContent = formatDuration(remaining);
    }
}

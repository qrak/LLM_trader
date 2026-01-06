/**
 * WebSocket client for real-time dashboard updates.
 */

let ws = null;
let reconnectAttempts = 0;
const maxReconnectAttempts = 10;
const reconnectDelay = 3000;

/**
 * Initialize WebSocket connection.
 */
export function initWebSocket() {
    connect();
}

/**
 * Connect to WebSocket server.
 */
function connect() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${location.host}/ws`);
    ws.onopen = () => {
        reconnectAttempts = 0;
        updateConnectionStatus('connected');
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

/**
 * Schedule a reconnection attempt.
 */
function scheduleReconnect() {
    if (reconnectAttempts >= maxReconnectAttempts) {
        console.warn('Max WebSocket reconnection attempts reached');
        return;
    }
    reconnectAttempts++;
    const delay = reconnectDelay * Math.min(reconnectAttempts, 5);
    setTimeout(connect, delay);
}

/**
 * Handle incoming WebSocket updates.
 */
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
        default:
            console.log('Unknown WS message type:', data.type);
    }
}

/**
 * Update countdown display from WebSocket data.
 */
function updateCountdownFromWS(data) {
    if (data.next_check_utc) {
        window._nextCheckUTC = new Date(data.next_check_utc);
    }
}

/**
 * Update position from WebSocket data.
 */
function updatePositionFromWS(data) {
    const event = new CustomEvent('position-update', { detail: data.data });
    document.dispatchEvent(event);
}

/**
 * Trigger a full refresh of all panels.
 */
function triggerRefreshAll() {
    const event = new CustomEvent('analysis-complete');
    document.dispatchEvent(event);
}

/**
 * Update connection status indicator.
 */
function updateConnectionStatus(status) {
    const indicator = document.getElementById('connection-status');
    if (!indicator) return;
    if (status === 'connected') {
        indicator.innerHTML = '🟢 Connected';
        indicator.className = 'connected';
    } else {
        indicator.innerHTML = '🔴 Disconnected';
        indicator.className = 'disconnected';
    }
}

/**
 * Format duration to human-readable string (e.g., "2h 35m").
 */
export function formatDuration(seconds) {
    if (seconds < 0) seconds = 0;
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    if (hours > 0) return `${hours}h ${minutes}m`;
    if (minutes > 0) return `${minutes}m ${secs}s`;
    return `${secs}s`;
}

/**
 * Start countdown timer loop.
 */
export function startCountdownLoop() {
    fetchInitialCountdown();
    setInterval(updateCountdownDisplay, 1000);
}

/**
 * Fetch initial countdown from REST API.
 */
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

/**
 * Update countdown display every second.
 */
function updateCountdownDisplay() {
    const element = document.getElementById('next-analysis');
    if (!element) return;
    if (!window._nextCheckUTC) {
        element.textContent = 'Next analysis: --';
        return;
    }
    const now = new Date();
    const remaining = (window._nextCheckUTC.getTime() - now.getTime()) / 1000;
    if (remaining <= 0) {
        element.textContent = 'Next analysis: soon...';
    } else {
        element.textContent = `Next analysis: ${formatDuration(remaining)}`;
    }
}

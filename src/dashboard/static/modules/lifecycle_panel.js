import { getLastPosition } from './position_panel.js?v=4.13';

function formatAge(seconds) {
    if (seconds === null || seconds === undefined) return '';
    const total = Math.max(0, Math.round(seconds));
    if (total < 60) return `${total}s ago`;
    const minutes = Math.floor(total / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 48) return `${hours}h ago`;
    return `${Math.floor(hours / 24)}d ago`;
}

function setChip(id, text, severity, title) {
    const element = document.getElementById(id);
    if (!element) return;
    if (element.textContent !== text) element.textContent = text;
    const nextClass = `state-chip state-${severity || 'neutral'}`;
    if (element.className !== nextClass) element.className = nextClass;
    const nextTitle = title || '';
    if (element.title !== nextTitle) element.title = nextTitle;
}

export function initLifecyclePanel() {
    const strip = document.getElementById('state-strip');
    if (!strip) return;
    updateLifecyclePanel();
}

export function updateLifecyclePanel() {
    const strip = document.getElementById('state-strip');
    if (!strip) return;
    const data = getLastPosition();
    const lifecycle = data && data.lifecycle;
    if (!lifecycle || lifecycle.available === false) {
        strip.hidden = true;
        return;
    }
    strip.hidden = false;
    const cycle = lifecycle.cycle || {};
    const position = lifecycle.position || {};
    const proof = position.verification || {};
    const approval = lifecycle.approval || {};
    const cycleAge = formatAge(cycle.age_seconds);
    setChip(
        'state-cycle',
        `Cycle: ${cycle.label || 'Idle'}${cycleAge ? ` · ${cycleAge}` : ''}`,
        cycle.severity,
        cycle.detail || ''
    );
    setChip('state-position', `Position: ${position.label || 'Flat'}`, position.severity, position.symbol || '');
    setChip('state-proof', `Proof: ${proof.label || 'Unverified'}`, proof.severity, proof.detail || '');
    setChip(
        'state-approval',
        `Approval: ${approval.label || 'None'}`,
        approval.severity,
        approval.oldest_expires_at || ''
    );
}

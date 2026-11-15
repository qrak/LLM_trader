import { escapeHtml } from './html_utils.js?v=1.0';

const RULE_TYPES = {
    anti_pattern: { label: 'Avoid', className: 'rule-tag-avoid' },
    corrective: { label: 'Improve', className: 'rule-tag-improve' },
    ai_mistake: { label: 'AI mistake', className: 'rule-tag-mistake' },
    best_practice: { label: 'Best practice', className: 'rule-tag-practice' },
};

export async function initRulesPanel() {
    window.updateRulesPanel = updateRulesPanel;
    await updateRulesPanel();
}

export async function updateRulesPanel() {
    const container = document.getElementById('rules-content');
    if (!container) return;
    try {
        const response = await fetch('/api/brain/rules');
        const rules = response.ok ? await response.json() : [];
        renderRules(container, Array.isArray(rules) ? rules : []);
    } catch (error) {
        console.error('Failed to load learned rules', error);
        container.innerHTML = DOMPurify.sanitize(
            `<p class="chain-empty">Learned rules could not be loaded right now.</p>`
        );
        setMeta('');
    }
}

function setMeta(text) {
    const meta = document.getElementById('rules-meta');
    if (meta) meta.textContent = text;
}

function formatNumber(value, digits = 1) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed.toFixed(digits) : null;
}

function detailBlock(label, value) {
    if (!value) return '';
    return `
        <div class="rule-detail">
            <span class="rule-detail-label">${escapeHtml(label)}</span>${escapeHtml(value)}
        </div>
    `;
}

function ruleCard(rule) {
    const type = RULE_TYPES[rule.rule_type] || RULE_TYPES.best_practice;
    const winRate = formatNumber(rule.win_rate, 0);
    const averagePnl = formatNumber(rule.avg_pnl_pct, 1);
    const profitFactor = formatNumber(rule.profit_factor, 1);
    const sourceTrades = rule.wins !== undefined && rule.losses !== undefined
        ? `${rule.wins}W / ${rule.losses}L`
        : `from ${rule.source_trades || '--'} trades`;
    const exitProfile = rule.dominant_exit_profile
        ? `<span>${escapeHtml(rule.dominant_exit_profile)}</span>` : '';
    const mistakeType = rule.mistake_type
        ? `<span>${escapeHtml(String(rule.mistake_type).replace(/_/g, ' '))}</span>` : '';
    const pnlClass = averagePnl !== null && Number(averagePnl) < 0 ? 'rule-meta-negative' : '';
    return `
        <article class="rule-card">
            <div class="rule-head">
                <span class="rule-tag ${type.className}">${escapeHtml(type.label)}</span>
                <span class="rule-source">${escapeHtml(sourceTrades)}</span>
            </div>
            <p class="rule-text">${escapeHtml(rule.rule_text || '')}</p>
            ${detailBlock('Why', rule.failure_reason)}
            ${detailBlock('Fix', rule.recommended_adjustment)}
            <div class="rule-meta">
                <span class="rule-win-rate">${winRate !== null ? `${winRate}% win rate` : 'win rate --'}</span>
                ${averagePnl !== null ? `<span class="${pnlClass}">${averagePnl >= 0 ? '+' : ''}${averagePnl}% avg P&amp;L</span>` : ''}
                ${profitFactor !== null && Number(profitFactor) < 90 ? `<span>PF ${escapeHtml(profitFactor)}</span>` : ''}
                ${exitProfile}
                ${mistakeType}
            </div>
        </article>
    `;
}

function renderRules(container, rules) {
    if (!rules.length) {
        setMeta('0 active');
        container.innerHTML = DOMPurify.sanitize(`
            <p class="chain-empty">No rules learned yet. Rules appear once enough closed trades share a
                pattern worth repeating or avoiding.</p>
        `);
        return;
    }
    setMeta(`${rules.length} active`);
    container.innerHTML = DOMPurify.sanitize(
        `<div class="rules-grid">${rules.map(ruleCard).join('')}</div>`
    );
}

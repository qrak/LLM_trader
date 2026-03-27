/**
 * News panel module - Displays latest crypto news articles.
 */

export async function initNewsPanel() {
    await updateNewsData();
}

export async function updateNewsData() {
    const container = document.getElementById('news-content');
    if (!container) return;
    try {
        const response = await fetch('/api/monitor/news');
        const data = await response.json();
        if (!data.articles || data.articles.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon" aria-hidden="true"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M4 22h16a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2H8a2 2 0 0 0-2 2v16a2 2 0 0 1-2 2Zm0 0a2 2 0 0 1-2-2v-9c0-1.1.9-2 2-2h2"/><path d="M18 14h-8"/><path d="M15 18h-5"/><path d="M10 6h8v4h-8V6Z"/></svg></div>
                    <p class="empty-state-text">No news articles available</p>
                    <p style="font-size: 0.85em; margin-top: 8px; color: var(--text-muted);">
                        News will appear after the next analysis cycle.
                    </p>
                </div>
            `;
            return;
        }
        container.innerHTML = renderNews(data.articles);
    } catch (e) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon" aria-hidden="true"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg></div>
                <p class="empty-state-text">Error loading news</p>
                <p style="font-size: 0.85em; margin-top: 8px; color: var(--accent-danger);"></p>
            </div>
        `;
        container.querySelector('p:last-child').textContent = e.message;
    }
}

function formatNewsDate(timestamp) {
    if (!timestamp) return 'Unknown';
    const date = new Date(timestamp * 1000);
    const now = new Date();
    const diffMs = now - date;
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffMins = Math.floor(diffMs / (1000 * 60));
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return new Intl.DateTimeFormat(navigator.language, { dateStyle: 'short' }).format(date);
}

function renderNews(articles) {
    return `
        <div class="news-grid">
            ${articles.slice(0, 30).map(article => `
                <div class="news-card">
                    <div class="news-header">
                        <div class="news-title">
                            <a href="${escapeHtml(sanitizeUrl(article.url || article.guid))}" target="_blank" rel="noopener">
                                ${escapeHtml(article.title || 'Untitled')}
                            </a>
                        </div>
                        <span class="news-date">${formatNewsDate(article.published_on || article.publishedOn)}</span>
                    </div>
                    <div class="news-body">
                        ${escapeHtml(truncateText(article.body || article.summary || '', 300))}
                    </div>
                    ${renderTopics(article)}
                    <div class="news-source">
                        Source: ${escapeHtml(article.source_info?.name || article.source || 'Unknown')}
                        ${article.detected_coins ? ` • Coins: ${article.detected_coins.map(c => escapeHtml(c)).join(', ')}` : ''}
                    </div>
                </div>
            `).join('')}
        </div>
    `;
}

function renderTopics(article) {
    const categories = article.categories ? article.categories.split('|') : [];
    const tags = article.tags ? article.tags.split('|') : [];
    const topics = [...new Set([...categories, ...tags])].slice(0, 5);
    if (topics.length === 0) return '';
    return `
        <div class="news-topics">
            ${topics.map(topic => `<span class="news-topic">${escapeHtml(topic)}</span>`).join('')}
        </div>
    `;
}

function truncateText(text, maxLength) {
    if (!text || text.length <= maxLength) return text;
    return text.substring(0, maxLength).trim() + '...';
}

function sanitizeUrl(url) {
    if (!url) return '#';
    const cleanUrl = url.trim();
    // Only allow http and https protocols to prevent XSS (javascript:, data:, etc.)
    if (/^https?:\/\//i.test(cleanUrl)) {
        return cleanUrl;
    }
    return '#';
}

function escapeHtml(text) {
    if (!text) return '';
    return String(text)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

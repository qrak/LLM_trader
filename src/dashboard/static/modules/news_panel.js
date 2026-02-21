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
            container.innerHTML = '<div class="empty-state">No news articles available. News will appear after the next analysis cycle.</div>';
            return;
        }
        container.innerHTML = renderNews(data.articles);
    } catch (e) {
        container.innerHTML = `<div class="empty-state">Error loading news: ${escapeHtml(e.message)}</div>`;
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
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

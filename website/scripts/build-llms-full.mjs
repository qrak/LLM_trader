/**
 * Build llms-full.txt from the generated site.
 *
 * Why: LLM tools and crawlers get a single plain-text document containing every
 * page, instead of having to strip markup from ten HTML files. It is generated
 * from `dist/` during the build so it can never drift from what is deployed.
 *
 * Usage: node scripts/build-llms-full.mjs   (wired into `npm run build`)
 */
import { readFileSync, writeFileSync, readdirSync, statSync } from 'node:fs';
import { join, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const dist = join(root, 'dist');

/** Page order and titles for the document. */
const ORDER = [
	['index.html', 'https://qrak.org/', 'Home'],
	['story/index.html', 'https://qrak.org/story/', 'The full development story & technology'],
	['articles/index.html', 'https://qrak.org/articles/', 'Articles'],
	['articles/chart-vision/index.html', 'https://qrak.org/articles/chart-vision/', 'How the bot reads charts'],
	['articles/vector-memory/index.html', 'https://qrak.org/articles/vector-memory/', 'Vector memory'],
	['articles/executor-separation/index.html', 'https://qrak.org/articles/executor-separation/', 'From paper to a live executor'],
	['about/index.html', 'https://qrak.org/about/', 'About the developer'],
	['contact/index.html', 'https://qrak.org/contact/', 'Contact'],
	['disclaimer/index.html', 'https://qrak.org/disclaimer/', 'Risk disclaimer'],
	['privacy-policy/index.html', 'https://qrak.org/privacy-policy/', 'Privacy policy'],
];

const ENTITIES = {
	'&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"', '&#39;': "'",
	'&nbsp;': ' ', '&mdash;': '—', '&ndash;': '–', '&hellip;': '…',
	'&times;': '×', '&minus;': '−', '&middle;': '·', '&#8203;': '',
};

function textOf(html) {
	return html
		.replace(/<script[\s\S]*?<\/script>/gi, '')
		.replace(/<style[\s\S]*?<\/style>/gi, '')
		.replace(/<svg[\s\S]*?<\/svg>/gi, '')
		.replace(/<\/(p|div|li|h[1-6]|tr|pre|section|article|dl|dt|dd)>/gi, '\n')
		.replace(/<br\s*\/?>/gi, '\n')
		.replace(/<li[^>]*>/gi, '  - ')
		.replace(/<[^>]+>/g, '')
		.replace(/&[a-z#0-9]+;/gi, (m) => (m in ENTITIES ? ENTITIES[m] : ' '))
		.split('\n')
		.map((line) => line.replace(/[ \t]+/g, ' ').trim())
		.filter((line, i, all) => line !== '' || all[i - 1] !== '')
		.join('\n')
		.replace(/\n{3,}/g, '\n\n')
		.trim();
}

function contentOf(file) {
	const html = readFileSync(file, 'utf8');
	const main = html.match(/<main[\s\S]*?<\/main>/i);
	const body = main ? main[0] : html;
	return textOf(body);
}

const missing = [];
const parts = [];
for (const [file, url, title] of ORDER) {
	const path = join(dist, file);
	try {
		statSync(path);
	} catch {
		missing.push(file);
		continue;
	}
	parts.push(`\n\n${'='.repeat(78)}\n## ${title}\nURL: ${url}\n${'='.repeat(78)}\n\n${contentOf(path)}\n`);
}

const header = `# Semantic Signal (qrak.org) — full site text

Source: https://qrak.org | Generated from the deployed build; every page of the
site as plain text, for LLM tools and crawlers.

Fact sheet in JSON: https://qrak.org/project.json
Short index: https://qrak.org/llms.txt
Source code: https://github.com/qrak/LLM_trader

Status carried with every page: paper trading only (simulated capital), executor
on an exchange testnet, profitability unproven, not financial advice.
`;

writeFileSync(join(dist, 'llms-full.txt'), header + parts.join(''), 'utf8');

const rel = relative(root, join(dist, 'llms-full.txt')).split(sep).join('/');
const size = statSync(join(dist, 'llms-full.txt')).size;
console.log(`[llms-full] ${rel} — ${parts.length} pages, ${(size / 1024).toFixed(1)} KB`);
if (missing.length) {
	console.warn(`[llms-full] WARNING: pages listed but not built: ${missing.join(', ')}`);
}

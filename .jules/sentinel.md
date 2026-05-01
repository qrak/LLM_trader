## 2025-03-07 - [DOM-based XSS in Error Rendering]
**Vulnerability:** In `src/dashboard/static/modules/log_viewer.js`, error messages caught from `fetch` responses were rendered using `viewer.innerHTML = "Error: " + e.message;`, opening the door for DOM-based XSS if the error message is manipulable or reflects an unsanitized API response.
**Learning:** Even internal API endpoints or general fetch errors can construct DOM payloads if dynamically inserted directly into HTML. Frontend logic in this repository is purely static JS (no framework protections), demanding explicit manual safety when injecting strings into the DOM.
**Prevention:** Use `textContent` instead of `innerHTML` when directly rendering variables, untrusted data, or error messages into the DOM to enforce plain-text interpretation and mitigate XSS.

## 2025-03-13 - [DOM-based XSS in Vector Panel UI Injection]
**Vulnerability:** In `src/dashboard/static/modules/vector_panel.js`, the `data.current_context` property (which could contain unvalidated or unsanitized data from the backend) was being injected directly into the DOM via `container.innerHTML` without being escaped.
**Learning:** The frontend dashboard lacks an automatic escaping mechanism for dynamically built HTML strings since it relies on Vanilla JS template literals. Variables passed from the backend API, even internally generated ones, must be assumed unsafe when generating raw HTML.
**Prevention:** Always use the local `escapeHtml()` utility function to sanitize dynamically injected variables before embedding them in an `innerHTML` template string.

## 2025-03-14 - [DOM-based XSS in Statistics Panel UI Injection]
**Vulnerability:** In `src/dashboard/static/modules/statistics_panel.js`, the `stats` properties fetched from the `/api/performance/stats` endpoint were being directly injected into the DOM via `container.innerHTML` without sanitization.
**Learning:** The frontend dashboard lacks an automatic escaping mechanism for dynamically built HTML strings since it relies on Vanilla JS template literals. Variables passed from the backend API must be assumed unsafe when generating raw HTML.
**Prevention:** Always use the local `escapeHtml()` utility function to sanitize dynamically injected variables before embedding them in an `innerHTML` template string.

## 2025-03-15 - [DOM-based XSS via escapeHtml Implementation]
**Vulnerability:** In `src/dashboard/static/modules/performance_chart.js`, the `escapeHtml` function was using `div.textContent = text; return div.innerHTML;` to escape user input. This implementation can lead to mutation XSS or trigger DOM-based vulnerabilities under certain circumstances (especially when injecting into attributes or scripts), compared to a plain-text regex replace.
**Learning:** Using DOM manipulation (`innerHTML`) to sanitize or escape data before injecting it back into another `innerHTML` or attribute can introduce subtle XSS vectors, such as Mutation XSS (mXSS).
**Prevention:** Always use regex-based string replacements for `escapeHtml` implementations, replacing `&`, `<`, `>`, `"`, and `'` with their respective HTML entities to ensure safe injection without triggering DOM parsing during the sanitization phase itself.

## 2024-05-18 - [Fix Path Traversal in RagFileHandler]
**Vulnerability:** Methods `load_json_file` and `save_json_file` in `src/rag/file_handler.py` loaded/saved files using unvalidated string paths without checking boundaries.
**Learning:** Functions accepting unvalidated input for file paths present a critical directory traversal vulnerability.
**Prevention:** Resolved both the requested path and the base data directory using `os.path.realpath` to resolve symlinks and checked `os.path.commonpath([abs_path, abs_base_dir]) == abs_base_dir` before reading or writing.
## 2025-03-02 - Defense-in-Depth against Clickjacking
**Vulnerability:** The dashboard UI lacked the `frame-ancestors 'none';` directive in its Content-Security-Policy header.
**Learning:** While the dashboard already implements `X-Frame-Options: DENY`, `X-Frame-Options` is deprecated in modern browsers in favor of the CSP `frame-ancestors` directive. Without `frame-ancestors`, certain legacy or misconfigured browser setups might ignore `X-Frame-Options`, potentially leaving the application susceptible to clickjacking.
**Prevention:** Added `frame-ancestors 'none';` directly to the `Content-Security-Policy` header in `src/dashboard/server.py` to ensure robust, modern defense-in-depth against clickjacking across all browsers.

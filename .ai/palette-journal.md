# Palette 🎨 UX & Accessibility Journal

## 2026-07-26 - Screen Reader Emoji Hiding in Admin Navigation
**Learning:** Screen readers announced raw Unicode emojis (`📊`, `🎮`, `⚙️`, `📜`, `🚪`) phonetically on every sidebar navigation focus.
**Action:** Wrapped decorative emojis in `<span aria-hidden="true">` elements across sidebar navigation links and logout button in `src/dashboard/static/admin/index.html`. Screen readers now cleanly read button labels without emoji distraction.

## 2026-07-26 - Admin Login Error Live Region & Double Click Guard
**Learning:** Screen readers did not announce authentication failure messages when `#login-error` became visible. In addition, rapid double-clicks on the submit button could dispatch duplicate login requests.
**Action:** Added `role="alert"` and `aria-live="assertive"` to `showLoginError()` and disabled `#login-form` submit button during network authentication.

## 2026-07-26 - Decision Pathways Structured Synopsis Cards & Pills
**Learning:** Dense unformatted text paragraphs in the Decision Pathways header reduced readability for multi-source situational summary metrics.
**Action:** Replaced plain paragraph rendering with a structured card layout: position state badge (`FLAT`/`LONG`/`SHORT`), action/confidence pill (`HOLD 72%`), trend badge (`BEARISH`), monospace context tags, border-accented rule/journal blocks, and friction warning callouts.

## 2026-07-27 - start.py Terminal Startup UX Rework & Progress Stream Synchronization
**Learning:** Raw `tqdm` progress bars from `SentenceTransformer` and dual `RichHandler` console instances broke Rich panel layout and caused line fragmentation during startup.
**Action:** Unified `Console` stream across `CompositionRoot` and `Logger`, enabled top-level Windows console UTF-8 stream reconfiguration, set `TQDM_DISABLE=1` during startup model load, integrated maintenance tasks into Stage 8 before summary table rendering, and removed redundant keyboard command log statements. Startup output is now visually clean and unfragmented.




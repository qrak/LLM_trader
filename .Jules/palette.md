## 2024-05-15 - Interactive custom elements and skip link targets
**Learning:** Using `span` or `div` as interactive elements (like tabs or buttons) instead of native `<button>` or `<a>` requires explicitly handling `Enter` and `Space` key events for keyboard accessibility, even if arrow keys are handled. Additionally, for a "Skip to main content" link to correctly move keyboard focus to a non-interactive element like `<main>`, that target element must have `tabindex="-1"`.
**Action:** When building custom interactive components or skip links, ensure `Enter`/`Space` are mapped to click events, and apply `tabindex="-1"` to skip link targets.

## 2024-05-18 - Missing State Semantics on UI Toggles
**Learning:** Found an accessibility pattern where UI toggle elements (like mobile menu buttons and panel minimizers) had `aria-label` but lacked `aria-expanded` attributes. This prevents screen readers from understanding the current state (open/closed) of the controlled content.
**Action:** Always pair `aria-label` with `aria-expanded` (and ideally `aria-controls`) on interactive elements that toggle visibility of other content. When the state changes, dynamically update the `aria-expanded` value via JavaScript and update `aria-label` to describe the *next* action.

## 2024-05-20 - Fullscreen Modals and aria-haspopup
**Learning:** Found that buttons opening modal dialogs (like fullscreen panels) incorrectly used `aria-expanded`. `aria-expanded` is meant for inline collapsible content. If the button itself becomes hidden inside an `aria-hidden="true"` container when the modal opens, updating its `aria-expanded` state provides no value to screen readers.
**Action:** Buttons that open modal dialogs must use `aria-haspopup="dialog"` and `aria-controls="[modal-id]"`. Avoid `aria-expanded` for these types of interactions.

## 2024-05-22 - Consistent Iconography vs Unicode Symbols
**Learning:** Using raw Unicode characters like `−`, `+`, `⛶`, `✕`, `🔍+` inside UI controls leads to inconsistent visual rendering across different operating systems and browsers. Furthermore, they lack the visual polish and distinct design characteristics of dedicated SVG icons.
**Action:** Always prefer consistent inline SVGs over text-based icons for UI controls. Additionally, ensure dynamically created modal trigger buttons properly wire `aria-haspopup="dialog"` and `aria-controls="[modal-id]"` for correct screen reader announcement.

## 2024-05-25 - Dynamic KPI Values and aria-live
**Learning:** In a live updating dashboard, screen readers do not automatically announce changes to text nodes unless instructed. Found a KPI metric (Session Cost) missing `aria-live="polite"`, causing updates to be silent, unlike sibling metrics.
**Action:** For dynamically updating dashboard metrics (such as KPI card values), ensure the target HTML element includes the `aria-live="polite"` attribute so screen readers announce asynchronous updates naturally without interrupting the user.

## 2025-02-28 - [Loading States for Refresh Buttons]
**Learning:** Icon-only refresh buttons often lack visual feedback during async operations, leading to user uncertainty ("Did it work?").
**Action:** When implementing refresh actions, always pass the button element to the handler and toggle a loading state/animation until the promise resolves.

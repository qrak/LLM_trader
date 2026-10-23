# Architecture Decision Records

Short records of decisions that shaped this repo. Format: context → decision →
alternatives (with the reason they lost) → consequences → how it was verified.

| ADR | Decision | Status |
|-----|----------|--------|
| [0001](0001-durable-execution-ledger.md) | Durable execution ledger in the executor, intent journal in the bot | accepted |
| [0002](0002-untrusted-context-boundary.md) | External text is sanitized data, not instructions | accepted |
| [0003](0003-cycle-checkpoints.md) | One persisted cycle checkpoint; duplicate suppression by decision hash | accepted |
| [0004](0004-approval-gate.md) | Human approval gate is opt-in and expires into a refusal | accepted |
| [0005](0005-real-price-exit-ownership.md) | The real price owns the exit; the exchange leg is a catastrophe backstop | accepted |

Add a new ADR for any change that alters where state lives, who may execute an order, or
what evidence a lesson may rest on. Supersede rather than rewrite.

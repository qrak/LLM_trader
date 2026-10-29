## What changed

## Change type

- [ ] Feature
- [ ] Fix
- [ ] Refactor / retirement (add the `refactor` label)
- [ ] Docs / tooling

## Deletion evidence (refactor / retirement only)

- [ ] The superseded symbols, their tests, doubles and fixtures are deleted in this pull request.
- [ ] `python scripts/check_dead_symbols.py` reports `0 dead, 0 test-only, N allowlisted`.
- [ ] `python scripts/check_net_diff.py --base origin/master --head HEAD --require-negative` reports `Net: -N lines` for the production scope.
- [ ] No parallel implementation, deprecation warning, `_v2` alias or compatibility shim was added beside the old code.

## Validation

Commands run and their results:

## Checklist

- [ ] No secret, API key or local runtime data is included.
- [ ] The documentation or changelog is updated where behaviour, configuration or a gate changed.
- [ ] The new tests extend an existing domain module.

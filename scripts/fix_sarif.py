#!/usr/bin/env python3
import json, sys

with open('results.sarif') as f:
    sarif = json.load(f)

fixed = 0
for run in sarif.get('runs', []):
    # Fix rules not being array
    driver = run.get('tool', {}).get('driver', {})
    rules = driver.get('rules')
    if rules is not None and not isinstance(rules, list):
        print(f"  Fixing rules type={type(rules).__name__} value={repr(rules)[:200]}")
        driver['rules'] = []
        fixed += 1
    elif isinstance(rules, list):
        print(f"  rules is already a list with {len(rules)} items")

    for result in run.get('results', []):
        for loc in result.get('locations', []):
            region = loc.get('physicalLocation', {}).get('region', {})
            if region.get('startLine', 1) < 1:
                region['startLine'] = 1
                fixed += 1
            if region.get('startColumn', 1) < 1:
                region['startColumn'] = 1
                fixed += 1
        if result.get('level') not in (None, 'none', 'note', 'warning', 'error'):
            print(f"  Fixing level: {result.get('level')}")
            result['level'] = 'warning'
            fixed += 1

with open('results.sarif', 'w') as f:
    json.dump(sarif, f)

print(f'Total: Fixed {fixed} SARIF issues')

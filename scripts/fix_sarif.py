#!/usr/bin/env python3
import json, sys

with open('results.sarif') as f:
    sarif = json.load(f)

fixed = 0
for i, run in enumerate(sarif.get('runs', [])):
    # Ensure runs have a results array
    if 'results' not in run or run['results'] is None:
        run['results'] = []
        fixed += 1
        print(f"  Run[{i}]: added missing results array")

    # Fix rules not being array
    driver = run.get('tool', {}).get('driver', {})
    rules = driver.get('rules')
    if rules is not None and not isinstance(rules, list):
        print(f"  Run[{i}]: Fixing rules type={type(rules).__name__}")
        driver['rules'] = []
        fixed += 1
    elif 'rules' not in driver or rules is None:
        if run.get('results'):
            driver['rules'] = []
            fixed += 1
        else:
            pass  # no results, no rules needed

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
    json.dump(sarif, f, indent=None, separators=(',', ':'))

print(f'Total: Fixed {fixed} SARIF issues')

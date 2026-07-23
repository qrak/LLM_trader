#!/usr/bin/env python3
import json, sys

with open('results.sarif') as f:
    sarif = json.load(f)

fixed = 0
for run in sarif.get('runs', []):
    for result in run.get('results', []):
        for loc in result.get('locations', []):
            region = loc.get('physicalLocation', {}).get('region', {})
            if region.get('startLine', 1) < 1:
                region['startLine'] = 1
                fixed += 1
        if result.get('level') not in (None, 'none', 'note', 'warning', 'error'):
            result['level'] = 'warning'
            fixed += 1

with open('results.sarif', 'w') as f:
    json.dump(sarif, f)

print(f'Fixed {fixed} SARIF issues')

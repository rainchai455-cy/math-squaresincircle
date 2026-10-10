"""Drift check: re-hash every file in a frozen directory against its MANIFEST."""
import hashlib
import json
import os
import sys

d = sys.argv[1] if len(sys.argv) > 1 else 'g2_frozen_narrow'
raw = json.load(open(os.path.join(d, 'MANIFEST.json'), encoding='utf-8'))
files = raw['files'] if isinstance(raw, dict) and 'files' in raw else raw
if isinstance(files, dict):
    items = [(k, v['sha256'] if isinstance(v, dict) else v) for k, v in files.items()]
else:
    items = [(e['file'], e['sha256']) if isinstance(e, dict) else (e[0], e[1]) for e in files]

drift = 0
for name, h in items:
    p = os.path.join(d, name)
    if not os.path.exists(p):
        print(f'MISSING    {name}')
        drift += 1
        continue
    digest = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    ok = digest == h or digest.startswith(str(h)[:16])
    print(('UNCHANGED  ' if ok else 'DRIFT      ') + f'{name} {digest[:16]}')
    drift += 0 if ok else 1
print(f'[DRIFT] {d}: files={len(list(items))} drift={drift}')

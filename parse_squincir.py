"""squincir_raw.html -> data/squincir_n1_50.csv + audit vs the table in fit scripts.

HTML layout: each <table> shows THREE pictures (n, n+1, n+2) and then ONE row holding
the THREE captions in the same left-to-right order. So a "next caption after this
picture" search returns the WRONG row whenever n is not the first of its triple.
Correct pairing: collect picture markers and captions in document order, zip by index.

Printed caption forms (all must parse):
    r = 1 / &radic;2 = .70710+<br>Trivial.          (leading-dot decimal)
    r = 5 &radic;17 / 16 = 1.28847+<br>Proved by..  (closed form + decimal)
    r = 1.97877+<br>Found by ..                    (decimal only)
    r = 1.688+<sup>*</sup><br>Proved by ..         (exact value is in the footnote)
    r = 5/2 = 2.5<br>Found by ..                   (no trailing +)
"""
import hashlib
import io
import math
import os
import re

SRC = 'https://erich-friedman.github.io/packing/squincir/'
RAW = 'squincir_raw.html'
html = io.open(RAW, 'r', encoding='utf-8', errors='replace').read()
sha_html = hashlib.sha256(open(RAW, 'rb').read()).hexdigest()

mark_re = re.compile(r'>\s*(\d+)\.\s*<br>\s*<img src="(\d+)\.gif"', re.I)
cap_re = re.compile(r'<font size=\+1>r\s*=\s*(.*?)<br>\s*(Trivial|Proved|Found)',
                    re.I | re.S)
ENT = [('&radic;', 'sqrt'), ('&ndash;', '-'), ('&nbsp;', ' '), ('&times;', '*')]


def strip_tags(s):
    s = re.sub(r'<[^>]+>', ' ', s)
    for k, v in ENT:
        s = s.replace(k, v)
    return re.sub(r'\s+', ' ', s).strip()


marks = [int(m.group(1)) for m in mark_re.finditer(html)]
caps = [(strip_tags(m.group(1)), m.group(2)) for m in cap_re.finditer(html)]
print('picture markers: %d (n=%d..%d, contiguous=%s)' %
      (len(marks), min(marks), max(marks), marks == list(range(1, len(marks) + 1))))
print('caption matches  : %d' % len(caps))
assert len(marks) == len(caps), 'cannot zip 1:1'

recs = {}
for n, (body, status) in zip(marks, caps):
    tail = re.search(r'(\.?\d+(?:\.\d+)?)\s*\+?\s*$', body)
    if tail:
        val = tail.group(1)
    else:
        nums = re.findall(r'\.?\d+(?:\.\d+)?', body)
        val = nums[-1] if nums else None
    assert val, (n, body)
    cut = body.rfind('=')
    closed = body[:cut].strip() if cut > 0 else '-'
    recs[n] = dict(r=val, closed=closed or '-', status=status,
                   truncated=body.rstrip().endswith('+'))

claim = re.search(r'Only the cases n=([0-9\-]+) have been proved optimal', html, re.I)
print('page statement: "%s"' % claim.group(0))

src_txt = io.open('fit_4param_stability.py', 'r', encoding='utf-8').read()
blk = re.search(r'\bR\s*=\s*\{(.*?)\}', src_txt, re.S)
local = {int(k): float(v) for k, v in re.findall(r'(\d+):\s*([0-9.]+)', blk.group(1))}

print('\n%-4s %-10s %-10s %-11s %-5s %-3s %s' %
      ('n', 'page', 'local', 'local-page', 'trunc', 'dp', 'closed form'))
over, below, coarse = [], [], []
for n in range(1, 51):
    page = float(recs[n]['r'])
    d = local[n] - page
    dp = len(recs[n]['r'].split('.')[1]) if '.' in recs[n]['r'] else 0
    if d < -1e-12:
        below.append((n, page, local[n], d))
    if d > 1e-4:
        over.append((n, page, local[n], d))
    if dp < 5:
        coarse.append((n, recs[n]['r']))
    print('%-4d %-10s %-10s %-11.2e %-5s %-3d %s' %
          (n, recs[n]['r'], local[n], d, 'yes' if recs[n]['truncated'] else 'no',
           dp, recs[n]['closed']))

print('\n[A1] local > page by more than the printed truncation (>1e-4): %s' % (over or 'none'))
print('[A2] local < page  (would CONTRADICT the page, which truncates down): %s' % (below or 'none'))
print('[A3] rows the page prints with fewer than 5 decimals            : %s' % (coarse or 'none'))
print('[A4] page sequence strictly increasing over n=1..50            : %s' %
      all(float(recs[n]['r']) < float(recs[n + 1]['r']) for n in range(1, 50)))
x6 = math.sqrt(19706163 + 13275064 * math.sqrt(2) - 40 * math.sqrt(
    443374242065 + 313512226176 * math.sqrt(2))) / 534.0
print('[A5] n=6 footnote exact %.6f vs local %.6f (|d|=%.1e); page prints %s'
      % (x6, local[6], abs(x6 - local[6]), recs[6]['r']))

if not os.path.isdir('data'):
    os.mkdir('data')
out = 'data/squincir_n1_50.csv'
with io.open(out, 'w', encoding='utf-8', newline='') as fh:
    fh.write('# source : %s\n' % SRC)
    fh.write('# retrieved 2026-10-10; raw html archived as squincir_raw.html, sha256=%s\n'
             % sha_html)
    fh.write('# page statement: "%s"\n' % claim.group(0))
    fh.write('# SEMANTICS: r_decimal_printed is the radius of the SMALLEST KNOWN circle\n'
             '# holding n unit squares, i.e. a CONSTRUCTION => it UPPER-bounds r_n.\n'
             '# The page truncates the decimal and marks it "+", so r_n <= value + 5e-4.\n'
             '# No row here lower-bounds r_n; only n<=7 are proved, so only those rows\n'
             '# equal r_n. Rows n>=8 carry NO information about N(R).\n')
    fh.write('n,r_decimal_printed,truncated,page_decimals,closed_form,optimality,status\n')
    for n in range(1, 51):
        rc = recs[n]
        dp = len(rc['r'].split('.')[1]) if '.' in rc['r'] else 0
        fh.write('%d,%s,%s,%d,"%s",%s,%s\n' %
                 (n, rc['r'], 'yes' if rc['truncated'] else 'no', dp,
                  rc['closed'].replace(',', ';'), 'yes' if n <= 7 else 'no',
                  rc['status']))
print('\nwrote %s (%d bytes)' % (out, os.path.getsize(out)))
print('sha256(csv) = %s' % hashlib.sha256(open(out, 'rb').read()).hexdigest())

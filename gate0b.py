import numpy as np, math, sys, json
from gate0 import (load_csv, pairwise_sat, pairwise_clip, corners, sep_sat,
                   enclosing_radius, ccw, clip, poly_area, signed_area, sha256)

rows = [l.split() for l in open('packing67.txt') if l.strip() and not l.startswith('#')]
COL = [r[0] for r in rows]
# exact integer certificate: doubled-cm coords, unit square => X1-X0 == 2
CX = np.array([(int(r[1])+int(r[3]))/4.0 for r in rows])
CY = np.array([(int(r[2])+int(r[4]))/4.0 for r in rows])
CT = np.zeros(len(rows))

log = []
def say(s):
    print(s); log.append(s)


def worst_sep(x, y, th, s):
    return pairwise_sat(x, y, th, s)[0]


def solve_side(x, y, th, lo=0.5, hi=1.2):
    """largest s for which the config is still pairwise disjoint: sep_worst(s)=0"""
    if worst_sep(x, y, th, hi) > 0:
        return hi
    for _ in range(200):
        mid = 0.5*(lo+hi)
        if worst_sep(x, y, th, mid) >= 0:
            lo = mid
        else:
            hi = mid
    return 0.5*(lo+hi)


def radius_at(x, y, th, s):
    return enclosing_radius(corners(x, y, th, s))[0]


say('='*74)
say('PART 1  provenance of the 68 baseline: what side is the surviving CSV?')
say('='*74)
p = 'D:/HeliumDownload/mmax68_R5_candidate.csv'
x, y, th, idx, txt = load_csv(p)
say(f'file {p}\nsha256 {sha256(p)}   N={len(x)}')
s_star = solve_side(x, y, th, 0.5, 1.5)
R_star = radius_at(x, y, th, s_star)
say(f'\nmax side making it pairwise disjoint : s* = {s_star:.9f}')
say(f'  separation at s*                   : {worst_sep(x, y, th, s_star):+.3e}  (target 0)')
say(f'  enclosing radius at s*             : R(s*) = {R_star:.9f}')
say(f'  EQUIVALENT UNIT-SQUARE RADIUS      : R(s*)/s* = {R_star/s_star:.9f}')
say('\n  candidate recorded baselines:')
for ref, who in ((5.031405, 'derived 5/0.993758'), (5.052404006121, 'project memory R_imp'),
                 (5.052605585675, 'snap68.log'), (5.052572999132, 'ver68.py ledger tag')):
    say(f'    {ref:.12f}  ({who:24s})  delta from R(s*)/s* = {R_star/s_star-ref:+.9f}')
say(f'\n  and 5/s* = {5.0/s_star:.9f}   (compare 5.031405)')

say('')
say('='*74)
say('PART 2  two independent implementations, pair-by-pair SIGN agreement')
say('='*74)
C1 = corners(x, y, th, 1.0)
n = len(x)
mis = 0
nA = nB = 0
for i in range(n):
    Qi = ccw([tuple(q) for q in C1[i]])
    for j in range(i+1, n):
        Qj = ccw([tuple(q) for q in C1[j]])
        a = sep_sat(x[i], y[i], th[i], x[j], y[j], th[j], 1.0)
        ovA = a < -1e-12
        inter = clip(Qi, Qj)
        ar = poly_area(inter) if len(inter) >= 3 else 0.0
        ovB = ar > 1e-12
        nA += ovA
        nB += ovB
        if ovA != ovB:
            mis += 1
            if mis <= 5:
                say(f'   MISMATCH pair ({i},{j}): A sep={a:+.3e}  B area={ar:.3e}')
say(f'  pairs={n*(n-1)//2}  A flags {nA} overlapping, B flags {nB} overlapping, '
    f' disagreements={mis}')

say('')
say('='*74)
say('PART 3  GATE-0 executed on the config whose ground truth is EXACT')
say('='*74)
say('packing67.txt : 67 axis-aligned unit squares, integer certificate, R*=5.000000000000')
say('expectation from the certificate: 8 corners satisfy x^2+y^2 = 100 exactly,')
say('  so containment is tight to 0.0, and G0-C must fail by EXACTLY 1e-5.')
sha67 = sha256('packing67.txt')
say(f'\nfile packing67.txt  sha256 {sha67}')
C67 = corners(CX, CY, CT, 1.0)
r67, arg, Rm = enclosing_radius(C67)
sep67, pi, pj = pairwise_sat(CX, CY, CT, 1.0)
ar67, ai, aj = pairwise_clip(C67)
say(f'impl A worst separation = {sep67:+.12e}  at pair ({pi},{pj})')
say(f'impl B max overlap area = {ar67:+.12e}  at pair ({ai},{aj})')
say(f'implicit enclosing radius (recomputed, float) = {r67:.12f}')
say(f'  baseline recorded (integer, x^2+y^2<=100)   = 5.000000000000')
say(f'  difference = {r67-5.0:+.3e}   vs TOL_CONTAIN=1e-9 -> '
    f'{"WITHIN tolerance" if abs(r67-5.0)<=1e-9 else "OUTSIDE tolerance"}')
say(f'#corners exactly on the circle (r>5-1e-12) = {(Rm>5-1e-12).sum()}')
say('\n  G0 tests (disjointness gate AND containment gate, reported separately):')
for tag, Rc, expect in (('G0-A baseline', 5.0, 'LEGAL'),
                        ('G0-B relaxed', 5.0+1e-5, 'LEGAL'),
                        ('G0-C tightened', 5.0-1e-5, 'ILLEGAL')):
    gd = sep67 >= -1e-9
    gc = r67 <= Rc + 1e-9
    verdict = 'LEGAL' if (gd and gc) else 'ILLEGAL'
    say(f'   {tag:16s} R={Rc:.12f}  disjoint={gd}  contained={gc}  -> {verdict:8s} '
        f'expected {expect:8s} {"PASS" if verdict==expect else "*** FAIL ***"}')
    if tag == 'G0-C':
        say(f'        measured containment excess = {r67-Rc:.12f} (closed form 1e-5)')
say('\n  sensitivity: smallest container change the chain resolves')
for eps in (1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-12):
    ok = r67 <= (5.0+eps) + 1e-9
    ok2 = r67 <= (5.0-eps) + 1e-9
    say(f'    R=5+{eps:.0e}: pass={ok}   R=5-{eps:.0e}: pass={ok2} '
        f'(TOL_CONTAIN=1e-9 absorbs everything below 1e-9)')

json.dump(dict(csv_sha=sha256(p), s_star=float(s_star), R_star=float(R_star),
               unit_equiv=float(R_star/s_star), pack67_R=float(r67),
               pack67_sep=float(sep67), ab_mismatch=int(mis)),
          open('gate0_summary.json', 'w'), indent=1)
open('gate0_log.txt', 'w', encoding='utf-8').write('\n'.join(log))

import numpy as np
import rate3 as R

rows = [l.split() for l in open('packing67.txt') if l.strip() and not l.startswith('#')]
COL = [r[0] for r in rows]
C = np.array([[(int(r[1])+int(r[3]))/4.0, (int(r[2])+int(r[4]))/4.0] for r in rows])
N = len(C)
TH = np.zeros(N)
cl = R.contacts(C, TH)

# how many contacts are TIED (corner-corner) -> disjunctive, my model over-constrains them
tied = 0
for (i, j, g, a, sg, ki, mi, kj, mj) in cl:
    d = C[j]-C[i]
    seen = {}
    for kk in (i, j):
        c, s = np.cos(TH[kk]), np.sin(TH[kk])
        for ax in (np.array([c, s]), np.array([-s, c])):
            key = (round(abs(ax[0]), 9), round(abs(ax[1]), 9))   # dedupe up to sign
            t = abs(d @ ax) - R.support(ax, TH[i]) - R.support(ax, TH[j])
            seen[key] = max(seen.get(key, -9e9), t)
    mx = max(seen.values())
    if sum(1 for t in seen.values() if t >= mx-1e-9) >= 2:
        tied += 1
print(f'active contacts = {len(cl)}   of which tied/corner-corner = {tied}   '
      f'unique-face = {len(cl)-tied}')

# rotation-lock: a square is rotation-locked if some contact enforces an axis that is
# ITS OWN edge normal (kink -> h can only grow -> any nonzero rotation penetrates at 1st order)
locked = np.zeros(N, bool)
deg = np.zeros(N, int)
for (i, j, g, a, sg, ki, mi, kj, mj) in cl:
    deg[i] += 1
    deg[j] += 1
    if ki > 0:
        locked[i] = True
    if kj > 0:
        locked[j] = True
print(f'rotation-locked squares (own-normal contact): {locked.sum()}/{N}'
      f'   unlocked: {[k for k in range(N) if not locked[k]]}')

# face-contact degree distribution
print('contact degree per square: min=%d max=%d mean=%.2f  hist=%s' %
      (deg.min(), deg.max(), deg.mean(), np.bincount(deg)))

# dof vs constraints
print(f'\ndimension count:  translation dof={2*N}  +rotation dof={3*N}  '
      f'active contact rows={len(cl)}  binding corner rows=8')
print(f'   translation-only:  {2*N} dof - {len(cl)} hard first-order constraints = {2*N-len(cl)}')
print(f'   with rotation:     {3*N} dof - {len(cl)} = {3*N-len(cl)}')

# per-corner mobility (which corners can move in ALONE)
import scipy.optimize as so
for k in range(N):
    pass
res0, nc, nb, nv, _, D = R.first_order_rate(C, False)
print(f'\nmax-min t* (T) = {res0.fun:+.6f}  ->  no collective inward motion of the 8')

# slack accounting at R=5
P, RR = R.corners(C, TH)
sl = 5.0 - RR
print(f'\ncorner slack at R=5:  #corners with slack 0 = {(sl < 1e-12).sum()}, '
      f'slack>0.5 = {(sl > 0.5).sum()}, max slack = {sl.max():.4f}')
# area accounting
print(f'area: disk 78.5398 - 67 squares 67.0 = waste {(np.pi*25-67):.4f} '
      f'({100*(np.pi*25-67)/(np.pi*25):.1f}% of disk)')

# colour breakdown of the binding owners
_, _, bic = None, None, None
P, RR = R.corners(C, TH)
owners = sorted({k for k in range(N) for m in range(4) if RR[k, m] > 5.0-1e-9})
from collections import Counter
print(f'binding-corner owners ({len(owners)}): ' +
      ', '.join(f'{k}:{COL[k]}' for k in owners))
print('colour counts overall:', dict(Counter(COL)))

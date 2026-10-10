"""Constructive UPPER bounds on tau(5), read together with the measure LP.

A point set P pierces the whole legal family as soon as its covering radius is below the
inradius of a unit square (0.5), because every unit square contains the disc of radius
0.5 centred on itself.  A legal square lies entirely inside the disc, so any point of P
that lands in it is inside the disc.  Hence a truncated lattice with covering radius
< 0.5 is an EXPLICIT piercing set and |P| is a rigorous upper bound on tau(5).

The truncated lattice can have a worse covering radius than the infinite one, so the
bound is checked against the actual candidate family instead of trusting the algebra.
"""
import math, json, time
import numpy as np
from scipy.spatial import cKDTree
import g2_tau as T

t0 = time.time()
RHO = 0.4999            # strictly below the inradius 0.5 => strictly inside


def lattices():
    out = {}
    g = RHO * math.sqrt(2.0)
    n = int(math.ceil(T.R / g)) + 2
    pts = [[i * g, j * g] for i in range(-n, n + 1) for j in range(-n, n + 1)
           if (i * g) ** 2 + (j * g) ** 2 <= T.R * T.R]
    out['square'] = (g, np.array(pts))
    a = RHO * math.sqrt(3.0)
    n = int(math.ceil(T.R / a)) + 2
    row = a * math.sqrt(3.0) / 2.0
    pts = []
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            x = j * a + (abs(i) % 2) * a / 2.0
            y = i * row
            if x * x + y * y <= T.R * T.R:
                pts.append([x, y])
    out['hexagonal'] = (a, np.array(pts))
    return out


def unpierced(P, F):
    """count placements containing NO point of P strictly inside (exact test)"""
    tree = cKDTree(P)
    miss = 0
    for x, y, th in F:
        ct, st = math.cos(th), math.sin(th)
        cand = tree.query_ball_point([x, y], 0.71)
        ok = False
        for k in cand:
            px, py = P[k]
            dx, dy = px - x, py - y
            u = dx * ct + dy * st
            v = -dx * st + dy * ct
            if max(abs(u), abs(v)) < 0.5:
                ok = True
                break
        if not ok:
            miss += 1
    return miss


res = {}
for name, (sp, P) in lattices().items():
    cov = sp / math.sqrt(2.0 if name == 'square' else 3.0)
    F = T.build_family(0.18, 19)
    miss = unpierced(P, F)
    res[name] = dict(spacing=sp, covering_radius=cov, n_points=int(len(P)),
                    placements_tested=int(len(F)), unpierced=int(miss))
    print(f'[BOUND] {name}: spacing={sp:.4f} covering_radius={cov:.4f} '
          f'|P|={len(P)} placements={len(F)} UNPIERCED={miss} '
          f'-> tau(5) <= {len(P) if miss == 0 else "NOT ESTABLISHED"} '
          f'({time.time()-t0:.1f}s)', flush=True)
json.dump(res, open('g2_tau_bounds.json', 'w'), indent=1)
print('[BOUND] reading: tau(5) is bracketed BELOW by 67 (weak duality on the verified '
      '67) and ABOVE by an explicit piercing set. A 67-point piercing set -- the only '
      'thing G2 can use -- would have to beat the naive lattice by a wide margin.', flush=True)

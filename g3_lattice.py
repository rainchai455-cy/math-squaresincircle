import numpy as np, json, sys, time
from math import isqrt
from scipy.optimize import milp, LinearConstraint, Bounds

# ============================================================
# G3 phase L : EXACT lattice probe (declared search space).
#   search space : 67 axis-aligned unit squares whose centres lie on the
#                  grid h*Z^2   (h = 1/4 by default)   -- NO rotation.
#   legality     : INTEGER-EXACT.  centre = (i,j)*h -> corners = (i±k, j±k)*h
#                  with k = h*2/... see below; containment is
#                       (i+sx)^2 + (j+sy)^2 <= floor(D^2 * R^2)
#                  with D = 1/h and sx,sy = +-D/2  (all integers).
#   positive control : at R=5.0 the certified 67 packing (0.5-grid) is inside
#                  this search space, so max count must be >= 67.
# ============================================================


def candidates(h, R):
    """lattice points whose whole unit square is inside radius R, integer-exact."""
    D = int(round(1/h))
    assert abs(D*h - 1) < 1e-12
    k = D//2                      # corner offset in lattice units (needs D even)
    LIM = int(np.floor(D*D*R*R + 1e-9))
    rng = int(np.ceil(D*R)) + 1
    pts = []
    for i in range(-rng, rng+1):
        for j in range(-rng, rng+1):
            # worst corner = (|i|+k, |j|+k)
            a, b = abs(i)+k, abs(j)+k
            if a*a + b*b <= LIM:
                pts.append((i, j))
    return D, k, LIM, np.array(pts, dtype=np.int64)


def conflicts(pts, D):
    """two lattice squares overlap  <=>  |di| < D and |dj| < D (both)."""
    n = len(pts)
    x, y = pts[:, 0], pts[:, 1]
    pairs = []
    order = np.lexsort((y, x))
    xs, ys = x[order], y[order]
    # bucket by x to keep it O(n * band)
    from collections import defaultdict
    buck = defaultdict(list)
    for t, (a, b) in enumerate(zip(xs, ys)):
        buck[int(a)].append(t)
    keys = sorted(buck)
    for a in keys:
        for a2 in range(a, a + D):
            if a2 not in buck:
                continue
            for p in buck[a]:
                for q in buck[a2]:
                    if q <= p:
                        continue
                    if abs(int(ys[q]) - int(ys[p])) < D:
                        pairs.append((int(order[p]), int(order[q])))
    return n, pairs


def greedy(pts, D, key):
    """weighted greedy placement: strong primal incumbent."""
    taken, occ = [], set()
    for t in np.argsort(key):
        i, j = int(pts[t, 0]), int(pts[t, 1])
        ok = True
        for a in range(i - D + 1, i + D):
            for b in range(j - D + 1, j + D):
                if (a, b) in occ:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            taken.append(int(t))
            for a in range(i, i + D):
                for b in range(j, j + D):
                    occ.add((a, b))
    return taken


def best_greedy(pts, D, tries=40, seed=0):
    rng = np.random.default_rng(seed)
    r = np.hypot(pts[:, 0], pts[:, 1])
    best = []
    for t in range(tries):
        k = r + (rng.random(len(pts))*1.6 if t else 0.0)
        s = greedy(pts, D, k)
        if len(s) > len(best):
            best = s
        # greedy-add: fill remaining gaps
        taken = set(s)
        occ = set()
        for u in s:
            i, j = int(pts[u, 0]), int(pts[u, 1])
            for a in range(i, i+D):
                for b in range(j, j+D):
                    occ.add((a, b))
        for u in np.argsort(r):
            u = int(u)
            if u in taken:
                continue
            i, j = int(pts[u, 0]), int(pts[u, 1])
            if all((a, b) not in occ for a in range(i-D+1, i+D)
                   for b in range(j-D+1, j+D)):
                taken.add(u)
                for a in range(i, i+D):
                    for b in range(j, j+D):
                        occ.add((a, b))
        if len(taken) > len(best):
            best = sorted(taken)
    return best


def solve(h, R, tl, verbose=True):
    t0 = time.time()
    D, k, LIM, pts = candidates(h, R)
    n, prs = conflicts(pts, D)
    g = best_greedy(pts, D)
    if verbose:
        print(f'[L] h={h}  R={R}  D={D}  LIM={LIM}  |cands|={n}  |conflict pairs|={len(prs)}'
              f'  greedy={len(g)}  ({time.time()-t0:.1f}s)')
    if len(prs) == 0:
        return n, pts[np.arange(n)], 'trivial', n
    P = np.array(prs)
    rows, cols, vals = [], [], []
    for e, (a, b) in enumerate(prs):
        rows += [e, e]; cols += [int(a), int(b)]; vals += [1.0, 1.0]
    from scipy.sparse import coo_matrix
    A = coo_matrix((vals, (rows, cols)), shape=(len(prs), n)).tocsr()
    lc = LinearConstraint(A, -np.inf, 1.0)
    lb = np.zeros(n); ub = np.ones(n)
    v = np.ones(n); v[g] = 1.0001          # tiny bias toward the greedy incumbent
    res = milp(c=-v, constraints=lc, integrality=np.ones(n),
               bounds=Bounds(lb, ub),
               options=dict(time_limit=tl, mip_rel_gap=1e-9, presolve=True))
    dt = time.time() - t0
    if res.x is None:
        if verbose:
            print(f'[L]   no incumbent  status={res.status} {res.message}  ({dt:.1f}s)')
        return 0, np.empty((0, 2), dtype=np.int64), res.message, n
    sel = np.where(res.x > 0.5)[0]
    cnt = len(sel)                                  # honest objective
    gap = getattr(res, 'mip_relaxation', None)
    if gap is None:
        ab = getattr(res, 'mip_absolute_gap', None)
        gap = None if ab is None else cnt + ab
    opt = 'OPTIMAL (proven)' if (gap is not None and abs(gap - cnt) < 0.5001) else f'incumbent (LP/relax bound {gap})'
    if verbose:
        print(f'[L]   MILP: max axis-aligned = {cnt}   status={res.status}  {opt}  ({dt:.1f}s)')
    return cnt, pts[sel], opt, n


def export(pts, D, h, R, path, tag):
    """integer-exact re-check, then write centres."""
    C = pts*D*h if len(pts) else np.empty((0, 2))
    if len(pts):
        worst = max((abs(i)+D//2)**2 + (abs(j)+D//2)**2 for i, j in pts)
        LIM = int(np.floor(D*D*R*R + 1e-9))
        ok = worst <= LIM
        xy = pts.astype(float)*h
        d = np.abs(xy[:, None, 0]-xy[None, :, 0]); e = np.abs(xy[:, None, 1]-xy[None, :, 1])
        n = len(pts)
        tri = np.triu_indices(n, 1)
        ov = int(((d[tri] < 1-1e-12) & (e[tri] < 1-1e-12)).sum())
        rmax = float(np.max(np.hypot(np.abs(xy)[:, 0]+0.5, np.abs(xy)[:, 1]+0.5)))
        print(f'[L] {tag}: N={n}  exact containment {ok} (worst corner^2 {worst} <= {LIM}, R={R})'
              f'  overlap pairs={ov}  implicit R = {rmax:.12f}')
        json.dump(dict(tag=tag, N=n, R=R, h=h, exact_containment=bool(ok),
                       overlaps=ov, r_implicit=rmax,
                       x=xy[:, 0].tolist(), y=xy[:, 1].tolist(),
                       int_pts=pts.tolist(), D=D), open(path, 'w'), indent=1)
        return n, ok, ov
    print(f'[L] {tag}: no selection')
    return 0, False, -1


def area_bound(R):
    """INVALID as an upper bound - kept only as a recorded self-correction.
    The argument 'centres are Linf-1-separated so their own unit squares tile the
    centre region {(|x|+.5)^2+(|y|+.5)^2<=R^2}' is FALSE: that region is not convex
    and the Linf ball around a centre is not contained in it.  The control run
    disproved it directly: area(5.0)=59.57 while the MILP finds 67 axis-aligned
    squares at R=5.0.  The only lattice-free bound left is the trivial
    N <= floor(pi R^2) (disjoint unit squares inside the disk)."""
    from scipy.integrate import quad
    from math import sqrt, pi
    xmax = sqrt(R*R - 0.25) - 0.5
    f = lambda x: max(0.0, sqrt(max(0.0, R*R - (x+0.5)**2)) - 0.5)
    a, err = quad(f, 0.0, xmax, limit=400, epsabs=1e-12, epsrel=1e-12)
    return 4.0*a, err, int(np.floor(pi*R*R))


if __name__ == '__main__':
    tl = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    h = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    Rs = [float(s) for s in (sys.argv[3] if len(sys.argv) > 3 else '5.0,4.99').split(',')]
    Dq = int(round(1/h))
    out = {'h': h, 'D': Dq, 'area_note': {}}
    for R in Rs:
        A, err, triv = area_bound(R)
        out['area_note'][str(R)] = dict(centre_region_area=A, quad_err=err,
                                        valid_trivial_bound=triv,
                                        note='centre_region_area is NOT an upper bound '
                                             '(falsified by the R=5.0 control: 59.57 < 67)')
        print(f'[L] R={R}: centre-region area={A:.9f} (INVALID as a bound - falsified by the '
              f'67 found at R=5.0);  valid lattice-free bound: N <= floor(pi R^2) = {triv}')
    out['runs'] = {}
    for R in Rs:
        cnt, pts, note, nc = solve(h, R, tl)
        out['runs'][str(R)] = dict(count=cnt, cands=nc, note=str(note))
        export(pts, Dq, h, R, f'g3_lattice_R{str(R).replace(".", "")}.json', f'R={R} h={h}')
    json.dump(out, open('g3_lattice_summary.json', 'w'), indent=1)
    if len(Rs) == 2:
        print(f'\n[L] axis-aligned {h}-lattice max count:  R={Rs[0]} -> {out["runs"][str(Rs[0])]["count"]}   '
              f'R={Rs[1]} -> {out["runs"][str(Rs[1])]["count"]}')
        v = 'REACHABLE in this lattice' if out['runs'][str(Rs[1])]['count'] >= 67 else \
            'NOT reachable within this declared lattice'
        print(f'[L] verdict: 67 squares with NO rotation at R={Rs[1]} are {v}   '
              f'(statement about the {h}-lattice + the lattice-free bound above, '
              f'NOT about packings that use rotation)')


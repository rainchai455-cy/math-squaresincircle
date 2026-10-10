import numpy as np, hashlib, json, sys, math

# ============================================================
# Gate-0 : baseline calibration for the packing search chain.
# Two INDEPENDENT geometry implementations, no optimiser, no
# movement of squares.  Tolerances are pre-registered below.
# ============================================================
TOL_OVERLAP = 1e-9      # linear penetration allowed between two square interiors
TOL_CONTAIN = 1e-9      # corner may exceed container by this
SIDE = 1.0              # unit squares (assumption under test, see probe)


def sha256(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def load_csv(path):
    txt = open(path, 'r', encoding='utf-8', errors='replace').read()
    lines = [l.strip() for l in txt.split('\n') if l.strip()]
    hdr = lines[0].split(',')
    assert hdr == ['index', 'x', 'y', 'theta_deg'], hdr
    idx, x, y, th = [], [], [], []
    for l in lines[1:]:
        p = l.split(',')
        idx.append(int(p[0])); x.append(float(p[1])); y.append(float(p[2])); th.append(float(p[3]))
    return np.array(x), np.array(y), np.radians(np.array(th)), idx, txt


# ---------- implementation A : SAT over the 4 edge normals ----------
def sep_sat(ax, ay, at, bx, by, bt, s=SIDE):
    """max over the 4 candidate axes of (|dc.a| - h_a - h_b). >=0 => disjoint."""
    dx, dy = bx-ax, by-ay
    best = -math.inf
    hs = s/2.0
    cand = []
    for t in (at, bt):
        c, si = math.cos(t), math.sin(t)
        cand.append((c, si)); cand.append((-si, c))
    for (a1, a2) in cand:
        ha = hs*(abs(a1*math.cos(at)+a2*math.sin(at)) + abs(-a1*math.sin(at)+a2*math.cos(at)))
        hb = hs*(abs(a1*math.cos(bt)+a2*math.sin(bt)) + abs(-a1*math.sin(bt)+a2*math.cos(bt)))
        best = max(best, abs(dx*a1+dy*a2) - ha - hb)
    return best


def pairwise_sat(x, y, th, s=SIDE):
    n = len(x)
    worst = math.inf
    wi = wj = -1
    for i in range(n):
        for j in range(i+1, n):
            v = sep_sat(x[i], y[i], th[i], x[j], y[j], th[j], s)
            if v < worst:
                worst, wi, wj = v, i, j
    return worst, wi, wj


# ---------- implementation B : Sutherland-Hodgman clipping of two convex quads ----------
def corners(x, y, th, s=SIDE):
    h = s/2.0
    off = [(-h, -h), (-h, h), (h, h), (h, -h)]
    out = np.empty((len(x), 4, 2))
    for k in range(len(x)):
        c, si = math.cos(th[k]), math.sin(th[k])
        R = np.array([[c, -si], [si, c]])
        out[k] = np.array(off) @ R.T + np.array([x[k], y[k]])
    return out


def poly_area(P):
    n = len(P)
    a = 0.0
    for i in range(n):
        a += P[i][0]*P[(i+1) % n][1] - P[(i+1) % n][0]*P[i][1]
    return abs(a)/2.0


def clip(subject, clipQuad):
    """Sutherland-Hodgman: clip convex polygon `subject` by convex polygon clipQuad."""
    out = list(subject)
    n = len(clipQuad)
    for i in range(n):
        if not out:
            break
        A, B = clipQuad[i], clipQuad[(i+1) % n]
        ex, ey = B[0]-A[0], B[1]-A[1]
        # inside = left of directed edge (clipQuad must be CCW)
        def inside(Pt):
            return ex*(Pt[1]-A[1]) - ey*(Pt[0]-A[0]) >= -1e-15
        def inter(P, Q):
            v1 = ex*(P[1]-A[1]) - ey*(P[0]-A[0])
            v2 = ex*(Q[1]-A[1]) - ey*(Q[0]-A[0])
            f = v1/(v1-v2) if v1 != v2 else 0.0
            return (P[0]+f*(Q[0]-P[0]), P[1]+f*(Q[1]-P[1]))
        nxt, last = [], out[-1]
        for pt in out:
            if inside(pt):
                if not inside(last):
                    nxt.append(inter(last, pt))
                nxt.append(pt)
            elif inside(last):
                nxt.append(inter(last, pt))
            last = pt
        out = nxt
    return out


def signed_area(P):
    n = len(P)
    a = 0.0
    for i in range(n):
        a += P[i][0]*P[(i+1) % n][1] - P[(i+1) % n][0]*P[i][1]
    return a/2.0


def ccw(Q):
    return list(reversed(Q)) if signed_area(Q) < 0 else Q


def pairwise_clip(C, s=SIDE):
    """independent overlap test: intersect every pair by polygon clipping."""
    n = len(C)
    worst_area = -math.inf
    wi = wj = -1
    for i in range(n):
        Qi = ccw([tuple(p) for p in C[i]])
        for j in range(i+1, n):
            Qj = ccw([tuple(p) for p in C[j]])
            inter = clip(Qi, Qj)
            a = poly_area(inter) if len(inter) >= 3 else 0.0
            if a > worst_area:
                worst_area, wi, wj = a, i, j
    return worst_area, wi, wj


def enclosing_radius(C):
    r = np.hypot(C[:, :, 0], C[:, :, 1])
    return r.max(), np.unravel_index(r.argmax(), r.shape), r


def probe_side(x, y):
    """the csv has no size column: recover side from the nearest-neighbour distance"""
    n = len(x)
    d = np.hypot(x[:, None]-x[None, :], y[:, None]-y[None, :])
    np.fill_diagonal(d, np.inf)
    return np.sort(d.min(axis=1))


if __name__ == '__main__':
    path = sys.argv[1]
    log = []

    def say(s):
        print(s); log.append(s)

    say('GATE-0  baseline calibration')
    say(f'config file      = {path}')
    say(f'sha256           = {sha256(path)}')
    say(f'pre-registered tolerances: TOL_OVERLAP={TOL_OVERLAP:g} (linear), '
        f'TOL_CONTAIN={TOL_CONTAIN:g} (radius)')
    x, y, th, idx, txt = load_csv(path)
    say(f'N = {len(x)}  indices contiguous 0..{len(x)-1}: {idx == list(range(len(x)))}')
    nn = probe_side(x, y)
    say(f'nearest-neighbour centre distance: min={nn.min():.9f} median={np.median(nn):.9f} '
        f'max={nn.max():.9f}')
    say(f'  -> consistent with SIDE={SIDE:g} only if min >= {SIDE:.9f}')
    C = corners(x, y, th)
    rmax, arg, R = enclosing_radius(C)
    say(f'\nimplicit enclosing radius R_cfg = {rmax:.12f}  at square {arg[0]} corner {arg[1]}')
    for ref in (5.052404006121, 5.052605585675, 5.052572999132, 5.031405, 5.0):
        say(f'   vs recorded baseline {ref:.12f} : delta = {rmax-ref:+.9f}')
    sat, si, sj = pairwise_sat(x, y, th)
    say(f'\nimpl A (SAT, 4 edge normals): worst separation = {sat:+.12e}  at pair ({si},{sj})')
    ar, ai, aj = pairwise_clip(C)
    say(f'impl B (Sutherland-Hodgman):  max overlap area  = {ar:+.12e}  at pair ({ai},{aj})')
    say(f'A/B agree on disjointness?  {sat >= -TOL_OVERLAP} vs {ar <= 1e-15}')
    say('\n--- G0 tests ---')
    for tag, Rc in (('G0-A baseline', rmax), ('G0-B relaxed', rmax+1e-5),
                    ('G0-C tightened', rmax-1e-5)):
        legal = (sat >= -TOL_OVERLAP) and (rmax <= Rc + TOL_CONTAIN)
        say(f'  {tag:16s} R={Rc:.12f}  containment {rmax:.12f} <= {Rc+TOL_CONTAIN:.12f} '
            f'-> {rmax <= Rc+TOL_CONTAIN}   => {"LEGAL" if legal else "ILLEGAL"}')
    json.dump(dict(file=path, sha256=sha256(path), N=len(x), R_cfg=rmax,
                   sat_worst=sat, clip_area=ar, tol=[TOL_OVERLAP, TOL_CONTAIN]),
              open('gate0_summary.json', 'w'), indent=1)
    open('gate0_log.txt', 'w', encoding='utf-8').write('\n'.join(log))

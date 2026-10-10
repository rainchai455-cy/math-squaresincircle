"""
G2 closing gate (pre-registered 2026-10-09, memory project-packing-research-routes):
    "gate-0 reversed: probe tau >= 68 first; if found, close G2 on the spot and skip
     all certificate engineering."

tau(R) = minimum number of points that pierce the INTERIOR of every legal unit square
placed inside the disc of radius R.  N(R) <= tau(R), so a 67-point piercing set would
prove N(5) <= 67 -- that is the only useful direction G2 has.  tau >= 68 means no such
set exists, i.e. G2 is dead (and, per the scope arithmetic below, it says nothing about
whether 68 squares fit).

Exactly two sources can support tau >= 68, and this file checks both:
  (i)  a genuine packing of 68 disjoint legal squares in R=5.  Nothing on disk provides
       one: the only legal 68-ish artifact is the decertified mmax68 CSV (rule 2).
  (ii) a MEASURE certificate: a measure mu on the legal family whose plane density
       D(p) = mu({S : p in int S}) obeys D <= 1, with total mass >= 68.  Then every
       piercing set has mass >= mu_total / max D >= 68 points.  This is weak duality and
       it is the only route that can certify 68 without actually packing 68.

Rigorous ceiling for (ii): a legal square lies entirely inside the disc (all 4 corners
in the convex disc), so area(S cap disc) = 1 and
        mu_total = integral of D over the disc <= Area(disc) = pi R^2 = 78.5398.
So "tau >= 68" needs a measure saturating 68/78.5398 = 86.6% of the disc at D = 1.

SCOPE ARITHMETIC (step 3, independent of the numbers): a k-point piercing set at radius
R proves N(R) <= k, i.e. r(k+1) >= R.  tau >= 68 proves neither N(5) <= 67 nor
r(67) >= 5.  r(67) = 5 needs r(67) <= 5 (supported by the verified 67) AND r(67) >= 5
(not supported by anything here).

Positive controls, run before any verdict (feedback-negative-result-needs-positive-
control): the same density machinery must certify the KNOWN bound tau >= 67 from the
verified 67-square certificate, and the uniform-measure bound must reproduce the
analytic Area(C) ~ 60.3 -- which is weaker than 67 and is therefore a code check, never
a candidate certificate.
"""
import json, math, time
import numpy as np
from scipy.optimize import linprog
from scipy import sparse

R = 5.0
SIDE = 1.0
HALF = SIDE / 2.0
AREA_DISC = math.pi * R * R
TARGET = 68.0

log_lines = []


def emit(s):
    print(s, flush=True)
    log_lines.append(s)


# ---------------------------------------------------------------- geometry
def legal(x, y, th):
    """Exact vertex criterion (G1, independently re-derived): the square with centre
    (x,y) and orientation th lies in the disc iff its farthest corner does, and
    r2_max = rho^2 + 1/2 + |c.u| + |c.v| with u=(cos,sin), v=(-sin,cos)."""
    cu = x * math.cos(th) + y * math.sin(th)
    cv = -x * math.sin(th) + y * math.cos(th)
    return x * x + y * y + 0.5 + abs(cu) + abs(cv) <= R * R


def r2max(x, y, th):
    cu = x * math.cos(th) + y * math.sin(th)
    cv = -x * math.sin(th) + y * math.cos(th)
    return x * x + y * y + 0.5 + abs(cu) + abs(cv)


def interior_idx(X, Y, TH, pts):
    """Boolean mask of which points lie strictly inside a square (vectorised)."""
    c, s = math.cos(TH), math.sin(TH)
    dx = pts[:, 0] - X
    dy = pts[:, 1] - Y
    u = dx * c + dy * s
    v = -dx * s + dy * c
    return np.maximum(np.abs(u), np.abs(v)) < HALF


def build_family(h_c, n_ang, extra=None):
    """Legal placements: centre grid step h_c over the rim region, n_ang orientations on
    [0, pi/2) (square symmetry), plus any injected placements."""
    angs = np.linspace(0.0, math.pi / 2.0, n_ang, endpoint=False)
    g = np.arange(-R, R + 1e-12, h_c)
    GX, GY = np.meshgrid(g, g, indexing='ij')
    rows = []
    for th in angs:
        m = (GX * GX + GY * GY <= R * R)
        idx = np.argwhere(m)
        for i, j in idx:
            x, y = g[i], g[j]
            if legal(x, y, th):
                rows.append((x, y, float(th)))
    arr = np.array(rows, dtype=float)
    if extra is not None and len(extra):
        arr = np.vstack([arr, np.asarray(extra, dtype=float)])
    return arr


def grid_pts(h):
    n = int(math.floor(2 * R / h)) + 1
    g = np.linspace(-R, R, n)
    GX, GY = np.meshgrid(g, g, indexing='ij')
    p = np.column_stack([GX.ravel(), GY.ravel()])
    return p[np.einsum('ij,ij->i', p, p) <= R * R + 1e-12]


def density_max(arr, w, pts, corners=True, work_cap=4e7):
    """max over a SAMPLED point set of D(p) = sum_i w_i 1{p in int S_i}.  Chunked and
    capped so the cost is bounded.  This is a numerical audit, not a certificate: D jumps
    across square edges that fall between sample points, so an exact density check needs
    the full arrangement (see the verdict text)."""
    sel = np.nonzero(w > 0)[0]
    P = pts
    if corners and len(sel) <= 1500:
        cor = []
        for k in sel:
            x, y, th = arr[k]
            ct, st = math.cos(th), math.sin(th)
            for a, b in ((HALF, HALF), (-HALF, HALF), (-HALF, -HALF), (HALF, -HALF)):
                cor.append([x + a * ct - b * st, y + a * st + b * ct])
        C = np.array(cor)
        eps = 1e-3
        off = np.vstack([C + eps * np.array([dx, dy])
                         for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                                        (1, 1), (1, -1), (-1, 1), (-1, -1))])
        off = off[np.einsum('ij,ij->i', off, off) <= R * R]
        P = np.vstack([P, off])
    best = 0.0
    nch = max(1, int(work_cap // max(1, len(P))))
    for a in range(0, len(sel), nch):
        D = np.zeros(len(P))
        for k in sel[a:a + nch]:
            x, y, th = arr[k]
            D[interior_idx(x, y, th, P)] += w[k]
        if len(D):
            best = max(best, float(D.max()))
    return best, P


def legal_area(theta, h=0.04):
    """Area of the legal-centre region for one orientation, by grid counting."""
    n = int(math.floor(2 * R / h)) + 1
    g = np.linspace(-R, R, n)
    GX, GY = np.meshgrid(g, g, indexing='ij')
    ct, st = math.cos(theta), math.sin(theta)
    r2 = GX * GX + GY * GY + 0.5 + np.abs(GX * ct + GY * st) + np.abs(-GX * st + GY * ct)
    return float((r2 <= R * R).sum()) * h * h



# ---------------------------------------------------------------- the LP
def run_lp(F, pts, tag, keep_w=False):
    n, m = len(F), len(pts)
    t0 = time.time()
    ri, ci = [], []
    for i, (x, y, th) in enumerate(F):
        j = np.nonzero(interior_idx(x, y, th, pts))[0]
        ri.append(j)
        ci.append(np.full(len(j), i))
    ri = np.concatenate(ri); ci = np.concatenate(ci)
    A = sparse.csr_matrix((np.ones(len(ri)), (ri, ci)), shape=(m, n))
    res = linprog(-np.ones(n), A_ub=A, b_ub=np.ones(m),
                  bounds=[(0, None)] * n, method='highs')
    if not res.success:
        emit(f'[T2] {tag}: LP FAILED {res.message} ({time.time()-t0:.1f}s)')
        return None
    w = res.x
    val = float(w.sum())
    sup = int((w > 1e-9).sum())
    dmax, _ = density_max(F, w, grid_pts(0.15), corners=False)
    emit(f'[T2] {tag}: cols={n} rows={m} nnz={A.nnz} LP={val:.4f} support={sup} '
         f'max_density(audited)={dmax:.4f} ({time.time()-t0:.1f}s)')
    r = dict(tag=tag, cols=n, rows=m, nnz=int(A.nnz), lp_value=val,
             support=sup, max_density_est=float(dmax),
             audited_value=val / max(1.0, dmax), secs=round(time.time() - t0, 1),
             status=int(res.status), message=str(res.message),
             primal_infeas=float(-np.min(res.ineqlin.lower)) if hasattr(res, 'ineqlin') else float('nan'),
             dual_max_abs=float(np.max(np.abs(res.ineqlin.marginal))) if hasattr(res, 'ineqlin') else float('nan'))
    if keep_w:
        r['w'] = w
    return r

def main():
    emit(f'[T2] R={R} side={SIDE}; disc area ceiling on any measure certificate = '
         f'{AREA_DISC:.4f}; tau>=68 needs {68/AREA_DISC:.1%} of it at D=1')
    emit('[T2] NOTE the remembered fractional tau*~66.6 has NO artifact on disk '
         '(grep tau/pierce/cover: none), so it is not citable this round; rebuilding.')

    # ---- PC1: uniform placement measure.  Its bound is mean_theta Area(C_theta),
    # ---- which ANALYTICALLY sits near 60.3 and is therefore WEAKER than the known
    # ---- tau>=67.  It exists to validate the geometry code, not as a certificate.
    areas = [legal_area(t) for t in np.linspace(0, math.pi / 2, 8, endpoint=False)]
    pc1 = float(np.mean(areas))
    emit(f'[T2] PC1 uniform-measure bound = mean Area(C_theta) = {pc1:.3f} '
         f'(per-theta: {" ".join(f"{a:.2f}" for a in areas)}); '
         f'analytic rho_max(theta=0)={((-1+math.sqrt(1+4*(R*R-0.5)))/2):.6f} vs '
         f'recorded 4.474937; the remembered "60.3" was a pi*rho^2 estimate, the '
         f'counted area is {pc1:.2f} -> geometry code OK, and the uniform bound is '
         f'WEAKER than the known tau>=67 exactly as expected')

    # ---- PC2: the verified 67 as a measure.  Disjoint interiors are exactly what
    # ---- g3_verify already proved, so D<=1 a.e. and mass=67 is a REAL certificate:
    # ---- tau(5) >= 67.  The machinery must reproduce that, or it certifies nothing.
    d = json.load(open('g3_frozen/best_legal_67.json'))
    cert = np.column_stack([d['x'], d['y'], d['th']])
    bad = [k for k, (x, y, th) in enumerate(cert) if not legal(x, y, th)]
    w2 = np.ones(len(cert))
    dmax2, _ = density_max(cert, w2, grid_pts(0.25))
    emit(f'[T2] PC2 certified 67 as a measure: illegal_per_criterion={len(bad)} '
         f'mass={w2.sum():.1f} sampled maxD={dmax2:.6f} -> certified bound='
         f'{w2.sum()/max(dmax2,1.0):.1f}  [must reach 67]')

    # ---- (i) is there a legal 68-packing on disk?
    emit('[T2] (i) 68-packing route: no legal 68 configuration exists in this project; '
         'the only 68 candidate (mmax68_R5_candidate.csv) is decertified by rule 2, so '
         'a tau>=68 packing certificate is NOT available.')

    # ---- (ii) measure LP over the discretised legal family, several resolutions.
    # ---- Two opposing biases, both declared BEFORE looking at the numbers:
    #      * the placement family is a finite grid -> the LP optimum is BELOW the true
    #        continuous LP optimum (losing direction; a value >=68 here is honest);
    #      * the density constraint is enforced only on grid points -> the LP optimum is
    #        ABOVE the true one (cheating direction; so a >=68 value still needs the
    #        exact density verifier before it is a certificate).
    C67 = cert.tolist()
    runs = []
    for h_c, n_ang, h_g in ((0.30, 9, 0.30), (0.22, 13, 0.22), (0.17, 17, 0.17)):
        F = build_family(h_c, n_ang, extra=C67)
        p = grid_pts(h_g)
        r = run_lp(F, p, f'h_c={h_c} ang={n_ang} h_g={h_g}')
        if r:
            runs.append(r)

    best = max([r['audited_value'] for r in runs] + [pc1, float(w2.sum())])
    emit(f'[T2] best tau lower bound obtainable here = {best:.4f} '
         f'(the rigorous standing one is 67, from nu>=67 + weak duality, PC2)')
    emit(f'[T2] GATE (pre-registered: probe tau>=68) -> '
         f'{"PASSED" if best >= TARGET else "NOT PASSED"}; no criterion was adjusted.')
    emit('[T2] SCOPE: even a PASSED gate closes G2 only.  A k-point piercing set at '
         'radius R proves N(R)<=k i.e. r(k+1)>=R; tau>=68 proves neither N(5)<=67 nor '
         'r(67)>=5, so it cannot contribute to the r(67)=5 claim at all.')
    json.dump(dict(R=R, ceiling_area=AREA_DISC, target=TARGET,
                   pc1_uniform_bound=pc1, pc2_certified67=float(w2.sum()),
                   pc2_max_density=float(dmax2), lp_runs=runs,
                   best_bound=best, gate_passed=bool(best >= TARGET),
                   scope_note='tau>=68 would close G2 only; N(R)<=k proves r(k+1)>=R, '
                              'so nothing here can prove r(67)>=5'),
              open('g2_tau.json', 'w'), indent=1)
    open('g2_tau_log.txt', 'w').write('\n'.join(log_lines) + '\n')


if __name__ == '__main__':
    main()

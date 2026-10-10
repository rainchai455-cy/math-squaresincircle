import numpy as np, json, sys, time, math, hashlib
from scipy.optimize import linprog

# =====================================================================
# G3 : search for 67 UNIT (side strictly 1) squares inside radius
#      R_TARGET, with continuous translation AND rotation.
#
#   R_SEARCH < R_TARGET on purpose: the search keeps a safety band so the
#   final configuration is verified strictly inside the declared target.
#
#   Legality is NEVER taken from this file's own linearised model: every
#   candidate is re-checked by the independent verifier g3_verify.py
#   (SAT support functions AND Sutherland-Hodgman clipping, all pairs,
#   all corners, SIDE = 1 exactly, TOL = 1e-9 pre-registered in gate0.py).
#
#   Outcomes are recorded as two distinct events:
#     FOUND_AND_VERIFIED  -> r(67) <= R_TARGET
#     BUDGET_EXHAUSTED    -> "not found" only.  NOT a non-existence proof,
#                            and it does not raise any lower bound.
# =====================================================================

OFF = np.array([[-.5, -.5], [-.5, .5], [.5, -.5], [.5, .5]])
N_SQ = 67


# ---------------- exact geometry (used only for measuring merit) ------
def corner_r(C, TH):
    c, s = np.cos(TH)[:, None], np.sin(TH)[:, None]
    px = C[:, 0:1] + OFF[:, 0]*c - OFF[:, 1]*s
    py = C[:, 1:2] + OFF[:, 0]*s + OFF[:, 1]*c
    return np.hypot(px, py), np.stack([px, py], axis=2)


def support(a, th):
    c, s = math.cos(th), math.sin(th)
    return 0.5*abs(a[0]*c + a[1]*s) + 0.5*abs(-a[0]*s + a[1]*c)


def support_split(a, th, eps=1e-4):
    """(kink_on_|w|, smooth_on_theta) for d h(a,theta).  When a is (nearly) an EDGE
    NORMAL of the square, h is at its MINIMUM over theta -> rotation can only widen
    -> the conservative model charges 0.5|omega| and must ALSO drop the smooth term,
    otherwise a spurious -+0.5*theta makes one rotation direction look free (this is
    exactly the retracted rate2.py error)."""
    c, s = math.cos(th), math.sin(th)
    au = a[0]*c + a[1]*s
    av = -a[0]*s + a[1]*c
    if min(abs(au), abs(av)) < eps:
        return 0.5, 0.0
    m = 0.5*math.copysign(1, au)*av - 0.5*math.copysign(1, av)*au
    return 0.0, m


def pair_data(C, TH, window=4.2):
    """one entry per pair that can touch: (i,j,gap,axis,sg,tied) using the exact
    max-over-4-edge-normals separation (>=0 proves disjoint)."""
    n = len(C)
    out = []
    for i in range(n):
        for j in range(i+1, n):
            d = C[j]-C[i]
            if d @ d > window:
                continue
            seen = {}
            for kk in (i, j):
                c, s = math.cos(TH[kk]), math.sin(TH[kk])
                for ax in ((c, s), (-s, c)):
                    key = (round(abs(ax[0]), 9), round(abs(ax[1]), 9))
                    t = abs(d[0]*ax[0]+d[1]*ax[1]) - support(np.array(ax), TH[i]) \
                        - support(np.array(ax), TH[j])
                    if key not in seen or t > seen[key][0]:
                        seen[key] = (t, np.array(ax))
            mx = max(v[0] for v in seen.values())
            tied = [v[1] for v in seen.values() if v[0] >= mx - 1e-9]
            out.append((i, j, mx, tied, d))
    return out


def metrics(C, TH, R):
    pd = pair_data(C, TH)
    ov = sum(max(0.0, -x[2]) for x in pd)
    mg = min((x[2] for x in pd), default=0.0)
    rr, _ = corner_r(C, TH)
    out = float(np.maximum(0.0, rr - R).sum())
    return ov, out, mg, float(rr.max())


# ---------------- conservative linearised feasibility step ------------
def build_lp(C, TH, R, rot, rho, wg, beta=0.0):
    n = len(C)
    D = 3 if rot else 2
    nv = D*n
    nw = n if rot else 0
    pd = pair_data(C, TH)
    rr, pp = corner_r(C, TH)
    A_c, b_c, kink = [], [], []
    # A_c rows hold P = grad(gap) in variable order [dc_i, dc_j, th, w]; the LP row is
    # then  -P.d - v_gap <= g   i.e.   gap + P.d >= -v_gap   (gaps must OPEN, not close).
    for (i, j, g, tied, d) in pd:
        row = np.zeros(nv + nw)
        if len(tied) >= 2:
            nn = np.zeros(2)
            for ax in tied:
                nn += ax*(1.0 if d @ ax >= 0 else -1.0)
            nn /= np.linalg.norm(nn)
            row[D*j] = nn[0];  row[D*j+1] = nn[1]
            row[D*i] = -nn[0]; row[D*i+1] = -nn[1]
            A_c.append(row); b_c.append(g); kink.append(None)
        else:
            ax = tied[0]
            sg = 1.0 if d @ ax >= 0 else -1.0
            row[D*j] = sg*ax[0];   row[D*j+1] = sg*ax[1]
            row[D*i] = -sg*ax[0];  row[D*i+1] = -sg*ax[1]
            ks = []
            if rot:
                for k in (i, j):
                    kk, mm = support_split(ax, TH[k])
                    row[D*k+2] -= mm
                    if kk:
                        row[nv+k] -= kk
                        ks.append(k)
            A_c.append(row); b_c.append(g); kink.append(ks)
    ng = len(A_c)
    A_r, b_r = [], []
    for k in range(n):
        for m in range(4):
            r = rr[k, m]
            if r < 1e-12:
                continue
            p = pp[k, m]
            c_, s_ = math.cos(TH[k]), math.sin(TH[k])
            dp = OFF[m][0]*np.array([-s_, c_]) + OFF[m][1]*np.array([-c_, -s_])
            row = np.zeros(nv + nw)
            row[D*k] = p[0]/r; row[D*k+1] = p[1]/r
            if rot:
                row[D*k+2] = (p @ dp)/r
            A_r.append(row); b_r.append(R - r)      # rate - vc <= R - r  (only outward
            #                                                        corners are pressed)
    nc = len(A_r)
    nvar = nv + nw + ng + nc
    M = np.zeros((ng+nc, nvar)); rhs = np.zeros(ng+nc)
    for e in range(ng):                       # g + rate >= -v   <=>  -rate - v <= g
        M[e, :nv+nw] = -A_c[e]
        M[e, nv+nw+e] = -1.0
        rhs[e] = b_c[e]
    for e in range(nc):                       # rate <= (r-R) + v <=> rate - v <= r-R
        M[ng+e, :nv+nw] = A_r[e]
        M[ng+e, nv+nw+ng+e] = -1.0
        rhs[ng+e] = b_r[e]
    obj = np.concatenate([np.zeros(nv+nw), np.full(ng, wg), np.ones(nc)])
    if beta:
        # compaction tie-break: gradient of beta*|c|^2/2 pulls every centre inward,
        # so a LEGAL-but-flat configuration keeps sliding toward the middle instead of
        # stalling at merit 0.  The contact rows are what stop it (jam -> no movement).
        for k in range(n):
            obj[D*k] += beta*C[k, 0]
            obj[D*k+1] += beta*C[k, 1]
    if rot:
        # FORCE w_k >= |theta_k|, otherwise the LP drives the kink charge to 0 and
        # rotation becomes free again (the retracted rate2.py error).
        Klink = np.zeros((2*n, nvar)); krhs = np.zeros(2*n)
        for k in range(n):
            Klink[2*k,   3*k+2] = 1.0;  Klink[2*k,   nv+k] = -1.0
            Klink[2*k+1, 3*k+2] = -1.0; Klink[2*k+1, nv+k] = -1.0
        M = np.vstack([M, Klink]); rhs = np.concatenate([rhs, krhs])
    th = rho/5.0
    bnds = []
    for k in range(n):
        bnds += [(-rho, rho), (-rho, rho)] + ([(-th, th)] if rot else [])
    bnds += [(0.0, None)]*nw + [(0.0, None)]*(ng+nc)
    return M, rhs, obj, bnds, nv, D


def try_relax(C, TH, R, rot=True, rho0=0.02, wg=300.0, iters=400, tl=1e-13,
              deadline=None, beta=0.0, stop_when_feasible=True):
    """LP-driven feasibility descent at FIXED container radius.
    beta>0 adds the compaction tie-break so that merit-flat jam-sliding is still
    progress (it banks slack at the rim instead of stalling)."""
    n = len(C)
    cur = metrics(C, TH, R)
    val = cur[0]+cur[1] + 0.5*beta*np.sum(C*C)
    rho = rho0
    it = 0
    for it in range(iters):
        if cur[0] <= tl and cur[1] <= tl:
            if stop_when_feasible or beta == 0.0:
                return C, TH, cur, it, True
        if deadline is not None and time.time() > deadline:
            return C, TH, cur, it, False
        M, rhs, obj, bnds, nv, D = build_lp(C, TH, R, rot, rho, wg, beta)
        res = linprog(obj, A_ub=M, b_ub=rhs, bounds=bnds, method='highs')
        if not res.success:
            rho *= 0.5
            if rho < 1e-9:
                return C, TH, cur, it, False
            continue
        d = res.x[:nv]
        if np.abs(d).max() < 1e-16:
            rho *= 0.5
            if rho < 1e-9:
                return C, TH, cur, it, False
            continue
        improved = False
        for sc in (1.0, 0.5, 0.2, 0.08):
            Cd = C + np.array([[d[D*k], d[D*k+1]] for k in range(n)])*sc
            THd = TH + np.array([d[D*k+2] for k in range(n)])*sc if rot else TH
            trial = metrics(Cd, THd, R)
            v2 = trial[0]+trial[1] + 0.5*beta*np.sum(Cd*Cd)
            if v2 < val - 1e-16:
                C, TH, cur, val, improved = Cd, THd, trial, v2, True
                break
        if improved:
            rho = min(rho*1.3, rho0)
        else:
            rho *= 0.35
            if rho < 1e-9:
                return C, TH, cur, it, False
    return C, TH, cur, iters, (cur[0] <= tl and cur[1] <= tl)


def sep_pair(ax, ay, at, bx, by, bt):
    """exact separation of two unit squares: max over the 4 edge normals (>=0 => disjoint)."""
    dx, dy = bx-ax, by-ay
    ca, sa, cb, sb = math.cos(at), math.sin(at), math.cos(bt), math.sin(bt)
    best = -1e9
    for (c, s) in ((ca, sa), (sa, -ca), (cb, sb), (sb, -cb)):
        ha = 0.5*abs(c*ca+s*sa) + 0.5*abs(c*sa-s*ca)
        hb = 0.5*abs(c*cb+s*sb) + 0.5*abs(c*sb-s*cb)
        v = abs(dx*c+dy*s) - ha - hb
        if v > best:
            best = v
    return best


def best_insert(C, TH, R, margin=1e-6, n_grid=80, angles=(0.0,), near=2.5):
    """place one more square at the hole with the largest exact minimum gap.
    Pairs farther than `near` are credited a gap of 0: still a rigorous LOWER bound
    (they are certainly disjoint, near > sqrt(2)), but evaluating a wider shell is
    what lets the scan DISTINGUISH a roomy hole from a barely-empty one."""
    n = len(C)
    best = (-9e9, None, None)
    lim = R - margin
    step = 2.0*lim/n_grid
    for gi in range(-n_grid, n_grid+1):
        x = gi*step
        if (abs(x)+0.5)**2 + 0.25 > lim**2:
            continue
        for gj in range(-n_grid, n_grid+1):
            y = gj*step
            if (abs(x)+0.5)**2 + (abs(y)+0.5)**2 > lim**2:
                continue
            if n == 0:
                return 9e9, np.array([x, y]), angles[0]
            dist = np.hypot(C[:, 0]-x, C[:, 1]-y)
            sel = np.where(dist < near)[0]
            for a in angles:
                g = 0.0
                for t in sel:
                    v = sep_pair(x, y, a, C[t, 0], C[t, 1], TH[t])
                    if v < g:
                        g = v
                if g > best[0]:
                    best = (g, np.array([x, y]), a)
    return best


def insert_sites(C, TH, R, margin=1e-6, n_grid=80, angles=(0.0,), near=2.5, k=8):
    """top-K distinct insertion sites (centres at least 0.5 apart), scored by the
    same rigorous lower bound on the exact min gap used by best_insert."""
    n = len(C)
    lim = R - margin
    step = 2.0*lim/n_grid
    scored = []
    for gi in range(-n_grid, n_grid+1):
        x = gi*step
        if (abs(x)+0.5)**2 + 0.25 > lim**2:
            continue
        for gj in range(-n_grid, n_grid+1):
            y = gj*step
            if (abs(x)+0.5)**2 + (abs(y)+0.5)**2 > lim**2:
                continue
            dist = np.hypot(C[:, 0]-x, C[:, 1]-y)
            sel = np.where(dist < near)[0]
            for a in angles:
                g = 0.0
                for t in sel:
                    v = sep_pair(x, y, a, C[t, 0], C[t, 1], TH[t])
                    if v < g:
                        g = v
                scored.append((g, x, y, a))
    scored.sort(key=lambda s: -s[0])
    out = []
    for (g, x, y, a) in scored:
        if all((x-o[1][0])**2 + (y-o[1][1])**2 > 0.25 for o in out):
            out.append((g, np.array([x, y]), a))
        if len(out) >= k:
            break
    return out

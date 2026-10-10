import numpy as np, math, time, json, sys
from scipy.optimize import linprog

# =====================================================================
# G3-GROW : re-parameterised search.  Container is FIXED at R_FIX and the
# 67 squares share one common side s, which is the variable being maximised.
#
#   scale-equivalence:  (67 squares of side s inside radius R_FIX)
#                   <-> (67 UNIT squares inside radius R_FIX / s)
#   so  s > 1  <=>  an enclosing radius < R_FIX, and the G3 target
#   R = 4.99 is exactly  s >= 5/4.99 = 500/499 = 1.002004008016...
#
#   This is a change of parameterisation, NOT an easier problem, and it buys
#   nothing by itself.  What it does buy: the descent direction is now
#   "grow the squares" instead of "squeeze the same rigid packing", which is a
#   different dynamics on the same feasible set.
#
#   Legality is still not decided here.  Any candidate is judged by
#   g3_verify (SAT + independent polygon clipping, all pairs, all corners) at
#   side s, and AGAIN after scaling by 1/s as unit squares at R_FIX/s.
#
#   Exact geometry, side-aware:
#     gap_ij = max over the 4 edge normals of  |dc.a| - s*(h_i(a)+h_j(a)),
#              h(a,th) = (|a.u|+|a.v|)/2      (gap >= 0 proves disjoint)
#     corner  = c + s*q,  q = ROT(th)@(+-1/2,+-1/2),  r = |corner|
#     dgap/ds = -(h_i+h_j)   (exact, gap is affine in s)
#     dr/ds   = (corner . q)/r
# =====================================================================

R_FIX = 5.0
N_SQ = 67
S_TARGET = 5.0/4.99            # 1.002004008016...
OFF = np.array([[-.5, -.5], [-.5, .5], [.5, -.5], [.5, .5]])
# The 1e-9 linear tolerance PRE-REGISTERED in gate0.py, re-imported here so the
# accept test below is the same statement the verifier will make (scaled by the
# side).  It is not a knob: a step is accepted only if the EXACT metrics of the
# whole state are inside the tolerance the independent verifier uses.
TOL_LINEAR = 1e-9
EPS_W = 1e-4                   # tie-break so the TR arm never pays for rotation
                               # the LP does not need (TR's feasible set contains
                               # T's exactly at omega=0; without this the TR arm
                               # picks gratuitous rotations, fails the exact
                               # check and dies -- an artifact, not a finding)



def supp_unit(a, th):
    """support of the UNIT square in direction a."""
    c, s = math.cos(th), math.sin(th)
    return 0.5*abs(a[0]*c + a[1]*s) + 0.5*abs(-a[0]*s + a[1]*c)


def supp_split_unit(a, th, eps=1e-4):
    """(kink coefficient on |omega|, smooth coefficient on theta) of d h.
    At an EDGE NORMAL the square is at its minimum width, so h has a V-kink:
    charge 0.5|omega| AND drop the smooth term (a spurious -+0.5*theta is the
    retracted rate2.py error)."""
    c, s = math.cos(th), math.sin(th)
    au = a[0]*c + a[1]*s
    av = -a[0]*s + a[1]*c
    if min(abs(au), abs(av)) < eps:
        return 0.5, 0.0
    return 0.0, 0.5*math.copysign(1, au)*av - 0.5*math.copysign(1, av)*au


def corners_s(C, TH, s):
    c, sn = np.cos(TH)[:, None], np.sin(TH)[:, None]
    qx = OFF[:, 0]*c - OFF[:, 1]*sn
    qy = OFF[:, 0]*sn + OFF[:, 1]*c
    px = C[:, 0:1] + s*qx
    py = C[:, 1:2] + s*qy
    return np.hypot(px, py), np.stack([px, py], axis=2), np.stack([qx, qy], axis=2)


def pair_data_s(C, TH, s, window=4.2):
    """(i, j, gap, tied axes, sigma, d) with sigma = h_i+h_j at the binding
    axis, i.e. the exact affine coefficient dgap/ds = -sigma."""
    n = len(C)
    out = []
    for i in range(n):
        for j in range(i+1, n):
            d = C[j]-C[i]
            if d @ d > window:
                continue
            seen = {}
            for kk in (i, j):
                c, s_ = math.cos(TH[kk]), math.sin(TH[kk])
                for ax in ((c, s_), (-s_, c)):
                    an = np.array(ax)
                    key = (round(abs(ax[0]), 9), round(abs(ax[1]), 9))
                    t = abs(d @ an) - s*(supp_unit(an, TH[i]) + supp_unit(an, TH[j]))
                    if key not in seen or t > seen[key][0]:
                        seen[key] = (t, an)
            mx = max(v[0] for v in seen.values())
            tied = [v[1] for v in seen.values() if v[0] >= mx - 1e-9]
            sg = sum(supp_unit(ax, TH[i]) + supp_unit(ax, TH[j]) for ax in tied)/len(tied)
            out.append((i, j, mx, tied, sg, d))
    return out


def metrics_s(C, TH, s, R=R_FIX):
    pd = pair_data_s(C, TH, s)
    ov = sum(max(0.0, -x[2]) for x in pd)
    mg = min((x[2] for x in pd), default=0.0)
    rr, _, _ = corners_s(C, TH, s)
    out = float(np.maximum(0.0, rr - R).sum())
    return ov, out, mg, float(rr.max())


def build_lp_s(C, TH, s, R, rot, rho, wg, rho_s):
    """one builder, two modes: rho_s>0 maximises the common side (grow),
    rho_s=0 is pure feasibility repair at fixed side."""
    n = len(C)
    D = 3 if rot else 2
    nv = D*n
    nw = n if rot else 0
    isv = nv + nw
    pd = pair_data_s(C, TH, s)
    rr, pp, qq = corners_s(C, TH, s)
    A_c, b_c, S_c = [], [], []
    for (i, j, g, tied, sg, d) in pd:
        row = np.zeros(nv + nw + 1)
        if len(tied) >= 2:
            # corner-corner: the true separating set is the single half-space with
            # the diagonal normal; rotation contributes 0 at first order (EXACT).
            nn = np.zeros(2)
            for ax in tied:
                nn += ax*(1.0 if d @ ax >= 0 else -1.0)
            nn /= np.linalg.norm(nn)
            sg = supp_unit(nn, TH[i]) + supp_unit(nn, TH[j])
            row[D*j] = nn[0];   row[D*j+1] = nn[1]
            row[D*i] = -nn[0];  row[D*i+1] = -nn[1]
        else:
            ax = tied[0]
            sg = 1.0 if d @ ax >= 0 else -1.0
            row[D*j] = sg*ax[0];  row[D*j+1] = sg*ax[1]
            row[D*i] = -sg*ax[0]; row[D*i+1] = -sg*ax[1]
            if rot:
                for k in (i, j):
                    kk, mm = supp_split_unit(ax, TH[k])
                    row[D*k+2] -= s*mm
                    if kk:
                        row[nv+k] -= s*kk
        row[isv] = -sg                                  # dgap/ds = -sigma
        A_c.append(row); b_c.append(g)
    ng = len(A_c)
    A_r, b_r, ncol = [], [], 0
    for k in range(n):
        for m in range(4):
            r = rr[k, m]
            if r < 1e-12:
                continue
            p = pp[k, m]; q = qq[k, m]
            c_, s_ = math.cos(TH[k]), math.sin(TH[k])
            dq = OFF[m][0]*np.array([-s_, c_]) + OFF[m][1]*np.array([-c_, -s_])
            row = np.zeros(nv + nw + 1)
            row[D*k] = p[0]/r; row[D*k+1] = p[1]/r
            if rot:
                row[D*k+2] = (p @ dq)/r
            row[isv] = (p @ q)/r                        # dr/ds
            A_r.append(row); b_r.append(R - r)
            ncol += 1
    nvar = nv + nw + 1 + ng + ncol
    M = np.zeros((ng+ncol, nvar)); rhs = np.zeros(ng+ncol)
    for e in range(ng):        # gap + P.d - sigma*ds + v >= 0
        M[e, :nv+nw+1] = -A_c[e]
        M[e, nv+nw+e] = -1.0
        rhs[e] = b_c[e]
    for e in range(ncol):      # r + grad.d + (dr/ds)ds - v <= R
        M[ng+e, :nvar-ng-ncol] = A_r[e]
        M[ng+e, nv+nw+ng+e] = -1.0
        rhs[ng+e] = b_r[e]
    obj = np.zeros(nvar)
    obj[nv+nw: nv+nw+ng] = wg
    obj[nv+nw+ng:] = 1.0
    obj[isv] = -1.0 if rho_s > 0 else 0.0
    if rot:
        obj[nv: nv+nw] = EPS_W                   # cheapest rotation, only if it pays
    if rot:                                            # w >= |theta|
        Kl = np.zeros((2*n, nvar)); kr = np.zeros(2*n)
        for k in range(n):
            Kl[2*k,   3*k+2] = 1.0;   Kl[2*k,   nv+k] = -1.0
            Kl[2*k+1, 3*k+2] = -1.0;  Kl[2*k+1, nv+k] = -1.0
        M = np.vstack([M, Kl]); rhs = np.concatenate([rhs, kr])
    th = rho/5.0
    bnds = []
    for k in range(n):
        bnds += [(-rho, rho), (-rho, rho)] + ([(-th, th)] if rot else [])
    bnds += [(0.0, None)]*nw + [(-rho_s, rho_s)] + [(0.0, None)]*(ng+ncol)
    return M, rhs, obj, bnds, nv, D, isv


def _apply(C, TH, x, D, n, sc, rot):
    Cd = C + np.array([[x[D*k], x[D*k+1]] for k in range(n)])*sc
    THd = TH + np.array([x[D*k+2] for k in range(n)])*sc if rot else TH
    return Cd, THd


def _proposals(C, TH, s, R, rot, rho, wg, rho_s):
    """LP proposals for the current state.  When rotation is ALLOWED, the
    translation-only optimum is offered as well.
    Reason: arm TR means "theta may change", not "theta must change".  Its own LP
    picks a translation that was computed to COMPLEMENT its rotations, so zeroing
    the rotation can leave an infeasible translation -- measured, TR then stalled
    at s=1.000025920 (806 LPs) where T reached 1.018882720 (695 LPs) with
    max|theta|=0, i.e. the TR arm was losing to a solver artifact and the A/B
    would have scored that as 'rotation does not help'.  With the T optimum in the
    candidate list, TR's search set contains T's at every step, so any TR gain is
    attributable to rotation and any tie is a real tie.  Cost: one extra LP (~7 ms).
    """
    M, rhs, obj, bnds, nv, D, isv = build_lp_s(C, TH, s, R, rot, rho, wg, rho_s)
    res = linprog(obj, A_ub=M, b_ub=rhs, bounds=bnds, method='highs')
    out = [(res, D, isv, rot)] if res.success else []
    if rot:
        M, rhs, obj, bnds, nv, D, isv = build_lp_s(C, TH, s, R, False, rho, wg, rho_s)
        res = linprog(obj, A_ub=M, b_ub=rhs, bounds=bnds, method='highs')
        if res.success:
            out.append((res, D, isv, False))
    return out


def repair_s(C, TH, s, R=R_FIX, rot=True, rho0=0.02, wg=300.0, iters=300,
             tl=1e-13, deadline=None):
    """drive ov,out -> 0 at FIXED side s."""
    cur = metrics_s(C, TH, s, R)
    val = cur[0]+cur[1]
    n = len(C)
    rho = rho0
    lps = 0
    for it in range(iters):
        if cur[0] <= tl and cur[1] <= tl:
            return C, TH, cur, lps, True
        if deadline is not None and time.time() > deadline:
            return C, TH, cur, lps, False
        props = _proposals(C, TH, s, R, rot, rho, wg, 0.0)
        lps += len(props) or 1
        if not props:
            rho *= 0.4
            if rho < 1e-9:
                return C, TH, cur, lps, False
            continue
        improved = False
        for res, D, isv, use_rot in props:
            if np.abs(res.x[:D*n]).max() < 1e-16 and (not use_rot or
                                                      np.abs(res.x[2*n:3*n]).max() < 1e-16):
                continue
            for sc in (1.0, 0.5, 0.2, 0.08):
                Cd, THd = _apply(C, TH, res.x, D, n, sc, use_rot)
                t2 = metrics_s(Cd, THd, s, R)
                if t2[0]+t2[1] < val - 1e-16:
                    C, TH, cur, val, improved = Cd, THd, t2, t2[0]+t2[1], True
                    break
            if improved:
                break
        if improved:
            rho = min(rho*1.3, rho0)
        else:
            rho *= 0.35
            if rho < 1e-9:
                return C, TH, cur, lps, False
    return C, TH, cur, lps, (cur[0] <= tl and cur[1] <= tl)


def grow(C, TH, s, R=R_FIX, rot=True, rho0=0.02, rho_s=0.002, wg=300.0,
         iters=300, polish=25, ftol=None, deadline=None, verbose=False):
    """maximise the common side while keeping the state inside the PRE-REGISTERED
    tolerance under the EXACT (not linearised) metrics.

    Why the exact check and not the LP value: the containment rows linearise
    r(s)=|c+s q|, which is CONVEX in s, so its tangent UNDER-states the true
    corner radius -> the LP is systematically optimistic about growth.  A step is
    therefore accepted only when the exact side-aware metrics confirm it; if the
    curvature made it slightly illegal a SHORT repair is tried before the step is
    rejected.  Accepting at ov,out <= s*1e-9 (the verifier's own per-pair margin,
    measured on the WHOLE state each time, so nothing accumulates) instead of
    1e-13 is what keeps the loop from burning 30 repair LPs per step.  Legality is
    still judged by g3_verify, never here.
    Returns (C, TH, s, metrics, iters, status) with status in
      {'JAMMED'   - growth LP says ds<=0 at this s (first-order ceiling),
       'ceiling'  - no growth step passed the exact check any more,
       'feasible' - grew and stayed feasible, budget of iterations used up,
       'stalled'  - could not be made feasible}"""
    ft = s*TOL_LINEAR if ftol is None else ftol
    C, TH, cur, it_r, ok = repair_s(C, TH, s, R, rot, rho0, wg, iters, deadline=deadline)
    if not ok:
        return C, TH, s, cur, it_r, 'stalled'
    n = len(C)
    tot = it_r
    rho_s_cur = rho_s
    status = 'feasible'
    for it in range(iters):
        if deadline is not None and time.time() > deadline:
            return C, TH, s, cur, tot, status
        props = _proposals(C, TH, s, R, rot, rho0, wg, rho_s_cur)
        tot += max(1, len(props))
        if not props:
            rho_s_cur *= 0.6
            if rho_s_cur < 1e-7:
                return C, TH, s, cur, tot, 'ceiling'
            continue
        if all(res.x[isv] <= 1e-12 for res, D, isv, ur in props):
            return C, TH, s, cur, tot, 'JAMMED'
        took = False
        dsc = (1.0, 0.5, 0.2, 0.05)
        tried = []
        for res, D, isv, use_rot in props:
            ds = res.x[isv]
            if ds <= 1e-12:
                continue
            cands = [(sc, False) for sc in dsc]
            if use_rot:
                cands += [(sc, True) for sc in dsc]     # omega-zeroed projection too
            for sc, zero_th in cands:
                x = res.x
                if zero_th:
                    x = x.copy()
                    for k in range(n):
                        x[D*k+2] = 0.0
                Cd, THd = _apply(C, TH, x, D, n, sc, use_rot)
                s2 = s + sc*ds
                t2 = metrics_s(Cd, THd, s2, R)
                tried.append((Cd, THd, s2, t2, use_rot))
                if t2[0] <= ft and t2[1] <= ft:
                    C, TH, s, cur, took = Cd, THd, s2, t2, True
                    break
            if took:
                break
        if not took:
            repair_left = 2
            for (Cd, THd, s2, t2, ur) in tried:
                if repair_left <= 0 or t2[0] + t2[1] > 1e-3:
                    continue
                repair_left -= 1
                C3, T3, m3, it3, ok3 = repair_s(Cd, THd, s2, R, rot, rho0, wg, polish,
                                                deadline=deadline)
                tot += it3
                if ok3:
                    C, TH, s, cur, took = C3, T3, s2, m3, True
                    break
        if not took:
            rho_s_cur *= 0.6
            if rho_s_cur < 1e-7:
                return C, TH, s, cur, tot, 'ceiling'
        elif verbose and it % 25 == 0:
            print(f'        grow: it={it} s={s:.9f} rho_s={rho_s_cur:.1e}')
    return C, TH, s, cur, tot, status


def sep_pair_s(ax, ay, at, bx, by, bt, s):
    """exact separation of two squares of COMMON side s: max over the 4 edge
    normals of |d.n| - s*(h_i+h_j).  >= 0 proves disjoint (SAT for convex polys)."""
    dx, dy = bx-ax, by-ay
    ca, sa, cb, sb = math.cos(at), math.sin(at), math.cos(bt), math.sin(bt)
    best = -1e9
    for (c, sn) in ((ca, sa), (sa, -ca), (cb, sb), (sb, -cb)):
        ha = 0.5*abs(c*ca+sn*sa) + 0.5*abs(c*sa-sn*ca)
        hb = 0.5*abs(c*cb+sn*sb) + 0.5*abs(c*sb-sn*cb)
        v = abs(dx*c+dy*sn) - s*(ha+hb)
        if v > best:
            best = v
    return best


def corner_ok(x, y, a, s, R, margin):
    """all 4 corners of a side-s square strictly inside radius R-margin."""
    c, sn = math.cos(a), math.sin(a)
    for (ox, oy) in ((-.5, -.5), (-.5, .5), (.5, -.5), (.5, .5)):
        px = x + s*(ox*c - oy*sn)
        py = y + s*(ox*sn + oy*c)
        if px*px + py*py > (R-margin)**2:
            return False
    return True


def rim_order(C, TH, s):
    """squares sorted by their own maximum corner radius, outermost first -- the
    boundary-contact pool that remove-k is allowed to touch."""
    rr, _, _ = corners_s(C, TH, s)
    return np.argsort(-rr.max(axis=1))


def refill(C, TH, s, holes, angles, R=R_FIX, margin=1e-9, offsets=((0.0, 0.0),),
           allow_pen=0.05, diag=None):
    """put the removed squares back into the holes that remove-k just opened.
    `holes` are the freed CENTRE POSITIONS (directions), `offsets` is a list of
    (radial, tangential) displacements applied to each: growth moves the whole rim
    outward, so the refill must scan a radial range, not just the old position.
    `offsets` and `angles` come from the SHARED pre-drawn schedule, so both arms
    are offered exactly the same sites; arm T is simply handed (0.0,) for angles.

    A site is taken even when it slightly OVERLAPS (down to allow_pen*s): making
    room for an inserted square is the polish stage's job, and requiring gap >= 0
    up front refused 100% of refills in g3_probe_rim (the dilated bulk fills the
    annulus, so no hole is a-priori free).  What is never allowed is a square
    outside the container -- corner_ok, because no descent can fix that at fixed s.
    ok=False only when no candidate site is inside the rim at all."""
    Cl, TL = list(C), list(TH)
    for pos in holes:
        pos = np.asarray(pos, dtype=float)
        dir_r = pos/max(1e-12, float(np.hypot(*pos)))
        dir_t = np.array([-dir_r[1], dir_r[0]])
        best = (-9e9, None, None)
        for (dr, dt) in offsets:
            p = pos + dr*dir_r + dt*dir_t
            for a in angles:
                if not corner_ok(p[0], p[1], a, s, R, margin):
                    continue
                g = 9e9
                for (q, t) in zip(Cl, TL):
                    v = sep_pair_s(p[0], p[1], a, q[0], q[1], t, s)
                    if v < g:
                        g = v
                if g > best[0]:
                    best = (g, np.array(p), a)
        if diag is not None:
            diag.append(float(best[0]))
        if best[1] is None or best[0] < -allow_pen*s:
            return np.array(Cl), np.array(TL), False
        Cl.append(best[1]); TL.append(best[2])
    return np.array(Cl), np.array(TL), True


if __name__ == '__main__':
    print(f'[G] controls for the fixed-R / variable-side solver.  '
          f'R_FIX={R_FIX}  S_TARGET(=5/4.99)={S_TARGET:.12f}')
    # ---- control 1: a single square.  analytic ceiling s = R*sqrt(2)
    C = np.array([[0.0, 0.0]]); TH = np.array([0.0])
    Cf, THf, s, m, it, st = grow(C, TH, 1.0, rot=False, iters=400, rho_s=0.05)
    print(f'[G] C1 one square   : s={s:.9f}  analytic max = {R_FIX*math.sqrt(2):.9f}  '
          f'({st}, {it} LPs)  ov={m[0]:.2e} out={m[1]:.2e}')

    # ---- control 2: 2x2 block whose centres are 1.05 apart, TRANSLATION FROZEN
    # ---- (rho0 -> 0) so that only s moves.  Then the only ceiling is the face
    # ---- contact: gap = 1.05 - s, so the solver must stop at EXACTLY 1.05.  This
    # ---- is the sharp test of the dgap/ds = -sigma coefficient (a wrong sigma
    # ---- would overshoot or stall).  Unfrozen, the block would simply drift
    # ---- apart and 1.05 would not be a ceiling at all.
    C = np.array([[-.525, -.525], [.525, -.525], [-.525, .525], [.525, .525]])
    TH = np.zeros(4)
    Cf, THf, s, m, it, st = grow(C, TH, 1.0, rot=False, rho0=1e-12, iters=400, rho_s=0.005)
    print(f'[G] C2 frozen 2x2, centre-dist 1.05: s={s:.9f}  analytic max = 1.050000000 '
          f'({st}, {it} LPs) ov={m[0]:.2e} out={m[1]:.2e}')

    # ---- control 3: NEGATIVE control.  the certified 67 at s=1 IS a first-order
    # ---- jam (independent squeeze LP measured t*=0), so growth must report 0.
    rows = [l.split() for l in open('packing67.txt') if l.strip() and not l.startswith('#')]
    C67 = np.array([[(int(r[1])+int(r[3]))/4.0, (int(r[2])+int(r[4]))/4.0] for r in rows])
    TH0 = np.zeros(len(C67))
    m = metrics_s(C67, TH0, 1.0)
    print(f'[G] C3 certified 67 @s=1: ov={m[0]:.3e} out={m[1]:.3e} minGap={m[2]:+.1e} '
          f'R_impl={m[3]:.12f}   (expect JAMMED: ds<=0)')
    for rot in (False, True):
        Cf, THf, s, mm, it, st = grow(C67.copy(), TH0.copy(), 1.0,
                                      rot=rot, iters=200, rho_s=0.002)
        print(f'[G] C3 {"TR" if rot else "T "} : s={s:.12f} status={st} it={it} '
              f'ov={mm[0]:.2e} out={mm[1]:.2e}')

    # ---- control 4: POSITIVE control.  the SAME config at s=0.99 is strictly
    # ---- interior, so growth MUST lift it.  Its own ceiling is s=1.000 (the
    # ---- certified packing is exactly at radius 5.0), therefore the lifted s
    # ---- must land at ~1.0 and NOT materially exceed it -- growing past 1.0
    # ---- here would mean the solver is reporting an infeasible state, since
    # ---- s>1 at R=5 is precisely the result we are hunting for.
    for rot in (False, True):
        Cf, THf, s, mm, it, st = grow(C67.copy(), TH0.copy(), 0.99,
                                      rot=rot, iters=600, rho_s=0.002)
        print(f'[G] C4 {"TR" if rot else "T "} from s=0.99: s={s:.12f} status={st} '
              f'it={it} ov={mm[0]:.2e} out={mm[1]:.2e} minGap={mm[2]:+.2e}  '
              f'-> implied unit-square radius = {R_FIX/s:.9f} (expect >= 5.0)')

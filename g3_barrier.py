import numpy as np, math, time, json
from scipy.optimize import linprog
import g3_grow as GR

# =====================================================================
# BARRIER MEASUREMENT, not a search.
#
# The paired A/B (g3_ab2) showed 18/18 trials ending at ov ~ 1e-14 with out
# STUCK at exactly 8.016e-02.  8.016e-02 = 8 x 0.010020 = 8 x 5*(S_TARGET-1):
# the overshoot of the 8 rim-pinned corners of the PURE DILATION.  So the
# polish's answer to "re-seat the 8 ring squares" was to undo the insertion and
# walk back to the dilated configuration.  Two explanations have to be
# separated before anything is called a barrier:
#   (i) GEOMETRIC: at s_target no inward move of a pinned square is possible
#       without opening a hole in the interior, because the certificate is
#       contact-saturated (every gap is exactly 0 on the contact set).
#   (ii) OBJECTIVE: repair_s weights overlap slack wg=300 against containment
#       slack 1, so the LP is *defined* to finish overlaps first and refuse any
#       trade that pays overlap to buy overshoot.  A weight artifact.
# Sweep wg over 5 orders of magnitude: if the frontier moves with wg, it is (ii)
# and the fix is a schedule; if out stays frozen at every wg, it is (i).
#
# PC (positive control, per the rule that a negative result needs a known-feasible
# instance run through the SAME code): two side-s squares on the +x rim, one
# overlapping the other and both overshooting.  A legal resolution exists and is
# written out below, so repair MUST reach (0,0) here; if it does not, this script
# measures nothing about the 67.
# =====================================================================

S = GR.S_TARGET
R = GR.R_FIX
OUT0 = None


def load67():
    rows = [l.split() for l in open('packing67.txt') if l.strip()
            and not l.startswith('#')]
    C = np.array([[(int(r[1])+int(r[3]))/4.0, (int(r[2])+int(r[4]))/4.0]
                  for r in rows])
    return C, np.zeros(len(C))


def hard_contact_lp(C, TH, s, R, rot, rho=0.05, eps=0.0):
    """minimise the total corner overshoot with the CONTACTS AS HARD
    CONSTRAINTS (gap >= -eps), side fixed, inside a trust region.

    This is the question the wg-sweep could not answer: repair_s descends on
    ov+out with l1 slacks, so a step that buys 0.010 of overshoot at the price
    of 0.035 of penetration raises ov+out and is refused -- the search rule
    conflates 'no descent of the penalty' with 'no feasible rearrangement'.
    Here penetration is not in the objective at all: we ask whether ANY
    first-order motion can pull the 8 rim corners in while keeping every one of
    the 196 contacts unbroken.  The linearisation is the same one the solver
    uses (pair_data_s tied axes, dgap/ds, dr/ds), but the slack structure is
    ours, so a bug here shows up as an implausible answer rather than confirming
    the search.  Returns (predicted overshoot, exact metrics after the step,
    number of LP rows).
    """
    n = len(C)
    D = 3 if rot else 2
    nv = D*n
    nw = n if rot else 0
    isv = nv + nw                       # ds column, pinned to 0 by its bounds
    pd = GR.pair_data_s(C, TH, s)
    rr, pp, qq = GR.corners_s(C, TH, s)
    ng = len(pd)
    ncol = 4*n
    nvar = isv + 1 + ncol
    M = np.zeros((ng+ncol+2*n, nvar))
    rhs = np.zeros(ng+ncol+2*n)
    for e, (i, j, g, tied, sg, d) in enumerate(pd):
        row = np.zeros(nvar)
        if len(tied) >= 2:
            nn = np.zeros(2)
            for ax in tied:
                nn += ax*(1.0 if d @ ax >= 0 else -1.0)
            nn /= np.linalg.norm(nn)
            row[D*j] = nn[0];   row[D*j+1] = nn[1]
            row[D*i] = -nn[0];  row[D*i+1] = -nn[1]
        else:
            ax = tied[0]
            sg = 1.0 if d @ ax >= 0 else -1.0
            row[D*j] = sg*ax[0];  row[D*j+1] = sg*ax[1]
            row[D*i] = -sg*ax[0]; row[D*i+1] = -sg*ax[1]
            if rot:
                for k in (i, j):
                    kk, mm = GR.supp_split_unit(ax, TH[k])
                    row[D*k+2] -= s*mm
                    if kk:
                        row[nv+k] -= s*kk
        M[e, :nvar-ncol] = -row[:nvar-ncol]
        rhs[e] = g + eps                        # gap >= -eps
    e = 0
    for k in range(n):
        for m in range(4):
            r = rr[k, m]
            c_, s_ = math.cos(TH[k]), math.sin(TH[k])
            dq = GR.OFF[m][0]*np.array([-s_, c_]) + GR.OFF[m][1]*np.array([-c_, -s_])
            row = np.zeros(nvar)
            row[D*k] = pp[k, m, 0]/r; row[D*k+1] = pp[k, m, 1]/r
            if rot:
                row[D*k+2] = (pp[k, m] @ dq)/r
            M[ng+e, :nvar-ncol] = row[:nvar-ncol]
            M[ng+e, isv+1+e] = -1.0             # v_e >= r_new - R
            rhs[ng+e] = R - r
            e += 1
    if rot:                                     # w >= |theta|
        for k in range(n):
            M[ng+ncol+2*k,     3*k+2] = 1.0;   M[ng+ncol+2*k,     nv+k] = -1.0
            M[ng+ncol+2*k+1,   3*k+2] = -1.0;  M[ng+ncol+2*k+1,   nv+k] = -1.0
    obj = np.zeros(nvar)
    obj[isv+1:] = 1.0
    bnds = []
    for k in range(n):
        bnds += [(-rho, rho), (-rho, rho)] + ([(-rho/5.0, rho/5.0)] if rot else [])
    bnds += [(0.0, None)]*nw + [(0.0, 0.0)] + [(0.0, None)]*ncol
    res = linprog(obj, A_ub=M, b_ub=rhs, bounds=bnds, method='highs')
    if not res.success:
        return None
    x = res.x
    Cd = C + np.array([[x[D*k], x[D*k+1]] for k in range(n)])
    THd = TH + np.array([x[D*k+2] for k in range(n)]) if rot else TH
    return (float(obj[isv+1:] @ x[isv+1:]), GR.metrics_s(Cd, THd, s, R),
            M.shape, x)


if __name__ == '__main__':
    C0, TH0 = load67()
    Cd = C0*S
    m = GR.metrics_s(Cd, TH0, S)
    print(f'[BR] dilated 67 at s={S:.12f} (=5/4.99): ov={m[0]:.3e} out={m[1]:.6f} '
          f'minGap={m[2]:+.3e} R_impl={m[3]:.9f}')
    OUT0 = m[1]

    # ---- A) saturation count ------------------------------------------------
    pd = GR.pair_data_s(Cd, TH0, S)
    cont = [x for x in pd if x[2] <= 1e-9]
    ties = sum(len(x[3]) for x in cont)
    n = len(Cd)
    print(f'[BR] A) pairs inside the window={len(pd)}  contacts (gap<=1e-9)={len(cont)}'
          f'  tied-axis rows={ties}  DOF=2N={2*n}  contacts/DOF={ties/(2*n):.2f}')
    rr, _, _ = GR.corners_s(Cd, TH0, S)
    over = np.maximum(0.0, rr - R)
    print(f'[BR]    corners over R: {(over.ravel() > 1e-12).sum()} of {4*n}, '
          f'sum={over.sum():.9f}, per-square max={over.max():.9f}')
    print(f'[BR]    squares carrying the overshoot: '
          f'{sorted(np.where(over.max(axis=1) > 1e-12)[0].tolist())}')

    # ---- PC) known-feasible two-square chain --------------------------------
    # side S, container R.  Legal resolution written out explicitly:
    #   c2 = (R - S/sqrt(2)) , c1 = c2 - S  ->  gap exactly 0, corner exactly R.
    # Start 0.05 overlapped and 0.01 overshooting; a descent exists (lengthen the
    # chain by moving c1 inward), so repair MUST clean it.
    c2 = math.sqrt(R*R - (S/2.0)**2) - S/2.0 + 0.01   # top-right corner 0.01 over R
    c1 = c2 - S + 0.05                                # 0.05 deep into its neighbour
    Cp = np.array([[c1, 0.0], [c2, 0.0]]); THp = np.zeros(2)
    mp = GR.metrics_s(Cp, THp, S)
    print(f'[BR] PC start: ov={mp[0]:.4f} out={mp[1]:.6f}  (resolution exists: '
          f'c2 -> {c2-0.01:.6f}, c1 -> {c2-0.01-S:.6f})')
    for rot in (False, True):
        t0 = time.time()
        A, B, mm, it, ok = GR.repair_s(Cp.copy(), THp.copy(), S, rot=rot, rho0=0.05,
                                       wg=300.0, iters=400)
        print(f'[BR] PC {"TR" if rot else "T "} -> ov={mm[0]:.2e} out={mm[1]:.2e} '
              f'ok={ok} LPs={it} ({time.time()-t0:.1f}s)  centres={A[:, 0].round(5)}')

    # ---- C) wg sweep on the 67 ----------------------------------------------
    print(f'[BR] sweep: can ANY weight trade reduce out below {OUT0:.6f}?  '
          f'(rho0=0.05, 300 iters)')
    rows = []
    for rot in (False, True):
        for wg in (300.0, 30.0, 3.0, 1.0, 0.3, 0.03, 0.003):
            t0 = time.time()
            A, B, mm, it, ok = GR.repair_s(Cd.copy(), TH0.copy(), S, rot=rot,
                                           rho0=0.05, wg=wg, iters=300)
            dev = float(np.abs(A - Cd).max())
            line = (f'[BR] {"TR" if rot else "T "} wg={wg:8.3f} -> ov={mm[0]:.3e} '
                    f'out={mm[1]:.6f} ({mm[1]/OUT0:6.1%} of dilation) '
                    f'moved_by={dev:.5f} LPs={it} ok={ok} ({time.time()-t0:.1f}s)')
            print(line); rows.append(line)
    # ---- D) HARD-CONTACT LP: is there ANY first-order motion that pulls the 8
    # ---- rim corners in while every contact stays unbroken?  (the wg-sweep above
    # ---- cannot answer this: its l1 penalty refuses the trade at every scale.)
    # PC for THIS lp on the 2-square chain: out must drop by ~0.01 with gap kept.
    two = np.array([[c1, 0.0], [c2, 0.0]])
    r2 = hard_contact_lp(two, np.zeros(2), S, R, False, rho=0.15, eps=0.0)
    m2s = GR.metrics_s(two, np.zeros(2), S)
    print(f'[BR] PC-LP 2-chain (rho=0.15, big enough for the known resolution): '
          f'start out={m2s[1]:.6f} -> pred={r2[0]:.6f} exact={r2[1][1]:.6f} '
          f'ov={r2[1][0]:.2e}  centres={np.round(r2[3][:4], 5)}')
    r2t = hard_contact_lp(two, np.zeros(2), S, R, True, rho=0.15, eps=0.0)
    # TR's pred=0 but exact_out=0.002302 is NOT a failure: the containment rows
    # linearise r=|c+s q|, which is CONVEX, so the tangent sits BELOW the true
    # radius.  The LP is allowed to be optimistic; the accept tests in the search
    # and the F-scan below are what keep the verdict honest.
    print(f'[BR] PC-LP 2-chain TR: pred={r2t[0]:.6f} exact={r2t[1][1]:.6f} '
          f'ov={r2t[1][0]:.2e}')

    print(f'[BR] D0) rho-independence at eps=0.  The dilated state already has '
          f'every gap >= 0, so the hard rows are a CONE: if a descent direction '
          f'existed the box size would not matter.  Check it anyway.')
    for rot in (False, True):
        for rho in (0.002, 0.02, 0.2, 1.0):
            o = hard_contact_lp(Cd, TH0, S, R, rot, rho=rho, eps=0.0)
            print(f'[BR]     {"TR" if rot else "T "} rho={rho:6.3f} -> '
                  f'pred_out={o[0]:.9f} ({o[0]/OUT0:8.2%})')
    print(f'[BR] D) hard-contact LP on the dilated 67 at s={S:.12f}, '
          f'out0={OUT0:.6f}')
    for rot in (False, True):
        for eps in (0.0, 1e-4, 1e-3, 5e-3, 0.01, 0.02):
            t0 = time.time()
            out = hard_contact_lp(Cd, TH0, S, R, rot, rho=0.05, eps=eps)
            if out is None:
                print(f'[BR]    {"TR" if rot else "T "} eps={eps:.1e} INFEASIBLE')
                continue
            pred, ex, shape, x = out
            moved = float(np.abs(x[:len(Cd)*(3 if rot else 2)]).max())
            print(f'[BR]    {"TR" if rot else "T "} eps={eps:.1e} rows={shape} '
                  f'pred_out={pred:.6f} ({pred/OUT0:6.1%}) exact_out={ex[1]:.6f} '
                  f'exact_ov={ex[0]:.3e} max_step={moved:.4f} ({time.time()-t0:.1f}s)')
    # ---- E) is the rigidity specific to s_target, or does the whole branch fail
    # ---- for every side > 1?  The contact cone is scale-covariant, so the honest
    # ---- prediction is 'every side' -- but that is a prediction, so measure it.
    print('[BR] E) same hard-contact LP along the dilation ray (eps=0):')
    for s_t in (1.0002, 1.001, S, 1.005, 1.012, 1.020):
        Cx = C0*s_t
        ox = GR.metrics_s(Cx, TH0, s_t)[1]
        if ox <= 0:
            print(f'[BR]     s={s_t:.9f} out=0 -> already contained, nothing to ask')
            continue
        line = f'[BR]     s={s_t:.9f} out0={ox:.6f} '
        for rot in (False, True):
            o = hard_contact_lp(Cx, TH0, s_t, R, rot, rho=0.02, eps=0.0)
            line += f'{"TR" if rot else "T "}={o[0]/ox:7.2%} '
        print(line + ' (fraction of the overshoot a first-order motion can remove)')
    # ---- F) INDEPENDENT check of the rigidity claim, with the EXACT metric and no
    # ---- linearisation at all: scan directions and look for a move that lowers the
    # ---- overshoot WITHOUT increasing the overlap.  This also catches the escapes
    # ---- the contact-linearisation cannot see (sliding along a tie at second
    # ---- order), which is the honest caveat on D/E.
    ov0 = m[0]
    n = len(Cd)
    rng = np.random.default_rng(20261009)
    cand = []
    for k in range(n):
        for ax in (0, 1):
            d = np.zeros((n, 2)); d[k, ax] = 1.0
            cand.append((f'sq{k}.{"xy"[ax]}', d, np.zeros(n)))
            d = np.zeros((n, 2)); d[k, ax] = -1.0
            cand.append((f'sq{k}.-{"xy"[ax]}', d, np.zeros(n)))
        cand.append((f'sq{k}.rot', np.zeros((n, 2)),
                     np.array([1.0 if j == k else 0.0 for j in range(n)])))
    for j in range(300):
        g = rng.standard_normal((n, 2)); g /= np.abs(g).max()
        cand.append((f'rand{j}', g, rng.standard_normal(n)/np.sqrt(n)))
    xlp = hard_contact_lp(Cd, TH0, S, R, True, rho=0.05, eps=0.0)[3]
    cand.append(('LP-eps0', np.array([[xlp[3*k], xlp[3*k+1]] for k in range(n)]),
                 np.array([xlp[3*k+2] for k in range(n)])))
    hits, best = 0, (None, OUT0, None)
    for tag, dC, dT in cand:
        for t in (1e-1, 1e-2, 1e-3, 1e-4):
            mm = GR.metrics_s(Cd + t*dC, TH0 + t*dT, S)
            if mm[0] <= ov0 + 1e-16 and mm[1] < OUT0 - 1e-15:
                hits += 1
                if mm[1] < best[1]:
                    best = (f'{tag}@t={t:g}', mm[1], mm[0])
    print(f'[BR] F) exact directional scan: {len(cand)} directions x 4 step sizes = '
          f'{len(cand)*4} states, ov0={ov0:.3e} out0={OUT0:.6f} -> '
          f'improving (ov not increased AND out lower): {hits}; '
          f'best={best}')
    json.dump(dict(S_target=S, out_dilation=OUT0, contacts=len(cont),
                   tied_rows=ties, dof=2*n, log=rows),
              open('g3_barrier.json', 'w'), indent=1)

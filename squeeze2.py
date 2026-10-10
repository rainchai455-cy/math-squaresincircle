import numpy as np, json, sys
from scipy.optimize import linprog
import squeeze_lp as S

OFF, N, C = S.OFF, S.N, S.C
COL = S.COL


def all_rows(Cw, TH, Rn, rot):
    """Full linearised model: EVERY close pair gets its (frozen-axis) separation row,
    EVERY corner gets its containment row.  No window -> no unguarded pairs.
    Frozen axis a is sufficient for disjointness: |dc.a| >= h_i(a)+h_j(a) => separate."""
    D = 3 if rot else 2
    nv = D*N
    gl = S.gap_list(Cw, TH)
    RR = S.corner_r(Cw, TH)
    P = S.corner_pts(Cw, TH)
    g_rows, c_rows = [], []
    for (i, j, g, a, sg) in gl:
        row = {}
        for idx, val in ((D*j, sg*a[0]), (D*j+1, sg*a[1]),
                         (D*i, -sg*a[0]), (D*i+1, -sg*a[1])):
            row[idx] = row.get(idx, 0) + val
        if rot:
            row[D*i+2] = row.get(D*i+2, 0) - S.dsupport(a, i, TH)
            row[D*j+2] = row.get(D*j+2, 0) - S.dsupport(a, j, TH)
        g_rows.append((row, g))
    for k in range(N):
        for m in range(4):
            r = RR[k, m]
            if r < 1e-12:
                continue
            p = P[k, m]
            c_, s_ = np.cos(TH[k]), np.sin(TH[k])
            dp = OFF[m][0]*np.array([-s_, c_]) + OFF[m][1]*np.array([-c_, -s_])
            row = {D*k: p[0]/r, D*k+1: p[1]/r}
            if rot:
                row[D*k+2] = (p @ dp)/r
            c_rows.append((row, r - Rn))
    ng, nr = len(g_rows), len(c_rows)
    n = nv + ng + nr
    M = np.zeros((ng+nr, n))
    rhs = np.zeros(ng+nr)
    # gap row:   g + grad.d >= -vg   <=>   -grad.d - vg <= g
    for e, (row, g) in enumerate(g_rows):
        for idx, v in row.items():
            M[e, idx] = -v
        M[e, nv+e] = -1.0
        rhs[e] = g
    # corner row:  r + grad.d <= Rn + vr   <=>   grad.d - vr <= Rn - r
    for e, (row, excess) in enumerate(c_rows):
        rr = ng+e
        for idx, v in row.items():
            M[rr, idx] = v
        M[rr, nv+ng+e] = -1.0
        rhs[rr] = -excess
    obj = np.concatenate([np.zeros(nv), np.full(ng, 20.0), np.ones(nr)])
    return M, rhs, obj, nv, ng, nr, D


def lp_solve(Cw, TH, Rn, rot, rho, th_rho):
    M, rhs, obj, nv, ng, nr, D = all_rows(Cw, TH, Rn, rot)
    if rot:
        b = []
        for k in range(N):
            b += [(-rho, rho), (-rho, rho), (-th_rho, th_rho)]
    else:
        b = [(-rho, rho)]*nv
    b += [(0.0, None)]*(ng+nr)
    res = linprog(obj, A_ub=M, b_ub=rhs, bounds=b, method='highs')
    return res, nv, ng, nr, D


def merit(Cw, TH, Rn):
    """exact violation: total overlap penetration + total corner excess past Rn."""
    gl = S.gap_list(Cw, TH)
    ov = sum(max(0.0, -x[2]) for x in gl)
    RR = S.corner_r(Cw, TH)
    out = float(np.maximum(0.0, RR - Rn).sum())
    return ov, out, min(x[2] for x in gl), RR.max()


def squeeze(rot, step=0.002, rho=0.004, th_rho=0.008, verbose=True, max_it=4000):
    TH = np.zeros(N)
    Cw = C.copy()
    R = 5.0
    st0, best = step, (R, Cw.copy(), TH.copy())
    fails = 0
    it = 0
    while step >= 2e-5 and fails < 10 and it < max_it:
        Rn = R - step
        rhos, ths = rho, th_rho
        ov1, out1, mg1, mr1 = merit(Cw, TH, Rn)
        won = False
        for inner in range(60):
            res, nv, ng, nr, D = lp_solve(Cw, TH, Rn, rot, rhos, ths)
            it += 1
            if not res.success:
                rhos *= 0.5
                ths *= 0.5
                if rhos < 1e-6:
                    break
                continue
            d = res.x[:nv]
            ov0, out0, mg0, mr0 = merit(Cw, TH, Rn)
            Cd = Cw + np.array([[d[D*k], d[D*k+1]] for k in range(N)])
            THd = (TH + np.array([d[D*k+2] for k in range(N)])) if rot else TH
            ov1, out1, mg1, mr1 = merit(Cd, THd, Rn)
            if ov1 + out1 < ov0 + out0 - 1e-14:
                Cw, TH = Cd, THd
                rhos = min(rhos*1.3, rho)
                ths = min(ths*1.3, th_rho)
            else:
                rhos *= 0.45
                ths *= 0.45
                if rhos < 1e-7:
                    break
                continue
            if ov1 <= 1e-12 and out1 <= 1e-12:
                won = True
                break
            if it >= max_it:
                break
        if won and mg1 >= -1e-12 and mr1 <= Rn + 1e-12:
            R = Rn
            best = (R, Cw.copy(), TH.copy())
            fails = 0
            step = min(step*1.4, st0)
            if verbose:
                print(f'  {"TR" if rot else "T "} R={R:.6f} gap={mg1:+.2e} worst={mr1:.6f} '
                      f'step={step:.5f} it={it}')
        else:
            fails += 1
            step *= 0.5
            if verbose:
                print(f'  {"TR" if rot else "T "} stall at Rn={Rn:.6f} '
                      f'ov={ov1:.3e} out={out1:.3e} (fails={fails})')
    return best, it


if __name__ == '__main__':
    st = float(sys.argv[1]) if len(sys.argv) > 1 else 0.002
    out = {}
    for rot, name in [(False, 'T'), (True, 'TR')]:
        (R, Cf, THf), its = squeeze(rot, step=st)
        mg, mr = S.state_metrics(Cf, THf)
        th = THf
        out[name] = dict(R=R, min_gap=mg, worst_corner=mr, its=its,
                         x=Cf[:, 0].tolist(), y=Cf[:, 1].tolist(),
                         th=(th.tolist() if rot else [0.0]*N), col=COL)
        print(f'== {name}: squeezed R={R:.6f}  minGap={mg:+.3e}  worstCorner={mr:.7f}  iters={its}')
    json.dump(out, open('squeeze_result.json', 'w'), indent=1)
    print(f'\nrotation gain  T - TR = {out["T"]["R"]-out["TR"]["R"]:+.6f}')
    print(f'saving vs certified 5.000000:  T {5-out["T"]["R"]:.6f}   TR {5-out["TR"]["R"]:.6f}')
    print(f'vs fit prediction 4.968155:    T {out["T"]["R"]-4.968155:+.6f}   '
          f'TR {out["TR"]["R"]-4.968155:+.6f}')

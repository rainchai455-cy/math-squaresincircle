import numpy as np, json, sys
from scipy.optimize import linprog
import squeeze_lp as S

OFF, N, C = S.OFF, S.N, S.C
COL = S.COL
WG_DEFAULT = 1000.0


def rows(Cw, TH, Rn, rot):
    """Full linearised feasibility model, no window.
    gap row   : g + grad.d >= -vg      (vg>=0 slack, heavily weighted)
    corner row: r + grad.d <= Rn + vr  (vr>=0 slack)"""
    D = 3 if rot else 2
    nv = D*N
    gl = S.gap_list(Cw, TH)
    RR = S.corner_r(Cw, TH)
    P = S.corner_pts(Cw, TH)
    gr, gh = [], []
    for (i, j, g, a, sg) in gl:
        row = np.zeros(nv)
        row[D*j] += sg*a[0]; row[D*j+1] += sg*a[1]
        row[D*i] += -sg*a[0]; row[D*i+1] += -sg*a[1]
        if rot:
            row[D*i+2] += -S.dsupport(a, i, TH)
            row[D*j+2] += -S.dsupport(a, j, TH)
        gr.append(row); gh.append(g)
    cr, ch = [], []
    for k in range(N):
        for m in range(4):
            r = RR[k, m]
            p = P[k, m]
            c_, s_ = np.cos(TH[k]), np.sin(TH[k])
            dp = OFF[m][0]*np.array([-s_, c_]) + OFF[m][1]*np.array([-c_, -s_])
            row = np.zeros(nv)
            row[D*k] = p[0]/r; row[D*k+1] = p[1]/r
            if rot:
                row[D*k+2] = (p @ dp)/r
            cr.append(row); ch.append(r - Rn)
    G = np.array(gr); B = np.array(cr)
    ng, nc = G.shape[0], B.shape[0]
    M = np.vstack([np.hstack([-G, np.eye(ng),  np.zeros((ng, nc))]),
                   np.hstack([B,  np.zeros((nc, ng)), -np.eye(nc)])])
    rhs = np.concatenate([np.array(gh), -np.array(ch)])
    obj = np.concatenate([np.zeros(nv), np.full(ng, WG_DEFAULT), np.ones(nc)])
    return M, rhs, obj, nv, ng, nc, D


def solve(Cw, TH, Rn, rot, rho, wg=1000.0):
    M, rhs, obj, nv, ng, nc, D = rows(Cw, TH, Rn, rot)
    obj = np.concatenate([np.zeros(nv), np.full(ng, wg), np.ones(nc)])
    if rot:
        th = rho/5.0          # a 1 rad rotation moves a rim corner ~5x more than a
        b = []                # 1 unit translation -> keep both effects comparable
        for k in range(N):
            b += [(-rho, rho), (-rho, rho), (-th, th)]
    else:
        b = [(-rho, rho)]*nv
    b += [(0.0, None)]*(ng+nc)
    return linprog(obj, A_ub=M, b_ub=rhs, bounds=b, method='highs'), nv, D


def merit(Cw, TH, Rn):
    gl = S.gap_list(Cw, TH)
    ov = sum(max(0.0, -x[2]) for x in gl)
    RR = S.corner_r(Cw, TH)
    out = float(np.maximum(0.0, RR - Rn).sum())
    return ov, out, min(x[2] for x in gl), RR.max()


def squeeze(rot, step=0.0004, rho=0.002, verbose=True, max_it=6000, floor=4.40):
    TH = np.zeros(N)
    Cw = C.copy()
    R = 5.0
    st0 = step
    best = (R, Cw.copy(), TH.copy())
    fails = 0
    it = 0
    while step >= 5e-5 and fails < 8 and R-step >= floor and it < max_it:
        Rn = R - step
        rh = rho
        cur = merit(Cw, TH, Rn)
        won = False
        for inner in range(40):
            res, nv, D = solve(Cw, TH, Rn, rot, rh)
            it += 1
            if not res.success:
                rh *= 0.5
                if rh < 1e-8:
                    break
                continue
            d = res.x[:nv]
            if np.abs(d).max() < 1e-15:
                break
            Cd = Cw + np.array([[d[D*k], d[D*k+1]] for k in range(N)])
            THd = (TH + np.array([d[D*k+2] for k in range(N)])) if rot else TH
            trial = merit(Cd, THd, Rn)
            if trial[0] + trial[1] < cur[0] + cur[1] - 1e-15:
                Cw, TH, cur = Cd, THd, trial
                rh = min(rh*1.25, rho)
            else:
                rh *= 0.4
                if rh < 1e-8:
                    break
            if cur[0] <= 1e-13 and cur[1] <= 1e-13:
                won = True
                break
            if it >= max_it:
                break
        ov1, out1, mg1, mr1 = cur
        if won:
            R = Rn
            best = (R, Cw.copy(), TH.copy())
            fails = 0
            step = min(step*1.35, st0)
            if verbose and int(round(R*1e5)) % 20 == 0:
                print(f'  {"TR" if rot else "T "} R={R:.6f} gap={mg1:+.2e} '
                      f'worst={mr1:.7f} it={it}')
        else:
            fails += 1
            step *= 0.5
            if verbose:
                print(f'  {"TR" if rot else "T "} stall at Rn={Rn:.6f} ov={ov1:.2e} '
                      f'out={out1:.2e} minGap={mg1:+.2e} worst={mr1:.7f} (fails={fails})')
    return best, it


if __name__ == '__main__':
    st = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0004
    rh = float(sys.argv[2]) if len(sys.argv) > 2 else 0.002
    out = {}
    for rot, name in [(False, 'T'), (True, 'TR')]:
        (R, Cf, THf), its = squeeze(rot, step=st, rho=rh)
        gl = S.gap_list(Cf, THf)
        RR = S.corner_r(Cf, THf)
        out[name] = dict(R=R, min_gap=min(x[2] for x in gl), worst_corner=RR.max(),
                         its=its, x=Cf[:, 0].tolist(), y=Cf[:, 1].tolist(),
                         th=THf.tolist(), col=COL)
        print(f'== {name}: best feasible R = {R:.7f}  minGap={out[name]["min_gap"]:+.3e}  '
              f'worstCorner={RR.max():.7f}  LPs={its}')
    out['_params'] = dict(step=st, rho=rh)
    json.dump(out, open(f'squeeze3_s{st:g}_r{rh:g}.json', 'w'), indent=1)
    print(f'\nrotation gain  T - TR = {out["T"]["R"]-out["TR"]["R"]:+.7f}')
    print(f'saving vs certified 5.0000000:  T {5-out["T"]["R"]:.7f}   TR {5-out["TR"]["R"]:.7f}')
    print(f'vs fit prediction 4.968155:     T {out["T"]["R"]-4.968155:+.6f}   '
          f'TR {out["TR"]["R"]-4.968155:+.6f}')

import numpy as np, json, sys
from scipy.optimize import linprog

# ---- exact certificate: doubled-cm integers, unit square => X1-X0 == 2 ----
rows = [l.split() for l in open('packing67.txt') if l.strip() and not l.startswith('#')]
COL = [r[0] for r in rows]
C = np.array([[(int(r[1])+int(r[3]))/4.0, (int(r[2])+int(r[4]))/4.0] for r in rows])
N = len(C)
OFF = np.array([[-.5, -.5], [-.5, .5], [.5, -.5], [.5, .5]])


def support(a, k, TH):
    c, s = np.cos(TH[k]), np.sin(TH[k])
    return 0.5*abs(a[0]*c + a[1]*s) + 0.5*abs(-a[0]*s + a[1]*c)


def dsupport(a, k, TH):
    c, s = np.cos(TH[k]), np.sin(TH[k])
    au = a[0]*c + a[1]*s
    av = -a[0]*s + a[1]*c
    return 0.5*np.sign(au)*av - 0.5*np.sign(av)*au


def corner_pts(C, TH):
    P = np.empty((len(C), 4, 2))
    for k in range(len(C)):
        c, s = np.cos(TH[k]), np.sin(TH[k])
        Rm = np.array([[c, -s], [s, c]])
        P[k] = C[k] + OFF @ Rm.T
    return P


def corner_r(C, TH):
    P = corner_pts(C, TH)
    return np.hypot(P[:, :, 0], P[:, :, 1])


def gap_list(C, TH):
    """one entry per pair that could touch. g = max over the 4 candidate axes of
    (|dc.a| - h_i(a) - h_j(a)); g >= 0 => provably disjoint along axis a."""
    n = len(C)
    out = []
    for i in range(n):
        for j in range(i+1, n):
            d = C[j] - C[i]
            if d @ d > 4.0:
                continue
            best, ba, bs = -9e9, None, 1.0
            for k in (i, j):
                c, s = np.cos(TH[k]), np.sin(TH[k])
                for a in ((c, s), (-s, c)):
                    a = np.array(a)
                    t = abs(d @ a) - support(a, i, TH) - support(a, j, TH)
                    if t > best:
                        best, ba = t, a
                        bs = 1.0 if d @ a >= 0 else -1.0
            out.append((i, j, best, ba, bs))
    return out


def state_metrics(Cw, TH):
    g = gap_list(Cw, TH)
    return min(x[2] for x in g), corner_r(Cw, TH).max()


def squeeze(rot, step=0.002, rho=0.02, th_rho=0.03, tg=0.03, tc=0.03, verbose=False, max_it=3000):
    TH = np.zeros(N)
    Cw = C.copy()
    R = 5.0
    st0 = step
    best = (R, Cw.copy(), TH.copy())
    fails, it = 0, 0
    while step >= 1e-4 and fails < 12 and it < max_it:
        Rn = R - step
        D = 3 if rot else 2
        nv = D*N
        gl = [x for x in gap_list(Cw, TH) if x[2] < tg]
        RR = corner_r(Cw, TH)
        ng = len(gl)
        g_rows = []
        for (i, j, g, a, sg) in gl:
            row = {}
            for idx, val in ((D*j+0, sg*a[0]), (D*j+1, sg*a[1]),
                             (D*i+0, -sg*a[0]), (D*i+1, -sg*a[1])):
                row[idx] = row.get(idx, 0) + val
            if rot:
                row[D*i+2] = row.get(D*i+2, 0) - dsupport(a, i, TH)
                row[D*j+2] = row.get(D*j+2, 0) - dsupport(a, j, TH)
            g_rows.append((row, -g))
        c_rows = []
        P = corner_pts(Cw, TH)
        for k in range(N):
            for m in range(4):
                r = RR[k, m]
                if r < Rn - tc or r < 1e-12:
                    continue
                p = P[k, m]
                c_, s_ = np.cos(TH[k]), np.sin(TH[k])
                dp_dth = OFF[m][0]*np.array([-s_, c_]) + OFF[m][1]*np.array([-c_, -s_])
                row = {D*k+0: p[0]/r, D*k+1: p[1]/r}
                if rot:
                    row[D*k+2] = (p @ dp_dth)/r
                c_rows.append((row, Rn - r))
        nr = len(c_rows)
        n = nv + ng + nr
        M = np.zeros((ng+nr, n))
        rhs = np.zeros(ng+nr)
        # gap:   grad.d + vg >= -g   <=>   -grad.d - vg <= g
        for e, (row, neg_g) in enumerate(g_rows):
            for idx, v in row.items():
                M[e, idx] = -v
            M[e, nv+e] = -1.0
            rhs[e] = -neg_g
        # corner: grad.d <= (Rn-r) + vr   <=>  grad.d - vr <= Rn-r
        for e, (row, bv) in enumerate(c_rows):
            rr = ng+e
            for idx, v in row.items():
                M[rr, idx] = v
            M[rr, nv+ng+e] = -1.0
            rhs[rr] = bv
        obj = np.concatenate([np.zeros(nv), np.full(ng, 10.0), np.ones(nr)])
        if rot:
            bounds = []
            for k in range(N):
                bounds += [(-rho, rho), (-rho, rho), (-th_rho, th_rho)]
        else:
            bounds = [(-rho, rho)]*nv
        bounds += [(0.0, None)]*(ng+nr)
        res = linprog(obj, A_ub=M, b_ub=rhs, bounds=bounds, method='highs')
        it += 1
        if not res.success:
            if verbose:
                print(f'   LP infeasible at Rn={Rn:.5f} step={step:.5f}: {res.message}')
            step *= 0.5
            fails += 1
            continue
        d = res.x[:nv]
        ok, used = False, None
        for scale in (1.0, 0.6, 0.36, 0.2, 0.1):
            Cd = Cw + np.array([[d[D*k], d[D*k+1]] for k in range(N)])*scale
            THd = (TH + np.array([d[D*k+2] for k in range(N)])*scale) if rot else TH.copy()
            mg, mr = state_metrics(Cd, THd)
            if mg >= -1e-12 and mr <= Rn + 1e-12:
                Cw, TH, R, used, ok = Cd, THd, Rn, scale, True
                best = (R, Cw.copy(), TH.copy())
                fails = 0
                step = min(step*1.25, st0)
                break
        if verbose and (it % 25 == 0 or not ok):
            print(f'  {("TR" if rot else "T ")} it={it} R={R:.5f} step={step:.5f} '
                  f'viol={res.x[nv:].sum():.2e} scale={used}')
        if not ok:
            fails += 1
            step *= 0.5
    return best, fails, it


if __name__ == '__main__':
    st = float(sys.argv[1]) if len(sys.argv) > 1 else 0.002
    out = {}
    for rot, name in [(False, 'T'), (True, 'TR')]:
        (R, Cf, THf), fails, its = squeeze(rot, step=st, verbose=True)
        mg, mr = state_metrics(Cf, THf)
        out[name] = dict(R=R, min_gap=mg, worst_corner=mr, fails=fails, its=its,
                         x=Cf[:, 0].tolist(), y=Cf[:, 1].tolist(), th=THf.tolist(), col=COL)
        print(f'== {name}: squeezed R={R:.6f}  minGap={mg:.3e}  worstCorner={mr:.6f}  '
              f'(iters={its}, stalls={fails})')
    json.dump(out, open('squeeze_result.json', 'w'), indent=1)
    print(f'\nrotation gain  T-TR = {out["T"]["R"]-out["TR"]["R"]:+.6f}')
    print(f'saving vs certified 5.000000:  T {5-out["T"]["R"]:.6f}   TR {5-out["TR"]["R"]:.6f}')
    print(f'gap to fit prediction 4.968155: T {out["T"]["R"]-4.968155:+.6f}  '
          f'TR {out["TR"]["R"]-4.968155:+.6f}')

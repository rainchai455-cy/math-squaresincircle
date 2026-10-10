# RETRACTED MODEL: uses Clarke subgradient (0) at the kink of h(theta)=0.5(|cos|+|sin|)
# at theta=0, which is the minimum-width orientation. Reports a spurious
# TR t*=-0.1. Correct model = rate3.py / rate4.py (|omega| kink terms).
import numpy as np, json, sys
import squeeze_lp as S
from scipy.optimize import linprog

OFF = S.OFF


def model(Cw, rot=False, act=1e-9, bind=1e-9):
    N = len(Cw)
    TH = np.zeros(N)
    D = 3 if rot else 2
    nv = D*N
    G, grhs = [], []
    for (i, j, g, a, sg) in S.gap_list(Cw, TH):
        if g >= act:
            continue
        row = np.zeros(nv)
        row[D*j] = sg*a[0]; row[D*j+1] = sg*a[1]
        row[D*i] = -sg*a[0]; row[D*i+1] = -sg*a[1]
        if rot:
            row[D*i+2] = -S.dsupport(a, i, TH)
            row[D*j+2] = -S.dsupport(a, j, TH)
        G.append(row); grhs.append(-g)
    RR = S.corner_r(Cw, TH)
    P = S.corner_pts(Cw, TH)
    R0 = RR.max()
    bic = [(k, m) for k in range(N) for m in range(4) if RR[k, m] > R0 - bind]
    B, brhs = [], []
    for (k, m) in bic:
        p = P[k, m]; r = RR[k, m]
        c_, s_ = np.cos(TH[k]), np.sin(TH[k])
        dp = OFF[m][0]*np.array([-s_, c_]) + OFF[m][1]*np.array([-c_, -s_])
        row = np.zeros(nv)
        row[D*k] = p[0]/r; row[D*k+1] = p[1]/r
        if rot:
            row[D*k+2] = (p @ dp)/r
        B.append(row)
    return np.array(G), np.array(grhs), np.array(B), bic, nv, D, N, R0


def lp_maxmin(G, grhs, B, nv, box=1.0):
    """min t  s.t.  G d >= grhs ,  B d - t <= 0 ,  |d| <= box   ->  t*<0 iff ALL
    binding corners can move inward at once while no contact penetrates."""
    ng, nc = G.shape[0], B.shape[0]
    A = np.vstack([np.hstack([-G, np.zeros((ng, 1))]),
                   np.hstack([B, -np.ones((nc, 1))])])
    b = np.concatenate([-grhs, np.zeros(nc)])
    obj = np.zeros(nv+1); obj[nv] = 1.0
    bounds = [(-box, box)]*nv + [(None, None)]
    r = linprog(obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    return r


def lp_sum(G, grhs, B, nv, box=1.0):
    ng = G.shape[0]
    A = np.vstack([-G]); b = np.array(-grhs)
    obj = B.sum(axis=0)
    bounds = [(-box, box)]*nv
    r = linprog(obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    return r


def lp_each(G, grhs, B, nv, box=1.0):
    A = -G
    b = np.array(-grhs)
    out = []
    for e in range(B.shape[0]):
        r = linprog(B[e], A_ub=A, b_ub=b, bounds=[(-box, box)]*nv, method='highs')
        out.append(None if not r.success else r.fun)
    return out


def report(tag, Cw, rot):
    G, grhs, B, bic, nv, D, N, R0 = model(Cw, rot)
    mm = lp_maxmin(G, grhs, B, nv)
    sm = lp_sum(G, grhs, B, nv)
    print(f'{tag} rot={rot}: dof={nv}  contacts={G.shape[0]}  binding corners={len(bic)}')
    print(f'   max-min t* = {mm.fun:+.6f}   (sum of all 8 rates min = {sm.fun:+.6f})')
    if mm.success and mm.fun < -1e-9:
        d = mm.x[:nv]
        mv = sum(1 for k in range(N) if np.linalg.norm(d[D*k:D*k+2]) > 1e-6)
        rt = sum(1 for k in range(N) if rot and abs(d[D*k+2]) > 1e-6)
        print(f'   -> SHRINKABLE. moving squares={mv}/{N}' + (f' rotating={rt}' if rot else ''))
    if len(bic) <= 12:
        ea = lp_each(G, grhs, B, nv)
        print('   per-corner min rate: ' + '  '.join(
            f'(sq{bic[e][0]},c{bic[e][1]})={ea[e]:+.4f}' for e in range(len(bic))))
    return mm.fun if mm.success else None


if __name__ == '__main__':
    rows = [l.split() for l in open('packing67.txt') if l.strip() and not l.startswith('#')]
    C67 = np.array([[(int(r[1])+int(r[3]))/4.0, (int(r[2])+int(r[4]))/4.0] for r in rows])
    print('=== 67 packing, R=5 ===')
    t_T = report('67', C67, False)
    t_TR = report('67', C67, True)
    print('\n=== POSITIVE CONTROL: drop squares, same code ===')
    # drop the squares owning binding corners one at a time -> should free a mechanism
    G, grhs, B, bic, nv, D, N, R0 = model(C67, False)
    owners = sorted({k for (k, m) in bic})
    print(f'binding-corner owner squares: {owners}')
    for drop in owners[:4]:
        keep = [i for i in range(N) if i != drop]
        t = report(f'67 minus sq{drop}', C67[keep], False)
    print('\n=== NEGATIVE CONTROL: drop a square that owns NO binding corner ===')
    free = [i for i in range(N) if i not in owners]
    for drop in free[:2]:
        keep = [i for i in range(N) if i != drop]
        t = report(f'67 minus sq{drop}', C67[keep], False)

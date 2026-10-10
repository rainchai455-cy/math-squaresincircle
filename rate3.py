import numpy as np, json, sys
from scipy.optimize import linprog

OFF = np.array([[-.5, -.5], [-.5, .5], [.5, -.5], [.5, .5]])
TIE = 1e-9


def support(a, th):
    c, s = np.cos(th), np.sin(th)
    return 0.5*abs(a[0]*c + a[1]*s) + 0.5*abs(-a[0]*s + a[1]*c)


def support_split(a, th):
    """Exact one-sided directional derivative of h(a,theta) in direction w.
    h = 0.5|a.u| + 0.5|a.v|.  If a is parallel to u or v (i.e. a is an EDGE NORMAL
    of the square) then h is at its MINIMUM over theta -> kink -> the derivative is
    +0.5|w| (rotation can only widen the square against that axis).  Otherwise smooth.
    Returns (coeff_on_w, coeff_on_theta) so  dh = k*|w| + m*w."""
    c, s = np.cos(th), np.sin(th)
    au = a[0]*c + a[1]*s
    av = -a[0]*s + a[1]*c
    kink = (abs(au) < 1e-12) or (abs(av) < 1e-12)
    m = 0.5*np.sign(au)*av - 0.5*np.sign(av)*au      # smooth part
    return (0.5 if kink else 0.0), m


def contacts(C, TH, gtol=1e-9):
    """active face contacts: pairs whose separation is 0 along >=1 SAT axis.
    Yields (i,j,g,axis,sgn,k_i,m_i,k_j,m_j)."""
    N = len(C)
    out = []
    for i in range(N):
        for j in range(i+1, N):
            d = C[j]-C[i]
            if d @ d > 4.2:
                continue
            cand = []
            for kk in (i, j):
                c, s = np.cos(TH[kk]), np.sin(TH[kk])
                cand += [np.array([c, s]), np.array([-s, c])]
            best = -9e9
            for a in cand:
                t = abs(d @ a) - support(a, TH[i]) - support(a, TH[j])
                if t > best:
                    best = t
            if best > gtol:
                continue
            # keep every axis that realises the max; enforce the FIRST one -> sufficient
            for a in cand:
                t = abs(d @ a) - support(a, TH[i]) - support(a, TH[j])
                if t >= best - TIE:
                    sg = 1.0 if d @ a >= 0 else -1.0
                    ki, mi = support_split(a, TH[i])
                    kj, mj = support_split(a, TH[j])
                    out.append((i, j, best, a, sg, ki, mi, kj, mj))
                    break
    return out


def corners(C, TH):
    N = len(C)
    P = np.empty((N, 4, 2))
    for k in range(N):
        c, s = np.cos(TH[k]), np.sin(TH[k])
        P[k] = C[k] + OFF @ np.array([[c, s], [-s, c]])
    return P, np.hypot(P[:, :, 0], P[:, :, 1])


def first_order_rate(C, rot, box=1.0):
    """min t  s.t.  every active contact keeps gap-rate >= 0 (rotation enters through
    its EXACT one-sided derivative, kink included), every binding corner moves inward
    at rate <= t.  t* < 0 => a provable squeeze direction exists."""
    N = len(C)
    TH = np.zeros(N)
    D = 3 if rot else 2
    nv = D*N
    cl = contacts(C, TH)
    P, RR = corners(C, TH)
    R0 = RR.max()
    bi = [(k, m) for k in range(N) for m in range(4) if RR[k, m] > R0 - 1e-9]
    nw = N if rot else 0
    # vars: u(2N) [th(N)] then w(N>=|th|) then t   ->  w block starts right after nv
    nvar = nv + nw + 1
    iw = nv
    A, b = [], []
    for (i, j, g, a, sg, ki, mi, kj, mj) in cl:
        row = np.zeros(nvar)
        # rate = sg*((u_j-u_i).a) - (ki*w_i+mi*th_i) - (kj*w_j+mj*th_j) >= -g
        # in A_ub form:  -rate <= g
        row[D*j] = -sg*a[0]; row[D*j+1] = -sg*a[1]
        row[D*i] = sg*a[0]; row[D*i+1] = sg*a[1]
        if rot:
            row[iw+i] = ki
            row[iw+j] = kj
            row[3*i+2] = mi
            row[3*j+2] = mj
        A.append(row); b.append(g)
    for (k, m) in bi:
        p = P[k, m]; r = RR[k, m]
        c_, s_ = np.cos(TH[k]), np.sin(TH[k])
        dp = OFF[m][0]*np.array([-s_, c_]) + OFF[m][1]*np.array([-c_, -s_])
        row = np.zeros(nvar)
        row[D*k] = p[0]/r; row[D*k+1] = p[1]/r
        if rot:
            row[3*k+2] = (p @ dp)/r
        row[nvar-1] = -1.0
        A.append(row); b.append(0.0)
    # w >= |theta|  ->  theta - w <= 0 ; -theta - w <= 0
    for k in range(N if rot else 0):
        r1 = np.zeros(nvar); r1[3*k+2] = 1.0; r1[iw+k] = -1.0
        A.append(r1); b.append(0.0)
        r2 = np.zeros(nvar); r2[3*k+2] = -1.0; r2[iw+k] = -1.0
        A.append(r2); b.append(0.0)
    A = np.array(A); b = np.array(b)
    obj = np.zeros(nvar); obj[nvar-1] = 1.0
    bounds = [(-box, box)]*nv + [(0.0, None)]*nw + [(None, None)]
    res = linprog(obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    return res, len(cl), len(bi), nv, N, D


def report(tag, C, rot):
    res, nc, nb, nv, N, D = first_order_rate(C, rot)
    if not res.success:
        print(f'{tag} rot={rot}: LP {res.status}  (contacts={nc}, binding={nb})')
        return None
    d = res.x[:nv]
    print(f'{tag} rot={rot} dof={nv} active contacts={nc} binding corners={nb} -> '
          f't* = {res.fun:+.6f}  {"SHRINKABLE" if res.fun < -1e-9 else "JAMMED (1st order)"}')
    if res.success and res.fun < -1e-9:
        mv = sum(1 for k in range(N) if np.linalg.norm(d[D*k:D*k+2]) > 1e-9)
        print(f'    moving={mv}/{N}' + (f' rotating={sum(1 for k in range(N) if abs(d[3*k+2])>1e-9)}'
                                        if rot else ''))
    return res.fun


if __name__ == '__main__':
    rows = [l.split() for l in open('packing67.txt') if l.strip() and not l.startswith('#')]
    C67 = np.array([[(int(r[1])+int(r[3]))/4.0, (int(r[2])+int(r[4]))/4.0] for r in rows])
    print('KINK-AWARE first-order squeeze rates')
    a = report('67 packing', C67, False)
    bq = report('67 packing', C67, True)
    print('\nPOSITIVE CONTROLS (same code, configurations that must be squeezable)')
    blk = np.array([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]])
    report('2x2 block, centred (optimal, expect JAMMED)', blk, False)
    report('2x2 block offset (0.15,0.15) (expect SHRINKABLE)', blk+np.array([0.15, 0.15]), False)
    row3 = np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    report('3-in-a-row offset (expect SHRINKABLE)', row3+np.array([0.2, 0.0]), False)
    report('3-in-a-row centred (optimal, expect JAMMED)', row3, False)
    report('3-in-a-row offset', row3+np.array([0.2, 0.0]), True)
    print('\n=== how much did the old kink-free model overstate? ===')
    print(f'  T : t*={a}   TR : t*={bq}')

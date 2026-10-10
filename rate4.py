import numpy as np
from scipy.optimize import linprog
import rate3 as R

OFF = R.OFF


def contact_rows(C, TH, exact_ties=True):
    """Rows encoding first-order impenetrability.
      face contact (unique max axis a):  rate = sg*(du.a) - dh_i - dh_j,
          with dh = k*|w| + m*theta  (k=0.5 exactly when a is that square's own
          edge normal, i.e. theta=0 is the MINIMUM-WIDTH orientation -> rotation
          can only widen -> kink, modelled with an |omega| variable)
      corner-corner contact (>=2 tied axes): the true separating-motion set is the
          single half-space (du).n >= 0 with n = the diagonal contact normal, and
          rotation contributes 0 at first order.  Using it is EXACT, not conservative.
    """
    N = len(C)
    rows, rh = [], []
    kinds = []
    for i in range(N):
        for j in range(i+1, N):
            d = C[j]-C[i]
            if d @ d > 4.2:
                continue
            seen = {}
            for kk in (i, j):
                c, s = np.cos(TH[kk]), np.sin(TH[kk])
                for ax in (np.array([c, s]), np.array([-s, c])):
                    key = (round(abs(ax[0]), 9), round(abs(ax[1]), 9))
                    t = abs(d @ ax) - R.support(ax, TH[i]) - R.support(ax, TH[j])
                    if key not in seen or t > seen[key][0]:
                        seen[key] = (t, ax)
            mx = max(v[0] for v in seen.values())
            if mx > 1e-9:
                continue
            tied = [v[1] for v in seen.values() if v[0] >= mx - 1e-9]
            if exact_ties and len(tied) >= 2:
                n = np.zeros(2)
                for ax in tied:
                    n += ax*(1.0 if d @ ax >= 0 else -1.0)
                n /= np.linalg.norm(n)
                row = np.zeros(4*N)
                row[3*j] = -n[0]; row[3*j+1] = -n[1]
                row[3*i] = n[0]; row[3*i+1] = n[1]
                rows.append(row); rh.append(mx)
                kinds.append('corner')
            else:
                ax = tied[0]
                sg = 1.0 if d @ ax >= 0 else -1.0
                ki, mi = R.support_split(ax, TH[i])
                kj, mj = R.support_split(ax, TH[j])
                row = np.zeros(3*N + N)
                row[3*j] = -sg*ax[0]; row[3*j+1] = -sg*ax[1]
                row[3*i] = sg*ax[0]; row[3*i+1] = sg*ax[1]
                row[3*i+2] = mi; row[3*j+2] = mj
                row[3*N+i] = ki; row[3*N+j] = kj
                rows.append(row); rh.append(mx)
                kinds.append('face')
    return np.array(rows), np.array(rh), kinds


def rate(C, rot, exact_ties):
    N = len(C)
    TH = np.zeros(N)
    P, RR = R.corners(C, TH)
    R0 = RR.max()
    bi = [(k, m) for k in range(N) for m in range(4) if RR[k, m] > R0 - 1e-9]
    A_c, b_c, kinds = contact_rows(C, TH, exact_ties)
    D = 3 if rot else 2
    nv = D*N
    nw = N if rot else 0
    nvar = nv + nw + 1
    iw = nv
    A, b = [], []
    for e in range(len(A_c)):
        row = np.zeros(nvar)
        is_corner = exact_ties and kinds[e] == 'corner'
        for t in range(N):                       # translation part (always present)
            row[D*t] += A_c[e][3*t]
            row[D*t+1] += A_c[e][3*t+1]
        if rot:
            for t in range(N):                   # smooth rotation part
                row[3*t+2] += A_c[e][3*t+2]
            if not is_corner:                    # kink |omega| part
                row[iw:iw+N] += A_c[e][3*N:4*N]
        A.append(row); b.append(b_c[e])
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
    for k in range(N if rot else 0):
        r1 = np.zeros(nvar); r1[3*k+2] = 1.0; r1[iw+k] = -1.0
        A.append(r1); b.append(0.0)
        r2 = np.zeros(nvar); r2[3*k+2] = -1.0; r2[iw+k] = -1.0
        A.append(r2); b.append(0.0)
    A = np.array(A); b = np.array(b)
    obj = np.zeros(nvar); obj[nvar-1] = 1.0
    bounds = [(-1.0, 1.0)]*nv + [(0.0, None)]*nw + [(None, None)]
    res = linprog(obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    ncorner = kinds.count('corner') if exact_ties else 0
    return res, len(kinds), ncorner, len(bi), A.shape


if __name__ == '__main__':
    rows = [l.split() for l in open('packing67.txt') if l.strip() and not l.startswith('#')]
    C = np.array([[(int(r[1])+int(r[3]))/4.0, (int(r[2])+int(r[4]))/4.0] for r in rows])
    print('conservatism check: strict single-axis vs exact corner-normal ties')
    for et in (False, True):
        for rot in (False, True):
            res, nc, ncor, nb, shape = rate(C, rot, et)
            tag = 'exact corner normals' if et else 'strict one-axis    '
            print(f'  {tag} rot={str(rot):5s}: rows={shape[0]} contacts={nc} '
                  f'(corner-ties={ncor}) binding={nb} -> t* = {res.fun:+.6f}  '
                  f'{"SHRINKABLE" if res.fun < -1e-9 else "JAMMED"}')
    # controls must still pass in the exact model
    print('\ncontrols under the exact-tie model:')
    blk = np.array([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]])
    for tag, cc in [('2x2 centred', blk), ('2x2 offset', blk+np.array([.15, .15]))]:
        res, *_ = rate(cc, False, True)
        print(f'  {tag:12s} t* = {res.fun:+.6f}  {"SHRINKABLE" if res.fun < -1e-9 else "JAMMED"}')

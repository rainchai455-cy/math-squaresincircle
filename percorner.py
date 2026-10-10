import numpy as np
from scipy.optimize import linprog
import rate4 as R4
import rate3 as R

rows = [l.split() for l in open('packing67.txt') if l.strip() and not l.startswith('#')]
COL = [r[0] for r in rows]
C = np.array([[(int(r[1])+int(r[3]))/4.0, (int(r[2])+int(r[4]))/4.0] for r in rows])
N = len(C)
TH = np.zeros(N)
P, RR = R.corners(C, TH)
bi = [(k, m) for k in range(N) for m in range(4) if RR[k, m] > 5.0 - 1e-9]

for et in (True, False):
    A_c, b_c, kinds = R4.contact_rows(C, TH, et)
    ncon = A_c.shape[0]
    print(f'\n--- tie model: {"exact corner normals" if et else "strict one axis"} '
          f'({kinds.count("corner")} corner / {kinds.count("face")} face) ---')
    for rot in (False, True):
        D = 3 if rot else 2
        nv = D*N
        nw = N if rot else 0
        nvar = nv + nw + 1
        iw = nv
        A, b = [], []
        for e in range(ncon):
            is_cor = et and kinds[e] == 'corner'
            row = np.zeros(nvar)
            for t in range(N):
                row[D*t] += A_c[e][3*t]; row[D*t+1] += A_c[e][3*t+1]
            if rot:
                for t in range(N):
                    row[3*t+2] += A_c[e][3*t+2]
                if not is_cor:
                    row[iw:iw+N] += A_c[e][3*N:4*N]
            A.append(row); b.append(b_c[e])
        for k in range(N if rot else 0):
            r1 = np.zeros(nvar); r1[3*k+2] = 1.0; r1[iw+k] = -1.0
            A.append(r1); b.append(0.0)
            r2 = np.zeros(nvar); r2[3*k+2] = -1.0; r2[iw+k] = -1.0
            A.append(r2); b.append(0.0)
        A = np.array(A); b = np.array(b)
        bnds = [(-1.0, 1.0)]*nv + [(0.0, None)]*nw
        # max-min over the 8 binding corners
        Am = np.zeros((len(A), nvar)); Am[:, :nvar-1] = A[:, :nvar-1]
        bm = b.copy()
        rows_c, rhs_c = [], []
        for (k, m) in bi:
            p = P[k, m]; r = RR[k, m]
            c_, s_ = np.cos(TH[k]), np.sin(TH[k])
            dp = R.OFF[m][0]*np.array([-s_, c_]) + R.OFF[m][1]*np.array([-c_, -s_])
            row = np.zeros(nvar)
            row[D*k] = p[0]/r; row[D*k+1] = p[1]/r
            if rot:
                row[3*k+2] = (p @ dp)/r
            row[nvar-1] = -1.0
            rows_c.append(row); rhs_c.append(0.0)
        Af = np.vstack([Am, np.array(rows_c)]); bf = np.concatenate([bm, np.array(rhs_c)])
        obj = np.zeros(nvar); obj[nvar-1] = 1.0
        r0 = linprog(obj, A_ub=Af, b_ub=bf, bounds=bnds+[(None, None)], method='highs')
        each = []
        for (k, m) in bi:
            p = P[k, m]; r = RR[k, m]
            c_, s_ = np.cos(TH[k]), np.sin(TH[k])
            dp = R.OFF[m][0]*np.array([-s_, c_]) + R.OFF[m][1]*np.array([-c_, -s_])
            g = np.zeros(nvar)
            g[D*k] = p[0]/r; g[D*k+1] = p[1]/r
            if rot:
                g[3*k+2] = (p @ dp)/r
            rr = linprog(g, A_ub=A, b_ub=b, bounds=bnds+[(None, None)], method='highs')
            each.append(rr.fun)
        tag = 'TR' if rot else 'T '
        print(f'  {tag}: max-min t* = {r0.fun:+.6f}   '
              f'{"SHRINKABLE" if r0.fun < -1e-9 else "JAMMED"}')
        print(f'      per-corner single rates: ' +
              ' '.join(f'{v:+.3f}' for v in each) +
              f'   (all can move in alone? {all(v < -1e-9 for v in each)})')

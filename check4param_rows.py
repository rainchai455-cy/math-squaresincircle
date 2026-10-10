"""Self-consistency of the reported 4-parameter rows.

For each reported row: fit the exact same basis+window by least squares (that RMS is
an unreachable floor for ANY coefficients), then evaluate the REPORTED coefficients
in the same formula to see whether they even produce the REPORTED r67/r68.
"""
import math
import numpy as np

R5 = {1: 0.707107, 2: 1.118034, 3: 1.288471, 4: 1.414214, 5: 1.581139, 6: 1.688543,
      7: 1.802776, 8: 1.97877, 9: 2.07759, 10: 2.12132, 11: 2.21386, 12: 2.23606,
      13: 2.3607, 14: 2.5, 15: 2.533, 16: 2.62309, 17: 2.67687, 18: 2.74146,
      19: 2.80150, 20: 2.893, 21: 2.91547, 22: 3.03239, 23: 3.06738, 24: 3.10940,
      25: 3.17450, 26: 3.20156, 27: 3.26050, 28: 3.33147, 29: 3.39309, 30: 3.45962,
      31: 3.52279, 32: 3.57340, 33: 3.60010, 34: 3.62852, 35: 3.64005, 36: 3.72853,
      37: 3.77126, 38: 3.83514, 39: 3.88461, 40: 3.93833, 41: 3.98879, 42: 4.03112,
      43: 4.06733, 44: 4.09999, 45: 4.14067, 46: 4.16573, 47: 4.19870, 48: 4.25610,
      49: 4.28307, 50: 4.30116}
NS = np.arange(1, 51, dtype=float)
EX = np.array([R5[int(n)] for n in NS]) - np.sqrt(NS / math.pi)   # excess y_n
B = lambda n: np.column_stack([np.ones(len(np.atleast_1d(n))), np.power(n, -0.5),
                               np.power(n, -1.0), np.power(n, -1.5)])
Lm = lambda n: math.sqrt(n / math.pi)
full = lambda p, n: Lm(n) + p[0] + p[1] / math.sqrt(n) + p[2] / n + p[3] / n ** 1.5

ROWS = [('exp1 10-50', 10, np.array([0.2246, 0.6821, -2.512, 2.119]), 0.0128,
         4.9978, 5.0329, 0.0351),
        ('exp2 20-50', 20, np.array([0.2569, 0.4215, -0.938, 0.231]), 0.0104,
         4.9991, 5.0332, 0.0351),
        ('base 1-50', 1, np.array([0.3683, -0.1350, 0.2768, -0.3608]), None,
         None, None, None)]

print('%-12s %8s %8s %8s | %8s %8s | %8s %8s' %
      ('row', 'LSQ RMS', 'rep RMS', 'verdict', 'rep r67', 'my r67@rep',
       'rep r68', 'my r68@rep'))
for nm, lo, p, rms_rep, r67_rep, r68_rep, step_rep in ROWS:
    sel = NS >= lo
    x, y = NS[sel], EX[sel]
    X = B(x)
    sol, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ sol
    rms_lsq = float(np.sqrt(np.mean(e ** 2)))
    rms_rep_c = float(np.sqrt(np.mean((y - X @ p) ** 2)))          # RMS AT his coeffs
    cmp_ = '' if rms_rep is None else ('IMPOSSIBLE (rep < LSQ floor)'
                                       if rms_rep < rms_lsq - 1e-12 else 'feasible')
    print('%-12s %8.5f %8.5f %8s | %s %s | %s %s' %
          (nm, rms_lsq, rms_rep_c, cmp_,
             '%.4f' % r67_rep if r67_rep else '   -  ', '%.4f' % full(p, 67.0),
             '%.4f' % r68_rep if r68_rep else '   -  ', '%.4f' % full(p, 68.0)))
    if r68_rep:
        print('             reported step %.4f vs step at reported coeffs %.4f'
              ' | LSQ coeffs %s' % (step_rep, full(p, 68.0) - full(p, 67.0),
                                    np.round(sol, 4).tolist()))
        print('             reported RMS %.5f is BELOW the least-squares floor %.5f'
              ' by %.5f  -> no coefficients of this basis/window give it'
              % (rms_rep, rms_lsq, rms_lsq - rms_rep))

print('\n[2] does any coefficient vector on that window reach the reported RMS?')
for nm, lo, p, rms_rep, _, _, _ in ROWS:
    if rms_rep is None:
        continue
    sel = NS >= lo
    X, y = B(NS[sel]), EX[sel]
    sol, *_ = np.linalg.lstsq(X, y, rcond=None)
    floor = float(np.sqrt(np.mean((y - X @ sol) ** 2)))
    # also the best RMS achievable with c FIXED at his value (3 free params)
    X3 = X[:, 1:]
    s3, *_ = np.linalg.lstsq(X3, y - X[:, 0] * p[0], rcond=None)
    f3 = float(np.sqrt(np.mean((y - X[:, 0] * p[0] - X3 @ s3) ** 2)))
    print('    %-12s free (C0,d,f,g) floor %.5f | with c pinned to %.4f floor %.5f'
          ' | reported %.5f -> reachable: %s'
          % (nm, floor, p[0], f3, rms_rep, rms_rep >= min(floor, f3) - 1e-12))

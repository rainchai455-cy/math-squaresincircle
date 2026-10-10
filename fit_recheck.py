"""Recheck of the reported fit for r_n = sqrt(n/pi) + c + d/sqrt(n*pi), then the
three follow-ups requested: (1) add e/n, (2) leave-one-out validation, (3) whether
the 67/68 prediction is stable rather than dragged by small n.

Basis used everywhere: [1, u_n] with u_n = 1/sqrt(n*pi)  (the user's stated model).
"""
import math
import numpy as np

R = {1: 0.707107, 2: 1.118034, 3: 1.288471, 4: 1.414214, 5: 1.581139, 6: 1.688543,
     7: 1.802776, 8: 1.97877, 9: 2.07759, 10: 2.12132, 11: 2.21386, 12: 2.23606,
     13: 2.3607, 14: 2.5, 15: 2.533, 16: 2.62309, 17: 2.67687, 18: 2.74146,
     19: 2.80150, 20: 2.893, 21: 2.91547, 22: 3.03239, 23: 3.06738, 24: 3.10940,
     25: 3.17450, 26: 3.20156, 27: 3.26050, 28: 3.33147, 29: 3.39309, 30: 3.45962,
     31: 3.52279, 32: 3.57340, 33: 3.60010, 34: 3.62852, 35: 3.64005, 36: 3.72853,
     37: 3.77126, 38: 3.83514, 39: 3.88461, 40: 3.93833, 41: 3.98879, 42: 4.03112,
     43: 4.06733, 44: 4.09999, 45: 4.14067, 46: 4.16573, 47: 4.19870, 48: 4.25610,
     49: 4.28307, 50: 4.30116}
NS = np.arange(1, 51)
E = np.array([R[n] for n in NS]) - np.sqrt(NS / math.pi)
u = lambda n: 1.0 / np.sqrt(np.asarray(n, dtype=float) * math.pi)
L = lambda n: np.sqrt(np.asarray(n, dtype=float) / math.pi)
L67, L68 = float(L(67)), float(L(68))
STEP = L68 - L67
CLAIM = [(5.031405, 0.385425, -0.096205, 0.02714, 4.996885, 5.031270, '1-50'),
         (None, 0.345938, 0.014536, 0.02282, 4.965031, 4.999359, '8-50'),
         (None, 0.329516, 0.071869, 0.02013, 4.952561, 4.986860, '20-50')]

print('[1] reproduce the reported rows exactly (basis [1, 1/sqrt(n pi)])')
print('%-8s %10s %10s %8s | %10s %10s %8s | %10s %10s' % (
    'window', 'c(lsq)', 'd(lsq)', 'RMS', 'c(reported)', 'd(reported)', 'd/d_lsq',
    'r67(lsq)', 'r68(lsq)'))
for _, c_r, d_r, rms_r, p67_r, p68_r, tag in CLAIM:
    lo = int(tag.split('-')[0])
    m = NS >= lo
    X = np.column_stack([np.ones(int(m.sum())), u(NS[m])])
    sol, *_ = np.linalg.lstsq(X, E[m], rcond=None)
    res = E[m] - X @ sol
    rms = float(np.sqrt(np.mean(res ** 2)))
    p67 = L67 + sol[0] + sol[1] * u(67)
    p68 = L68 + sol[0] + sol[1] * u(68)
    print('%-8s %10.6f %10.6f %8.5f | %10.6f %11.6f %8.4f | %10.6f %10.6f'
          % (tag, sol[0], sol[1], rms, c_r, d_r, d_r / sol[1], p67, p68))
    print('%-8s reported pair (c=%.6f, d=%.6f) gives RMS %8.5f vs %.5f claimed; '
          'its r67/r68 %9.6f %9.6f (reported %9.6f %9.6f)'
          % ('', c_r, d_r, float(np.sqrt(np.mean((E[m] - (c_r + d_r * u(NS[m]))) ** 2))),
             rms_r, L67 + c_r + d_r * u(67), L68 + c_r + d_r * u(68), p67_r, p68_r))
    print('%-8s d_lsq/pi = %.6f  (reported %.6f, diff %.2e)   d_lsq/sqrt(pi) = %.6f'
          % ('', sol[1] / math.pi, d_r, abs(sol[1] / math.pi - d_r), sol[1] / math.sqrt(math.pi)))

print('\n[2] is the reported pair even a least-squares solution at fixed c?')
for _, c_r, d_r, _, _, _, tag in CLAIM:
    lo = int(tag.split('-')[0])
    m = NS >= lo
    y = E[m] - c_r
    x = u(NS[m])
    d_at_c = float(np.dot(x, y) / np.dot(x, x))
    print('    %-6s best d given c=%.6f is %+.6f ; reported %+.6f ; pi-scaled %+.6f'
          % (tag, c_r, d_at_c, d_r, d_at_c / math.pi))

print('\n[3] add the e/n term:  r_n = sqrt(n/pi) + c + d/sqrt(n pi) + e/n')
print('%-8s %9s %9s %9s %8s | %9s %9s | %s' % (
    'window', 'c', 'd', 'e', 'RMS', 'r67', 'r68', 'N(5) implied'))
for lo, hi in ((1, 50), (8, 50), (20, 50), (30, 50)):
    m = (NS >= lo) & (NS <= hi)
    X = np.column_stack([np.ones(int(m.sum())), u(NS[m]), 1.0 / NS[m]])
    sol, *_ = np.linalg.lstsq(X, E[m], rcond=None)
    res = E[m] - X @ sol
    rms = float(np.sqrt(np.mean(res ** 2)))
    sig2 = float(res @ res) / (int(m.sum()) - 3)
    XtXi = np.linalg.inv(X.T @ X)
    xn = np.array([1.0, u(68), 1.0 / 68.0])
    se68 = math.sqrt(sig2 * (1.0 + float(xn @ XtXi @ xn)))
    p67 = L67 + sol[0] + sol[1] * u(67) + sol[2] / 67.0
    p68 = L68 + sol[0] + sol[1] * u(68) + sol[2] / 68.0
    print('%-8s %9.6f %9.6f %9.6f %8.5f | %9.6f %9.6f | %d  (r68-5 = %+.5f, SE %.5f)'
          % ('%d-%d' % (lo, hi), sol[0], sol[1], sol[2], rms, p67, p68,
             68 if p68 <= 5 else 67, p68 - 5.0, se68))
    se = np.sqrt(sig2 * np.diag(XtXi))
    print('%-8s SE: c %.5f d %.5f e %.5f  => e is %.2f SE from 0'
          % ('', se[0], se[1], se[2], abs(sol[2]) / se[2]))

print('\n[4] leave-one-out validation (2-parameter model, fit on 49 pts, predict the 50th)')
for lo, hi in ((1, 50), (8, 50), (20, 50)):
    m_all = (NS >= lo) & (NS <= hi)
    errs = []
    for i in np.nonzero(m_all)[0]:
        sel = m_all.copy()
        sel[i] = False
        X = np.column_stack([np.ones(int(sel.sum())), u(NS[sel])])
        sol, *_ = np.linalg.lstsq(X, E[sel], rcond=None)
        pred = L(NS[i]) + sol[0] + sol[1] * u(NS[i])
        errs.append(pred - R[NS[i]])
    errs = np.array(errs)
    print('    %-6s LOO RMSE %.5f  max|e| %.5f (n=%d)  mean bias %+.5f'
          % ('%d-%d' % (lo, hi), float(np.sqrt(np.mean(errs ** 2))),
             float(np.max(np.abs(errs))), int(NS[np.argmax(np.abs(errs))]),
             float(np.mean(errs))))
    n_out = int(np.sum(np.abs(errs) > 3 * float(np.sqrt(np.mean(errs ** 2)))))
    print('    %-6s points beyond 3x LOO-RMSE: %d %s'
          % ('', n_out, [int(NS[i]) for i in range(len(errs))
                         if abs(errs[i]) > 3 * float(np.sqrt(np.mean(errs ** 2)))]))

print('\n[5] stability of the 67/68 call across windows and model order (2-par model)')
calls = []
for lo, hi in ((1, 50), (2, 50), (5, 50), (8, 50), (11, 50), (15, 50), (20, 50),
               (21, 50), (25, 50), (30, 50), (31, 50), (35, 50), (41, 50)):
    m = (NS >= lo) & (NS <= hi)
    X = np.column_stack([np.ones(int(m.sum())), u(NS[m])])
    sol, *_ = np.linalg.lstsq(X, E[m], rcond=None)
    p68 = L68 + sol[0] + sol[1] * u(68)
    p67 = L67 + sol[0] + sol[1] * u(67)
    calls.append((lo, hi, p67, p68))
    print('    %-6s d=%+8.5f  r67=%.6f (%+8.5f vs anchor 5)  r68=%.6f (%+8.5f)  call=%d'
          % ('%d-%d' % (lo, hi), sol[1], p67, p67 - 5.0, p68, p68 - 5.0,
             67 if p68 > 5 else 68))
n67 = sum(1 for v in calls if v[3] > 5)
print('    windows predicting r68 > 5 (=> N(5)=67): %d of %d ; r68 range %.6f..%.6f '
      '(span %.4f = %.2f area steps)'
      % (n67, len(calls), min(v[3] for v in calls), max(v[3] for v in calls),
         max(v[3] for v in calls) - min(v[3] for v in calls),
         (max(v[3] for v in calls) - min(v[3] for v in calls)) / STEP))
print('    anchor r67=5 vs predictions: gap range %+.5f..%+.5f'
      % (min(v[2] for v in calls) - 5.0, max(v[2] for v in calls) - 5.0))

print('\n[6] sensitivity of d to the r68 anchor used (r67=5 fixed)')
for tag, val in (('5.031405 (this msg, = 5/0.993758 derived)', 5.031405),
                 ('5.031415 (previous msg)', 5.031415),
                 ('5.031425 (msg before)', 5.031425),
                 ('5.052404006121 (project verified)', 5.052404006121)):
    e68 = val - L68
    e67 = 5.0 - L67
    d = (e67 - e68) / (u(67) - u(68))
    print('    %-44s e68=%.6f  d=%+10.4f  c=%+.6f' % (tag, e68, d, e67 - d * u(67)))
print('    1e-5 in r68 moves d by %.4f (%.2f%% of d=5.74) ; the three quoted r68 values '
      'span d = %.3f..%.3f' % (1e-5 / (u(67) - u(68)),
                               1e-5 / (u(67) - u(68)) / 5.74 * 100, 5.7220, 5.7416))

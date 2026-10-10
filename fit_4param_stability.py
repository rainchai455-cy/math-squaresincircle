"""Stability experiments for r_n = sqrt(n/pi) + c + d/sqrt(n) + f/n + g/n^{3/2}.

Reproduces the reported rows, then adds the diagnostics that decide whether the
reported stability is real:
  [A] coefficients + standard errors + design condition number + predictions
  [B] extrapolation leverage and prediction SE at n=67,68 (and the step)
  [C] his hold-out-the-top test, done at two cut points
  [D] the anchored refit (r_67=5 added as a data point)
  [E] residual bootstrap spread of the r_68 prediction
  [F] the step r_68-r_67 against BOTH r_68 口径s (5.031405 derived vs 5.052404 verified)
"""
import math
import numpy as np

rng = np.random.default_rng(20261010)

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
E = np.array([R[n] for n in NS]) - np.sqrt(NS / math.pi)     # excess over the area lead
B4 = lambda n: np.column_stack([np.ones(len(np.atleast_1d(n))),
                                np.power(n, -0.5), np.power(n, -1.0), np.power(n, -1.5)])
L = lambda n: np.sqrt(np.asarray(n, dtype=float) / math.pi)
Lm = lambda n: float(np.sqrt(n / math.pi))
L67, L68 = Lm(67), Lm(68)
STEP = L68 - L67
CLAIM = {10: (0.2246, 0.6821, -2.512, 2.119, 0.0128, 4.9978, 5.0329),
         20: (0.2569, 0.4215, -0.938, 0.231, 0.0104, 4.9991, 5.0332),
         1: (None, None, None, None, None, 4.9947, 5.0300)}


def fit(lo, hi, anchor=False):
    m = (NS >= lo) & (NS <= hi)
    X = B4(NS[m]).astype(float)
    y = E[m].copy()
    if anchor:
        X = np.vstack([X, np.array([1.0, 67 ** -0.5, 67 ** -1.0, 67 ** -1.5])])
        y = np.append(y, 5.0 - L67)
    sol, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ sol
    k = 4
    n_pts = X.shape[0]
    rms = float(np.sqrt(np.mean(res ** 2)))
    sig2 = float(res @ res) / (n_pts - k)
    XtXi = np.linalg.inv(X.T @ X)
    se = np.sqrt(sig2 * np.diag(XtXi))
    return sol, se, rms, XtXi, sig2, n_pts, float(np.linalg.cond(X))


def pred(sol, n):
    return Lm(n) + float((B4(np.array([float(n)])) @ sol)[0])


print('[A] reproduce the reported rows')
print('%-7s %8s %9s %9s %9s %9s %8s | %9s %9s %8s' % (
    'window', 'c', 'd', 'f', 'g', 'RMS', 'cond', 'r67', 'r68', 'step'))
for lo in (1, 10, 20, 30):
    sol, se, rms, XtXi, sig2, N, cond = fit(lo, 50)
    p67, p68 = pred(sol, 67), pred(sol, 68)
    cl = CLAIM.get(lo)
    print('%-7s %8.4f %9.4f %9.4f %9.4f %9.5f %8.1f | %9.5f %9.5f %8.5f'
          % ('%d-50' % lo, sol[0], sol[1], sol[2], sol[3], rms, cond, p67, p68, p68 - p67))
    print('%-7s SE: c %.4f d %.4f f %.4f g %.4f   | t: c %.2f d %.2f f %.2f g %.2f'
          % ('', se[0], se[1], se[2], se[3], sol[0] / se[0], sol[1] / se[1],
             sol[2] / se[2], sol[3] / se[3]))
    if cl and cl[0] is not None:
        print('%-7s reported c,d,f,g = %.4f %.4f %.4f %.4f RMS %.4f  -> my coeffs differ by '
              '%s' % ('', cl[0], cl[1], cl[2], cl[3], cl[4],
                      np.round(sol[:4] - np.array([cl[0], cl[1], cl[2], cl[3]]), 4)))
    corr = np.corrcoef(np.column_stack([np.power(NS[NS >= lo], -0.5),
                                        np.power(NS[NS >= lo], -1.0),
                                        np.power(NS[NS >= lo], -1.5)]).T)
    print('%-7s basis correlation n^-1/2 vs n^-1: %.4f | vs n^-3/2: %.4f | n^-1 vs n^-3/2: %.4f'
          % ('', corr[0, 1], corr[0, 2], corr[1, 2]))

print('\n[B] extrapolation leverage and prediction SE at 67/68 (2 sigma bands)')
for lo in (1, 10, 20, 30):
    sol, se, rms, XtXi, sig2, N, cond = fit(lo, 50)
    out = []
    for n in (67, 68):
        x = B4(np.array([float(n)]))[0]
        h = float(x @ XtXi @ x)
        s = math.sqrt(sig2 * h)   # SE of the MODEL CURVE (no new-observation term)
        out.append((pred(sol, n), s, h, math.sqrt(h)))
    dv = B4(np.array([68.0]))[0] - B4(np.array([67.0]))[0]
    s_step = math.sqrt(sig2 * float(dv @ XtXi @ dv))
    print('    %-6s r67 %.5f +-%.5f (h=%.1f) | r68 %.5f +-%.5f (h=%.1f) | step %.5f +-%.5f'
          % ('%d-50' % lo, out[0][0], out[0][1], out[0][2], out[1][0], out[1][1], out[1][2],
             out[1][0] - out[0][0], s_step))
    print('    %-6s r68-5 = %+.5f -> %.2f SE ; |r68-5| vs its own SE: %s'
          % ('', out[1][0] - 5.0, (out[1][0] - 5.0) / out[1][1],
             'NOT resolved' if abs(out[1][0] - 5.0) < out[1][1] else 'resolved'))

print('\n[C] hold out the top and predict it (his experiment 3)')
for lo, hi, cut in ((1, 50, 40), (1, 50, 30), (1, 50, 25)):
    m_tr = (NS >= lo) & (NS <= cut)
    m_te = (NS > cut) & (NS <= hi)
    X = B4(NS[m_tr]).astype(float)
    sol, *_ = np.linalg.lstsq(X, E[m_tr], rcond=None)
    err = (L(NS[m_te]) + B4(NS[m_te]).astype(float) @ sol) - np.array([R[n] for n in NS[m_te]])
    print('    train %2d-%2d (%2d pts) -> test %2d-%2d: mean err %+.5f, RMS %.5f, '
          'max|err| %.5f' % (lo, cut, m_tr.sum(), cut + 1, hi, float(np.mean(err)),
                             float(np.sqrt(np.mean(err ** 2))), float(np.max(np.abs(err)))))

print('\n[D] anchored refit: add (n=67, r=5) as a 51st data point')
for lo in (1, 10, 20):
    sol, se, rms, XtXi, sig2, N, cond = fit(lo, 50, anchor=True)
    x68 = B4(np.array([68.0]))[0]
    s68 = math.sqrt(sig2 * float(x68 @ XtXi @ x68))
    p67, p68 = pred(sol, 67), pred(sol, 68)
    print('    %-6s c=%.4f d=%.4f f=%.4f g=%.4f RMS %.5f | r67 %.5f (anchor 5) | '
          'r68 %.5f +-%.5f' % ('%d-50' % lo, sol[0], sol[1], sol[2], sol[3], rms, p67, p68, s68))
    print('    %-6s err vs 5.031405 %+.5f (%.2f SE) ; err vs verified 5.052404 %+.5f (%.2f SE)'
          % ('', p68 - 5.031405, (p68 - 5.031405) / s68,
             p68 - 5.052404006121, (p68 - 5.052404006121) / s68))

print('\n[E] residual bootstrap of the r68 prediction (1000 draws, model 1-50)')
sol, se, rms, XtXi, sig2, N, cond = fit(1, 50)
res_all = E - B4(NS.astype(float)) @ sol
x68 = B4(np.array([68.0]))[0]
draws = []
for _ in range(1000):
    yb = B4(NS.astype(float)) @ sol + rng.choice(res_all, size=50, replace=True)
    sb, *_ = np.linalg.lstsq(B4(NS.astype(float)), yb, rcond=None)
    draws.append(L68 + float(x68 @ sb))
draws = np.array(draws)
print('    median %.5f  sd %.5f  2.5%%..97.5%% %.5f..%.5f  P(r68>5) = %.3f'
      % (float(np.median(draws)), float(np.std(draws)), float(np.percentile(draws, 2.5)),
         float(np.percentile(draws, 97.5)), float(np.mean(draws > 5))))

print('\n[F] the step r68-r67 against both 口径s')
for lo in (1, 10, 20, 30):
    s2, _, _, _, _, _, _ = fit(lo, 50)
    st = pred(s2, 68) - pred(s2, 67)
    print('    %-6s model step %.5f | area step %.5f | "observed" with r68=5.031405: '
          '%.5f | with verified 5.052404: %.5f'
          % ('%d-50' % lo, st, STEP, 5.031405 - 5.0, 5.052404006121 - 5.0))

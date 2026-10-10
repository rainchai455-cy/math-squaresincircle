"""Evaluate the three handwritten forms against a HELD-OUT validation value r_68.

Nothing here refits on the validation value: it is used only as a target after the
parameters are fixed from n=1..50 (plus the assumed anchor R(67)=5).

Blocks:
  [G] provenance gate for the supplied number 5.031425
  [V] held-out errors, required coefficient shift, and exact linear-model
      prediction SE (sigma^2 = RSS/(N-k), h = x'(X'X)^-1 x, extrapolation included)
  [D] discrimination: spread across forms vs |err|, in units of the step 0.034336
"""
import math
import numpy as np

NS = np.arange(1, 51)
R = {1: 0.707107, 2: 1.118034, 3: 1.288471, 4: 1.414214, 5: 1.581139, 6: 1.688543,
     7: 1.802776, 8: 1.97877, 9: 2.07759, 10: 2.12132, 11: 2.21386, 12: 2.23606,
     13: 2.3607, 14: 2.5, 15: 2.533, 16: 2.62309, 17: 2.67687, 18: 2.74146,
     19: 2.80150, 20: 2.893, 21: 2.91547, 22: 3.03239, 23: 3.06738, 24: 3.10940,
     25: 3.17450, 26: 3.20156, 27: 3.26050, 28: 3.33147, 29: 3.39309, 30: 3.45962,
     31: 3.52279, 32: 3.57340, 33: 3.60010, 34: 3.62852, 35: 3.64005, 36: 3.72853,
     37: 3.77126, 38: 3.83514, 39: 3.88461, 40: 3.93833, 41: 3.98879, 42: 4.03112,
     43: 4.06733, 44: 4.09999, 45: 4.14067, 46: 4.16573, 47: 4.19870, 48: 4.25610,
     49: 4.28307, 50: 4.30116}
RV = np.array([R[n] for n in NS])
LEAD = np.sqrt(NS / math.pi)
Y = RV - LEAD
L67, L68 = math.sqrt(67 / math.pi), math.sqrt(68 / math.pi)
ANCHOR_C = 5.0 - L67                      # 0.381909
STEP = L68 - L67                          # 0.034336
VAL = 5.031425                            # user-supplied validation target
VERIFIED = 5.052404006121                 # project-verified 68 construction radius

FORMS = {
    'F1  1/n': lambda n: 1.0 / n,
    'F2  1/sqrt(pi n)': lambda n: 1.0 / np.sqrt(math.pi * n),
    'F3a 1/(pi n^2)': lambda n: 1.0 / (math.pi * n ** 2),
    'F3b sqrt(n/pi+1/(pi n^2))': lambda n: np.sqrt(n / math.pi + 1.0 / (math.pi * n ** 2))
                                          - np.sqrt(n / math.pi),
}

print('[G] provenance gate on %.6f' % VAL)
print('    project verified r_68                 = %.12f  (diff %+.6f)'
      % (VERIFIED, VAL - VERIFIED))
print('    established DERIVED quantity 5/0.993758 = %.6f  (diff %+.2e)'
      % (5 / 0.993758, VAL - 5 / 0.993758))
print('    reverse scale 5/VAL                   = %.9f' % (5 / VAL))
print('    anchor + pure area step               = %.6f  (diff %+.6f = %+.4f step)'
      % (5 + STEP, VAL - (5 + STEP), (VAL - 5 - STEP) / STEP))
print('    implied excess c(68)=VAL-sqrt(68/pi)  = %.6f  vs anchor c(67) %.6f'
      % (VAL - L68, ANCHOR_C))
print('    observed step VAL-5 = %.6f vs area step %.6f -> ratio %.4f'
      % (VAL - 5, STEP, (VAL - 5) / STEP))
print('    historical excess envelope (n=1..50)  = %.6f' % Y.max())


def se_pred(xnew, X, resid, k):
    """Exact homoscedastic linear-model prediction SE at a new design row."""
    sigma2 = float(resid @ resid) / (len(resid) - k)
    XtXi = np.linalg.inv(X.T @ X)
    xn = np.asarray(xnew, dtype=float)
    h = float(xn @ XtXi @ xn)
    return math.sqrt(sigma2 * (1.0 + h)), h, math.sqrt(sigma2), XtXi


print('\n%-26s | %-25s | %-25s' % ('', 'A: R(67)=5 hard anchor', 'B: anchor as 51st pt'))
print('%-26s | %9s %5s %7s %8s | %9s %5s %7s %8s' % (
    'form', 'err', 'err/SE', 'da*', 'da*/SEa', 'err', 'err/SE', 'da*', 'da*/SEa'))
rows = []
for nm, c in FORMS.items():
    x = c(NS)
    x68 = float(c(np.array([68.0]))[0])
    c67 = float(c(np.array([67.0]))[0])
    out = []
    # ---- A: C0 = ANCHOR_C - a*c67; one free parameter, design b = x - c67
    b = x - c67
    XA = b.reshape(-1, 1)
    aA = float(-np.dot(Y - ANCHOR_C, b) / np.dot(b, b))
    rA = (Y - ANCHOR_C) - XA[:, 0] * aA
    predA = 5.0 + STEP + aA * (x68 - c67)
    seA, _, sigA, _ = se_pred([x68 - c67], XA, rA, 1)
    se_aA = sigA / math.sqrt(float(np.dot(b, b)))
    dA = (predA - VAL) / (c67 - x68)                # a-shift required to hit VAL
    out.append((predA - VAL, (predA - VAL) / seA, dA, abs(dA) / se_aA, aA, seA, predA))
    # ---- B: free (C0, a) on the 51-point excess (anchor as data point)
    nB = np.append(NS, 67.0)
    yB = np.append(Y, ANCHOR_C)
    XB = np.column_stack([np.ones(len(nB)), c(nB)])
    sol, *_ = np.linalg.lstsq(XB, yB, rcond=None)
    predB = L68 + float(sol[0] + sol[1] * x68)
    rB = (yB - XB @ sol)[:50]                       # sigma from the 50 historical pts
    seB, hB, sigB, XtXiB = se_pred([1.0, x68], XB, rB, 2)
    se_aB = sigB * math.sqrt(float(XtXiB[1, 1]))
    dB = (predB - VAL) / x68                        # a-shift at fixed C0
    out.append((predB - VAL, (predB - VAL) / seB, dB, abs(dB) / se_aB,
                float(sol[1]), seB, predB))
    rows.append((nm, out))
    print('%-26s | %9.6f %5.2f %7.3f %8.1f | %9.6f %5.2f %7.3f %8.1f'
          % (nm, out[0][0], out[0][1], out[0][2], out[0][3],
             out[1][0], out[1][1], out[1][2], out[1][3]))

print('\n[same table, scored against the VERIFIED 5.052404006121 instead]')
for nm, out in rows:
    print('%-26s A: %+.6f (%.2f SE) | B: %+.6f (%.2f SE)'
          % (nm, out[0][6] - VERIFIED, (out[0][6] - VERIFIED) / out[0][5],
             out[1][6] - VERIFIED, (out[1][6] - VERIFIED) / out[1][5]))

pA = [o[1][0][6] for o in rows]
pB = [o[1][1][6] for o in rows]
eA = [o[1][0][0] for o in rows]
eB = [o[1][1][0] for o in rows]
print('\n[D] discrimination (prediction spread across forms vs distance to target)')
print('    A: span %.2e = %.4f step ; |err| %.2e..%.2e' % (
    max(pA) - min(pA), (max(pA) - min(pA)) / STEP, min(abs(v) for v in eA),
    max(abs(v) for v in eA)))
print('    B: span %.2e = %.4f step ; |err| %.2e..%.2e' % (
    max(pB) - min(pB), (max(pB) - min(pB)) / STEP, min(abs(v) for v in eB),
    max(abs(v) for v in eB)))
print('    A errors all %+.6f..%+.6f ; B errors all %+.6f..%+.6f'
      % (min(eA), max(eA), min(eB), max(eB)))
print('    => the target sits between the two treatments; it can rank forms, '
      'not settle the anchor treatment.')

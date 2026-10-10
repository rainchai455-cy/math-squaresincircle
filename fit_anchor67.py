"""Refit the three handwritten candidate forms, with R(67)=5 imposed.

Forms (correction term only; the leading sqrt(n/pi) is fixed by area):
  F1  R(n) = sqrt(n/pi) + 1/n            + C0
  F2  R(n) = sqrt(n/pi) + 1/sqrt(pi*n)   + C0
  F3a R(n) = sqrt(n/pi) + 1/(pi*n**2)    + C0        (radical over n/pi only)
  F3b R(n) = sqrt(n/pi + 1/(pi*n**2))    + C0        (radical over the sum)
  F0  R(n) = sqrt(n/pi)                  + C0        (no correction, reference)
  FD  R(n) = sqrt(n/pi) + C0 + D/sqrt(n)             (the previously unidentifiable one)

Two fittings per form:
  (A) anchor as HARD constraint: C0 = 5 - sqrt(67/pi) - a*corr(67), `a` by LSQ on n=1..50
  (B) anchor as one more data point, free (C0, a) by LSQ
Reported: parameters, RMS / max residual on the 50 historical points, R(68), and the
decision margin 0.034335 = sqrt(68/pi)-sqrt(67/pi) that R(68) must clear to say N(5)=67.
Data: Erich Friedman squares-in-circles, n=1..50 (values are constructions => upper
bounds on the true r_n; n<=7 are proven optimal).  Nothing here is a certificate.
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
RV = np.array([R[n] for n in NS])
LEAD = np.sqrt(NS / math.pi)
Y = RV - LEAD                                   # observed excess c(n)
ANCHOR_C = 5.0 - math.sqrt(67.0 / math.pi)      # excess required by R(67)=5
STEP = math.sqrt(68.0 / math.pi) - math.sqrt(67.0 / math.pi)

FORMS = {
    'F0  none': lambda n: 0.0 * n,
    'F1  1/n': lambda n: 1.0 / n,
    'F2  1/sqrt(pi n)': lambda n: 1.0 / np.sqrt(math.pi * n),
    'F3a 1/(pi n^2)': lambda n: 1.0 / (math.pi * n ** 2),
    'F3b sqrt(n/pi+1/(pi n^2)) lead': lambda n: np.sqrt(n / math.pi + 1.0 / (math.pi * n ** 2)) - np.sqrt(n / math.pi),
    'FD  D/sqrt(n)': None,                       # handled as two free columns
}


def fit_anchor(c):
    """(A) C0 = ANCHOR_C - a*c(67); a by LSQ on the 50 points."""
    x = c(NS)
    c67 = float(c(np.array([67.0]))[0])
    b = x - c67                                   # residual = (Y - ANCHOR_C) + a*b
    den = float(np.dot(b, b))
    a = 0.0 if den == 0.0 else float(-np.dot(Y - ANCHOR_C, b) / den)
    e = Y - (ANCHOR_C - a * c67 + a * x)
    return a, ANCHOR_C - a * c67, e


def fit_free(c):
    """(B) anchor added as data point n=67,R=5; LSQ on (C0, a) for the EXCESS."""
    n = np.append(NS, 67.0)
    y = np.append(Y, ANCHOR_C)                    # excess over sqrt(n/pi), anchor included
    A = np.column_stack([np.ones(len(n)), c(n)])
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    e = y - A @ sol
    return float(sol[1]), float(sol[0]), e[:50]


def predict(a, C0, c, n):
    return math.sqrt(n / math.pi) + C0 + a * float(c(np.array([float(n)]))[0])


print('anchor excess required by R(67)=5: %.6f   step sqrt(68/pi)-sqrt(67/pi): %.6f'
      % (ANCHOR_C, STEP))
print('decision: R(68) > 5  <=>  a*(c(67)-c(68)) < %.6f\n' % STEP)
print('%-34s %10s %10s %8s %8s %9s %8s %9s' % (
    'form (A: R(67)=5 hard)', 'a*', 'C0', 'RMS50', 'max|e|', 'R(68)', 'margin', 'need|a*|'))
for nm, c in FORMS.items():
    if c is None:
        continue
    a, C0, e = fit_anchor(c)
    p68 = predict(a, C0, c, 68.0)
    x = c(NS); c67 = float(c(np.array([67.0]))[0]); c68 = float(c(np.array([68.0]))[0])
    need = STEP / abs(c67 - c68) if abs(c67 - c68) > 1e-12 else float('nan')
    print('%-34s %10.4f %10.6f %8.5f %8.5f %9.6f %8.5f %9.3g' % (
        nm, a, C0, float(np.sqrt(np.mean(e ** 2))), float(np.max(np.abs(e))),
        p68, p68 - 5.0, need))

print('%-34s %10s %10s %8s %8s %9s %8s' % (
    'form (B: anchor as data pt)', 'a*', 'C0', 'RMS50', 'max|e|', 'R(68)', 'R(67)'))
for nm, c in FORMS.items():
    if c is None:
        continue
    a, C0, e = fit_free(c)
    print('%-34s %10.4f %10.6f %8.5f %8.5f %9.6f %9.6f' % (
        nm, a, C0, float(np.sqrt(np.mean(e ** 2))), float(np.max(np.abs(e))),
        predict(a, C0, c, 68.0), predict(a, C0, c, 67.0)))

# two-free-column reference fit (C0 + D/sqrt n), the previously unidentifiable variant
X = np.column_stack([np.ones(51), 1.0 / np.sqrt(np.append(NS, 67.0))])
y = np.append(Y, ANCHOR_C)
sol, *_ = np.linalg.lstsq(X, y, rcond=None)
e = (y - X @ sol)[:50]
print('\nFD  C0 + D/sqrt(n), anchor as data point: C0=%.6f D=%.6f RMS50=%.5f max|e|=%.5f'
      ' R(68)=%.6f' % (sol[0], sol[1], float(np.sqrt(np.mean(e ** 2))),
                       float(np.max(np.abs(e))),
                       math.sqrt(68 / math.pi) + sol[0] + sol[1] / math.sqrt(68.0)))

# self-check on a proven point: model must reproduce r_4 = sqrt(2) exactly-ish
for nm, c in FORMS.items():
    if c is None:
        continue
    a, C0, _ = fit_free(c)
    print('check %-32s R(4)=%.6f (exact 1.414214)  R(67)=%.6f (anchor 5.0)'
          % (nm, predict(a, C0, c, 4.0), predict(a, C0, c, 67.0)))

# residual scale vs decision step, and the sign-of-correction test
a, C0, e = fit_anchor(FORMS['F2  1/sqrt(pi n)'])
print('\nsign test: proven n<=7 excess %s' % np.round(Y[:7], 5).tolist())
print('           n=41..50 excess      %s' % np.round(Y[40:], 5).tolist())
print('all three corrections DECAY in n, i.e. the model can only produce a falling')
print('excess, while the data rise from %.3f (n=1) to a ~%.3f plateau.' % (Y[0], float(np.mean(Y[20:]))))
print('\nverified 68 construction r_68<=5.052404 vs model predictions above;')
print('RMS50 / step = %.2f  (model noise as a multiple of the 67-vs-68 step)'
      % (float(np.sqrt(np.mean(fit_anchor(FORMS['F1  1/n'])[2] ** 2))) / STEP))

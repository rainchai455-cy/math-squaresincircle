"""Leave-out test for r_n = sqrt(n/pi) + c + d/sqrt(n*pi), plus arithmetic checks.

Blocks:
  [P] provenance / arithmetic audit of the supplied derivation (e_67, e_68, d, c and
      the spot values quoted at n=50, n=7, n=1)
  [I] identity check: this family is a reparametrisation of c + D/sqrt(n) (D = d/sqrt(pi)),
      so it cannot change fit quality -- asserted numerically
  [L] THE leave-out test: (c,d) fitted on n=1..50 ONLY, then r_67, r_68 predicted.
      Windows reported because the small-n region is not in the same regime.
  [T] two-point model implied by (r_67, r_68): tension with the n=1..50 cloud.
  [N] exact inversion N(R)=max{n: r_n<=R} under this model + its large-R expansion.

All numbers are fit values / construction values, not certificates.
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
PROVEN = 7          # n <= 7 are proven optimal; the rest are constructions (upper bounds)
TRUNC = 5e-4        # '+' truncation in the published decimals

L = lambda n: np.sqrt(n / math.pi)
U = lambda n: 1.0 / np.sqrt(n * math.pi)
NS = np.arange(1, 51)
RVv = np.array([R[n] for n in NS])
EV = RVv - np.sqrt(NS / math.pi)

L67, L68 = L(67), L(68)
A67 = 5.0 - L67                                   # e_67 required by r_67 = 5
STEP = L68 - L67
R68_A = 5.031415                                  # this message
R68_B = 5.031425                                  # previous message
R68_V = 5.052404006121                            # project-verified construction

print('[P] arithmetic audit')
print('    sqrt(67/pi)=%.6f  sqrt(68/pi)=%.6f  area step=%.6f  anchor e_67=%.6f'
      % (L67, L68, STEP, A67))
for tag, val in (('r68=5.031415 (this msg)', R68_A), ('r68=5.031425 (prev msg)', R68_B),
                 ('r68=5.052404 (verified)', R68_V)):
    e68 = val - L68
    d = (A67 - e68) / (U(67) - U(68))
    c = A67 - d * U(67)
    print('    %-26s e68=%.6f  de=%.6f  d=%.4f  c=%.6f   [quoted: 0.378989 / 0.002921 / '
          '5.742 / -0.01384]' % (tag, e68, A67 - e68, d, c))
d_q, c_q = 5.742, -0.01384
print('    spot checks of the quoted two-point model r_n = sqrt(n/pi) %+.5f + %.3f/sqrt(n pi):'
      % (c_q, d_q))
for n in (1, 4, 7, 50, 67, 68):
    pred = L(n) + c_q + d_q * U(n)
    print('      n=%-3d model %8.5f | published %8.5f | err %+8.5f (%+6.4f step)  %s'
          % (n, pred, R.get(n, (5.0 if n == 67 else R68_A)),
             pred - R.get(n, (5.0 if n == 67 else R68_A)),
             (pred - R.get(n, (5.0 if n == 67 else R68_A))) / STEP,
             'OK' if abs(pred - R.get(n, 5.0)) < 0.002 else 'MISMATCH'))
print('      n<=7 proven values: max err %.5f at n=%d' % tuple(
    (lambda es: (float(np.max(np.abs(es))), int(NS[np.argmax(np.abs(es))]) + 1))(
        np.array([L(n) + c_q + d_q * U(n) - R[n] for n in range(1, 8)]))))

# ---- [I] reparametrisation identity
Xpi = np.column_stack([np.ones(50), U(NS)])
Xsq = np.column_stack([np.ones(50), 1.0 / np.sqrt(NS)])
s1 = np.linalg.lstsq(Xpi, EV, rcond=None)[0]
s2 = np.linalg.lstsq(Xsq, EV, rcond=None)[0]
print('\n[I] identity: d/sqrt(n pi) == (d/sqrt(pi))/sqrt(n), so the earlier c+D/sqrt(n) '
      'family is the SAME model')
print('    50-pt fit here: c=%.6f d=%.6f | D=d/sqrt(pi)=%.6f | earlier family: c=%.6f '
      'D=%.6f' % (s1[0], s1[1], s1[1] / math.sqrt(math.pi), s2[0], s2[1]))
print('    RSS identical to %.2e  =>  the "more natural" scale changes bookkeeping only'
      % abs(float(EV @ EV - 2 * s1 @ (Xpi.T @ EV) + s1 @ (Xpi.T @ Xpi) @ s1)
            - float(EV @ EV - 2 * s2 @ (Xsq.T @ EV) + s2 @ (Xsq.T @ Xsq) @ s2)))


def fit_window(lo, hi):
    m = (NS >= lo) & (NS <= hi)
    X = np.column_stack([np.ones(int(m.sum())), U(NS[m])])
    y = EV[m]
    sol, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ sol
    n_pts = int(m.sum())
    sig2 = float(res @ res) / (n_pts - 2)
    XtXi = np.linalg.inv(X.T @ X)
    se_c = math.sqrt(sig2 * XtXi[0, 0])
    se_d = math.sqrt(sig2 * XtXi[1, 1])
    rms = math.sqrt(float(res @ res) / n_pts)
    return sol[0], sol[1], se_c, se_d, rms, XtXi, n_pts


def N_of_R(c, d, Rv):
    """Exact inversion of R = sqrt(n/pi) + c + d/sqrt(n pi): x=sqrt(n/pi) solves
    x^2 - (R-c)x + d/pi = 0; N = pi x_+^2. Returns (N, n_at_min or None, R_min or None)."""
    S = Rv - c
    disc = S * S - 4.0 * d / math.pi
    if disc < 0:
        return float('nan'), (d if d > 0 else None), None
    x = 0.5 * (S + math.sqrt(disc))
    if d > 0:
        return math.pi * x * x, d, 2.0 * math.sqrt(d / math.pi) + c
    return math.pi * x * x, None, None


print('\n[L] LEAVE-OUT: (c,d) from n in window only; 67 and 68 never used')
print('%-12s %4s %9s %8s %9s %8s %8s | %9s %8s | %9s %8s' % (
    'window', 'N', 'c', 'se_c', 'd', 'se_d', 'RMS', 'pred r67', 'err', 'pred r68', 'err'))
for lo, hi in ((1, 50), (2, 50), (8, 50), (11, 50), (21, 50), (31, 50), (41, 50)):
    c, d, se_c, se_d, rms, XtXi, N = fit_window(lo, hi)
    p67 = L67 + c + d * U(67)
    p68 = L68 + c + d * U(68)
    s67 = math.sqrt((1 + np.array([1.0, U(67)]) @ XtXi @ np.array([1.0, U(67)])))
    s68 = math.sqrt((1 + np.array([1.0, U(68)]) @ XtXi @ np.array([1.0, U(68)])))
    s67 *= rms * math.sqrt(N / (N - 2.0))
    s68 *= rms * math.sqrt(N / (N - 2.0))
    # delta-method SE of N(5)=pi x_+^2 from the 2x2 covariance of (c,d)
    C = (rms ** 2) * (N / (N - 2.0)) * XtXi
    h = 1e-6
    g = np.array([(N_of_R(c + h, d, 5.0)[0] - N_of_R(c - h, d, 5.0)[0]) / (2 * h),
                  (N_of_R(c, d + h, 5.0)[0] - N_of_R(c, d - h, 5.0)[0]) / (2 * h)])
    se_N = math.sqrt(float(g @ C @ g))
    print('%-12s %4d %9.6f %8.6f %9.6f %8.6f %8.5f | %9.6f %+8.5f | %9.6f %+8.5f'
          % ('%d-%d' % (lo, hi), N, c, se_c, d, se_d, rms, p67, p67 - 5.0, p68,
             p68 - R68_A))
    print('%-12s %4s r68 pred is %+.5f %s 5 with SE %.5f -> z = %+.2f | N(5) = %.3f +- %.3f'
          % ('', '', p68 - 5.0, 'above' if p68 > 5 else 'below', s68,
             (p68 - 5.0) / s68, N_of_R(c, d, 5.0)[0], se_N))
    print('%-12s %4s pred r67 err = %.2f prediction-SE | err/area-step = %.3f'
          % ('', '', (p67 - 5.0) / s67, (p67 - 5.0) / STEP))
    print('%-12s %4s => model N(5) = %d (r68 pred %.5f %s 5) | err vs verified r68 '
          '5.052404: %+.5f'
          % ('', '', 67 if p68 > 5 else 68, p68, '>' if p68 > 5 else '<',
             p68 - R68_V))

print('\n[T] tension between the two-point model and the n=1..50 cloud')
c50, d50, se_c50, se_d50, rms50, _, _ = fit_window(1, 50)
print('    50-pt leave-out fit: c=%.6f+-%.6f  d=%.6f+-%.6f' % (c50, se_c50, d50, se_d50))
print('    two-point model:     c=%.6f          d=%.6f' % (c_q, d_q))
print('    d shift needed = %.4f = %.1f SE(d) ; c shift = %.4f = %.1f SE(c)'
      % (d_q - d50, (d_q - d50) / se_d50, c_q - c50, (c_q - c50) / se_c50))
print('    RMS50 of the two-point model on n=1..50 = %.5f (vs best-fit RMS %.5f)'
      % (float(np.sqrt(np.mean((EV - (c_q + d_q * U(NS))) ** 2))), rms50))
print('    max |residual| of the two-point model on n=1..50 = %.5f at n=%d'
      % tuple((lambda a: (float(np.max(np.abs(a))), int(NS[np.argmax(np.abs(a))])))(
          EV - (c_q + d_q * U(NS)))))
print('    implied d if c is held at %.5f: ' % c_q, end='')
print(' '.join('n=%d:%.2f' % (n, (R[n] - L(n) - c_q) / U(n)) for n in (1, 4, 7, 25, 41, 50)))


def N_of_R(c, d, Rv):
    """Exact inversion of R = sqrt(n/pi) + c + d/sqrt(n pi): x=sqrt(n/pi) solves
    x^2 - (R-c)x + d/pi = 0; N = pi x_+^2. Returns (N, n_at_min or None, R_min or None)."""
    S = Rv - c
    disc = S * S - 4.0 * d / math.pi
    if disc < 0:
        return float('nan'), (d if d > 0 else None), None
    x = 0.5 * (S + math.sqrt(disc))
    if d > 0:
        return math.pi * x * x, d, 2.0 * math.sqrt(d / math.pi) + c
    return math.pi * x * x, None, None


print('\n[N] inversion  N(R) = max{n : r_n <= R}')
print('    closed form: x=sqrt(n/pi) solves x^2-(R-c)x+d/pi=0, x_+ = [(R-c)+sqrt((R-c)^2'
      '-4d/pi)]/2, N = pi x_+^2')
print('    large-R:  N(R) = pi (R-c)^2 - 2d + O(1/(R-c))   ->  the 1/sqrt(n pi) term enters'
      ' at O(1), NOT as an O(R) perimeter term')
print('%-12s %9s %9s %10s %9s %9s' % ('window', 'c', 'd', 'N(5) exact', 'floor', 'dN/dd'))
for lo, hi in ((1, 50), (2, 50), (8, 50), (11, 50), (21, 50), (31, 50), (41, 50)):
    c, d, se_c, se_d, rms, XtXi, N = fit_window(lo, hi)
    Nv, n_min, R_min = N_of_R(c, d, 5.0)
    print('%-12s %9.6f %9.6f %10.4f %9d %9.3f' % ('%d-%d' % (lo, hi), c, d, Nv,
                                                  int(math.floor(Nv)),
                                                  -2.0 if d > 0 else -2.0))
    if n_min is not None:
        print('             model minimum at n=%.3f (R_min=%.4f): r_n DECREASING below that'
              % (n_min, R_min))
for tag, (c, d) in (('two-point quoted (c=%.5f,d=%.3f)' % (c_q, d_q), (c_q, d_q)),):
    Nv, n_min, R_min = N_of_R(c, d, 5.0)
    S = 5.0 - c
    print('    %-32s N(5)=%.4f -> floor %d ; pi(R-c)^2-2d = %.4f (diff %.4f)'
          % (tag, Nv, int(math.floor(Nv)), math.pi * S * S - 2 * d,
             Nv - (math.pi * S * S - 2 * d)))
    print('        with d rounded to 3 decimals the model gives r_67 = %.5f > 5, so its own'
          % (L67 + c + d * U(67)))
    print('        N(5) lands %.4f below 67 -> floor flips with any d-error > %.1e (dN/dd=-2)'
          % (67 - Nv, (67 - Nv) / 2))
print('    area ceiling N(5)<=floor(pi*25)=%d unchanged; none of these are certificates.'
      % int(math.floor(math.pi * 25)))

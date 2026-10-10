"""Invert r_n = sqrt(n/pi) + c + d/sqrt(n) + f/n + g/n^{3/2} for n.

His algebra: with x=sqrt(n), r/1: (1/sqrt(pi))x^4 - (r-c)x^3 + d x^2 + f x + g = 0,
n = x^2 at the positive root; 1st order n ~ pi(r-c)^2; with the d-correction
n ~ (sqrt(pi)(r-c) - d/(r-c))^2.

Checks here:
  [P] the algebra itself (coefficient map, round-trip r -> n -> r)
  [M] monotonicity: where dr/dx changes sign -> is "the positive root" unique?
  [Q] N(R) = largest positive root vs the literal max{n integer: model(n)<=R}
  [A] every coefficient set: LSQ (mine) vs his reported tuples
  [S] delta-method SE of N(5) and the asymptotic-series truncation error
  [T] tautology test: does N(5.031405)=68 say anything?
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
E = np.array([R5[int(n)] for n in NS]) - np.sqrt(NS / math.pi)
SPI = math.sqrt(math.pi)
DERIVED = 5.031405          # = 5/0.993758, a scaled equivalence radius, NOT measured
VERIFIED = 5.052404006121   # project verified unit-square r_68 口径

B4 = lambda n: np.column_stack([np.ones(len(np.atleast_1d(n))), np.power(n, -0.5),
                                np.power(n, -1.0), np.power(n, -1.5)])
Lm = lambda n: math.sqrt(n / math.pi)
model = lambda p, n: Lm(n) + float((B4(np.array([float(n)])) @ p)[0])


def fit(lo, hi, anchor=False):
    sel = (NS >= lo) & (NS <= hi)
    x, y = NS[sel], E[sel]
    if anchor:
        x = np.append(x, 67.0)
        y = np.append(y, 5.0 - Lm(67.0))
    X = B4(x)
    sol, *_ = np.linalg.lstsq(X, y, rcond=None)
    k = 4
    npts = len(x)
    e = (y - X @ sol)[:50 if npts > k else npts] if npts > k else y - X @ sol
    e = y - X @ sol
    e = e[:50]                                     # sigma from historical pts only
    dof = max(npts - k, 1)
    sig2 = float(e @ e) / dof
    rms = float(np.sqrt(np.mean(e ** 2)))
    return sol, sig2 * np.linalg.inv(X.T @ X), rms, npts


def roots4(c, d, f, g, rv):
    """positive real roots of (1/sqrt(pi))x^4-(r-c)x^3+d x^2+f x+g=0."""
    rts = np.roots([1.0 / SPI, -(rv - c), d, f, g])
    return sorted(z.real for z in rts if abs(z.imag) < 1e-8 and z.real > 0.0)


def N_root(p, rv):
    pos = roots4(*p, rv)
    return (pos[-1] ** 2 if pos else float('nan')), len(pos)


def N_int(p, rv, cap=400):
    s = [n for n in range(1, cap + 1) if model(p, n) <= rv]
    return (max(s) if s else 0), len(s)


def N_asym1(rv, c):
    return math.pi * (rv - c) ** 2


def N_asym2(rv, c, d):
    R = rv - c
    return (SPI * R - d / R) ** 2


def monotone(p):
    c, d, f, g = [float(v) for v in p]
    # dr/dx = 1/sqrt(pi) - d/x^2 - 2f/x^3 - 3g/x^4  -> sign changes at roots of
    xr = np.roots([1.0 / SPI, 0.0, -d, -2.0 * f, -3.0 * g])
    pos = sorted(z.real for z in xr if abs(z.imag) < 1e-8 and z.real > 0.0)
    return pos


print('[P] algebra audit / round trip with the true 1-50 LSQ coefficients')
p150, cov150, rms150, n150 = fit(1, 50)
print('    LSQ 1-50   c=%.4f d=%.4f f=%.4f g=%.4f  RMS=%.5f' %
      (p150[0], p150[1], p150[2], p150[3], rms150))
for n in (4.0, 50.0, 67.0, 68.0):
    rv = model(p150, n)
    nn, nr = N_root(p150, rv)
    print('    n=%-5.1f model r=%.6f -> invert: largest root n=%.6f (%d positive roots)'
          '  err=%.2e' % (n, rv, nn, nr, nn - n))

print('\n[M] monotonicity of the model curve in x=sqrt(n)  (dr/dx=0 locations)')
for nm, p in [('LSQ 1-50', p150)]:
    pos = monotone(p)
    print('    %s sign-change x = %s -> n = %s' %
          (nm, ['%.4f' % z for z in pos], ['%.3f' % (z * z) for z in pos]))
    print('    model decreasing for n below the largest such n => "the positive root"'
          ' is ambiguous when R is small')

SETS = [('LSQ 1-50', p150), ('LSQ 10-50', fit(10, 50)[0]), ('LSQ 20-50', fit(20, 50)[0]),
        ('LSQ 30-50', fit(30, 50)[0]), ('anchored LSQ 1-50+67', fit(1, 50, True)[0]),
        ('his exp1 (claims 10-50)', np.array([0.2246, 0.6821, -2.512, 2.119])),
        ('his exp2 (claims 20-50)', np.array([0.2569, 0.4215, -0.938, 0.231])),
        ('his 2-term+e/n (1-50)', np.array([0.3683, -0.1350, 0.2768, -0.3608]))]

print('\n[Q] N(R): largest-root vs literal max{n: model(n)<=R}, and root count')
RS = [('R=5', 5.0), ('R=4.99', 4.99), ('R=5.031405', DERIVED),
      ('R=5.052404 (verified)', VERIFIED), ('R=5.10', 5.10)]
print('    %-24s %-8s %-9s %-9s %s' % ('set', 'r68', 'r69', 'r70',
                                        ' | '.join('%s' % a for a, _ in RS)))
for nm, p in SETS:
    cells = []
    for _, rv in RS:
        nr = N_root(p, rv)[1]
        cells.append('%d/%d' % (N_root(p, rv)[0], nr))
    print('    %-24s %8.4f %9.4f %8.4f  %s' %
          (nm, model(p, 68.0), model(p, 69.0), model(p, 70.0), '  '.join(cells)))
print('    (cells are N(R)/#positive-roots)')

print('\n[Q2] agreement between the two definitions of N(R)')
bad = 0
for nm, p in SETS:
    for _, rv in RS:
        a = N_root(p, rv)[0]
        b, cnt = N_int(p, rv)
        if not (abs(a - b) <= 1.0):
            bad += 1
            print('    MISMATCH %-24s R=%.6f root %.3f vs integer-count %d (count=%d set)'
                  % (nm, rv, a, b, cnt))
print('    mismatches = %d out of %d' % (bad, len(SETS) * len(RS)))

print('\n[S] truncation error of his asymptotic inversion (set = LSQ 1-50)')
print('    %-22s %10s %10s %10s %10s' % ('R', 'pi(r-c)^2', 'with -d term', 'quartic',
                                         'exact N'))
for nm, rv in RS + [('R=r68(model)', model(p150, 68.0))]:
    print('    %-22s %10.3f %10.3f %10.3f %10.3f' %
          (nm, N_asym1(rv, p150[0]), N_asym2(rv, p150[0], p150[1]),
           N_root(p150, rv)[0], N_int(p150, rv)[0]))
print('    => the 2-term closed form misses f,g; how much is a function of the window')

print('\n[S2] SE of N(5) by delta method (model-curve covariance)')
for nm, lo in (('1-50', 1), ('10-50', 10), ('20-50', 20)):
    p, cov, rms, npts = fit(lo, 50)
    g = np.zeros(4)
    base = N_root(p, 5.0)[0]
    for i in range(4):
        h = 1e-5 * max(1.0, abs(p[i]))
        pp, pm = list(p), list(p)
        pp[i] += h
        pm[i] -= h
        g[i] = (N_root(np.array(pp), 5.0)[0] - N_root(np.array(pm), 5.0)[0]) / (2 * h)
    se = math.sqrt(float(g @ cov @ g))
    print('    %-6s N(5)=%.4f  dN/dc=%.2f dN/dd=%.2f dN/df=%.2f dN/dg=%.2f  SE=%.3f'
          '  -> 67/68 resolved? %s' % (nm, base, g[0], g[1], g[2], g[3], se,
                                       'NO (SE covers 67.5)' if se > 0.5 else 'see value'))

print('\n[T] tautology test on "N(5.031405) ~ 68 => the estimator works"')
for nm, p in SETS:
    r68, r69 = model(p, 68.0), model(p, 69.0)
    inn = r68 <= DERIVED < r69
    innv = r68 <= VERIFIED < r69
    print('    %-24s r68=%.5f r69=%.5f | 5.031405 inside [r68,r69): %-5s -> N=%d'
          ' | verified inside: %-5s -> N=%d' %
          (nm, r68, r69, inn, N_int(p, DERIVED)[0], innv, N_int(p, VERIFIED)[0]))
print('    N(R)=68 iff model(68)<=R<model(69); the window is one model step wide')
print('    print step width = r69-r68 = %.5f (area step %.5f)' %
      (model(p150, 69.0) - model(p150, 68.0), math.sqrt(69 / math.pi) - math.sqrt(68 / math.pi)))
print('    so feeding R=5.031405 (itself a claimed r68) and reading back 68 is a')
print('    round trip through the same model, not an independent check.')

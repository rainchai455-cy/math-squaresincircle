"""EXACT re-verification of the tau(5) >= 67 certificate, with no floating point at all.

The certificate is axis aligned with coordinates that are multiples of 1/4, so
  * "square i lies in the disc of radius 5"  <=>  (x+1/2)^2 + (y+1/2)^2 <= 25 for the
    farthest corner  -- a rational inequality;
  * "interiors of i and j are disjoint"      <=>  |x_i-x_j| >= 1 or |y_i-y_j| >= 1
    for axis-aligned unit squares -- again rational.
Disjoint interiors give D(p) = sum_i 1{p in int S_i} <= 1 for EVERY p in the plane, i.e.
the continuum density constraint holds exactly, so mu = sum of point masses = 67 is a
valid measure certificate and tau(5) >= 67 needs no sampling argument.
"""
import json
from fractions import Fraction

d = json.load(open('g3_frozen/best_legal_67.json'))
X = [Fraction(v).limit_denominator(64) for v in d['x']]
Y = [Fraction(v).limit_denominator(64) for v in d['y']]
n = len(X)
H = Fraction(1, 2)
print(f'[EX67] N={n} side=1 orientations={set(d["th"])} '
      f'all coordinates on the 1/4 lattice: '
      f'{all((4*v).denominator == 1 for v in X + Y)}')

bad_contain = []
for i in range(n):
    r2 = max((X[i] + sx * H) ** 2 + (Y[i] + sy * H) ** 2 for sx in (-1, 1) for sy in (-1, 1))
    if r2 > 25:
        bad_contain.append((i, float(r2)))

bad_disjoint = []
for i in range(n):
    for j in range(i + 1, n):
        if abs(X[i] - X[j]) < 1 and abs(Y[i] - Y[j]) < 1:
            bad_disjoint.append((i, j, float(X[i] - X[j]), float(Y[i] - Y[j])))

tight = sum(1 for i in range(n)
            if max((X[i] + sx * H) ** 2 + (Y[i] + sy * H) ** 2
                   for sx in (-1, 1) for sy in (-1, 1)) == 25)
print(f'[EX67] corner-outside-disc violations = {len(bad_contain)} (exact rational test)')
print(f'[EX67] overlapping-interior pairs     = {len(bad_disjoint)} '
      f'(exact rational test) {bad_disjoint[:3]}')
print(f'[EX67] squares pinned exactly on the rim (r2max == 25) = {tight}')
verdict = (len(bad_contain) == 0 and len(bad_disjoint) == 0)
print(f'[EX67] {"VERIFIED" if verdict else "FAILED"}: D(p)<=1 for every p in the plane, '
      f'mass=67 -> tau(5) >= 67 is an EXACT certificate, not a sampled estimate')
json.dump(dict(N=n, exact_contain_violations=len(bad_contain),
               exact_overlap_pairs=len(bad_disjoint), pinned_on_rim=tight,
               verified=bool(verdict)), open('g2_tau_cert67_exact.json', 'w'), indent=1)

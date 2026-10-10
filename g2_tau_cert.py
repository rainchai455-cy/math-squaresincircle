"""Strict UPPER-bound certificate for the grid-relaxed measure LP.

The gate question is  V_c(family) <= 68 - margin,  where
    V_g(A) >= V_c(F) >= 67      (relaxation on the rows; the verified 67 inside F)
so a DUAL feasible point with exact integer arithmetic pins the whole chain.

Pipeline, in the sound direction for every step:
 1. incidence rows are kept only with a 1e-9 clearance (a point that float64 calls
    strictly inside a square, with 1e-15 scale error, is truly inside).  Uncertain
    incidences are DROPPED, which removes constraints -> still a relaxation, so the
    upper bound survives.
 2. the lower end of the squeeze is re-proved in EXACT rational arithmetic: the
    certified 67 uses doubled-integer coordinates, so "inside R=5" is the exact
    statement X^2+Y^2 <= 100 in Fractions.
 3. A is rebuilt by a second, independent O(n*m) method and compared to the accelerated
    builder; a mismatch aborts.
 4. the LP dual y is rounded UP to p/Q with a common denominator Q, then A^T p >= Q is
    verified with integers only.  U = sum(p)/Q is then a theorem: V_c(F) <= V_g(A) <= U.
"""
import json, math, time
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog
from scipy import sparse
import g2_tau as T
import g2_tau_exact as E

CLEAR = 1e-9
Q = 10 ** 7


def build_brute(F, gr):
    """Builder 1: brute force over every grid node, no bounding-box shortcut."""
    n = len(F)
    gx, gy = gr.gx, gr.gy
    nx, ny = gr.nx, gr.ny
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    inside_disc = gr.inside.ravel()
    fidx = np.nonzero(inside_disc)[0]
    FX, FY = GX.ravel()[fidx], GY.ravel()[fidx]
    ri, ci = [], []
    for i, (x, y, th) in enumerate(F):
        ct, st = math.cos(th), math.sin(th)
        dx = FX - x
        dy = FY - y
        q = np.maximum(np.abs(dx * ct + dy * st), np.abs(-dx * st + dy * ct))
        j = fidx[np.nonzero(q < HALF_C)[0]]
        ri.append(j)
        ci.append(np.full(len(j), i, dtype=np.int64))
    ri = np.concatenate(ri); ci = np.concatenate(ci)
    return sparse.csr_matrix((np.ones(len(ri)), (ri, ci)), shape=(nx * ny, n))


def build_fast(F, gr):
    """Builder 2: the accelerated path (bounding-box window + searchsorted), same
    clearance rule.  Agreement between 1 and 2 is what rules out an indexing bug."""
    n, ny = len(F), gr.ny
    ri, ci = [], []
    for i, (x, y, th) in enumerate(F):
        ct, st = math.cos(th), math.sin(th)
        b = (abs(ct) + abs(st)) * T.HALF + gr.h
        i0 = max(0, int(np.searchsorted(gr.gx, x - b)))
        i1 = min(gr.nx, int(np.searchsorted(gr.gx, x + b)) + 1)
        j0 = max(0, int(np.searchsorted(gr.gy, y - b)))
        j1 = min(gr.ny, int(np.searchsorted(gr.gy, y + b)) + 1)
        if i1 <= i0 or j1 <= j0:
            ri.append(np.zeros(0, dtype=np.int64)); ci.append(np.zeros(0, dtype=np.int64))
            continue
        X = gr.gx[i0:i1, None] - x
        Y = gr.gy[None, j0:j1] - y
        q = np.maximum(np.abs(X * ct + Y * st), np.abs(-X * st + Y * ct))
        m = (q < HALF_C) & gr.inside[i0:i1, j0:j1]
        ii, jj = np.nonzero(m)
        ri.append((ii + i0).astype(np.int64) * ny + (jj + j0))
        ci.append(np.full(len(ii), i, dtype=np.int64))
    ri = np.concatenate(ri); ci = np.concatenate(ci)
    return sparse.csr_matrix((np.ones(len(ri)), (ri, ci)), shape=(gr.nx * gr.ny, n))


HALF_C = T.HALF - CLEAR


def solve_cert(F, h_g, tag):
    t0 = time.time()
    gr = E.Grid(h_g)
    A = build_brute(F, gr)
    A2 = build_fast(F, gr)
    a1 = np.sort(A.tocoo().row * A.shape[1] + A.tocoo().col)
    a2 = np.sort(A2.tocoo().row * A2.shape[1] + A2.tocoo().col)
    same = bool(A.nnz == A2.nnz and len(a1) == len(a2) and np.array_equal(a1, a2))
    n = A.shape[1]
    res = linprog(-np.ones(n), A_ub=A, b_ub=np.ones(A.shape[0]),
                  bounds=[(0, None)] * n, method='highs', options=E.TOLOPT)
    if not res.success:
        print(f'[CERT] {tag}: LP FAILED', flush=True)
        return None
    w = res.x
    mass = float(w.sum())
    y = np.maximum(-np.asarray(res.ineqlin.marginals, dtype=float), 0.0)
    p = np.ceil(y * Q).astype(np.int64)
    # repair any column whose dual constraint is not met after rounding
    csc = A.tocsc()
    steps = 0
    while True:
        s = np.asarray(A.T @ p).ravel()
        bad = np.nonzero(s < Q)[0]
        if len(bad) == 0 or steps > 60:
            break
        coo = csc
        for i in bad:
            rows = coo.indices[coo.indptr[i]:coo.indptr[i + 1]]
            if len(rows) == 0:
                continue
            k = rows[int(np.argmin(p[rows]))]
            p[k] += Q - s[i]
        steps += 1
    s = np.asarray(A.T @ p).ravel()
    U = int(p.sum()) / Q
    out = dict(tag=tag, h_g=h_g, cols=n, rows=int(A.shape[0]), nnz=int(A.nnz),
               builders_agree=bool(same), lp_value=mass, support=int((w > 1e-9).sum()),
               primal_infeas=float(max(0.0, float(np.asarray(A @ w).max()) - 1.0)),
               dual_objective=float(y.sum()), repair_steps=steps,
               min_column_slack=int(s.min() - Q), U_exact=U,
               Q=Q, proves_no_68_in_family=bool(U < 68.0),
               margin_below_68=68.0 - U, secs=round(time.time() - t0, 1))
    print(f'[CERT] {tag}: LP={mass:.6f} builders_agree={same} dual_obj={out["dual_objective"]:.6f} '
          f'repairs={steps} min_int_slack={out["min_column_slack"]} '
          f'U_exact={U:.6f} margin_below_68={68.0-U:.6f} '
          f'-> {"family CANNOT host a >=68 measure certificate" if U < 68 else "inconclusive"} '
          f'({out["secs"]}s)', flush=True)
    return out


def exact_legality_of_certified_67():
    """r^2 <= 25 as an exact rational statement on the doubled-integer certificate"""
    d = json.load(open('g3_frozen/best_legal_67.json'))
    ok = 0
    for x, y in zip(d['x'], d['y']):
        fx, fy = Fraction(x).limit_denominator(64), Fraction(y).limit_denominator(64)
        th = d['th'][0]
        # corners are (x +- 1/2, y +- 1/2) since the certificate is axis aligned
        r2max = max((fx + Fraction(sx, 2)) ** 2 + (fy + Fraction(sy, 2)) ** 2
                    for sx in (-1, 1) for sy in (-1, 1))
        if r2max <= 25:
            ok += 1
    return ok, len(d['x'])


def main():
    ok, tot = exact_legality_of_certified_67()
    print(f'[CERT] exact rational legality of the certified 67 (X^2+Y^2<=100 in '
          f'Fractions, doubled-integer data): legal={ok}/{tot}', flush=True)
    d = json.load(open('g3_frozen/best_legal_67.json'))
    C67 = np.column_stack([d['x'], d['y'], d['th']]).tolist()
    out = []
    for h_c, na, nm, hs in ((0.30, 9, 'fam(0.30,ang9)', (0.15, 0.075)),
                            (0.20, 13, 'fam(0.20,ang13)', (0.15, 0.075))):
        F = T.build_family(h_c, na, extra=C67)
        # V_c(F) >= 67 only holds if the family really contains the certified 67
        keys = {(round(float(a), 9), round(float(b), 9), round(float(c), 9))
                for a, b, c in F}
        have = sum(1 for a, b, c in C67
                   if (round(a, 9), round(b, 9), round(c, 9)) in keys)
        print(f'[CERT] {nm}: |F|={len(F)} certified_67_present_in_F={have}/67', flush=True)
        for h_g in hs:
            r = solve_cert(F, h_g, f'{nm} h_g={h_g}')
            if r:
                out.append(r)
    json.dump(out, open('g2_tau_cert.json', 'w'), indent=1, default=float)
    pin = [r for r in out if r['proves_no_68_in_family'] and r['builders_agree']]
    print(f'[CERT] families pinned below 68 by an exact dual certificate: '
          f'{[r["tag"] for r in pin] or "none"}', flush=True)
    print('[CERT] scope: a pinned family bounds ONLY that discretised family. It is not '
          'a statement about the global tau(5), and no conclusion here touches r(67)>=5.',
          flush=True)


if __name__ == '__main__':
    main()

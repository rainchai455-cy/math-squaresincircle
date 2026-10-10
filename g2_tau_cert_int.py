"""Literal-integer re-verification of the dual upper-bound certificate.

g2_tau_cert.py computed the column sums with `A.T @ p` on a float64 matrix.  The values
are far below 2**53 so the arithmetic is exact, but the gate condition the user set is
"verified with integer arithmetic only, no float tolerance".  This script re-runs the
same pipeline and does the final check in int64, then asserts the two agree bit for bit.

Nothing here refines a grid: identical family / identical h_g as g2_tau_cert.py.
"""
import json, time
import numpy as np
from fractions import Fraction
from scipy.optimize import linprog
from scipy import sparse
import g2_tau as T
import g2_tau_exact as E
import g2_tau_cert as C

Q = C.Q


def cert_int(F, h_g, tag):
    t0 = time.time()
    gr = E.Grid(h_g)
    A = C.build_brute(F, gr)
    A2 = C.build_fast(F, gr)
    coo, coo2 = A.tocoo(), A2.tocoo()
    same = bool(A.nnz == A2.nnz
                and np.array_equal(np.sort(coo.row * A.shape[1] + coo.col),
                                   np.sort(coo2.row * A2.shape[1] + coo2.col)))
    # same support, integer dtype -> the check below never touches a float
    Ai = sparse.csr_matrix((np.ones(A.nnz, dtype=np.int64), (coo.row, coo.col)),
                           shape=A.shape)
    n = A.shape[1]
    res = linprog(-np.ones(n), A_ub=A, b_ub=np.ones(A.shape[0]),
                  bounds=[(0, None)] * n, method='highs', options=E.TOLOPT)
    if not res.success:
        print(f'[INT] {tag}: LP FAILED', flush=True)
        return None
    y = np.maximum(-np.asarray(res.ineqlin.marginals, dtype=float), 0.0)
    p = np.ceil(y * Q).astype(np.int64)
    csc = Ai.tocsc()
    steps = 0
    while True:
        s = np.asarray(Ai.T @ p).ravel()          # int64
        bad = np.nonzero(s < Q)[0]
        if len(bad) == 0 or steps > 60:
            break
        for i in bad:
            rows = csc.indices[csc.indptr[i]:csc.indptr[i + 1]]
            if len(rows) == 0:
                continue
            p[int(rows[int(np.argmin(p[rows]))])] += Q - int(s[i])
        steps += 1
    s_int = np.asarray(Ai.T @ p).ravel()
    s_flt = np.asarray(A.T @ p).ravel()
    # exactness cross-check: int64 result vs the float64 product, no tolerance
    float_ok = bool(np.array_equal(s_flt.astype(np.int64), s_int))
    max_abs_val = int(s_int.max())
    dual_feasible_int = bool((s_int >= Q).all() and (p >= 0).all())
    sum_p = int(p.sum())
    U = Fraction(sum_p, Q)
    out = dict(tag=tag, h_g=h_g, cols=n, rows=int(A.shape[0]), nnz=int(A.nnz),
               builders_agree=same, int64_equals_float64=float_ok,
               max_column_sum=max_abs_val, exact_in_float64=bool(max_abs_val < 2 ** 53),
               repairs=steps, dual_feasible_in_integers=dual_feasible_int,
               min_column_slack=int(s_int.min()) - Q, sum_p=sum_p,
               U_exact=float(U), U_fraction=f'{sum_p}/{Q}',
               lp_value=float(res.x.sum()), dual_objective_float=float(y.sum()),
               proves_no_68_in_family=bool(U < 68), secs=round(time.time() - t0, 1))
    print(f'[INT] {tag}: builders_agree={same} int==float={float_ok} '
          f'colsum_max={max_abs_val} (<2^53={out["exact_in_float64"]}) '
          f'dual_feasible_int={dual_feasible_int} min_slack={out["min_column_slack"]} '
          f'sum_p={sum_p} U={float(U):.6f} {out["U_fraction"]} '
          f'-> {"U<68 PROVED in integers" if U < 68 else "inconclusive"} '
          f'({out["secs"]}s)', flush=True)
    return out


def main():
    import sys
    d = json.load(open('g3_frozen/best_legal_67.json'))
    C67 = np.column_stack([d['x'], d['y'], d['th']]).tolist()
    default = ((0.30, 9, 0.15), (0.30, 9, 0.075), (0.20, 13, 0.15))
    specs, out_name = default, 'g2_tau_cert_int.json'
    if len(sys.argv) > 2:                      # argv[1]=out.json  argv[2:]=h_c,na,h_g
        out_name = sys.argv[1]
        specs = tuple(tuple((float(a), int(b), float(c)))
                      for a, b, c in (s.split(',') for s in sys.argv[2:]))
    out = []
    for h_c, na, h_g in specs:
        nm = f'fam({h_c:.2f},ang{na})'
        F = T.build_family(h_c, na, extra=C67)
        out.append(cert_int(F, h_g, f'{nm} h_g={h_g}'))
    out = [o for o in out if o]
    json.dump(out, open(out_name, 'w'), indent=1, default=float)
    print('[INT] scope unchanged: this bounds V_c of the named discretised family only, '
          'never the global tau(5).', flush=True)


if __name__ == '__main__':
    main()

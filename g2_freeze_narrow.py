"""Step 1 of the user's plan: freeze the two narrow-family integer dual certificates.

Writes, per (family, h_g):
  poses        the candidate family F actually solved over
  grid         the row/audit grid parameters
  matrix       the incidence matrix as canonical (row, col) pairs -- the input matrix
  dual         the integer dual vector p (int64) and the raw LP dual/primal
  per-column   every column's integer sum and slack, plus the LP objective value
and then RE-LOADS the frozen files and re-runs the per-column check from (row,col)+p
alone, with np.add.at on int64 -- no scipy, no float, no reuse of in-memory state.
"""
import hashlib, json, os, time
import numpy as np
from fractions import Fraction
from scipy.optimize import linprog
from scipy import sparse
import g2_tau as T
import g2_tau_exact as E
import g2_tau_cert as C

Q = C.Q
OUT = 'g2_frozen_narrow'


def build_and_certify(F, h_g, tag):
    t0 = time.time()
    gr = E.Grid(h_g)
    A = C.build_brute(F, gr)
    A2 = C.build_fast(F, gr)
    coo, coo2 = A.tocoo(), A2.tocoo()
    agree = bool(A.nnz == A2.nnz
                 and np.array_equal(np.sort(coo.row * A.shape[1] + coo.col),
                                    np.sort(coo2.row * A2.shape[1] + coo2.col)))
    rows = coo.row.astype(np.int64)
    cols = coo.col.astype(np.int64)
    m, n = A.shape
    Ai = sparse.csr_matrix((np.ones(A.nnz, dtype=np.int64), (rows, cols)), shape=(m, n))
    res = linprog(-np.ones(n), A_ub=A, b_ub=np.ones(m),
                  bounds=[(0, None)] * n, method='highs', options=E.TOLOPT)
    assert res.success, res.message
    y = np.maximum(-np.asarray(res.ineqlin.marginals, dtype=float), 0.0)
    p = np.ceil(y * Q).astype(np.int64)
    csc = Ai.tocsc()
    repairs = 0
    while True:
        s = np.asarray(Ai.T @ p).ravel()
        bad = np.nonzero(s < Q)[0]
        if len(bad) == 0 or repairs > 60:
            break
        for j in bad:
            rr = csc.indices[csc.indptr[j]:csc.indptr[j + 1]]
            if len(rr) == 0:
                continue
            p[int(rr[int(np.argmin(p[rr]))])] += Q - int(s[j])
        repairs += 1
    s_int = np.asarray(Ai.T @ p).ravel()
    s_flt = np.asarray(A.T @ p.ravel().astype(float)).ravel()
    sum_p = int(p.sum())
    U = Fraction(sum_p, Q)
    chk = dict(
        tag=tag, h_g=h_g, cols=int(n), rows_in_grid=int(m), nnz=int(A.nnz),
        builders_agree=agree,
        incidence_rule=f'kept only q < HALF - 1e-9 ({C.CLEAR}); uncertain incidences DROPPED',
        dual_nonneg_in_int=bool((p >= 0).all()),
        per_column_int_feasible=bool((s_int >= Q).all()),
        int64_equals_float64=bool(np.array_equal(s_flt.astype(np.int64), s_int)),
        max_column_sum=int(s_int.max()), exact_in_float64=bool(int(s_int.max()) < 2 ** 53),
        repair_steps=int(repairs),
        min_column_sum=int(s_int.min()), min_column_slack=int(s_int.min()) - Q,
        argmin_column=int(np.argmin(s_int)),
        sum_p=sum_p, U_fraction=f'{sum_p}/{Q}', U=float(U),
        U_below_68=bool(U < 68), margin_below_68=68.0 - float(U),
        lp_primal_value=float(res.x.sum()), dual_objective_float=float(y.sum()),
    )
    chk['gate_pass'] = bool(chk['builders_agree'] and chk['dual_nonneg_in_int']
                            and chk['per_column_int_feasible'] and chk['U_below_68'])
    os.makedirs(OUT, exist_ok=True)
    base = f'{OUT}/{tag}'
    np.savez_compressed(base + '_certificate.npz',
                        rows=rows, cols=cols, p=p, s_int=s_int, w=res.x, y=y,
                        grid_x=gr.gx, grid_y=gr.gy, poses=np.asarray(F, dtype=float))
    with open(base + '_per_column.txt', 'w') as f:
        f.write(f'# {tag}: column j -> integer sum s_j = sum_i A_ij p_i, slack s_j - Q'
                f'  (Q={Q})\n# columns={n}  min={chk["min_column_sum"]}'
                f'  max={chk["max_column_sum"]}  gate_pass={chk["gate_pass"]}\n')
        for j in range(n):
            f.write(f'{j}\t{int(s_int[j])}\t{int(s_int[j]) - Q}\n')
    json.dump(chk, open(base + '_checks.json', 'w'), indent=1, default=float)

    # ---- independent reload: re-derive the per-column sums from the frozen file only
    z = np.load(base + '_certificate.npz')
    r2, c2, p2, s_ref = z['rows'], z['cols'], z['p'], z['s_int']
    s3 = np.zeros(int(c2.max()) + 1, dtype=np.int64)
    np.add.at(s3, c2, p2[r2])
    reload_ok = bool(np.array_equal(s3, s_ref) and (s3 >= Q).all() and (p2 >= 0).all())
    sum_p2 = int(p2.sum())
    U2 = Fraction(sum_p2, Q)
    chk['reload_check'] = dict(per_column_sums_reproduced=reload_ok,
                               sum_p_recomputed=sum_p2, U_recomputed=float(U2),
                               matches_sum_p=sum_p2 == sum_p, U2_below_68=bool(U2 < 68))
    json.dump(chk, open(base + '_checks.json', 'w'), indent=1, default=float)
    print(f'[FRZ] {tag}: nnz={A.nnz} builders_agree={agree} min_slack=+{chk["min_column_slack"]}'
          f' sum_p={sum_p} U={float(U):.7f} gate_pass={chk["gate_pass"]}'
          f' reload_ok={reload_ok} ({time.time()-t0:.1f}s)', flush=True)
    return chk


def main():
    d = json.load(open('g3_frozen/best_legal_67.json'))
    C67 = np.column_stack([d['x'], d['y'], d['th']]).tolist()
    F = T.build_family(0.30, 9, extra=C67)
    keys = {(round(float(a), 9), round(float(b), 9), round(float(c), 9)) for a, b, c in F}
    have = sum(1 for a, b, c in C67 if (round(a, 9), round(b, 9), round(c, 9)) in keys)
    print(f'[FRZ] fam(0.30,ang9) |F|={len(F)} certified_67_present={have}/67 '
          f'(by explicit injection: 0.25 is not on the 0.30 centre grid)', flush=True)
    assert have == 67
    recs = [build_and_certify(F, h_g, f'fam030_ang9_hg{h_g}') for h_g in (0.15, 0.075)]
    gate = all(r['gate_pass'] and r['reload_check']['per_column_sums_reproduced'] for r in recs)
    meta = dict(
        date='2026-10-09', route='G2 measure-certificate, narrow candidate family',
        family='build_family(h_c=0.30, n_ang=9) on [-5,5] centre grid, angles [0,pi/2) x9, '
               'plus the frozen certified 67 injected',
        certificates=recs,
        squeeze='67 <= V_c(fam(0.30,ang9)) <= min(U) = '
                f'{min(r["U"] for r in recs):.7f}  (lower end: exact Fraction certificate, '
                'interior convention D<=1 at every point)',
        scope_limit='V_c(F) <= V* for every subfamily F of the continuum family, so a '
                    'family-restricted upper bound CANNOT bound the global continuum value; '
                    'this is a finite-candidate-set conclusion, never tau(5)=67, never r(67)>=5.',
        gate_all_pass=bool(gate),
        recording_rules='Gate-0 stays PARTIALLY PASSED; mmax68_R5_candidate.csv is not a legal '
                        'baseline (sha256 227158f7...1394f, rechecked unchanged); radius 口径 '
                        '5.052404006121 / 5.052605585675 kept separate; packing67.txt untouched.',
        no_further_work='No additional grid refinement on this family (user directive).',
    )
    json.dump(meta, open(f'{OUT}/GATE.json', 'w'), indent=1, default=float)
    manifest = {}
    for fn in sorted(os.listdir(OUT)):
        if fn == 'MANIFEST.json':
            continue
        b = open(os.path.join(OUT, fn), 'rb').read()
        manifest[fn] = dict(bytes=len(b), sha256=hashlib.sha256(b).hexdigest())
    json.dump(manifest, open(f'{OUT}/MANIFEST.json', 'w'), indent=1)
    print(f'[FRZ] both gates pass = {gate}; hashed files = {len(manifest)} (MANIFEST.json excluded)', flush=True)
    for k, v in sorted(manifest.items()):
        if k != 'MANIFEST.json':
            print(f'[FRZ]   {v["bytes"]:>10}  {k}  {v["sha256"][:16]}', flush=True)


if __name__ == '__main__':
    main()

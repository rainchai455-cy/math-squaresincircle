"""Phase cost of the certificate pipeline, WITHOUT calling the LP solver.

Why: the running int64 gate job prints only a total, so the bottleneck cannot be attributed
post-hoc.  Here every non-LP phase is timed on the same matrix the gate job uses; the LP time is
then total minus this, instead of an inference from aggregate CPU seconds.  The control case is
fam(0.30,ang9) @ h_g=0.075, whose whole pipeline (including LP) took 88.1 s, so a correct
decomposition must leave a plausible LP residual there.

Nothing is solved: no linprog call, no file writes, the solver chain is untouched.
"""
import json
import time

import numpy as np
from scipy import sparse

import g2_tau as T
import g2_tau_exact as E
from g2_tau_cert import HALF_C, build_brute, build_fast

Q = 10 ** 7


def timed(fn):
    t = time.perf_counter()
    r = fn()
    return round(time.perf_counter() - t, 2), r


def phases(F, h_g, tag):
    gr = E.Grid(h_g)
    out = {'tag': tag, 'h_g': h_g}
    t, A2 = timed(lambda: build_fast(F, gr))
    out['build_fast_s'] = t
    out['cols'] = int(A2.shape[1])
    out['grid_rows'] = int(A2.shape[0])
    out['nnz'] = int(A2.nnz)
    t, A = timed(lambda: build_brute(F, gr))
    out['build_brute_s'] = t
    coo, coo2 = A.tocoo(), A2.tocoo()
    t, same = timed(lambda: bool(coo.nnz == coo2.nnz and np.array_equal(
        np.sort(coo.row * A.shape[1] + coo.col),
        np.sort(coo2.row * A2.shape[1] + coo2.col))))
    out['agree_check_s'] = t
    out['builders_agree'] = same
    t, Ai = timed(lambda: sparse.csr_matrix(
        (np.ones(coo.nnz, dtype=np.int64), (coo.row, coo.col)), shape=A.shape))
    out['int64_matrix_s'] = t
    # the dual vector is indexed by GRID ROWS (s = A^T p has one entry per column)
    p = np.full(int(Ai.shape[0]), 1000, dtype=np.int64)
    t, s = timed(lambda: np.asarray(Ai.T @ p).ravel())
    out['int_colsum_pass_s'] = t
    out['colsum_len'] = int(len(s))
    out['colsum_dtype'] = str(s.dtype)
    t, csc = timed(lambda: Ai.tocsc())
    out['tocsc_s'] = t
    # 60 repair passes is the loop cap in the gate script: one pass = full colsum + per-column updates
    t, _ = timed(lambda: [np.asarray(Ai.T @ p).ravel() for _ in range(10)])
    out['int_colsum_10_passes_s'] = t
    node = coo.row.astype(np.int64)      # grid node index, matches p's indexing
    pose = coo.col.astype(np.int64)      # candidate-pose (= column) index

    def addat():
        s3 = np.zeros(int(A.shape[1]), dtype=np.int64)
        np.add.at(s3, pose, p[node])
        return s3
    t, _ = timed(addat)
    out['reload_addat_s'] = t
    used = int((np.asarray(A.sum(axis=1)).ravel() > 0).sum())
    out['used_rows'] = used
    out['nonlp_total_s'] = round(sum(v for k, v in out.items() if k.endswith('_s')), 2)
    print(f'[PHASE] {tag} h_g={h_g}: fast={out["build_fast_s"]} brute={out["build_brute_s"]} '
          f'agree={out["agree_check_s"]} int64={out["int64_matrix_s"]} '
          f'colsum={out["int_colsum_pass_s"]} tocsc={out["tocsc_s"]} '
          f'10passes={out["int_colsum_10_passes_s"]} addat={out["reload_addat_s"]} '
          f'-> non-LP total={out["nonlp_total_s"]}s  (LP = gate total - this)', flush=True)
    return out


def main():
    d = json.load(open('g3_frozen/best_legal_67.json'))
    C67 = np.column_stack([d['x'], d['y'], d['th']]).tolist()
    res = []
    for h_c, na, nm in ((0.30, 9, 'fam(0.30,ang9)'), (0.20, 13, 'fam(0.20,ang13)')):
        F = T.build_family(h_c, na, extra=C67)
        res.append(phases(F, 0.075, nm))
    json.dump(res, open('g2_phase_cost.json', 'w'), indent=1, default=float)
    print('[PHASE] no LP was solved here; gate jobs and solver chain untouched', flush=True)


if __name__ == '__main__':
    main()

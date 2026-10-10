import numpy as np, math, time, json, sys
import g3_grow as GR, g3_verify as V

# =====================================================================
# DILATE-then-REPAIR : the direct feasibility probe for a target side.
#
# Scaling the certified 67 by lambda is EXACTLY gap-preserving (every gap and
# every contact scales with it), so dilating by s_t keeps all 67 squares disjoint
# and pushes every corner to s_t * r.  At s_t > 1 the state is illegal only by the
# rim overshoot out ~ 5*(s_t - 1).  Whether a legal 67 at side s_t exists is then
# the question "can repair absorb that overshoot without closing a gap", which is
# the finite-step version of the squeeze that measured t* = 0 at first order.
#
# This is the sharpest available test of the user's A/B question, because the
# operation is IDENTICAL in both arms and only the repair's freedom (theta frozen
# or not) differs.  k = how many outermost squares are removed before dilating,
# then put back after the repair, so a rim square that cannot absorb its own
# overshoot can be re-seated elsewhere.
# =====================================================================

R = GR.R_FIX
rows = [l.split() for l in open('packing67.txt') if l.strip() and not l.startswith('#')]
C0 = np.array([[(int(r[1])+int(r[3]))/4.0, (int(r[2])+int(r[4]))/4.0] for r in rows])
TH0 = np.zeros(len(C0))
OFFS = [(dr, dt) for dr in (0.0, .04, -.04, .08, -.08, .12, -.16)
        for dt in (0.0, .05, -.05, .10)]


def attempt(s_t, rot, k, iters=400, refill_after=True):
    C, TH = C0*s_t, TH0.copy()
    if k:
        rr, _, _ = GR.corners_s(C, TH, s_t)
        rm = np.argsort(-rr.max(axis=1))[:k]
        keep = np.setdiff1d(np.arange(len(C)), rm)
        holes = [C0[i]*s_t for i in rm]
        C, TH = C[keep], TH[keep]
    C, TH, m, it, ok = GR.repair_s(C, TH, s_t, rot=rot, iters=iters)
    if ok and k and refill_after:
        C, TH, ok2 = GR.refill(C, TH, s_t, holes, (0.0,) if not rot else (0.0, math.pi/4),
                               offsets=OFFS, allow_pen=0.30)
        if ok2:
            C, TH, m, it2, ok = GR.repair_s(C, TH, s_t, rot=rot, iters=iters)
            it += it2
        else:
            ok = False
    return C, TH, s_t, m, it, ok


if __name__ == '__main__':
    if '--controls' in sys.argv:
        # ---- POSITIVE CONTROLS.  A negative result for the 67-dilate is only
        # ---- worth reporting if the SAME pipeline can produce and verify an
        # ---- s > 1 arrangement for a smaller N.  Without this the failure would
        # ---- be ambiguous between "the geometry refuses it" and "side > 1 is not
        # ---- supported by my code".
        lat = json.load(open('g3_lattice_R499.json'))
        x, y = np.array(lat['x']), np.array(lat['y'])
        th = np.zeros(len(x))
        s_max_lat = R/lat['r_implicit']
        for s_try in (GR.S_TARGET, s_max_lat*0.999999):
            v1 = V.verify(x*s_try, y*s_try, th, R, side=s_try)
            v2 = V.verify(x, y, th, R/s_try)
            print(f'[PC1] N=65 quarter-lattice at side s={s_try:.12f}: '
                  f's-frame LEGAL={v1["LEGAL"]} (R_impl={v1["enclosing_radius"]:.12f}), '
                  f'unit-frame LEGAL={v2["LEGAL"]} (R_impl={v2["enclosing_radius"]:.12f})  '
                  f'=> s_target {GR.S_TARGET:.9f} IS reachable at N=65, '
                  f'own ceiling {s_max_lat:.9f}', flush=True)
        for k in (0, 8):
            C, TH = C0*GR.S_TARGET, TH0.copy()
            if k:
                rr, _, _ = GR.corners_s(C, TH, GR.S_TARGET)
                rm = np.argsort(-rr.max(axis=1))[:k]
                C, TH = C[np.setdiff1d(np.arange(len(C)), rm)], TH[np.setdiff1d(
                    np.arange(len(C)), rm)]
            C, TH, m, it, ok = GR.repair_s(C, TH, GR.S_TARGET, rot=False, iters=200)
            print(f'[PC2] dilate 67 by s_target, remove k={k}, NO refill: N={len(C)} '
                  f'FEASIBLE={ok} ({it} iters) ov={m[0]:.2e} out={m[1]:.2e} '
                  f'R_impl={m[3]:.9f}', flush=True)
        sys.exit(0)

    quick = '--quick' in sys.argv
    log = []
    for s_t in ((1.002, 1.01) if quick else (1.0005, 1.001, 1.002, 1.005, 1.01, 1.02)):
        for k in (0, 8):
            for rot in (False, True):
                t0 = time.time()
                C, TH, st, m, it, ok = attempt(s_t, rot, k, iters=200 if not quick else 60)
                fp = (int((np.abs(TH) > 1e-6).sum()), float(np.abs(TH).sum()))
                line = (f's_t={s_t:.5f} k={k:2d} arm={"TR" if rot else "T "} '
                        f'N={len(C):2d} FEASIBLE={ok} ({it} iters,{time.time()-t0:.1f}s) '
                        f'ov={m[0]:.3e} out={m[1]:.3e} minGap={m[2]:+.1e} '
                        f'R_impl={m[3]:.6f} rotated_squares={fp[0]} sum|th|={fp[1]:.4f}')
                print(line, flush=True); log.append(line)
                if ok:
                    v1 = V.verify(C[:, 0], C[:, 1], TH, R, side=s_t)
                    v2 = V.verify(C[:, 0]/s_t, C[:, 1]/s_t, TH, R/s_t)
                    line2 = (f'    VERIFIED s_t={s_t:.9f}: s-frame LEGAL={v1["LEGAL"]} '
                             f'unit-frame LEGAL={v2["LEGAL"]} R_equiv={v2["enclosing_radius"]:.12f}'
                             f' -> r(67) <= {R/s_t:.9f}')
                    print(line2, flush=True); log.append(line2)
                    if v1['LEGAL'] and v2['LEGAL'] and len(C) == 67:
                        json.dump(dict(x=C[:, 0].tolist(), y=C[:, 1].tolist(),
                                       th=TH.tolist(), side=float(s_t), R_fix=R,
                                       op=f'dilate-repair k={k} rot={rot}'),
                                  open(f'g3_dilate_WIN_{s_t:.5f}.json', 'w'), indent=1)
                        print('    *** WIN: 67 squares verified at side', s_t,
                              flush=True)
    open('g3_dilate_log.txt', 'w', encoding='utf-8').write('\n'.join(log)+'\n')

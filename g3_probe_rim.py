import numpy as np, math, time, sys
import g3_grow as GR

rows = [l.split() for l in open('packing67.txt') if l.strip() and not l.startswith('#')]
C = np.array([[(int(r[1])+int(r[3]))/4.0, (int(r[2])+int(r[4]))/4.0] for r in rows])
TH = np.zeros(len(C))
rr, _, _ = GR.corners_s(C, TH, 1.0)
mx = rr.max(axis=1)
pinned = np.where(mx > 5.0 - 1e-9)[0]
order = np.argsort(-mx)
print(f'pinned-at-5.0 squares = {len(pinned)} {sorted(pinned.tolist())}', flush=True)
OFFS = [(dr, dt) for dr in (0.0, .04, -.04, .08, -.08, .12, -.12, .16, -.16)
        for dt in (0.0, .05, -.05, .10, -.10)]

for k in (8, 12):
    rm = sorted(order[:k].tolist())
    keep = np.setdiff1d(np.arange(len(C)), rm)
    for rot in (False, True):
        t0 = time.time()
        Cg, Tg, sg, m, it, st = GR.grow(C[keep].copy(), TH[keep].copy(), 1.0,
                                        rot=rot, iters=200, rho_s=0.002,
                                        deadline=t0+60.0)
        print(f'  k={k:2d} N={len(keep):2d} arm={"TR" if rot else "T "}: '
              f's={sg:.9f} ({st},{it} LPs,{time.time()-t0:.1f}s) '
              f'ov={m[0]:.2e} out={m[1]:.2e} R_impl={m[3]:.6f} '
              f'| max|th|={float(np.abs(Tg).max()):.4f} '
              f'| dilate-limit={5.0/m[3]:.6f}', flush=True)
        # how deep would the refill have to bite?  diag = best min-gap per hole
        for fac in (1.0, 0.98, 0.95):
            dg = []
            Cf, Tf, ok = GR.refill(Cg, Tg, sg*fac, [C[i] for i in rm], (0.0,),
                                   offsets=OFFS, allow_pen=0.5, diag=dg)
            m2 = GR.metrics_s(Cf, Tf, sg*fac)
            print(f'      refill s={sg*fac:.9f} ok={ok} worst_offered_gap='
                  f'{min(dg):+.6f} -> post ov={m2[0]:.3e} out={m2[1]:.3e} '
                  f'N={len(Cf)}', flush=True)

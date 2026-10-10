import numpy as np, math, json, time
import g3_grow as GR

# =====================================================================
# Is "remove-k -> grow the remainder -> put k back" geometrically ALIVE at all,
# before spending any LP budget on the paired A/B?  Pure geometry, no search:
# for each (k, s) we hand the refill the best site it could ask for and report
# the gap it offers.  worst_gap >= 0 means the k-th square can be seated with no
# overlap at that side; worst_gap < 0 says how deep it would have to bite.
#
# Two bulk variants are compared because they answer different questions:
#   'dil'   = the keep-squares simply DILATED by s (gaps scale, so the only
#             possible illegality is rim overshoot -> repair makes room and stops
#             there).  This is the "give the annulus back" bulk.
#   'grow'  = the keep-squares grown to their OWN ceiling at side sg, then s is
#             brought back down.  This is what g3_ab.run_arm currently does.
# =====================================================================

R = GR.R_FIX
OFFS = [(dr, dt) for dr in (0.0, .04, -.04, .08, -.08, .12, -.12, .16, -.16,
                            .24, -.24, .32, -.32)
        for dt in (0.0, .05, -.05, .10, -.10, .20, -.20, .30, -.30)]


def load67():
    rows = [l.split() for l in open('packing67.txt') if l.strip()
            and not l.startswith('#')]
    C = np.array([[(int(r[1])+int(r[3]))/4.0, (int(r[2])+int(r[4]))/4.0]
                  for r in rows])
    return C, np.zeros(len(C))


def shells(C, TH):
    rr, _, _ = GR.corners_s(C, TH, 1.0)
    mx = rr.max(axis=1)
    order = np.argsort(-mx)
    pinned = np.where(mx > R - 1e-9)[0]
    return order, pinned, mx


def hole_gaps(Cb, Tb, s, holes, angles=(0.0,)):
    """per-hole best offered gap over the site grid, greedy (each taken site
    blocks the next), and the count of holes seatable with gap >= 0."""
    Cl, TL = list(Cb), list(Tb)
    gaps, seat = [], 0
    for pos in holes:
        pos = np.asarray(pos, float)
        ur = pos/max(1e-12, float(np.hypot(*pos)))
        ut = np.array([-ur[1], ur[0]])
        best = -9e9
        bp, ba = pos.copy(), angles[0]
        for (dr, dt) in OFFS:
            p = pos + dr*ur + dt*ut
            for a in angles:
                if not GR.corner_ok(p[0], p[1], a, s, R, 1e-9):
                    continue
                g = min(GR.sep_pair_s(p[0], p[1], a, q[0], q[1], t, s)
                        for (q, t) in zip(Cl, TL))
                if g > best:
                    best = g
                    bp, ba = p, a
        gaps.append(best)
        if best >= 0.0:
            seat += 1
        Cl.append(bp); TL.append(ba)
    return gaps, seat


if __name__ == '__main__':
    C0, TH0 = load67()
    order, pinned, mx = shells(C0, TH0)
    print(f'[SP] N={len(C0)}  pinned at R={R}: {len(pinned)} -> {sorted(pinned.tolist())}')
    print(f'[SP] shell radii (desc): '
          f'{[round(float(mx[i]), 6) for i in order[:24].tolist()]}')
    log = []
    for k in (8, 12, 16, 20, 26):
        rm = sorted(set([int(p) for p in pinned] +
                        [int(order[i]) for i in range(8, k)]))
        keep = np.setdiff1d(np.arange(len(C0)), rm)
        holes0 = [C0[i] for i in rm]
        print(f'\n[SP] === k={len(rm)} removed={rm} ===')
        for s_t in (GR.S_TARGET, 1.005, 1.012, 1.020):
            for variant in ('dil', 'grow'):
                t0 = time.time()
                if variant == 'dil':
                    Cb, Tb = C0[keep]*s_t, TH0[keep].copy()
                    m = GR.metrics_s(Cb, Tb, s_t)
                    Cb, Tb, m, it, ok = GR.repair_s(Cb, Tb, s_t, rot=False,
                                                    rho0=0.02, iters=200)
                else:
                    sg = 1.0
                    Cb, Tb, sg, m, it, st = GR.grow(C0[keep].copy(),
                                                    TH0[keep].copy(), 1.0,
                                                    rot=False, iters=200, rho_s=0.002)
                    sc = s_t/sg
                    Cb, Tb = Cb*sc, Tb
                    Cb, Tb, m, it2, ok = GR.repair_s(Cb, Tb, s_t, rot=False,
                                                     iters=200)
                    it += it2
                gaps, seat = hole_gaps(Cb, Tb, s_t, holes0)
                line = (f'[SP] k={len(rm):2d} s={s_t:.9f} {variant:4s} bulk: '
                        f'ov={m[0]:.1e} out={m[1]:.1e} minGap={m[2]:+.2e} '
                        f'R_impl={m[3]:.6f} LPs={it} ({time.time()-t0:.1f}s)  '
                        f'hole gaps: min={min(gaps):+.4f} med={np.median(gaps):+.4f} '
                        f'seated(gap>=0)={seat}/{len(rm)}')
                print(line); log.append(line)
    open('g3_seed_probe_log.txt', 'w', encoding='utf-8').write('\n'.join(log)+'\n')

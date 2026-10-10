import numpy as np, math, time, json, sys, os

# =====================================================================
# G3 S3 : GLOBAL feasibility search by penalty annealing / basin hopping.
#
# Why this phase exists (measured, not assumed):
#   - the certified 67 at R=5.0 is a FIRST-ORDER JAM.  Pure shrink from it
#     stalls at out=8.40e-02 = 8 rim corners x 1.05e-02, i.e. R_impl pinned at
#     5.000002, and removing 1 square does not free the coupling.
#   - the 1/4-lattice 65 base at R=4.99 is already saturated (its best
#     insertion site scores -0.259, i.e. it needs 0.254 of overlap to open).
#   Both LP routes therefore explore ONE basin.  This phase explores many:
#     min  ovsum(C,TH) + outsum(C,TH)  at fixed R,  by Metropolis moves on
#     individual squares (translation + rotation), restarted from several seed
#     families.  Nothing here certifies anything - a state with both sums
#     exactly 0 is handed to g3_verify (independent SAT + polygon clipping).
#
# Geometry is written fresh here (not imported from the searcher's LP model):
#   disjointness  : SAT over the 4 edge normals of the two squares.  For two
#                   convex quadrilaterals an axis among those four separates
#                   them iff any axis does, so max over the four >= 0 is an
#                   EXACT proof of disjointness.
#   containment   : max_k sqrt(|c_k|^2 + 1/2 + |c_k.u_k| + |c_k.v_k|) is the
#                   largest of the 4 rotated corners, algebraically.
# Only squares closer than a square's diagonal (sqrt(2)) can overlap, so pairs
# beyond that are skipped - that is a proof-grade filter, not a heuristic.
# =====================================================================

SIDE = 1.0
N_SQ = 67
DIAG2 = 2.000001          # (sqrt(2))^2 + margin: centre-distance overlap limit


class Feas:
    """exact (conservative-free) feasibility merit at a fixed radius."""

    def __init__(self, n, R):
        self.n, self.R = n, R
        self.I, self.J = np.triu_indices(n, 1)

    def gaps(self, C, TH):
        U = np.stack([np.cos(TH), np.sin(TH)], 1)
        V = np.stack([-np.sin(TH), np.cos(TH)], 1)
        d = C[self.J] - C[self.I]
        near = (d * d).sum(1) < DIAG2
        if not near.any():
            return np.empty(0), np.zeros(0, dtype=bool)
        di, ui, vi, uj, vj = d[near], U[self.I][near], V[self.I][near], \
            U[self.J][near], V[self.J][near]

        def term(dx, a, b1, b2):
            # |d.a| - 0.5 - 0.5*(|a.b1| + |a.b2|)   (a is an edge normal of one
            # square, 0.5 its own support, b1/b2 the other square's normals)
            return np.abs(dx[:, 0]*a[:, 0] + dx[:, 1]*a[:, 1]) - 0.5 \
                - 0.5*(np.abs(a[:, 0]*b1[:, 0] + a[:, 1]*b1[:, 1]) +
                       np.abs(a[:, 0]*b2[:, 0] + a[:, 1]*b2[:, 1]))

        g = np.maximum.reduce([term(di, ui, uj, vj), term(di, vi, uj, vj),
                               term(di, uj, ui, vi), term(di, vj, ui, vi)])
        return g, near

    def radii(self, C, TH):
        cu = C[:, 0]*np.cos(TH) + C[:, 1]*np.sin(TH)
        cv = -C[:, 0]*np.sin(TH) + C[:, 1]*np.cos(TH)
        r2 = (C*C).sum(1) + 0.5 + np.abs(cu) + np.abs(cv)
        return np.sqrt(r2)

    def __call__(self, C, TH):
        g, _ = self.gaps(C, TH)
        ov = float(np.maximum(0.0, -g).sum())
        r = self.radii(C, TH)
        out = float(np.maximum(0.0, r - self.R).sum())
        return ov + out, ov, out, r.max(), (g.min() if len(g) else 0.0)


def anneal(ev, C, TH, iters, T0, T1, rng, sigma=0.09):
    n = len(C)
    C, TH = C.copy(), TH.copy()
    f, ov, out, rmax, gmin = ev(C, TH)
    best = (f, C.copy(), TH.copy(), ov, out, rmax)
    sp, sr, acc, tries = sigma, 0.5*sigma, 0, 0
    alpha = (T1/T0)**(1.0/max(1, iters))
    T = T0
    for it in range(iters):
        k = rng.integers(n)
        C2 = C.copy(); TH2 = TH.copy()
        C2[k] += rng.normal(0.0, sp, 2)
        TH2[k] += rng.normal(0.0, sr)
        f2, ov2, out2, rmax2, gmin2 = ev(C2, TH2)
        tries += 1
        if f2 <= f or rng.random() < math.exp(min(0.0, -(f2-f))/max(T, 1e-15)):
            C, TH, f = C2, TH2, f2
            acc += 1
            if f < best[0] - 1e-18:
                best = (f, C.copy(), TH.copy(), ov2, out2, rmax2)
            if f == 0.0:
                break
        if it % 120 == 119:
            r = acc/max(tries, 1)
            if r > 0.45:
                sp *= 1.2; sr *= 1.2
            elif r < 0.15:
                sp *= 0.8; sr *= 0.8
            acc = tries = 0
        T *= alpha
    return best, (f, ov, out, rmax, gmin), it


def seed(fam, rng, C67, TH67, lat, R, i):
    s = R/5.0
    if fam == 'cert':
        return C67*s, np.zeros(len(C67))
    if fam == 'jit':
        sig = (0.02, 0.05, 0.10, 0.20, 0.35)[i % 5]
        return C67*s + rng.normal(0, sig, C67.shape), rng.normal(0, 0.15*sig+0.005, len(C67))
    if fam == 'lattice':
        C = np.array([lat['x'], lat['y']]).T*s
        cc = rng.uniform(-1, 1, (2, 2))*(R-0.8)
        return np.vstack([C, cc]), np.zeros(len(C)+2)
    if fam == 'brick':
        pts = []
        row = 0
        y = -(R-0.6)
        step = 1.0 + 0.02*rng.random()
        while y < R and len(pts) < N_SQ:
            x = -(R-0.6) + (0.5*step if row % 2 else 0.0)
            while x < R and len(pts) < N_SQ:
                pts.append([x + rng.normal(0, 0.03), y + rng.normal(0, 0.03)]); x += step
            y += step; row += 1
        pts = np.array(pts[:N_SQ])
        return pts, np.zeros(len(pts))
    if fam == 'random':
        pts = []
        while len(pts) < N_SQ:
            p = rng.uniform(-R, R, 2)
            if (abs(p[0])+0.5)**2 + (abs(p[1])+0.5)**2 <= R*R:
                pts.append(p)
        return np.array(pts), np.zeros(N_SQ)
    raise ValueError(fam)


def run(bud, log, R, C67, TH67, iters=4000, T0=0.05, T1=3e-5, fams=None,
        verbose=True, polish=None):
    """restarts until the budget expires; returns (found, C, TH)."""
    lat = json.load(open('g3_lattice_R499.json'))
    fams = fams or ['jit', 'jit', 'cert', 'brick', 'random', 'lattice']
    ev = Feas(N_SQ, R)
    rng = np.random.default_rng(int(time.time()) % 2**32)
    hist = []
    global_best = (9e9, None, None)
    t0 = time.time()
    restart = 0
    n_eval = 0
    while bud.left() > max(8.0, 0.05*bud.sec) and restart < 400:
        fam = fams[restart % len(fams)]
        C, TH = seed(fam, rng, C67, TH67, lat, R, restart)
        if len(C) != N_SQ:
            restart += 1; continue
        m0 = ev(C, TH)
        (f, bc, bTH, bov, bout, brmax), mfin, it = anneal(ev, C, TH, iters, T0, T1, rng)
        n_eval += it
        dt = time.time()-t0
        if f < global_best[0]:
            global_best = (f, bc.copy(), bTH.copy())
        rec = dict(restart=restart, fam=fam, f_start=m0[0], f_best=f, ov=bov, out=bout,
                   R_impl=float(brmax), iters=it, cum_sec=round(dt, 1))
        hist.append(rec)
        if verbose:
            print(f'[S3] #{restart} {fam:8s} start={m0[0]:.4f} best={f:.4e} '
                  f'ov={bov:.3e} out={bout:.3e} R_impl={brmax:.6f} it={it} '
                  f'({dt:.0f}s, budget left {bud.left():.0f}s)')
        if f == 0.0:
            log.append(f'[S3] #{restart} {fam}: EXACT feasibility at R={R} after {it} moves')
            return True, bc, bTH
        if bov < 1e-4 and polish is not None and bud.left() > 20:
            Cp, THp, wp = polish(bc, bTH, bud, tag=f'S3 #{restart} polish')
            if wp:
                log.append(f'[S3] #{restart} {fam}: LP polish reached exact legality')
                return True, Cp, THp
        restart += 1
    dt = time.time()-t0
    log.append(f'[S3] restarts={restart}  wall={dt:.1f}s  moves={n_eval} '
               f'({n_eval/max(dt,1e-9):.0f}/s)  best merit={global_best[0]:.6e}')
    if verbose:
        print(f'[S3] end: restarts={restart} wall={dt:.1f}s moves={n_eval} '
              f'({n_eval/max(dt,1e-9):.0f}/s) best merit={global_best[0]:.4e}')
    json.dump(dict(R=R, restarts=restart, wall=dt, moves=n_eval,
                   best_merit=global_best[0], history=hist),
              open('g3_anneal_summary.json', 'w'), indent=1)
    return False, global_best[1], global_best[2]


if __name__ == '__main__':
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    R = float(sys.argv[2]) if len(sys.argv) > 2 else 4.9895
    rows = [l.split() for l in open('packing67.txt') if l.strip() and not l.startswith('#')]
    C67 = np.array([[(int(r[1])+int(r[3]))/4.0, (int(r[2])+int(r[4]))/4.0] for r in rows])
    t = time.time()
    ev = Feas(N_SQ, R)
    f, ov, out, rmax, gmin = ev(C67, np.zeros(N_SQ))
    print(f'[calib] one merit eval = {(time.time()-t)*1000:.3f} ms   '
          f'(certified 67 at R={R}: f={f:.4f} ov={ov:.4f} out={out:.4f} '
          f'R_impl={rmax:.9f} minGap={gmin:+.3e})')
    # sanity: the known-exact config must reproduce the known implicit radius
    t = time.time()
    for _ in range(200):
        ev(C67, np.zeros(N_SQ))
    rate = 200/(time.time()-t)
    print(f'[calib] eval rate = {rate:.0f}/s')
    if len(sys.argv) > 3 and sys.argv[3] == 'go':
        class B:
            def __init__(s, sec):
                s.t0 = time.time(); s.sec = sec; s.lps = 0
            def left(s):
                return s.sec-(time.time()-s.t0)
            def expired(s):
                return s.left() <= 0
        log = []
        bud = B(budget)
        found, C, TH = run(bud, log, R, C67, np.zeros(N_SQ))
        print('\n'.join(log))
        if found:
            import g3_verify as V
            c = V.verify(C[:, 0], C[:, 1], TH, 4.99)
            V.report(c, 'standalone anneal candidate @R=4.99', log)
            json.dump(dict(N=len(C), x=C[:, 0].tolist(), y=C[:, 1].tolist(),
                           th=TH.tolist(), side=1.0), open('g3_cand_S3.json', 'w'), indent=1)
        open('g3_anneal_log.txt', 'w', encoding='utf-8').write('\n'.join(log)+'\n')

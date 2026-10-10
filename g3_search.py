import numpy as np, json, sys, time, math, hashlib, os
import g3_core as K
import g3_verify as V

# =====================================================================
# G3 driver.  Target: N_SQ squares of side EXACTLY 1 inside R_TARGET,
# free dofs = continuous translation + continuous rotation.
# Nothing here certifies legality - every candidate goes to the
# independent verifier (g3_verify: SAT AND polygon clipping, all pairs,
# all corners, tolerances pre-registered in gate0.py).
# =====================================================================
R_TARGET = 4.99
R_SEARCH = 4.9895          # safety band, verified strictly at R_TARGET
SEED_R5 = 'packing67.txt'
LATTICE = 'g3_lattice_R499.json'
BETA = 2e-3                # compaction tie-break weight
TOL = 1e-13                # internal "legal yet?" threshold (NOT the verdict)


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest() if os.path.exists(p) else None


def load_cert():
    rows = [l.split() for l in open(SEED_R5) if l.strip() and not l.startswith('#')]
    C = np.array([[(int(r[1])+int(r[3]))/4.0, (int(r[2])+int(r[4]))/4.0] for r in rows])
    return C, np.zeros(len(C)), [r[0] for r in rows]


def dump(path, C, TH, extra=None):
    d = dict(N=len(C), x=C[:, 0].tolist(), y=C[:, 1].tolist(), th=TH.tolist(),
             side=1.0, R_search=R_SEARCH, R_target=R_TARGET)
    if extra:
        d.update(extra)
    json.dump(d, open(path, 'w'), indent=1)
    return path


class Budget:
    def __init__(self, sec):
        self.t0 = time.time(); self.sec = sec; self.lps = 0

    def left(self):
        return self.sec - (time.time()-self.t0)

    def expired(self):
        return self.left() <= 0


def relax(C, TH, bud, rot=True, rho0=0.015, iters=300, tag='', R=None, beta=0.0,
          stop=True, wg=300.0):
    t0 = time.time()
    R = R_SEARCH if R is None else R
    dl = time.time() + max(3.0, min(60.0, bud.left()))
    C, TH, cur, it, won = K.try_relax(C, TH, R, rot=rot, rho0=rho0, wg=wg,
                                      iters=iters, deadline=dl, beta=beta,
                                      stop_when_feasible=stop)
    dt = time.time()-t0
    bud.lps += it
    print(f'      {tag}: {"LEGAL" if won else "stalled"} ov={cur[0]:.2e} out={cur[1]:.2e} '
          f'minGap={cur[2]:+.2e} R_impl={cur[3]:.9f} sum|c|2={np.sum(C*C):.3f} '
          f'it={it} {dt:.1f}s')
    return C, TH, cur, won


def compact_polish(C, TH, bud, tag, log):
    """pull everything inward (banks slack at the rim), then clear any residual
    violation with beta=0 so the state is strictly legal again."""
    C, TH, m, _ = relax(C, TH, bud, rot=True, beta=BETA, stop=False,
                        iters=200, tag=f'{tag} compact')
    C, TH, m, won = relax(C, TH, bud, rot=True, beta=0.0, iters=200,
                          tag=f'{tag} polish')
    log.append(f'[{tag}] compact+polish: N={len(C)} legal={won} ov={m[0]:.2e} '
               f'out={m[1]:.2e} R_impl={m[3]:.9f} sum|c|2={np.sum(C*C):.3f}')
    return C, TH, m, won


def try_place(C, TH, bud, log, tag, angles=(0.0, 0.06, -0.06, 0.15, -0.15, 0.3, -0.3),
              k=8, shakes=3):
    """insert one square at the best available site; accept only if the state
    becomes strictly legal at R_SEARCH."""
    sites = K.insert_sites(C, TH, R_SEARCH, angles=angles, k=k)
    if not sites:
        log.append(f'[{tag}] no insertion site'); return None, None, None
    log.append(f'[{tag}] site scores: ' + ', '.join(f'{g:+.4f}' for g, _, _ in sites))
    print(f'[{tag}] best site gap={sites[0][0]:+.5f} at '
          f'({sites[0][1][0]:+.3f},{sites[0][1][1]:+.3f})  budget {bud.left():.0f}s')
    rng = np.random.default_rng(abs(hash((len(C), round(sites[0][0], 6)))) % 2**32)
    for (g, cc, a) in sites:
        for sh in range(shakes+1):
            Cn = np.vstack([C, cc]); THn = np.concatenate([TH, [a]])
            if sh:
                Cn[:len(C)] += rng.normal(0, 0.006, (len(C), 2))
                THn[:len(TH)] += rng.normal(0, 0.01, len(TH))
            Cf, THf, m, won = relax(Cn, THn, bud, rot=True, iters=250,
                                    tag=f'{tag} site={g:+.3f} shake{sh}')
            if won:
                log.append(f'[{tag}] ACCEPTED insertion at site gap={g:+.5f} shake={sh} '
                           f'-> N={len(Cf)} legal at R_SEARCH={R_SEARCH}')
                return Cf, THf, m
            if bud.expired():
                return None, None, None
    return None, None, None


def phase_control(log, bud):
    log.append('--- PHASE S0 : positive controls ---')
    C, TH, COL = load_cert()
    print(f'[S0] certified packing N={len(C)}  sha256(packing67.txt)={sha(SEED_R5)}')
    ok = V.report(V.verify(C[:, 0], C[:, 1], TH, 5.0), 'S0a certified 67 @R=5.0 (expect LEGAL)', log)
    bad = V.verify(C[:, 0], C[:, 1], TH, R_TARGET)
    V.report(bad, f'S0b same config @R={R_TARGET} (expect ILLEGAL by containment)', log)
    sols = {}
    for rot, nm in [(False, 'T'), (True, 'TR')]:
        Cs = C*0.998
        m0 = K.metrics(Cs, TH, 5.0)
        print(f'[S0c-{nm}] solver control: 67 centres scaled x0.998 at R=5.0 -> '
              f'ov={m0[0]:.2e} minGap={m0[2]:+.1e}; a legal config EXISTS at 5.0 so this '
              f'MUST be repaired')
        Cf, THf, mf, won = relax(Cs, TH.copy(), bud, rot=rot, iters=300, tag=f'S0c {nm} @5.0',
                                 R=5.0)
        chk = V.verify(Cf[:, 0], Cf[:, 1], THf, 5.0)
        sols[nm] = won and chk['LEGAL']
        log.append(f'[S0c-{nm}] repaired={won}, independent verifier LEGAL={chk["LEGAL"]} '
                   f'(sep {chk["implA_sat_worst_separation"]:+.2e}, R_impl '
                   f'{chk["enclosing_radius"]:.10f})')
    # informative only, NOT part of the pass criterion: the pure-shrink route should
    # stall here, reproducing the earlier first-order jamming result in this machinery.
    for rho in (0.9999, 0.999):
        Cs = C67_shrink = C*rho
        Cf, THf, mf, won = relax(Cs, TH.copy(), bud, rot=True, iters=300,
                                 tag=f'S0d pure shrink rho={rho}')
        log.append(f'[S0d] pure shrink rho={rho}: legal={won} ov={mf[0]:.2e} '
                   f'out={mf[1]:.2e} R_impl={mf[3]:.9f}  (expected to stall: reproduces '
                   f'the first-order jam)')
    verdict = ok and (not bad['LEGAL']) and sols['T'] and sols['TR']
    log.append(f'[S0] CONTROL VERDICT: {"PASS" if verdict else "FAIL"}  '
               f'(legal@5.0={ok}, illegal@{R_TARGET}={not bad["LEGAL"]}, '
               f'repair T={sols["T"]} TR={sols["TR"]})')
    print(f'[S0] CONTROL VERDICT: {"PASS" if verdict else "FAIL"}')
    return verdict, C, TH


def phase_insert(bud, log):
    """S1: lattice base + compaction + sequential insertion."""
    log.append('--- PHASE S1 : lattice base + compact + sequential insertion '
               '(rotation available) ---')
    best = [0, None, None]          # best count that is LEGAL at R_SEARCH
    if not os.path.exists(LATTICE):
        log.append(f'[S1] {LATTICE} missing, phase skipped'); return False, None, None
    d = json.load(open(LATTICE))
    C = np.array([d['x'], d['y']]).T; TH = np.zeros(len(C))
    log.append(f'[S1] base = {LATTICE} N={len(C)} h={d["h"]} '
               f'lattice-exact={d["exact_containment"]} sha256={sha(LATTICE)}')
    print(f'[S1] base N={len(C)} from {LATTICE}')
    C, TH, m, won = compact_polish(C, TH, bud, 'S1 base', log)
    if won:
        best = [len(C), C.copy(), TH.copy()]
    while len(C) < K.N_SQ and not bud.expired():
        Cf, THf, mf = try_place(C, TH, bud, log, f'S1 ->N={len(C)+1}')
        if Cf is None:
            log.append(f'[S1] STALLED at N={len(C)}: no insertion site could be made legal')
            print(f'[S1] STALLED at N={len(C)}')
            break
        C, TH = Cf, THf
        if len(C) > best[0]:
            best = [len(C), C.copy(), TH.copy()]
        C, TH, m, won = compact_polish(C, TH, bud, f'S1 N={len(C)}', log)
    print(f'[S1] end: N={len(C)}/{K.N_SQ}  best legal={best[0]}  LPs={bud.lps}')
    log.append(f'[S1] end: N={len(C)}/{K.N_SQ}  best LEGAL count at R_SEARCH = {best[0]}')
    if len(C) == K.N_SQ:
        dump('g3_cand_S1.json', C, TH, dict(seed='lattice+compact+insertion'))
        return True, C, TH
    return False, best[1], best[2]


def phase_excursion(bud, log, C67, TH67):
    """S2: the jam at 5.0 is a *coupling*, so delete k squares, shrink the rest into
    the target, relax (now unblocked), then refill the k vacancies."""
    log.append('--- PHASE S2 : remove-k + shrink + relax + refill (excursion search) ---')
    rim = np.argsort(-np.hypot(C67[:, 0], C67[:, 1]))
    rng = np.random.default_rng(20261009)
    best = [0, None, None]          # only counts states LEGAL at R_SEARCH
    for k in (1, 2, 3, 4):
        for mode in ('rim', 'random'):
            for trial in range(3):
                if bud.expired():
                    return False, best[1], best[2]
                if mode == 'rim':
                    # distinct sets: slide the window over the rim ordering instead of
                    # re-running the identical top-k choice three times
                    pool = rim[:12]
                    pick = pool[(trial*k) % 12:(trial*k) % 12 + k]
                    idx = pick if len(pick) == k else np.roll(pool, trial*k)[:k]
                else:
                    idx = rng.choice(len(C67), size=k, replace=False)
                keep = np.setdiff1d(np.arange(len(C67)), idx)
                s = R_SEARCH/5.0
                C = C67[keep]*s; TH = np.zeros(len(C))
                m0 = K.metrics(C, TH, R_SEARCH)
                print(f'[S2] k={k} {mode}#{trial}: removed {[int(i) for i in idx]} '
                      f'-> ov={m0[0]:.2e} out={m0[1]:.2e} minGap={m0[2]:+.1e}  '
                      f'budget {bud.left():.0f}s')
                log.append(f'[S2] k={k} {mode}#{trial} removed={sorted(int(i) for i in idx)} '
                           f'start ov={m0[0]:.2e} out={m0[1]:.2e} minGap={m0[2]:+.2e}')
                C, TH, m, won = relax(C, TH, bud, rot=True, iters=400,
                                      tag=f'S2 k={k} {mode}#{trial} relax')
                if not won:
                    C, TH, m, won = compact_polish(C, TH, bud, f'S2 k={k} {mode}#{trial}', log)
                if won and len(C) > best[0]:
                    best = [len(C), C.copy(), TH.copy()]
                if not won:
                    continue
                done = True
                for j in range(k):
                    Cf, THf, mf = try_place(C, TH, bud, log,
                                            f'S2 k={k} {mode}#{trial} refill{j}')
                    if Cf is None:
                        done = False; break
                    C, TH = Cf, THf
                    if len(C) > best[0]:
                        best = [len(C), C.copy(), TH.copy()]
                if done and len(C) == K.N_SQ:
                    log.append(f'[S2] SUCCESS: 67 restored at R_SEARCH via k={k} {mode}#{trial}')
                    dump('g3_cand_S2.json', C, TH,
                         dict(seed=f'excursion k={k} {mode} trial{trial}'))
                    return True, C, TH
    log.append(f'[S2] end: best intermediate count = {best[0]}/{K.N_SQ}')
    print(f'[S2] end: best N={best[0]}')
    return False, best[1], best[2]


KICKS = ('pos', 'rot', 'both', 'rotbig', 'mosaic', 'rim')


def kick(C67, TH67, rng, mode, sp, st, rim):
    """a basin-hopping kick on the shrunk certified 67."""
    s = R_SEARCH/5.0
    C = C67*s
    TH = np.zeros(len(C))
    if mode == 'pos':
        C = C + rng.normal(0, sp, C.shape)
    elif mode == 'rot':
        TH = rng.normal(0, st, len(TH))
    elif mode == 'both':
        C = C + rng.normal(0, sp, C.shape); TH = rng.normal(0, st, len(TH))
    elif mode == 'rotbig':                       # every angle redrawn
        C = C + rng.normal(0, sp, C.shape)
        TH = rng.uniform(0.0, math.pi/4, len(TH))
    elif mode == 'mosaic':                       # 0 / pi-4 mix: interlocking family
        C = C + rng.normal(0, sp, C.shape)
        TH = (rng.random(len(TH)) < 0.35)*math.pi/4 + rng.normal(0, 0.05, len(TH))
    elif mode == 'rim':                          # only the binding rim squares
        C = C.copy()
        idx = rim[:12]
        C[idx] += rng.normal(0, sp, (len(idx), 2))
        TH = np.zeros(len(TH)); TH[idx] = rng.normal(0, st, len(idx))
    return C, TH


KICK_SCALE = {
    'pos':    ([0.005, 0.01, 0.02, 0.05], [0.0]),
    'rot':    ([0.0], [0.02, 0.05, 0.10, 0.20]),
    'both':   ([0.01, 0.03, 0.06], [0.02, 0.06, 0.12]),
    'rim':    ([0.01, 0.02, 0.04, 0.08], [0.02, 0.05, 0.10, 0.20]),
    'rotbig': ([0.02, 0.05, 0.10], [0.0]),
    'mosaic': ([0.02, 0.05, 0.10, 0.20], [0.0]),
}


def probe_chain(C, TH, bud, log, tag, m0, loose=1e-6):
    """repair probe, then descending probes.  The repair probe starts at the kicked
    state's OWN enclosing radius (+ a hair), because a heavy kick is not remotely
    feasible at 5.005 and a chain that begins below the state's radius can only ever
    report 'dead' - which would be a property of the probe, not of the basin.
    Kicks needing more than 0.10 of repair are dropped outright: budget is for
    moderate kicks that can actually reach the wall.
    Returns (C, TH, metrics, reached_target, status)."""
    r0 = max(5.005, m0[3] + 0.010)
    if r0 > 5.115:
        return C, TH, m0, False, 'kick too violent'
    rungs = [(r0, 200)]
    if r0 > 5.016:
        rungs += [(round(float(x), 5), 150) for x in np.linspace(r0, 5.005, 4)[1:]]
    rungs += [(4.9950, 220), (R_SEARCH, 300)]
    status = 'repair failed'
    for (R, iters) in rungs:
        C, TH, m, won = relax(C, TH, bud, iters=iters, R=R, rho0=0.03,
                              tag=f'{tag} probe R={R:.5f}')
        alive = won or (m[0]+m[1] <= loose)
        if not alive:
            return C, TH, m, False, status
        status = f'alive@{R:.5f}'
        if R <= R_SEARCH + 1e-12:
            return C, TH, m, bool(m[0] <= TOL and m[1] <= TOL), status
        if bud.expired():
            return C, TH, m, False, status
    return C, TH, m, False, status


def phase_hop(bud, log, C67, TH67, seed=20261009):
    """S3: basin hopping.  The LP relaxer is the strong local optimiser (it repairs
    ov=3.9e-01 to 1e-14 in 2 iterations), so the global budget is spent on KICKS
    (position / rotation / both / all-angles-redrawn / 0-or-45deg mosaic / rim-only)
    followed by a 3-probe descent.  A gap-weight trade pass (wg=30, 8) is interleaved
    because wg=300 is feasibility-first and structurally forbids buying containment
    with a little overlap - the only trade that could cross the 5.0 wall."""
    log.append('--- PHASE S3 : basin hopping (kick families x 3-probe descent '
               '5.005 / 4.995 / 4.9895) ---')
    rng = np.random.default_rng(seed)
    s = R_SEARCH/5.0
    rim = np.argsort(-np.hypot(C67[:, 0], C67[:, 1]))
    best = (9e9, None, None, None)
    by_mode = {}
    attempts = 0
    while not bud.expired():
        attempts += 1
        mode = str(rng.choice(KICKS))
        wg = float(rng.choice([300.0, 30.0, 8.0]))
        ps, ts = KICK_SCALE[mode]
        sp = float(rng.choice(ps)) if ps else 0.0
        st = float(rng.choice(ts)) if ts else 0.0
        C, TH = kick(C67, TH67, rng, mode, sp, st, rim)
        m0 = K.metrics(C, TH, R_SEARCH)
        print(f'[S3] #{attempts} kick={mode:7s} wg={wg:g} sp={sp} st={st}: '
              f'start ov={m0[0]:.2e} out={m0[1]:.2e} R_impl={m0[3]:.6f}  '
              f'budget left {bud.left():.0f}s')
        t0 = time.time()
        if wg < 300.0:
            C, TH, m, _ = relax(C, TH, bud, iters=100, wg=wg, rho0=0.03,
                                tag=f'S3 #{attempts} trade wg={wg:g}')
            m0 = K.metrics(C, TH, R_SEARCH)
        C, TH, mfin, found, status = probe_chain(C, TH, bud, log, f'S3 #{attempts}', m0)
        dt = time.time()-t0
        rank = float(status.split('@')[1]) if '@' in status else 5.115
        if found:
            C, TH, mfin, won2 = relax(C, TH, bud, iters=300, R=R_SEARCH, rho0=0.03,
                                      tag=f'S3 #{attempts} final')
            found = bool(won2)
        val = mfin[0]+mfin[1]
        by_mode.setdefault(mode, []).append((rank, val, dt))
        if val < best[0]:
            best = (val, C.copy(), TH.copy(), (mode, wg, sp, st))
        if rank < 5.0 - 1e-9 and not found:
            # a branch alive strictly below the certified radius: try to close it
            # exactly and let the INDEPENDENT verifier judge it at its own radius.
            Cp, THp, mp, wonp = relax(C, TH, bud, iters=400, R=rank, rho0=0.02,
                                      tag=f'S3 #{attempts} close@{rank:.5f}')
            if wonp:
                chk = V.verify(Cp[:, 0], Cp[:, 1], THp, rank+1e-9)
                line = (f'[S3] #{attempts} BYPRODUCT: exact legality at R={rank:.6f} '
                        f'(< 5.0); verifier LEGAL={chk["LEGAL"]}  R_impl='
                        f'{chk["enclosing_radius"]:.12f}')
                log.append(line); print(line)
                if chk['LEGAL']:
                    dump(f'g3_byproduct_{rank:.5f}.json', Cp, THp,
                         dict(seed=f'hop kick={mode} wg={wg} sp={sp} st={st}',
                              R_certified=float(rank),
                              note='byproduct of G3, NOT the G3 target 4.99'))
        log.append(f'[S3] #{attempts} kick={mode} wg={wg:g} sp={sp} st={st}: '
                   f'start merit={m0[0]+m0[1]:.4e} -> ov={mfin[0]:.3e} '
                   f'out={mfin[1]:.3e} R_impl={mfin[3]:.9f} status={status} '
                   f'legal_at_target={found} ({dt:.1f}s)')
        if found:
            log.append(f'[S3] SUCCESS on attempt #{attempts}: 67 legal at R_SEARCH='
                       f'{R_SEARCH} via kick={mode} wg={wg} sp={sp} st={st}')
            dump('g3_cand_S3.json', C, TH,
                 dict(seed=f'hop kick={mode} wg={wg} sp={sp} st={st} attempt={attempts}'))
            return True, C, TH
    fam = {}
    for k, v in by_mode.items():
        fam[k] = dict(n=len(v), best_alive_R=float(min(x[0] for x in v)),
                      min_merit=float(min(x[1] for x in v)),
                      mean_sec=round(float(np.mean([x[2] for x in v])), 1))
    all_rank = [x[0] for v in by_mode.values() for x in v]
    best_rank = min(all_rank) if all_rank else None
    hist = {}
    for r in all_rank:
        hist[r] = hist.get(r, 0)+1
    log.append(f'[S3] end: attempts={attempts}  best merit={best[0]:.6e} via {best[3]}')
    log.append(f'[S3] branch-survival histogram (R : attempts still alive at/below it) '
               f'-> ' + '  '.join(f'{k:.5f}:{v}' for k, v in sorted(hist.items())))
    log.append(f'[S3] per-kick distribution -> ' + '  '.join(
        f'{k}: n={d["n"]} bestR={d["best_alive_R"]:.5f} mer={d["min_merit"]:.2e} '
        f'{d["mean_sec"]}s' for k, d in sorted(fam.items(),
                                               key=lambda z: z[1]['best_alive_R'])))
    print(f'[S3] end: attempts={attempts} best merit={best[0]:.4e} best alive R='
          f'{best_rank}  hist={hist}')
    json.dump(dict(attempts=attempts, best_alive_R=best_rank,
                   survival_histogram=hist, per_kick=fam,
                   note='best_alive_R=5.006 means no branch was alive below 5.005'),
              open('g3_hop_summary.json', 'w'), indent=1)
    return False, best[1], best[2]


if __name__ == '__main__':
    bud_sec = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
    phases = (sys.argv[2] if len(sys.argv) > 2 else 'S0,S1,S2,S3').split(',')
    log = [f'G3 search log   start={time.strftime("%Y-%m-%d %H:%M:%S")}',
           f'R_TARGET={R_TARGET}  R_SEARCH={R_SEARCH} (safety band)  SIDE=1.0 exactly',
           'dofs: continuous translation + continuous rotation',
           f'budget={bud_sec:g}s  phases={phases}  BETA={BETA}',
           f'seeds: {SEED_R5} sha256={sha(SEED_R5)}  {LATTICE} sha256={sha(LATTICE)}',
           'GATE-0 status = PARTIALLY PASSED (judging-chain controls passed; the '
           'specified 68-square baseline calibration is incomplete).',
           'The radii 5.052404006121 and 5.052605585675 keep their own provenance and '
           'are NOT part of G3 success criteria. mmax68_R5_candidate.csv is NOT a legal '
           'baseline; it is kept for traceability only.']
    result = dict(R_target=R_TARGET, R_search=R_SEARCH, side=1.0, N_target=K.N_SQ,
                  budget_sec=bud_sec, phases=phases, gate0_status='PARTIALLY PASSED',
                  excluded_radii_note='5.052404006121 / 5.052605585675 are not G3 criteria',
                  seed_hashes={SEED_R5: sha(SEED_R5), LATTICE: sha(LATTICE)})
    bud = Budget(bud_sec)
    ctrl, C67, TH67 = phase_control(log, bud)
    result['control'] = ctrl
    if not ctrl:
        log.append('ABORT: positive control failed -> the judging chain is not trustworthy')
        print('ABORT: positive control failed')
        result['outcome'] = 'CONTROL_FAILED_NO_CLAIM'
    else:
        found, Cb, THb = False, None, None
        searched = any(p in phases for p in ('S1', 'S2', 'S3'))
        if not searched:
            result['outcome'] = 'CONTROLS_ONLY_NO_SEARCH_RUN'
            log.append('RESULT: CONTROLS_ONLY_NO_SEARCH_RUN. Only phase S0 executed; no '
                       'claim of any kind about R=' + str(R_TARGET) + ' is recorded here.')
        if 'S1' in phases:
            found, C1, TH1 = phase_insert(bud, log)
            if C1 is not None:
                Cb, THb = C1, TH1
        if not found and 'S2' in phases:
            found, C2, TH2 = phase_excursion(bud, log, C67, TH67)
            if C2 is not None and (Cb is None or len(C2) > len(Cb)):
                Cb, THb = C2, TH2
        if not found and 'S3' in phases:
            found, C3, TH3 = phase_hop(bud, log, C67, TH67)
            if C3 is not None and (Cb is None or len(C3) >= len(Cb)):
                Cb, THb = C3, TH3
        if found:
            chk = V.verify(Cb[:, 0], Cb[:, 1], THb, R_TARGET)
            V.report(chk, 'G3 candidate @ R_TARGET (independent verifier)', log)
            result['outcome'] = ('FOUND_AND_VERIFIED' if chk['LEGAL']
                                 else 'FOUND_BUT_NOT_VERIFIED')
            result['verification'] = chk
            result['impl_R'] = chk['enclosing_radius']
            if chk['LEGAL']:
                log.append(f'RESULT: FOUND_AND_VERIFIED, implicit R = '
                           f'{chk["enclosing_radius"]:.12f} <= {R_TARGET}  =>  '
                           f'r(67) <= {R_TARGET}')
        else:
            result['outcome'] = 'BUDGET_EXHAUSTED_NOT_FOUND'
            if Cb is not None:
                m = K.metrics(Cb, THb, R_SEARCH)
                result['best_state'] = dict(N=int(len(Cb)), ov=float(m[0]),
                                            out=float(m[1]), minGap=float(m[2]),
                                            R_impl=float(m[3]),
                                            note='NOT legal; N only counts squares')
                dump('g3_best_partial.json', Cb, THb,
                     dict(note='best partial state, NOT a result'))
            log.append('RESULT: BUDGET_EXHAUSTED_NOT_FOUND. This records only that THIS '
                       'search (these seeds, this budget) produced no independently verified '
                       f'legal 67-square arrangement at R={R_TARGET}. It is NOT a proof that '
                       'none exists and it does not raise any lower bound.')
    result['lps'] = bud.lps
    result['wall_sec'] = round(bud_sec - bud.left(), 1)
    log.append(f'[stats] LP solves={bud.lps}  wall={result["wall_sec"]}s')
    print(f'\n[stats] outcome={result["outcome"]}  LPs={bud.lps}  '
          f'wall={result["wall_sec"]}s  best_state={result.get("best_state")}')
    json.dump(result, open('g3_result.json', 'w'), indent=1)
    open('g3_log.txt', 'w', encoding='utf-8').write('\n'.join(log)+'\n')

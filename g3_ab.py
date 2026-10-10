import numpy as np, math, time, json, sys
import g3_grow as GR, g3_verify as V

# =====================================================================
# G3-GROW  PAIRED A/B : does ROTATION help cross the local-structure barrier?
#
#   Parameterisation locked by the user: container FIXED at R=5, the 67 squares
#   share ONE common side s, and s is MAXIMISED.  Scale-equivalence makes
#     verified s > 1          <=>  r(67) < 5,
#     verified s >= 5/4.99    <=>  r(67) <= 4.99      (S_TARGET = 1.002004008016)
#
#   arm T  : theta frozen at its initial value (translation + side only)
#   arm TR : theta free (translation + rotation + side)
#   THE ONLY DIFFERENCE is whether theta may change.  Identical initial
#   configuration (the certified 67, theta == 0), identical PRE-DRAWN schedule
#   (same removal sets, same candidate offsets, same drawn angles -- arm T is
#   handed (0.0,) as its angle list), identical solver parameters, identical
#   budget, identical tolerances.
#
#   WHY THIS OPERATION SET (measured in g3_probe_rim.py and g3_seed_probe.py, not
#   assumed): the certified 67 has EXACTLY 8 squares with a corner at
#   5.000000000000 and the next shell sits at 4.924428901.  Uniform dilation keeps
#   every gap at its current value, so growing s is possible iff no corner stays
#   pinned -- removing 2 or 4 of the 8 leaves the rest pinned and the growth LP
#   returns ds = 0 (JAMMED).  That is why the first-generation run (k=3..4, output
#   preserved in g3_ab_log.txt / g3_ab_summary.json) could never score anything:
#   it never freed the pinned set.  Given that, the ORDER of the two remaining
#   steps is decided by measurement, not taste: growing the 59 survivors to their
#   own ceiling (s=1.024) fills the annulus the rim squares need and buries every
#   hole 0.4-0.7 deep, while simply DILATING them to the target side is already
#   legal with 0 LPs and leaves every hole 0.0396 deep.  So: remove-k -> dilate to
#   s -> re-seat -> polish at fixed s -> GROW the restored 67.
#
#   Output goes to g3_ab2_* so the first-generation artifacts stay readable.
#
#   PRIMARY METRIC = largest s that passes INDEPENDENT verification, judged twice:
#   in the s-frame at radius 5, and again with every coordinate divided by s as
#   UNIT squares at radius 5/s.  Never the LP objective, never the penalty value.
#
#   Existing artifacts (g3_result.json, g3_pass600.txt, lattice logs) are not
#   touched; all output here goes to g3_ab_* / g3_probe_*.
# =====================================================================

R = GR.R_FIX
N = 67
S_TARGET = GR.S_TARGET
# shared candidate-site grid for the refill (identical in both arms)
OFFS = [(dr, dt) for dr in (0.0, .04, -.04, .08, -.08, .12, -.12, .16, -.16)
        for dt in (0.0, .05, -.05, .10, -.10)]
# how deep a re-seated ring square may bite into the bulk before the site is
# refused.  g3_seed_probe.py: at k=8, s=s_target every hole is only 0.0396 deep
# over a dilated bulk, so 0.10 covers that case with margin; deeper bites are
# left to trials that remove more of the rim.  Fixed here, identical in both arms.
PEN = 0.10


def load67():
    rows = [l.split() for l in open('packing67.txt') if l.strip()
            and not l.startswith('#')]
    C = np.array([[(int(r[1])+int(r[3]))/4.0, (int(r[2])+int(r[4]))/4.0]
                  for r in rows])
    return C, np.zeros(len(C))


def verify_both(C, TH, s):
    """dual-implementation AND dual-frame judgement (never this file's model)."""
    a = V.verify(C[:, 0], C[:, 1], TH, R, side=s)
    b = V.verify(C[:, 0]/s, C[:, 1]/s, TH, R/s, side=1.0)
    return dict(LEGAL=bool(a['LEGAL'] and b['LEGAL']),
                s=float(s), R_equiv=float(R/s),
                s_frame=dict(worst_sep=a['implA_sat_worst_separation'],
                             max_area=a['implB_clip_max_overlap_area'],
                             R_impl=a['enclosing_radius'],
                             flaggedA=a['implA_flagged_pairs'],
                             flaggedB=a['implB_flagged_pairs'],
                             agree=a['implA_implB_agreement'], LEGAL=a['LEGAL']),
                unit_frame=dict(worst_sep=b['implA_sat_worst_separation'],
                                max_area=b['implB_clip_max_overlap_area'],
                                R_impl=b['enclosing_radius'],
                                agree=b['implA_implB_agreement'], LEGAL=b['LEGAL']))


def fingerprint(C, TH, s):
    """is this a DIFFERENT structure from the original eight-rim-corner one?"""
    rr, _, _ = GR.corners_s(C, TH, s)
    mx = rr.max(axis=1)
    return dict(pinned_at_R=int((mx > R - 1e-9).sum()),
                at_rim_1e_3=int((mx > R - 1e-3).sum()),
                rotated_squares=int((np.abs(TH) > 1e-6).sum()),
                sum_abs_theta=float(np.abs(TH).sum()),
                R_impl=float(mx.max()),
                min_gap=float(GR.metrics_s(C, TH, s)[2]))


def draw_schedule(trials, seed):
    """ONE arm-independent sequence, drawn once and handed to both arms.
    k ranges over [8,22): 8 is the floor because every one of the 8 rim-pinned
    squares must be freed before s can move at all (dilation preserves gaps), and
    the upper end reaches into the 4.609772 shell, which the probe says is where
    the deep re-seating sites are."""
    rng = np.random.default_rng(seed)
    out = []
    for t in range(trials):
        k = int(rng.integers(8, 22))
        out.append(dict(t=t, k=k,
                        angle=float(rng.uniform(0.0, math.pi/2)),
                        perm=int(rng.integers(0, 97))))
    return out


def rim_pool(C0, TH0, s):
    rr, _, _ = GR.corners_s(C0, TH0, s)
    order = np.argsort(-rr.max(axis=1))
    pinned = np.where(rr.max(axis=1) > R - 1e-9)[0]
    return order, pinned


def run_arm(name, rot, sched, budget, log, grow_iters=80, polish_iters=80):
    """arm driver: for each scheduled op -> remove-k (always all 8 pinned), dilate
    the remainder to the target side, re-seat the holes, polish at fixed s, then
    GROW the restored 67.  Keeps the largest VERIFIED s.
    Returns (summary, best state)."""
    C0, TH0 = load67()
    order, pinned = rim_pool(C0, TH0, 1.0)
    t_end = time.time() + budget
    lps = trials = 0
    tm_ref = tm_pol = tm_grow = 0.0
    best_s, best_state = 1.0, (C0.copy(), TH0.copy())
    rec = dict(refill_ok=0, refill_fail=0, polish_ok=0, polish_fail=0,
               above_1_unverified=0, verifications=0, illegal_verifications=0)
    lines = []
    for d in sched:
        if time.time() > t_end - 5.0:
            break
        trials += 1
        t0 = time.time()
        rm = sorted(set([int(p) for p in pinned] +
                        [int(order[8 + (d['perm'] + 3*i) % 16])
                         for i in range(max(0, d['k'] - len(pinned)))]))
        holes = [C0[i] for i in rm]
        keep = np.setdiff1d(np.arange(N), rm)
        k = len(rm)
        # --- SEED AT THE TARGET SIDE.  g3_seed_probe.py measured why the earlier
        # --- "grow the reduced bulk to its own ceiling, then refill" ordering is
        # --- structurally dead: the expanded bulk FILLS THE ANNULUS the rim squares
        # --- need.  At k=8 it buries every hole 0.4-0.7 deep; the same kept squares
        # --- merely DILATED to s are already legal (0 LPs: gaps scale, so only the
        # --- removed corners could overshoot and they are gone) and leave each hole
        # --- only -0.0396 deep -- a bite polish can absorb.  So growth comes AFTER
        # --- the refill, from the full 67, never before.
        s_try = S_TARGET
        Cg, Tg = C0[keep]*s_try, TH0[keep].copy()
        angles = (0.0,) if not rot else (0.0, d['angle'], math.pi/4)
        t_ref = time.time()
        A, B, ok = GR.refill(Cg, Tg, s_try, [h*s_try for h in holes], angles,
                             offsets=OFFS, allow_pen=PEN)
        t_ref = time.time()-t_ref
        tm_ref += t_ref
        if not ok:
            rec['refill_fail'] += 1
            lines.append(f'[{name}] t={d["t"]:3d} k={k:2d} REFILL REFUSED at '
                         f's={s_try:.9f} ({t_ref:.1f}s)')
            continue
        rec['refill_ok'] += 1
        m_ref = GR.metrics_s(A, B, s_try)
        # --- polish the refilled 67 at the FIXED target side
        tp = time.time()
        C2, T2, m2, it2, won = GR.repair_s(A, B, s_try, rot=rot, rho0=0.02,
                                           iters=polish_iters, deadline=t_end-4.0)
        tm_pol += time.time()-tp
        d_pol = time.time()-tp
        lps += it2
        if not won:
            rec['polish_fail'] += 1
            lines.append(f'[{name}] t={d["t"]:3d} k={k:2d} refill@s_try '
                         f'POLISH FAILED ov={m2[0]:.3e} out={m2[1]:.3e} '
                         f'(refill ov={m_ref[0]:.2e}) LPs={it2} '
                         f'ref={t_ref:.1f}s tot={time.time()-t0:.1f}s')
            continue
        rec['polish_ok'] += 1
        fp_pre = fingerprint(C2, T2, s_try)
        # --- and grow from the refilled, polished 67
        tp = time.time()
        C3, T3, s3, m3, it3, st3 = GR.grow(C2, T2, s_try, rot=rot, iters=grow_iters,
                                           rho_s=0.002, deadline=t_end - 4.0)
        tm_grow += time.time()-tp
        d_grw = time.time()-tp
        lps += it3
        fp = fingerprint(C3, T3, s3)
        note = ''
        if s3 > best_s + 1e-12:
            rec['above_1_unverified'] += 1
            if s3 > 1.0 + 1e-9:
                rec['verifications'] += 1
                v = verify_both(C3, T3, s3)
                if v['LEGAL']:
                    best_s, best_state = s3, (C3.copy(), T3.copy())
                    note = (f'  **VERIFIED s={s3:.12f} -> r(67) <= {R/s3:.12f}'
                            f'{"  **TARGET MET**" if s3 >= S_TARGET else ""}')
                else:
                    rec['illegal_verifications'] += 1
                    note = f'  (verified ILLEGAL: {v["s_frame"]} | {v["unit_frame"]})'
        lines.append(f'[{name}] t={d["t"]:3d} k={k:2d} POLISH OK@s_try '
                     f'(pinned={fp_pre["pinned_at_R"]} minGap={fp_pre["min_gap"]:+.2e}) '
                     f'-> grow s={s3:.9f}({st3},{it3}LPs) '
                     f'ov={m3[0]:.1e} out={m3[1]:.1e} pinned={fp["pinned_at_R"]} '
                     f'rot_sq={fp["rotated_squares"]} sumTh={fp["sum_abs_theta"]:.4f} '
                     f'R_impl={fp["R_impl"]:.9f} '
                     f'[ref {t_ref:.1f}s pol {d_pol:.1f}s grw {d_grw:.1f}s] '
                     f'tot={time.time()-t0:.1f}s{note}')
    txt = '\n'.join(lines)
    print(txt); log.append(txt)
    dur = budget - max(0.0, t_end - time.time() - budget)
    line = (f'[{name}] END trials={trials} LPs={lps} refill_ok={rec["refill_ok"]}/'
            f'{trials} polish_ok={rec["polish_ok"]}/{rec["refill_ok"]} '
            f'above1={rec["above_1_unverified"]} verified_illegal='
            f'{rec["illegal_verifications"]} BEST VERIFIED s={best_s:.12f} '
            f'-> r(67) <= {R/best_s:.12f}  | phase s/trial: refill='
            f'{tm_ref/max(1, trials):.2f} polish={tm_pol/max(1, trials):.2f} '
            f'grow={tm_grow/max(1, trials):.2f}')
    print(line); log.append(line)
    Cb, Tb = best_state
    json.dump(dict(x=Cb[:, 0].tolist(), y=Cb[:, 1].tolist(), th=Tb.tolist(),
                   side=float(best_s), R_fix=R, arm=name,
                   note='best VERIFIED state of this arm'),
              open(f'g3_ab2_best_{name}.json', 'w'), indent=1)
    return dict(arm=name, rot=bool(rot), trials=trials, LPs=lps,
                seconds=round(time.time()-(t_end-budget), 1),
                sec_per_trial_refill=round(tm_ref/max(1, trials), 3),
                sec_per_trial_polish=round(tm_pol/max(1, trials), 3),
                sec_per_trial_grow=round(tm_grow/max(1, trials), 3),
                best_verified_s=float(best_s), best_R_equiv=float(R/best_s),
                reached_target=bool(best_s >= S_TARGET),
                fingerprint=fingerprint(Cb, Tb, best_s), **rec), best_state


if __name__ == '__main__':
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20261009
    log = []

    def say(s):
        print(s, flush=True); log.append(s)

    say(f'[AB] G3-GROW paired A/B.  R_FIX={R}  s_target={S_TARGET:.12f} (=5/4.99)  '
        f'budget/arm={budget}s  seed={seed}')
    say('[AB] Gate-0 = PARTIALLY PASSED (judging-chain controls passed; the 68-square '
        'baseline calibration is incomplete).  5.052404006121 / 5.052605585675 are a '
        'different 口径, are not substituted for each other and are NOT part of this '
        'criterion.')
    say('[AB] arms differ ONLY in whether theta may change.')

    C0, TH0 = load67()
    v0 = verify_both(C0, TH0, 1.0)
    say(f'[AB] control: initial 67 at s=1 -> LEGAL={v0["LEGAL"]}  R_impl(s-frame)='
        f'{v0["s_frame"]["R_impl"]:.12f}  flaggedA/B={v0["s_frame"]["flaggedA"]}/'
        f'{v0["s_frame"]["flaggedB"]}  agree={v0["s_frame"]["agree"]}  '
        f'unit-frame LEGAL={v0["unit_frame"]["LEGAL"]}')
    if not v0['LEGAL']:
        say('[AB] ABORT: the initial configuration does not verify.')
        sys.exit(1)

    order, pinned = rim_pool(C0, TH0, 1.0)
    say(f'[AB] pinned-at-rim squares = {len(pinned)}; rim pool (outermost 12) = '
        f'{order[:12].tolist()}')
    sched = draw_schedule(400, seed)
    say(f'[AB] shared schedule drawn ONCE for both arms: first four ops = '
        f'{[(d["k"], round(d["angle"], 6), d["perm"]) for d in sched[:4]]}')

    out = {}
    for name, rot in (('T', False), ('TR', True)):
        say(f'[AB] === arm {name} (rot={rot}) ===')
        summ, _ = run_arm(name, rot, sched, budget, log)
        out[name] = summ
    a, b = out['T'], out['TR']
    say(f'[AB] PAIRED  T : best verified s={a["best_verified_s"]:.12f} -> r(67) <= '
        f'{a["best_R_equiv"]:.12f}   trials={a["trials"]} LPs={a["LPs"]} '
        f'({a["seconds"]}s)  fp={a["fingerprint"]}')
    say(f'[AB] PAIRED  TR: best verified s={b["best_verified_s"]:.12f} -> r(67) <= '
        f'{b["best_R_equiv"]:.12f}   trials={b["trials"]} LPs={b["LPs"]} '
        f'({b["seconds"]}s)  fp={b["fingerprint"]}')
    top = max(a['best_verified_s'], b['best_verified_s'])
    if top >= S_TARGET:
        say(f'[AB] OUTCOME FOUND_AND_VERIFIED: s >= {S_TARGET:.12f}  =>  r(67) <= 4.99')
    elif top > 1.0 + 1e-9:
        say('[AB] OUTCOME PARTIAL: a VERIFIED s strictly between 1 and s_target '
            '=> NEW upper bound below 5, 4.99 gate not met.')
    else:
        say('[AB] OUTCOME BUDGET_EXHAUSTED_NOT_FOUND: this search, at this budget, '
            'produced no verified s > 1.  NOT a proof that none exists; raises no '
            'lower bound and improves no certificate.')
    json.dump(out, open('g3_ab2_summary.json', 'w'), indent=1)
    open('g3_ab2_log.txt', 'w', encoding='utf-8').write('\n'.join(log)+'\n')

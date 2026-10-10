import numpy as np, json, hashlib, os, shutil, time, math
import g3_verify as V, g3_grow as GR

# =====================================================================
# STEP 1 OF THE USER'S DIRECTIVE: FREEZE THE G3 RECORD.
#
#   * nothing outside g3_frozen/ is written; nothing there is ever overwritten
#     (the script aborts if the directory already exists),
#   * the best legal coordinates are RE-VERIFIED from packing67.txt by the
#     independent verifier before being frozen, so the frozen "best state" is a
#     verified object, not a search log,
#   * the verdict text is the user's table, with the scope limits written into
#     the same file that carries the numbers: the measurements evidence a rigid
#     obstacle in THIS branch; they do not prove that no 67-square arrangement
#     shrinks below radius 5, and they raise no lower bound.
# =====================================================================

FROZEN = 'g3_frozen'
COPIES = [
    # the certificate itself (NOT modified -- this is the object being frozen)
    'packing67.txt', 'verify67.mjs',
    # judging chain
    'gate0.py', 'g3_verify.py', 'gate0_log.txt', 'gate0_summary.json',
    # fixed-R / variable-side branch
    'g3_grow.py', 'g3_grow_controls.log', 'g3_grow_controls2.log',
    'g3_ab.py', 'g3_ab_log.txt', 'g3_ab_summary.json',
    'g3_ab2_log.txt', 'g3_ab2_summary.json', 'g3_ab2_calib.log',
    'g3_ab2_best_T.json', 'g3_ab2_best_TR.json',
    'g3_seed_probe.py', 'g3_seed_probe.log', 'g3_probe_rim.log',
    'g3_dilate_log.txt', 'g3_dilate_quick.log',
    'g3_barrier.py', 'g3_barrier.log', 'g3_barrier.json',
    'g3_byproduct_64.json', 'g3_byproduct_64_verify.json',
]

VERDICT = {
    'branch': 'G3 = fix container R=5, one common side s for all 67 squares, '
              'maximise s (scale-equivalent to r(67) <= 5/s)',
    'best_verified_side': 1.0,
    'best_verified_side_digits': '1.000000000000',
    'corresponding_legal_radius': 5.0,
    'target': 'R = 4.99, i.e. s >= 5/4.99 = 1.002004008016',
    'target_met': False,
    'paired_T_TR': 'neither arm left the original contact structure; both arms '
                   'end with pinned=8, rotated_squares=0, sum|theta|=0',
    'directional_probe_in_fixed_structure': '2544 exact metric evaluations over '
                   '636 directions (coordinate, single-square rotation, 300 '
                   'random joint directions, the hard-contact LP solution '
                   'itself): 0 directions lower the rim overshoot without '
                   'increasing the overlap',
    'outcome': 'BUDGET_EXHAUSTED_NOT_FOUND',
    'evidenced': [
        'the dilation ray is first-order rigid at EVERY side above 1 '
        '(s=1.0002..1.020 all return pred_out = 100.00%, T and TR identically)',
        'the freeze is not an artifact of the penalty weight (wg 300 -> 0.003 '
        'moves the state by 0.00000) nor of the trust region (rho 0.002 -> 1.0 '
        'give the identical value)',
        'the contact graph is saturated: 196 pairs at gap <= 1e-9, 252 tied-axis '
        'rows against 2N = 134 degrees of freedom',
    ],
    'NOT_evidenced': [
        'NOT a proof that no 67-square arrangement fits below radius 5 '
        '(global infeasibility is not established by a local/first-order probe)',
        'NO lower bound r(67) >= 5 is raised; the certificate r(67) <= 5 is '
        'unchanged',
        'NOT a proof that the 67/68 decision at R=5 is settled in either '
        'direction',
    ],
    'recording_rules_upheld': [
        'Gate-0 = PARTIALLY PASSED (judging-chain controls pass; the 68-square '
        'baseline calibration is incomplete -- never written as fully passed)',
        'D:/HeliumDownload/mmax68_R5_candidate.csv is NOT a legal baseline; kept '
        'for traceability only; no calibration was fabricated by scaling '
        'coordinates or resizing squares',
        '5.052404006121 and 5.052605585675 keep separate provenance, are not '
        'substituted for each other, and are not part of the G3 criterion',
        'the falsified R=5.0 area upper bound is not used anywhere in this '
        'branch',
    ],
}


def sha256(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


if __name__ == '__main__':
    if os.path.exists(FROZEN):
        raise SystemExit(f'[FREEZE] {FROZEN}/ already exists -- refusing to '
                         f'overwrite a frozen record. Inspect it instead.')
    os.makedirs(FROZEN)

    # ---- re-verify the best legal state from the certificate, then freeze it
    rows = [l.split() for l in open('packing67.txt') if l.strip()
            and not l.startswith('#')]
    C = np.array([[(int(r[1])+int(r[3]))/4.0, (int(r[2])+int(r[4]))/4.0]
                  for r in rows])
    TH = np.zeros(len(C))
    chk = V.verify(C[:, 0], C[:, 1], TH, 5.0, side=1.0)
    print(f'[FREEZE] best legal state re-verified independently: N={chk["N"]} '
          f'LEGAL={chk["LEGAL"]}  implA worst sep={chk["implA_sat_worst_separation"]:+.3e} '
          f'implB max area={chk["implB_clip_max_overlap_area"]:+.3e} '
          f'A/B agree={chk["implA_implB_agreement"]}  R_impl={chk["enclosing_radius"]:.12f} '
          f'corners over 5: {chk["corners_over_R"]}')
    if not chk['LEGAL']:
        shutil.rmtree(FROZEN)
        raise SystemExit('[FREEZE] ABORT: the state to be frozen does not verify.')
    json.dump(dict(x=C[:, 0].tolist(), y=C[:, 1].tolist(), th=TH.tolist(),
                   side=1.0, R=5.0, N=len(C),
                   note='G3 best VERIFIED state: the certified 67 at s=1, i.e. '
                        'r(67) <= 5.000000000000. Unchanged by this branch.'),
              open(os.path.join(FROZEN, 'best_legal_67.json'), 'w'), indent=1)
    json.dump(chk, open(os.path.join(FROZEN, 'best_legal_67_verify.json'), 'w'),
              indent=1)
    json.dump(VERDICT, open(os.path.join(FROZEN, 'VERDICT.json'), 'w'),
              indent=1, ensure_ascii=False)

    missing = []
    for f in COPIES:
        if os.path.exists(f):
            shutil.copy2(f, os.path.join(FROZEN, f))
        else:
            missing.append(f)
    print(f'[FREEZE] copied {len(COPIES)-len(missing)}/{len(COPIES)} artifacts'
          + (f'; MISSING {missing}' if missing else ''))

    man = {'frozen_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'working_dir': os.getcwd().encode('utf-8', 'replace').decode('utf-8'),
           'files': {}}
    for f in sorted(os.listdir(FROZEN)):
        p = os.path.join(FROZEN, f)
        man['files'][f] = {'sha256': sha256(p), 'bytes': os.path.getsize(p)}
    man['verdict_sha256'] = man['files']['VERDICT.json']['sha256']
    man['packing67_sha256'] = man['files']['packing67.txt']['sha256']
    man['originals_untouched'] = {'packing67.txt': sha256('packing67.txt'),
                                 'g3_ab_summary.json(first-gen)': sha256('g3_ab_summary.json')}
    json.dump(man, open(os.path.join(FROZEN, 'MANIFEST.json'), 'w'), indent=1,
              ensure_ascii=False)
    print(f'[FREEZE] MANIFEST written: {len(man["files"])} files, '
          f'packing67.txt sha256={man["packing67_sha256"][:16]}... (identical to '
          f'the live copy: {man["packing67_sha256"] == sha256("packing67.txt")})')
    print(f'[FREEZE] branch is CLOSED FOR FURTHER T/TR SEARCH at this contact '
          f'structure, per the user directive. Next: G2 tau>=68 closing gate.')

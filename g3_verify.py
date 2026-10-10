import numpy as np, json, sys, math, hashlib
import gate0 as G

# =====================================================================
# G3 INDEPENDENT VERIFIER.
# Deliberately shares NO code with the searcher's linearised model:
#   impl A = SAT over the 4 edge normals  (gate0.sep_sat)
#   impl B = Sutherland-Hodgman polygon clipping + shoelace area (gate0.clip)
#   containment = explicit rotation of all 4 corners of every square
# Tolerances are the ones PRE-REGISTERED in gate0.py (1e-9), not tuned here.
#
# SIDE: the searcher may propose a COMMON side s (G3-GROW parameterisation).
# The pre-registered 1e-9 tolerances are for UNIT squares, so for side s they
# are used in their DERIVED, dimensionally-correct form
#     linear tolerance   -> s   * 1e-9      (a length scales like s)
#     area  tolerance    -> s^2 * 1e-9      (an area scales like s^2)
# Both reduce to EXACTLY 1e-9 at s=1, so no previous unit-square verdict moves.
# This is scale-equivalence, not a fitted slack: verifying (x,y,th,s) at radius
# R with these tolerances is the same statement as verifying the arrangement
# scaled by 1/s as unit squares at radius R/s with the original 1e-9.  verify()
# is called BOTH ways by the grow driver, so a candidate is judged by the s-frame
# pass and again by the unit-frame pass.
#
# TOL_AREA is DERIVED, not fitted: for two unit squares an overlap of linear
# penetration depth d has intersection area <= d * 1, so "area <= TOL_OVERLAP"
# is the area-space restatement of the same 1e-9 linear tolerance.  Using a
# tighter magic number (an earlier draft used 1e-15) would make impl B reject
# float noise rather than geometry, and looser would weaken the test.
# =====================================================================


def tol_area(side):
    """area-space restatement of the pre-registered linear tolerance at side s."""
    return side*side*G.TOL_OVERLAP


def flagged_sets(x, y, th, C, s=G.SIDE):
    """per-pair verdicts from BOTH implementations, as sets of flagged pairs."""
    n = len(x)
    A, B = set(), set()
    to, ta = s*G.TOL_OVERLAP, tol_area(s)
    for i in range(n):
        Qi = G.ccw([tuple(p) for p in C[i]])
        for j in range(i+1, n):
            if G.sep_sat(x[i], y[i], th[i], x[j], y[j], th[j], s) < -to:
                A.add((i, j))
            Qj = G.ccw([tuple(p) for p in C[j]])
            inter = G.clip(Qi, Qj)
            a = G.poly_area(inter) if len(inter) >= 3 else 0.0
            if a > ta:
                B.add((i, j))
    return A, B


def verify(x, y, th, R, side=1.0, tol_ov=None, tol_ct=None):
    tol_ov = side*G.TOL_OVERLAP if tol_ov is None else tol_ov
    tol_ct = side*G.TOL_CONTAIN if tol_ct is None else tol_ct
    ta = tol_area(side)
    n = len(x)
    C = G.corners(x, y, th, side)
    sat, si, sj = G.pairwise_sat(x, y, th, side)
    ar, ai, aj = G.pairwise_clip(C, side)
    rmax, arg, rr = G.enclosing_radius(C)
    A, B = flagged_sets(x, y, th, C, side)
    checks = {
        'N': n,
        'side_exact': side,
        'implA_sat_worst_separation': sat,
        'implA_worst_pair': [si, sj],
        'implA_disjoint': bool(sat >= -tol_ov),
        'implB_clip_max_overlap_area': ar,
        'implB_worst_pair': [ai, aj],
        'implB_disjoint': bool(ar <= ta),
        'tol_area_derived_from': 'TOL_OVERLAP (linear penetration) -> area <= depth*side, '
                                 'i.e. s^2*1e-9 at side s (= 1e-9 at s=1)',
        'implA_flagged_pairs': len(A),
        'implB_flagged_pairs': len(B),
        'implA_implB_agreement': (A == B),
        'implA_implB_disagreements': len(A ^ B),
        'corner_count_checked': int(n*4),
        'enclosing_radius': rmax,
        'worst_corner': [int(arg[0]), int(arg[1])],
        'corners_over_R': int((rr > R + tol_ct).sum()),
        'contained': bool(rmax <= R + tol_ct),
        'target_R': R,
        'tolerances': [tol_ov, tol_ct, ta],
        'margin_contain': R - rmax,
        'margin_sep': sat,
        'n_pairs': n*(n-1)//2,
    }
    checks['LEGAL'] = (checks['implA_disjoint'] and checks['implB_disjoint']
                       and checks['contained'] and checks['implA_implB_agreement'])
    return checks


def report(checks, tag, log):
    s = (f'[{tag}] N={checks["N"]}  pairs={checks["n_pairs"]}  corners={checks["corner_count_checked"]}\n'
         f'        implA (SAT)   worst separation = {checks["implA_sat_worst_separation"]:+.12e} '
         f'at {checks["implA_worst_pair"]}  -> disjoint={checks["implA_disjoint"]} '
         f'(flagged {checks["implA_flagged_pairs"]})\n'
         f'        implB (clip)  max overlap area = {checks["implB_clip_max_overlap_area"]:+.12e} '
         f'at {checks["implB_worst_pair"]}  -> disjoint={checks["implB_disjoint"]} '
         f'(flagged {checks["implB_flagged_pairs"]})\n'
         f'        A/B agreement on the flagged set: {checks["implA_implB_agreement"]} '
         f'(disagreements {checks["implA_implB_disagreements"]})\n'
         f'        containment   R_impl = {checks["enclosing_radius"]:.12f} vs target '
         f'{checks["target_R"]:.12f}  (margin {checks["margin_contain"]:+.3e}, '
         f'{checks["corners_over_R"]} corners over) -> {checks["contained"]}\n'
         f'        VERDICT: {"LEGAL" if checks["LEGAL"] else "ILLEGAL"}')
    log.append(s)
    print(s)
    return checks['LEGAL']


if __name__ == '__main__':
    path, R = sys.argv[1], float(sys.argv[2])
    d = json.load(open(path))
    side = float(d.get('side', 1.0))
    log = [f'verifying {path}  sha256={hashlib.sha256(open(path,"rb").read()).hexdigest()}',
           f'target R = {R:.12f}   side = {side:.12f}   tolerances = '
           f'{side*G.TOL_OVERLAP:g} / {side*G.TOL_CONTAIN:g} / area {tol_area(side):g} '
           f'(1e-9 pre-registered in gate0.py, scaled by side)']
    c = verify(np.array(d['x']), np.array(d['y']), np.array(d['th']), R, side)
    report(c, path, log)
    json.dump(c, open(path.replace('.json', '_verify.json'), 'w'), indent=1)
    open('g3_verify_log.txt', 'a', encoding='utf-8').write('\n'.join(log)+'\n')

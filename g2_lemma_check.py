"""Checker validation of the deterministic pose-covering lemma (NOT the proof itself).

Lemma (two cases).  Let h be the family centre spacing, d0 = h/sqrt(2) the lattice
covering radius, rho_in = (sqrt(100)-sqrt(2))/2, rho_max = (sqrt(99)-1)/2,
r_s = rho_in - d0, Delta = rho_max - rho_in.  For every legal centre p (|p| <= rho_max):

  Case A  |p| <= r_s      : nearest lattice node q has |q| <= r_s + d0 = rho_in,
                            so q is legal at EVERY angle, |p-q| <= d0.
  Case B  r_s < |p|       : project p radially to p' with |p'| = r_s, take the nearest
                            node q to p'; |q| <= rho_in so q is legal at every angle, and
                            |p-q| <= (|p| - r_s) + d0 <= Delta + 2*d0.

Both cases give |p-q| <= Delta + 2*d0; adding the rotation term (sqrt2/2)*pi/(4*n_ang)
yields eps_det.  Note disc(rho_in) subset K_theta subset disc(rho_max) is all that is used
-- convexity of K_theta is not needed.

This script only measures how loose that bound is on the real families.
"""
import json, math
import numpy as np
from scipy.spatial import cKDTree
import g2_tau as T

RHO_IN = (math.sqrt(100) - math.sqrt(2)) / 2
RHO_MAX = (math.sqrt(99) - 1) / 2
BUDGET_EPS = 0.00372


def r2(x, y, t):
    return x * x + y * y + 0.5 + np.abs(x * np.cos(t) + y * np.sin(t)) \
        + np.abs(-x * np.sin(t) + y * np.cos(t))


def check(h_c, n_ang, family):
    d0 = h_c / math.sqrt(2)
    det_centre = (RHO_MAX - RHO_IN) + 2 * d0
    r_s = RHO_IN - d0
    ang_rad = math.pi / (4.0 * n_ang)
    eps = det_centre + math.sqrt(2) / 2 * ang_rad

    angs = np.unique(np.round(np.mod(family[:, 2], math.pi / 2), 12))
    th_mod = np.mod(family[:, 2], math.pi / 2)
    trees = {float(t): cKDTree(family[np.isclose(th_mod, float(t), atol=1e-12)][:, :2])
             for t in angs}

    # worst-case query angle = halfway between two family angles (max angular gap)
    th_q = float(angs[0]) + ang_rad
    tf = float(min(angs, key=lambda a: abs(a - th_q)))
    r = np.linspace(0.0, RHO_MAX, 400)
    phi = np.linspace(0.0, 2 * math.pi, 720, endpoint=False)
    RR, PP = np.meshgrid(r, phi, indexing='ij')
    X, Y = RR * np.cos(PP), RR * np.sin(PP)
    keep = r2(X, Y, th_q) <= 25.0
    X, Y, RR = X[keep], Y[keep], RR[keep]
    dd, _ = trees[tf].query(np.column_stack([X, Y]))
    inner, outer = dd[RR <= r_s], dd[RR > r_s]
    res = dict(
        h_c=h_c, n_ang=n_ang, rho_in=RHO_IN, rho_max=RHO_MAX, delta=RHO_MAX - RHO_IN,
        lattice_covering_radius_d0=d0, r_s=r_s,
        det_centre_bound=det_centre, angle_term=math.sqrt(2) / 2 * ang_rad,
        eps_det=eps, footprint_ratio=(1 + 2 * eps) ** 2, blocked_vs_budget=eps / BUDGET_EPS,
        legal_test_points=int(len(dd)),
        case_A_max=float(inner.max()) if len(inner) else 0.0,
        case_A_bound=d0, case_A_ok=bool(len(inner) == 0 or inner.max() <= d0 + 1e-9),
        case_B_max=float(outer.max()) if len(outer) else 0.0,
        case_B_bound=det_centre,
        case_B_ok=bool(len(outer) == 0 or outer.max() <= det_centre + 1e-9),
        looseness_factor=det_centre / max(float(dd.max()), 1e-12),
    )
    print(f'[LEM] h_c={h_c}, n_ang={n_ang}: d0={d0:.6f} Delta={RHO_MAX-RHO_IN:.6f} '
          f'det_centre={det_centre:.6f} | A max={res["case_A_max"]:.6f}<={d0:.6f} '
          f'{res["case_A_ok"]} | B max={res["case_B_max"]:.6f}<={det_centre:.6f} '
          f'{res["case_B_ok"]} | eps_det={eps:.6f} ratio={res["footprint_ratio"]:.4f} '
          f'blocked x{res["blocked_vs_budget"]:.0f} | bound is {res["looseness_factor"]:.2f}x '
          f'looser than measured worst ({len(dd)} test points)', flush=True)
    return res


def main():
    d = json.load(open('g3_frozen/best_legal_67.json'))
    C67 = np.column_stack([d['x'], d['y'], d['th']]).tolist()
    out = []
    for h_c, na in ((0.30, 9), (0.20, 13)):
        out.append(check(h_c, na, T.build_family(h_c, na, extra=C67)))
    json.dump(out, open('g2_lemma_check.json', 'w'), indent=1, default=float)
    print('[LEM] validation only: the bound itself is proved analytically above; this '
          'measures its looseness. It refutes the CURRENT envelope upgrade at the CURRENT '
          'covering precision -- it does NOT show no continuous-pose argument can exist.',
          flush=True)


if __name__ == '__main__':
    main()

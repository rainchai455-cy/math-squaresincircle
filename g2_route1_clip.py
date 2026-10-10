"""Route (i): exact deterministic covering radius by clipping Voronoi cells of the
LEGAL-at-family-angle node set against the legal target region.

No radial projection is used, so the h-independent Delta term of the two-case lemma
does not appear in this bound.  Everything is computed from the same candidate set the
certificates used (g2_tau.build_family), and checked by three independent controls.
"""
import json, math
import numpy as np
from scipy.spatial import Voronoi, cKDTree
import g2_tau as T

R = T.R
RHO_IN = (math.sqrt(100.0) - math.sqrt(2.0)) / 2.0     # 4.292893219 (worst, delta=45deg)
RHO_MAX = (math.sqrt(99.0) - 1.0) / 2.0                # 4.474937186 (universal outer bound)


def nodes_for_angle(g, th):
    """Lattice nodes legal at angle th -- identical criterion to build_family."""
    out = []
    for x in g:
        for y in g:
            if x * x + y * y <= R * R and T.legal(x, y, th):
                out.append((x, y))
    return np.array(out, dtype=float)


def cross2(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def order_convex(poly):
    """Sort vertices cyclically around the centroid and return (poly, orientation)."""
    P = np.asarray(poly, dtype=float)
    c = P.mean(axis=0)
    ang = np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0])
    P = P[np.argsort(ang)]
    area = 0.0
    n = len(P)
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        area += cross2(a, b)
    return P, (1.0 if area >= 0 else -1.0)


def clip_cell_maxdist(site, poly, rad):
    """max_{p in conv(poly) cap disc(rad)} |p - site|, computed exactly.

    |p-site|^2 is convex, so the maximum over the convex clipped region is attained at
    an extreme point: an original vertex inside the disc, or an edge/circle crossing.
    Along a circular arc the only interior maximiser is the antipodal point -rad*site_hat.
    """
    P, orient = order_convex(poly)
    n = len(P)
    inside = np.einsum('ij,ij->i', P, P) <= rad * rad
    cand = [P[i] for i in range(n) if inside[i]]

    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        da, db = inside[i], inside[(i + 1) % n]
        if da == db:
            continue
        d = b - a
        A = float(d @ d)
        B = float(2.0 * (a @ d))
        C = float(a @ a) - rad * rad
        disc = B * B - 4.0 * A * C
        if disc < 0:
            continue
        sq = math.sqrt(disc)
        for t in ((-B - sq) / (2.0 * A), (-B + sq) / (2.0 * A)):
            if -1e-12 <= t <= 1.0 + 1e-12:
                cand.append(a + min(max(t, 0.0), 1.0) * d)

    # antipodal point on the circle, kept only if it lies in the unclipped cell
    ns = math.hypot(site[0], site[1])
    if ns > 1e-12:
        far = np.array([-rad * site[0] / ns, -rad * site[1] / ns])
        edge_ok = True
        for i in range(n):
            a, b = P[i], P[(i + 1) % n]
            if orient * cross2(b - a, far - a) < -1e-12:
                edge_ok = False
                break
        if edge_ok:
            cand.append(far)

    if not cand:
        return 0.0, None
    C = np.array(cand)
    d2 = np.einsum('ij,ij->i', C - site, C - site)
    k = int(np.argmax(d2))
    return math.sqrt(d2[k]), C[k]


def cell_polygons(pts, sites, rad):
    """Voronoi cells of every site, bounded by a far guard ring so all regions are finite.

    Guards sit at radius 60: any p with |p| <= rad is >= 55 from a guard and <= ~0.3 from
    a real site, so guards cannot claim any point of the region of interest.
    """
    m = len(sites)
    ring = np.array([[60.0 * math.cos(a), 60.0 * math.sin(a)]
                     for a in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False)])
    vor = Voronoi(np.vstack([sites, ring]))
    out = {}
    for i, s in enumerate(sites):
        reg = vor.point_region[i]
        vs = vor.regions[reg]
        if -1 in vs or len(vs) == 0:
            continue
        poly = vor.vertices[vs]
        # expand cell to a closed convex polygon; clip later against the disc
        out[i] = (tuple(s), poly)
    return out


def covering_radius(V, rad, npts_grid=0):
    """exact sup_{p in disc(rad)} dist(p, V) via clipped Voronoi cells."""
    cells = cell_polygons(None, V, rad)
    tree = cKDTree(V)
    best = 0.0
    arg = None
    for idx, (site, poly) in cells.items():
        d, p = clip_cell_maxdist(np.array(site), poly, rad)
        if d > best:
            best, arg = d, (site, p)
    # independent re-check: the claimed worst point's KDTree distance must match
    witness = None
    if arg is not None:
        site, p = arg
        kd = float(tree.query(p)[0])
        witness = dict(site=[float(site[0]), float(site[1])],
                       point=[float(p[0]), float(p[1])],
                       cell_dist=best, kdtree_dist=kd,
                       agree=abs(kd - best) < 1e-9)
    return best, witness


def main():
    res = []
    for h_c, n_ang, nm in ((0.30, 9, 'fam(0.30,ang9)'), (0.20, 13, 'fam(0.20,ang13)')):
        g = np.arange(-R, R + 1e-12, h_c)
        angs = np.linspace(0.0, math.pi / 2.0, n_ang, endpoint=False)
        dth = (math.pi / 2.0) / n_ang
        ang_term = (math.sqrt(2.0) / 2.0) * dth / 2.0
        d0 = h_c / math.sqrt(2.0)

        F = T.build_family(h_c, n_ang)
        tot_family = len(F)

        # control 1 does not depend on the angle: all lattice nodes legal, same clip code
        allnodes = np.array([(x, y) for x in g for y in g if x * x + y * y <= R * R])
        d_all, _ = covering_radius(allnodes, RHO_MAX)

        per_ang, cnt_tot = [], 0
        for th in angs:
            V = nodes_for_angle(g, th)
            cnt_tot += len(V)
            d_exact, wit = covering_radius(V, RHO_MAX)
            per_ang.append(dict(theta=float(th), n_legal=int(len(V)),
                                delta_full_lattice=d_all, delta_exact=d_exact,
                                witness=wit))
        # control 2: candidate set must equal the certificate family exactly
        family_match = (cnt_tot == tot_family)
        cent = max(p['delta_exact'] for p in per_ang)
        ctrl = max(p['delta_full_lattice'] for p in per_ang)
        old_bound = (RHO_MAX - RHO_IN) + math.sqrt(2.0) * h_c
        res.append(dict(tag=nm, h_c=h_c, n_ang=n_ang, dtheta=dth,
                        d0_ideal=d0, angle_term=ang_term,
                        n_family_poses=int(tot_family),
                        legal_node_count_match_family=bool(family_match),
                        control_full_lattice_delta=ctrl,
                        delta_centre_exact=cent,
                        delta_pose_additive=cent + ang_term,
                        delta_pose_max=max(cent, ang_term),
                        old_two_case_centre_bound=old_bound,
                        exact_leq_old_bound=bool(cent <= old_bound + 1e-12)))
        print('[R1] %s: exact centre=%.6f (full-lattice control %.6f vs ideal %.6f) '
              'angle=%.6f => pose additive=%.6f / max=%.6f | old two-case bound %.6f | '
              'family match=%s' % (nm, cent, ctrl, d0, ang_term, cent + ang_term,
                                   max(cent, ang_term), old_bound, family_match), flush=True)

    budget = 0.0037172
    for r in res:
        need = budget
        r['budget_epsilon'] = need
        r['blocked_factor_vs_budget'] = r['delta_pose_additive'] / need
        r['area_ratio'] = (1.0 + 2.0 * r['delta_pose_additive']) ** 2
        r['meets_budget'] = bool(r['delta_pose_additive'] <= need)
        print('[R1] %s: budget eps=%.6f, exact additive eps=%.6f => %.1fx too coarse, '
              'area ratio %.4f vs allowed 1.014924' % (r['tag'], need,
              r['delta_pose_additive'], r['blocked_factor_vs_budget'], r['area_ratio']),
              flush=True)

    json.dump(res, open('g2_route1_clip.json', 'w'), indent=1, default=float)
    print('[R1] wrote g2_route1_clip.json', flush=True)


if __name__ == '__main__':
    main()

"""Route (i), sector-exact version.

For each family angle theta_k we bound
    delta_c(k) = sup { |p - v| : p in P_k, v nearest legal node of V_k }
where
    V_k = { lattice nodes legal at theta_k }            (the certificate family itself)
    P_k = union over phi in [theta_k - dth/2, theta_k + dth/2] of K_phi
        = { p : rho^2 + 1/2 + rho*(cos d + sin d) <= 25 },  d = folded distance of
          the direction of p to the sector   (this equality is exact, see notes)
P_k is outer-approximated by a convex polygon whose offsets are the exact radial boundary
plus a certified support-function margin support_offset(L, rho_min), derived from an
analytic Lipschitz bound on rho*.  A chord-size margin O(pi/N) is NOT enough: the kink of
rho* at the sector edge makes the true support gap ~L^2/(2 rho) = 2.3e-2, and control C
(random poses) / control D (deterministic gap) are there precisely to catch that.
delta_c is therefore a deterministic UPPER bound.

No radial projection anywhere: the h-independent Delta=0.182044 term of the two-case
lemma cannot appear, because we clip against the true sector region instead of annulus
arguments.
"""
import json, math
import numpy as np
from scipy.spatial import Voronoi, cKDTree
import g2_tau as T

R = T.R
SQ2 = math.sqrt(2.0)


def sector_rhorange(th_k, w, N):
    """radial boundary rho*(psi) of P_k, plus supporting-halfplane offsets h_m."""
    psi = 2.0 * math.pi * np.arange(N) / N
    # folded angular distance from psi to the sector [th_k-w, th_k+w], period pi/2
    a = (psi - th_k + math.pi / 4.0) % (math.pi / 2.0) - math.pi / 4.0
    d = np.maximum(0.0, np.abs(a) - w)
    g = np.cos(d) + np.sin(d)                      # in [1, sqrt 2]
    rho_star = (-g + np.sqrt(g * g + 4.0 * 24.5)) / 2.0
    return psi, rho_star


def max_slope(psi, rho):
    """empirical sup |d rho*/d psi| (rho is pi/2-periodic and piecewise smooth)."""
    k = 1
    return float(np.max(np.abs(np.diff(rho)) / np.diff(psi)[0])) if k else 0.0


L_ANA = 0.45   # analytic Lipschitz bound for rho*:  sup |d rho/dd| = rho_max/(2 rho_max+1)
               # = 4.474937/9.949874 = 0.449751 < 0.45, and |dd/dpsi| <= 1.


def support_offset(L, rho_min, n=400001):
    """certified sup over psi of [rho(psi)cos(psi-psi_m) - rho(psi_m)].

    With rho L-Lipschitz, for t = |psi-psi_m| in [0, pi/2] (beyond that cos <= 0 and the
    halfplane is trivially satisfied):
        rho(psi)cos t <= (rho(psi_m) + L t) cos t
    so the excess over rho(psi_m) is at most  L t cos t - rho(psi_m)(1 - cos t), which is
    largest for the smallest rho.  Maxima of that one-function on a fine grid are a valid
    upper bound up to the grid modulus, which the printed margin re-checks.
    """
    t = np.linspace(0.0, math.pi / 2.0, n)
    phi = L * t * np.cos(t) - rho_min * (1.0 - np.cos(t))
    return float(np.max(phi))


RHO_IN = (10.0 - math.sqrt(2.0)) / 2.0     # min over all sectors of rho* (g = sqrt 2)
OFFSET = support_offset(L_ANA, RHO_IN)     # certified radial margin of the outer polygon


def sector_polygon(th_k, w, N=512, Lh=None):
    """convex outer approximation of P_k: intersection of x.u_m <= h_m."""
    psi, rho = sector_rhorange(th_k, w, N)
    off = OFFSET if Lh is None else Lh              # certified support margin
    h = rho + off
    u = np.column_stack([np.cos(psi), np.sin(psi)])
    return u, h, psi, rho


def clip_poly_halfplanes(poly, u, h):
    """Sutherland-Hodgman: clip convex `poly` by halfplanes {x : x.u_m <= h_m}."""
    P = poly
    for um, hm in zip(u, h):
        if len(P) == 0:
            return P
        val = P @ um
        keep = val <= hm + 1e-12
        newP = []
        n = len(P)
        for i in range(n):
            a, b = P[i], P[(i + 1) % n]
            va, vb = val[i], val[(i + 1) % n]
            if keep[i]:
                newP.append(a)
            if keep[i] != keep[(i + 1) % n]:
                t = (hm - va) / (vb - va)
                newP.append(a + t * (b - a))
        P = np.array(newP) if newP else np.zeros((0, 2))
    return P


def voronoi_cells(sites):
    ring = np.array([[60.0 * math.cos(a), 60.0 * math.sin(a)]
                     for a in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False)])
    vor = Voronoi(np.vstack([sites, ring]))
    out = []
    for i in range(len(sites)):
        reg = vor.point_region[i]
        vs = vor.regions[reg]
        if -1 in vs or len(vs) == 0:
            return None                             # unbounded: guards failed
        out.append(vor.vertices[vs])
    return out


def order_ccw(P):
    c = P.mean(axis=0)
    ang = np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0])
    return P[np.argsort(ang)]


def delta_centre(V, th_k, w, N=512, Lh=None):
    """max over clipped cells of |p - site|, p in the convex outer polygon of P_k.

    cell (convex) intersect region (convex) is convex, and |p-site|^2 is convex, so the
    maximum is attained at a vertex of the clipped polygon -- no sampling involved.

    A halfplane can only cut the cell if it is already exceeded by a cell vertex (linear
    functions on a convex polytope attain their max at a vertex, and clipping only shrinks
    the set), so selecting those first is exact, not a heuristic.
    """
    u, h, psi, rho = sector_polygon(th_k, w, N, Lh)
    cells = voronoi_cells(V)
    if cells is None:
        raise RuntimeError('unbounded Voronoi region: guard ring too close')
    best, arg, n_cut = 0.0, None, 0
    for site, cell in zip(V, cells):
        cp = order_ccw(np.asarray(cell))
        excess = cp @ u.T - h
        sel = np.nonzero(excess.max(axis=0) > 1e-12)[0]
        if len(sel):
            n_cut += 1
            clipped = clip_poly_halfplanes(cp, u[sel], h[sel])
            if len(clipped) == 0:
                continue
            cand = clipped
        else:
            cand = cp
        d2 = np.einsum('ij,ij->i', cand - site, cand - site)
        j = int(np.argmax(d2))
        if d2[j] > best:
            best = float(d2[j])
            arg = (tuple(site), tuple(cand[j]))
    return math.sqrt(best), arg, (float(np.max(rho)), float(np.min(rho)), n_cut)


def nodes_for_angle(g, th):
    return np.array([(x, y) for x in g for y in g
                     if x * x + y * y <= R * R and T.legal(x, y, th)], dtype=float)


def main():
    out = []
    for h_c, n_ang, nm in ((0.30, 9, 'fam(0.30,ang9)'), (0.20, 13, 'fam(0.20,ang13)')):
        g = np.arange(-R, R + 1e-12, h_c)
        angs = np.linspace(0.0, math.pi / 2.0, n_ang, endpoint=False)
        dth = (math.pi / 2.0) / n_ang
        w = dth / 2.0
        ang_term = (SQ2 / 2.0) * w
        d0 = h_c / SQ2
        F = T.build_family(h_c, n_ang)

        # control A: machinery check -- full lattice, region = the lattice's own covering
        allnodes = np.array([(x, y) for x in g for y in g if x * x + y * y <= R * R])

        rows, sl_max, cut_tot = [], 0.0, 0
        Vcache = {}
        Nf = 8192
        psi_f = 2.0 * math.pi * np.arange(Nf) / Nf
        gap_max, rho_min_all = -np.inf, np.inf
        for th in angs:
            V = nodes_for_angle(g, th)
            Vcache[float(th)] = V
            psi, rho = sector_rhorange(th, w, 4096)
            sl_max = max(sl_max, max_slope(psi, rho))
            d, arg, meta = delta_centre(V, th, w)
            cut_tot += meta[2]
            rho_min_all = min(rho_min_all, meta[1])
            # control D (deterministic): true support-function gap vs the offset actually used
            rf = sector_rhorange(th, w, Nf)[1]
            u, h, psi_c, rho_c = sector_polygon(th, w)
            M = rf[None, :] * np.cos(psi_f[None, :] - psi_c[:, None])
            gap_max = max(gap_max, float(np.max(np.maximum(M, 0.0) - rho_c[:, None])))
            rows.append(dict(theta=float(th), n_legal=int(len(V)), delta_centre=d,
                             rho_max=meta[0], rho_min=meta[1], cells_clipped=meta[2],
                             site=list(arg[0]), point=list(arg[1])))
        offset_used = OFFSET                              # what the clipping actually used
        offset_tight = support_offset(L_ANA, rho_min_all)  # per-family (smaller) version
        grid_miss = (L_ANA + R) * (math.pi / Nf)      # sup over the unsampled mid-arcs
        offset_valid = bool(gap_max + grid_miss <= offset_used)
        # control B: sampled legal target poses must never exceed the deterministic bound.
        # the centre direction psi and the pose angle phi are drawn INDEPENDENTLY, so the
        # sample really sweeps K_phi for every phi, not just the best-covered ray psi=phi.
        rng = np.random.default_rng(20261010)
        Ns = 200000
        phi = rng.uniform(0.0, math.pi / 2.0, Ns)          # pose angle, pi/2-periodic
        psi_p = rng.uniform(0.0, 2.0 * math.pi, Ns)        # centre direction
        rad = R * np.sqrt(rng.uniform(0.0, 1.0, Ns))
        px, py = rad * np.cos(psi_p), rad * np.sin(psi_p)
        c, s = np.cos(phi), np.sin(phi)
        cu = px * c + py * s
        cv = -px * s + py * c
        ok = (px * px + py * py + 0.5 + np.abs(cu) + np.abs(cv)) <= R * R
        px, py, phi = px[ok], py[ok], phi[ok]
        ks = (np.round(((phi + math.pi / 4.0) % (math.pi / 2.0) - math.pi / 4.0)
                       / dth).astype(int)) % n_ang
        dd = np.empty(len(px))
        for k in range(n_ang):
            m = (ks == k)
            if not m.any():
                continue
            dd[m] = cKDTree(Vcache[float(angs[k])]).query(np.column_stack([px[m], py[m]]))[0]
        aa = np.abs((phi - angs[ks] + math.pi / 4.0) % (math.pi / 2.0) - math.pi / 4.0)
        samp = float(np.max(np.maximum(dd, (SQ2 / 2.0) * aa)))
        samp_c = float(np.max(dd))
        jworst = int(np.argmax(np.maximum(dd, (SQ2 / 2.0) * aa)))

        # control C: the sampled points lie in P_k by construction, so any positive excess
        # against the halfplanes of Poly(k) falsifies the outer approximation (slack too small).
        max_excess = -np.inf
        for k in range(n_ang):
            m = (ks == k)
            if not m.any():
                continue
            um, hm, _, _ = sector_polygon(float(angs[k]), w)
            Q = np.column_stack([px[m], py[m]])
            for i0 in range(0, len(Q), 2000):
                max_excess = float(max(max_excess, np.max(Q[i0:i0 + 2000] @ um.T - hm)))

        cent = max(r['delta_centre'] for r in rows)
        worst = max(rows, key=lambda r: r['delta_centre'])
        ctrlA, _, _ = delta_centre(allnodes, angs[0], w)
        n_family = int(sum(r['n_legal'] for r in rows))
        # outer-approximation validity is asserted by control C (random poses) and control D
        # (deterministic support-function gap), not by this printed comparison.
        rho_max_analytic = (math.sqrt(99.0) - 1.0) / 2.0
        out.append(dict(tag=nm, h_c=h_c, n_ang=n_ang, dtheta=dth, angle_term=ang_term,
                        d0_ideal=d0, legal_node_total=n_family,
                        family_poses=len(F), node_count_matches_family=bool(n_family == len(F)),
                        cells_clipped_total=cut_tot,
                        lip_bound_used=L_ANA,
                        lip_analytic=rho_max_analytic / (2.0 * rho_max_analytic + 1.0),
                        lip_empirical=sl_max, rho_min_region=rho_min_all,
                        offset_used=offset_used, offset_tight=offset_tight,
                        gap_max=gap_max, grid_miss=grid_miss,
                        offset_valid=offset_valid,
                        control_full_lattice=ctrlA,
                        outer_polygon_max_excess=max_excess,
                        outer_polygon_contains_samples=bool(max_excess <= 0.0),
                        delta_centre_exact=cent, delta_pose_additive=cent + ang_term,
                        delta_pose_max=max(cent, ang_term),
                        sampled_delta_max=samp, sampled_centre_max=samp_c,
                        sampled_leq_det=bool(samp <= cent + 1e-9),
                        sampled_worst=dict(p=[float(px[jworst]), float(py[jworst])],
                                           phi=float(phi[jworst]), k=int(ks[jworst]),
                                           centre_err=samp_c, ang_err=float((SQ2/2.0)*aa[jworst])),
                        old_two_case_centre=0.606308 if n_ang == 9 else 0.464887,
                        improvement_vs_old_bound=(0.606308 if n_ang == 9 else 0.464887) / cent,
                        worst_angle=worst['theta'], worst_site=worst['site'],
                        worst_point=worst['point'], per_angle=rows))
        r = out[-1]
        print('[S1] %s: delta_centre=%.6f  angle=%.6f  => additive %.6f / max %.6f'
              % (nm, r['delta_centre_exact'], ang_term, r['delta_pose_additive'],
                 r['delta_pose_max']), flush=True)
        print('[S1]      offset %.6f (L=%.4f vs analytic %.4f / empirical %.4f) | det. gap '
              '%.6f + grid %.6f <= offset? %s | clipped cells %d/%d'
              % (offset_used, L_ANA, r['lip_analytic'], sl_max, gap_max, grid_miss,
                 offset_valid, cut_tot, n_family), flush=True)
        print('[S1]      outer polygon: max halfplane excess over %d sampled poses '
              '= %.3e -> contains? %s' % (len(px), max_excess,
                                          r['outer_polygon_contains_samples']), flush=True)
        print('[S1]      old two-case %.6f => improvement %.2fx | full-lattice control '
              '%.6f (ideal %.6f) | sampled centre %.6f vs det %.6f -> ok? %s | combined '
              '%.6f ok? %s | node count==|F|? %s (%d/%d)'
              % (r['old_two_case_centre'], r['improvement_vs_old_bound'], ctrlA, d0,
                 samp_c, cent, samp_c <= cent + 1e-9, samp, r['sampled_leq_det'],
                 r['node_count_matches_family'], n_family, len(F)), flush=True)

    budget = 0.0037172
    for r in out:
        r['budget'] = budget
        r['blocked_factor'] = r['delta_pose_additive'] / budget
        r['area_ratio'] = (1.0 + 2.0 * r['delta_pose_additive']) ** 2
        r['meets_budget'] = bool(r['delta_pose_additive'] <= budget)
        r['soundness'] = dict(
            node_count_eq_family=r['node_count_matches_family'],
            offset_certified=r['offset_valid'],
            polygon_contains_sampled_poses=r['outer_polygon_contains_samples'],
            sampled_leq_deterministic=r['sampled_leq_det'],
            lip_empirical_leq_used=bool(r['lip_empirical'] <= L_ANA),
            full_lattice_control_in_range=bool(r['d0_ideal'] - 1e-9 <= r['control_full_lattice']
                                              <= r['d0_ideal'] + 2.0 * OFFSET + 1e-9))
        r['soundness_all'] = all(r['soundness'].values())
        r['verdict'] = ('bound is a valid deterministic certificate' if r['soundness_all']
                        else 'SOUNDNESS CONTROL FAILED - number unusable')
        r['upgrade_breaks_gate'] = bool(r['soundness_all'] and r['meets_budget'])
        r['cert_value_if_upgraded'] = r['area_ratio'] * (69.649006 if r['n_ang'] == 13
                                                         else 67.0001319)
        print('[S1] %s: soundness %s %s | budget %.6f vs exact %.6f => %.1fx coarse; '
              'area ratio %.4f; upgraded wide U would be %.3f (gate 68)' % (
                  r['tag'], r['soundness_all'], r['verdict'], budget,
                  r['delta_pose_additive'], r['blocked_factor'], r['area_ratio'],
                  r['cert_value_if_upgraded']), flush=True)
        if not r['soundness_all']:
            print('[S1]      failed gates: %s' % [k for k, v in r['soundness'].items()
                                                  if not v], flush=True)
    json.dump(out, open('g2_route1_sector.json', 'w'), indent=1, default=float)
    print('[S1] wrote g2_route1_sector.json', flush=True)


if __name__ == '__main__':
    main()

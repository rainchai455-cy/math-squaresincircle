"""Step 3 of the user's plan: does the candidate family actually cover the continuous
pose space it claims to reason about?

Two separate questions, and they must not be conflated:
  (a) F subset of the continuum family  -- soundness of the certificate's SCOPE
      (an illegal pose inside F would make the LP a different problem entirely).
  (b) continuum family subset of a neighbourhood of F -- coverage / refinement
      (how far F is from the full 3-D pose space, i.e. how much mass the family
      simply cannot see).
Also computes the legal pose-space volume, so the sampled fraction is a number.
"""
import json, math, time
import numpy as np
from scipy.spatial import cKDTree
import g2_tau as T

R = T.R
CLEAR = 1e-9


def r2max_arr(x, y, th):
    """Vectorised vertex criterion: a unit square is legal iff r2max <= R^2."""
    ct, st = np.cos(th), np.sin(th)
    return x * x + y * y + 0.5 + np.abs(x * ct + y * st) + np.abs(-x * st + y * ct)


def pose_gap_audit(F, name, n_samples=200000, seed=7):
    """Sample continuum legal poses (rho,phi,theta) by rejection; measure the distance
    from each to the nearest pose of F in a metric that bounds the geometry:
        d = max( |centre difference| , (sqrt(2)/2) * |angle difference| )
    because a unit square rotated by a moves every point by <= (sqrt2/2)*a."""
    rng = np.random.default_rng(seed)
    cx, cy = F[:, 0], F[:, 1]
    th = np.mod(F[:, 2], math.pi / 2.0)
    tree = cKDTree(np.column_stack([cx, cy, th * math.sqrt(2.0) / 2.0]))
    got_x, got_y, got_t, n_drawn = [], [], [], 0
    while sum(len(a) for a in got_x) < n_samples:
        b = 5.0
        x = rng.uniform(-b, b, 50000)
        y = rng.uniform(-b, b, 50000)
        t = rng.uniform(0.0, math.pi / 2.0, 50000)
        n_drawn += len(x)
        m = r2max_arr(x, y, t) <= R * R
        got_x.append(x[m]); got_y.append(y[m]); got_t.append(t[m])
    X = np.concatenate(got_x)[:n_samples]
    Y = np.concatenate(got_y)[:n_samples]
    TH = np.mod(np.concatenate(got_t)[:n_samples], math.pi / 2.0)
    Q = np.column_stack([X, Y, TH * math.sqrt(2.0) / 2.0])
    d, i = tree.query(Q, k=1)
    dc = np.hypot(X - cx[i], Y - cy[i])
    da = np.abs(TH - th[i])
    da = np.minimum(da, math.pi / 2.0 - da)
    return dict(
        name=name, n_family=int(len(F)), n_legal_samples=int(len(X)),
        n_draws_before_rejection=int(n_drawn),
        pose_gap_max=float(d.max()), pose_gap_p99=float(np.quantile(d, 0.99)),
        pose_gap_median=float(np.median(d)),
        centre_gap_max=float(dc.max()), angle_gap_max=float(da.max()),
        frac_gap_gt_0p30=float((d > 0.30).mean()),
        frac_gap_gt_0p45=float((d > 0.45).mean()),
    )


def illegal_poses_in_family(F, name):
    """(a) soundness of scope: F must be a SUBSET of the continuum family, i.e. every pose
    legal in the CLOSED sense r2max <= R^2.  Poses with r2max == 25 exactly are the rim-pinned
    ones -- legal, and the certified 67 contributes 8 of them (proved in Fractions elsewhere),
    so a 'strict with clearance' test here would be the wrong criterion."""
    r2 = r2max_arr(F[:, 0], F[:, 1], F[:, 2])
    worst = float(r2.max())
    return dict(name=name, n=int(len(F)), limit=R * R, max_r2max=worst,
                poses_beyond_rim_beyond_float_noise=int((r2 > R * R + 1e-9).sum()),
                poses_exactly_on_rim=int(np.sum(np.abs(r2 - R * R) <= 1e-9)),
                every_pose_legal=bool(worst <= R * R + 1e-9))


def pose_space_volume(h=0.05, n_ang=91):
    """Vol = int_0^{pi/2} Area({centres legal at theta}) dtheta  (trapezoid on a fine grid)."""
    angs = np.linspace(0.0, math.pi / 2.0, n_ang)
    areas = [T.legal_area(t, h=h) for t in angs]
    vol = float(np.trapezoid(areas, angs))
    return dict(h_centre=h, n_ang=n_ang, volume_pose_space=vol,
                area_at_theta0=float(areas[0]), area_at_theta45=float(areas[n_ang // 2]),
                min_area=float(min(areas)), max_area=float(max(areas)))


def sym_diff_area(F, tree, n_poses=4000, pts_per=3000, seed=11):
    """Monte-Carlo estimate of |S \\\\ S'| for sampled legal continuum poses S and their
    nearest family pose S'.  This is the geometric size of what a delta-net family 'misses';
    diagnostic only, it is not turned into a lemma here."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_poses):
        while True:
            x, y = rng.uniform(-5, 5, 2)
            t = rng.uniform(0.0, math.pi / 2.0)
            if r2max_arr(x, y, t) <= R * R:
                break
        d, i = tree.query([x, y, t * math.sqrt(2.0) / 2.0])
        X, Y, TH = F[i]
        # sample points in S (rotated frame), test membership in S' (its rotated frame)
        a = rng.uniform(-0.5, 0.5, pts_per)
        b = rng.uniform(-0.5, 0.5, pts_per)
        px = x + a * math.cos(t) - b * math.sin(t)
        py = y + a * math.sin(t) + b * math.cos(t)
        u = (px - X) * math.cos(TH) + (py - Y) * math.sin(TH)
        v = -(px - X) * math.sin(TH) + (py - Y) * math.cos(TH)
        out.append(float(np.mean(np.maximum(np.abs(u), np.abs(v)) > 0.5)))
    out = np.array(out)
    # both squares have area 1, so |S \\ S'| = |S' \\ S| and |S delta S'| = 2 * mean
    return dict(symmetric_diff_area_mean=float(out.mean() * 2.0),
                symmetric_diff_area_max=float(out.max() * 2.0),
                n_poses_sampled=int(len(out)))


def main():
    t0 = time.time()
    d = json.load(open('g3_frozen/best_legal_67.json'))
    C67 = np.column_stack([d['x'], d['y'], d['th']]).tolist()
    out = []
    vol = pose_space_volume()
    print(f'[COV] legal pose-space volume (theta in [0,pi/2), centres) = '
          f'{vol["volume_pose_space"]:.2f}; area(centre region) at theta=0 '
          f'{vol["area_at_theta0"]:.2f}, at 45deg {vol["area_at_theta45"]:.2f}', flush=True)
    for h_c, na in ((0.30, 9), (0.20, 13)):
        F = T.build_family(h_c, na, extra=C67)
        name = f'fam({h_c:.2f},ang{na})'
        il = illegal_poses_in_family(F, name)
        gap = pose_gap_audit(F, name)
        tree = cKDTree(np.column_stack([F[:, 0], F[:, 1],
                                        np.mod(F[:, 2], math.pi / 2.0) * math.sqrt(2.0) / 2.0]))
        sd = sym_diff_area(F, tree)
        cell = h_c * h_c * ((math.pi / 2.0) / na)
        gap['family_cell_volume_per_pose'] = cell
        gap['volume_tiled_by_cells'] = cell * len(F)
        gap['pose_volume'] = vol['volume_pose_space']
        gap['is_a_lattice_net_not_a_sparse_subset'] = bool(cell * len(F) >= 0.9 * vol['volume_pose_space'])
        gap['angle_covering_radius_exact'] = math.pi / (4.0 * na)
        gap.update(sd)
        rec = dict(volume=vol, legality=il, coverage=gap)
        out.append(rec)
        print(f'[COV] {name}: |F|={len(F)} every_pose_legal={il["every_pose_legal"]} '
              f'(beyond-rim={il["poses_beyond_rim_beyond_float_noise"]}, '
              f'exactly-on-rim={il["poses_exactly_on_rim"]}) | pose-gap delta_max='
              f'{gap["pose_gap_max"]:.4f} p99={gap["pose_gap_p99"]:.4f} '
              f'med={gap["pose_gap_median"]:.4f} (angle radius {gap["angle_covering_radius_exact"]:.4f}) '
              f'| cells tile {100*gap["volume_tiled_by_cells"]/vol["volume_pose_space"]:.0f}% of pose volume '
              f'| sym-diff area mean={sd["symmetric_diff_area_mean"]:.4f} max={sd["symmetric_diff_area_max"]:.4f} '
              f'({time.time()-t0:.0f}s)', flush=True)
        assert il['every_pose_legal'] and il['poses_beyond_rim_beyond_float_noise'] == 0
    json.dump(out, open('g2_coverage_audit.json', 'w'), indent=1, default=float)
    print('[COV] SCOPE LEMMA: V_c(F) <= V* for every subfamily F of the continuum family, so '
          'a family-restricted dual upper bound NEVER bounds the global measure route; the '
          'families above are lattice nets (delta_max ~0.32 pose metric), so upgrading needs an '
          'explicit pose-envelope relaxation -- without it every closure stays a '
          'finite-candidate-set conclusion, never tau(5)=67 and never r(67)>=5.', flush=True)


if __name__ == '__main__':
    main()

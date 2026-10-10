"""Grid-relaxation diagnostic for the G2 measure LP.

The density constraint D(p)<=1 is a CONTINUUM constraint.  Enforcing it only on a point
grid is a relaxation, so the LP value must DECREASE toward the true continuous LP optimum
as the grid refines.  Any value above the rigorous ceiling Area(disc)=78.5398 is
self-evidently cheating (no measure with D<=1 a.e. can carry more mass than the area).

Each solution is therefore also audited on a STAGGERED grid (offset h/2), which is exactly
where the relaxed LP hides its excess density.
"""
import time, json, math
import numpy as np
import g2_tau as T


def staggered(h):
    n = int(math.floor(2 * T.R / h)) + 1
    g = np.linspace(-T.R + h / 2, T.R + h / 2, n)
    GX, GY = np.meshgrid(g, g, indexing='ij')
    p = np.column_stack([GX.ravel(), GY.ravel()])
    return p[np.einsum('ij,ij->i', p, p) <= T.R * T.R]


def run(F, h_g, tag, out):
    t0 = time.time()
    pts = T.grid_pts(h_g)
    r = T.run_lp(F, pts, tag, keep_w=True)
    if not r:
        return
    d1, _ = T.density_max(F, r.pop('w'), staggered(h_g), corners=False)
    r['staggered_max_density'] = float(d1)
    r['conservative_cert'] = r['lp_value'] / max(1.0, d1)
    r['above_area_ceiling'] = bool(r['lp_value'] > T.AREA_DISC)
    out.append(r)
    print(f'[CONV] {tag}: rows={r["rows"]} LP={r["lp_value"]:.4f} '
          f'staggered_maxD={d1:.4f} conservative_cert={r["conservative_cert"]:.4f} '
          f'{"**ABOVE AREA CEILING = CHEATING**" if r["above_area_ceiling"] else ""} '
          f'({time.time()-t0:.1f}s)', flush=True)


d = json.load(open('g3_frozen/best_legal_67.json'))
cert = np.column_stack([d['x'], d['y'], d['th']])
C67 = cert.tolist()
out = []
print(f'[CONV] rigorous ceiling on any measure certificate: Area(disc) = '
      f'{T.AREA_DISC:.4f}; target tau>=68 is {68/T.AREA_DISC:.1%} of it', flush=True)

F1 = T.build_family(0.30, 9, extra=C67)
for h_g in (0.30, 0.15, 0.075, 0.04):
    run(F1, h_g, f'fam(h_c=0.30,ang=9,n={len(F1)}) h_g={h_g}', out)

F2 = T.build_family(0.20, 13, extra=C67)
for h_g in (0.15, 0.075):
    run(F2, h_g, f'fam(h_c=0.20,ang=13,n={len(F2)}) h_g={h_g}', out)

ok = [r for r in out if not r['above_area_ceiling']]
best = max([r['conservative_cert'] for r in ok] + [67.0])
seq = ' '.join(f"{r['lp_value']:.3f}" for r in out)
print(f'[CONV] LP values along the grid-refinement order = {seq}', flush=True)
print(f'[CONV] best conservative tau certificate from the measure LP = {best:.4f} '
      f'(standing rigorous value = 67, from nu>=67)', flush=True)
print(f'[CONV] GATE tau>=68 -> {"PASSED" if best >= 68 else "NOT PASSED"}; '
      f'criterion unchanged from the pre-registered 68', flush=True)
json.dump(out, open('g2_tau_conv.json', 'w'), indent=1)

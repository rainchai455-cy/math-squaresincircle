import time, numpy as np, json, math
import g2_tau as T

t0 = time.time()
areas = [T.legal_area(x) for x in np.linspace(0, math.pi / 2, 8, endpoint=False)]
print('PC1 areas', [round(a, 3) for a in areas], 'mean', round(float(np.mean(areas)), 4),
      f'({time.time()-t0:.1f}s)', flush=True)

d = json.load(open('g3_frozen/best_legal_67.json'))
cert = np.column_stack([d['x'], d['y'], d['th']])
print('th unique', np.unique(cert[:, 2]), 'coord min/max', cert[:, :2].min(), cert[:, :2].max())
bad = [k for k, (x, y, th) in enumerate(cert) if not T.legal(x, y, th)]
print('illegal by vertex criterion:', len(bad))
t1 = time.time()
dm, _ = T.density_max(cert, np.ones(len(cert)), T.grid_pts(0.25))
print('PC2 sampled maxD', dm, f'({time.time()-t1:.1f}s)', flush=True)

t1 = time.time()
F = T.build_family(0.30, 9, extra=cert.tolist())
print('family size', len(F), f'({time.time()-t1:.1f}s)', flush=True)
t1 = time.time()
r = T.run_lp(F, T.grid_pts(0.30), 'smoke h=0.30 ang=9')
print(r, f'({time.time()-t1:.1f}s)', flush=True)

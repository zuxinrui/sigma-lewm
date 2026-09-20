"""非 free（剔除开局即成功）成功率：Push-T 与 Cube，逐 episode 复算。
Push-T free 判据 = env.eval_state 原式：||goal[:4]-s[:4]||<20 且 角差<pi/9，goal=start+offset 的 state。
Cube  free 判据 = ||cube(t+off)-cube(t)|| <= 4cm。"""
from repro_paths import resolve_path as _resolve_artifact
import re, glob, os, h5py, numpy as np, collections, json

R_IDX = re.compile(r"valid starting points found for evaluation\.(.*?)CEM solve time", re.S)
R_ARR = re.compile(r"\[([\s\d]+)\]", re.S)
R_ES  = re.compile(r"'episode_successes': array\((\[.*?\])\)", re.S)

def parse(fn):
    t = open(fn, 'rb').read().decode('utf8', 'ignore')
    m = R_IDX.search(t); a = R_ARR.search(m.group(1)) if m else None
    e = R_ES.search(t)
    if not (a and e): return None, None
    idx = np.array([int(x) for x in a.group(1).split()])
    es = np.array([x == 'True' for x in re.findall(r"True|False", e.group(1))])
    return (idx, es) if len(idx) == len(es) else (None, None)

D = _resolve_artifact('stable:datasets')
with h5py.File(f'{D}/cube_single_expert.h5', 'r') as f:
    CUBE = f['privileged_block_0_pos'][:]
with h5py.File(f'{D}/pusht_expert_train.h5', 'r') as f:
    PT = f['state'][:]

def free_cube(idx, off):
    return np.linalg.norm(CUBE[idx + off] - CUBE[idx], axis=1) <= 0.04

def free_pusht(idx, off):
    s, g = PT[idx], PT[idx + off]
    pos = np.linalg.norm(g[:, :4] - s[:, :4], axis=1)
    ang = np.abs(g[:, 4] - s[:, 4]); ang = np.minimum(ang, 2 * np.pi - ang)
    return (pos < 20) & (ang < np.pi / 9)

FREE = {'pusht': free_pusht, 'cube': free_cube}
agg = collections.defaultdict(lambda: [0, 0, 0, 0])   # key -> [free, freeS, nonfree, nonfreeS]

def add(task, arm, axis, x, idx, es, off):
    fr = FREE[task](idx, off)
    a = agg[(task, arm, axis, x)]
    a[0] += int(fr.sum()); a[1] += int(es[fr].sum())
    a[2] += int((~fr).sum()); a[3] += int(es[~fr].sum())

# --- CEM 阶梯 (h=5, offset 25) 与 horizon 轴 (offset 5h) 的世界模型臂
for fn in sorted(glob.glob('logs/t4_cem_ladder_20260908/*.log')):
    m = re.match(r'(pusht|cube)_(lewm|ours)_c(\d+)_s(\d)\.log$', os.path.basename(fn))
    if not m: continue
    idx, es = parse(fn)
    if idx is None: continue
    add(m[1], m[2], 'cem', int(m[3]), idx, es, 25)
for fn in sorted(glob.glob('logs/t6_horizon_20260908/*.log')):
    m = re.match(r'(pusht|cube)_(lewm|ours)_h(\d+)_s(\d)\.log$', os.path.basename(fn))
    if not m: continue
    idx, es = parse(fn)
    if idx is None: continue
    add(m[1], m[2], 'h', int(m[3]), idx, es, 5 * int(m[3]))
# --- RC-DB-DCL 与其配对的 stock CEM
for fn in sorted(glob.glob('logs/rcdb_axes_20260909/*.log')):
    m = re.match(r'(pusht|cube)_(cem|safe_dcl_portfolio)_s(\d)_cem(\d+)_h(\d+)_', os.path.basename(fn))
    if not m: continue
    idx, es = parse(fn)
    if idx is None: continue
    arm = 'rcdb' if m[2] == 'safe_dcl_portfolio' else 'stock'
    c, h = int(m[4]), int(m[5])
    if h == 5: add(m[1], arm, 'cem', c, idx, es, 25)
    if c == 30: add(m[1], arm, 'h', h, idx, es, 5 * h)

rows = []
print(f"{'task':6s} {'axis':4s} {'x':>3s} {'arm':6s} {'n':>4s} {'free':>5s} {'raw%':>6s} {'非free%':>8s}")
for (task, arm, axis, x), (F, FS, NF, NFS) in sorted(agg.items()):
    n = F + NF
    raw, nf = 100 * (FS + NFS) / n, 100 * NFS / NF if NF else float('nan')
    print(f"{task:6s} {axis:4s} {x:3d} {arm:6s} {n:4d} {F:5d} {raw:6.1f} {nf:8.1f}")
    rows.append(dict(task=task, arm=arm, axis=axis, x=x, n=n, free=F, free_succ=FS,
                     nonfree=NF, nonfree_succ=NFS, raw_pct=round(raw, 2), nonfree_pct=round(nf, 2)))
json.dump(rows, open('logs/nonfree_20260910.json', 'w'), indent=1)
print("\n-> logs/nonfree_20260910.json")

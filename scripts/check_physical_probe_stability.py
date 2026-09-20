#!/usr/bin/env python3
"""Secondary five-fold episode-grouped robustness check; not independent data."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import sys,json
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.rank_physical_probes import candidates,score,ROOT,OUT,ALPHAS


def main():
    results=[]
    for task in ['cube','gear']:
        source=ROOT/'outputs/cube_gear_physical_atlas_v1'/task
        sample=dict(np.load(source/'sample.npz'));episodes=sample['episodes']
        specs=candidates(task,sample)
        ys=np.concatenate([s['y'] for s in specs],1);offsets=np.cumsum([0]+[s['y'].shape[1] for s in specs])
        embs={m:np.load(source/f'emb_{m}.npy').astype(float) for m in ['lewm','ours']}
        for fold,(development,test) in enumerate(GroupKFold(n_splits=5,shuffle=True,random_state=20260910).split(ys,groups=episodes)):
            groups=np.random.default_rng(20260911+fold).permutation(np.unique(episodes[development]))
            pivot=int(.75*len(groups))
            train=np.flatnonzero(np.isin(episodes,groups[:pivot]));val=np.flatnonzero(np.isin(episodes,groups[pivot:]))
            assert not set(episodes[train])&set(episodes[test])
            yt=StandardScaler().fit(ys[train]);ytrain=yt.transform(ys[train]);paired={}
            for model,z in embs.items():
                x=StandardScaler().fit(z[train]).transform(z)
                pv,pt=[],[]
                for alpha in ALPHAS:
                    probe=Ridge(alpha=alpha).fit(x[train],ytrain)
                    pv.append(yt.inverse_transform(probe.predict(x[val])))
                    pt.append(yt.inverse_transform(probe.predict(x[test])))
                for j,s in enumerate(specs):
                    sl=slice(offsets[j],offsets[j+1])
                    alpha=int(np.nanargmax([score(s['y'][val],p[:,sl],s['kind']) for p in pv]))
                    paired[model,s['field']]=score(s['y'][test],pt[alpha][:,sl],s['kind'])
            for s in specs:
                a,b=paired['lewm',s['field']],paired['ours',s['field']]
                results.append(dict(task=task,fold=fold,field=s['field'],label=s['label'],kind=s['kind'],lewm=a,ours=b,gain=b-a))
            print(task,'fold',fold,'complete',flush=True)
    summary=[]
    for task in ['cube','gear']:
        for field in dict.fromkeys(r['field'] for r in results if r['task']==task):
            rs=[r for r in results if r['task']==task and r['field']==field]
            summary.append(dict(task=task,field=field,label=rs[0]['label'],kind=rs[0]['kind'],
                                mean_lewm=float(np.mean([r['lewm'] for r in rs])),mean_ours=float(np.mean([r['ours'] for r in rs])),
                                gains=[r['gain'] for r in rs],wins=sum(r['gain']>0 for r in rs)))
    (OUT/'five_fold_stability.json').write_text(json.dumps(dict(caveat='Secondary resplitting of already inspected data; not independent confirmation or five independent training seeds',folds=results,summary=summary),indent=2)+'\n')
    report=ROOT/'outputs/PHYSICAL_PROBE_RANKING.md'
    marker='## Secondary five-fold episode robustness'
    content=report.read_text().split(marker)[0].rstrip()
    lines=['', '', marker, '',
           'All candidate probes are repeated with five episode-disjoint test folds. Within each development fold, a separate grouped validation split selects alpha. This reuses the inspected atlas: it is a stability check, not five independent model seeds or independent confirmation.', '',
           '| Task | Quantity | Metric family | Mean LeWM | Mean ours | Positive-gain folds |',
           '|---|---|---|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['task']} | {r['label']} | {r['kind']} | {r['mean_lewm']:.3f} | {r['mean_ours']:.3f} | {r['wins']}/5 |")
    # Report all three relative axes together, not just the largest-gap y axis.
    saved=dict(np.load(OUT/'cube/test_predictions.npz'))
    errors={m:np.linalg.norm(np.column_stack([
        (saved[f'effector_minus_block_{axis}_{m}']-saved[f'effector_minus_block_{axis}_truth'])[:,0]
        for axis in 'xyz']),axis=1) for m in ['lewm','ours']}
    _,inverse=np.unique(saved['episodes'],return_inverse=True)
    counts=np.bincount(inverse);rng=np.random.default_rng(20260912)
    weights=rng.multinomial(len(counts),np.full(len(counts),1/len(counts)),1000)
    improvements=np.bincount(inverse,weights=errors['lewm']-errors['ours'])
    ci=np.quantile((weights@improvements)/(weights@counts),[.025,.975])
    aggregate=dict(metric='Mean Euclidean error of decoded effector-minus-block xyz (mm)',
                   lewm=float(errors['lewm'].mean()),ours=float(errors['ours'].mean()),
                   improvement_ci=ci.tolist(),note='Post-hoc physical aggregate of all three axes; primary grouped probe-test predictions, pointwise episode-bootstrap interval')
    (OUT/'cube_relative_xyz_summary.json').write_text(json.dumps(aggregate,indent=2)+'\n')
    lines.extend(['', '## Recommended emphasis (exploratory, not a new confirmatory endpoint)', '',
                  f"Cube all-axis summary: mean 3D relative-position error **{aggregate['lewm']:.3f} mm vs {aggregate['ours']:.3f} mm**, pointwise 95% interval for error reduction [{ci[0]:.3f}, {ci[1]:.3f}] mm. This explicitly includes x, y and z; it is a post-hoc aggregate, not a pre-registered endpoint.", '',
                  '- Cube: foreground effector-minus-block position recovery, particularly y; report x and z as well to avoid presenting a hand-picked axis alone. Gripper opening is a stable complementary readout. Future effector/block displacement is also stronger, but cannot be interpreted as causal action-conditioned dynamics from a single current-frame probe.',
                  '- Gear: foreground average precision for the instantaneous strict geometry predicate (primary probe-test AP 0.227 vs 0.871; prevalence 0.143; AUROC 0.606 vs 0.971). This is not persistent success or policy rollout success. Pair it with millimetre MAE for relative x/y, lateral error and seating height.',
                  '- Gear height R2 is sensitive to rare outlier episodes: the bootstrap interval and fold results are broad. The original probe-test MAE is 5.746 vs 2.900 mm; report this alongside R2, rather than relying on a flattering large R2 gap.',
                  '- Do not foreground near-ceiling Cube absolute positions, Cube target-relative y (small/uncertain primary-test gain and near-zero absolute R2), either task\'s poorly decoded rotation dynamics, or raw effective rank. These are retained in the full table.',
                  '- The strongest recommended relative-position and strict-predicate gains have the same sign in all five folds. Freeze these hypotheses and verify on a fresh independent episode bank before treating the selected advantages as paper-confirmatory results.', '',
                  'Stability reproduction: `.venv/bin/python scripts/check_physical_probe_stability.py`.'])
    report.write_text(content+'\n'.join(lines)+'\n')
    print('Stability saved',flush=True)


if __name__=='__main__':
    from scripts.archive_embedding_round import archive_round
    with archive_round('physical_probe_stability'):
        main()

#!/usr/bin/env python3
"""Exploratory, episode-grouped physical recoverability audit of frozen latents.

No model training, simulator or GPU: only small ridge readouts. Select alpha
and six display variables on validation; report untouched probe-test episodes.
The source atlas has already been inspected, so this is not a sealed paper test.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,json,csv,hashlib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score,roc_auc_score,balanced_accuracy_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.cube_gear_physical_atlas import CUBE_GROUPS,GEAR_GROUPS

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/physical_probe_ranking_v1'
REPORT=ROOT/'outputs/PHYSICAL_PROBE_RANKING.md'
ALPHAS=[.1,1.,10.,100.,1000.]
SEED=20260909


def grouped_split(episodes):
    unique=np.unique(episodes); perm=np.random.default_rng(SEED).permutation(unique)
    a,b=int(.6*len(perm)),int(.8*len(perm))
    groups=dict(train=perm[:a],validation=perm[a:b],test=perm[b:])
    assert not set(groups['train'])&set(groups['validation'])
    assert not set(groups['train'])&set(groups['test'])
    return {k:np.flatnonzero(np.isin(episodes,g)) for k,g in groups.items()},groups


def r2(y,p):
    denominator=np.square(y-y.mean(0)).sum()
    return float(1-np.square(y-p).sum()/denominator) if denominator>1e-15 else float('nan')


def score(y,p,kind):
    if kind=='binary':return float(average_precision_score(y[:,0],p[:,0]))
    if kind=='category':return float(balanced_accuracy_score(y.argmax(1),p.argmax(1)))
    return r2(y,p)


def bootstrap_delta(y,a,b,episodes,kind,repeats=1000):
    unique,inverse=np.unique(episodes,return_inverse=True)
    rng=np.random.default_rng(SEED+1)
    if kind not in ('binary','category'):
        # Cluster bootstrap via sufficient statistics, preserving paired errors.
        counts=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),repeats)
        n=np.bincount(inverse)
        sums=np.stack([np.bincount(inverse,weights=y[:,j]) for j in range(y.shape[1])],1)
        squares=np.bincount(inverse,weights=np.square(y).sum(1))
        delta_sse=np.bincount(inverse,weights=(np.square(y-a)-np.square(y-b)).sum(1))
        denominator=counts@squares-np.square(counts@sums).sum(1)/(counts@n)
        values=(counts@delta_sse)/np.maximum(denominator,1e-15)
    else:
        buckets=[np.flatnonzero(inverse==j) for j in range(len(unique))]
        values=[]
        for _ in range(repeats):
            idx=np.concatenate([buckets[j] for j in rng.integers(0,len(unique),len(unique))])
            if kind=='binary' and len(np.unique(y[idx]))<2:continue
            values.append(score(y[idx],b[idx],kind)-score(y[idx],a[idx],kind))
    return [float(x) for x in np.quantile(values,[.025,.975])]


def candidates(task,sample):
    groups=CUBE_GROUPS if task=='cube' else GEAR_GROUPS
    result=[]
    for slug,_,variables in groups:
        for field,label,unit,_,kind in variables:
            values=sample[field].astype(float)
            if np.ptp(values)<1e-10:continue
            kind=kind if kind in ('angle','angle_error','binary','category') else 'continuous'
            if kind in ('angle','angle_error'):
                y=np.column_stack([np.cos(np.deg2rad(values)),np.sin(np.deg2rad(values))])
            elif kind=='category':y=np.eye(3)[values.astype(int)+1]
            else:y=values[:,None]
            result.append(dict(field=field,label=label,unit=unit,kind=kind,y=y,family=slug))
    if task=='cube':
        for axis in 'xyz':
            for prefix,label,base in [('goal_minus_block','Target - block','goal'),('effector_minus_block','Effector - block','control')]:
                result.append(dict(field=f'{prefix}_{axis}',label=f'{label} {axis}',unit='mm',kind='continuous',
                                   y=(sample[f'{base}_{axis}']-sample[f'block_{axis}'])[:,None].astype(float),family='relative_geometry'))
    else:
        for axis in 'xy':
            result.append(dict(field=f'control_minus_fixture_{axis}',label=f'Control target - fixture {axis}',unit='mm',
                               kind='continuous',y=(sample[f'control_{axis}']-sample[f'fixed_{axis}'])[:,None].astype(float),family='relative_geometry'))
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    all_rows=[]
    for task in ['cube','gear']:
        folder=OUT/task;folder.mkdir(exist_ok=True)
        source=ROOT/'outputs/cube_gear_physical_atlas_v1'/task
        sample=dict(np.load(source/'sample.npz'));episodes=sample['episodes']
        split,groups=grouped_split(episodes); specs=candidates(task,sample)
        protocol=dict(seed=SEED,alpha_grid=ALPHAS,probe='feature-standardized ridge; same capacity/grid for both models',
                      split_groups={k:v.tolist() for k,v in groups.items()},split_rows={k:v.tolist() for k,v in split.items()},
                      candidates=[{k:v for k,v in s.items() if k!='y'} for s in specs],
                      display_selection='six continuous physical quantities by descending validation R2 gain, restricted to positive ours validation R2',
                      limitations='Exploratory: atlas already inspected; probe-test groups are disjoint from probe fitting/selection, not a sealed new representation-training-disjoint bank.',
                      bootstrap='1000 paired episode resamples; pointwise 95% intervals, no multiplicity correction',
                      source_provenance_sha256=hashlib.sha256((source/'provenance.json').read_bytes()).hexdigest())
        # Written before fitting/looking at test outcomes; refuse silent split changes.
        protocol_file=folder/'protocol.json'
        if protocol_file.exists():assert json.loads(protocol_file.read_text())==protocol
        else:protocol_file.write_text(json.dumps(protocol,indent=2)+'\n')
        ys=np.concatenate([s['y'] for s in specs],1)
        offsets=np.cumsum([0]+[s['y'].shape[1] for s in specs])
        yt=StandardScaler().fit(ys[split['train']]); ytrain=yt.transform(ys[split['train']])
        predictions={}; rows=[]
        for model in ['lewm','ours']:
            z=np.load(source/f'emb_{model}.npy').astype(float)
            normalizer=StandardScaler().fit(z[split['train']]);x=normalizer.transform(z)
            val_preds=[];test_preds=[]
            for alpha in ALPHAS:
                probe=Ridge(alpha=alpha).fit(x[split['train']],ytrain)
                val_preds.append(yt.inverse_transform(probe.predict(x[split['validation']])))
                test_preds.append(yt.inverse_transform(probe.predict(x[split['test']])) )
            for j,s in enumerate(specs):
                sl=slice(offsets[j],offsets[j+1]);yv=s['y'][split['validation']]
                vals=[score(yv,p[:,sl],s['kind']) for p in val_preds]
                chosen=int(np.nanargmax(vals))
                predictions[(model,s['field'])]=dict(validation_score=vals[chosen],alpha=ALPHAS[chosen],
                                                    test=test_preds[chosen][:,sl],validation=val_preds[chosen][:,sl])
        # Select displayed variables using validation only; test values read afterwards.
        eligible=[s for s in specs if s['kind']=='continuous' and predictions['ours',s['field']]['validation_score']>0]
        selected=sorted(eligible,key=lambda s:predictions['ours',s['field']]['validation_score']-predictions['lewm',s['field']]['validation_score'],reverse=True)[:6]
        selected_fields=[s['field'] for s in selected]
        (folder/'validation_selected.json').write_text(json.dumps(selected_fields,indent=2)+'\n')
        saved_predictions={}
        for s in specs:
            field,kind=s['field'],s['kind'];y=s['y'][split['test']]
            a=predictions['lewm',field];b=predictions['ours',field]
            ci=bootstrap_delta(y,a['test'],b['test'],episodes[split['test']],kind)
            row=dict(task=task,field=field,label=s['label'],family=s['family'],kind=kind,
                     metric='AP' if kind=='binary' else 'balanced_accuracy' if kind=='category' else 'circular_R2' if kind in ('angle','angle_error') else 'R2',
                     selected=field in selected_fields,lewm_alpha=a['alpha'],ours_alpha=b['alpha'],
                     val_lewm=a['validation_score'],val_ours=b['validation_score'],val_gain=b['validation_score']-a['validation_score'],
                     test_lewm=score(y,a['test'],kind),test_ours=score(y,b['test'],kind),ci_low=ci[0],ci_high=ci[1],
                     test_rows=len(y),test_episodes=len(groups['test']),unit=s['unit'])
            row['test_gain']=row['test_ours']-row['test_lewm']
            if kind=='continuous':
                row.update(lewm_mae=float(np.abs(y-a['test']).mean()),ours_mae=float(np.abs(y-b['test']).mean()))
            elif kind in ('angle','angle_error'):
                for name,pred in [('lewm',a),('ours',b)]:
                    error=np.angle(np.exp(1j*(np.arctan2(pred['test'][:,1],pred['test'][:,0])-np.arctan2(y[:,1],y[:,0]))))
                    row[name+'_mae']=float(np.abs(np.rad2deg(error)).mean())
            elif kind=='binary':
                row.update(prevalence=float(y.mean()),lewm_auc=float(roc_auc_score(y[:,0],a['test'][:,0])),ours_auc=float(roc_auc_score(y[:,0],b['test'][:,0])))
            saved_predictions.update({field+'_truth':y,field+'_lewm':a['test'],field+'_ours':b['test']})
            rows.append(row)
        np.savez_compressed(folder/'test_predictions.npz',episodes=episodes[split['test']],rows=sample['rows'][split['test']],**saved_predictions)
        (folder/'results.json').write_text(json.dumps(rows,indent=2)+'\n');all_rows.extend(rows)
        print(task,'selected',[(r['field'],round(r['test_lewm'],3),round(r['test_ours'],3),[round(r[k],3) for k in ['ci_low','ci_high']]) for r in rows if r['selected']],flush=True)
    fields=list(dict.fromkeys(k for r in all_rows for k in r))
    with (OUT/'all_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(all_rows)
    lines=['# Physical recoverability: exploratory variable ranking','',
           'Frozen 192-D embeddings; no encoder training, simulator or GPU. Episode-grouped 60/20/20 probe split; train-only standardization. Same Ridge alpha grid for both models; alpha and six highlighted continuous quantities selected on validation, not probe-test outcomes. All candidates and non-wins retained below.','',
           'These are new probes on an already inspected atlas, not a sealed confirmatory experiment. Cube representation-training overlap is unknown. Future-motion/target labels may reflect trajectory correlations, not causal action information. Pointwise bootstrap intervals do not correct for the candidate search.','',
           'Continuous: R2; circular: joint sin/cos R2 with angle MAE; binary: average precision (AP), with prevalence and AUROC in raw results; three-way rotation: balanced accuracy. Different metric families must not share a numerical leaderboard.','']
    for task in ['cube','gear']:
        lines.extend([f'## {task.title()}: validation-selected highlights','',
                      '| Physical quantity | LeWM test R2 | Ours test R2 | Gain | Pointwise 95% interval | MAE LeWM / ours |','|---|---:|---:|---:|---|---|'])
        ranked=sorted([r for r in all_rows if r['task']==task],key=lambda r:r['val_gain'],reverse=True)
        for r in ranked:
            if r['selected']:lines.append(f"| {r['label']} | {r['test_lewm']:.3f} | {r['test_ours']:.3f} | {r['test_gain']:+.3f} | [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | {r['lewm_mae']:.3f} / {r['ours_mae']:.3f} {r['unit']} |")
        lines.extend(['',f'### All {task} candidates (grouped by metric, then validation gain)','',
                      '| Quantity | Metric | Validation gain | Test LeWM | Test ours | Test gain |','|---|---|---:|---:|---:|---:|'])
        for r in sorted(ranked,key=lambda r:(r['metric'],-r['val_gain'])):lines.append(f"| {r['label']} | {r['metric']} | {r['val_gain']:+.3f} | {r['test_lewm']:.3f} | {r['test_ours']:.3f} | {r['test_gain']:+.3f} |")
        lines.append('')
    lines.extend(['## Reproduction and sources','',
                  '` .venv/bin/python scripts/rank_physical_probes.py `','',
                  'Protocol/splits, validation selections, all scores, and paired predictions: `outputs/physical_probe_ranking_v1/`. These are analysis artifacts; existing paper results and figures are not overwritten.','',
                  'Grouped splitting rationale: [scikit-learn grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html). Metric definitions: [R2](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html), [average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).'])
    REPORT.write_text('\n'.join(lines)+'\n');print(REPORT,flush=True)


if __name__=='__main__':
    from scripts.archive_embedding_round import archive_round
    with archive_round('physical_probe_ranking'):
        main()

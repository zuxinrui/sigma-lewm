#!/usr/bin/env python3
"""CPU-only physical readout views, with explicit supervised/unsupervised controls.

Never select individual points or camera angles for a model-specific advantage.
Keep original PCA figures. Fit readouts only on episode-disjoint probe training
groups; use existing validation-chosen alpha, display every probe-test point.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
import sys, json, shutil
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import pusht_physical_atlas as atlas
from scripts.rank_physical_probes import grouped_split, r2, ALPHAS, candidates
sys.path.insert(0, str(atlas.STYLE))
from _common import apply_style, plt
from matplotlib import colors, cm

OUT = ROOT/'outputs/focused_physical_atlas_v1'
FIELDS = {
    'pusht': ['block_x','block_y','pusher_x','pusher_y','pusher_rel_x','pusher_rel_y'],
    'cube': ['effector_minus_block_x','effector_minus_block_y','effector_minus_block_z',
             'opening','control_motion','block_motion'],
    'gear': ['relative_x','relative_y','height','xy_error','control_minus_fixture_y','strict'],
}
LABELS = {
    'pusht': ['Block x','Block y','Pusher x','Pusher y','Pusher - block x','Pusher - block y'],
    'cube': ['Effector - block x','Effector - block y','Effector - block z','Gripper opening','Effector motion / 5','Block motion / 5'],
    'gear': ['Gear - fixture x','Gear - fixture y','Seating height','Lateral error','Control - fixture y','Strict geometry'],
}
SHORT_LABELS = {
    'pusht': ['Block x','Block y','Pusher x','Pusher y','Relative x','Relative y'],
    'cube': ['Relative x','Relative y','Relative z','Gripper opening','Effector motion','Block motion'],
    'gear': ['Relative x','Relative y','Seating height','Lateral error','Control offset y','Strict geometry'],
}

def message(task, mode):
    if task=='gear':
        return ('Insertion geometry is readable, but not in the leading PCs'
                if mode not in ('readout','paired') else 'A linear readout exposes strict insertion geometry')
    if task=='cube':return 'Relative object-hand geometry is more recoverable'
    return 'Pusher position is more recoverable; block position is mixed'

def source(task):
    return ROOT/('outputs/fig9_original_recovery/atlas' if task=='pusht' else f'outputs/cube_gear_physical_atlas_v1/{task}')

def prepare(task):
    src=source(task); sample=dict(np.load(src/'sample.npz'))
    split, groups=grouped_split(sample['episodes'])
    z={k:np.load(src/f'emb_{k}.npy').astype(float) for k in ['lewm','ours']}
    if task=='pusht':
        ys={k:sample[k][:,None] for k in FIELDS[task]}; prior={}
    else:
        ys={s['field']:s['y'] for s in candidates(task,sample)}
        prior={r['field']:r for r in json.loads((ROOT/f'outputs/physical_probe_ranking_v1/{task}/results.json').read_text())}
        saved=json.loads((ROOT/f'outputs/physical_probe_ranking_v1/{task}/protocol.json').read_text())
        assert saved['split_rows']=={k:v.tolist() for k,v in split.items()}
    output={}; metrics=[]; audits=[]
    for model in ['lewm','ours']:
        scaler=StandardScaler().fit(z[model][split['train']]); x=scaler.transform(z[model])
        pc=PCA(3,svd_solver='full').fit(z[model][split['train']])
        xyz=pc.transform(z[model])
        output[model]={'pca':xyz[split['test']], 'readout':[]}
        for field in FIELDS[task]:
            y=ys[field]; train, val, test=[split[k] for k in ['train','validation','test']]
            fits=[Ridge(alpha=a).fit(x[train],y[train]) for a in ALPHAS]
            alpha=prior[field][model+'_alpha'] if field in prior else ALPHAS[int(np.argmax([r2(y[val],p.predict(x[val])) for p in fits]))]
            fit=fits[ALPHAS.index(alpha)]; pred=fit.predict(x).reshape(-1)
            # A task-directed axis, NOT an unsupervised embedding: held-out
            # readout vs label-free PC1 residualized against that readout.
            nuisance=Ridge(alpha=1.).fit(pred[train,None],xyz[train,0])
            residual=xyz[:,0]-nuisance.predict(pred[:,None])
            output[model]['readout'].append(np.column_stack([pred[test],residual[test]]))
            truth=y[test,0]
            from sklearn.metrics import average_precision_score
            metric='AP' if field=='strict' else 'MAE'
            value=float(average_precision_score(truth,pred[test])) if metric=='AP' else float(np.abs(truth-pred[test]).mean())
            metrics.append(dict(task=task,model=model,field=field,metric=metric,value=value,alpha=alpha,
                                r2=None if metric=='AP' else r2(y[test],pred[test,None])))
            if field in prior:
                expected=prior[field]['test_'+model] if metric=='AP' else prior[field][model+'_mae']
                np.testing.assert_allclose(value,expected,rtol=1e-5,atol=1e-6)
            # Fixed 20-NN on unsupervised PCA, no camera/point search.
            for dim in [2,3]:
                knn=KNeighborsRegressor(20).fit(xyz[train,:dim],y[train])
                p=knn.predict(xyz[test,:dim])
                score=float(average_precision_score(truth,p[:,0])) if metric=='AP' else r2(y[test],p)
                audits.append(dict(model=model,field=field,dimensions=dim,metric='AP' if metric=='AP' else 'R2',score=score))
    folder=OUT/task; folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'coordinates.npz',rows=sample['rows'][split['test']],episodes=sample['episodes'][split['test']],
        **{f'{k}_pca':v['pca'] for k,v in output.items()},
        **{f'{k}_readout':np.stack(v['readout']) for k,v in output.items()},
        **{f'truth_{f}':ys[f][split['test'],0] for f in FIELDS[task]})
    (folder/'audit.json').write_text(json.dumps(dict(fields=FIELDS[task],split_rows={k:v.tolist() for k,v in split.items()},
        source=str(src),metrics=metrics,pca_2d_3d=audits,selection='Post-hoc task-relevant highlights; no point selection; all probe-test points.',
        readout='Supervised ridge axis; alpha validation-selected. Orthogonal nuisance is PC1 residualized against prediction on train.',
        caveat='Exploratory previously inspected bank; Cube/PushT encoder-training disjointness unverified. Not planning success.'),indent=2)+'\n')
    return sample,split,ys,output,metrics

def draw(task, prepared, mode):
    sample,split,ys,output,metrics=prepared
    paired=mode=='paired'
    p=apply_style(); fig=plt.figure(figsize=(9,5.8 if paired else 3.7))
    gs=fig.add_gridspec(5 if paired else 3,6,left=.058,right=.735,bottom=.16 if paired else .20,top=.78 if paired else .73,
                       height_ratios=[1,1,1,1,.06] if paired else [1,1,.06],wspace=.04,hspace=.14 if paired else .08)
    from textwrap import fill
    for j,field in enumerate(FIELDS[task]):
        y=ys[field][split['test'],0]
        signed=('minus' in field or '_rel_' in field or field in ['relative_x','relative_y','height'])
        kind='binary' if field=='strict' else 'signed' if signed else 'linear'
        cmap=atlas.theme_colormap(kind,p)
        lo,hi=float(y.min()),float(y.max())
        if signed: hi=max(abs(lo),abs(hi));lo=-hi
        # All values retained. Shared nonlinear physical scale reveals the
        # small insertion band without clipping rare large-error states.
        norm=(colors.SymLogNorm(linthresh=3.,vmin=lo,vmax=hi) if task=='gear' and signed
              else colors.PowerNorm(.45,vmin=lo,vmax=hi) if task=='gear' and kind=='linear'
              else colors.Normalize(lo,hi))
        both=np.concatenate([output[k]['readout'][j][:,0] for k in ['lewm','ours']])
        for row,k in enumerate(['lewm','ours']*(2 if paired else 1)):
            view=('pca2d' if row<2 else 'readout') if paired else mode
            ax=fig.add_subplot(gs[row,j],projection='3d' if mode=='pca3d' else None)
            coords=output[k]['readout'][j] if view=='readout' else output[k]['pca']
            kw=dict(c=y,cmap=cmap,norm=norm,s=2.0,alpha=.85,linewidths=0,rasterized=True)
            if mode=='pca3d':
                ax.scatter(*coords.T,depthshade=False,**kw);ax.view_init(elev=24,azim=-55)
                ax.set_box_aspect((1,1,1));ax.set_zticks([])
            else:
                ax.scatter(coords[:,0],coords[:,1],**kw);ax.set_box_aspect(1)
                if view=='readout':
                    pad=max(np.ptp(both)*.03,1e-6);ax.set_xlim(both.min()-pad,both.max()+pad)
            ax.set_xticks([]);ax.set_yticks([]);ax.grid(False)
            for s in ax.spines.values():s.set_color(p['grid'])
            if row==0:ax.set_title(fill(SHORT_LABELS[task][j],15),fontsize=8,pad=4)
            if j==0 and mode!='pca3d':
                label=('LeWM' if k=='lewm' else 'Σ-LeWM')
                if paired:label=('PCA' if row<2 else 'Readout')+' / '+label
                ax.set_ylabel(label,fontsize=8,color=p[k],labelpad=3)
            if j==0 and mode=='pca3d':ax.text2D(-.12,.5,'LeWM' if k=='lewm' else 'Σ-LeWM',transform=ax.transAxes,rotation=90,fontsize=8,color=p[k])
        cax=fig.add_subplot(gs[4 if paired else 2,j]);cb=fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal')
        cb.set_ticks([lo,hi]);cb.set_ticklabels(['no','yes'] if kind=='binary' else [f'{lo:.1f}',f'{hi:.1f}'])
        cb.ax.tick_params(labelsize=6,length=1,pad=1);cb.outline.set_visible(False)
        cb.ax.get_xticklabels()[0].set_ha('left');cb.ax.get_xticklabels()[-1].set_ha('right')
    bars=fig.add_gridspec(6,1,left=.79,right=.963,bottom=.23 if paired else .20,top=.75,hspace=1.2)
    for j,field in enumerate(FIELDS[task]):
        ax=fig.add_subplot(bars[j,0]); rr=[next(r for r in metrics if r['model']==k and r['field']==field) for k in ['lewm','ours']]
        unit='px' if task=='pusht' else 'a.u.' if field=='opening' else 'mm'
        title=SHORT_LABELS[task][j]+(' | AP' if field=='strict' else f' | {unit}')
        for i,(k,r) in enumerate(zip(['lewm','ours'],rr)):
            ax.barh(1-i,r['value'],height=.7,color=p[k]);ax.text(r['value'],1-i,f"  {r['value']:.2f}",fontsize=8,va='center')
        ax.set_xlim(0,max(r['value'] for r in rr)*1.4);ax.set_xticks([]);ax.set_yticks([])
        ax.set_title(title,fontsize=7.5,loc='left',pad=2);ax.grid(False)
        for s in ax.spines.values():s.set_visible(False)
    kind={'paired':'PCA + SUPERVISED READOUT','readout':'SUPERVISED READOUT','pca2d':'UNSUPERVISED PCA (2D)','pca3d':'UNSUPERVISED PCA (3D)'}[mode]
    fig.text(.045,.972,f'{task.title()}  /  {kind}',fontsize=8,color=p['muted'])
    fig.text(.045,.90,message(task,mode),fontsize=11,weight='bold')
    detail={'pusht':'Block x/y  |  Pusher x/y  |  Relative = pusher minus block',
            'cube':'Relative = effector minus block  |  Motion = displacement over 5 steps',
            'gear':'Relative = gear minus fixture  |  Strict = instantaneous geometry, not rollout success'}[task]
    fig.text(.045,.835,detail,fontsize=7.5,color=p['muted'])
    fig.text(.79,.972,'FULL-LATENT READOUT',fontsize=8,color=p['muted'])
    fig.text(.79,.835,'Mean absolute error: shorter is better' if task!='gear' else 'Error: shorter better | AP: longer better',fontsize=6.7)
    guide=('Top two rows: unsupervised PCA. Bottom two: supervised prediction axis. Same points and colours; one shared score panel.' if paired else
           'Read left to right: colour should track the predicted value. Each column is a separate supervised axis.'
           if mode=='readout' else 'Look for colour organization, not cloud shape. The right-hand bars test the full embedding, not this projection.')
    fig.text(.045,.098,guide,fontsize=7.5)
    fig.text(.045,.038,f"All {len(split['test'])} paired test points; no point selection. Colour = true physical value. Bars: grey LeWM, blue Σ-LeWM.",fontsize=7,color=p['muted'])
    stem=f'figA_{task}_focused_{mode}'
    for ext in ['pdf','png','svg']:fig.savefig(atlas.FIGS/f'{stem}.{ext}',dpi=240)
    plt.close(fig)

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--refresh-original-atlases',action='store_true')
    args=parser.parse_args()
    for task in FIELDS:
        if args.refresh_original_atlases:
            from scripts.cube_gear_physical_atlas import CUBE_GROUPS, GEAR_GROUPS
            atlas.OUT=source(task)
            sample=dict(np.load(atlas.OUT/'sample.npz'))
            emb={k:np.load(atlas.OUT/f'emb_{k}.npy') for k in ['lewm','ours']}
            frozen={k:dict(np.load(atlas.OUT/f'pca_{k}.npz')) for k in emb}
            groups=atlas.GROUPS if task=='pusht' else CUBE_GROUPS if task=='cube' else GEAR_GROUPS
            explanations=None if task=='pusht' else dict(
                positions='Recorded physical positions; control target is commanded in Gear.',
                goals='Logged Cube target / Gear instantaneous strict geometry; not rollout success.',
                motion='Measured motion over five raw steps; yaw sign is about world z.')
            atlas.render(sample,emb,json.loads((atlas.OUT/'provenance.json').read_text()),
                         frozen_pca=frozen,task=task,groups=groups,explanations=explanations)
        prepared=prepare(task)
        for mode in ['readout','pca2d','pca3d']+([] if task=='pusht' else ['paired']):draw(task,prepared,mode)
        print(task,'done',flush=True)

if __name__=='__main__':
    from scripts.archive_embedding_round import archive_round
    with archive_round('focused_physical_atlas'):
        main()

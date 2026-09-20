#!/usr/bin/env python3
"""Paired Cube/Gear physical atlas, frozen paper checkpoints; CPU only."""
from repro_paths import resolve_path as _resolve_artifact
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
import sys
import json
import csv
import hashlib
import argparse
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import pusht_physical_atlas as atlas
from scripts.recover_fig9_atlas import original_rows

ROOT = atlas.ROOT
OUT = ROOT/'outputs/cube_gear_physical_atlas_v1'
CUBE = Path(_resolve_artifact('stable:datasets/cube_single_expert.h5'))
GEAR = ROOT/'outputs/isaaclab_automate/gear_01036_eval_set_v2_seed5242_n1000/eval_trajectories.h5'


def v(field, label, unit='mm', kind='linear'):
    return (field, label, unit, 'paper', kind)


CUBE_GROUPS = [
    ('positions', 'Cube | object and end-effector position', [
        v('block_x','Block x'), v('block_y','Block y'), v('block_z','Block z'),
        v('control_x','End-effector x'), v('control_y','End-effector y'), v('control_z','End-effector z')]),
    ('goals', 'Cube | recorded target and task geometry', [
        v('goal_x','Target x'), v('goal_y','Target y'), v('goal_z','Target z'),
        v('goal_distance','Target distance'), v('goal_angle_error','Target yaw error','deg','angle_error'),
        v('opening','Gripper opening','a.u.')]),
    ('motion', 'Cube | orientation, motion and contact', [
        v('angle','Block yaw','deg','angle'), v('rotation_angle','Yaw change / 5','deg','signed'),
        v('rotation','Rotation sign','','category'), v('block_motion','Block motion / 5'),
        v('control_motion','Effector motion / 5'), v('contact','Gripper contact','','binary')])]

GEAR_GROUPS = [
    ('positions', 'Gear | object, fixture and controller target', [
        v('held_x','Gear x'), v('held_y','Gear y'), v('fixed_x','Fixture x'), v('fixed_y','Fixture y'),
        v('control_x','Control target x'), v('control_y','Control target y')]),
    ('goals', 'Gear | insertion geometry and strict predicate', [
        v('relative_x','Gear - fixture x','mm','signed'), v('relative_y','Gear - fixture y','mm','signed'),
        v('height','Seating height','mm','signed'), v('xy_error','Lateral error'),
        v('orientation_error','Orientation error','deg'), v('strict','Strict frame','','binary')]),
    ('motion', 'Gear | orientation and measured motion', [
        v('angle','Gear yaw','deg','angle'), v('rotation_angle','Yaw change / 5','deg','signed'),
        v('rotation','Rotation sign','','category'), v('speed','Gear speed','mm/s'),
        v('angular_speed','Angular speed','deg/s'), v('block_motion','Gear motion / 5')])]


def quat_yaw(q):
    """Isaac wxyz unit quaternion; positive right-handed yaw about world z."""
    q = q/np.linalg.norm(q, axis=-1, keepdims=True)
    w,x,y,z = np.moveaxis(q, -1, 0)
    return np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def rotation_features(path):
    delta = np.rad2deg(atlas.wrapped_angle(np.diff(path,axis=1)).sum(1))
    return delta, np.where(delta>1,1,np.where(delta < -1,-1,0))


def cube_data(f, rows):
    take = lambda name: np.asarray(f[name])[rows]
    block, control = take('privileged_block_0_pos'), take('proprio_effector_pos')
    goal = take('privileged_target_block_pos')
    yaw = take('privileged_block_0_yaw')[:,0]
    yaw_all = np.asarray(f['privileged_block_0_yaw'])[:,0]
    delta, direction = rotation_features(yaw_all[rows[:,None]+np.arange(6)])
    features = {**{f'block_{a}':block[:,i]*1000 for i,a in enumerate('xyz')},
                **{f'control_{a}':control[:,i]*1000 for i,a in enumerate('xyz')},
                **{f'goal_{a}':goal[:,i]*1000 for i,a in enumerate('xyz')},
                'goal_distance':np.linalg.norm(goal-block,axis=1)*1000,
                'goal_angle_error':np.rad2deg(atlas.wrapped_angle(take('privileged_target_block_yaw')[:,0]-yaw)),
                'opening':take('proprio_gripper_opening')[:,0],
                'angle':np.rad2deg(yaw % (2*np.pi)), 'rotation_angle':delta,'rotation':direction,
                'block_motion':np.linalg.norm(np.asarray(f['privileged_block_0_pos'])[rows+5]-block,axis=1)*1000,
                'control_motion':np.linalg.norm(np.asarray(f['proprio_effector_pos'])[rows+5]-control,axis=1)*1000,
                'contact':(take('proprio_gripper_contact')[:,0]>0).astype(int)}
    metric_state = np.column_stack([block, np.cos(yaw),np.sin(yaw),control])
    return features,metric_state


def gear_data(f, rows):
    from scripts.isaac_gear_eval_env import _strict_mask
    take = lambda name: np.asarray(f[name])[rows]
    held,fixed,control = take('held_pos'),take('fixed_pos'),take('ctrl_target_fingertip_midpoint_pos')
    q, fq = take('held_quat'), take('fixed_quat')
    strict,height,xy,error = _strict_mask(held,q,fixed,fq,max_height_m=.003,max_xy_error_m=.003,max_orientation_deg=5)
    yaw_all = quat_yaw(np.asarray(f['held_quat']))
    yaw = yaw_all[rows]
    delta,direction = rotation_features(yaw_all[rows[:,None]+np.arange(6)])
    vel = take('held_root_vel')
    relative=held-fixed
    features={**{f'held_{a}':held[:,i]*1000 for i,a in enumerate('xy')},
              **{f'fixed_{a}':fixed[:,i]*1000 for i,a in enumerate('xy')},
              **{f'control_{a}':control[:,i]*1000 for i,a in enumerate('xy')},
              'relative_x':relative[:,0]*1000, 'relative_y':relative[:,1]*1000,
              'height':height*1000,'xy_error':xy*1000,'orientation_error':error,'strict':strict.astype(int),
              'angle':np.rad2deg(yaw % (2*np.pi)),'rotation_angle':delta,'rotation':direction,
              'speed':np.linalg.norm(vel[:,:3],axis=1)*1000,
              'angular_speed':np.rad2deg(np.linalg.norm(vel[:,3:],axis=1)),
              'block_motion':np.linalg.norm(np.asarray(f['held_pos'])[rows+5]-held,axis=1)*1000}
    # Relative task geometry, not global fixture placement or temporal index.
    metric_state = np.column_stack([relative, np.cos(yaw),np.sin(yaw),control-fixed])
    return features,metric_state


def encode(f, rows, checkpoint, output):
    import torch
    import stable_worldmodel as swm
    from slewm.training.utilities import get_img_preprocessor
    ident={'checkpoint':str(checkpoint),'sha256':atlas.sha256(checkpoint),
           'config_sha256':atlas.sha256(checkpoint.parent/'config.json'),
           'rows_sha256':hashlib.sha256(rows.tobytes()).hexdigest(),
           'dataset':f.filename,'dataset_mtime_ns':Path(f.filename).stat().st_mtime_ns,
           'preprocess_sha256':atlas.sha256(ROOT/'slewm/training/utilities.py')}
    stamp=output.with_suffix('.json')
    if output.exists() and stamp.exists() and json.loads(stamp.read_text())==ident:
        return np.load(output),ident
    model=swm.wm.utils.load_pretrained(str(checkpoint)).eval()
    pre=get_img_preprocessor('pixels','pixels',224)
    batches=[]
    with torch.inference_mode():
        for i in range(0,len(rows),16):
            px=np.stack([f['pixels'][int(r)] for r in rows[i:i+16]])
            px=pre({'pixels':torch.from_numpy(px).permute(0,3,1,2).contiguous()})['pixels']
            batches.append(model.encode({'pixels':px[:,None]})['emb'][:,0].float().numpy())
            if i%640==0:print(checkpoint.parent.name,min(i+16,len(rows)),len(rows),'CPU',flush=True)
    z=np.concatenate(batches); assert z.shape==(len(rows),192) and np.isfinite(z).all()
    np.save(output,z);stamp.write_text(json.dumps(ident,indent=2)+'\n')
    return z,ident


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=['cube','gear','all'],default='all');args=parser.parse_args()
    import torch,h5py,hdf5plugin
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import trustworthiness
    sys.path.insert(0,str(atlas.STYLE));import latent_metrics as old
    torch.set_num_threads(4);torch.manual_seed(3072)
    for task in (['cube','gear'] if args.task=='all' else [args.task]):
        atlas.OUT=OUT/task;atlas.OUT.mkdir(parents=True,exist_ok=True)
        data=CUBE if task=='cube' else GEAR
        if task=='cube':
            ckpts={k:atlas.CKPT_ROOT/name/'weights.pt' for k,name in
                   {'lewm':'models--quentinll--lewm-cube','ours':'t3_ours_cube_plan'}.items()}
            selection='Existing paper champion: cube_C_T_dcl_idm_sigreg_s3072 epoch 8; plan-export encoder tensors match exactly. Not selected from atlas.'
        else:
            records={k:json.loads((ROOT/f'outputs/paper_new_tasks_v1/gear/{arm}/selection.json').read_text())
                     for k,arm in [('lewm','l0'),('ours','c0')]}
            ckpts={k:Path(v['checkpoint']) for k,v in records.items()}
            for k,p in ckpts.items():assert atlas.sha256(p)==records[k]['checkpoint_sha256']
            selection='Paper L0 epoch11 vs C0 epoch18, frozen validation-loss-composite EMA selection; not success-selected.'
        with h5py.File(data,'r') as f:
            lengths,offsets=np.asarray(f['ep_len']),np.asarray(f['ep_offset'])
            if task=='cube':rows,episodes,starts=original_rows(lengths,offsets)
            else:
                eps=np.arange(len(lengths));rng=np.random.default_rng(3072)
                starts=np.array([rng.integers(0,int(lengths[e])-15) for e in eps])
                rows=(offsets[eps,None]+starts[:,None]+np.array([0,5,10])).ravel();episodes=np.repeat(eps,3)
            assert np.all(rows+5<offsets[episodes]+lengths[episodes])
            assert np.array_equal(np.asarray(f['ep_idx'])[rows],episodes)
            features,state=(cube_data if task=='cube' else gear_data)(f,rows)
            assert all(np.isfinite(v).all() for v in features.values())
            np.savez_compressed(atlas.OUT/'sample.npz',rows=rows,episodes=episodes,starts=starts,metric_state=state,**features)
            with (atlas.OUT/'physical_states.csv').open('w') as out:
                w=csv.writer(out);w.writerow(['point_id','row','episode',*features]);w.writerows(zip(np.arange(len(rows)),rows,episodes,*features.values()))
            embeddings,identities,frozen,metrics={},{},{},{}
            sz=StandardScaler().fit_transform(state)
            for k,p in ckpts.items():
                z,ident=encode(f,rows,p,atlas.OUT/f'emb_{k}.npy');embeddings[k]=z;identities[k]=ident
                pfile=atlas.OUT/f'pca_{k}.npz'
                if pfile.exists():frozen[k]=dict(np.load(pfile))
                else:
                    pc=PCA(2,svd_solver='covariance_eigh').fit(z)
                    frozen[k]=dict(coordinates=pc.transform(z),components=pc.components_,mean=pc.mean_,explained_variance_ratio=pc.explained_variance_ratio_)
                # Match sklearn.transform's arithmetic order in float32.
                expected_xy=z @ frozen[k]['components'].T-frozen[k]['mean'][None] @ frozen[k]['components'].T
                np.testing.assert_allclose(expected_xy,frozen[k]['coordinates'],atol=1e-5)
                metrics[k]=dict(effective_rank=old.effective_rank(z),
                                knn_overlap_at20=old.knn_overlap(z[2::3],sz[2::3],20),
                                trustworthiness_at20=float(trustworthiness(sz[2::3],z[2::3],n_neighbors=20)),
                                spearman=old.spearman_dist(z[2::3],sz[2::3]),
                                pc1_pc2_var=float(frozen[k]['explained_variance_ratio'].sum()))
                with (atlas.OUT/f'coordinates_{k}.csv').open('w') as out:
                    w=csv.writer(out);w.writerow(['point_id','row','episode','pc1','pc2']);w.writerows(zip(np.arange(len(rows)),rows,episodes,*frozen[k]['coordinates'].T))
                print(task,k,metrics[k],flush=True)
        manifest=dict(task=task,n_frames=len(rows),n_distinct_episodes=len(np.unique(episodes)),
                      selection=selection,models=identities,metrics=metrics,no_training_or_planning=True,
                      sampling='Original Fig9 1000 clips with replacement, last-decile episodes' if task=='cube' else 'One random clip in each of all 1000 separately collected evaluation episodes; no success filtering',
                      split_caveat='Cube last-decile is not verified training-disjoint; Gear is a separately collected evaluation sidecar',
                      physical_metric_state='block xyz, cos/sin yaw, effector xyz' if task=='cube' else 'gear-minus-fixture xyz, cos/sin gear yaw, controller-target-minus-fixture xyz',
                      target_contract='Logged privileged_target_block_pos/yaw; not future substitute' if task=='cube' else 'Measured fixture pose; controller target is commanded pose, not measured end-effector',
                      strict_contract=None if task=='cube' else 'Instantaneous 0<height<=3mm, lateral<=3mm, quaternion error<=5deg; NOT three-step persistent success',
                      rotation_contract='Sum wrapped world-z yaw increments over 5 steps; positive right-handed; 1deg display deadband')
        explanations=({'positions':'Recorded world positions; mm. End-effector position is measured.',
                       'goals':'Target is logged at the current frame; no future-goal substitution.',
                       'motion':'Yaw sign is about world z; contact is the logged positive-contact indicator.'}
                      if task=='cube' else
                      {'positions':'Recorded world positions; mm. Control target is commanded, not measured.',
                       'goals':'Strict frame: 0 < height <= 3 mm, lateral <= 3 mm, rotation <= 5 deg; no persistence.',
                       'motion':'Measured gear velocities; yaw sign is about world z, not screen clockwise.'})
        atlas.render(features,embeddings,manifest,frozen_pca=frozen,task=task,
                     groups=CUBE_GROUPS if task=='cube' else GEAR_GROUPS,explanations=explanations)


if __name__=='__main__':
    from scripts.archive_embedding_round import archive_round
    with archive_round('cube_gear_atlas'):
        main()

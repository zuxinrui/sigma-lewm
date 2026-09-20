#!/usr/bin/env python3
"""Reproducible, paired PushT PCA atlas; no training or simulator rollout."""
from __future__ import annotations
from repro_paths import resolve_path as _resolve_artifact

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import textwrap
import logging

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / 'outputs/figs'
STYLE = ROOT / 'scripts/plotting'
OUT = ROOT / 'outputs/pusht_physical_atlas_v1'
DATA = Path(_resolve_artifact('stable:datasets/pusht_expert_train.h5'))
CKPT_ROOT = Path(_resolve_artifact('stable:checkpoints'))
MODELS = {'lewm': 'models--quentinll--lewm-pusht', 'ours': 't3_ours_pusht_plan'}
SEED = 3072


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def wrapped_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def physical_features(state, next_state, goal, angle_path):
    """Angles are radians in source state; colour values use degrees.

    y points downward in the renderer, so positive angle change appears CW.
    Summing wrapped one-raw-step differences avoids the 0/2pi seam.
    """
    relative = state[:, :2] - state[:, 2:4]
    error = goal[:, 2:4] - state[:, 2:4]
    delta = np.rad2deg(wrapped_angle(np.diff(angle_path, axis=1)).sum(axis=1))
    rotation = np.where(delta > 1., 1, np.where(delta < -1., -1, 0))
    return {
        'block_x': state[:, 2], 'block_y': state[:, 3],
        'pusher_x': state[:, 0], 'pusher_y': state[:, 1],
        'pusher_rel_x': relative[:, 0], 'pusher_rel_y': relative[:, 1],
        'goal_x': goal[:, 2], 'goal_y': goal[:, 3],
        'goal_dx': error[:, 0], 'goal_dy': error[:, 1],
        'goal_distance': np.linalg.norm(error, axis=1),
        'goal_angle_error': np.rad2deg(wrapped_angle(goal[:, 4]-state[:, 4])),
        'block_angle': np.rad2deg(state[:, 4] % (2*np.pi)),
        'angle_change': delta, 'rotation': rotation,
        'pusher_speed': np.linalg.norm(state[:, 5:7], axis=1),
        'block_motion': np.linalg.norm(next_state[:, 2:4]-state[:, 2:4], axis=1),
        'pusher_motion': np.linalg.norm(next_state[:, :2]-state[:, :2], axis=1),
    }


def sample_rows(lengths, offsets, n_episodes):
    candidates = np.arange(round(len(lengths)*.9), len(lengths))
    candidates = candidates[lengths[candidates] >= 36]
    if n_episodes > len(candidates):
        raise ValueError('Not enough distinct sufficiently long episodes')
    rng = np.random.default_rng(SEED)
    episodes = np.sort(rng.choice(candidates, n_episodes, replace=False))
    starts = np.array([rng.integers(0, int(lengths[e])-35) for e in episodes])
    rows = (offsets[episodes, None]+starts[:, None]+np.array([0, 5, 10])).reshape(-1)
    ids = np.repeat(episodes, 3)
    assert np.all(rows+25 < offsets[ids]+lengths[ids])
    return rows, ids


GROUPS = [
    ('positions', 'Position and relative geometry', [
        ('block_x', 'Block x', 'px', 'viridis', 'linear'),
        ('block_y', 'Block y', 'px', 'plasma', 'linear'),
        ('pusher_x', 'Control point x', 'px', 'viridis', 'linear'),
        ('pusher_y', 'Control point y', 'px', 'plasma', 'linear'),
        ('pusher_rel_x', 'Control point - block: x', 'px', 'RdBu_r', 'signed'),
        ('pusher_rel_y', 'Control point - block: y', 'px', 'RdBu_r', 'signed')]),
    ('goals', 'Future reference goal and pose error', [
        ('goal_x', 'Reference-goal x', 'px', 'viridis', 'linear'),
        ('goal_y', 'Reference-goal y', 'px', 'plasma', 'linear'),
        ('goal_distance', 'Block-to-goal distance', 'px', 'cividis', 'linear'),
        ('goal_dx', 'Goal - block: x', 'px', 'RdBu_r', 'signed'),
        ('goal_dy', 'Goal - block: y', 'px', 'RdBu_r', 'signed'),
        ('goal_angle_error', 'Goal - block: angle', 'deg', 'RdBu_r', 'angle_error')]),
    ('motion', 'Orientation, rotation and motion', [
        ('block_angle', 'Block orientation', 'deg', 'twilight', 'angle'),
        ('angle_change', 'Signed rotation / 5 steps', 'deg', 'RdBu_r', 'signed'),
        ('rotation', 'Rotation direction', '', 'coolwarm', 'category'),
        ('pusher_speed', 'Control-point speed', 'px/s', 'magma', 'linear'),
        ('block_motion', 'Block displacement / 5 steps', 'px', 'cividis', 'linear'),
        ('pusher_motion', 'Control displacement / 5 steps', 'px', 'magma', 'linear')]),
]


def theme_colormap(kind, palette):
    """Physical semantics use paper hues, identically for both model rows."""
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    if kind == 'category':
        return ListedColormap([palette['backward'], palette['lewm_light'], palette['inverse']], name='paper_rotation')
    if kind == 'binary':
        return ListedColormap([palette['lewm_light'], palette['ours']], name='paper_binary')
    if kind in ('signed', 'angle_error'):
        return LinearSegmentedColormap.from_list('paper_signed', [palette['backward'], '#D8DCE2', palette['inverse']])
    if kind == 'angle':
        return LinearSegmentedColormap.from_list('paper_cyclic', [palette['ours'], palette['backward'], palette['inverse'], palette['action'], palette['ours']])
    return LinearSegmentedColormap.from_list('paper_blue', [palette['ours_fill'], palette['ours_light'], palette['ours'], '#284E89'])


def render(features, embeddings, provenance, frozen_pca=None, *, task='pusht', groups=None, explanations=None):
    sys.path.insert(0, str(STYLE))
    FIGS.mkdir(parents=True, exist_ok=True)
    from _common import apply_style, plt
    from matplotlib import colors, cm
    from sklearn.decomposition import PCA

    palette = apply_style()
    logging.getLogger('fontTools').setLevel(logging.ERROR)
    names = {'lewm': 'LeWM', 'ours': 'Σ-LeWM'}
    coordinates, variance, limits = {}, {}, {}
    for key, z in embeddings.items():
        if frozen_pca is None:
            pca = PCA(n_components=2, svd_solver='full').fit(z)
            saved = dict(coordinates=pca.transform(z), components=pca.components_,
                         mean=pca.mean_, explained_variance_ratio=pca.explained_variance_ratio_)
        else:
            saved = frozen_pca[key]
            assert len(saved['coordinates']) == len(z)
        coordinates[key] = saved['coordinates']
        variance[key] = saved['explained_variance_ratio'].tolist()
        xy = coordinates[key]
        low, high = xy.min(0), xy.max(0)
        pad = np.maximum((high-low)*.04, 1e-6)
        limits[key] = (low-pad, high+pad)
        np.savez_compressed(OUT/f'pca_{key}.npz', **saved)
    # Same draw order in every panel, independent of physical colour.
    n = len(next(iter(features.values())))
    order = (np.arange(n) if frozen_pca is not None else
             np.random.default_rng(SEED+1).permutation(n))
    summaries = []
    layout_audit = {}
    for slug, title, variables in (GROUPS if groups is None else groups):
        fig = plt.figure(figsize=(7.6, 3.1))
        gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, .075],
                              left=.045, right=.77, bottom=.255, top=.82,
                              wspace=.045, hspace=.08)
        headlines={
            ('pusht','positions'):'PushT | Where are the block and the pusher?',
            ('pusht','goals'):'PushT | Future position is a reference, not a recorded target',
            ('pusht','motion'):'PushT | Separate orientation from subsequent motion',
            ('cube','positions'):'Cube | Absolute position is not the key distinction',
            ('cube','goals'):'Cube | Logged targets and gripper state describe different factors',
            ('cube','motion'):'Cube | Motion, yaw and contact are distinct physical signals',
            ('gear','positions'):'Gear | Global placement is not insertion accuracy',
            ('gear','goals'):'Gear | Judge insertion by relative error, not cloud shape',
            ('gear','motion'):'Gear | A velocity colour is not evidence of action sensitivity',
        }
        fig.suptitle(headlines.get((task,slug),title), x=.045, y=.99, ha='left', fontsize=9,weight='bold')
        fig.text(.045,.915,'Read each column: top LeWM, bottom Σ-LeWM. Same points; colour changes with the named physical quantity.',
                 fontsize=6.8,color=palette['muted'])
        data_axes, colour_tick_labels = [], []
        for j, (field, label, unit, cmap_name, kind) in enumerate(variables):
            col = j
            values = features[field]
            valid_values = values[np.isfinite(values)]
            if not len(valid_values):
                raise ValueError(f'No valid colour values for {field}')
            constant = kind == 'linear' and float(np.ptp(valid_values)) == 0.
            if kind == 'angle':
                low, high, norm = 0., 360., colors.Normalize(0, 360)
            elif kind == 'angle_error':
                low, high, norm = -180., 180., colors.TwoSlopeNorm(0, -180, 180)
            elif kind == 'category':
                low, high = -1., 1.
                norm = colors.BoundaryNorm([-1.5, -.5, .5, 1.5], 3)
            elif kind == 'binary':
                low, high = 0., 1.
                norm = colors.BoundaryNorm([-.5, .5, 1.5], 2)
            elif kind == 'signed':
                high = max(float(np.max(np.abs(valid_values))), 1e-6)
                low, norm = -high, colors.TwoSlopeNorm(0, -high, high)
            else:
                low, high = float(valid_values.min()), float(valid_values.max())
                if low == high:
                    high = low+1e-6
                norm = colors.Normalize(low, high)
            cmap = theme_colormap(kind, palette)
            if constant:
                centre = float(valid_values[0])
                low, high = centre-1e-6, centre+1e-6
                norm = colors.Normalize(low, high)
                cmap = colors.ListedColormap([cmap(.55)], name='paper_blue_constant')
            cmap = cmap.copy()
            cmap.set_bad('#c8cbd0', alpha=.65)
            for row, key in enumerate(('lewm', 'ours')):
                ax = fig.add_subplot(gs[row, col])
                data_axes.append(ax)
                ax.set_box_aspect(1)
                xy = coordinates[key]
                ax.scatter(xy[order, 0], xy[order, 1], c=values[order], cmap=cmap,
                           norm=norm, s=1.1, alpha=.80, linewidths=0, rasterized=True,
                           plotnonfinite=True)
                ax.set_xlim(limits[key][0][0], limits[key][1][0])
                ax.set_ylim(limits[key][0][1], limits[key][1][1])
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                for spine in ax.spines.values():
                    spine.set_visible(True); spine.set_color(palette['grid'])
                if row == 0:
                    short = label.replace('Control point', 'Pusher' if task=='pusht' else 'Control')
                    short = short.replace('Control-point', 'Pusher').replace('Reference-goal', 'Ref. goal')
                    short = short.replace('displacement', 'motion').replace('orientation', 'angle')
                    ax.set_title(textwrap.fill(short, 16), fontsize=6.2, pad=4, loc='center')
                if col == 0:
                    ax.set_ylabel(names[key], fontsize=6.3,
                                  color=palette[key])
            cax = fig.add_subplot(gs[2, col])
            cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                              orientation='horizontal')
            ticks = [centre] if constant else [-1, 0, 1] if kind == 'category' else [0, 1] if kind == 'binary' else [low, 0, high] if kind in ('signed','angle_error') else [low, high]
            cb.set_ticks(ticks)
            category_labels = ['CCW', '0', 'CW'] if task == 'pusht' else ['−', '0', '+']
            cb.set_ticklabels([f'{centre:g} (const.)'] if constant else category_labels if kind == 'category' else ['no', 'yes'] if kind == 'binary'
                              else [f'{t:.0f}' if abs(high-low)>5 else f'{t:.2f}' for t in ticks])
            cb.ax.get_xticklabels()[0].set_ha('left')
            cb.ax.get_xticklabels()[-1].set_ha('right')
            if constant:
                cb.ax.get_xticklabels()[0].set_ha('center')
            colour_tick_labels.extend(cb.ax.get_xticklabels())
            cb.ax.tick_params(labelsize=5.5, length=1.5, pad=1)
            cb.outline.set_visible(False)
            if unit:
                cb.ax.set_xlabel(unit, fontsize=5.5, labelpad=1)
            summaries.append({'field': field, 'unit': unit, 'cmap': cmap.name,
                              'normalization': 'constant' if constant else kind, 'vmin': low, 'vmax': high,
                              'observed_min': float(valid_values.min()),
                              'observed_max': float(valid_values.max()),
                              'mean': float(valid_values.mean()),
                              'std': float(valid_values.std()),
                              'n_valid': len(valid_values), 'n_missing': len(values)-len(valid_values)})
        explanation = ('Goal = same-episode state 25 raw steps later; not a logged task target.'
                       if slug == 'goals' else
                       'Positive rotation is clockwise on screen; still means |rotation| <= 1 deg.'
                       if slug == 'motion' else
                       'Coordinates use the 512-pixel workspace, not the resized 224-pixel image.')
        if explanations is not None:
            explanation = explanations[slug]
        missing_goals = task == 'pusht' and slug == 'goals' and np.isnan(features['goal_x']).any()
        fig.text(.062, .070 if missing_goals else .085, explanation, fontsize=6, color=palette['muted'])
        if missing_goals:
            fig.text(.062, .112, f"Grey: {np.isnan(features['goal_x']).sum()} states lack a same-episode t+25 frame; all points retained.",
                     fontsize=6, color=palette['muted'])
        axbar = fig.add_axes([.805, .43, .17, .28])
        for i, key in enumerate(('lewm', 'ours')):
            value = 100*sum(variance[key])
            axbar.barh(1-i, value, height=.48, color=palette[key])
            axbar.text(value+1, 1-i, f'{value:.1f}%', va='center', fontsize=7)
        axbar.set_xlim(0, 100); axbar.set_ylim(-.6, 1.6)
        axbar.set_yticks([]); axbar.set_xticks([0,50,100])
        axbar.tick_params(labelsize=6)
        axbar.set_title('Projection coverage (%)', fontsize=7)
        axbar.grid(axis='y', visible=False)
        fig.text(.805, .27, 'Variance kept by PC1 + PC2\nGrey LeWM | Blue Σ-LeWM\nNot accuracy or success', fontsize=6, color=palette['muted'])
        fig.text(.062, .025, f"Same {len(order):,} states; fixed PCA per model.",
                 fontsize=6, color=palette['muted'])
        stem = f'figA_{task}_{slug}'
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ratios = [float(ax.get_window_extent(renderer).width / ax.get_window_extent(renderer).height)
                  for ax in data_axes]
        assert all(abs(r-1)<1e-6 for r in ratios), ratios
        boxes = [label.get_window_extent(renderer) for label in colour_tick_labels]
        overlaps = [(i,j) for i,a in enumerate(boxes) for j,b in enumerate(boxes[i+1:],i+1)
                    if a.overlaps(b)]
        assert not overlaps, (stem, 'overlapping colour ticks', overlaps)
        layout_audit[slug] = {'columns': 6, 'model_rows': 2, 'panel_width_over_height': ratios,
                              'colour_tick_overlap_count': len(overlaps)}
        for extension in ('pdf', 'png', 'svg'):
            fig.savefig(FIGS/f'{stem}.{extension}', dpi=300)
        plt.close(fig)
    with (OUT/'color_axes.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    provenance['pca_variance'] = variance
    provenance['color_axes'] = summaries
    provenance['layout_audit'] = layout_audit
    (OUT/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = str(args.threads)
    import h5py
    import hdf5plugin  # noqa: F401
    import torch
    import stable_worldmodel as swm
    sys.path.insert(0, str(ROOT))
    from slewm.training.utilities import get_img_preprocessor

    torch.set_num_threads(args.threads)
    torch.manual_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    with h5py.File(DATA, 'r') as f:
        lengths, offsets = np.asarray(f['ep_len']), np.asarray(f['ep_offset'])
        rows, episodes = sample_rows(lengths, offsets, args.episodes)
        all_state = np.asarray(f['state'])
        assert np.array_equal(np.asarray(f['episode_idx'])[rows], episodes)
        assert np.array_equal(np.asarray(f['episode_idx'])[rows+25], episodes)
        state, nxt, goal = all_state[rows], all_state[rows+5], all_state[rows+25]
        paths = all_state[rows[:, None]+np.arange(6), 4]
        features = physical_features(state, nxt, goal, paths)
        assert all(np.isfinite(v).all() for v in features.values())
        manifest = {'version': 1, 'dataset': str(DATA), 'dataset_size': DATA.stat().st_size,
                    'dataset_mtime_ns': DATA.stat().st_mtime_ns, 'seed': SEED,
                    'n_frames': len(rows), 'n_distinct_episodes': len(np.unique(episodes)),
                    'episode_index_min': int(episodes.min()), 'episode_index_max': int(episodes.max()),
                    'row_sha256': hashlib.sha256(rows.tobytes()).hexdigest(),
                    'state_sha256': hashlib.sha256(np.concatenate([state, nxt, goal], axis=1).tobytes()).hexdigest(),
                    'goal_contract': 'same-episode state at t+25 raw steps; not a logged environment target',
                    'sampling': f'{args.episodes} distinct last-decile episodes, three observations 5 raw steps apart per clip',
                    'split_caveat': 'descriptive slice; checkpoint training overlap is not established',
                    'no_training_or_planning': True, 'models': {}}
        np.savez_compressed(OUT/'sample.npz', rows=rows, episodes=episodes, state=state,
                            next_state=nxt, goal=goal, angle_path=paths, **features)
        with (OUT/'physical_states.csv').open('w') as out:
            writer = csv.writer(out)
            writer.writerow(['row', 'episode', 'reference_goal_row', *features])
            writer.writerows(zip(rows, episodes, rows+25, *features.values()))
        embeddings = {}
        preprocessor = get_img_preprocessor('pixels', 'pixels', 224)
        for key, name in MODELS.items():
            checkpoint = CKPT_ROOT/name/'weights.pt'
            identity = {'weights_sha256': sha256(checkpoint),
                        'config_sha256': sha256(checkpoint.parent/'config.json'),
                        'checkpoint': str(checkpoint), 'row_sha256': manifest['row_sha256'],
                        'state_sha256': manifest['state_sha256'], 'dataset_mtime_ns': manifest['dataset_mtime_ns'],
                        'preprocess_source_sha256': sha256(ROOT/'slewm/training/utilities.py')}
            manifest['models'][key] = identity
            cache, stamp = OUT/f'emb_{key}.npy', OUT/f'emb_{key}.json'
            if cache.exists() and stamp.exists() and json.loads(stamp.read_text()) == identity:
                embeddings[key] = np.load(cache)
                print(f'{key}: verified cache', flush=True)
                continue
            model = swm.wm.utils.load_pretrained(str(checkpoint)).eval()
            batches = []
            with torch.inference_mode():
                for start in range(0, len(rows), 16):
                    px = np.stack([f['pixels'][int(r)] for r in rows[start:start+16]])
                    px = torch.from_numpy(px).permute(0, 3, 1, 2).contiguous()
                    px = preprocessor({'pixels': px})['pixels']
                    emb = model.encode({'pixels': px[:, None]})['emb'][:, 0]
                    batches.append(emb.float().numpy())
                    if start % 320 == 0:
                        print(f'{key}: {min(start+16, len(rows))}/{len(rows)} frames (CPU)', flush=True)
            embeddings[key] = np.concatenate(batches)
            assert embeddings[key].shape == (len(rows), 192)
            assert np.isfinite(embeddings[key]).all()
            np.save(cache, embeddings[key]); stamp.write_text(json.dumps(identity, indent=2)+'\n')
            del model
    render(features, embeddings, manifest)
    print('Atlas complete:', OUT, flush=True)


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
    from scripts.archive_embedding_round import archive_round
    with archive_round('pusht_atlas'):
        main()

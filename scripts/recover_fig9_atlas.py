#!/usr/bin/env python3
"""Recover original Fig. 9 points from archived Claude caches and recolour them.

Never overwrites the original figure or its quantitative results. No encoder,
training, simulator or GPU is needed. Original PCA defaults resolve to
covariance_eigh in sklearn 1.7.2; persist the resulting coordinates for reuse.
"""
from pathlib import Path
import csv
import json
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import pusht_physical_atlas as atlas

ROOT = atlas.ROOT
OUT = ROOT/'outputs/fig9_original_recovery'


def original_rows(lengths, offsets):
    eps = np.arange(round(len(lengths)*.9), len(lengths))
    eligible = eps[lengths[eps] > 21]
    rng = np.random.default_rng(3072)
    episodes = rng.choice(eligible, size=1000, replace=True)
    starts = np.array([rng.integers(0, int(lengths[e])-21) for e in episodes])
    rows = (offsets[episodes, None]+starts[:, None]+np.array([0, 5, 10])).ravel()
    return rows, np.repeat(episodes, 3), starts


def reference_goals(all_state, rows, episodes, lengths, offsets):
    valid = rows+25 < offsets[episodes]+lengths[episodes]
    goal = np.full((len(rows), all_state.shape[1]), np.nan, dtype=all_state.dtype)
    goal[valid] = all_state[rows[valid]+25]
    return goal, valid


def main():
    import h5py
    import sklearn
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import trustworthiness
    sys.path.insert(0, str(atlas.STYLE))
    import latent_metrics as old

    atlas.OUT = OUT/'atlas'
    atlas.OUT.mkdir(parents=True, exist_ok=True)
    old_rows = list(csv.DictReader((OUT/'original/fig9_latent.csv').open()))
    with h5py.File(atlas.DATA, 'r') as f:
        lengths, offsets = np.asarray(f['ep_len']), np.asarray(f['ep_offset'])
        rows, episodes, starts = original_rows(lengths, offsets)
        all_state = np.asarray(f['state'])
        assert np.array_equal(np.asarray(f['episode_idx'])[rows], episodes)
        assert np.all(rows+5 < offsets[episodes]+lengths[episodes])
        goal, valid = reference_goals(all_state, rows, episodes, lengths, offsets)
        state, nxt = all_state[rows], all_state[rows+5]
        paths = all_state[rows[:, None]+np.arange(6), 4]
    features = atlas.physical_features(state, nxt, goal, paths)
    np.savez_compressed(atlas.OUT/'sample.npz', rows=rows, episodes=episodes,
                        starts=starts, state=state, next_state=nxt, goal=goal,
                        goal_valid=valid, angle_path=paths, **features)
    with (atlas.OUT/'physical_states.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['point_id', 'row', 'episode', 'reference_goal_row', *features])
        writer.writerows(zip(np.arange(len(rows)), rows, episodes,
                             np.where(valid, rows+25, -1), *features.values()))
    st = np.column_stack([state[:, :4], np.cos(state[:, 4]), np.sin(state[:, 4])])
    stz = StandardScaler().fit_transform(st)
    embeddings, frozen, checks = {}, {}, {}
    for key in ('lewm', 'ours'):
        cache = OUT/f'cache/pusht_{key}.npz'
        enc = dict(np.load(cache))
        z = enc['emb'].reshape(-1, 192)
        embeddings[key] = z
        pca_file = atlas.OUT/f'pca_{key}.npz'
        if pca_file.exists():
            frozen[key] = dict(np.load(pca_file))
        else:
            pca = PCA(2, svd_solver='covariance_eigh').fit(z)
            frozen[key] = dict(coordinates=pca.transform(z), components=pca.components_,
                               mean=pca.mean_, explained_variance_ratio=pca.explained_variance_ratio_)
        saved = frozen[key]
        np.testing.assert_allclose((z-saved['mean']) @ saved['components'].T,
                                   saved['coordinates'], atol=1e-5)
        expected = next(r for r in old_rows if r['task']=='pusht' and
                        r['model']==('LeWM' if key=='lewm' else 'ours'))
        checks[key] = dict(
            cache_sha256=atlas.sha256(cache),
            effective_rank=old.effective_rank(z),
            knn_overlap_at20=old.knn_overlap(enc['emb'][:, -1], stz.reshape(1000,3,6)[:, -1], 20),
            trustworthiness_at20=float(trustworthiness(stz.reshape(1000,3,6)[:, -1], enc['emb'][:, -1], n_neighbors=20)),
            spearman_rho_latent_vs_state=old.spearman_dist(enc['emb'][:, -1], stz.reshape(1000,3,6)[:, -1]),
            pc1_pc2_var=float(saved['explained_variance_ratio'].sum()),
            action_sensitivity=float(np.linalg.norm(enc['p_true']-enc['p_cf'],axis=1).mean()))
        for metric, actual in checks[key].items():
            if metric == 'cache_sha256':
                continue
            tolerance = .00051 if metric=='effective_rank' else .000051
            assert abs(actual-float(expected[metric])) < tolerance, (key, metric, actual, expected[metric])
        np.save(atlas.OUT/f'emb_{key}.npy', z)
        with (atlas.OUT/f'coordinates_{key}.csv').open('w') as f:
            writer = csv.writer(f); writer.writerow(['point_id','row','episode','pc1','pc2'])
            writer.writerows(zip(np.arange(len(rows)),rows,episodes,*saved['coordinates'].T))
        print(key, checks[key], flush=True)
    manifest = dict(version=2, source='recovered original fig9 Claude cache',
                    n_frames=len(rows), n_distinct_episodes=len(np.unique(episodes)),
                    sampling='original seed3072: 1000 clips sampled with replacement, 3 frames per clip',
                    row_sha256=atlas.hashlib.sha256(rows.tobytes()).hexdigest(),
                    original_pdf_sha256=atlas.sha256(OUT/'original/fig9_latent.pdf'),
                    dataset=str(atlas.DATA), dataset_mtime_ns=atlas.DATA.stat().st_mtime_ns,
                    goal_valid=int(valid.sum()), goal_missing=int((~valid).sum()),
                    goal_contract='same episode t+25; unavailable values are NaN and drawn grey, never resampled',
                    sklearn_version=sklearn.__version__, pca_solver='covariance_eigh',
                    pca_recovery='2D coordinates were not cached originally; regenerated from original embeddings using original resolved PCA solver',
                    no_training_or_planning=True, original_metrics_checks=checks,
                    split_caveat='last decile is not verified disjoint from checkpoint training',
                    cache_caveat='original caches contain no checkpoint hashes; identity supported by original run log and reproduced metrics')
    atlas.render(features, embeddings, manifest, frozen_pca=frozen)
    print('Saved original points and recoloured atlas:', atlas.OUT, flush=True)


if __name__ == '__main__':
    from scripts.archive_embedding_round import archive_round
    with archive_round('recover_fig9_atlas'):
        main()

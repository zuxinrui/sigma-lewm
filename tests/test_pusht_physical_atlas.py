import numpy as np

from scripts.pusht_physical_atlas import physical_features, sample_rows, wrapped_angle


def test_angle_seam_and_rotation_direction():
    state = np.zeros((3, 7))
    paths = np.deg2rad([[358, 359, 0, 1, 2, 3], [3, 2, 1, 0, 359, 358], [20]*6])
    state[:, 4] = paths[:, 0]
    nxt = state.copy(); nxt[:, 4] = paths[:, -1]
    goal = state.copy(); goal[:, 2] = 3; goal[:, 3] = 4
    features = physical_features(state, nxt, goal, paths)
    np.testing.assert_allclose(features['angle_change'], [5, -5, 0], atol=1e-9)
    np.testing.assert_array_equal(features['rotation'], [1, -1, 0])
    np.testing.assert_allclose(features['goal_distance'], 5)


def test_rows_do_not_cross_episode_boundary():
    lengths = np.arange(40, 240)
    offsets = np.r_[0, np.cumsum(lengths[:-1])]
    rows, episodes = sample_rows(lengths, offsets, 12)
    assert len(rows) == 36
    assert len(np.unique(episodes)) == 12
    assert np.all(rows+25 < offsets[episodes]+lengths[episodes])
    assert np.all(rows >= offsets[episodes])
    assert np.all(np.diff(rows) > 0)


def test_wrap_has_no_spurious_full_rotation():
    np.testing.assert_allclose(wrapped_angle(np.array([2*np.pi-.1, -2*np.pi+.1])), [-.1, .1])


def test_original_sampling_matches_old_algorithm_without_resampling():
    from scripts.recover_fig9_atlas import original_rows
    lengths = np.arange(20, 220)
    offsets = np.r_[0, np.cumsum(lengths[:-1])]
    rows, episodes, starts = original_rows(lengths, offsets)
    rng = np.random.default_rng(3072)
    candidates = np.arange(round(len(lengths)*.9), len(lengths))
    picked = rng.choice(candidates[lengths[candidates]>21], size=1000, replace=True)
    expected_starts = np.array([rng.integers(0, int(lengths[e])-21) for e in picked])
    np.testing.assert_array_equal(episodes, np.repeat(picked, 3))
    np.testing.assert_array_equal(starts, expected_starts)
    np.testing.assert_array_equal(rows.reshape(-1, 3)[:, 0], offsets[picked]+expected_starts)
    assert len(rows)==3000


def test_missing_goal_never_uses_next_episode_and_keeps_point():
    from scripts.recover_fig9_atlas import reference_goals
    state = np.arange(80*7, dtype=float).reshape(80, 7)
    rows = np.array([0, 14, 15, 39, 40, 55])
    ids = np.array([0, 0, 0, 0, 1, 1])
    goal, valid = reference_goals(state, rows, ids, np.array([40, 40]), np.array([0, 40]))
    np.testing.assert_array_equal(valid, [True, True, False, False, True, False])
    assert len(goal)==len(rows)
    assert np.isnan(goal[~valid]).all()
    np.testing.assert_array_equal(goal[valid], state[rows[valid]+25])


def test_gear_quaternion_yaw_order_and_sign_invariance():
    from scripts.cube_gear_physical_atlas import quat_yaw, rotation_features
    angle=np.deg2rad(np.array([0, 90, 179, -179]))
    q=np.column_stack([np.cos(angle/2),np.zeros(4),np.zeros(4),np.sin(angle/2)])
    np.testing.assert_allclose(quat_yaw(q),angle,atol=1e-9)
    np.testing.assert_allclose(quat_yaw(-q),angle,atol=1e-9)
    delta,direction=rotation_features(np.deg2rad([[178,179,-180,-179,-178,-177]]))
    np.testing.assert_allclose(delta,[5],atol=1e-9)
    np.testing.assert_array_equal(direction,[1])


def test_theme_scale_endpoints_and_cyclic_continuity():
    import json
    from matplotlib.colors import to_rgba
    from scripts.pusht_physical_atlas import theme_colormap, STYLE
    p=json.loads((STYLE/'palette.json').read_text())
    np.testing.assert_allclose(theme_colormap('angle',p)(0.),theme_colormap('angle',p)(1.))
    np.testing.assert_allclose(theme_colormap('signed',p)(0.),to_rgba(p['backward']))
    np.testing.assert_allclose(theme_colormap('signed',p)(1.),to_rgba(p['inverse']))

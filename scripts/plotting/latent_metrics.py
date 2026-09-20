"""Latent-geometry metrics: effective rank, kNN overlap, distance rank correlation."""
import numpy as np

SEED = 3072


def effective_rank(z: np.ndarray) -> float:
    x = z.reshape(-1, z.shape[-1]).astype(np.float64)
    x = x - x.mean(0, keepdims=True)
    cov = x.T @ x / max(len(x) - 1, 1)
    ev = np.clip(np.linalg.eigvalsh(cov), 0, None)
    p = ev / max(ev.sum(), 1e-12)
    ent = -(p * np.log(np.clip(p, 1e-12, None))).sum()
    return float(np.exp(ent))


def knn_sets(x: np.ndarray, k: int) -> np.ndarray:
    from scipy.spatial.distance import cdist

    d = cdist(x, x)
    np.fill_diagonal(d, np.inf)
    return np.argpartition(d, k, axis=1)[:, :k]


def knn_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    na, nb = knn_sets(a, k), knn_sets(b, k)
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(na, nb)]))


def spearman_dist(a: np.ndarray, b: np.ndarray, n_pairs: int = 200_000) -> float:
    from scipy.stats import spearmanr

    rng = np.random.default_rng(SEED)
    i = rng.integers(0, len(a), n_pairs)
    j = rng.integers(0, len(a), n_pairs)
    m = i != j
    i, j = i[m], j[m]
    da = np.linalg.norm(a[i] - a[j], axis=1)
    db = np.linalg.norm(b[i] - b[j], axis=1)
    return float(spearmanr(da, db).statistic)

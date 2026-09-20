"""Data split support for the archived multistep recipes."""
import hashlib
import torch


def _episode_level_split(dataset, train_fraction, generator):
    """Split clip indices by episode, never by overlapping window.

    Returns ``(train_subset, val_subset, train_episode_ids, val_episode_ids)``.
    The dataset transform remains shared, while no internal episode contributes
    a clip to both subsets.
    """
    if not hasattr(dataset, "clip_indices"):
        raise TypeError("episode-level split requires dataset.clip_indices")
    episode_ids = sorted({int(ep) for ep, _ in dataset.clip_indices})
    if len(episode_ids) < 2:
        raise ValueError("episode-level split requires at least two usable episodes")
    train_fraction = float(train_fraction)
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie strictly between zero and one")

    order = torch.randperm(len(episode_ids), generator=generator).tolist()
    n_train = min(
        len(episode_ids) - 1,
        max(1, int(len(episode_ids) * train_fraction)),
    )
    train_episodes = sorted(episode_ids[i] for i in order[:n_train])
    val_episodes = sorted(episode_ids[i] for i in order[n_train:])
    train_episode_set = set(train_episodes)
    train_indices = [
        idx for idx, (ep, _) in enumerate(dataset.clip_indices)
        if int(ep) in train_episode_set
    ]
    val_indices = [
        idx for idx, (ep, _) in enumerate(dataset.clip_indices)
        if int(ep) not in train_episode_set
    ]
    if not train_indices or not val_indices:
        raise RuntimeError("episode-level split produced an empty window subset")
    return (
        torch.utils.data.Subset(dataset, train_indices),
        torch.utils.data.Subset(dataset, val_indices),
        train_episodes,
        val_episodes,
    )

def _state_dict_sha256(model, excluded_prefixes=()):
    """Hash deterministic initial shared parameters without checkpoint I/O."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if any(name.startswith(prefix) for prefix in excluded_prefixes):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()

"""Distributed support for the archived multistep recipes."""
import os
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn


def _prelaunch_global_rank(environ=None):
    """Resolve rank before Lightning initializes the process group."""
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    environ = os.environ if environ is None else environ
    # Native srun owns rank assignment whenever SLURM_PROCID is present.  Give
    # it priority over stale RANK/LOCAL_RANK values that `sbatch --export=ALL`
    # could otherwise carry in from the submission shell.
    slurm_rank = environ.get("SLURM_PROCID")
    if slurm_rank is not None:
        try:
            return int(slurm_rank)
        except (TypeError, ValueError) as error:
            raise ValueError(f"malformed SLURM_PROCID={slurm_rank!r}") from error
    for key in ("RANK", "LOCAL_RANK", "JSM_NAMESPACE_RANK"):
        value = environ.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0

def _collective_device(device):
    """Pick a device the initialized process group can actually reduce on.

    2026-08-24 (Snellius): the execution audit runs from on_fit_end, by which point
    Lightning has moved the module back to CPU, so pl_module.device is "cpu".  A
    NCCL-only process group has no backend registered for CPU tensors and
    all_reduce dies with "No backend type associated with device type cpu".  The
    local 2x3090 box did not hit this because its group also carried gloo.  Reduce
    on CUDA whenever the group is NCCL and a GPU is visible; otherwise fall back to
    the caller's device so CPU-only and gloo runs are unaffected.
    """
    try:
        if (dist.is_available() and dist.is_initialized()
                and torch.cuda.is_available() and dist.get_backend() == "nccl"):
            return torch.device("cuda", torch.cuda.current_device())
    except Exception:
        pass
    return device

def _dist_active():
    """Return whether this process is participating in initialized DDP."""
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

def _all_gather_with_grad(x):
    """Concatenate equal-shaped tensors from every rank without cutting gradients."""
    if not _dist_active():
        return x
    return torch.cat(dist_nn.all_gather(x), dim=0)

def _all_gather_no_grad(x):
    """Concatenate equal-shaped non-trainable tensors from every rank."""
    if not _dist_active():
        return x
    gathered = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, x.contiguous())
    return torch.cat(gathered, dim=0)

def _shared_randint(low, high, device, enabled=False):
    """Sample one integer on rank zero and broadcast it to all ranks."""
    if not (enabled and _dist_active()):
        return int(torch.randint(low, high, (1,), device=device).item())
    value = torch.empty(1, dtype=torch.long, device=device)
    if dist.get_rank() == 0:
        value.random_(low, high)
    dist.broadcast(value, src=0)
    return int(value.item())

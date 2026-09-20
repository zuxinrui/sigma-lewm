"""CPU tests for the cross-step GC-IDM heads (GC-IDM arXiv:2605.08732).
Validates: cross-step sampling logic/range/uniformity, both heads' shapes,
encoder-shaping gradient (our no-sg deviation), horizon sensitivity."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch, collections
from trim.planning.gcidm import GCIDMHead, FlowGCIDMHead, sample_crossstep

torch.manual_seed(0)
print("=== cross-step sampler: h~U[1,h_max], t~U[0,T-1-h] (per paper) ===")
T, B = 8, 20000
t, h = sample_crossstep(T, B, h_max=7, device="cpu")
assert h.min() >= 1 and h.max() <= T - 1, (h.min().item(), h.max().item())
assert (t >= 0).all() and ((t + h) <= T - 1).all(), "t+h OUT OF RANGE"
c = collections.Counter(h.tolist())
print(" h dist:", {k: round(v / B, 3) for k, v in sorted(c.items())})
print(" t+h max =", (t + h).max().item(), "<= T-1 =", T - 1, " OK")

print("=== GCIDMHead (paper-faithful MLP) ===")
g = GCIDMHead(D=192, A=10)
zt = torch.randn(64, 192, requires_grad=True); zg = torch.randn(64, 192)
hh = torch.randint(1, 8, (64,))
a = g(zt, zg, hh); assert a.shape == (64, 10)
(a - torch.randn(64, 10)).pow(2).mean().backward()
assert zt.grad.abs().sum() > 0, "NO encoder-shaping grad!"
print(f" out{tuple(a.shape)} grad->enc={zt.grad.abs().sum():.2f} "
      f"params={sum(p.numel() for p in g.parameters())/1e6:.2f}M")

print("=== FlowGCIDMHead (ACID arch x cross-step) ===")
f = FlowGCIDMHead(D=192, A=10)
zt2 = torch.randn(32, 192, requires_grad=True); zg2 = torch.randn(32, 192)
x = torch.randn(32, 10); tau = torch.rand(32, 1); h2 = torch.randint(1, 8, (32,))
v = f(x, tau, zt2, zg2, h2); assert v.shape == (32, 10)
v.sum().backward(); assert zt2.grad.abs().sum() > 0, "NO encoder-shaping grad!"
print(f" v{tuple(v.shape)} grad->enc={zt2.grad.abs().sum():.2f} "
      f"params={sum(p.numel() for p in f.parameters())/1e6:.2f}M")
f.eval()
with torch.no_grad():
    v1 = f(x, tau, zt2.detach(), zg2, torch.ones(32, dtype=torch.long))
    v7 = f(x, tau, zt2.detach(), zg2, torch.full((32,), 7, dtype=torch.long))
d = (v1 - v7).abs().mean().item()
assert d > 0, "flow head ignores horizon!"
print(f" horizon sensitivity |v(h=1)-v(h=7)| = {d:.4f} OK")
print("ALL GCIDM TESTS PASSED")

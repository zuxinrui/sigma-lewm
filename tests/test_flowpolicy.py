"""CPU tests for the Qantara-style flow-matching policy head.
The load-bearing property: it must condition on the PAST ONLY (z_<=t) — if it could see
z_{t+1} it would be just another co-trained verifier (our law #9) and buy nothing."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from trim.planning.flowpolicy import FlowPolicyHead, flowpolicy_loss

torch.manual_seed(0)

def test_shapes_and_encoder_grad():
    h = FlowPolicyHead(D=192, A=10, ctx=3)
    z = torch.randn(8, 3, 192, requires_grad=True)
    x, tau = torch.randn(8, 10), torch.rand(8, 1)
    v = h(x, tau, z); assert v.shape == (8, 10), v.shape
    v.sum().backward()
    assert z.grad is not None and z.grad.abs().sum() > 0, "no encoder-shaping gradient"
    print(f"PASS shapes {tuple(v.shape)}, grad->encoder={z.grad.abs().sum():.2f}, "
          f"params={sum(p.numel() for p in h.parameters())/1e6:.2f}M")

def test_past_only_no_future_leakage():
    """THE decisive test: the loss for a_t must not depend on z_{t+1}."""
    h = FlowPolicyHead(D=192, A=10, ctx=3).eval()
    emb = torch.randn(4, 6, 192, requires_grad=True)
    a = torch.randn(4, 6, 10)
    torch.manual_seed(1); loss, _ = flowpolicy_loss(h, emb, a)
    loss.backward()
    g = emb.grad.abs().sum(dim=(0, 2))          # per-timestep gradient
    # windows cover t=ctx-1..T-2 -> z_0..z_{T-2} get gradient; the LAST latent z_{T-1} never does
    assert g[-1].item() == 0.0, f"z_{{T-1}} (the future) received gradient {g[-1]} — LEAKAGE!"
    assert g[:-1].abs().sum() > 0, "no gradient anywhere"
    print(f"PASS past-only: per-t grad {[round(x,2) for x in g.tolist()]} — last (future) is exactly 0")

def test_sample_roundtrip():
    h = FlowPolicyHead(D=192, A=10, ctx=3)
    a = h.sample(torch.randn(5, 3, 192))
    assert a.shape == (5, 10), a.shape
    print(f"PASS 1-step Euler sample {tuple(a.shape)}")

if __name__ == "__main__":
    test_shapes_and_encoder_grad(); test_past_only_no_future_leakage(); test_sample_roundtrip()
    print("\nALL FLOWPOLICY TESTS PASSED")

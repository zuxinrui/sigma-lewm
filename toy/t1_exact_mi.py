"""T1: exact one-vs-two mutual informations on discrete (Z, A, Z') pmfs (no learning).

I_F = I(Z'; Z, A), I_B = I(Z; Z', A), I_A = I(A; Z, Z'); TC, DTC; identity I_F+I_B+I_A = TC+DTC.
"""
import csv
import os

import numpy as np

from common import ACTION, BACKWARD, FORWARD, GREY, INK, INVERSE, LIGHT, MUTED, ICLR_W, plt, savefig, style

HERE = os.path.dirname(os.path.abspath(__file__))
LOG2 = np.log(2.0)


def H(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / LOG2)


def mi_terms(P):
    """P: array [nz, na, nz'] joint pmf. Returns dict of bits."""
    P = P / P.sum()
    Hz = H(P.sum((1, 2)))
    Ha = H(P.sum((0, 2)))
    Hz2 = H(P.sum((0, 1)))
    Hza = H(P.sum(2))
    Haz2 = H(P.sum(0))
    Hzz2 = H(P.sum(1))
    Hall = H(P)
    I_F = Hz2 + Hza - Hall
    I_B = Hz + Haz2 - Hall
    I_A = Ha + Hzz2 - Hall
    TC = Hz + Ha + Hz2 - Hall
    DTC = Hza + Haz2 + Hzz2 - 2 * Hall
    # residual conditional entropies: what each loss actually minimises once SIGReg pins the marginals
    res_F = Hall - Hza      # H(Z' | Z, A)
    res_B = Hall - Haz2     # H(Z  | Z', A)
    res_A = Hall - Hzz2     # H(A  | Z, Z')
    sat = lambda num, den: (num / den) if den > 1e-12 else float("nan")
    return dict(I_F=I_F, I_B=I_B, I_A=I_A, TC=TC, DTC=DTC, H_Z=Hz, H_A=Ha, H_Z2=Hz2,
                H_Z2_given_ZA=res_F, H_Z_given_Z2A=res_B, H_A_given_ZZ2=res_A,
                sat_F=sat(I_F, Hz2), sat_B=sat(I_B, Hz), sat_A=sat(I_A, Ha),
                identity_gap=I_F + I_B + I_A - (TC + DTC))


def deterministic(nz, na, f, pz=None, pa=None, nz2=None):
    """Joint pmf for z' = f(z, a) with Z, A independent."""
    pz = np.full(nz, 1 / nz) if pz is None else pz
    pa = np.full(na, 1 / na) if pa is None else pa
    outs = {}
    for z in range(nz):
        for a in range(na):
            outs.setdefault(f(z, a), len(outs))
    nz2 = len(outs) if nz2 is None else nz2
    P = np.zeros((nz, na, nz2))
    for z in range(nz):
        for a in range(na):
            P[z, a, outs[f(z, a)]] += pz[z] * pa[a]
    return P


def cases(n=4):
    C = {}
    # (i) group / XOR dynamics z' = z + a mod n
    C["(i) group  z'=z⊕a"] = deterministic(n, n, lambda z, a: (z + a) % n)
    # (ii) autonomous bijective flow z' = f(z), A independent
    perm = [1, 2, 3, 0]
    C["(ii) autonomous flow  z'=f(z)"] = deterministic(n, n, lambda z, a: perm[z])
    # (iii) teleport z' = a
    C["(iii) teleport  z'=a"] = deterministic(n, n, lambda z, a: a)
    # (iv) noise embedding z' = (z⊕a, ε), ε fresh uniform on m symbols
    m = 4
    P = np.zeros((n, n, n * m))
    for z in range(n):
        for a in range(n):
            for e in range(m):
                P[z, a, ((z + a) % n) * m + e] += 1 / (n * n * m)
    C["(iv) noise-embedded  z'=(z⊕a, ε)"] = P
    # (v) PI example: Z' constant, A two clusters, Z aligned with A (z=a) or independent
    P = np.zeros((2, 2, 1))
    for z in range(2):
        for a in range(2):
            P[z, a, 0] = 0.5 * (1.0 if z == a else 0.0)
    C["(v-a) static z', z aligned with a"] = P
    P = np.zeros((2, 2, 1))
    for z in range(2):
        for a in range(2):
            P[z, a, 0] = 0.5 * (0.9 if z == a else 0.1)
    C["(v-b) static z', z 90% aligned"] = P
    P = np.full((2, 2, 1), 0.25)
    C["(v-c) static z', z indep. of a"] = P
    # (vi) merging dynamics: z' = max(z, a); wall: z' = clip(z + a, 0, n-1) with a in {-1,0,+1}
    C["(vi-a) merge  z'=max(z,a)"] = deterministic(n, n, lambda z, a: max(z, a))
    C["(vi-b) wall  z'=clip(z+a)"] = deterministic(n, 3, lambda z, a: int(np.clip(z + a - 1, 0, n - 1)))
    return C


def main():
    style()
    rng = np.random.default_rng(0)
    C = cases()
    rows = []
    for name, P in C.items():
        t = mi_terms(P)
        rows.append(dict(case=name, **t))
    # random pmfs: verify the identity
    gaps = []
    for _ in range(200):
        shape = tuple(rng.integers(2, 6, size=3))
        P = rng.gamma(0.5, size=shape)
        gaps.append(abs(mi_terms(P)["identity_gap"]))
    max_gap = max(gaps)
    max_gap_cases = max(abs(r["identity_gap"]) for r in rows)
    with open(os.path.join(HERE, "t1_counterexamples.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
        w.writerow(dict(case=f"random pmfs (200): max |I_F+I_B+I_A-(TC+DTC)| = {max_gap:.2e}",
                        **{k: "" for k in rows[0] if k != "case"}))
    print(f"identity gap: cases {max_gap_cases:.2e}, random {max_gap:.2e}")
    for r in rows:
        print(f"{r['case']:38s} I_F={r['I_F']:.3f} I_B={r['I_B']:.3f} I_A={r['I_A']:.3f} "
              f"TC={r['TC']:.3f} DTC={r['DTC']:.3f} | resid F/B/A={r['H_Z2_given_ZA']:.3f}/{r['H_Z_given_Z2A']:.3f}/{r['H_A_given_ZZ2']:.3f}")

    # ---------------- figure: left = the three MIs (bits); right = residual conditional entropies
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(ICLR_W, 3.4), gridspec_kw=dict(width_ratios=[1.45, 1.0]), sharey=True)
    names = [r["case"] for r in rows]
    y = np.arange(len(rows))[::-1]
    h = 0.26
    spec = [(h, "I_F", "H_Z2_given_ZA", FORWARD, "$I_F=I(Z';Z,A)$", "$H(Z'\\mid Z,A)$"),
            (0, "I_B", "H_Z_given_Z2A", BACKWARD, "$I_B=I(Z;Z',A)$", "$H(Z\\mid Z',A)$"),
            (-h, "I_A", "H_A_given_ZZ2", INVERSE, "$I_A=I(A;Z,Z')$", "$H(A\\mid Z,Z')$")]
    for off, key, rkey, col, lab, rlab in spec:
        vals = np.array([r[key] for r in rows])
        ax.barh(y + off, vals, height=h * 0.92, color=col, edgecolor="none", label=lab)
        for yi, v in zip(y + off, vals):
            ax.text(max(v, 0) + 0.05, yi, f"{v:.2f}", va="center", ha="left", fontsize=6, color=INK)
        res = np.array([r[rkey] for r in rows])
        ax2.barh(y + off, res, height=h * 0.92, color=col, edgecolor="none", label=rlab)
        for yi, v in zip(y + off, res):
            ax2.text(max(v, 0) + 0.05, yi, f"{v:.2f}", va="center", ha="left", fontsize=6, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("mutual information (bits)")
    ax.set_xlim(0, 2.9)
    ax2.set_xlabel("residual entropy (bits)")
    ax2.set_xlim(0, 2.6)
    for a_ in (ax, ax2):
        a_.xaxis.grid(True, color=LIGHT, lw=0.5)
        a_.set_axisbelow(True)
        a_.tick_params(axis="y", length=0)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.55, -0.10), ncol=3, frameon=False, fontsize=6.3,
               columnspacing=1.6, handlelength=1.2)
    ax.set_title("one-vs-two views of (Z, A, Z')", loc="left", fontsize=7)
    ax2.set_title("what each loss minimises", loc="left", fontsize=7)
    fig.text(0.0, -0.15, "Z, A uniform on 4 symbols ((v): 2 symbols); (iv) has H(Z')=4 bits, so I_F=2 is only half-saturated: "
             "H(Z'|Z,A)=2 bits of fresh noise.", fontsize=6, color=MUTED)
    fig.tight_layout()
    savefig(fig, os.path.join(HERE, "t1_counterexamples"))


if __name__ == "__main__":
    main()

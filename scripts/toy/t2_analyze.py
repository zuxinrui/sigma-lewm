"""T2 aggregation: runs/*.json -> t2_results.csv, t2_probe_matrix, t2_rollout_error, t2_planning."""
import csv
import glob
import json
import os

import numpy as np

from common import (ARM_COLORS, ARM_LABEL, ARM_ORDER, INK, LIGHT, MUTED, ICLR_W, TEAL, TEAL_CMAP,
                    plt, savefig, style)
from t2_toy_world import RUNS_DIR, Z_DIM, eval_planning_reference

HERE = os.path.dirname(os.path.abspath(__file__))
SUF = "" if Z_DIM == 8 else f"_d{Z_DIM}"
PM = "t2_probe_matrix" if Z_DIM == 8 else ("t2b_probe_matrix" if Z_DIM == 4 else f"t2b_probe_matrix_d{Z_DIM}")


def main():
    style()
    runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(HERE, RUNS_DIR, "*.json")))]
    ref_path = os.path.join(HERE, "t2_planning_reference.json")
    ref = eval_planning_reference() if (Z_DIM == 8 or not os.path.exists(ref_path)) else json.load(open(ref_path))
    json.dump(ref, open(os.path.join(HERE, f"t2_planning_reference{SUF}.json"), "w"), indent=1)
    print("planning reference:", ref)
    keys = ["lin_r2_p", "lin_r2_o", "lin_r2_d", "lin_r2_n", "mlp_r2_p", "mlp_r2_o", "mlp_r2_d", "mlp_r2_n",
            "lin_r2_action", "mlp_r2_action", "action_sensitivity", "rankme", "plan_success", "plan_final_dist"]
    # per-run CSV
    with open(os.path.join(HERE, f"t2_runs{SUF}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "seed"] + keys + [f"rollout_rmse_h{h}" for h in range(21)])
        for r in runs:
            w.writerow([r["arm"], r["seed"]] + [r[k] for k in keys] + r["rollout_rmse"])
    # summary CSV (mean +- sd over seeds)
    summ = {}
    for arm in ARM_ORDER:
        rs = [r for r in runs if r["arm"] == arm]
        if not rs:
            continue
        summ[arm] = {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs]))) for k in keys}
        R = np.array([r["rollout_rmse"] for r in rs])
        summ[arm]["rollout"] = (R.mean(0), R.std(0))
        summ[arm]["copy"] = np.array([r["copy_rmse"] for r in rs]).mean(0)
        summ[arm]["n"] = len(rs)
    with open(os.path.join(HERE, f"t2_results{SUF}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "n_seeds"] + [f"{k}_{s}" for k in keys for s in ("mean", "sd")]
                   + [f"rollout_rmse_h{h}_mean" for h in (1, 5, 10, 20)])
        for arm, d in summ.items():
            w.writerow([arm, d["n"]] + [f"{d[k][i]:.4f}" for k in keys for i in (0, 1)]
                       + [f"{d['rollout'][0][h]:.4f}" for h in (1, 5, 10, 20)])
    # markdown table for the report
    lines = ["| arm | R² p | R² o | R² d | R² n | R² a (mlp) | act.sens. | RankMe | plan succ. | roll h1 | roll h10 | roll h20 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm, d in summ.items():
        f = lambda k: f"{d[k][0]:.2f}±{d[k][1]:.2f}"
        lines.append(f"| {arm} | {f('lin_r2_p')} | {f('lin_r2_o')} | {f('lin_r2_d')} | {f('lin_r2_n')} | {f('mlp_r2_action')} | "
                     f"{f('action_sensitivity')} | {f('rankme')} | {f('plan_success')} | {d['rollout'][0][1]:.3f} | "
                     f"{d['rollout'][0][10]:.3f} | {d['rollout'][0][20]:.3f} |")
    lines.append(f"| oracle (true-state CEM) | | | | | | | | {ref['oracle']:.2f} | | | |")
    lines.append(f"| random actions | | | | | | | | {ref['random']:.2f} | | | |")
    open(os.path.join(HERE, f"t2_table{SUF}.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    arms = [a for a in ARM_ORDER if a in summ]
    # ---------------- probe matrix (+ CSV of exactly what is plotted)
    with open(os.path.join(HERE, PM + ".csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "n_seeds", "lin_r2_p", "lin_r2_o", "lin_r2_d", "lin_r2_n", "mlp_r2_action", "action_sensitivity", "rankme",
                    "rollout_rmse_h0", "rollout_rmse_h10", "sd_lin_r2_p", "sd_lin_r2_o", "sd_lin_r2_d", "sd_mlp_r2_action"])
        for a in arms:
            d = summ[a]
            w.writerow([a, d["n"]] + [f"{d[k][0]:.4f}" for k in ("lin_r2_p", "lin_r2_o", "lin_r2_d", "lin_r2_n", "mlp_r2_action", "action_sensitivity", "rankme")]
                       + [f"{d['rollout'][0][0]:.4f}", f"{d['rollout'][0][10]:.4f}"]
                       + [f"{d[k][1]:.4f}" for k in ("lin_r2_p", "lin_r2_o", "lin_r2_d", "mlp_r2_action")])
    fig, ax = plt.subplots(figsize=(ICLR_W, 2.1))
    cols = ["lin_r2_p", "lin_r2_o", "lin_r2_d", "lin_r2_n", "mlp_r2_action", "action_sensitivity"]
    col_lab = ["agent $p$", "object $o$", "distractor $d$", "noise $n$", "action $a$\nfrom $(z,z')$", "action\nsensitivity"]
    M = np.array([[summ[a][c][0] for c in cols] for a in arms])
    Msd = np.array([[summ[a][c][1] for c in cols] for a in arms])
    show = M.copy()
    show[:, 5] = 0.5                                   # sensitivity column: one flat mid tone (values nearly equal)
    r2_cells = show[:, :5][show[:, :5] > 0.05]         # colour range spans the R^2 values actually shown
    vmin = float(np.floor((r2_cells.min() - 0.05) * 10) / 10) if r2_cells.size else 0.0
    im = ax.imshow(show, cmap=TEAL_CMAP, vmin=vmin, vmax=1, aspect="auto")
    for i in range(len(arms)):
        for j in range(len(cols)):
            v = M[i, j]
            txt = f"{v:.2f}" if j < 5 else f"{v:.3f}"
            ax.text(j, i, f"{txt}\n±{Msd[i, j]:.2f}" if j < 5 else txt, ha="center", va="center", fontsize=6,
                    color="white" if (j < 5 and show[i, j] > vmin + 0.9 * (1 - vmin)) else INK)
    # hairline grid so the cells read as a table in print
    ax.set_xticks(np.arange(len(cols) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(arms) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(col_lab, fontsize=6.5)
    ax.set_yticks(range(len(arms))); ax.set_yticklabels([ARM_LABEL[a] for a in arms], fontsize=6.5)
    for lab, a in zip(ax.get_yticklabels(), arms):
        lab.set_color(TEAL if a == "F+B+I+SIG" else INK)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"Held-out linear-probe $R^2$ of the latent, D={Z_DIM} for a 14-D state (8 predictable + 4 noise + 2 agent... see report)" if False else f"Held-out linear-probe $R^2$ of the latent, D={Z_DIM} (mean ± sd over seeds)", loc="left", fontsize=8)
    savefig(fig, os.path.join(HERE, PM))

    # ---------------- rollout error
    fig, ax = plt.subplots(figsize=(ICLR_W, 2.2))
    hs = np.arange(21)
    for a in arms:
        m, sd = summ[a]["rollout"]
        ax.plot(hs, m, color=ARM_COLORS[a], lw=1.6 if a == "F+B+I+SIG" else 1.1, label=ARM_LABEL[a])
        ax.fill_between(hs, m - sd, m + sd, color=ARM_COLORS[a], alpha=0.12, lw=0)
    ax.plot(hs, summ[arms[0]]["copy"], color=MUTED, lw=0.8, ls=":", label="copy $z_0$ (no dynamics)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=6.3)
    ax.set_xlabel("open-loop rollout horizon (steps)")
    ax.set_ylabel("RMSE of probed $[p, o]$ (world units)")
    ax.set_xlim(0, 20)
    ax.yaxis.grid(True, color=LIGHT, lw=0.5); ax.set_axisbelow(True)
    ax.set_title(f"Open-loop latent rollout error, decoded by a fixed linear probe (D={Z_DIM})", loc="left", fontsize=8)
    fig.subplots_adjust(right=0.72)
    savefig(fig, os.path.join(HERE, f"t2_rollout_error{SUF}"))

    # ---------------- planning (D = 8 only; the proxy is uninformative, see report)
    if Z_DIM != 8:
        return
    fig, ax = plt.subplots(figsize=(ICLR_W, 2.0))
    xs = np.arange(len(arms))
    for i, a in enumerate(arms):
        vals = [r["plan_success"] for r in runs if r["arm"] == a]
        ax.bar(i, np.mean(vals), color=ARM_COLORS[a], width=0.66)
        ax.scatter([i] * len(vals), vals, s=8, color=INK, zorder=3)
        ax.text(i, np.mean(vals) + 0.03, f"{100 * np.mean(vals):.0f}%", ha="center", fontsize=6.5, color=INK)
    ax.axhline(ref["oracle"], color=MUTED, lw=0.7, ls="--")
    ax.text(len(arms) - 0.5, ref["oracle"] + 0.015, f"true-state CEM {100 * ref['oracle']:.0f}%", ha="right", fontsize=6, color=MUTED)
    ax.axhline(ref["random"], color=MUTED, lw=0.7, ls=":")
    ax.text(len(arms) - 0.5, ref["random"] + 0.015, f"random actions {100 * ref['random']:.0f}%", ha="right", fontsize=6, color=MUTED)
    ax.set_xticks(xs); ax.set_xticklabels([ARM_LABEL[a] for a in arms], fontsize=6.2, rotation=12)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("success (object within 0.15)")
    ax.set_title("Latent CEM planning, horizon 8, full commit, 100 episodes × 3 seeds", loc="left", fontsize=8)
    savefig(fig, os.path.join(HERE, f"t2_planning{SUF}"))


if __name__ == "__main__":
    main()

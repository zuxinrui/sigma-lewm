"""Turn the micro-benchmark JSONs into the planning-latency table.

    python -m slewm.evaluation.latency_summary <dir-of-micro_*.json>

Per (task, method) it reports the median solve time at each CEM budget; the table averages
those over every task and method, because the planner is the same object for all of them at
test time and only the encoder and predictor weights differ.
"""
import csv
import json
import statistics as st
import sys
from pathlib import Path

TASKS = ["pusht", "cube", "tworoom", "reacher"]
ARMS = ["lewm", "ours", "subjepa", "rcaux"]
CEMS = [30, 8, 4, 2]


def main() -> None:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/latency")
    rows = []
    for t in TASKS:
        for a in ARMS:
            f = d / f"micro_{t}_{a}_c30.json"
            if not f.exists():
                continue
            r = json.loads(f.read_text())
            s, prep = r["solve_ms"], r["prep_ms"]["median"]
            med = {c: s[str(c)]["median"] for c in CEMS}
            tot = {c: med[c] + prep for c in CEMS}          # what the controller waits for
            for c in CEMS:
                rows.append(dict(task=t, arm=a, cem=c, solve_median_ms=med[c], total_ms=tot[c],
                                 control_rate_hz=1e3 / tot[c], measured_speedup=tot[30] / tot[c],
                                 fixed_ms=r["linear_fit"]["fixed_ms"],
                                 per_iteration_ms=r["linear_fit"]["per_iteration_ms"],
                                 gpu=r["meta"].get("gpu")))
    if not rows:
        sys.exit(f"no micro_*.json in {d}")
    with open(d / "latency_single_env.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    runs = len({(r["task"], r["arm"]) for r in rows})
    print(f"{runs} (task, method) runs on {rows[0]['gpu']}\n")
    print(f"{'':28s}" + "".join(f"{'CEM' + str(c):>16s}" for c in CEMS))
    print(f"{'Nominal budget reduction':28s}" + "".join(f"{'x' + format(30 / c, 'g'):>16s}" for c in CEMS))
    for label, key, fmt in (("Solve latency (ms)", "solve_median_ms", "{:.1f} +- {:.1f}"),
                            ("Replanning rate (Hz)", "control_rate_hz", "{:.2f} +- {:.2f}"),
                            ("Measured solve speed-up", "measured_speedup", "x{:.2f} +- {:.2f}")):
        cells = []
        for c in CEMS:
            v = [r[key] for r in rows if r["cem"] == c]
            cells.append(fmt.format(st.mean(v), st.stdev(v) if len(v) > 1 else 0.0))
        print(f"{label:28s}" + "".join(f"{x:>16s}" for x in cells))
    fx = [r["fixed_ms"] for r in rows if r["cem"] == 30]
    pit = [r["per_iteration_ms"] for r in rows if r["cem"] == 30]
    print(f"\nlinear fit: {st.mean(fx):.1f} ms fixed + {st.mean(pit):.1f} ms per CEM iteration")
    print(f"-> {d / 'latency_single_env.csv'}")


if __name__ == "__main__":
    main()

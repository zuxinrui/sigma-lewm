#!/usr/bin/env python3
"""Incrementally summarize and visualize the formal Gear/Franka matrix.

This script is read-only with respect to experiment inputs. It consumes only
hash-committed epoch records/results and writes regenerated tables and figures.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/paper_new_tasks_v1"
ARMS = ("L0", "L1", "C0", "C1")
COLORS = {"L0": "#6b7280", "L1": "#3b82f6", "C0": "#f59e0b", "C1": "#dc2626"}
COMPONENTS = ("teacher", "free_run", "terminal", "action", "sigreg")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def committed_json(path: Path, *, schema: str, count: int) -> dict | None:
    marker = path.with_suffix(path.suffix + ".done")
    if not path.is_file() or not marker.is_file():
        return None
    try:
        done = json.loads(marker.read_text(encoding="utf-8"))
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not (
        done.get("status") == "complete"
        and done.get("result_schema_version") == schema
        and int(done.get("record_count", -1)) == count
        and done.get("result_sha256") == sha256(path)
    ):
        return None
    return payload


def wilson(successes: int, count: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if count <= 0:
        return math.nan, math.nan
    p = successes / count
    denominator = 1 + z * z / count
    centre = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def exact_paired_binomial(a: Iterable[bool], b: Iterable[bool]) -> dict:
    pairs = [(bool(x), bool(y)) for x, y in zip(a, b)]
    a_only = sum(x and not y for x, y in pairs)
    b_only = sum(y and not x for x, y in pairs)
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, k) * 0.5**discordant
            for k in range(0, min(a_only, b_only) + 1)
        )
        p_value = min(1.0, 2.0 * tail)
    return {
        "a_only": a_only,
        "b_only": b_only,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def paired_bootstrap_delta(
    baseline: Iterable[bool], method: Iterable[bool], *, seed: int, draws: int = 100000
) -> dict:
    baseline = np.asarray(list(baseline), dtype=np.float64)
    method = np.asarray(list(method), dtype=np.float64)
    if baseline.shape != method.shape or baseline.size == 0:
        raise ValueError("paired vectors must have the same nonzero size")
    delta = method - baseline
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(delta), size=(draws, len(delta)))
    distribution = delta[samples].mean(axis=1)
    return {
        "count": int(len(delta)),
        "delta": float(delta.mean()),
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "ci95": np.quantile(distribution, [0.025, 0.975]).tolist(),
        "probability_delta_positive": float((distribution > 0).mean()),
    }


def load_training(task: str, arm: str) -> list[dict]:
    directory = OUTPUT / task / arm.lower() / "training"
    records = []
    for path in sorted(directory.glob("epoch_*.json")):
        payload = committed_json(path, schema="paper-training-epoch-v1", count=1)
        if payload is not None:
            records.append(payload)
    return records


def load_h5(task: str, arm: str, *, split_directory: str = "h5_dev30") -> list[dict]:
    directory = OUTPUT / task / arm.lower() / split_directory
    records = []
    for path in sorted(directory.glob("epoch_*/eval.json")):
        payload = committed_json(
            path, schema="paper-h5-checkpoint-aggregate-v1", count=30
        )
        if payload is not None:
            records.append(payload)
    return records


def _gear_violation(height_mm: float, xy_mm: float, orientation_deg: float) -> float:
    score = max(
        max(0.0, float(height_mm)) / 3.0,
        float(xy_mm) / 3.0,
        float(orientation_deg) / 5.0,
    )
    if float(height_mm) <= 0.0:
        score += 1.0 + abs(float(height_mm)) / 3.0
    return score


def progress_values(task: str, record: dict) -> tuple[float, float]:
    if task == "gear":
        source = record["paired_source"]
        initial = _gear_violation(
            source["init_height_mm"],
            source["init_xy_mm"],
            source["init_orientation_deg"],
        )
        best = float(record["best"]["normalized_violation"])
        final = _gear_violation(
            record["final_height_mm"],
            record["final_xy_mm"],
            record["final_orientation_deg"],
        )
        return initial - best, initial - final
    best = record["best"]
    return (
        float(best["active_cube_goal_progress_best"]),
        float(best["active_cube_goal_progress_final"]),
    )


def _save_figure(figure: plt.Figure, destination: Path) -> None:
    """Write the same traceable figure as PNG and vector PDF."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    png = destination.with_suffix(".png")
    pdf = destination.with_suffix(".pdf")
    figure.savefig(png, dpi=180)
    figure.savefig(pdf)
    plt.close(figure)


def write_epoch_table(task: str, destination: Path) -> list[dict]:
    rows = []
    for arm in ARMS:
        train_by_epoch = {
            int(item["epoch"]): item for item in load_training(task, arm)
        }
        h5_by_epoch = {
            int(item["epoch"]): item for item in load_h5(task, arm)
        }
        for epoch in sorted(set(train_by_epoch) | set(h5_by_epoch)):
            train = train_by_epoch.get(epoch, {})
            decision = train.get("loss_control_decision", {})
            telemetry = train.get("telemetry_record", {})
            evaluation = h5_by_epoch.get(epoch)
            successes = None if evaluation is None else int(evaluation["successes"])
            low, high = (math.nan, math.nan) if successes is None else wilson(successes, 30)
            progress = (
                []
                if evaluation is None
                else [progress_values(task, record) for record in evaluation["records"]]
            )
            train_components = decision.get("train_components", {})
            validation_components = decision.get("validation_components", {})
            learning_rates = telemetry.get("learning_rates") or []
            rows.append(
                {
                    "task": task,
                    "arm": arm,
                    "epoch": epoch,
                    "validation_factor": decision.get("validation_factor"),
                    "validation_ema": decision.get("validation_ema"),
                    "best_epoch": decision.get("best_epoch"),
                    "stop": decision.get("stop"),
                    "stop_reason": decision.get("stop_reason"),
                    "h5_successes": successes,
                    "h5_rate": None if successes is None else successes / 30,
                    "h5_wilson_low": low,
                    "h5_wilson_high": high,
                    "h5_best_progress_median": (
                        None if not progress else float(np.median([value[0] for value in progress]))
                    ),
                    "h5_final_progress_median": (
                        None if not progress else float(np.median([value[1] for value in progress]))
                    ),
                    **{
                        f"train_{name}": train_components.get(name)
                        for name in COMPONENTS
                    },
                    **{
                        f"validation_{name}": validation_components.get(name)
                        for name in COMPONENTS
                    },
                    "generalization_gap": decision.get("generalization_gap"),
                    "embedding_std_mean": decision.get("embedding_std_mean"),
                    "embedding_effective_rank": decision.get("embedding_effective_rank"),
                    "embedding_covariance_condition": decision.get(
                        "embedding_covariance_condition"
                    ),
                    "learning_rate": learning_rates[0] if learning_rates else None,
                    "epoch_wall_seconds": telemetry.get("epoch_wall_seconds"),
                    "examples_per_second": telemetry.get("examples_per_second"),
                    "gradient_norm_mean": telemetry.get("gradient_norm_mean"),
                    "peak_cuda_memory_bytes": telemetry.get("peak_cuda_memory_reserved_bytes"),
                }
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["task"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_learning(task: str, rows: list[dict], destination: Path) -> None:
    if not rows:
        return
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        train = [row for row in arm_rows if row["validation_ema"] is not None]
        h5 = [row for row in arm_rows if row["h5_rate"] is not None]
        if train:
            epochs = np.array([row["epoch"] for row in train])
            raw = np.array([row["validation_factor"] for row in train])
            ema = np.array([row["validation_ema"] for row in train])
            axes[0, 0].plot(epochs, raw, marker=".", color=COLORS[arm], alpha=0.25)
            axes[0, 0].plot(epochs, ema, marker="o", color=COLORS[arm], label=arm)
            best_epoch = int(train[-1]["best_epoch"])
            best_rows = [row for row in train if int(row["epoch"]) == best_epoch]
            if best_rows:
                axes[0, 0].scatter(
                    [best_epoch], [best_rows[0]["validation_ema"]], marker="*",
                    s=120, color=COLORS[arm], edgecolor="black", linewidth=0.5,
                    zorder=5,
                )
            if bool(train[-1]["stop"]):
                axes[0, 0].axvline(
                    train[-1]["epoch"], color=COLORS[arm], linestyle=":", alpha=0.7
                )
            axes[1, 0].plot(
                epochs,
                [row["gradient_norm_mean"] for row in train],
                marker="o",
                color=COLORS[arm],
                label=arm,
            )
            axes[1, 1].plot(
                epochs,
                [row["examples_per_second"] for row in train],
                marker="o",
                color=COLORS[arm],
                label=arm,
            )
        if h5:
            epochs = np.array([row["epoch"] for row in h5])
            rate = np.array([row["h5_rate"] for row in h5])
            low = np.array([row["h5_wilson_low"] for row in h5])
            high = np.array([row["h5_wilson_high"] for row in h5])
            axes[0, 1].plot(epochs, rate, marker="o", color=COLORS[arm], label=arm)
            axes[0, 1].fill_between(epochs, low, high, color=COLORS[arm], alpha=0.12)
    axes[0, 0].set(
        title=(
            f"{task.title()} validation composite\n"
            "raw=faint, EMA=solid, ★=loss-best, dotted vertical=loss-stop"
        ),
        ylabel="factor (lower better)",
    )
    axes[0, 1].set(
        title="H5-dev30 strict success (Wilson 95% CI)",
        ylabel="rate",
        ylim=(0, 1),
    )
    axes[1, 0].set(title="Raw pre-clip gradient norm", xlabel="epoch")
    axes[1, 1].set(title="Training throughput", xlabel="epoch", ylabel="examples/s")
    for axis in axes.flat:
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.2)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(ncol=2)
    _save_figure(figure, destination)


def plot_loss_components(task: str, rows: list[dict], destination: Path) -> None:
    if not rows:
        return
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, component in zip(axes.flat, COMPONENTS):
        for arm in ARMS:
            arm_rows = [
                row for row in rows
                if row["arm"] == arm and row[f"validation_{component}"] is not None
            ]
            if not arm_rows:
                continue
            epochs = [row["epoch"] for row in arm_rows]
            axis.plot(
                epochs,
                [row[f"validation_{component}"] for row in arm_rows],
                marker="o",
                color=COLORS[arm],
                label=f"{arm} val",
            )
            train_values = [row[f"train_{component}"] for row in arm_rows]
            if all(value is not None for value in train_values):
                axis.plot(
                    epochs,
                    train_values,
                    linestyle="--",
                    alpha=0.55,
                    color=COLORS[arm],
                    label=f"{arm} train",
                )
        axis.set_title(component.replace("_", " ").title())
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.2)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        axes.flat[-1].legend(handles, labels, loc="center", ncol=2)
    figure.suptitle(f"{task.title()} train / validation loss components")
    _save_figure(figure, destination)


def plot_training_diagnostics(task: str, rows: list[dict], destination: Path) -> None:
    if not rows:
        return
    panels = (
        ("generalization_gap", "Generalization gap"),
        ("learning_rate", "Learning rate"),
        ("embedding_std_mean", "Embedding std (mean)"),
        ("embedding_effective_rank", "Embedding effective rank"),
        ("embedding_covariance_condition", "Embedding covariance condition"),
        ("h5_progress", "H5 best / final progress (median)"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, (field, title) in zip(axes.flat, panels):
        for arm in ARMS:
            if field == "h5_progress":
                arm_rows = [
                    row for row in rows
                    if row["arm"] == arm
                    and row["h5_best_progress_median"] is not None
                    and row["h5_final_progress_median"] is not None
                ]
                if arm_rows:
                    epochs = [row["epoch"] for row in arm_rows]
                    axis.plot(
                        epochs,
                        [row["h5_best_progress_median"] for row in arm_rows],
                        color=COLORS[arm], marker="o", label=f"{arm} best",
                    )
                    axis.plot(
                        epochs,
                        [row["h5_final_progress_median"] for row in arm_rows],
                        color=COLORS[arm], linestyle="--", alpha=0.75,
                        label=f"{arm} final",
                    )
                continue
            arm_rows = [
                row for row in rows
                if row["arm"] == arm and row[field] is not None
            ]
            if arm_rows:
                axis.plot(
                    [row["epoch"] for row in arm_rows],
                    [row[field] for row in arm_rows],
                    marker="o",
                    color=COLORS[arm],
                    label=arm,
                )
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.2)
        if field == "embedding_covariance_condition":
            axis.set_yscale("log")
        if field == "h5_progress":
            handles, labels = axis.get_legend_handles_labels()
            if handles:
                axis.legend(fontsize=7, ncol=2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        axes.flat[0].legend(handles, labels, loc="best", ncol=2, fontsize=8)
    figure.suptitle(f"{task.title()} optimization and representation diagnostics")
    _save_figure(figure, destination)


def plot_case_heatmap(task: str, destination: Path, source_csv: Path) -> None:
    available = {arm: load_h5(task, arm) for arm in ARMS}
    epochs = sorted({int(item["epoch"]) for values in available.values() for item in values})
    if not epochs:
        return
    figure, axes = plt.subplots(1, 4, figsize=(18, 7), sharex=True, sharey=True, constrained_layout=True)
    outcome_cmap = ListedColormap(["#b91c1c", "#15803d"])
    outcome_cmap.set_bad("#f3f4f6")
    if len(epochs) <= 10:
        tick_positions = list(range(len(epochs)))
    else:
        tick_positions = sorted(
            set(np.linspace(0, len(epochs) - 1, 10, dtype=int).tolist())
        )
    source_rows = []
    for axis, arm in zip(axes, ARMS):
        by_epoch = {int(item["epoch"]): item for item in available[arm]}
        matrix = np.full((30, len(epochs)), np.nan)
        canonical_cases = None
        for column, epoch in enumerate(epochs):
            result = by_epoch.get(epoch)
            if result is None:
                continue
            case_ids = [record["case_id"] for record in result["records"]]
            if canonical_cases is None:
                canonical_cases = case_ids
            if case_ids != canonical_cases:
                raise ValueError(f"{task}/{arm}: frozen case order changed")
            matrix[:, column] = [bool(record["success"]) for record in result["records"]]
            source_rows.extend(
                {
                    "task": task,
                    "arm": arm,
                    "epoch": epoch,
                    "case_position": position,
                    "case_id": record["case_id"],
                    "success": int(bool(record["success"])),
                }
                for position, record in enumerate(result["records"])
            )
        axis.imshow(
            matrix,
            vmin=0,
            vmax=1,
            cmap=outcome_cmap,
            aspect="auto",
            interpolation="nearest",
        )
        axis.set_title(arm)
        axis.set_xlabel("epoch")
        axis.set_xticks(tick_positions)
        axis.set_xticklabels(
            [str(epochs[position]) for position in tick_positions],
            rotation=45 if len(epochs) > 10 else 0,
            ha="right" if len(epochs) > 10 else "center",
        )
        if np.isnan(matrix).all():
            axis.text(
                0.5,
                0.5,
                "pending",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#6b7280",
                fontsize=10,
            )
    axes[0].set_ylabel("frozen case position")
    figure.suptitle(
        f"{task.title()} H5-dev30 case × epoch strict success\n"
        "red=failure · green=success · gray=not evaluated"
    )
    source_csv.parent.mkdir(parents=True, exist_ok=True)
    with source_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("task", "arm", "epoch", "case_position", "case_id", "success"),
        )
        writer.writeheader()
        writer.writerows(source_rows)
    _save_figure(figure, destination)


def final_statistics(task: str) -> dict | None:
    outcomes = {}
    sources = {}
    for arm in ARMS:
        path = OUTPUT / task / arm.lower() / "final_test" / "primary" / "eval.json"
        result = committed_json(
            path, schema="paper-h5-checkpoint-aggregate-v1", count=30
        )
        if result is None:
            return None
        sources[arm] = {"path": str(path.resolve()), "sha256": sha256(path)}
        outcomes[arm] = {
            record["case_id"]: bool(record["success"])
            for record in result["records"]
        }
    case_order = list(outcomes["L0"])
    if any(list(outcomes[arm]) != case_order for arm in ARMS):
        raise ValueError(f"{task}: final test case order differs across arms")
    rows = {}
    for arm in ARMS:
        successes = sum(outcomes[arm].values())
        rows[arm] = {
            "successes": successes,
            "count": 30,
            "rate": successes / 30,
            "wilson95": wilson(successes, 30),
        }
    comparisons = {}
    baseline = [outcomes["L0"][case] for case in case_order]
    for arm in ("L1", "C0", "C1"):
        method = [outcomes[arm][case] for case in case_order]
        comparisons[f"{arm}_minus_L0"] = {
            **paired_bootstrap_delta(baseline, method, seed=20260906),
            "mcnemar": exact_paired_binomial(baseline, method),
        }
    return {"task": task, "rows": rows, "comparisons": comparisons, "sources": sources}


def main() -> None:
    figures = OUTPUT / "figures"
    tables = OUTPUT / "tables"
    summaries = {}
    for task in ("gear", "franka"):
        rows = write_epoch_table(task, tables / f"{task}_epoch_metrics.csv")
        plot_learning(task, rows, figures / f"{task}_learning_curves.png")
        plot_loss_components(task, rows, figures / f"{task}_loss_components.png")
        plot_training_diagnostics(
            task, rows, figures / f"{task}_training_diagnostics.png"
        )
        plot_case_heatmap(
            task,
            figures / f"{task}_case_epoch_heatmap.png",
            tables / f"{task}_case_epoch_success.csv",
        )
        summaries[task] = {
            "completed_training_epochs": {
                arm: len(load_training(task, arm)) for arm in ARMS
            },
            "completed_h5_epochs": {arm: len(load_h5(task, arm)) for arm in ARMS},
            "final_statistics": final_statistics(task),
        }
    output = OUTPUT / "comparisons/incremental_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(output), **summaries}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

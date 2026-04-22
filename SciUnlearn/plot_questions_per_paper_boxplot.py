from __future__ import annotations

import argparse
import json
import random
import statistics as stats
from collections import Counter
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def extract_paper_id(example_id: str) -> str:
    if not example_id:
        return ""
    first = example_id.split("|", 1)[0].strip()
    if first.lower().endswith(".pdf"):
        return first[:-4]
    return first


def per_paper_question_counts(jsonl_path: Path) -> List[int]:
    counts = Counter()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            paper_id = extract_paper_id(str(row.get("id", "")))
            if paper_id:
                counts[paper_id] += 1
    return sorted(counts.values())


def default_input_files(data_dir: Path) -> Dict[str, Path]:
    return {
        "forget_sc_1": data_dir / "forget_sc_1.jsonl",
        "forget_sc_2": data_dir / "forget_sc_2.jsonl",
        "retain_external": data_dir / "retain_external_sc.jsonl",
        "retain_internal": data_dir / "retain_internal_sc.jsonl",
    }


def plot_boxplot(series: Dict[str, List[int]], output_path: Path, show_plot: bool) -> None:
    labels = list(series.keys())
    values = [series[k] for k in labels]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(values, labels=labels, patch_artist=True)
    ax.set_title("Questions Per Paper Across Common-Filtered Splits")
    ax.set_ylabel("Questions per paper")
    ax.set_xlabel("Dataset split")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    if show_plot:
        plt.show()
    plt.close(fig)


def plot_clean(series: Dict[str, List[int]], output_path: Path, show_plot: bool) -> None:
    labels = list(series.keys())
    values = [series[k] for k in labels]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(
        values,
        labels=labels,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops={"color": "#1f1f1f", "linewidth": 2.0},
        whiskerprops={"color": "#555555", "linewidth": 1.2},
        capprops={"color": "#555555", "linewidth": 1.2},
        boxprops={"edgecolor": "#555555", "linewidth": 1.1},
    )

    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(colors[i % len(colors)])
        box.set_alpha(0.35)

    ax.set_title("Questions Per Paper Across Splits", pad=12)
    ax.set_ylabel("Questions per paper")
    ax.set_xlabel("Dataset split")
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=15)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    if show_plot:
        plt.show()
    plt.close(fig)


def plot_advanced(series: Dict[str, List[int]], output_path: Path, show_plot: bool) -> None:
    labels = list(series.keys())
    values = [series[k] for k in labels]
    positions = list(range(1, len(labels) + 1))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]

    fig, ax = plt.subplots(figsize=(12, 7))

    v = ax.violinplot(values, positions=positions, showmeans=False, showmedians=False, showextrema=False)
    for i, body in enumerate(v["bodies"]):
        body.set_facecolor(colors[i % len(colors)])
        body.set_edgecolor("#222222")
        body.set_alpha(0.22)

    ax.boxplot(
        values,
        positions=positions,
        widths=0.22,
        patch_artist=True,
        showfliers=True,
        boxprops={"facecolor": "#FFFFFF", "edgecolor": "#222222"},
        medianprops={"color": "#222222", "linewidth": 1.8},
        whiskerprops={"color": "#222222"},
        capprops={"color": "#222222"},
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "#222222", "alpha": 0.35},
    )

    rng = random.Random(13)
    for i, ys in enumerate(values):
        x0 = positions[i]
        x_jitter = [x0 + rng.uniform(-0.07, 0.07) for _ in ys]
        ax.scatter(x_jitter, ys, s=12, color=colors[i % len(colors)], alpha=0.55, linewidths=0)

        n = len(ys)
        mean_v = sum(ys) / n
        median_v = stats.median(ys)
        q1, q3 = stats.quantiles(ys, n=4, method="inclusive")[0], stats.quantiles(ys, n=4, method="inclusive")[2]
        iqr = q3 - q1
        note = f"n={n}\nmean={mean_v:.1f}\nmed={median_v:.1f}\nIQR={iqr:.1f}"
        ax.text(x0, max(ys) + 0.7, note, ha="center", va="bottom", fontsize=8, color="#333333")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_title("Common-Filtered Questions Per Paper: Distribution + Summary")
    ax.set_ylabel("Questions per paper")
    ax.set_xlabel("Dataset split")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    if show_plot:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot box plot of question counts per paper for common-filtered forget/retain datasets."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("export_data"),
        help="Directory containing common-filtered dataset JSONL files (default: export_data).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/questions_per_paper_boxplot_common.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plot window in addition to saving image.",
    )
    parser.add_argument(
        "--style",
        choices=["clean", "advanced", "basic"],
        default="clean",
        help="Plot style: clean (minimal), advanced (violin+box+points), or basic (box only).",
    )
    args = parser.parse_args()

    input_files = default_input_files(args.data_dir)
    missing = [name for name, path in input_files.items() if not path.exists()]
    if missing:
        missing_msg = ", ".join(missing)
        raise FileNotFoundError(f"Missing input JSONL files for: {missing_msg}")

    data_series: Dict[str, List[int]] = {}
    for name, path in input_files.items():
        counts = per_paper_question_counts(path)
        if not counts:
            raise ValueError(f"No valid rows found in {path}")
        data_series[name] = counts

    if args.style == "clean":
        plot_clean(data_series, args.output, args.show)
    elif args.style == "advanced":
        plot_advanced(data_series, args.output, args.show)
    else:
        plot_boxplot(data_series, args.output, args.show)

    print(f"Saved {args.style} plot to: {args.output}")
    for name, counts in data_series.items():
        print(f"{name}: papers={len(counts)}, min={min(counts)}, median={counts[len(counts)//2]}, max={max(counts)}")


if __name__ == "__main__":
    main()
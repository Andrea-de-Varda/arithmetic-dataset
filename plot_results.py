#!/usr/bin/env python3
"""
Plot arithmetic evaluation results.

Reads per-dataset summary JSONs from:
results/<model_safe>/<mode>/<dataset>_summary.json
and produces one bar plot per dataset/condition.

Model ordering:
1) Families sorted by numeric accuracy (ascending, worst to best).
2) Within each family, models sorted by numeric accuracy (ascending).
3) The same order is used across all plots.

Colors and nice labels are mapped via the dictionaries below.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re


# Family / label / color maps (extend as needed)
model_family = {
    # Gemma: group all 2/3 variants under one family
    "gemma-2-27b": "Gemma",
    "gemma-2-27b-instruct": "Gemma",
    "gemma-2-9b": "Gemma",
    "gemma-2-9b-instruct": "Gemma",
    "gemma-2b": "Gemma",
    "gemma-2b-instruct": "Gemma",
    "gemma-7b": "Gemma",
    "gemma-7b-instruct": "Gemma",
    "gemma-3-1b-pt": "Gemma",
    "qwen2-1-5b": "Qwen",
    "qwen2-1-5b-instruct": "Qwen",
    "qwen2-7b": "Qwen",
    "qwen2-7b-instruct": "Qwen",
    "qwen2-72b": "Qwen",
    "qwen2-72b-instruct": "Qwen",
    "qwen3-0.6b": "Qwen 3",
    "qwen3-4b": "Qwen 3",
    "qwen3-8b": "Qwen 3",
    "qwen3-30b-a3b-base": "Qwen 3",
    "qwen3-32b": "Qwen 3",
    # Llama: group 3 / 3.1 / 3.2 together
    "llama-3-1-8b": "Llama 3",
    "llama-3-1-8b-instruct": "Llama 3",
    "llama-3.1-8b": "Llama 3",
    "llama-3-70b": "Llama 3",
    "llama-3-70b-instruct": "Llama 3",
    "llama-3-8b": "Llama 3",
    "llama-3-8b-instruct": "Llama 3",
    "llama-3.2-1b": "Llama 3",
    "llama-3.2-3b": "Llama 3",
    "mistral-7b": "Mistral",
    "mistral-7b-instruct": "Mistral",
    "mistral-nemo-12b-2407": "Mistral NeMo",
    "mistral-nemo-12b-instruct-2407": "Mistral NeMo",
    "mixtral-8x7b-instruct-v0-1": "Mixtral",
    "phi-2": "Phi-2",
    "phi-4": "Phi-4",
    "olmo-3-1025-7b": "Olmo 3",
    "olmo-3-1125-32b": "Olmo 3",
}

model_nice = {
    "gemma-2-27b": "Gemma-2$_{27B}$",
    "gemma-2-27b-instruct": "Gemma-2-I$_{27B}$",
    "gemma-2-9b": "Gemma-2$_{9B}$",
    "gemma-2-9b-instruct": "Gemma-2-I$_{9B}$",
    "gemma-2b": "Gemma$_{2B}$",
    "gemma-2b-instruct": "Gemma-I$_{2B}$",
    "gemma-7b": "Gemma$_{7B}$",
    "gemma-7b-instruct": "Gemma-I$_{7B}$",
    "gemma-3-1b-pt": "Gemma-3$_{1B}$",
    "qwen2-1-5b": "Qwen-2$_{1.5B}$",
    "qwen2-1-5b-instruct": "Qwen-2-I$_{1.5B}$",
    "qwen2-7b": "Qwen-2$_{7B}$",
    "qwen2-7b-instruct": "Qwen-2-I$_{7B}$",
    "qwen2-72b": "Qwen-2$_{72B}$",
    "qwen2-72b-instruct": "Qwen-2-I$_{72B}$",
    "qwen3-0.6b": "Qwen-3$_{0.6B}$",
    "qwen3-4b": "Qwen-3$_{4B}$",
    "qwen3-8b": "Qwen-3$_{8B}$",
    "qwen3-30b-a3b-base": "Qwen-3$_{30B}$",
    "qwen3-32b": "Qwen-3$_{32B}$",
    "llama-3-1-8b": "Llama-3.1$_{8B}$",
    "llama-3-1-8b-instruct": "Llama-3.1-I$_{8B}$",
    "llama-3.1-8b": "Llama-3.1$_{8B}$",
    "llama-3-70b": "Llama-3$_{70B}$",
    "llama-3-70b-instruct": "Llama-3-I$_{70B}$",
    "llama-3-8b": "Llama-3$_{8B}$",
    "llama-3-8b-instruct": "Llama-3-I$_{8B}$",
    "llama-3.2-1b": "Llama-3.2$_{1B}$",
    "llama-3.2-3b": "Llama-3.2$_{3B}$",
    "mistral-7b": "Mistral$_{7B}$",
    "mistral-7b-instruct": "Mistral-I$_{7B}$",
    "mistral-nemo-12b-2407": "Mistral NeMo$_{12B}$",
    "mistral-nemo-12b-instruct-2407": "Mistral NeMo-I$_{12B}$",
    "mixtral-8x7b-instruct-v0-1": "Mixtral$_{8×7B}$",
    "phi-2": "Phi-2",
    "phi-4": "Phi-4",
    "olmo-3-1025-7b": "Olmo-3$_{7B}$",
    "olmo-3-1125-32b": "Olmo-3$_{32B}$",
}

palette_d = {
    "Llama 3": "steelblue",
    "Llama 3.1": "teal",
    "Llama 3.2": "#4c72b0",
    "Gemma 2": "yellowgreen",
    "Gemma 3": "#9acd32",
    "Mistral NeMo": "salmon",
    "Mistral": "tomato",
    "Phi-2": "slategray",
    "Phi-4": "#708090",
    "Qwen": "lightsteelblue",
    "Qwen 3": "#9bb7d6",
    "Gemma": "gold",
    "Mixtral": "darkorange",
    "Olmo 3": "#8c564b",
}


alias_map = {
    # HF ids -> canonical keys used in the dictionaries above
    "mistralai/mistral-7b-v0.1": "mistral-7b",
    "mistralai-mistral-7b-v0.1": "mistral-7b",
    "mistralai__mistral-7b-v0.1": "mistral-7b",
    "mistralai/mistral-7b": "mistral-7b",
    "mistralai/mistral-7b-instruct": "mistral-7b-instruct",
    "mistralai/mixtral-8x7b-instruct-v0.1": "mixtral-8x7b-instruct-v0-1",
    "google/gemma-2-2b": "gemma-2b",
    "google/gemma-2-2b-it": "gemma-2b-instruct",
    "google/gemma-2-7b": "gemma-7b",
    "google/gemma-2-7b-it": "gemma-7b-instruct",
    "google/gemma-2-9b": "gemma-2-9b",
    "google/gemma-2-9b-it": "gemma-2-9b-instruct",
    "google/gemma-2-27b": "gemma-2-27b",
    "google/gemma-2-27b-it": "gemma-2-27b-instruct",
    "google/gemma-3-1b-pt": "gemma-3-1b-pt",
    "Qwen/Qwen3-0.6B": "qwen3-0.6b",
    "Qwen/Qwen3-4B": "qwen3-4b",
    "Qwen/Qwen3-8B": "qwen3-8b",
    "Qwen/Qwen3-30B-A3B-Base": "qwen3-30b-a3b-base",
    "Qwen/Qwen3-32B": "qwen3-32b",
    "allenai/Olmo-3-1025-7B": "olmo-3-1025-7b",
    "allenai/Olmo-3-1125-32B": "olmo-3-1125-32b",
    "meta-llama/Llama-3.1-8B": "llama-3.1-8b",
    "meta-llama/Llama-3.2-1B": "llama-3.2-1b",
    "meta-llama/Llama-3.2-3B": "llama-3.2-3b",
    "microsoft/phi-4": "phi-4",
}


def normalize_model_key(name: str) -> str:
    """Normalize HF model id to our canonical key if possible."""
    if not name:
        return name
    # direct alias match
    if name in alias_map:
        return alias_map[name]
    lower_name = name.lower()
    if lower_name in alias_map:
        return alias_map[lower_name]
    # replace path separators with dash
    key = name.replace("/", "-").replace("__", "-").lower()
    key = key.replace(" ", "")
    return key


def infer_family(model_key: str) -> str:
    """Best-effort family inference from model key."""
    if model_key in model_family:
        return model_family[model_key]
    if "qwen" in model_key:
        return "Qwen"
    if "llama" in model_key:
        return "Llama 3"
    if "gemma" in model_key:
        return "Gemma"
    if "nemo" in model_key:
        return "Mistral NeMo"
    if "mistral" in model_key:
        return "Mistral"
    if "mixtral" in model_key:
        return "Mixtral"
    if "phi-2" in model_key or "phi2" in model_key:
        return "Phi-2"
    return "Other"


def infer_nice_name(model_key: str, raw_name: str) -> str:
    """Return nice display name; fallback to heuristic formatting."""
    if model_key in model_nice:
        return model_nice[model_key]
    name = raw_name.replace("/", "-")
    # extract size like 7b, 70b, 1.5b, 8x7b
    size = ""
    m = re.search(r"(\\d+(?:\\.\\d+)?(?:x\\d+)?b)", name, re.IGNORECASE)
    if m:
        size = m.group(1).upper()
    base = name.split("-")[0].title()
    inst = " I" if ("instruct" in name or "-it" in name) else ""
    if size:
        return f"{base}{inst}_${size}$"
    return f"{base}{inst}"


def recover_model_name(folder_model: str, summary_model_field: str) -> str:
    """Prefer the model name recorded in the summary file; fallback to folder name."""
    if summary_model_field:
        return summary_model_field
    return folder_model.replace("__", "/")


def load_results(results_dir: Path, mode: str, datasets: list[str]) -> pd.DataFrame:
    rows = []
    for model_dir in sorted(results_dir.iterdir()):
        mode_dir = model_dir / mode
        if not mode_dir.is_dir():
            continue
        for ds in datasets:
            summary_path = mode_dir / f"{ds}_summary.json"
            if not summary_path.exists():
                continue
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            model_name = recover_model_name(model_dir.name, summary.get("model"))
            rows.append(
                {
                    "model": model_name,
                    "dataset": ds,
                    "accuracy": summary.get("accuracy", 0.0),
                }
            )
    if not rows:
        raise RuntimeError("No results found. Check paths, mode, and datasets.")
    return pd.DataFrame(rows)


def build_order(df: pd.DataFrame) -> tuple[list[str], dict[str, float]]:
    """Return ordered model list and numeric acc map."""
    numeric = df[df["dataset"] == "numeric"].copy()
    if numeric.empty:
        raise RuntimeError("Numeric results missing; cannot build ordering.")
    numeric["model_key"] = numeric["model"].apply(normalize_model_key)
    numeric["family"] = numeric["model_key"].apply(infer_family)

    # Family order by mean numeric accuracy (asc: worst -> best)
    family_order = (
        numeric.groupby("family")["accuracy"]
        .mean()
        .sort_values(ascending=True)
        .index.tolist()
    )

    # Within each family, order models by numeric accuracy (asc)
    model_order = []
    numeric_acc = dict(zip(numeric["model"], numeric["accuracy"]))
    for fam in family_order:
        fam_models = numeric[numeric["family"] == fam]
        ordered = fam_models.sort_values("accuracy", ascending=True)["model"].tolist()
        model_order.extend(ordered)

    return model_order, numeric_acc


def compute_positions(model_order: list[str]) -> tuple[dict[str, float], list[tuple[float, float, str]]]:
    """Assign x positions with gaps between families; return positions and bracket ranges."""
    positions = {}
    brackets = []
    gap = 0.6
    x = 1.0
    current_family = None
    start_pos = None

    for m in model_order:
        fam_key = normalize_model_key(m)
        fam = model_family.get(fam_key, "Other")
        if fam != current_family:
            if current_family is not None:
                brackets.append((start_pos, prev_pos, current_family))
                x += gap
            current_family = fam
            start_pos = x
        positions[m] = x
        prev_pos = x
        x += 1.0

    if current_family is not None:
        brackets.append((start_pos, prev_pos, current_family))

    return positions, brackets


def add_bracket(ax, pos1, pos2, text, height_frac=0.05, weight="normal"):
    mid = (pos1 + pos2) / 2
    ylim = ax.get_ylim()
    height = (ylim[1] - ylim[0]) * height_frac
    y = ylim[1] - height * 1.5
    ax.plot([pos1, pos1, pos2, pos2], [y, y + height, y + height, y], color="black", lw=2.0)
    ax.text(mid, y + height * 1.2, text, ha="center", va="bottom", fontsize=13, weight=weight)


def plot_condition(df: pd.DataFrame, dataset: str, model_order: list[str], positions: dict[str, float],
                   brackets: list[tuple[float, float, str]], output_dir: Path, mode: str):
    df = df.copy()
    df = df[df["dataset"] == dataset]
    if df.empty:
        print(f"[warn] No data for dataset={dataset}, skipping plot.")
        return

    df["model_key"] = df["model"].apply(normalize_model_key)
    df["family"] = df["model_key"].apply(infer_family)
    df["model_nice"] = df.apply(lambda r: infer_nice_name(r["model_key"], r["model"]), axis=1)
    df["color"] = df["family"].map(palette_d).fillna("gray")
    df["x"] = df["model"].map(positions)
    # preserve global ordering
    order_idx = {m: i for i, m in enumerate(model_order)}
    df["order_idx"] = df["model"].map(order_idx).fillna(len(model_order) + 1)
    df = df.sort_values("order_idx")

    plt.figure(figsize=(7,7), dpi=300)
    sns.set_context("talk")
    ax = plt.gca()

    bars = ax.bar(
        df["x"],
        df["accuracy"] * 100.0,
        color=df["color"],
        edgecolor=".2",
        lw=2.2,
    )

    # Set ylim with margin before brackets
    ymax = (df["accuracy"].max() * 100.0) if not df.empty else 0
    ax.set_ylim(0, 110)

    # Brackets per family
    # for start, end, fam in brackets:
    #     add_bracket(ax, start, end, fam, height_frac=0.05, weight="bold")

    #ax.set_title(f"{dataset.capitalize()}", fontsize=25, weight="bold", pad=28)
    ax.set_ylabel("Accuracy (%)", fontsize=20)
    ax.set_xticks(df["x"])
    ax.set_xticklabels(df["model_nice"], rotation=55, ha="right", fontsize=16)
    sns.despine()
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{dataset}_{mode}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def plot_scatter_numeric_vs_english(df: pd.DataFrame, model_order: list[str], output_dir: Path, mode: str):
    """Scatter plot: numeric (x) vs english (y) accuracy, colored by family, with regression line."""
    wide = df.pivot(index="model", columns="dataset", values="accuracy")
    if "numeric" not in wide.columns or "english" not in wide.columns:
        print("[warn] Missing numeric or english results; skipping scatter plot.")
        return
    wide = wide.dropna(subset=["numeric", "english"]).reset_index()
    wide["model_key"] = wide["model"].apply(normalize_model_key)
    wide["family"] = wide["model_key"].apply(infer_family)
    wide["color"] = wide["family"].map(palette_d).fillna("gray")
    wide["model_nice"] = wide.apply(lambda r: infer_nice_name(r["model_key"], r["model"]), axis=1)
    order_idx = {m: i for i, m in enumerate(model_order)}
    wide["order_idx"] = wide["model"].map(order_idx).fillna(len(model_order) + 1)
    wide = wide.sort_values("order_idx")

    x = wide["numeric"] * 100.0
    y = wide["english"] * 100.0

    plt.figure(figsize=(7, 7), dpi=300)
    sns.set_context("talk")
    ax = plt.gca()

    # Regression line
    sns.regplot(x=x, y=y, scatter=False, ax=ax, color="black", line_kws={"lw": 2, "alpha": 0.6})

    # Scatter points
    ax.scatter(x, y, c=wide["color"], edgecolors=".2", linewidths=1.2, s=100, alpha=0.9)

    # Legend by family
    handles = []
    for fam in sorted(wide["family"].unique()):
        handles.append(
            plt.Line2D([0], [0], marker="o", color="w", label=fam,
                       markerfacecolor=palette_d.get(fam, "gray"), markeredgecolor=".2", markersize=8)
        )
    ax.legend(handles=handles, title="Family", fontsize=12, title_fontsize=13, frameon=False, loc="lower right")

    ax.set_xlabel("Numeric accuracy (%)", fontsize=18)
    ax.set_ylabel("English accuracy (%)", fontsize=18)
    #ax.set_xlim(0, 110)
    #ax.set_ylim(0, 110)
    sns.despine()
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"numeric_vs_english_{mode}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot arithmetic evaluation results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"),
                        help="Directory with evaluation results")
    parser.add_argument("--mode", type=str, default="standard",
                        choices=["standard", "reasoning"],
                        help="Evaluation mode to plot")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["numeric", "english", "spanish", "italian", "embedded", "embedded_verbal"],
                        help="Datasets/conditions to plot")
    parser.add_argument("--output-dir", type=Path, default=Path("plots"),
                        help="Directory to save figures")

    args = parser.parse_args()

    df = load_results(args.results_dir, args.mode, args.datasets)
    model_order, numeric_acc = build_order(df)
    positions, brackets = compute_positions(model_order)

    for ds in args.datasets:
        plot_condition(df, ds, model_order, positions, brackets, args.output_dir, args.mode)

    # Scatter numeric vs english
    plot_scatter_numeric_vs_english(df, model_order, args.output_dir, args.mode)


if __name__ == "__main__":
    main()



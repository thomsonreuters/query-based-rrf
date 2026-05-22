#!/usr/bin/env python3
"""
Tradeoff plot: per-query inference latency vs. avg metric gain over RRF baseline.

Single-panel usage:
    from plot_tradeoff import plot_tradeoff

    models = [
        ("rrf",               0,     0.000),
        ("mow",               0.25,  0.019),
        ("roberta regression",10.0,  0.023),
        ("DAT-gpt5.2",        250.0, 0.040),
    ]
    plot_tradeoff(models, metric_name="NDCG@10", dataset_name="MS MARCO")

2×2 grid usage:
    from plot_tradeoff import plot_tradeoff_grid

    datasets = [
        ("NDCG@10", "ACORD",    [...models...]),
        ("MRR@10",  "MS MARCO", [...models...]),
        ("NDCG@10", "NFCorpus", [...models...]),
        ("MRR@10",  "NQ",       [...models...]),
    ]
    plot_tradeoff_grid(datasets, output="tradeoff_all.png")

Caller only provides (name, latency_ms, metric_value).
Marker shape encodes model family; color+fill distinguishes models within a family.
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["font.size"] = 11

# ---------------------------------------------------------------------------
# Auto-styling config
# ---------------------------------------------------------------------------

# Tier-based marker shapes — one shape per model family.
# Models not listed fall back to "o".
MODEL_TIER_MARKERS = {
    "rrf":                       "X",   # tier 1: baselines
    "mow":                       "X",
    "ridge regression":          "s",   # tier 2: linear
    "roberta regression":        "^",   # tier 3a: RoBERTa
    "roberta interval":          "^",
    "moderbert regression":      "o",   # tier 3b: ModernBERT
    "moderbert interval":        "o",
    "DAT-qwen3":                 "D",   # tier 4: DAT-based / retrieval-aware
    "DAT-minstral3":             "D",
    "dat-moderbert-interval":    "o",
    "llm-fs-qwen-3-mbw":         "v",   # tier 5: LLM few-shot
    "llm-fs-qwen-3-interval":    "v",
    "llm-fs-mistral-mbw":        "v",
    "llm-fs-mistral-3-interval": "v",
}

# Okabe-Ito palette — color-blind safe (deuteranopia, protanopia, tritanopia).
# Reference: Okabe & Ito (2008), https://jfly.uni-koeln.de/color/
MODEL_COLORS = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow (use cautiously on white backgrounds)
    "#000000",  # black
]

# X position for zero-latency models on the log scale
ZERO_X = 0.1
X_LIM  = (0.06, 3000)

# Marker size and edge-width overrides keyed by marker code.
_MARKER_SIZE = {"*": 14, "^": 11, "v": 11, "<": 11, ">": 11}
_MARKER_EW   = {"x": 2.0, "X": 2.0}

# Display-name overrides applied only in the legend (data pipeline names are unchanged).
LEGEND_DISPLAY_NAMES = {
    "rrf":                    "RRF",
    "mow":                    "Mean Optimal Weight",
    "roberta regression":     "RoBERTa regression",
    "roberta interval":       "RoBERTa interval",
    "moderbert regression":    "ModernBERT regression",
    "moderbert interval":      "ModernBERT interval",
    "dat-moderbert-interval":  "DAT ModernBERT interval",
}

# Canonical names pinned to the bottom of the Models legend, in this order.
# All other models are sorted alphabetically above them.
LEGEND_TAIL = ["mow", "rrf"]

# Models that are retrieval-aware (passage-conditioned); drawn with a dot inside.
RETRIEVAL_AWARE_MODELS = {"DAT-qwen3", "DAT-minstral3", "dat-moderbert-interval"}




def _latency_bucket_idx(latency_ms):
    if latency_ms == 0:
        return 0
    if latency_ms <= 1:
        return 1
    if latency_ms <= 10:
        return 2
    if latency_ms <= 100:
        return 3
    return 4


def _build_style_map(model_names):
    """Assign stable (color, filled, marker) to each unique model in order of first appearance."""
    unique = list(dict.fromkeys(model_names))
    style_map = {}
    for i, name in enumerate(unique):
        color  = MODEL_COLORS[i % len(MODEL_COLORS)]
        filled = i < len(MODEL_COLORS)  # first 8 filled, next 8 hollow
        marker = MODEL_TIER_MARKERS.get(name, "o")
        style_map[name] = (color, filled, marker)
    return style_map


def _draw_panel(ax, models, metric_name, dataset_name, delta_mode, style_map):
    """
    Draw one tradeoff panel onto *ax*.

    Returns the list of point dicts (used by callers to build the legend).
    Does NOT add any legend to ax.
    """
    points = []
    for name, latency, value in models:
        color, filled, marker = style_map[name]
        x = ZERO_X if latency == 0 else latency
        points.append({"x": x, "y": value, "name": name,
                       "color": color, "filled": filled, "marker": marker,
                       "latency": latency,
                       "retrieval_aware": name in RETRIEVAL_AWARE_MODELS})

    y_min = min(pt["y"] for pt in points)
    y_max = max(pt["y"] for pt in points)
    y_range = y_max - y_min

    LOG_SPREAD = 0.06  # log10 units of horizontal spread per point within a bucket
    # Clamp slightly below gridlines so no point lands on the dashed line itself.
    _BUCKET_UPPER = {1: 0.95, 2: 9.5, 3: 95.0}
    bucket_groups: dict[int, list[int]] = {}
    for idx, pt in enumerate(points):
        bucket_groups.setdefault(_latency_bucket_idx(pt["latency"]), []).append(idx)
    for indices in bucket_groups.values():
        if len(indices) <= 1:
            continue
        sorted_idx = sorted(indices, key=lambda i: points[i]["x"])
        n = len(sorted_idx)
        for rank, idx in enumerate(sorted_idx):
            offset = (rank - (n - 1) / 2) * LOG_SPREAD
            new_x = points[idx]["x"] * (10 ** offset)
            bucket = _latency_bucket_idx(points[idx]["latency"])
            if bucket in _BUCKET_UPPER:
                new_x = min(new_x, _BUCKET_UPPER[bucket])
            points[idx]["x"] = new_x

    for pt in points:
        fc = pt["color"] if pt["filled"] else "none"
        ms = _MARKER_SIZE.get(pt["marker"], 10)
        ew = _MARKER_EW.get(pt["marker"], 1.5)
        ax.plot(pt["x"], pt["y"], marker=pt["marker"], color=pt["color"],
                markerfacecolor=fc, markeredgecolor=pt["color"],
                markersize=ms, markeredgewidth=ew,
                linestyle="none", zorder=3)
        if pt["retrieval_aware"]:
            ax.plot(pt["x"], pt["y"], marker=".", color="white",
                    markersize=ms * 0.38, linestyle="none", zorder=4)

    pad = y_range * 0.15 if y_range > 0 else 0.005
    ax.set_xscale("log")
    ax.set_xlim(*X_LIM)
    ax.set_xticks([ZERO_X, 1, 10, 100, 1000])
    ax.set_xticklabels(["0 ms", "~1 ms", "~10 ms", "~100 ms", "~1000 ms"])
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlabel("Per-query inference latency (log scale)", labelpad=8)
    ax.set_ylim(y_min - pad, y_max + pad)
    tick_interval = 0.005 if y_range < 0.05 else (0.01 if y_range < 0.1 else 0.02)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
    if delta_mode:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"+{v:.3f}" if v >= 0 else f"{v:.3f}"))
        ax.set_ylabel(f"Avg Δ{metric_name}\nover RRF", rotation=0, ha="right", va="top",
                      position=(0, 1.02), labelpad=10)
    else:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.3f}"))
        ax.set_ylabel(f"Avg {metric_name}", rotation=0, ha="right", va="top",
                      position=(0, 1.02), labelpad=10)

    if dataset_name:
        ax.set_title(dataset_name, pad=12)

    ax.yaxis.grid(True, linestyle="-", linewidth=0.4, color="lightgray", zorder=0)
    for vx in (1, 10, 100):
        ax.axvline(vx, linestyle="--", linewidth=0.7, color="lightgray", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return points


def _model_legend_handles(points):
    """One colored Line2D handle per unique model.

    Handles are sorted alphabetically by display name, with LEGEND_TAIL models
    pinned to the bottom in their declared order.
    """
    seen = set()
    handles = []
    for pt in points:
        if pt["name"] in seen:
            continue
        seen.add(pt["name"])
        fc = pt["color"] if pt["filled"] else "none"
        label = LEGEND_DISPLAY_NAMES.get(pt["name"], pt["name"])
        handles.append(
            matplotlib.lines.Line2D(
                [0], [0], marker=pt["marker"], color=pt["color"],
                markerfacecolor=fc, markeredgecolor=pt["color"],
                markersize=8, linestyle="none", label=label,
            )
        )

    tail_labels = {LEGEND_DISPLAY_NAMES.get(n, n) for n in LEGEND_TAIL}
    body = sorted((h for h in handles if h.get_label() not in tail_labels),
                  key=lambda h: h.get_label())
    tail = [h for n in LEGEND_TAIL
            for h in handles if h.get_label() == LEGEND_DISPLAY_NAMES.get(n, n)]

    note = matplotlib.lines.Line2D(
        [0], [0], linestyle="none", marker=".", color="white",
        markeredgecolor="dimgray", markeredgewidth=0.8, markersize=6,
        label="· = retrieval aware",
    )
    return body + tail + [note]


def _place_panel_legend(ax, model_handles):
    """Attach the Models legend to a single-panel axes."""
    ax.legend(
        handles=model_handles,
        loc="upper left", bbox_to_anchor=(1.02, 1),
        fontsize=9, framealpha=0.9, edgecolor="lightgray",
        title="Models", title_fontsize=9,
    )


def _place_grid_legend(fig, model_handles):
    """Attach the Models legend to the right of a multi-panel figure."""
    fig.legend(
        handles=model_handles,
        loc="upper left", bbox_to_anchor=(0.76, 0.97),
        ncol=1,
        fontsize=9, framealpha=0.9, edgecolor="lightgray",
        title="Models", title_fontsize=9,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _save_all_formats(fig, output: str) -> None:
    """Save *fig* as SVG and PDF for poster-quality vector output."""
    stem = Path(output).with_suffix("")
    for fmt in ("svg", "pdf"):
        path = f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved {path}")


def plot_tradeoff(models, metric_name="NDCG@10", dataset_name=None,
                  output="tradeoff_plot.png", delta_mode=True, show=True):
    """
    Generate a single tradeoff plot and save it as PNG, SVG, and PDF.

    Parameters
    ----------
    models : list of (name, latency_ms, metric_value)
        latency_ms = 0 means no per-query inference cost.
    metric_name : str
        Label for the y-axis, e.g. "NDCG@10" or "MRR@10".
    dataset_name : str or None
        Optional title shown above the plot.
    output : str
        Output file path (extension is replaced; SVG and PDF are always written).
    """
    style_map = _build_style_map([name for name, _, _ in models])
    fig, ax = plt.subplots(figsize=(9, 6.5))
    points = _draw_panel(ax, models, metric_name, dataset_name, delta_mode, style_map)
    _place_panel_legend(ax, _model_legend_handles(points))
    plt.tight_layout()
    _save_all_formats(fig, output)
    if show:
        plt.show()
    plt.close(fig)


def plot_tradeoff_grid(datasets, output="tradeoff_all.png", delta_mode=False, show=True):
    """
    Generate a 2×2 grid of tradeoff subplots (one per dataset) and save as PNG, SVG, and PDF.

    Parameters
    ----------
    datasets : list of (metric_name, dataset_name, models)
        models is a list of (name, latency_ms, metric_value).
    output : str
        Output file path (extension is replaced; SVG and PDF are always written).
    """
    ncols = 2
    nrows = (len(datasets) + 1) // 2

    all_names = [name for _, _, models in datasets for name, _, _ in models]
    style_map = _build_style_map(all_names)

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5.5 * nrows), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    all_points = []
    for ax, (metric_name, dataset_name, models) in zip(axes_flat, datasets):
        all_points.extend(_draw_panel(ax, models, metric_name, dataset_name, delta_mode, style_map))

    for ax in axes_flat[len(datasets):]:
        ax.set_visible(False)

    _place_grid_legend(fig, _model_legend_handles(all_points))

    plt.tight_layout(rect=[0, 0, 0.75, 1])
    _save_all_formats(fig, output)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Example — replace with your actual measured values
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    MODELS = [
        # name                    latency_ms   metric_value
        ("rrf",                   0,           0.012),
        ("mow",                   0.25,        0.019),
        ("ridge regression",      0.50,        0.019),
        ("roberta regression",    10.0,        0.023),
        ("moderbert regression",  15.0,        0.017),
        ("llm-fs-qwen3",          200.0,       0.030),
        ("DAT-qwen3",             180.0,       0.019),
        ("DAT-gpt5.2",            250.0,       0.040),
    ]

    DATASETS = [
        ("NDCG@10", "ACORD",    MODELS),
        ("MRR@10",  "MS MARCO", MODELS),
        ("NDCG@10", "NFCorpus", MODELS),
        ("MRR@10",  "NQ",       MODELS),
    ]

    plot_tradeoff_grid(DATASETS, output="tradeoff_all.png")

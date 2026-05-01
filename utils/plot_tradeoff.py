#!/usr/bin/env python3
"""
Tradeoff plot: per-query inference latency vs. avg metric gain over RRF baseline.

Usage:
    from plot_tradeoff import plot_tradeoff

    models = [
        ("RRF",        0,     0.000),
        ("Mean",       0.25,  0.019),
        ("RoBERTa",    10.0,  0.023),
        ("DAT-GPT5.2", 250.0, 0.040),
    ]
    plot_tradeoff(models, metric_name="NDCG@10", dataset_name="MS MARCO")

Caller only provides (name, latency_ms, metric_value).
Marker shape and color are assigned automatically.
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

matplotlib.rcParams["font.family"] = "monospace"
matplotlib.rcParams["font.size"] = 11

# ---------------------------------------------------------------------------
# Auto-styling config
# ---------------------------------------------------------------------------

# Latency buckets → (marker, filled, legend_label)
# Buckets are checked in order; first match wins.
LATENCY_BUCKETS = [
    (0,   0,   "x",  True,  "no per-query cost"),   # exactly 0
    (0,   1,   "o",  False, "< 1 ms"),               # (0, 1]
    (1,   10,  "o",  True,  "1–10 ms"),              # (1, 10]
    (10,  100, "^",  False,  "10–100 ms"),            # (10, 100]
    (100, float("inf"), "D", True, "100 ms+"),         # (100, ∞)
]

# One color per model, cycled if more models than colors
MODEL_COLORS = [
    "#888888",  # gray
    "#4C72B0",  # blue
    "#55A868",  # green
    "#C44E52",  # red
    "#8172B2",  # purple
    "#CCB974",  # gold
    "#64B5CD",  # cyan
    "#DD8452",  # orange
    "#BAB0AC",  # taupe
]

# X position for zero-latency models on the log scale
ZERO_X = 0.1
X_LIM  = (0.06, 3000)


def _marker_style(latency_ms):
    """Return (marker, filled) for the given latency."""
    for lo, hi, marker, filled, _ in LATENCY_BUCKETS:
        if latency_ms == 0 and lo == 0 and hi == 0:
            return marker, filled
        if latency_ms > 0 and lo == 0 and hi == 0:
            continue
        if lo < latency_ms <= hi:
            return marker, filled
    return "o", True  # fallback


def _latency_bucket_idx(latency_ms):
    """Return the index into LATENCY_BUCKETS for the given latency."""
    for i, (lo, hi, _, _, _) in enumerate(LATENCY_BUCKETS):
        if latency_ms == 0 and lo == 0 and hi == 0:
            return i
        if latency_ms > 0 and lo == 0 and hi == 0:
            continue
        if lo < latency_ms <= hi:
            return i
    return len(LATENCY_BUCKETS)  # fallback: own bucket




def plot_tradeoff(models, metric_name="NDCG@10", dataset_name=None, output="tradeoff_plot.png", delta_mode=True, show=True):
    """
    Parameters
    ----------
    models : list of (name, latency_ms, metric_value)
        latency_ms = 0 means no per-query inference cost.
    metric_name : str
        Label for the y-axis, e.g. "NDCG@10" or "MRR@10".
    dataset_name : str or None
        Optional title shown above the plot.
    output : str
        Output PNG file path.
    """
    fig, ax = plt.subplots(figsize=(9, 6.5))

    y_min = min(v for _, _, v in models)
    y_max = max(v for _, _, v in models)
    y_range = y_max - y_min

    # --- pass 1: collect per-point info ---
    points = []
    for i, (name, latency, value) in enumerate(models):
        color  = MODEL_COLORS[i % len(MODEL_COLORS)]
        x = ZERO_X if latency == 0 else latency
        points.append({"x": x, "y": value, "name": name,
                       "color": color, "latency": latency})

    # --- apply horizontal jitter within each latency bucket ---
    # Points in the same bucket have nearly identical x on the log scale.
    # Spread them by a small multiplicative factor so overlapping markers
    # become distinguishable, while staying visually within their bucket zone.
    LOG_SPREAD = 0.04  # log10 units between adjacent points in the same bucket
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
            points[idx]["x"] = points[idx]["x"] * (10 ** offset)

    # --- pass 2: plot markers at (possibly jittered) x positions ---
    for pt in points:
        marker, filled = _marker_style(pt["latency"])
        fc = pt["color"] if filled else "none"
        ms = 14 if marker == "*" else (11 if marker == "^" else 10)
        ew = 2.0 if marker == "x" else 1.5
        ax.plot(pt["x"], pt["y"], marker=marker, color=pt["color"],
                markerfacecolor=fc, markeredgecolor=pt["color"],
                markersize=ms, markeredgewidth=ew,
                linestyle="none", zorder=3)

    # --- axes setup ---
    pad = y_range * 0.15
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

    # Grid + spines
    ax.yaxis.grid(True, linestyle="-", linewidth=0.4, color="lightgray", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right-side legend: one colored entry per model
    model_handles = []
    for pt in points:
        marker, filled = _marker_style(pt["latency"])
        fc = pt["color"] if filled else "none"
        model_handles.append(
            matplotlib.lines.Line2D([0], [0], marker=marker,
                                    color=pt["color"],
                                    markerfacecolor=fc,
                                    markeredgecolor=pt["color"],
                                    markersize=8, linestyle="none",
                                    label=pt["name"])
        )
    model_leg = ax.legend(handles=model_handles,
                          loc="upper left", bbox_to_anchor=(1.02, 1),
                          fontsize=9, framealpha=0.9, edgecolor="lightgray",
                          title="Models", title_fontsize=9)
    ax.add_artist(model_leg)

    # Bottom-right legend: latency bucket shapes
    used_buckets = {_marker_style(lat) for _, lat, _ in models}
    bucket_handles = []
    for lo, hi, marker, filled, label in LATENCY_BUCKETS:
        if (marker, filled) in used_buckets:
            fc = "black" if filled else "none"
            bucket_handles.append(
                matplotlib.lines.Line2D([0], [0], marker=marker, color="black",
                                        markerfacecolor=fc, markeredgecolor="black",
                                        markersize=8, linestyle="none", label=label)
            )
            used_buckets.discard((marker, filled))
    ax.legend(handles=bucket_handles, loc="lower right", fontsize=9,
              framealpha=0.9, edgecolor="lightgray", title="Latency", title_fontsize=9)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved {output}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Example — replace with your actual measured values
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    MODELS = [
        # name           latency_ms   metric_value
        ("RRF",          0,           0.012),
        ("Mean",         0.25,        0.019),
        ("Ridge",        0.50,        0.019),
        ("RoBERTa",      10.0,        0.023),
        ("ModernBERT",   15.0,        0.017),
        ("LLM-Qwen3",    200.0,       0.030),
        ("DAT-Qwen3",    180.0,       0.019),
        ("DAT-GPT5.2",   250.0,       0.040),
    ]

    plot_tradeoff(MODELS, metric_name="NDCG@10", dataset_name="MS MARCO", output="tradeoff_plot.png")

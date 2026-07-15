"""
Student's t 95% CI for per-query latency averages, per sub-tier.

Each row in the source CSV is a within-dataset mean over hundreds-to-thousands of queries.
By CLT each such input is ~normal, so a t-interval on the collection of inputs is valid.

Sub-tier groupings (which method rows belong together):
    T1b  ridge regression                                                (n=16, one method)
    T2a  moderbert interval + roberta interval                           (n=32, pooled)
    T2b  moderbert passage cond                                          (n=16, one method)
    T3a  DAT-qwen3 + DAT-minstral3                                       (n=32, pooled)
    T3b  llm-fs-qwen-3-interval + llm-fs-mistral-3-interval              (n=32, pooled)

Pooling means: take the union of all 32 within-dataset means and treat them as a single
sample. Between-method variance then enters the sample SD honestly. We do NOT average the
two method means first -- that would throw away most of the information.

Formulas (the n=16 guideline, generalized to arbitrary n):
    mean = (1/n) * sum(x_i)
    sd   = sqrt( (1/(n-1)) * sum((x_i - mean)**2) )    # sample SD
    se   = sd / sqrt(n)                                # SE of the mean
    ci   = mean +/- t(0.975, df=n-1) * se              # 95% two-sided

t critical values used:
    df=15 -> 2.1314  (n=16)
    df=31 -> 2.0395  (n=32)

Reporting: 3 decimal places (ms) for the raw numbers; also a "scientific" form rounded
to the uncertainty's significant figures (1 sig fig, or 2 if leading digit is 1).
"""

import argparse
import csv
import math
from collections import defaultdict
from statistics import mean, stdev

DEFAULT_CSV_PATH = (
    ""
)

# Two-sided 95% t critical values (scipy.stats.t.ppf(0.975, df))
T_CRIT = {
    15: 2.131449545568623,
    31: 2.039513446396525,
}

# Sub-tier groupings: tier label -> (display name, list of model strings in the CSV)
SUBTIERS = [
    ("T1b", "Linear Regression",           ["ridge regression"]),
    ("T2a", "Small Encoder LM (interval)", ["moderbert interval", "roberta interval"]),
    ("T2b", "Small Encoder LM (pass-cond)", ["moderbert passage cond"]),
    ("T3a", "DAT (zero-shot)",             ["DAT-qwen3", "DAT-minstral3"]),
    ("T3b", "Few-shot LLM",                ["llm-fs-qwen-3-interval",
                                            "llm-fs-mistral-3-interval"]),
]


def load_rows(csv_path: str) -> dict[str, list[float]]:
    """method -> list of avg latency (ms) values."""
    by_method: dict[str, list[float]] = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = (row.get("model") or "").strip()
            avg_str = (row.get("avg latency (ms)") or "").strip()
            if not method or not avg_str:
                continue
            try:
                avg = float(avg_str)
            except ValueError:
                continue
            by_method[method].append(avg)
    return by_method


def t_critical(df: int) -> float:
    if df in T_CRIT:
        return T_CRIT[df]
    raise ValueError(f"No tabulated t critical for df={df}; add it to T_CRIT.")


def ci_stats(values: list[float]) -> tuple[int, float, float, float, float, float]:
    n = len(values)
    m = mean(values)
    sd = stdev(values)              # sample SD, denominator n-1
    se = sd / math.sqrt(n)
    t = t_critical(n - 1)
    half = t * se
    return n, m, sd, se, t, half


def round_to_sig(unc: float) -> tuple[float, int]:
    """Round uncertainty to 1 sig fig (2 if leading digit is 1).
    Returns (rounded_value, decimal_places_to_use_for_mean)."""
    if unc == 0:
        return 0.0, 3
    exp = math.floor(math.log10(abs(unc)))
    lead = abs(unc) / (10 ** exp)
    sig = 2 if int(lead) == 1 else 1
    dp = max(0, sig - 1 - exp)
    rounded_unc = round(unc, dp)
    return rounded_unc, dp


def scientific_form(mean_val: float, half_val: float, unit: str = "ms") -> str:
    """Format 'mean ± half' rounding uncertainty to 1-2 sig figs and mean to match."""
    runc, dp = round_to_sig(half_val)
    rmean = round(mean_val, dp)
    if dp == 0:
        return f"{int(round(rmean))} ± {int(round(runc))} {unit}"
    return f"{rmean:.{dp}f} ± {runc:.{dp}f} {unit}"


def maybe_us(mean_ms: float, half_ms: float) -> tuple[float, float, str]:
    """If the value is sub-millisecond, switch to microseconds for readability."""
    if mean_ms < 1.0:
        return mean_ms * 1000.0, half_ms * 1000.0, "µs"
    return mean_ms, half_ms, "ms"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=DEFAULT_CSV_PATH,
                    help="Path to the latency CSV (model, avg latency (ms), ...).")
    args = ap.parse_args()

    by_method = load_rows(args.csv)

    print(f"{'tier':<5} {'name':<30} {'n':>3} {'mean (ms)':>12} {'SD (ms)':>10} "
          f"{'SE (ms)':>10} {'t':>7} {'half (ms)':>11} {'95% CI (ms)':>26}  scientific")
    print("-" * 140)

    for tier, name, methods in SUBTIERS:
        values: list[float] = []
        missing: list[str] = []
        for m_name in methods:
            if m_name not in by_method:
                missing.append(m_name)
            else:
                values.extend(by_method[m_name])
        if missing:
            print(f"WARN: missing methods for {tier}: {missing}")
            continue

        n, m, sd, se, t, half = ci_stats(values)
        lo, hi = m - half, m + half
        m_disp, h_disp, unit = maybe_us(m, half)
        sci = scientific_form(m_disp, h_disp, unit)
        print(f"{tier:<5} {name:<30} {n:>3} {m:>12.3f} {sd:>10.3f} {se:>10.3f} "
              f"{t:>7.4f} {half:>11.3f} "
              f"{'[' + format(lo, '.3f') + ', ' + format(hi, '.3f') + ']':>26}  {sci}")

    print()
    print("Per-method (n=16) reference values:")
    print(f"{'method':<32} {'n':>3} {'mean (ms)':>12} {'SD (ms)':>10} "
          f"{'half (ms)':>11} {'95% CI (ms)':>26}  scientific")
    print("-" * 120)
    for method, xs in by_method.items():
        n, m, sd, se, t, half = ci_stats(xs)
        lo, hi = m - half, m + half
        m_disp, h_disp, unit = maybe_us(m, half)
        sci = scientific_form(m_disp, h_disp, unit)
        print(f"{method:<32} {n:>3} {m:>12.3f} {sd:>10.3f} {half:>11.3f} "
              f"{'[' + format(lo, '.3f') + ', ' + format(hi, '.3f') + ']':>26}  {sci}")


if __name__ == "__main__":
    main()

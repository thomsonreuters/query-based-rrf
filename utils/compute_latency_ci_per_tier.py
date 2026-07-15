"""
Student's t 95% CI for per-query latency averages, per tier (T1/T2/T3).

Same guideline as compute_latency_ci.py, but the pooling is one level coarser:
all methods that belong to a tier are concatenated into a single sample, and we
compute one CI per tier.

Why the t-interval still applies at the tier level:
    Each individual input is still a within-dataset mean over hundreds-to-thousands
    of per-query latencies, so by CLT each input is approximately normal. Pooling
    48 or 64 such inputs widens the effective sample (degrees of freedom grow) and
    lets between-method as well as between-dataset variance enter the sample SD.

Tier groupings:
    T1   ridge regression                                                (n=16,  one method)
    T2   moderbert interval + roberta interval + moderbert passage cond  (n=48,  three methods pooled)
    T3   DAT-qwen3 + DAT-minstral3 + llm-fs-qwen-3-interval
         + llm-fs-mistral-3-interval                                     (n=64,  four methods pooled)

T1a ("Mean Optimal Weight") is not measured in the CSV (reported as < 1 µs from
first principles), so T1 here equals T1b.

Formulas (unchanged):
    mean = (1/n) * sum(x_i)
    sd   = sqrt( (1/(n-1)) * sum((x_i - mean)**2) )
    se   = sd / sqrt(n)
    ci   = mean +/- t(0.975, df=n-1) * se

t critical values used (scipy.stats.t.ppf(0.975, df)):
    df=15 -> 2.1314  (n=16,  T1)
    df=47 -> 2.0117  (n=48,  T2)
    df=63 -> 1.9983  (n=64,  T3)
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
    47: 2.011740513729434,
    63: 1.998340543828851,
}

# Tier groupings: tier label -> (display name, list of model strings in the CSV)
TIERS = [
    ("T1", "Linear Regression (T1b only)",
        ["ridge regression"]),
    ("T2", "Small Encoder LM",
        ["moderbert interval", "roberta interval", "moderbert passage cond"]),
    ("T3", "DAT + Few-shot LLM",
        ["DAT-qwen3", "DAT-minstral3",
         "llm-fs-qwen-3-interval", "llm-fs-mistral-3-interval"]),
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

    print(f"{'tier':<5} {'name':<32} {'n':>3} {'mean (ms)':>12} {'SD (ms)':>10} "
          f"{'SE (ms)':>10} {'t':>7} {'half (ms)':>11} "
          f"{'95% CI (ms)':>26}  scientific")
    print("-" * 140)

    for tier, name, methods in TIERS:
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
        print(f"{tier:<5} {name:<32} {n:>3} {m:>12.3f} {sd:>10.3f} {se:>10.3f} "
              f"{t:>7.4f} {half:>11.3f} "
              f"{'[' + format(lo, '.3f') + ', ' + format(hi, '.3f') + ']':>26}  {sci}")


if __name__ == "__main__":
    main()

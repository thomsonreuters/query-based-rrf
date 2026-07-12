#!/usr/bin/env python3
"""
Runs the three query-analysis steps in sequence with a single, explicit
configuration block:

  Step 1 (1-sort_scores_by_error.py): sort each method/combo CSV by the gap
    between `predicted` and the best-weight interval -> sorted_scores/
  Step 2 (2-top_bottom_predictions_per_method.py): build per-method/dataset
    good/weak prediction files from the top/bottom N% of each combo ->
    top_bottom_predictions/
  Step 3 (3-query-features-analysis.py): compute query text features and
    correlate them with the good/weak label -> query_features_analysis/

Each step is invoked as a subprocess using THIS interpreter (sys.executable,
so it's guaranteed to be the environment with spacy/nltk/scipy installed),
with an explicit argument LIST -- never a shell string -- so there is no
shell-injection surface, and each step's own argparse validation still runs
(e.g. dataset name spelling).

The pipeline is fail-fast: if a step exits non-zero, no later step is run,
and this script exits with that same non-zero code. It also checks each
step's real precondition (e.g. sorted_scores/ must exist and be non-empty
before step 2) rather than trusting a step to fail loudly -- steps 2 and 3
print a message and return 0 if their input directory is simply missing,
which would otherwise look like a silent no-op success.

Edit the CONFIG block below to control which datasets/methods are processed
and each step's parameters. `methods=None` means "all methods discovered by
each step" (each step's own default); to restrict, set an explicit list of
method folder names or numeric prefixes, e.g. ["01", "10"]. `output_root=None`
means each step writes to (and reads from) its own default location -- this
script's directory. Set it to a Path to relocate the whole sorted_scores/,
top_bottom_predictions/, query_features_analysis/ tree elsewhere.

Usage:
    python error-analysis-pipeline.py                 # run all 3 steps
    python error-analysis-pipeline.py --dry-run        # print commands only
    python error-analysis-pipeline.py --steps 3        # re-run step 3 only
    python error-analysis-pipeline.py --steps 2,3      # re-run steps 2 and 3
"""
import argparse
import importlib.util
import subprocess
import sys
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent

STEP1_SCRIPT = ROOT / "1-sort_scores_by_error.py"
STEP2_SCRIPT = ROOT / "2-top_bottom_predictions_per_method.py"
STEP3_SCRIPT = ROOT / "3-query-features-analysis.py"

SORTED_DIRNAME = "sorted_scores"
TOP_BOTTOM_DIRNAME = "top_bottom_predictions"
FEATURES_DIRNAME = "query_features_analysis"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load step 1 as a module (not `import`, since its filename starts with a
# digit) purely to reuse its canonical DATASETS list, discover_methods(), and
# method_matches() -- a single source of truth, so this pipeline can never
# drift out of sync with the actual dataset/method vocabulary the steps use.
_step1_mod = _load_module(STEP1_SCRIPT, "_step1_sort_scores_by_error")
ALL_DATASETS = list(_step1_mod.DATASETS)


# ---------------------------------------------------------------------------
# CONFIG -- edit these to control what the pipeline processes. Validated at
# startup by PipelineConfig.validate() before any step runs.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineConfig:
    datasets: List[str] = field(default_factory=lambda: list(ALL_DATASETS))
    methods: Optional[List[str]] = None  # e.g. ["01", "10"]; None = all methods

    # Where sorted_scores/, top_bottom_predictions/, and query_features_analysis/
    # are read from and written to. None = each step's own default (its own
    # script directory, i.e. this repo's error-analysis/ folder).
    output_root: Optional[Path] = None

    # Step 1 (1-sort_scores_by_error.py)
    step1_ascending: bool = False  # False = largest gap first (step 1's default)

    # Step 2 (2-top_bottom_predictions_per_method.py)
    step2_pct: float = 5.0  # top/bottom N% of rows taken from each combo

    # Step 3 (3-query-features-analysis.py)
    step3_rare_threshold: int = 3
    step3_alpha: float = 0.05

    def resolved_output_root(self) -> Path:
        return self.output_root if self.output_root is not None else ROOT

    def sorted_dir(self) -> Path:
        return self.resolved_output_root() / SORTED_DIRNAME

    def top_bottom_dir(self) -> Path:
        return self.resolved_output_root() / TOP_BOTTOM_DIRNAME

    def features_dir(self) -> Path:
        return self.resolved_output_root() / FEATURES_DIRNAME

    def validate(self):
        if not self.datasets:
            raise ValueError("config.datasets must not be empty.")
        unknown = sorted(set(self.datasets) - set(ALL_DATASETS))
        if unknown:
            raise ValueError(f"Unknown dataset(s) in config: {unknown}. Must be a subset of {ALL_DATASETS}.")
        if self.methods is not None and len(self.methods) == 0:
            raise ValueError("config.methods must be None (all methods) or a non-empty list, not [].")
        if not (0 < self.step2_pct <= 50):
            raise ValueError(f"config.step2_pct should be in (0, 50], got {self.step2_pct}.")
        if self.step3_rare_threshold < 0:
            raise ValueError(f"config.step3_rare_threshold must be >= 0, got {self.step3_rare_threshold}.")
        if not (0 < self.step3_alpha < 1):
            raise ValueError(f"config.step3_alpha should be in (0, 1), got {self.step3_alpha}.")


CONFIG = PipelineConfig(
    datasets=list(ALL_DATASETS),
    methods=None,
    output_root=None,
    step1_ascending=False,
    step2_pct=5.0,
    step3_rare_threshold=3,
    step3_alpha=0.05,
)


# ---------------------------------------------------------------------------
# Step command builders. Each returns an explicit argv list (no shell=True).
# ---------------------------------------------------------------------------

def _method_args(methods: Optional[List[str]]) -> List[str]:
    return ["--methods", *methods] if methods else []


def _output_root_args(config: PipelineConfig) -> List[str]:
    return ["--output-root", str(config.output_root)] if config.output_root is not None else []


def build_step1_cmd(config: PipelineConfig) -> List[str]:
    cmd = [sys.executable, str(STEP1_SCRIPT), "--datasets", *config.datasets]
    cmd += _method_args(config.methods)
    if config.step1_ascending:
        cmd.append("--ascending")
    cmd += _output_root_args(config)
    return cmd


def build_step2_cmd(config: PipelineConfig) -> List[str]:
    cmd = [sys.executable, str(STEP2_SCRIPT), "--datasets", *config.datasets]
    cmd += _method_args(config.methods)
    cmd += ["--pct", str(config.step2_pct)]
    cmd += _output_root_args(config)
    return cmd


def build_step3_cmd(config: PipelineConfig) -> List[str]:
    cmd = [sys.executable, str(STEP3_SCRIPT), "--datasets", *config.datasets]
    cmd += _method_args(config.methods)
    cmd += ["--rare-threshold", str(config.step3_rare_threshold)]
    cmd += ["--alpha", str(config.step3_alpha)]
    cmd += _output_root_args(config)
    return cmd


def _precondition_sorted_dir_exists(config: PipelineConfig):
    sorted_dir = config.sorted_dir()
    if not sorted_dir.exists() or not any(sorted_dir.iterdir()):
        raise RuntimeError(
            f"{sorted_dir}/ is missing or empty -- step 2 needs step 1's output. "
            f"Run with --steps 1,2,3 (or at least include step 1)."
        )


def _precondition_top_bottom_dir_exists(config: PipelineConfig):
    top_bottom_dir = config.top_bottom_dir()
    if not top_bottom_dir.exists() or not any(top_bottom_dir.iterdir()):
        raise RuntimeError(
            f"{top_bottom_dir}/ is missing or empty -- step 3 needs step 2's output. "
            f"Run with --steps 2,3 (or at least include step 2)."
        )


def _precondition_step3_dependencies(_config: PipelineConfig):
    """Step 3 needs spaCy's en_core_web_sm model and nltk's wordnet corpus.
    Checked up front so a missing dependency fails immediately with a clear
    remediation message, instead of after steps 1/2 have already run."""
    missing = []
    try:
        import spacy
        try:
            spacy.load("en_core_web_sm")
        except OSError:
            missing.append("spaCy model 'en_core_web_sm' -- run: python -m spacy download en_core_web_sm")
    except ImportError:
        missing.append("spacy package -- run: pip install spacy")
    try:
        from nltk.corpus import wordnet
        wordnet.synsets("test")
    except LookupError:
        missing.append("nltk 'wordnet' corpus -- run: python -c \"import nltk; nltk.download('wordnet')\"")
    except ImportError:
        missing.append("nltk package -- run: pip install nltk")
    try:
        import scipy  # noqa: F401
    except ImportError:
        missing.append("scipy package -- run: pip install scipy")
    if missing:
        raise RuntimeError("Step 3 dependencies missing:\n  - " + "\n  - ".join(missing))


STEPS = {
    1: dict(label="sort scores by error", script=STEP1_SCRIPT, build_cmd=build_step1_cmd,
            preconditions=[]),
    2: dict(label="build top/bottom predictions", script=STEP2_SCRIPT, build_cmd=build_step2_cmd,
            preconditions=[_precondition_sorted_dir_exists]),
    3: dict(label="query features analysis", script=STEP3_SCRIPT, build_cmd=build_step3_cmd,
            preconditions=[_precondition_top_bottom_dir_exists, _precondition_step3_dependencies]),
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_step(step_num: int, config: PipelineConfig, dry_run: bool) -> int:
    spec = STEPS[step_num]
    label, script_path, build_cmd = spec["label"], spec["script"], spec["build_cmd"]
    if not script_path.exists():
        raise FileNotFoundError(f"Step {step_num} script not found: {script_path}")

    print(f"\n{'=' * 70}\nStep {step_num}: {label}\n{'=' * 70}")

    if not dry_run:
        for precondition in spec["preconditions"]:
            precondition(config)

    cmd = build_cmd(config)
    print("$ " + " ".join(cmd))
    if dry_run:
        print("(dry run -- not executed)")
        return 0

    start = time.monotonic()
    result = subprocess.run(cmd)
    elapsed = time.monotonic() - start
    print(f"--- step {step_num} finished in {elapsed:.1f}s with exit code {result.returncode} ---")
    return result.returncode


def parse_steps_arg(raw: str) -> List[int]:
    try:
        steps = sorted({int(s) for s in raw.split(",")})
    except ValueError:
        raise argparse.ArgumentTypeError(f"--steps must be comma-separated integers, got: {raw!r}")
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown step number(s): {unknown}. Valid steps: {sorted(STEPS)}")
    return steps


def main():
    # Without this, the parent's own print()s are block-buffered whenever
    # stdout isn't a TTY (e.g. piped to a log file), while each subprocess's
    # inherited stdout is not -- so status lines and step output interleave
    # out of order. Line-buffering keeps everything in the order it's emitted.
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=parse_steps_arg, default=sorted(STEPS),
                         help="Comma-separated step numbers to (re)run, e.g. '3' or '2,3'. Default: 1,2,3.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print the commands that would run, without executing them or checking preconditions.")
    args = parser.parse_args()

    CONFIG.validate()

    matched_methods = [
        m.name for m in _step1_mod.discover_methods() if _step1_mod.method_matches(m, CONFIG.methods)
    ]
    if not matched_methods:
        print(f"CONFIG.methods={CONFIG.methods!r} matched no method folders under {ROOT}. Aborting.")
        sys.exit(1)

    print("Pipeline configuration:")
    for f in fields(CONFIG):
        print(f"  {f.name} = {getattr(CONFIG, f.name)}")
    print(f"  steps to run = {args.steps}")
    print(f"  methods matched by config.methods: {matched_methods}")
    if args.dry_run:
        print("  (DRY RUN -- no scripts will be executed, no preconditions checked)")

    overall_start = time.monotonic()
    for step_num in args.steps:
        try:
            rc = run_step(step_num, CONFIG, args.dry_run)
        except (RuntimeError, FileNotFoundError) as exc:
            print(f"\nPipeline ABORTED before step {step_num} could run: {exc}")
            sys.exit(1)
        if rc != 0:
            print(f"\nPipeline ABORTED: step {step_num} exited with code {rc}. Later steps were not run.")
            sys.exit(rc)

    overall_elapsed = time.monotonic() - overall_start
    print(f"\n{'=' * 70}")
    if args.dry_run:
        print("Dry run complete -- no scripts were executed.")
    else:
        print(f"Pipeline complete in {overall_elapsed:.1f}s.")
        print("Outputs:")
        print(f"  {CONFIG.sorted_dir()}/       (step 1)")
        print(f"  {CONFIG.top_bottom_dir()}/   (step 2)")
        print(f"  {CONFIG.features_dir()}/     (step 3)")


if __name__ == "__main__":
    main()

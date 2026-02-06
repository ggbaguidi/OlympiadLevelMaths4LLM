#!/usr/bin/env python3
"""Cross-validation script for AIMO3 configurations.

Usage:
    python scripts/cross_val_configs.py --problems reference.csv --configs all
    python scripts/cross_val_configs.py --problems reference.csv --configs A,B
    python scripts/cross_val_configs.py --problems reference.csv --problem-ids "id1,id2"
    python scripts/cross_val_configs.py --list-configs

This script tests different solver configurations on a set of problems
and compares results to find the best settings for hard problems.

Expected CSV format:
    id,problem,answer
    prob1,"Find x such that...",42
    prob2,"How many...",123
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add src/ to path for local development (no pip install needed)
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import pandas as pd

# Configuration presets for cross-validation
CONFIG_PRESETS: dict[str, dict[str, Any]] = {
    # Baseline: minimal features, simple config (your 38/50 submission)
    "baseline": {
        "name": "Baseline (Simple)",
        "env": {
            "AIMO3_WICKELGREN": "0",
            "AIMO3_PROTOCOL": "0",
            "AIMO3_SECOND_STAGE_VERIFY_ENABLED": "0",
            "AIMO3_CONTRADICTION_RETRY_ENABLED": "0",
            "AIMO3_TIEBREAK_ENABLED": "0",
            "AIMO3_ATTEMPTS": "8",
            "AIMO3_TEMPERATURE": "0.95",
            "AIMO3_EARLY_STOP": "5",
        },
    },
    # Maximum exploration for hard problems
    "exploration": {
        "name": "Maximum Exploration",
        "env": {
            "AIMO3_ATTEMPTS": "16",
            "AIMO3_TEMPERATURE": "0.95",
            "AIMO3_TEMPERATURE_EXPLORATION": "1.0",
            "AIMO3_EXPLORATION_ATTEMPTS": "4",
            "AIMO3_ADAPTIVE_BUDGET_MAX_EXTENSION": "2.5",
            "AIMO3_EARLY_STOP": "5",
            "AIMO3_EARLY_STOP_MIN_VERIFIED": "2",
            "AIMO3_CONCLUDE_NUDGE_TOKENS": "18000",
        },
    },
    # Deep reasoning with more turns
    "deep_reasoning": {
        "name": "Deep Reasoning",
        "env": {
            "AIMO3_ATTEMPTS": "8",
            "AIMO3_TEMPERATURE": "0.7",
            "AIMO3_TEMPERATURE_MAIN": "0.6",
            "AIMO3_TURNS": "256",
            "AIMO3_CONCLUDE_NUDGE_TOKENS": "20000",
            "AIMO3_BASE_PROBLEM_TIMEOUT": "450",
            "AIMO3_JUPYTER_TIMEOUT": "45",
        },
    },
    # Tool-heavy for computational problems
    "tool_heavy": {
        "name": "Tool-Heavy (Computational)",
        "env": {
            "AIMO3_JUPYTER_TIMEOUT": "60",
            "AIMO3_PYTHON_TOOL_TIMEOUT_CAP_S": "300",
            "AIMO3_ABORT_ATTEMPT_AFTER_PYTHON_ERRORS": "6",
            "AIMO3_DISABLE_PROMPTS": "analytic",
            "AIMO3_ATTEMPTS": "12",
            "AIMO3_CONCLUDE_NUDGE_TOKENS": "14000",
        },
    },
    # Low temperature for precision
    "precise": {
        "name": "Precise (Low Temp)",
        "env": {
            "AIMO3_TEMPERATURE": "0.6",
            "AIMO3_TEMPERATURE_EXPLORATION": "0.7",
            "AIMO3_TEMPERATURE_MAIN": "0.5",
            "AIMO3_ATTEMPTS": "8",
            "AIMO3_EARLY_STOP": "3",
            "AIMO3_EARLY_STOP_MIN_VERIFIED": "2",
        },
    },
    # High creativity
    "creative": {
        "name": "Creative (High Temp)",
        "env": {
            "AIMO3_TEMPERATURE": "1.0",
            "AIMO3_TEMPERATURE_EXPLORATION": "1.0",
            "AIMO3_TEMPERATURE_MAIN": "0.9",
            "AIMO3_ATTEMPTS": "12",
            "AIMO3_EXPLORATION_ATTEMPTS": "6",
            "AIMO3_EARLY_STOP": "4",
        },
    },
    # Aggressive early exit (for easy problems)
    "fast_easy": {
        "name": "Fast Easy Exit",
        "env": {
            "AIMO3_EASY_EXIT_ENABLED": "1",
            "AIMO3_EASY_EXIT_TIME_THRESHOLD_S": "45",
            "AIMO3_EASY_EXIT_MIN_VOTES": "2",
            "AIMO3_EASY_EXIT_MIN_VERIFIED": "1",
            "AIMO3_EARLY_STOP": "3",
            "AIMO3_ATTEMPTS": "8",
        },
    },
    # Maximum budget for hard problems
    "max_budget": {
        "name": "Maximum Budget",
        "env": {
            "AIMO3_ADAPTIVE_BUDGET_ENABLED": "1",
            "AIMO3_ADAPTIVE_BUDGET_FLEX_POOL_FRACTION": "0.25",
            "AIMO3_ADAPTIVE_BUDGET_MAX_EXTENSION": "3.0",
            "AIMO3_ADAPTIVE_BUDGET_HARDNESS_TRIGGER": "0.4",
            "AIMO3_BASE_PROBLEM_TIMEOUT": "400",
            "AIMO3_HIGH_PROBLEM_TIMEOUT": "1200",
            "AIMO3_ATTEMPTS": "12",
        },
    },
    # All features enabled (current default)
    "full_features": {
        "name": "Full Features",
        "env": {
            "AIMO3_WICKELGREN": "1",
            "AIMO3_PROTOCOL": "1",
            "AIMO3_SECOND_STAGE_VERIFY_ENABLED": "1",
            "AIMO3_CONTRADICTION_RETRY_ENABLED": "1",
            "AIMO3_TIEBREAK_ENABLED": "1",
            "AIMO3_ADAPTIVE_BUDGET_ENABLED": "1",
            "AIMO3_CONCLUDE_NUDGE_ENABLED": "1",
            "AIMO3_EASY_EXIT_ENABLED": "1",
            "AIMO3_ATTEMPTS": "8",
        },
    },
    # No verification overhead
    "no_verify": {
        "name": "No Verification",
        "env": {
            "AIMO3_SECOND_STAGE_VERIFY_ENABLED": "0",
            "AIMO3_PYTHON_TOOL_VERIFY_REQUIRE_MARKER": "0",
            "AIMO3_TIEBREAK_ENABLED": "0",
            "AIMO3_ATTEMPTS": "10",
            "AIMO3_EARLY_STOP": "4",
        },
    },
}


@dataclass
class ProblemResult:
    """Result of solving one problem with one config."""

    problem_id: str
    config_name: str
    answer: int | None
    correct: bool | None  # None if ground truth unknown
    elapsed_s: float
    n_attempts: int
    n_distinct_answers: int
    top_votes: int
    verified_count: int
    extension_granted: bool = False


@dataclass
class ConfigResult:
    """Aggregated results for one config across all problems."""

    config_name: str
    problems: list[ProblemResult] = field(default_factory=list)

    @property
    def n_solved(self) -> int:
        return sum(1 for p in self.problems if p.correct is True)

    @property
    def n_answered(self) -> int:
        return sum(1 for p in self.problems if p.answer is not None)

    @property
    def accuracy(self) -> float:
        if not self.problems:
            return 0.0
        known = [p for p in self.problems if p.correct is not None]
        if not known:
            return 0.0
        return self.n_solved / len(known)

    @property
    def avg_time_s(self) -> float:
        if not self.problems:
            return 0.0
        return sum(p.elapsed_s for p in self.problems) / len(self.problems)

    @property
    def extensions_used(self) -> int:
        return sum(1 for p in self.problems if p.extension_granted)


def list_configs() -> None:
    """Print available configuration presets."""
    print("\n📋 Available Configuration Presets:\n")
    print("-" * 70)
    for key, cfg in CONFIG_PRESETS.items():
        print(f"\n🔧 {key}: {cfg['name']}")
        print("   Environment variables:")
        for k, v in cfg["env"].items():
            print(f"      {k}={v}")
    print("\n" + "-" * 70)


def load_problems(
    path: str, problem_ids: list[str] | None = None
) -> list[dict[str, Any]]:
    """Load problems from CSV or JSON file.

    CSV format (preferred):
        id,problem,answer
        prob1,"Find x such that...",42

    JSON format (legacy):
        [{"id": "prob1", "text": "...", "answer": 42}, ...]
    """
    path_obj = Path(path)

    if path_obj.suffix.lower() == ".csv":
        df = pd.read_csv(path)

        # Filter by problem IDs if specified
        if problem_ids:
            df = df[df["id"].isin(problem_ids)]

        problems = []
        for _, row in df.iterrows():
            prob = {
                "id": str(row["id"]),
                "text": str(row.get("problem", row.get("text", ""))),
            }
            if "answer" in row and pd.notna(row["answer"]):
                prob["answer"] = int(row["answer"])
            problems.append(prob)
        return problems

    else:
        # JSON format
        with open(path) as f:
            data = json.load(f)

        if isinstance(data, list):
            problems = data
        elif isinstance(data, dict) and "problems" in data:
            problems = data["problems"]
        else:
            raise ValueError(f"Unknown problems format in {path}")

        # Filter by problem IDs if specified
        if problem_ids:
            problems = [p for p in problems if p.get("id") in problem_ids]

        return problems


def run_single_problem(
    problem: dict[str, Any],
    config_name: str,
    config_env: dict[str, str],
    trace_dir: Path,
) -> ProblemResult:
    """Run solver on a single problem with given config.

    Note: This requires the full solver stack. For lightweight testing,
    you may want to mock this or use a simpler evaluation.
    """
    from olympiad_llm.aimo3.config import AIMO3Config
    from olympiad_llm.aimo3.solver import AIMO3Solver

    # Set environment variables for this config
    old_env = {}
    for k, v in config_env.items():
        old_env[k] = os.environ.get(k)
        os.environ[k] = v

    # Enable tracing
    trace_path = trace_dir / f"{config_name}_{problem.get('id', 'unknown')}.jsonl"
    os.environ["AIMO3_TRACE"] = "1"
    os.environ["AIMO3_TRACE_PATH"] = str(trace_path)
    os.environ["AIMO3_TRACE_RESET_ON_START"] = "1"

    try:
        cfg = AIMO3Config.from_env()
        solver = AIMO3Solver(cfg=cfg)

        start = time.time()
        answer = solver.solve_problem(problem["text"])
        elapsed = time.time() - start

        solver.close()

        # Parse trace for details
        n_attempts = 0
        n_distinct = 0
        top_votes = 0
        verified_count = 0
        extension_granted = False

        if trace_path.exists():
            with open(trace_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        if rec.get("event") == "solve_end":
                            n_attempts = rec.get("n_attempts", 0)
                            decision = rec.get("decision", {})
                            ranked = decision.get("ranked", [])
                            if ranked:
                                top_votes = ranked[0].get("votes", 0)
                                verified_count = ranked[0].get("verified", 0)
                            n_distinct = len(ranked)
                        elif rec.get("event") == "budget_extension":
                            extension_granted = True
                    except json.JSONDecodeError:
                        continue

        # Check correctness if ground truth available
        correct = None
        if "answer" in problem:
            correct = answer == problem["answer"]

        return ProblemResult(
            problem_id=problem.get("id", "unknown"),
            config_name=config_name,
            answer=answer,
            correct=correct,
            elapsed_s=elapsed,
            n_attempts=n_attempts,
            n_distinct_answers=n_distinct,
            top_votes=top_votes,
            verified_count=verified_count,
            extension_granted=extension_granted,
        )

    finally:
        # Restore environment
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def run_cross_validation(
    problems: list[dict[str, Any]],
    config_names: list[str],
    trace_dir: Path,
    dry_run: bool = False,
) -> dict[str, ConfigResult]:
    """Run cross-validation across configs and problems."""

    results: dict[str, ConfigResult] = {}

    for cfg_name in config_names:
        if cfg_name not in CONFIG_PRESETS:
            print(f"⚠️  Unknown config: {cfg_name}, skipping")
            continue

        cfg = CONFIG_PRESETS[cfg_name]
        print(f"\n{'='*60}")
        print(f"🔧 Testing config: {cfg_name} ({cfg['name']})")
        print(f"{'='*60}")

        results[cfg_name] = ConfigResult(config_name=cfg_name)

        for i, problem in enumerate(problems):
            pid = problem.get("id", f"problem_{i}")
            print(f"\n  [{i+1}/{len(problems)}] Problem: {pid}")

            if dry_run:
                print(f"    [DRY RUN] Would test with: {list(cfg['env'].keys())}")
                continue

            try:
                result = run_single_problem(
                    problem=problem,
                    config_name=cfg_name,
                    config_env=cfg["env"],
                    trace_dir=trace_dir,
                )
                results[cfg_name].problems.append(result)

                status = (
                    "✅"
                    if result.correct
                    else ("❌" if result.correct is False else "❓")
                )
                print(
                    f"    {status} Answer: {result.answer} | Time: {result.elapsed_s:.1f}s | "
                    f"Attempts: {result.n_attempts} | Votes: {result.top_votes}"
                )

            except Exception as e:
                print(f"    ❌ Error: {e}")

    return results


def print_summary(results: dict[str, ConfigResult]) -> None:
    """Print summary comparison of all configs."""

    print("\n" + "=" * 80)
    print("📊 CROSS-VALIDATION SUMMARY")
    print("=" * 80)

    # Header
    print(
        f"\n{'Config':<20} {'Accuracy':<12} {'Answered':<12} {'Avg Time':<12} {'Extensions':<12}"
    )
    print("-" * 68)

    # Sort by accuracy descending
    sorted_results = sorted(results.values(), key=lambda r: r.accuracy, reverse=True)

    for r in sorted_results:
        n_problems = len(r.problems)
        acc_str = f"{r.accuracy*100:.1f}%" if n_problems > 0 else "N/A"
        ans_str = f"{r.n_answered}/{n_problems}" if n_problems > 0 else "N/A"
        time_str = f"{r.avg_time_s:.1f}s" if n_problems > 0 else "N/A"
        ext_str = f"{r.extensions_used}" if n_problems > 0 else "N/A"

        print(
            f"{r.config_name:<20} {acc_str:<12} {ans_str:<12} {time_str:<12} {ext_str:<12}"
        )

    print("\n" + "-" * 68)

    # Best config
    if sorted_results and sorted_results[0].problems:
        best = sorted_results[0]
        print(
            f"\n🏆 Best config: {best.config_name} ({best.accuracy*100:.1f}% accuracy)"
        )
        print(f"   Settings: {CONFIG_PRESETS[best.config_name]['env']}")


def create_sample_problems_file(path: str) -> None:
    """Create a sample problems CSV file for testing."""
    sample_data = {
        "id": ["sample_1", "sample_2", "sample_3"],
        "problem": [
            "Find the remainder when 2^100 is divided by 7.",
            "How many positive integers less than 1000 are divisible by neither 2 nor 3?",
            "Let f(x) = x^2 + 1. Find f(f(f(1))).",
        ],
        "answer": [2, 333, 677],
    }

    df = pd.DataFrame(sample_data)
    df.to_csv(path, index=False)

    print(f"✅ Created sample problems file: {path}")
    print(f"   Format: id, problem, answer")
    print(f"   Problems: {len(df)}")


def main():
    parser = argparse.ArgumentParser(
        description="Cross-validation script for AIMO3 configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available configs
  python scripts/cross_val_configs.py --list-configs
  
  # Create sample problems file (CSV)
  python scripts/cross_val_configs.py --create-sample problems.csv
  
  # Run all configs (dry run)
  python scripts/cross_val_configs.py --problems reference.csv --configs all --dry-run
  
  # Run specific configs
  python scripts/cross_val_configs.py --problems reference.csv --configs baseline,exploration
  
  # Test specific problems only
  python scripts/cross_val_configs.py --problems reference.csv --problem-ids "id1,id2,id3" --configs baseline
  
  # Full run with output
  python scripts/cross_val_configs.py --problems reference.csv --configs all --output results.json

CSV format:
  id,problem,answer
  prob1,"Find x such that...",42
  prob2,"How many integers...",123
        """,
    )

    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available configuration presets",
    )
    parser.add_argument(
        "--create-sample",
        type=str,
        metavar="PATH",
        help="Create a sample problems CSV file",
    )
    parser.add_argument(
        "--problems", type=str, help="Path to problems CSV or JSON file"
    )
    parser.add_argument(
        "--problem-ids", type=str, help="Comma-separated problem IDs to test (optional)"
    )
    parser.add_argument(
        "--configs",
        type=str,
        default="all",
        help="Comma-separated config names or 'all'",
    )
    parser.add_argument(
        "--trace-dir",
        type=str,
        default="tmp/cross_val_traces",
        help="Directory for trace files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without running",
    )
    parser.add_argument("--output", type=str, help="Output results to JSON file")

    args = parser.parse_args()

    if args.list_configs:
        list_configs()
        return

    if args.create_sample:
        create_sample_problems_file(args.create_sample)
        return

    if not args.problems:
        parser.print_help()
        print("\n❌ Error: --problems is required")
        return

    # Parse config names
    if args.configs.lower() == "all":
        config_names = list(CONFIG_PRESETS.keys())
    else:
        config_names = [c.strip() for c in args.configs.split(",")]

    # Parse problem IDs if specified
    problem_ids = None
    if args.problem_ids:
        problem_ids = [pid.strip() for pid in args.problem_ids.split(",")]

    # Load problems
    problems = load_problems(args.problems, problem_ids=problem_ids)
    filter_info = f" (filtered to {len(problem_ids)} IDs)" if problem_ids else ""
    print(f"📚 Loaded {len(problems)} problems from {args.problems}{filter_info}")
    print(f"🔧 Testing {len(config_names)} configs: {config_names}")

    # Create trace directory
    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)

    # Run cross-validation
    results = run_cross_validation(
        problems=problems,
        config_names=config_names,
        trace_dir=trace_dir,
        dry_run=args.dry_run,
    )

    # Print summary
    if not args.dry_run:
        print_summary(results)

        # Save results
        if args.output:
            output_data = {
                cfg_name: {
                    "accuracy": r.accuracy,
                    "n_answered": r.n_answered,
                    "avg_time_s": r.avg_time_s,
                    "extensions_used": r.extensions_used,
                    "problems": [
                        {
                            "id": p.problem_id,
                            "answer": p.answer,
                            "correct": p.correct,
                            "elapsed_s": p.elapsed_s,
                        }
                        for p in r.problems
                    ],
                }
                for cfg_name, r in results.items()
            }
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\n💾 Results saved to {args.output}")


if __name__ == "__main__":
    main()

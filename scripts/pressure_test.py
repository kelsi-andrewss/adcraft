"""Pressure test script for the AdCraft tuning loop.

Automates calibration, smoke testing, full batch runs, and reporting.
Produces a summary with dimension breakdown and rubric tuning guidance.

Usage:
    uv run python scripts/pressure_test.py --stage all
    uv run python scripts/pressure_test.py --stage calibrate
    uv run python scripts/pressure_test.py --stage smoke --rpm 8
"""

from __future__ import annotations

import argparse
import time

from dotenv import load_dotenv

load_dotenv()

from src.analytics.cost import get_performance_per_token  # noqa: E402
from src.db.init_db import init_db  # noqa: E402
from src.db.queries import get_dimension_averages  # noqa: E402
from src.evaluate.calibrate import run_calibration  # noqa: E402
from src.models.calibration import CalibrationResult  # noqa: E402
from src.output.exporter import (  # noqa: E402
    export_ad_library,
    export_decision_log,
    export_summary_stats,
)
from src.pipeline.main import BatchPipeline  # noqa: E402

WEAK_DIMENSION_THRESHOLD = 6.0


def print_header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_dimension_breakdown(dim_avgs: dict[str, float]) -> list[str]:
    """Print dimension averages and return names of weak dimensions."""
    weak = []
    if not dim_avgs:
        print("  (no evaluation data yet)")
        return weak

    print(f"  {'Dimension':<30} {'Avg Score':>10}  {'Status':>8}")
    print(f"  {'-' * 50}")
    for dim, avg in sorted(dim_avgs.items()):
        status = "OK" if avg >= WEAK_DIMENSION_THRESHOLD else "WEAK"
        if status == "WEAK":
            weak.append(dim)
        print(f"  {dim:<30} {avg:>10.2f}  {status:>8}")
    return weak


def stage_calibrate() -> CalibrationResult | None:
    """Run calibration against the gold set and return structured metrics."""
    print_header("STAGE 1: CALIBRATION")
    try:
        result = run_calibration()
        print(f"\n  Alpha: {result.alpha_overall:.3f}")
        print(f"  Spearman rho: {result.spearman_rho:.3f}")
        for dim, mae in result.per_dimension_mae.items():
            print(f"  MAE {dim}: {mae:.2f}")
        print(f"  Status: {'PASSED' if result.passed else 'FAILED'}")
        if not result.passed:
            print("  Suggested fixes:")
            print("    - Review data/reference_ads/calibration_gold_set.json for label accuracy")
            print("    - Adjust dimension weights in src/evaluate/rubrics.py")
            print("    - Check evaluator prompt for scoring drift")
        return result
    except FileNotFoundError as exc:
        print(f"\n  Calibration ERROR: {exc}")
        print("  Fix: ensure data/reference_ads/calibration_gold_set.json exists with gold set ads")
        return None
    except Exception as exc:
        print(f"\n  Calibration ERROR: {type(exc).__name__}: {exc}")
        print("  Check GEMINI_API_KEY is set and the evaluation engine is functional")
        return None


def stage_smoke(db_path: str, rpm: int) -> dict | None:
    """Run 3 briefs through the full pipeline."""
    print_header("STAGE 2: SMOKE TEST (3 briefs)")
    try:
        pipeline = BatchPipeline(db_path=db_path, rpm_limit=rpm)
        briefs = pipeline.generate_brief_matrix()[:3]
        result = pipeline.run(briefs)
        print(f"\n  Passed: {result.passed}/{result.total_briefs}")
        print(f"  Avg score: {result.avg_score:.2f}")
        print(f"  Duration: {result.duration_seconds:.1f}s")
        return {
            "total": result.total_briefs,
            "passed": result.passed,
            "failed": result.failed,
            "errors": result.errors,
            "avg_score": result.avg_score,
            "duration": result.duration_seconds,
        }
    except Exception as exc:
        print(f"\n  Smoke test ERROR: {type(exc).__name__}: {exc}")
        return None


def stage_full(db_path: str, rpm: int) -> dict | None:
    """Run all seed briefs through the full pipeline."""
    print_header("STAGE 3: FULL BATCH (all seed briefs)")
    try:
        pipeline = BatchPipeline(db_path=db_path, rpm_limit=rpm)
        result = pipeline.run()
        print(f"\n  Passed: {result.passed}/{result.total_briefs}")
        print(f"  Failed: {result.failed} (errors: {result.errors})")
        print(f"  Avg score: {result.avg_score:.2f}")
        print(f"  Total cycles: {result.total_cycles}")
        print(f"  Duration: {result.duration_seconds:.1f}s")
        return {
            "total": result.total_briefs,
            "passed": result.passed,
            "failed": result.failed,
            "errors": result.errors,
            "avg_score": result.avg_score,
            "total_cycles": result.total_cycles,
            "duration": result.duration_seconds,
            "total_cost_usd": result.total_cost_usd,
        }
    except Exception as exc:
        print(f"\n  Full batch ERROR: {type(exc).__name__}: {exc}")
        return None


def stage_report(
    db_path: str,
    calibration_result: CalibrationResult | None,
    smoke_result: dict | None,
    full_result: dict | None,
) -> None:
    """Print summary report with dimension breakdown and export results."""
    print_header("REPORT")

    conn = init_db(db_path)

    # Calibration status and reliability metrics
    if calibration_result is None:
        print("  Calibration: SKIPPED")
    else:
        status = "PASSED" if calibration_result.passed else "FAILED"
        print(f"  Calibration: {status}")
        print(f"    Krippendorff's Alpha: {calibration_result.alpha_overall:.4f}")
        print(f"    Spearman rho:         {calibration_result.spearman_rho:.4f}")
        print("    Per-dimension MAE:")
        for dim, mae in sorted(calibration_result.per_dimension_mae.items()):
            print(f"      {dim:<25} {mae:.3f}")

    # Smoke test summary
    if smoke_result is not None:
        rate = smoke_result["passed"] / smoke_result["total"] if smoke_result["total"] else 0
        print(
            f"\n  Smoke test: {smoke_result['passed']}/{smoke_result['total']} "
            f"passed ({rate:.0%}), avg={smoke_result['avg_score']:.2f}"
        )

    # Full batch summary
    if full_result is not None:
        rate = full_result["passed"] / full_result["total"] if full_result["total"] else 0
        print(
            f"\n  Full batch: {full_result['passed']}/{full_result['total']} "
            f"passed ({rate:.0%}), avg={full_result['avg_score']:.2f}"
        )
        print(f"  Total cycles: {full_result.get('total_cycles', 'N/A')}")
        print(f"  Duration: {full_result['duration']:.1f}s")
        print(f"  Estimated cost: ${full_result.get('total_cost_usd', 0):.4f}")

    # Cost trend from quality snapshots
    snapshots = get_performance_per_token(conn)
    if snapshots:
        latest = snapshots[0]
        qpd = latest.get("quality_per_dollar")
        spend = latest.get("token_spend_usd")
        if qpd is not None:
            print(f"\n  Latest snapshot: spend=${spend or 0:.4f}, quality/dollar={qpd:.2f}")
        elif spend is not None:
            print(f"\n  Latest snapshot: spend=${spend:.4f}, quality/dollar=N/A")

    # Dimension breakdown
    print("\n  Dimension Averages:")
    dim_avgs = get_dimension_averages(conn)
    weak_dims = print_dimension_breakdown(dim_avgs)

    # Export results after full batch
    if full_result is not None:
        print("\n  Exporting results to output/...")
        export_ad_library(conn, "csv", "output/ad_library.csv")
        export_ad_library(conn, "json", "output/ad_library.json")
        export_decision_log(conn, "output/decisions.json")
        export_summary_stats(conn, "output/summary_stats.json")
        print("  Exported: ad_library.csv, ad_library.json, decisions.json, summary_stats.json")

    # Rubric tuning guide
    print(f"\n{'=' * 70}")
    if weak_dims:
        print(f"  Weak dimensions: {', '.join(weak_dims)}")
        print("  See src/evaluate/rubrics.py to tighten scoring criteria.")
    else:
        print("  All dimensions above threshold. No rubric changes needed.")
    print(f"{'=' * 70}")

    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="AdCraft pressure test — tuning loop automation")
    parser.add_argument(
        "--stage",
        choices=["calibrate", "smoke", "full", "all"],
        default="all",
        help="Which stage(s) to run (default: all)",
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=10,
        help="Rate limit: requests per minute (default: 10)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="data/ads.db",
        help="Database path (default: data/ads.db)",
    )
    args = parser.parse_args()

    start = time.time()
    calibration_result: CalibrationResult | None = None
    smoke_result: dict | None = None
    full_result: dict | None = None

    stages = ["calibrate", "smoke", "full"] if args.stage == "all" else [args.stage]

    for stage in stages:
        if stage == "calibrate":
            calibration_result = stage_calibrate()
        elif stage == "smoke":
            smoke_result = stage_smoke(args.db, args.rpm)
        elif stage == "full":
            full_result = stage_full(args.db, args.rpm)

    stage_report(args.db, calibration_result, smoke_result, full_result)

    elapsed = time.time() - start
    print(f"\n  Total pressure test time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

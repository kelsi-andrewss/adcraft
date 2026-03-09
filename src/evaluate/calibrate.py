"""Calibration script for AdCraft evaluator.

Run with: uv run python -m src.evaluate.calibrate

Loads reference ads from data/reference_ads/labeled_ads.json, runs each through
both evaluation modes (iteration and final), prints a score table, and asserts
that all "great" ads score >= 7.0 and all "bad" ads score < 5.0.

Requires GEMINI_API_KEY in environment or .env file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.decisions.logger import log_decision  # noqa: E402
from src.evaluate.engine import EvaluationEngine  # noqa: E402
from src.models.ad import AdCopy  # noqa: E402


def load_reference_ads() -> list[dict]:
    """Load labeled reference ads from JSON file."""
    path = Path("data/reference_ads/labeled_ads.json")
    with open(path) as f:
        data = json.load(f)
    return data["reference_ads"]


def run_calibration() -> bool:
    """Run calibration and return True if all assertions pass."""
    engine = EvaluationEngine()
    reference_ads = load_reference_ads()

    results: list[dict] = []
    all_passed = True

    print("\n" + "=" * 80)
    print("ADCRAFT EVALUATOR CALIBRATION")
    print("=" * 80)

    for ref_ad in reference_ads:
        ad = AdCopy(
            id=ref_ad["id"],
            primary_text=ref_ad["primary_text"],
            headline=ref_ad["headline"],
            description=ref_ad["description"],
            cta_button=ref_ad["cta_button"],
        )

        print(f"\n--- {ref_ad['id']} ({ref_ad['label']}) ---")

        # Run iteration mode
        iter_result = engine.evaluate_iteration(ad)
        print(f"  Iteration mode: weighted_avg={iter_result.weighted_average:.2f}")
        for s in iter_result.scores:
            print(f"    {s.dimension}: {s.score:.1f}")

        # Run final mode
        final_result = engine.evaluate_final(ad)
        print(f"  Final mode:     weighted_avg={final_result.weighted_average:.2f}")
        for s in final_result.scores:
            print(f"    {s.dimension}: {s.score:.1f}")

        # Check calibration criteria
        label = ref_ad["label"]
        iter_ok = (
            (label == "great" and iter_result.weighted_average >= 7.0)
            or (label == "bad" and iter_result.weighted_average < 5.0)
        )
        final_ok = (
            (label == "great" and final_result.weighted_average >= 7.0)
            or (label == "bad" and final_result.weighted_average < 5.0)
        )

        status = "PASS" if (iter_ok and final_ok) else "FAIL"
        if not (iter_ok and final_ok):
            all_passed = False
        print(f"  Calibration: {status}")

        results.append(
            {
                "id": ref_ad["id"],
                "label": label,
                "human_avg": sum(ref_ad["human_scores"].values()) / len(ref_ad["human_scores"]),
                "iteration_avg": iter_result.weighted_average,
                "final_avg": final_result.weighted_average,
                "status": status,
            }
        )

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'ID':<18} {'Label':<8} {'Human':>8} {'Iter':>8} {'Final':>8} {'Status':>8}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['id']:<18} {r['label']:<8} {r['human_avg']:>8.2f} "
            f"{r['iteration_avg']:>8.2f} {r['final_avg']:>8.2f} {r['status']:>8}"
        )
    print("=" * 80)
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'CALIBRATION FAILED'}")

    log_decision(
        "calibration",
        "calibration_run",
        f"Calibration {'passed' if all_passed else 'failed'} — "
        f"{sum(1 for r in results if r['status'] == 'PASS')}/{len(results)} ads met criteria",
        {"results": results},
    )

    return all_passed


if __name__ == "__main__":
    passed = run_calibration()
    sys.exit(0 if passed else 1)

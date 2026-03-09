"""Calibration script for AdCraft evaluator.

Run with: uv run python -m src.evaluate.calibrate

Loads a held-out gold set of human-scored ads, runs each through iteration-mode
evaluation, and computes inter-rater reliability metrics: Krippendorff's Alpha
(ordinal), Spearman rank correlation, and per-dimension MAE.

Returns a CalibrationResult instead of a simple bool — downstream consumers
(dashboard, pipeline gate) get structured metrics, not just pass/fail.

Requires GEMINI_API_KEY in environment or .env file.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import krippendorff
import numpy as np
from dotenv import load_dotenv
from scipy.stats import spearmanr

load_dotenv()

from src.db.init_db import init_db  # noqa: E402
from src.db.queries import insert_calibration_run  # noqa: E402
from src.decisions.logger import log_decision  # noqa: E402
from src.evaluate.engine import EvaluationEngine  # noqa: E402
from src.evaluate.rubrics import DIMENSIONS, FEW_SHOT_EXAMPLES  # noqa: E402
from src.models.ad import AdCopy  # noqa: E402
from src.models.calibration import CalibrationResult  # noqa: E402

ALPHA_THRESHOLD = 0.67


def load_gold_set() -> list[dict]:
    """Load held-out gold set ads from JSON file."""
    path = Path("data/reference_ads/calibration_gold_set.json")
    if not path.exists():
        raise FileNotFoundError(
            f"Gold set file not found at {path}. Run story-589 to create calibration_gold_set.json."
        )
    with open(path) as f:
        data = json.load(f)
    return data["gold_ads"]


def check_overlap(gold_ads: list[dict]) -> None:
    """Guard against circular validation: gold set ads must not appear in few-shot examples.

    Compares each gold ad's primary_text against all FEW_SHOT_EXAMPLES strings.
    Raises ValueError if any overlap is found.
    """
    gold_texts = {ad["id"]: ad["primary_text"] for ad in gold_ads}

    for gold_id, gold_text in gold_texts.items():
        for dimension, example_str in FEW_SHOT_EXAMPLES.items():
            if gold_text in example_str:
                raise ValueError(
                    f"Overlap detected: gold ad '{gold_id}' primary_text appears in "
                    f"FEW_SHOT_EXAMPLES['{dimension}']. Gold set must be held out from "
                    f"few-shot examples to avoid circular validation."
                )

    log_decision(
        "calibration",
        "overlap_guard_passed",
        f"No overlap between {len(gold_ads)} gold ads and few-shot examples",
        {"gold_ad_ids": [ad["id"] for ad in gold_ads]},
    )


def calculate_metrics(
    gold_ads: list[dict],
    eval_results: list[dict],
) -> tuple[float, float, dict[str, float]]:
    """Compute inter-rater reliability metrics between human and LLM scores.

    Returns (alpha_overall, spearman_rho, per_dimension_mae).
    """
    human_flat: list[float] = []
    llm_flat: list[float] = []
    per_dim_scores: dict[str, list[tuple[float, float]]] = {d: [] for d in DIMENSIONS}

    for ad, result in zip(gold_ads, eval_results):
        human_scores = ad["human_scores"]
        llm_scores = result["llm_scores"]
        for dim in DIMENSIONS:
            h_score = float(human_scores[dim])
            llm_score = float(llm_scores[dim])
            human_flat.append(h_score)
            llm_flat.append(llm_score)
            per_dim_scores[dim].append((h_score, llm_score))

    # Krippendorff's Alpha (ordinal) — 2 raters x N*D items
    reliability_data = np.array([human_flat, llm_flat])
    alpha_overall = krippendorff.alpha(
        reliability_data=reliability_data,
        level_of_measurement="ordinal",
    )

    # Spearman rank correlation
    rho, _p_value = spearmanr(human_flat, llm_flat)

    # Per-dimension MAE
    per_dimension_mae: dict[str, float] = {}
    for dim in DIMENSIONS:
        pairs = per_dim_scores[dim]
        mae = sum(abs(h - m) for h, m in pairs) / len(pairs)
        per_dimension_mae[dim] = round(mae, 3)

    return float(alpha_overall), float(rho), per_dimension_mae


def run_calibration() -> CalibrationResult:
    """Run calibration against the gold set and return structured results."""
    engine = EvaluationEngine()
    gold_ads = load_gold_set()
    check_overlap(gold_ads)

    n = len(gold_ads)
    print("ADCRAFT EVALUATOR CALIBRATION", flush=True)
    print(f"Gold set: {n} ads, {len(DIMENSIONS)} dimensions", flush=True)
    print("=" * 70, flush=True)

    eval_results: list[dict] = []

    for i, ad in enumerate(gold_ads):
        ad_id = ad["id"]
        print(f"Scoring {i + 1}/{n}: {ad_id}...", flush=True)

        ad_copy = AdCopy(
            id=ad_id,
            primary_text=ad["primary_text"],
            headline=ad["headline"],
            description=ad["description"],
            cta_button=ad["cta_button"],
        )

        result = engine.evaluate_iteration(ad_copy)

        llm_scores = {s.dimension: s.score for s in result.scores}
        human_scores = ad["human_scores"]
        human_avg = sum(human_scores.values()) / len(human_scores)

        eval_results.append(
            {
                "ad_id": ad_id,
                "label": ad["label"],
                "human_scores": human_scores,
                "llm_scores": llm_scores,
                "human_avg": round(human_avg, 2),
                "llm_avg": result.weighted_average,
            }
        )

    # Compute metrics
    alpha, rho, per_dim_mae = calculate_metrics(gold_ads, eval_results)

    # Print per-ad results table
    print("\n" + "=" * 70, flush=True)
    print(
        f"{'ID':<12} {'Label':<10} {'Human Avg':>10} {'LLM Avg':>10} {'Delta':>8}",
        flush=True,
    )
    print("-" * 70, flush=True)
    for r in eval_results:
        delta = r["human_avg"] - r["llm_avg"]
        print(
            f"{r['ad_id']:<12} {r['label']:<10} {r['human_avg']:>10.2f} "
            f"{r['llm_avg']:>10.2f} {delta:>+8.2f}",
            flush=True,
        )

    # Print per-dimension MAE table
    print("\n" + "-" * 40, flush=True)
    print(f"{'Dimension':<25} {'MAE':>10}", flush=True)
    print("-" * 40, flush=True)
    for dim in DIMENSIONS:
        print(f"{dim:<25} {per_dim_mae[dim]:>10.3f}", flush=True)

    # Determine pass/fail
    passed = alpha >= ALPHA_THRESHOLD

    # Print summary
    print("\n" + "=" * 70, flush=True)
    print(f"Alpha: {alpha:.3f}", flush=True)
    print(f"Spearman rho: {rho:.3f}", flush=True)
    print(f"Status: {'PASSED' if passed else 'FAILED'}", flush=True)
    print("=" * 70, flush=True)

    # Build CalibrationResult
    result = CalibrationResult(
        alpha_overall=round(alpha, 4),
        spearman_rho=round(rho, 4),
        per_dimension_mae=per_dim_mae,
        passed=passed,
        ad_count=n,
        model_version=engine._model,
        timestamp=datetime.now(timezone.utc).isoformat(),
        details={"per_ad_results": eval_results},
    )

    # Persist to DB
    conn = init_db(os.environ.get("DATABASE_PATH", "data/ads.db"))
    try:
        insert_calibration_run(
            conn,
            model_version=engine._model,
            alpha_overall=alpha,
            spearman_rho=rho,
            mae_per_dimension=per_dim_mae,
            ad_count=n,
            passed=passed,
            details={"per_ad_results": eval_results},
        )
    finally:
        conn.close()

    # Log decision
    log_decision(
        "calibration",
        "calibration_run",
        f"Calibration {'passed' if passed else 'failed'} — "
        f"alpha={alpha:.3f}, spearman={rho:.3f}, ads={n}",
        {
            "alpha": alpha,
            "spearman_rho": rho,
            "per_dimension_mae": per_dim_mae,
            "passed": passed,
        },
    )

    return result


if __name__ == "__main__":
    result = run_calibration()
    sys.exit(0 if result.passed else 1)

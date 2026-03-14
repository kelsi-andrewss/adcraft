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
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import krippendorff
import numpy as np
from dotenv import load_dotenv
from scipy.stats import spearmanr

load_dotenv()

from src.db.init_db import init_db  # noqa: E402
from src.db.queries import get_recent_calibration_runs, insert_calibration_run  # noqa: E402
from src.decisions.logger import log_decision  # noqa: E402
from src.evaluate.engine import EvaluationEngine  # noqa: E402
from src.evaluate.rubrics import DIMENSIONS, FEW_SHOT_EXAMPLES  # noqa: E402
from src.models.ad import AdCopy  # noqa: E402
from src.models.calibration import CalibrationResult, DriftAlert  # noqa: E402

ALPHA_THRESHOLD = 0.67
MID_RANGE_BOUNDS: tuple[float, float] = (4.0, 6.0)
DRIFT_WINDOW: int = 5
CONSECUTIVE_FAILURES: int = 3
MAE_INCREASE_RUNS: int = 3

MAE_DIMENSIONS = [f"mae_{d}" for d in DIMENSIONS]


def check_gold_set_overlap(few_shot_path: Path, gold_set_path: Path) -> None:
    """Guard against circular validation: IDs in few-shot file must not appear in gold set file.

    Both files are JSON arrays of objects with an "id" field.
    Raises ValueError listing overlapping IDs if any are found.
    """
    with open(few_shot_path) as f:
        few_shot_ids = {item["id"] for item in json.load(f)}
    with open(gold_set_path) as f:
        gold_set_ids = {item["id"] for item in json.load(f)}

    overlap = few_shot_ids & gold_set_ids
    if overlap:
        raise ValueError(
            f"Overlap detected between few-shot and gold set ad IDs: {', '.join(sorted(overlap))}. "
            f"Gold set must be held out from few-shot examples to avoid circular validation."
        )


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
    human_scores: list[dict] | list[float],
    llm_scores: list[dict] | list[float] | None = None,
) -> dict[str, float]:
    """Compute inter-rater reliability metrics between human and LLM scores.

    Supports two calling conventions:
      - calculate_metrics(gold_ads, eval_results): structured dicts with per-dimension scores
      - calculate_metrics(human_flat, llm_flat): flat numeric lists (pure math, no extraction)

    Returns dict with keys: alpha, spearman_rho, mae.
    """
    if llm_scores is None:
        raise ValueError("llm_scores is required")

    # Detect calling convention: flat numeric lists vs structured dicts
    if human_scores and isinstance(human_scores[0], (int, float)):
        human_flat = [float(x) for x in human_scores]
        llm_flat = [float(x) for x in llm_scores]  # type: ignore[union-attr]
    else:
        # Structured dict mode (gold_ads + eval_results)
        human_flat = []
        llm_flat = []
        for ad, result in zip(human_scores, llm_scores):  # type: ignore[arg-type]
            h_scores = ad["human_scores"]
            l_scores = result["llm_scores"]
            for dim in DIMENSIONS:
                human_flat.append(float(h_scores[dim]))
                llm_flat.append(float(l_scores[dim]))

    # Krippendorff's Alpha (ordinal) — 2 raters x N*D items
    reliability_data = np.array([human_flat, llm_flat])
    alpha_overall = krippendorff.alpha(
        reliability_data=reliability_data,
        level_of_measurement="ordinal",
    )

    # Spearman rank correlation
    rho, _p_value = spearmanr(human_flat, llm_flat)

    # Overall MAE
    mae = sum(abs(h - m) for h, m in zip(human_flat, llm_flat)) / len(human_flat)

    return {
        "alpha": float(alpha_overall),
        "spearman_rho": float(rho),
        "mae": mae,
    }


def filter_mid_range_scores(
    human_scores: list[float],
    llm_scores: list[float],
    bounds: tuple[float, float] = MID_RANGE_BOUNDS,
) -> tuple[list[float], list[float]]:
    """Extract pairs where EITHER human or LLM score falls within bounds (inclusive).

    Raises ValueError if lists have different lengths.
    """
    if len(human_scores) != len(llm_scores):
        raise ValueError(
            f"human_scores length ({len(human_scores)}) != llm_scores length ({len(llm_scores)})"
        )
    lo, hi = bounds
    human_out: list[float] = []
    llm_out: list[float] = []
    for h, m in zip(human_scores, llm_scores):
        if (lo <= h <= hi) or (lo <= m <= hi):
            human_out.append(h)
            llm_out.append(m)
    return human_out, llm_out


def calculate_mid_range_metrics(
    human_scores: list[float],
    llm_scores: list[float],
    bounds: tuple[float, float] = MID_RANGE_BOUNDS,
) -> dict[str, float | int]:
    """Compute Alpha and MAE for mid-range subset.

    Returns dict with keys: alpha, mae, count, pct_of_total.
    Returns insufficient_data=True when fewer than 3 pairs qualify.
    """
    total = len(human_scores)
    h_mid, l_mid = filter_mid_range_scores(human_scores, llm_scores, bounds)
    count = len(h_mid)

    if count < 3:
        return {
            "insufficient_data": True,
            "count": count,
            "pct_of_total": round(count / total * 100, 1) if total else 0.0,
        }

    metrics = calculate_metrics(h_mid, l_mid)
    return {
        "alpha": metrics["alpha"],
        "mae": metrics["mae"],
        "count": count,
        "pct_of_total": round(count / total * 100, 1) if total else 0.0,
    }


def _calculate_metrics_structured(
    gold_ads: list[dict],
    eval_results: list[dict],
) -> tuple[float, float, dict[str, float]]:
    """Structured variant used by run_calibration(). Returns (alpha, rho, per_dim_mae)."""
    human_flat: list[float] = []
    llm_flat: list[float] = []
    per_dim_scores: dict[str, list[tuple[float, float]]] = {d: [] for d in DIMENSIONS}

    for ad, result in zip(gold_ads, eval_results):
        human_scores_d = ad["human_scores"]
        llm_scores_d = result["llm_scores"]
        for dim in DIMENSIONS:
            h_score = float(human_scores_d[dim])
            llm_score = float(llm_scores_d[dim])
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


def detect_drift(conn: sqlite3.Connection) -> list[DriftAlert]:
    """Fetch the last DRIFT_WINDOW calibration_runs ordered by timestamp DESC.

    Checks:
    1. Alpha drift: CONSECUTIVE_FAILURES consecutive runs with alpha < ALPHA_THRESHOLD.
    2. MAE drift: any dimension where MAE increases for MAE_INCREASE_RUNS consecutive runs.

    Returns list of DriftAlert. Each alert logged via log_decision inside this function.
    """
    runs = get_recent_calibration_runs(conn, limit=DRIFT_WINDOW)
    alerts: list[DriftAlert] = []

    if not runs:
        return alerts

    # Alpha drift: check from newest to oldest for consecutive failures
    consecutive_alpha_failures = 0
    for run in runs:
        if run["alpha_overall"] < ALPHA_THRESHOLD:
            consecutive_alpha_failures += 1
        else:
            break

    if consecutive_alpha_failures >= CONSECUTIVE_FAILURES:
        alphas = [run["alpha_overall"] for run in runs[:consecutive_alpha_failures]]
        alert = DriftAlert(
            alert_type="alpha_drift",
            message=(
                f"Alpha below {ALPHA_THRESHOLD} for {consecutive_alpha_failures} consecutive runs"
            ),
            detail={"consecutive_failures": consecutive_alpha_failures, "alphas": alphas},
        )
        alerts.append(alert)
        log_decision(
            "calibration",
            alert.alert_type,
            alert.message,
            alert.detail,
            conn=conn,
        )

    # MAE drift: reverse to chronological order and check monotonic increase
    if len(runs) >= MAE_INCREASE_RUNS:
        chronological = list(reversed(runs))
        for mae_col in MAE_DIMENSIONS:
            values = [r[mae_col] for r in chronological if r[mae_col] is not None]
            if len(values) < MAE_INCREASE_RUNS:
                continue
            # Check the last MAE_INCREASE_RUNS values for monotonic increase
            tail = values[-MAE_INCREASE_RUNS:]
            increasing = all(tail[i] < tail[i + 1] for i in range(len(tail) - 1))
            if increasing:
                dim_name = mae_col.removeprefix("mae_")
                alert = DriftAlert(
                    alert_type="mae_drift",
                    message=(
                        f"MAE for {dim_name} increased over {MAE_INCREASE_RUNS} consecutive runs"
                    ),
                    detail={"dimension": dim_name, "values": tail},
                )
                alerts.append(alert)
                log_decision(
                    "calibration",
                    alert.alert_type,
                    alert.message,
                    alert.detail,
                    conn=conn,
                )

    return alerts


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
    alpha, rho, per_dim_mae = _calculate_metrics_structured(gold_ads, eval_results)

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

    # Mid-range audit
    human_flat: list[float] = []
    llm_flat: list[float] = []
    for ad, er in zip(gold_ads, eval_results):
        for dim in DIMENSIONS:
            human_flat.append(float(ad["human_scores"][dim]))
            llm_flat.append(float(er["llm_scores"][dim]))

    mid_range = calculate_mid_range_metrics(human_flat, llm_flat)

    print("\n" + "-" * 40, flush=True)
    print("MID-RANGE AUDIT", flush=True)
    print("-" * 40, flush=True)
    if mid_range.get("insufficient_data"):
        print(
            f"  Insufficient data ({mid_range['count']} pairs, "
            f"{mid_range['pct_of_total']:.1f}% of total)",
            flush=True,
        )
    else:
        print(f"  Alpha: {mid_range['alpha']:.3f}", flush=True)
        print(f"  MAE:   {mid_range['mae']:.3f}", flush=True)
        print(
            f"  Pairs: {mid_range['count']} ({mid_range['pct_of_total']:.1f}% of total)",
            flush=True,
        )

    log_decision(
        "calibration",
        "mid_range_audit",
        f"Mid-range audit: {mid_range['count']} pairs "
        f"in [{MID_RANGE_BOUNDS[0]}, {MID_RANGE_BOUNDS[1]}]",
        mid_range,
    )

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
        details={"per_ad_results": eval_results, "mid_range_audit": mid_range},
    )

    # Persist to DB and check for drift
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
        alerts = detect_drift(conn)
        for alert in alerts:
            log_decision(
                "calibration",
                alert.alert_type,
                alert.message,
                alert.detail,
                conn=conn,
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

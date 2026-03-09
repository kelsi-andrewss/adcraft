"""Export subsystem: ad library, decision log, and summary stats.

Exports to CSV and JSON formats. All exporters create parent directories
if they don't exist.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Literal

from src.db import queries
from src.decisions.logger import log_decision
from src.evaluate.rubrics import DIMENSION_WEIGHTS, PASSING_THRESHOLD


def export_ad_library(
    db_conn: sqlite3.Connection,
    format: Literal["csv", "json"],
    output_path: str | Path,
) -> Path:
    """Export ads joined with their final evaluation scores.

    CSV: one row per ad with flattened dimension scores.
    JSON: preserves nested structure with scores as sub-object.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = queries.get_ads_with_scores(db_conn)

    log_decision(
        "export",
        "export_ad_library",
        f"Exporting {len(rows)} ads to {format.upper()} at {path}",
        {"format": format, "count": len(rows), "path": str(path)},
        conn=db_conn,
    )

    if format == "csv":
        _write_ad_csv(rows, path)
    else:
        _write_ad_json(rows, path)

    return path


def _write_ad_csv(rows: list[dict], path: Path) -> None:
    """Write ads as flat CSV with one row per ad."""
    if not rows:
        path.write_text("")
        return

    # Group scores by ad_id
    ads = _group_ad_scores(rows)

    dimensions = list(DIMENSION_WEIGHTS.keys())
    fieldnames = [
        "ad_id",
        "primary_text",
        "headline",
        "description",
        "cta_button",
        "model_id",
        "cost_usd",
        "weighted_average",
    ] + [f"score_{d}" for d in dimensions]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ad in ads:
            row = {
                "ad_id": ad["id"],
                "primary_text": ad["primary_text"],
                "headline": ad["headline"],
                "description": ad["description"],
                "cta_button": ad["cta_button"],
                "model_id": ad.get("model_id", ""),
                "cost_usd": ad.get("cost_usd", ""),
                "weighted_average": ad.get("weighted_average", ""),
            }
            for d in dimensions:
                row[f"score_{d}"] = ad.get("scores", {}).get(d, "")
            writer.writerow(row)


def _write_ad_json(rows: list[dict], path: Path) -> None:
    """Write ads as JSON with nested scores."""
    if not rows:
        path.write_text("[]")
        return

    ads = _group_ad_scores(rows)
    path.write_text(json.dumps(ads, indent=2, default=str))


def _group_ad_scores(rows: list[dict]) -> list[dict]:
    """Group flat ad+evaluation rows into per-ad dicts with nested scores."""
    ad_map: dict[str, dict] = {}

    for row in rows:
        ad_id = row["ad_id"]
        if ad_id not in ad_map:
            ad_map[ad_id] = {
                "id": ad_id,
                "primary_text": row.get("primary_text", ""),
                "headline": row.get("headline", ""),
                "description": row.get("description", ""),
                "cta_button": row.get("cta_button", ""),
                "model_id": row.get("model_id", ""),
                "cost_usd": row.get("ad_cost_usd"),
                "scores": {},
                "weighted_average": None,
            }

        dim = row.get("dimension")
        score = row.get("score")
        if dim and score is not None:
            ad_map[ad_id]["scores"][dim] = score

    # Calculate weighted averages
    for ad in ad_map.values():
        scores = ad["scores"]
        if scores:
            weighted = sum(
                scores.get(dim, 0.0) * weight for dim, weight in DIMENSION_WEIGHTS.items()
            )
            ad["weighted_average"] = round(weighted, 4)

    return list(ad_map.values())


def export_decision_log(
    db_conn: sqlite3.Connection,
    output_path: str | Path,
) -> Path:
    """Export all decisions ordered by timestamp as a JSON array."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    decisions = queries.get_all_decisions(db_conn)

    log_decision(
        "export",
        "export_decision_log",
        f"Exporting {len(decisions)} decisions to {path}",
        {"count": len(decisions), "path": str(path)},
        conn=db_conn,
    )

    path.write_text(json.dumps(decisions, indent=2, default=str))
    return path


def export_summary_stats(
    db_conn: sqlite3.Connection,
    output_path: str | Path,
) -> Path:
    """Export summary statistics as JSON.

    Includes: total_ads, pass_rate, avg_weighted_score, dimension_averages,
    total_token_spend, avg_quality_per_dollar, iteration_stats.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    db_conn.row_factory = sqlite3.Row

    # Total ads
    total_ads_row = db_conn.execute("SELECT COUNT(*) as cnt FROM ads").fetchone()
    total_ads = total_ads_row["cnt"] if total_ads_row else 0

    # Dimension averages
    dimension_averages = queries.get_dimension_averages(db_conn)

    # Weighted averages per ad for pass rate
    ad_ids = [r["id"] for r in db_conn.execute("SELECT id FROM ads").fetchall()]
    weighted_avgs: list[float] = []
    for ad_id in ad_ids:
        evals = db_conn.execute(
            "SELECT dimension, score FROM evaluations WHERE ad_id = ? "
            "AND (eval_mode = 'final' OR eval_mode IS NULL)",
            (ad_id,),
        ).fetchall()
        if not evals:
            continue
        scores = {r["dimension"]: r["score"] for r in evals}
        weighted = sum(scores.get(d, 0.0) * w for d, w in DIMENSION_WEIGHTS.items())
        weighted_avgs.append(weighted)

    pass_count = sum(1 for w in weighted_avgs if w >= PASSING_THRESHOLD)
    pass_rate = pass_count / len(weighted_avgs) if weighted_avgs else 0.0
    avg_weighted_score = sum(weighted_avgs) / len(weighted_avgs) if weighted_avgs else 0.0

    # Token spend
    spend_row = db_conn.execute("SELECT COALESCE(SUM(cost_usd), 0.0) as total FROM ads").fetchone()
    eval_spend_row = db_conn.execute(
        "SELECT COALESCE(SUM(cost_usd), 0.0) as total FROM evaluations"
    ).fetchone()
    total_token_spend = (spend_row["total"] if spend_row else 0.0) + (
        eval_spend_row["total"] if eval_spend_row else 0.0
    )

    # Quality per dollar from snapshots
    snapshots = queries.list_quality_snapshots(db_conn)
    qpd_values = [s["quality_per_dollar"] for s in snapshots if s.get("quality_per_dollar")]
    avg_quality_per_dollar = sum(qpd_values) / len(qpd_values) if qpd_values else None

    # Iteration stats
    iter_rows = db_conn.execute(
        "SELECT source_ad_id, MAX(cycle_number) as max_cycle, "
        "MAX(delta_weighted_avg) as best_delta FROM iterations GROUP BY source_ad_id"
    ).fetchall()
    avg_cycles = sum(r["max_cycle"] for r in iter_rows) / len(iter_rows) if iter_rows else 0.0
    improved = sum(1 for r in iter_rows if (r["best_delta"] or 0) > 0)
    improvement_rate = improved / len(iter_rows) if iter_rows else 0.0

    stats = {
        "total_ads": total_ads,
        "pass_rate": round(pass_rate, 4),
        "avg_weighted_score": round(avg_weighted_score, 4),
        "dimension_averages": dimension_averages,
        "total_token_spend": round(total_token_spend, 6),
        "avg_quality_per_dollar": (
            round(avg_quality_per_dollar, 4) if avg_quality_per_dollar else None
        ),
        "iteration_stats": {
            "avg_cycles_per_ad": round(avg_cycles, 2),
            "improvement_rate": round(improvement_rate, 4),
        },
    }

    log_decision(
        "export",
        "export_summary_stats",
        f"Exported summary: {total_ads} ads, pass_rate={pass_rate:.1%}, "
        f"avg_score={avg_weighted_score:.2f}",
        stats,
        conn=db_conn,
    )

    path.write_text(json.dumps(stats, indent=2, default=str))
    return path

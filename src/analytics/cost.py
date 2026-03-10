"""Cost analytics: token counting, API cost calculation, quality snapshots.

Uses LiteLLM's completion_cost() as primary pricing source with a manual
fallback table for models not yet in LiteLLM's registry.
"""

from __future__ import annotations

import sqlite3

from src.db import queries
from src.decisions.logger import log_decision

# Fallback pricing per 1M tokens: (input_price, output_price)
FALLBACK_PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
}

# Flat rate per image at 1K resolution (not per-token — separate from FALLBACK_PRICING)
IMAGE_PRICING: dict[str, float] = {
    "gemini-2.5-flash-image": 0.039,
    "gemini-3-pro-image-preview": 0.08,
    "imagen-4.0-generate-001": 0.04,
    "imagen-4.0-fast-generate-001": 0.02,
}

PASS_THRESHOLD = 7.0


def calculate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate the USD cost for an API call.

    Tries LiteLLM first; falls back to the manual pricing table.
    """
    try:
        import litellm

        cost = litellm.completion_cost(
            model=model_id,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )
        if cost > 0:
            return cost
    except Exception:
        pass

    # Fallback: manual pricing table
    pricing = FALLBACK_PRICING.get(model_id)
    if pricing is None:
        log_decision(
            "cost",
            "unknown_model_pricing",
            f"No pricing data for model '{model_id}', returning 0.0",
            {"model_id": model_id},
        )
        return 0.0

    input_price, output_price = pricing
    cost = (input_tokens * input_price / 1_000_000) + (output_tokens * output_price / 1_000_000)
    return cost


def calculate_image_cost(model_id: str) -> float:
    """Return the flat-rate USD cost for one image generation call.

    Looks up the model in IMAGE_PRICING. Logs a decision and returns 0.0
    for unknown models (same pattern as calculate_cost fallback).
    """
    price = IMAGE_PRICING.get(model_id)
    if price is None:
        log_decision(
            "cost",
            "unknown_image_model_pricing",
            f"No image pricing data for model '{model_id}', returning 0.0",
            {"model_id": model_id},
        )
        return 0.0
    return price


def record_api_cost(
    db_conn: sqlite3.Connection,
    table: str,
    row_id: str,
    model_id: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Calculate cost and update token/cost columns on the specified table row.

    Returns the calculated cost_usd.
    """
    cost_usd = calculate_cost(model_id, input_tokens, output_tokens)

    db_conn.execute(
        f"UPDATE {table} SET input_tokens = ?, output_tokens = ?, cost_usd = ? WHERE id = ?",  # noqa: E501
        (input_tokens, output_tokens, cost_usd, row_id),
    )
    db_conn.commit()

    log_decision(
        "cost",
        "recorded_api_cost",
        f"Recorded cost for {table}.{row_id}: ${cost_usd:.6f} "
        f"({input_tokens} in + {output_tokens} out tokens, model={model_id})",
        {
            "table": table,
            "row_id": row_id,
            "model_id": model_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
        },
        conn=db_conn,
    )

    return cost_usd


def compute_quality_snapshot(db_conn: sqlite3.Connection, cycle_number: int) -> dict:
    """Calculate and persist a quality snapshot for the given cycle.

    Queries ads and evaluations for the cycle, computes performance-per-token,
    and inserts into quality_snapshots. Returns the snapshot dict.
    """
    db_conn.row_factory = sqlite3.Row

    # Get all ads (cycle filtering relies on iteration records if available)
    # Include image_cost_usd; fall back to without it for pre-migration DBs
    try:
        ads_rows = db_conn.execute("SELECT id, cost_usd, image_cost_usd FROM ads").fetchall()
        _has_image_col = True
    except sqlite3.OperationalError:
        ads_rows = db_conn.execute("SELECT id, cost_usd FROM ads").fetchall()
        _has_image_col = False
    total_ads = len(ads_rows)
    ad_ids = [r["id"] for r in ads_rows]

    if total_ads == 0:
        snapshot = {
            "cycle_number": cycle_number,
            "avg_weighted_score": None,
            "dimension_averages": {},
            "ads_above_threshold": 0,
            "total_ads": 0,
            "token_spend_usd": 0.0,
            "quality_per_dollar": None,
        }
        queries.insert_quality_snapshot(db_conn, **snapshot)
        return snapshot

    # Calculate token spend from ads (copy + image) and evaluations
    ad_spend = sum(r["cost_usd"] or 0.0 for r in ads_rows)
    image_spend = sum(r["image_cost_usd"] or 0.0 for r in ads_rows) if _has_image_col else 0.0
    eval_spend_row = db_conn.execute(
        "SELECT COALESCE(SUM(cost_usd), 0.0) as total FROM evaluations"
    ).fetchone()
    token_spend_usd = ad_spend + image_spend + (eval_spend_row["total"] if eval_spend_row else 0.0)

    # Calculate dimension averages from final-mode evaluations (or all if no final)
    eval_rows = db_conn.execute(
        "SELECT dimension, score FROM evaluations WHERE eval_mode = 'final'"
    ).fetchall()
    if not eval_rows:
        eval_rows = db_conn.execute("SELECT dimension, score FROM evaluations").fetchall()

    dim_scores: dict[str, list[float]] = {}
    for row in eval_rows:
        dim_scores.setdefault(row["dimension"], []).append(row["score"])

    dimension_averages = {dim: sum(scores) / len(scores) for dim, scores in dim_scores.items()}

    # Calculate weighted averages per ad
    from src.evaluate.rubrics import DIMENSION_WEIGHTS

    weighted_avgs: list[float] = []
    for ad_id in ad_ids:
        ad_evals = db_conn.execute(
            "SELECT dimension, score FROM evaluations WHERE ad_id = ? "
            "AND (eval_mode = 'final' OR eval_mode IS NULL)",
            (ad_id,),
        ).fetchall()
        if not ad_evals:
            continue

        ad_dim_scores = {r["dimension"]: r["score"] for r in ad_evals}
        weighted = sum(
            ad_dim_scores.get(dim, 0.0) * weight for dim, weight in DIMENSION_WEIGHTS.items()
        )
        weighted_avgs.append(weighted)

    avg_weighted_score = sum(weighted_avgs) / len(weighted_avgs) if weighted_avgs else None
    ads_above = sum(1 for w in weighted_avgs if w >= PASS_THRESHOLD)

    # Performance per dollar — avoid division by zero
    quality_per_dollar = None
    if token_spend_usd > 0 and avg_weighted_score is not None:
        quality_per_dollar = avg_weighted_score / token_spend_usd

    snapshot = {
        "cycle_number": cycle_number,
        "avg_weighted_score": avg_weighted_score,
        "dimension_averages": dimension_averages,
        "ads_above_threshold": ads_above,
        "total_ads": total_ads,
        "token_spend_usd": token_spend_usd,
        "quality_per_dollar": quality_per_dollar,
    }

    queries.insert_quality_snapshot(db_conn, **snapshot)

    log_decision(
        "cost",
        "quality_snapshot",
        f"Cycle {cycle_number}: avg_score={avg_weighted_score:.2f}, "
        f"spend=${token_spend_usd:.4f}, q/dollar={quality_per_dollar}"
        if avg_weighted_score is not None
        else f"Cycle {cycle_number}: no scored ads",
        snapshot,
        conn=db_conn,
    )

    return snapshot


def get_performance_per_token(db_conn: sqlite3.Connection) -> list[dict]:
    """Return all quality snapshots ordered by cycle for trend display."""
    return queries.list_quality_snapshots(db_conn)


def compute_creative_unit_cost(db_conn: sqlite3.Connection, ad_id: str) -> dict:
    """Calculate the full cost breakdown for one creative unit (ad).

    Sums copy generation cost, image generation cost, and all evaluation
    costs for the given ad. Returns 0.0 for any NULL cost fields.
    """
    db_conn.row_factory = sqlite3.Row

    ad_row = db_conn.execute(
        """SELECT id, cost_usd, image_cost_usd, image_model, image_path
           FROM ads WHERE id = ?""",
        (ad_id,),
    ).fetchone()

    if ad_row is None:
        log_decision(
            "cost",
            "creative_unit_cost_not_found",
            f"No ad found with id '{ad_id}'",
            {"ad_id": ad_id},
            conn=db_conn,
        )
        return {
            "ad_id": ad_id,
            "copy_gen_cost": 0.0,
            "image_gen_cost": 0.0,
            "eval_cost": 0.0,
            "total_cost": 0.0,
            "image_model": None,
            "has_image": False,
        }

    copy_gen_cost = ad_row["cost_usd"] or 0.0
    image_gen_cost = ad_row["image_cost_usd"] or 0.0

    eval_row = db_conn.execute(
        "SELECT COALESCE(SUM(cost_usd), 0.0) as total FROM evaluations WHERE ad_id = ?",
        (ad_id,),
    ).fetchone()
    eval_cost = eval_row["total"] if eval_row else 0.0

    total_cost = copy_gen_cost + image_gen_cost + eval_cost
    has_image = image_gen_cost > 0 or ad_row["image_path"] is not None

    breakdown = {
        "ad_id": ad_id,
        "copy_gen_cost": copy_gen_cost,
        "image_gen_cost": image_gen_cost,
        "eval_cost": eval_cost,
        "total_cost": total_cost,
        "image_model": ad_row["image_model"],
        "has_image": has_image,
    }

    log_decision(
        "cost",
        "creative_unit_cost",
        f"Ad {ad_id}: copy=${copy_gen_cost:.4f} + image=${image_gen_cost:.4f} "
        f"+ eval=${eval_cost:.4f} = total=${total_cost:.4f}",
        breakdown,
        conn=db_conn,
    )

    return breakdown


def compare_image_model_efficiency(db_conn: sqlite3.Connection) -> list[dict]:
    """Compare quality-per-dollar across image generation models.

    For each model that has generated at least one image, returns cost
    aggregates and average visual evaluation scores (brand_consistency,
    composition_quality, text_image_synergy).
    """
    model_costs = queries.get_image_costs_by_model(db_conn)
    if not model_costs:
        return []

    # Visual evaluation dimensions stored in evaluations table
    visual_dims = ("brand_consistency", "composition_quality", "text_image_synergy")

    results = []
    for mc in model_costs:
        model = mc["image_model"]

        # Average visual scores for ads generated by this model
        db_conn.row_factory = sqlite3.Row
        score_row = db_conn.execute(
            f"""SELECT AVG(e.score) as avg_visual_score
                FROM evaluations e
                JOIN ads a ON a.id = e.ad_id
                WHERE a.image_model = ?
                  AND e.dimension IN ({",".join("?" for _ in visual_dims)})""",
            (model, *visual_dims),
        ).fetchone()

        avg_visual_score = score_row["avg_visual_score"] if score_row else None
        avg_image_cost = mc["avg_cost"] or 0.0

        quality_per_dollar = None
        if avg_visual_score is not None and avg_image_cost > 0:
            quality_per_dollar = avg_visual_score / avg_image_cost

        results.append(
            {
                "image_model": model,
                "ad_count": mc["ad_count"],
                "total_image_cost": mc["total_cost"] or 0.0,
                "avg_image_cost": avg_image_cost,
                "avg_visual_score": avg_visual_score,
                "quality_per_dollar": quality_per_dollar,
            }
        )

    log_decision(
        "cost",
        "image_model_efficiency_comparison",
        f"Compared {len(results)} image models by quality-per-dollar",
        {"models": results},
        conn=db_conn,
    )

    return results

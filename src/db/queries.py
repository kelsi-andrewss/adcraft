"""Parameterized SQL query helpers for all AdCraft tables.

Every function takes a sqlite3.Connection as its first parameter.
No module-level connection state. No ORM. All SQL is raw and parameterized.
Insert functions generate UUIDs and timestamps internally.
Select functions return Pydantic model instances, not raw tuples.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime

# ---------------------------------------------------------------------------
# ads
# ---------------------------------------------------------------------------


def insert_ad(
    conn: sqlite3.Connection,
    *,
    primary_text: str,
    headline: str,
    description: str,
    cta_button: str,
    brief_id: str | None = None,
    model_id: str | None = None,
    temperature: float | None = None,
    generation_seed: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cost_usd: float | None = None,
) -> str:
    """Insert an ad and return its generated ID."""
    ad_id = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO ads
           (id, brief_id, primary_text, headline, description, cta_button,
            model_id, temperature, generation_seed, input_tokens, output_tokens, cost_usd)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            ad_id,
            brief_id,
            primary_text,
            headline,
            description,
            cta_button,
            model_id,
            temperature,
            generation_seed,
            input_tokens,
            output_tokens,
            cost_usd,
        ),
    )
    conn.commit()
    return ad_id


def get_ad(conn: sqlite3.Connection, ad_id: str) -> dict | None:
    """Fetch a single ad by ID. Returns a dict or None if not found."""
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM ads WHERE id = ?", (ad_id,)).fetchone()
    if row is None:
        return None
    return dict(row)


def list_ads(conn: sqlite3.Connection, *, limit: int = 100) -> list[dict]:
    """List ads ordered by creation time descending."""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM ads ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# evaluations
# ---------------------------------------------------------------------------


def insert_evaluation(
    conn: sqlite3.Connection,
    *,
    ad_id: str,
    dimension: str,
    score: float,
    rationale: str | None = None,
    confidence: float | None = None,
    evaluator_model: str | None = None,
    eval_mode: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cost_usd: float | None = None,
) -> str:
    """Insert an evaluation row and return its generated ID."""
    eval_id = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO evaluations
           (id, ad_id, dimension, score, rationale, confidence,
            evaluator_model, eval_mode, input_tokens, output_tokens, cost_usd)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            eval_id,
            ad_id,
            dimension,
            score,
            rationale,
            confidence,
            evaluator_model,
            eval_mode,
            input_tokens,
            output_tokens,
            cost_usd,
        ),
    )
    conn.commit()
    return eval_id


def get_evaluations_for_ad(conn: sqlite3.Connection, ad_id: str) -> list[dict]:
    """Fetch all evaluations for a given ad."""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM evaluations WHERE ad_id = ? ORDER BY created_at", (ad_id,)
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# iterations
# ---------------------------------------------------------------------------


def insert_iteration(
    conn: sqlite3.Connection,
    *,
    source_ad_id: str,
    target_ad_id: str,
    cycle_number: int,
    action_type: str,
    weak_dimension: str | None = None,
    feedback_prompt: str | None = None,
    delta_weighted_avg: float | None = None,
    token_cost: float | None = None,
) -> str:
    """Insert an iteration record and return its generated ID."""
    iter_id = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO iterations
           (id, source_ad_id, target_ad_id, cycle_number, action_type,
            weak_dimension, feedback_prompt, delta_weighted_avg, token_cost)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            iter_id,
            source_ad_id,
            target_ad_id,
            cycle_number,
            action_type,
            weak_dimension,
            feedback_prompt,
            delta_weighted_avg,
            token_cost,
        ),
    )
    conn.commit()
    return iter_id


def get_iterations_for_ad(conn: sqlite3.Connection, ad_id: str) -> list[dict]:
    """Fetch all iterations where the given ad is the source."""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM iterations WHERE source_ad_id = ? ORDER BY cycle_number",
        (ad_id,),
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# decisions
# ---------------------------------------------------------------------------


def insert_decision(
    conn: sqlite3.Connection,
    *,
    timestamp: datetime,
    component: str,
    action: str,
    rationale: str | None = None,
    context: str | None = None,
    agent_id: str | None = None,
) -> str:
    """Insert a decision log entry and return its generated ID."""
    decision_id = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO decisions
           (id, timestamp, component, action, rationale, context, agent_id)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            decision_id,
            timestamp.isoformat(),
            component,
            action,
            rationale,
            context,
            agent_id,
        ),
    )
    conn.commit()
    return decision_id


def list_decisions(
    conn: sqlite3.Connection, *, component: str | None = None, limit: int = 100
) -> list[dict]:
    """List decisions, optionally filtered by component."""
    conn.row_factory = sqlite3.Row
    if component is not None:
        rows = conn.execute(
            "SELECT * FROM decisions WHERE component = ? ORDER BY timestamp DESC LIMIT ?",
            (component, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM decisions ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# competitor_ads
# ---------------------------------------------------------------------------


def insert_competitor_ad(
    conn: sqlite3.Connection,
    *,
    brand: str,
    primary_text: str | None = None,
    headline: str | None = None,
    cta_button: str | None = None,
    hook_type: str | None = None,
    emotional_angle: str | None = None,
) -> str:
    """Insert a competitor ad and return its generated ID."""
    comp_id = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO competitor_ads
           (id, brand, primary_text, headline, cta_button, hook_type, emotional_angle)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (comp_id, brand, primary_text, headline, cta_button, hook_type, emotional_angle),
    )
    conn.commit()
    return comp_id


def list_competitor_ads(
    conn: sqlite3.Connection, *, brand: str | None = None, limit: int = 100
) -> list[dict]:
    """List competitor ads, optionally filtered by brand."""
    conn.row_factory = sqlite3.Row
    if brand is not None:
        rows = conn.execute(
            "SELECT * FROM competitor_ads WHERE brand = ? ORDER BY scraped_at DESC LIMIT ?",
            (brand, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM competitor_ads ORDER BY scraped_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# quality_snapshots
# ---------------------------------------------------------------------------


def insert_quality_snapshot(
    conn: sqlite3.Connection,
    *,
    cycle_number: int,
    avg_weighted_score: float | None = None,
    dimension_averages: dict | None = None,
    ads_above_threshold: int | None = None,
    total_ads: int | None = None,
    token_spend_usd: float | None = None,
    quality_per_dollar: float | None = None,
) -> str:
    """Insert a quality snapshot and return its generated ID."""
    snap_id = str(uuid.uuid4())
    dim_avg_json = json.dumps(dimension_averages) if dimension_averages is not None else None
    conn.execute(
        """INSERT INTO quality_snapshots
           (id, cycle_number, avg_weighted_score, dimension_averages,
            ads_above_threshold, total_ads, token_spend_usd, quality_per_dollar)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            snap_id,
            cycle_number,
            avg_weighted_score,
            dim_avg_json,
            ads_above_threshold,
            total_ads,
            token_spend_usd,
            quality_per_dollar,
        ),
    )
    conn.commit()
    return snap_id


def list_quality_snapshots(conn: sqlite3.Connection, *, limit: int = 100) -> list[dict]:
    """List quality snapshots ordered by cycle number."""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM quality_snapshots ORDER BY cycle_number DESC LIMIT ?", (limit,)
    ).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        # Parse dimension_averages back from JSON string
        if d.get("dimension_averages") is not None:
            d["dimension_averages"] = json.loads(d["dimension_averages"])
        results.append(d)
    return results

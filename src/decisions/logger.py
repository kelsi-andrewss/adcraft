"""Decision logger for AdCraft.

Standalone import: `from src.decisions.logger import log_decision`

Every pipeline component that makes a choice calls log_decision() before
executing that choice. This creates an auditable trail of system behavior.
"""

from __future__ import annotations

import json
import os
import sqlite3

from src.db.queries import insert_decision
from src.models.decision import DecisionEntry


def log_decision(
    component: str,
    action: str,
    rationale: str,
    context: dict | None = None,
    *,
    conn: sqlite3.Connection | None = None,
    agent_id: str = "system",
) -> str:
    """Log a decision entry and return its ID.

    Validates input by constructing a DecisionEntry (Pydantic raises on bad data).
    Persists to the decisions table via insert_decision().

    If conn is None, opens a temporary connection to the default database path
    and closes it after writing. If conn is provided, the caller manages lifecycle.
    """
    entry = DecisionEntry(
        component=component,
        action=action,
        rationale=rationale,
        context=context if context is not None else {},
        agent_id=agent_id,
    )

    context_json = json.dumps(entry.model_dump()["context"])

    owns_conn = conn is None
    if owns_conn:
        db_path = os.environ.get("DATABASE_PATH", "data/ads.db")
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys=ON")

    try:
        decision_id = insert_decision(
            conn,
            timestamp=entry.timestamp,
            component=entry.component,
            action=entry.action,
            rationale=entry.rationale,
            context=context_json,
            agent_id=entry.agent_id,
        )
        return decision_id
    finally:
        if owns_conn:
            conn.close()


def log_circuit_breaker_event(
    tier: str,
    failure_rate: float,
    ads_affected: int,
    total_ads: int,
    action: str,
    *,
    conn: sqlite3.Connection | None = None,
) -> str:
    """Log a circuit breaker trigger with structured failure stats.

    Delegates to log_decision() with component="visual_circuit_breaker".
    Returns the decision_id from log_decision().
    """
    pct = failure_rate * 100
    rationale = (
        f"Tier {tier} {action}: {pct:.1f}% failure rate "
        f"({ads_affected}/{total_ads} ads), halting image generation"
    )

    return log_decision(
        component="visual_circuit_breaker",
        action=f"tier_{tier}_{action}",
        rationale=rationale,
        context={
            "tier": tier,
            "failure_rate": failure_rate,
            "ads_affected": ads_affected,
            "total_ads": total_ads,
            "action": action,
        },
        conn=conn,
    )

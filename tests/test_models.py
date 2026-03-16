"""Smoke tests for Pydantic models and database schema."""

import sqlite3
from datetime import datetime

import pytest
from pydantic import ValidationError

from src.db.init_db import init_db
from src.models import (
    AdBrief,
    AdCopy,
    DecisionEntry,
    DimensionScore,
    EvaluationResult,
    IterationRecord,
)

# --- AdBrief ---


class TestAdBrief:
    def test_round_trip(self):
        data = {
            "audience_segment": "parents of high school juniors",
            "product_offer": "SAT prep tutoring",
            "campaign_goal": "lead generation",
            "tone": "empathetic and authoritative",
            "competitive_context": "Princeton Review emphasizes score guarantees",
        }
        brief = AdBrief.model_validate(data)
        assert brief.model_dump() == data

    def test_optional_competitive_context(self):
        brief = AdBrief(
            audience_segment="students",
            product_offer="SAT prep",
            campaign_goal="awareness",
            tone="casual",
        )
        assert brief.competitive_context == ""

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            AdBrief.model_validate({"audience_segment": "students"})


# --- AdCopy ---


class TestAdCopy:
    def test_round_trip(self):
        data = {
            "id": "ad-001",
            "primary_text": "Your child deserves expert SAT prep.",
            "headline": "Boost SAT Scores 200+ Points",
            "description": "1-on-1 tutoring tailored to your student's needs.",
            "cta_button": "Learn More",
            "brief_id": "brief-001",
            "model_id": "gemini-2.5-flash",
            "generation_config": {"temperature": 0.7},
            "token_count": 150,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
        }
        ad = AdCopy.model_validate(data)
        dumped = ad.model_dump()
        assert dumped == data

    def test_defaults(self):
        ad = AdCopy(
            primary_text="Test",
            headline="Test",
            description="Test",
            cta_button="Sign Up",
        )
        assert ad.id == ""
        assert ad.token_count == 0
        assert ad.generation_config == {}

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            AdCopy.model_validate({"primary_text": "Test"})


# --- DimensionScore ---


class TestDimensionScore:
    def test_round_trip(self):
        data = {
            "dimension": "clarity",
            "score": 8.5,
            "rationale": "Clear value proposition with specific claim.",
            "confidence": 0.9,
        }
        score = DimensionScore.model_validate(data)
        assert score.model_dump() == data

    def test_default_confidence(self):
        score = DimensionScore(dimension="cta", score=7.0, rationale="Decent CTA.")
        assert score.confidence == 1.0


# --- EvaluationResult ---


class TestEvaluationResult:
    def test_round_trip(self):
        data = {
            "ad_id": "ad-001",
            "scores": [
                {"dimension": "clarity", "score": 8.0, "rationale": "Clear.", "confidence": 0.9},
                {"dimension": "cta", "score": 6.0, "rationale": "Weak CTA.", "confidence": 0.8},
            ],
            "weighted_average": 7.2,
            "passed_threshold": True,
            "hard_gate_failures": [],
            "evaluator_model": "gemini-2.5-pro",
            "token_count": 500,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
        }
        result = EvaluationResult.model_validate(data)
        dumped = result.model_dump()
        assert dumped == data
        assert len(result.scores) == 2

    def test_hard_gate_failures(self):
        result = EvaluationResult(
            ad_id="ad-002",
            scores=[],
            weighted_average=4.0,
            passed_threshold=False,
            hard_gate_failures=["brand_voice"],
        )
        assert result.hard_gate_failures == ["brand_voice"]

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            EvaluationResult.model_validate({"ad_id": "ad-001"})


# --- DecisionEntry ---


class TestDecisionEntry:
    def test_round_trip(self):
        now = datetime(2025, 1, 15, 12, 0, 0)
        data = {
            "timestamp": now,
            "component": "evaluator",
            "action": "selected gemini-2.5-pro for evaluation",
            "rationale": "Cross-model avoids self-bias",
            "context": {"ad_id": "ad-001"},
            "agent_id": "pipeline",
        }
        entry = DecisionEntry.model_validate(data)
        dumped = entry.model_dump()
        assert dumped["component"] == "evaluator"
        assert dumped["context"] == {"ad_id": "ad-001"}

    def test_defaults(self):
        entry = DecisionEntry(
            component="generator",
            action="chose temperature 0.7",
            rationale="Balance creativity and consistency",
        )
        assert entry.agent_id == ""
        assert entry.context == {}
        assert isinstance(entry.timestamp, datetime)


# --- IterationRecord ---


class TestIterationRecord:
    def test_round_trip(self):
        data = {
            "source_ad_id": "ad-001",
            "target_ad_id": "ad-002",
            "cycle_number": 1,
            "action_type": "component_fix",
            "weak_dimension": "cta",
            "delta_scores": {"cta": 2.0, "clarity": -0.5},
            "token_cost": 0.003,
        }
        record = IterationRecord.model_validate(data)
        assert record.model_dump() == data

    def test_full_regen_action_type(self):
        record = IterationRecord(
            source_ad_id="ad-001",
            target_ad_id="ad-003",
            cycle_number=2,
            action_type="full_regen",
            weak_dimension="brand_voice",
        )
        assert record.action_type == "full_regen"

    def test_invalid_action_type(self):
        with pytest.raises(ValidationError):
            IterationRecord(
                source_ad_id="ad-001",
                target_ad_id="ad-002",
                cycle_number=1,
                action_type="invalid",
                weak_dimension="cta",
            )


# --- Schema & Database ---


class TestSchema:
    def test_schema_creates_all_tables(self):
        """Verify schema.sql creates all six expected tables."""
        conn = sqlite3.connect(":memory:")
        from pathlib import Path

        schema_path = Path(__file__).parent.parent / "src" / "db" / "schema.sql"
        conn.executescript(schema_path.read_text())
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        conn.close()
        expected = {
            "ads",
            "evaluations",
            "iterations",
            "decisions",
            "competitor_ads",
            "quality_snapshots",
            "calibration_runs",
            "performance_feedback",
        }
        assert tables == expected

    def test_init_db_creates_tables(self):
        """Verify init_db returns a connection with all tables created."""
        conn = init_db(":memory:")
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        conn.close()
        expected = {
            "ads",
            "evaluations",
            "iterations",
            "decisions",
            "competitor_ads",
            "quality_snapshots",
            "calibration_runs",
            "performance_feedback",
        }
        assert tables == expected

    def test_init_db_idempotent(self):
        """Verify init_db can be called twice without error."""
        conn = init_db(":memory:")
        # Execute schema again — should not raise
        from pathlib import Path

        schema_path = Path(__file__).parent.parent / "src" / "db" / "schema.sql"
        conn.executescript(schema_path.read_text())
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        conn.close()
        assert len(tables) == 8

    def test_insert_and_query_ad(self):
        """Verify we can insert and retrieve an ad row."""
        conn = init_db(":memory:")
        conn.execute(
            "INSERT INTO ads (id, primary_text, headline, description, cta_button) "
            "VALUES (?, ?, ?, ?, ?)",
            ("ad-test", "Test text", "Test headline", "Test desc", "Learn More"),
        )
        row = conn.execute("SELECT id, headline FROM ads WHERE id = ?", ("ad-test",)).fetchone()
        conn.close()
        assert row == ("ad-test", "Test headline")

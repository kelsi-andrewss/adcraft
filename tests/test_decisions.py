"""Tests for the decision logger."""

import json

import pytest
from pydantic import ValidationError

from src.db.queries import list_decisions
from src.decisions.logger import log_decision
from src.models.decision import DecisionEntry


class TestDecisionEntryValidation:
    """DecisionEntry model validates fields correctly."""

    def test_requires_component(self):
        with pytest.raises(ValidationError):
            DecisionEntry(action="test", rationale="test")

    def test_requires_action(self):
        with pytest.raises(ValidationError):
            DecisionEntry(component="test", rationale="test")

    def test_requires_rationale(self):
        with pytest.raises(ValidationError):
            DecisionEntry(component="test", action="test")

    def test_auto_generates_timestamp(self):
        entry = DecisionEntry(component="test", action="act", rationale="why")
        assert entry.timestamp is not None

    def test_default_context_is_empty_dict(self):
        entry = DecisionEntry(component="test", action="act", rationale="why")
        assert entry.context == {}

    def test_default_agent_id_is_empty_string(self):
        entry = DecisionEntry(component="test", action="act", rationale="why")
        assert entry.agent_id == ""


class TestLogDecision:
    """log_decision() persists entries and returns IDs."""

    def test_basic_round_trip(self, db_conn):
        decision_id = log_decision(
            "generator",
            "selected temperature",
            "balancing creativity and consistency",
            conn=db_conn,
        )
        assert decision_id is not None
        rows = list_decisions(db_conn)
        assert len(rows) == 1
        assert rows[0]["component"] == "generator"
        assert rows[0]["action"] == "selected temperature"
        assert rows[0]["rationale"] == "balancing creativity and consistency"

    def test_returned_id_matches_persisted_row(self, db_conn):
        decision_id = log_decision(
            "evaluator", "chose model", "cross-model reduces bias", conn=db_conn
        )
        rows = list_decisions(db_conn)
        assert rows[0]["id"] == decision_id

    def test_context_dict_serialized_to_json(self, db_conn):
        ctx = {"ad_id": "ad-001", "score": 7.5}
        log_decision(
            "iterator",
            "retry generation",
            "score below threshold",
            context=ctx,
            conn=db_conn,
        )
        rows = list_decisions(db_conn)
        persisted_ctx = json.loads(rows[0]["context"])
        assert persisted_ctx == ctx

    def test_default_agent_id_is_system(self, db_conn):
        log_decision("pipeline", "start batch", "scheduled run", conn=db_conn)
        rows = list_decisions(db_conn)
        assert rows[0]["agent_id"] == "system"

    def test_custom_agent_id_preserved(self, db_conn):
        log_decision(
            "pipeline",
            "start batch",
            "manual trigger",
            conn=db_conn,
            agent_id="operator-kelsi",
        )
        rows = list_decisions(db_conn)
        assert rows[0]["agent_id"] == "operator-kelsi"

    def test_multiple_decisions_get_unique_ids(self, db_conn):
        id1 = log_decision("comp1", "act1", "reason1", conn=db_conn)
        id2 = log_decision("comp2", "act2", "reason2", conn=db_conn)
        assert id1 != id2

    def test_none_context_stored_as_empty_dict(self, db_conn):
        log_decision("test", "act", "why", context=None, conn=db_conn)
        rows = list_decisions(db_conn)
        persisted_ctx = json.loads(rows[0]["context"])
        assert persisted_ctx == {}

    def test_creates_missing_parent_directory(self, tmp_path, monkeypatch):
        db_path = tmp_path / "cold" / "start" / "ads.db"
        monkeypatch.setenv("DATABASE_PATH", str(db_path))
        assert not db_path.parent.exists()

        # Patch insert_decision to isolate directory creation from schema concerns
        monkeypatch.setattr("src.decisions.logger.insert_decision", lambda *a, **kw: "dec-test")

        decision_id = log_decision("pipeline", "cold-start", "dir was missing")

        assert decision_id == "dec-test"
        assert db_path.parent.exists()

    def test_context_with_datetime_and_path_serializes(self, db_conn):
        import datetime
        import pathlib

        ctx = {
            "generated_at": datetime.datetime(2026, 3, 15, 12, 0, 0),
            "output_file": pathlib.Path("/tmp/ads/output.json"),
        }
        # Before fix: raises TypeError; after fix: succeeds with string coercion
        decision_id = log_decision(
            "pipeline",
            "serialize-test",
            "context contains non-primitive types",
            context=ctx,
            conn=db_conn,
        )
        assert decision_id is not None
        rows = list_decisions(db_conn)
        persisted = json.loads(rows[0]["context"])
        assert persisted["generated_at"] == "2026-03-15 12:00:00"
        assert persisted["output_file"] == "/tmp/ads/output.json"

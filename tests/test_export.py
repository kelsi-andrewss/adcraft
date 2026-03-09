"""Tests for export subsystem: ad library, decision log, summary stats."""

from __future__ import annotations

import csv
import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from src.db.init_db import init_db
from src.db.queries import insert_ad, insert_decision, insert_evaluation
from src.output.exporter import export_ad_library, export_decision_log, export_summary_stats


@pytest.fixture()
def db_conn():
    conn = init_db(":memory:")
    yield conn
    conn.close()


@pytest.fixture()
def seeded_db(db_conn):
    """DB with ads, evaluations, and decisions."""
    ad_id = insert_ad(
        db_conn,
        primary_text="Test ad primary text",
        headline="Test Headline",
        description="Test description here",
        cta_button="Learn More",
        model_id="gemini-2.5-flash",
        cost_usd=0.001,
    )
    for dim, score in [
        ("clarity", 8.0),
        ("value_prop", 7.5),
        ("cta_effectiveness", 7.0),
        ("brand_voice", 6.5),
        ("emotional_resonance", 7.0),
    ]:
        insert_evaluation(
            db_conn,
            ad_id=ad_id,
            dimension=dim,
            score=score,
            rationale=f"Good {dim}",
            eval_mode="final",
            cost_usd=0.0005,
        )

    insert_decision(
        db_conn,
        timestamp=datetime(2026, 3, 9, 10, 0, 0),
        component="generator",
        action="generation_start",
        rationale="Test decision",
        context='{"test": true}',
    )

    return db_conn, ad_id


class TestExportAdLibrary:
    def test_export_csv(self, seeded_db) -> None:
        db_conn, ad_id = seeded_db
        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_ad_library(db_conn, "csv", Path(tmpdir) / "ads.csv")
            assert path.exists()
            with path.open() as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["headline"] == "Test Headline"
            assert "score_clarity" in rows[0]
            assert float(rows[0]["score_clarity"]) == 8.0

    def test_export_json(self, seeded_db) -> None:
        db_conn, ad_id = seeded_db
        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_ad_library(db_conn, "json", Path(tmpdir) / "ads.json")
            assert path.exists()
            data = json.loads(path.read_text())
            assert len(data) == 1
            assert data[0]["headline"] == "Test Headline"
            assert "scores" in data[0]
            assert data[0]["scores"]["clarity"] == 8.0

    def test_export_creates_parent_dirs(self, seeded_db) -> None:
        db_conn, _ = seeded_db
        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_ad_library(
                db_conn, "json", Path(tmpdir) / "nested" / "deep" / "ads.json"
            )
            assert path.exists()

    def test_export_empty_db(self, db_conn) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_ad_library(db_conn, "json", Path(tmpdir) / "empty.json")
            assert path.exists()
            data = json.loads(path.read_text())
            assert data == []


class TestExportDecisionLog:
    def test_export_decisions(self, seeded_db) -> None:
        db_conn, _ = seeded_db
        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_decision_log(db_conn, Path(tmpdir) / "decisions.json")
            assert path.exists()
            data = json.loads(path.read_text())
            # At least the seeded decision + decisions from logging
            assert len(data) >= 1
            assert any(d["component"] == "generator" for d in data)


class TestExportSummaryStats:
    def test_summary_structure(self, seeded_db) -> None:
        db_conn, _ = seeded_db
        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_summary_stats(db_conn, Path(tmpdir) / "stats.json")
            assert path.exists()
            stats = json.loads(path.read_text())

            assert stats["total_ads"] == 1
            assert 0 <= stats["pass_rate"] <= 1.0
            assert stats["avg_weighted_score"] > 0
            assert "dimension_averages" in stats
            assert "iteration_stats" in stats
            assert "avg_cycles_per_ad" in stats["iteration_stats"]
            assert "improvement_rate" in stats["iteration_stats"]

    def test_summary_empty_db(self, db_conn) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_summary_stats(db_conn, Path(tmpdir) / "stats.json")
            stats = json.loads(path.read_text())
            assert stats["total_ads"] == 0
            assert stats["pass_rate"] == 0.0

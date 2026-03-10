"""Tests for cost analytics: pricing calculation, quality snapshots, image costs."""

from __future__ import annotations

import sqlite3

import pytest

from src.analytics.cost import (
    IMAGE_PRICING,
    calculate_cost,
    calculate_image_cost,
    compare_image_model_efficiency,
    compute_creative_unit_cost,
    compute_quality_snapshot,
    get_performance_per_token,
    record_api_cost,
)
from src.db.init_db import init_db
from src.db.queries import (
    get_image_costs_by_model,
    insert_ad,
    insert_evaluation,
    update_ad_image,
)


@pytest.fixture()
def db_conn():
    conn = init_db(":memory:")
    yield conn
    conn.close()


class TestCalculateCost:
    def test_gemini_flash_pricing(self) -> None:
        cost = calculate_cost("gemini-2.5-flash", 1000, 500)
        # $0.30/1M input + $2.50/1M output
        expected = (1000 * 0.30 / 1_000_000) + (500 * 2.50 / 1_000_000)
        assert abs(cost - expected) < 1e-10

    def test_gemini_pro_pricing(self) -> None:
        cost = calculate_cost("gemini-2.5-pro", 1000, 500)
        expected = (1000 * 1.25 / 1_000_000) + (500 * 10.00 / 1_000_000)
        assert abs(cost - expected) < 1e-10

    def test_claude_sonnet_pricing(self) -> None:
        cost = calculate_cost("claude-sonnet-4-6", 1000, 500)
        expected = (1000 * 3.00 / 1_000_000) + (500 * 15.00 / 1_000_000)
        assert abs(cost - expected) < 1e-10

    def test_unknown_model_returns_zero(self) -> None:
        cost = calculate_cost("unknown-model-xyz", 1000, 500)
        assert cost == 0.0

    def test_zero_tokens(self) -> None:
        cost = calculate_cost("gemini-2.5-flash", 0, 0)
        assert cost == 0.0


class TestCalculateImageCost:
    def test_known_models_return_flat_rate(self) -> None:
        for model_id, expected_price in IMAGE_PRICING.items():
            assert calculate_image_cost(model_id) == expected_price

    def test_unknown_model_returns_zero(self) -> None:
        assert calculate_image_cost("some-unknown-image-model") == 0.0

    def test_gemini_flash_image_price(self) -> None:
        assert calculate_image_cost("gemini-2.5-flash-image") == 0.039

    def test_imagen_fast_price(self) -> None:
        assert calculate_image_cost("imagen-4.0-fast-generate-001") == 0.02


class TestRecordApiCost:
    def test_updates_ad_row(self, db_conn) -> None:
        ad_id = insert_ad(
            db_conn,
            primary_text="Test ad",
            headline="Test",
            description="Desc",
            cta_button="Learn More",
        )

        cost = record_api_cost(db_conn, "ads", ad_id, "gemini-2.5-flash", 500, 200)
        assert cost > 0

        row = db_conn.execute(
            "SELECT input_tokens, output_tokens, cost_usd FROM ads WHERE id = ?", (ad_id,)
        ).fetchone()
        assert row[0] == 500
        assert row[1] == 200
        assert row[2] == cost


class TestComputeQualitySnapshot:
    def test_empty_db_returns_zero_snapshot(self, db_conn) -> None:
        snapshot = compute_quality_snapshot(db_conn, cycle_number=1)
        assert snapshot["total_ads"] == 0
        assert snapshot["quality_per_dollar"] is None

    def test_snapshot_with_data(self, db_conn) -> None:
        ad_id = insert_ad(
            db_conn,
            primary_text="Test",
            headline="H",
            description="D",
            cta_button="Learn More",
            cost_usd=0.001,
        )
        for dim, score in [
            ("clarity", 8.0),
            ("value_prop", 7.0),
            ("cta_effectiveness", 7.5),
            ("brand_voice", 8.0),
            ("emotional_resonance", 7.0),
        ]:
            insert_evaluation(
                db_conn,
                ad_id=ad_id,
                dimension=dim,
                score=score,
                eval_mode="final",
            )

        snapshot = compute_quality_snapshot(db_conn, cycle_number=1)
        assert snapshot["total_ads"] == 1
        assert snapshot["avg_weighted_score"] is not None
        assert snapshot["avg_weighted_score"] > 0
        assert snapshot["ads_above_threshold"] >= 0
        assert snapshot["token_spend_usd"] >= 0

    def test_zero_spend_quality_per_dollar_is_none(self, db_conn) -> None:
        ad_id = insert_ad(
            db_conn,
            primary_text="Test",
            headline="H",
            description="D",
            cta_button="Learn More",
            cost_usd=0.0,
        )
        insert_evaluation(db_conn, ad_id=ad_id, dimension="clarity", score=8.0)

        snapshot = compute_quality_snapshot(db_conn, cycle_number=1)
        # With zero spend, quality_per_dollar should be None (no div by zero)
        assert snapshot["quality_per_dollar"] is None

    def test_snapshot_includes_image_costs(self, db_conn) -> None:
        ad_id = insert_ad(
            db_conn,
            primary_text="Test",
            headline="H",
            description="D",
            cta_button="Learn More",
            cost_usd=0.001,
        )
        update_ad_image(
            db_conn,
            ad_id,
            image_path="/tmp/img.png",
            visual_prompt="test prompt",
            image_model="gemini-2.5-flash-image",
            image_cost_usd=0.039,
        )
        insert_evaluation(
            db_conn,
            ad_id=ad_id,
            dimension="clarity",
            score=8.0,
            eval_mode="final",
            cost_usd=0.005,
        )

        snapshot = compute_quality_snapshot(db_conn, cycle_number=1)
        # token_spend_usd should include copy (0.001) + image (0.039) + eval (0.005)
        assert abs(snapshot["token_spend_usd"] - 0.045) < 1e-10

    def test_snapshot_text_only_ad_no_image_cost(self, db_conn) -> None:
        """Text-only ads (no image columns set) still compute correctly."""
        ad_id = insert_ad(
            db_conn,
            primary_text="Test",
            headline="H",
            description="D",
            cta_button="Learn More",
            cost_usd=0.002,
        )
        insert_evaluation(
            db_conn,
            ad_id=ad_id,
            dimension="clarity",
            score=7.0,
            eval_mode="final",
            cost_usd=0.003,
        )

        snapshot = compute_quality_snapshot(db_conn, cycle_number=1)
        # Only copy + eval cost, no image cost
        assert abs(snapshot["token_spend_usd"] - 0.005) < 1e-10

    def test_snapshot_pre_migration_db(self) -> None:
        """Snapshot works on a DB without image_cost_usd column."""
        conn = sqlite3.connect(":memory:")
        conn.execute("PRAGMA foreign_keys=ON")
        # Create ads table WITHOUT image columns (pre-migration state)
        conn.execute(
            """CREATE TABLE ads (
                id TEXT PRIMARY KEY,
                brief_id TEXT,
                primary_text TEXT NOT NULL,
                headline TEXT NOT NULL,
                description TEXT NOT NULL,
                cta_button TEXT NOT NULL,
                model_id TEXT,
                temperature REAL,
                generation_seed TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost_usd REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        conn.execute(
            """CREATE TABLE evaluations (
                id TEXT PRIMARY KEY,
                ad_id TEXT NOT NULL REFERENCES ads(id),
                dimension TEXT NOT NULL,
                score REAL NOT NULL,
                rationale TEXT,
                confidence REAL,
                evaluator_model TEXT,
                eval_mode TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost_usd REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        conn.execute(
            """CREATE TABLE quality_snapshots (
                id TEXT PRIMARY KEY,
                cycle_number INTEGER NOT NULL,
                avg_weighted_score REAL,
                dimension_averages JSON,
                ads_above_threshold INTEGER,
                total_ads INTEGER,
                token_spend_usd REAL,
                quality_per_dollar REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        conn.execute(
            """CREATE TABLE decisions (
                id TEXT PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                component TEXT NOT NULL,
                action TEXT NOT NULL,
                rationale TEXT,
                context TEXT,
                agent_id TEXT
            )"""
        )
        conn.execute(
            "INSERT INTO ads (id, primary_text, headline, description, cta_button, cost_usd) "
            "VALUES ('a1', 'text', 'head', 'desc', 'cta', 0.01)"
        )
        conn.commit()

        snapshot = compute_quality_snapshot(conn, cycle_number=1)
        assert snapshot["total_ads"] == 1
        # Should not error; image cost defaults to 0
        assert snapshot["token_spend_usd"] == 0.01
        conn.close()


class TestGetPerformancePerToken:
    def test_returns_snapshots(self, db_conn) -> None:
        from src.db.queries import insert_quality_snapshot

        insert_quality_snapshot(
            db_conn,
            cycle_number=1,
            avg_weighted_score=7.5,
            quality_per_dollar=100.0,
            token_spend_usd=0.075,
        )
        results = get_performance_per_token(db_conn)
        assert len(results) >= 1


class TestComputeCreativeUnitCost:
    def test_text_only_ad(self, db_conn) -> None:
        ad_id = insert_ad(
            db_conn,
            primary_text="Test",
            headline="H",
            description="D",
            cta_button="CTA",
            cost_usd=0.001,
        )
        insert_evaluation(db_conn, ad_id=ad_id, dimension="clarity", score=8.0, cost_usd=0.005)

        result = compute_creative_unit_cost(db_conn, ad_id)
        assert result["ad_id"] == ad_id
        assert result["copy_gen_cost"] == 0.001
        assert result["image_gen_cost"] == 0.0
        assert result["eval_cost"] == 0.005
        assert abs(result["total_cost"] - 0.006) < 1e-10
        assert result["image_model"] is None
        assert result["has_image"] is False

    def test_ad_with_image(self, db_conn) -> None:
        ad_id = insert_ad(
            db_conn,
            primary_text="Test",
            headline="H",
            description="D",
            cta_button="CTA",
            cost_usd=0.001,
        )
        update_ad_image(
            db_conn,
            ad_id,
            image_path="/tmp/img.png",
            visual_prompt="a tutoring ad",
            image_model="gemini-2.5-flash-image",
            image_cost_usd=0.039,
        )
        insert_evaluation(db_conn, ad_id=ad_id, dimension="clarity", score=8.0, cost_usd=0.005)

        result = compute_creative_unit_cost(db_conn, ad_id)
        assert result["copy_gen_cost"] == 0.001
        assert result["image_gen_cost"] == 0.039
        assert result["eval_cost"] == 0.005
        assert abs(result["total_cost"] - 0.045) < 1e-10
        assert result["image_model"] == "gemini-2.5-flash-image"
        assert result["has_image"] is True

    def test_nonexistent_ad_returns_zeros(self, db_conn) -> None:
        result = compute_creative_unit_cost(db_conn, "nonexistent-id")
        assert result["total_cost"] == 0.0
        assert result["has_image"] is False

    def test_multiple_evaluations_summed(self, db_conn) -> None:
        ad_id = insert_ad(
            db_conn,
            primary_text="Test",
            headline="H",
            description="D",
            cta_button="CTA",
            cost_usd=0.001,
        )
        for dim in ["clarity", "value_prop", "brand_voice"]:
            insert_evaluation(db_conn, ad_id=ad_id, dimension=dim, score=7.0, cost_usd=0.003)

        result = compute_creative_unit_cost(db_conn, ad_id)
        assert abs(result["eval_cost"] - 0.009) < 1e-10
        assert abs(result["total_cost"] - 0.010) < 1e-10


class TestCompareImageModelEfficiency:
    def test_no_image_ads_returns_empty(self, db_conn) -> None:
        # Insert a text-only ad
        insert_ad(
            db_conn,
            primary_text="Test",
            headline="H",
            description="D",
            cta_button="CTA",
        )
        results = compare_image_model_efficiency(db_conn)
        assert results == []

    def test_single_model_with_scores(self, db_conn) -> None:
        ad_id = insert_ad(
            db_conn,
            primary_text="Test",
            headline="H",
            description="D",
            cta_button="CTA",
            cost_usd=0.001,
        )
        update_ad_image(
            db_conn,
            ad_id,
            image_path="/tmp/img.png",
            visual_prompt="prompt",
            image_model="gemini-2.5-flash-image",
            image_cost_usd=0.039,
        )
        # Add visual evaluation scores
        for dim, score in [
            ("brand_consistency", 8.0),
            ("composition_quality", 7.5),
            ("text_image_synergy", 7.0),
        ]:
            insert_evaluation(db_conn, ad_id=ad_id, dimension=dim, score=score)

        results = compare_image_model_efficiency(db_conn)
        assert len(results) == 1
        assert results[0]["image_model"] == "gemini-2.5-flash-image"
        assert results[0]["ad_count"] == 1
        assert results[0]["total_image_cost"] == 0.039
        assert results[0]["avg_image_cost"] == 0.039
        assert results[0]["avg_visual_score"] is not None
        assert results[0]["quality_per_dollar"] is not None
        # avg_visual_score = (8.0 + 7.5 + 7.0) / 3 = 7.5
        assert abs(results[0]["avg_visual_score"] - 7.5) < 1e-10
        # quality_per_dollar = 7.5 / 0.039
        assert abs(results[0]["quality_per_dollar"] - 7.5 / 0.039) < 1e-6

    def test_multiple_models(self, db_conn) -> None:
        for model, cost in [
            ("gemini-2.5-flash-image", 0.039),
            ("imagen-4.0-generate-001", 0.04),
        ]:
            ad_id = insert_ad(
                db_conn,
                primary_text="Test",
                headline="H",
                description="D",
                cta_button="CTA",
            )
            update_ad_image(
                db_conn,
                ad_id,
                image_path=f"/tmp/{model}.png",
                visual_prompt="prompt",
                image_model=model,
                image_cost_usd=cost,
            )

        results = compare_image_model_efficiency(db_conn)
        assert len(results) == 2
        models = {r["image_model"] for r in results}
        assert "gemini-2.5-flash-image" in models
        assert "imagen-4.0-generate-001" in models

    def test_model_without_visual_scores(self, db_conn) -> None:
        ad_id = insert_ad(
            db_conn,
            primary_text="Test",
            headline="H",
            description="D",
            cta_button="CTA",
        )
        update_ad_image(
            db_conn,
            ad_id,
            image_path="/tmp/img.png",
            visual_prompt="prompt",
            image_model="gemini-2.5-flash-image",
            image_cost_usd=0.039,
        )
        # No visual evaluations added

        results = compare_image_model_efficiency(db_conn)
        assert len(results) == 1
        assert results[0]["avg_visual_score"] is None
        assert results[0]["quality_per_dollar"] is None


class TestGetImageCostsByModel:
    def test_empty_returns_empty(self, db_conn) -> None:
        results = get_image_costs_by_model(db_conn)
        assert results == []

    def test_groups_by_model(self, db_conn) -> None:
        for i, (model, cost) in enumerate(
            [
                ("model-a", 0.04),
                ("model-a", 0.04),
                ("model-b", 0.08),
            ]
        ):
            ad_id = insert_ad(
                db_conn,
                primary_text=f"Test {i}",
                headline="H",
                description="D",
                cta_button="CTA",
            )
            update_ad_image(
                db_conn,
                ad_id,
                image_path=f"/tmp/img{i}.png",
                visual_prompt="prompt",
                image_model=model,
                image_cost_usd=cost,
            )

        results = get_image_costs_by_model(db_conn)
        by_model = {r["image_model"]: r for r in results}

        assert len(results) == 2
        assert by_model["model-a"]["ad_count"] == 2
        assert abs(by_model["model-a"]["total_cost"] - 0.08) < 1e-10
        assert abs(by_model["model-a"]["avg_cost"] - 0.04) < 1e-10
        assert by_model["model-b"]["ad_count"] == 1
        assert abs(by_model["model-b"]["total_cost"] - 0.08) < 1e-10

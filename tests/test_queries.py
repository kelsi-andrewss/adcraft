"""Tests for parameterized SQL query helpers."""


from src.db.queries import (
    get_ad,
    get_evaluations_for_ad,
    get_iterations_for_ad,
    insert_ad,
    insert_competitor_ad,
    insert_evaluation,
    insert_iteration,
    insert_quality_snapshot,
    list_ads,
    list_competitor_ads,
    list_decisions,
    list_quality_snapshots,
)


class TestAdsQueries:
    """Round-trip tests for the ads table."""

    def test_insert_and_get(self, db_conn):
        ad_id = insert_ad(
            db_conn,
            primary_text="Expert SAT tutoring for your child.",
            headline="Boost Your SAT Score",
            description="Personalized 1-on-1 sessions.",
            cta_button="Start Now",
            brief_id="brief-001",
            model_id="gemini-2.5-flash",
        )
        row = get_ad(db_conn, ad_id)
        assert row is not None
        assert row["id"] == ad_id
        assert row["primary_text"] == "Expert SAT tutoring for your child."
        assert row["headline"] == "Boost Your SAT Score"
        assert row["brief_id"] == "brief-001"

    def test_get_nonexistent_returns_none(self, db_conn):
        assert get_ad(db_conn, "nonexistent-id") is None

    def test_list_ads_empty(self, db_conn):
        result = list_ads(db_conn)
        assert result == []

    def test_list_ads_returns_inserted(self, db_conn):
        insert_ad(
            db_conn,
            primary_text="Ad 1",
            headline="H1",
            description="D1",
            cta_button="CTA1",
        )
        insert_ad(
            db_conn,
            primary_text="Ad 2",
            headline="H2",
            description="D2",
            cta_button="CTA2",
        )
        result = list_ads(db_conn)
        assert len(result) == 2

    def test_insert_generates_unique_ids(self, db_conn):
        id1 = insert_ad(
            db_conn,
            primary_text="A",
            headline="H",
            description="D",
            cta_button="C",
        )
        id2 = insert_ad(
            db_conn,
            primary_text="B",
            headline="H",
            description="D",
            cta_button="C",
        )
        assert id1 != id2

    def test_optional_fields_default_to_none(self, db_conn):
        ad_id = insert_ad(
            db_conn,
            primary_text="Text",
            headline="Head",
            description="Desc",
            cta_button="CTA",
        )
        row = get_ad(db_conn, ad_id)
        assert row["model_id"] is None
        assert row["temperature"] is None
        assert row["cost_usd"] is None


class TestEvaluationsQueries:
    """Round-trip tests for the evaluations table."""

    def test_insert_and_get_for_ad(self, db_conn):
        ad_id = insert_ad(
            db_conn,
            primary_text="T",
            headline="H",
            description="D",
            cta_button="C",
        )
        eval_id = insert_evaluation(
            db_conn,
            ad_id=ad_id,
            dimension="clarity",
            score=8.5,
            rationale="Clear and compelling.",
            confidence=0.9,
            evaluator_model="gemini-2.5-pro",
            eval_mode="final",
        )
        evals = get_evaluations_for_ad(db_conn, ad_id)
        assert len(evals) == 1
        assert evals[0]["id"] == eval_id
        assert evals[0]["dimension"] == "clarity"
        assert evals[0]["score"] == 8.5
        assert evals[0]["eval_mode"] == "final"

    def test_get_evaluations_empty(self, db_conn):
        result = get_evaluations_for_ad(db_conn, "no-such-ad")
        assert result == []

    def test_multiple_evaluations_for_same_ad(self, db_conn):
        ad_id = insert_ad(
            db_conn,
            primary_text="T",
            headline="H",
            description="D",
            cta_button="C",
        )
        insert_evaluation(db_conn, ad_id=ad_id, dimension="clarity", score=8.0)
        insert_evaluation(db_conn, ad_id=ad_id, dimension="cta_strength", score=6.0)
        evals = get_evaluations_for_ad(db_conn, ad_id)
        assert len(evals) == 2


class TestIterationsQueries:
    """Round-trip tests for the iterations table."""

    def test_insert_and_get_for_ad(self, db_conn):
        src_id = insert_ad(
            db_conn,
            primary_text="V1",
            headline="H",
            description="D",
            cta_button="C",
        )
        tgt_id = insert_ad(
            db_conn,
            primary_text="V2",
            headline="H",
            description="D",
            cta_button="C",
        )
        iter_id = insert_iteration(
            db_conn,
            source_ad_id=src_id,
            target_ad_id=tgt_id,
            cycle_number=1,
            action_type="component_fix",
            weak_dimension="cta",
            delta_weighted_avg=1.5,
        )
        iters = get_iterations_for_ad(db_conn, src_id)
        assert len(iters) == 1
        assert iters[0]["id"] == iter_id
        assert iters[0]["action_type"] == "component_fix"
        assert iters[0]["weak_dimension"] == "cta"

    def test_get_iterations_empty(self, db_conn):
        result = get_iterations_for_ad(db_conn, "no-such-ad")
        assert result == []


class TestDecisionsQueries:
    """Tests for the decisions table (tested more thoroughly in test_decisions.py)."""

    def test_list_decisions_empty(self, db_conn):
        result = list_decisions(db_conn)
        assert result == []

    def test_filter_by_component(self, db_conn):
        from datetime import datetime

        from src.db.queries import insert_decision

        insert_decision(
            db_conn,
            timestamp=datetime.utcnow(),
            component="generator",
            action="act1",
            rationale="r1",
        )
        insert_decision(
            db_conn,
            timestamp=datetime.utcnow(),
            component="evaluator",
            action="act2",
            rationale="r2",
        )
        gen_decisions = list_decisions(db_conn, component="generator")
        assert len(gen_decisions) == 1
        assert gen_decisions[0]["component"] == "generator"


class TestCompetitorAdsQueries:
    """Smoke tests for the competitor_ads table."""

    def test_insert_and_list(self, db_conn):
        comp_id = insert_competitor_ad(
            db_conn,
            brand="Kaplan",
            headline="Guaranteed Score Improvement",
            hook_type="guarantee",
        )
        result = list_competitor_ads(db_conn)
        assert len(result) == 1
        assert result[0]["id"] == comp_id
        assert result[0]["brand"] == "Kaplan"

    def test_list_empty(self, db_conn):
        result = list_competitor_ads(db_conn)
        assert result == []

    def test_filter_by_brand(self, db_conn):
        insert_competitor_ad(db_conn, brand="Kaplan")
        insert_competitor_ad(db_conn, brand="Princeton Review")
        kaplan = list_competitor_ads(db_conn, brand="Kaplan")
        assert len(kaplan) == 1
        assert kaplan[0]["brand"] == "Kaplan"


class TestQualitySnapshotsQueries:
    """Smoke tests for the quality_snapshots table."""

    def test_insert_and_list(self, db_conn):
        dim_avgs = {"clarity": 8.0, "cta": 6.5}
        snap_id = insert_quality_snapshot(
            db_conn,
            cycle_number=1,
            avg_weighted_score=7.25,
            dimension_averages=dim_avgs,
            ads_above_threshold=3,
            total_ads=5,
            token_spend_usd=0.05,
            quality_per_dollar=145.0,
        )
        result = list_quality_snapshots(db_conn)
        assert len(result) == 1
        assert result[0]["id"] == snap_id
        assert result[0]["dimension_averages"] == dim_avgs
        assert result[0]["avg_weighted_score"] == 7.25

    def test_list_empty(self, db_conn):
        result = list_quality_snapshots(db_conn)
        assert result == []

    def test_null_dimension_averages(self, db_conn):
        insert_quality_snapshot(db_conn, cycle_number=1)
        result = list_quality_snapshots(db_conn)
        assert result[0]["dimension_averages"] is None

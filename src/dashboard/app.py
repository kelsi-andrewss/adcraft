"""AdCraft Dashboard — Streamlit presentation layer for the Autonomous Ad Engine.

Six-tab interface: Ad Library, Quality Trends, Iteration Inspector,
Decision Log, Cost Tracker, Pipeline Runner. Zero business logic —
all data via queries.py, all charts via trends.py / cost.py.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from src.analytics.trends import (
    before_after_comparison,
    convergence_curves,
    cost_efficiency_trend,
    dimension_breakdown,
    score_distribution,
)
from src.db.init_db import init_db
from src.db.queries import (
    get_ad,
    get_all_decisions,
    get_evaluations_for_ad,
    get_iterations_for_ad,
    get_quality_snapshots,
    list_ads,
)
from src.evaluate.rubrics import DIMENSION_WEIGHTS, PASSING_THRESHOLD

DB_PATH = "data/ads.db"

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Ad Engine Dashboard",
    page_icon=":zap:",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS for professional demo appearance
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* Metric card alignment */
    div[data-testid="stMetric"] {
        background-color: rgba(23, 226, 234, 0.06);
        border: 1px solid rgba(23, 226, 234, 0.15);
        border-radius: 8px;
        padding: 12px 16px;
    }
    /* Tighter expander spacing */
    div[data-testid="stExpander"] {
        margin-bottom: 4px;
    }
    /* Tab content top padding */
    div[data-testid="stTabContent"] {
        padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------


def get_connection() -> sqlite3.Connection:
    """Return a database connection, initializing the schema if needed."""
    return init_db(DB_PATH)


def _db_exists() -> bool:
    return Path(DB_PATH).exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_weighted_score(evals: list[dict]) -> float | None:
    """Compute weighted average from a list of evaluation dicts."""
    if not evals:
        return None
    dim_scores = {e["dimension"]: e["score"] for e in evals if e.get("dimension")}
    if not dim_scores:
        return None
    return sum(dim_scores.get(dim, 0.0) * weight for dim, weight in DIMENSION_WEIGHTS.items())


def _pass_icon(passed: bool) -> str:
    return "Pass" if passed else "Fail"


def _truncate(text: str | None, length: int = 60) -> str:
    if not text:
        return ""
    return text[:length] + "..." if len(text) > length else text


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("AdCraft Engine")
    st.caption("Autonomous Ad Generation & Evaluation")
    st.divider()

    if _db_exists():
        st.success("Database connected")
    else:
        st.warning("No database file — run the pipeline first")

    conn = get_connection()

    # Summary stats
    ads = list_ads(conn, limit=10_000)
    total_ads = len(ads)

    # Compute pass rate from evaluations
    passed_count = 0
    for ad in ads:
        evals = get_evaluations_for_ad(conn, ad["id"])
        score = _compute_weighted_score(evals)
        if score is not None and score >= PASSING_THRESHOLD:
            passed_count += 1

    pass_rate = (passed_count / total_ads * 100) if total_ads > 0 else 0.0

    # Total iterations
    conn.row_factory = sqlite3.Row
    iter_count_row = conn.execute("SELECT COUNT(*) as cnt FROM iterations").fetchone()
    total_iterations = iter_count_row["cnt"] if iter_count_row else 0

    # Total cost
    cost_row = conn.execute("SELECT COALESCE(SUM(cost_usd), 0.0) as total FROM ads").fetchone()
    ad_cost = cost_row["total"] if cost_row else 0.0
    eval_cost_row = conn.execute(
        "SELECT COALESCE(SUM(cost_usd), 0.0) as total FROM evaluations"
    ).fetchone()
    eval_cost = eval_cost_row["total"] if eval_cost_row else 0.0
    total_cost = ad_cost + eval_cost

    col1, col2 = st.columns(2)
    col1.metric("Total Ads", total_ads)
    col2.metric("Pass Rate", f"{pass_rate:.0f}%")

    col3, col4 = st.columns(2)
    col3.metric("Iterations", total_iterations)
    col4.metric("Total Cost", f"${total_cost:.4f}")

    st.divider()
    if st.button("Refresh", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_library, tab_trends, tab_iter, tab_decisions, tab_cost, tab_runner = st.tabs(
    [
        "Ad Library",
        "Quality Trends",
        "Iteration Inspector",
        "Decision Log",
        "Cost Tracker",
        "Pipeline Runner",
    ]
)

# ---- Tab 1: Ad Library ----------------------------------------------------

with tab_library:
    st.header("Ad Library")

    if total_ads == 0:
        st.info("No ads yet — run the pipeline to generate your first batch.")
    else:
        # Filters
        filter_col1, filter_col2 = st.columns([1, 3])
        with filter_col1:
            status_filter = st.selectbox("Status", ["All", "Pass", "Fail"], key="lib_status")
        with filter_col2:
            search_query = st.text_input("Search headlines & primary text", key="lib_search")

        # Build display data
        rows = []
        for ad in ads:
            evals = get_evaluations_for_ad(conn, ad["id"])
            score = _compute_weighted_score(evals)
            passed = score is not None and score >= PASSING_THRESHOLD

            # Apply filters
            if status_filter == "Pass" and not passed:
                continue
            if status_filter == "Fail" and passed:
                continue
            if search_query:
                needle = search_query.lower()
                haystack = (
                    (ad.get("headline") or "") + " " + (ad.get("primary_text") or "")
                ).lower()
                if needle not in haystack:
                    continue

            rows.append(
                {
                    "id": ad["id"][:8],
                    "headline": ad.get("headline", ""),
                    "primary_text": _truncate(ad.get("primary_text"), 80),
                    "cta": ad.get("cta_button", ""),
                    "score": round(score, 2) if score is not None else None,
                    "status": _pass_icon(passed),
                    "created": ad.get("created_at", ""),
                    "_full_id": ad["id"],
                    "_evals": evals,
                    "_ad": ad,
                }
            )

        if rows:
            st.caption(f"Showing {len(rows)} of {total_ads} ads")

            # Dataframe overview
            df = pd.DataFrame(rows)[
                ["id", "headline", "primary_text", "cta", "score", "status", "created"]
            ]
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "id": st.column_config.TextColumn("ID", width="small"),
                    "headline": st.column_config.TextColumn("Headline", width="medium"),
                    "primary_text": st.column_config.TextColumn("Primary Text", width="large"),
                    "cta": st.column_config.TextColumn("CTA", width="small"),
                    "score": st.column_config.NumberColumn("Score", format="%.2f", width="small"),
                    "status": st.column_config.TextColumn("Status", width="small"),
                    "created": st.column_config.TextColumn("Created", width="medium"),
                },
            )

            # Expanders for detail
            for row in rows:
                ad_data = row["_ad"]
                evals_data = row["_evals"]
                with st.expander(f"{row['headline']} — {row['status']} ({row['score']})"):
                    dcol1, dcol2 = st.columns(2)
                    with dcol1:
                        st.subheader("Ad Copy")
                        st.markdown(f"**Headline:** {ad_data.get('headline', '')}")
                        st.markdown(f"**Primary Text:** {ad_data.get('primary_text', '')}")
                        st.markdown(f"**Description:** {ad_data.get('description', '')}")
                        st.markdown(f"**CTA:** {ad_data.get('cta_button', '')}")
                        st.markdown(f"**Model:** {ad_data.get('model_id', 'N/A')}")
                    with dcol2:
                        st.subheader("Dimension Scores")
                        for ev in evals_data:
                            dim_name = (ev.get("dimension") or "").replace("_", " ").title()
                            st.markdown(f"**{dim_name}:** {ev.get('score', 'N/A')}")
                            if ev.get("rationale"):
                                st.caption(ev["rationale"])
        else:
            st.info("No ads match the current filters.")

# ---- Tab 2: Quality Trends ------------------------------------------------

with tab_trends:
    st.header("Quality Trends")

    if total_ads == 0:
        st.info("No data yet — run the pipeline to see quality trends.")
    else:
        st.subheader("Score Distribution")
        st.caption(
            "Histogram of weighted average scores across all evaluated ads. "
            "The dashed line marks the passing threshold."
        )
        st.plotly_chart(score_distribution(conn), use_container_width=True)

        st.subheader("Dimension Breakdown")
        st.caption("Average score per evaluation dimension across all ads.")
        st.plotly_chart(dimension_breakdown(conn), use_container_width=True)

        st.subheader("Convergence Curves")
        st.caption(
            "Average weighted score per iteration cycle, with per-dimension trends. "
            "Tracks whether the system improves over successive cycles."
        )
        st.plotly_chart(convergence_curves(conn), use_container_width=True)

        st.subheader("Cost Efficiency Trend")
        st.caption(
            "Quality-per-dollar and token spend over cycles — "
            "measures whether we're getting better results per dollar spent."
        )
        st.plotly_chart(cost_efficiency_trend(conn), use_container_width=True)

# ---- Tab 3: Iteration Inspector -------------------------------------------

with tab_iter:
    st.header("Iteration Inspector")

    if total_iterations == 0:
        st.info(
            "No iteration history yet — run the pipeline to generate ads with iteration cycles."
        )
    else:
        # Find ads that have been iterated on (they appear as source_ad_id)
        conn.row_factory = sqlite3.Row
        source_rows = conn.execute(
            "SELECT DISTINCT source_ad_id FROM iterations ORDER BY source_ad_id"
        ).fetchall()
        source_ids = [r["source_ad_id"] for r in source_rows]

        # Build a label map: id -> headline
        ad_labels: dict[str, str] = {}
        for sid in source_ids:
            ad_data = get_ad(conn, sid)
            if ad_data:
                headline = ad_data.get("headline", sid[:8])
                ad_labels[sid] = f"{headline} ({sid[:8]})"
            else:
                ad_labels[sid] = sid[:8]

        selected_label = st.selectbox(
            "Select an ad to inspect",
            options=list(ad_labels.values()),
            key="iter_select",
        )

        # Reverse-lookup the ID
        selected_id = next((k for k, v in ad_labels.items() if v == selected_label), None)

        if selected_id:
            # Build the iteration chain: source -> target -> target -> ...
            chain_ids = [selected_id]
            current_id = selected_id
            visited: set[str] = {current_id}
            while True:
                iters = get_iterations_for_ad(conn, current_id)
                if not iters:
                    break
                next_id = iters[0]["target_ad_id"]
                if next_id in visited:
                    break
                chain_ids.append(next_id)
                visited.add(next_id)
                current_id = next_id

            # Display the chain as a vertical timeline
            prev_score: float | None = None
            for idx, ad_id in enumerate(chain_ids):
                ad_data = get_ad(conn, ad_id)
                if not ad_data:
                    continue
                evals = get_evaluations_for_ad(conn, ad_id)
                score = _compute_weighted_score(evals)

                # Get iteration metadata for this step (if not the first)
                iter_meta = None
                if idx > 0:
                    parent_iters = get_iterations_for_ad(conn, chain_ids[idx - 1])
                    for it in parent_iters:
                        if it["target_ad_id"] == ad_id:
                            iter_meta = it
                            break

                # Determine pass/fail
                passed = score is not None and score >= PASSING_THRESHOLD
                status_color = "green" if passed else "red"

                with st.container(border=True):
                    step_label = "Original" if idx == 0 else f"Cycle {idx}"
                    st.markdown(
                        f"### :{status_color}[{step_label}] — {ad_data.get('headline', 'N/A')}"
                    )

                    mcol1, mcol2, mcol3 = st.columns(3)
                    mcol1.metric(
                        "Weighted Score",
                        f"{score:.2f}" if score is not None else "N/A",
                        delta=(
                            f"{score - prev_score:+.2f}"
                            if score is not None and prev_score is not None
                            else None
                        ),
                    )
                    if iter_meta:
                        mcol2.metric("Action", iter_meta.get("action_type", "N/A"))
                        mcol3.metric(
                            "Weak Dimension",
                            (iter_meta.get("weak_dimension") or "N/A").replace("_", " ").title(),
                        )
                    else:
                        mcol2.metric("Action", "Initial Generation")
                        mcol3.metric("Status", _pass_icon(passed))

                    # Show ad copy and scores in expander
                    with st.expander("Details"):
                        dcol1, dcol2 = st.columns(2)
                        with dcol1:
                            st.markdown(f"**Primary Text:** {ad_data.get('primary_text', '')}")
                            st.markdown(f"**Description:** {ad_data.get('description', '')}")
                            st.markdown(f"**CTA:** {ad_data.get('cta_button', '')}")
                        with dcol2:
                            for ev in evals:
                                dim_name = (ev.get("dimension") or "").replace("_", " ").title()
                                st.markdown(f"**{dim_name}:** {ev.get('score', 'N/A')}")

                prev_score = score

            # Before/after chart for the original ad
            if len(chain_ids) > 1:
                st.subheader("Before / After Comparison")
                st.caption(
                    "Dimension scores for the final version of this ad, "
                    "comparing iteration-mode vs final-mode evaluations."
                )
                final_ad_id = chain_ids[-1]
                st.plotly_chart(
                    before_after_comparison(conn, final_ad_id),
                    use_container_width=True,
                )

# ---- Tab 4: Decision Log --------------------------------------------------

with tab_decisions:
    st.header("Decision Log")

    all_decisions = get_all_decisions(conn)

    if not all_decisions:
        st.info("No decisions logged yet — run the pipeline to populate the log.")
    else:
        # Filters
        fcol1, fcol2 = st.columns([1, 3])
        with fcol1:
            components = sorted({d["component"] for d in all_decisions})
            selected_components = st.multiselect(
                "Filter by component", options=components, key="dec_components"
            )
        with fcol2:
            dec_search = st.text_input("Search action / rationale", key="dec_search")

        # Filter
        filtered = all_decisions
        if selected_components:
            filtered = [d for d in filtered if d["component"] in selected_components]
        if dec_search:
            needle = dec_search.lower()
            filtered = [
                d
                for d in filtered
                if needle in (d.get("action") or "").lower()
                or needle in (d.get("rationale") or "").lower()
            ]

        # Sort most-recent first for display
        filtered = list(reversed(filtered))

        st.caption(f"Showing {len(filtered)} of {len(all_decisions)} decisions")

        if filtered:
            df_dec = pd.DataFrame(filtered)[
                ["timestamp", "component", "action", "rationale", "agent_id"]
            ]
            df_dec["rationale"] = df_dec["rationale"].apply(lambda x: _truncate(x, 100))
            st.dataframe(
                df_dec,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "timestamp": st.column_config.TextColumn("Timestamp", width="medium"),
                    "component": st.column_config.TextColumn("Component", width="small"),
                    "action": st.column_config.TextColumn("Action", width="medium"),
                    "rationale": st.column_config.TextColumn("Rationale", width="large"),
                    "agent_id": st.column_config.TextColumn("Agent", width="small"),
                },
            )

            # Expanders for full details
            for d in filtered[:50]:  # Limit expanders for performance
                with st.expander(
                    f"{d.get('timestamp', '')} — {d.get('component', '')} / "
                    f"{_truncate(d.get('action', ''), 50)}"
                ):
                    st.markdown(f"**Component:** {d.get('component', '')}")
                    st.markdown(f"**Action:** {d.get('action', '')}")
                    st.markdown(f"**Rationale:** {d.get('rationale', 'N/A')}")
                    st.markdown(f"**Context:** {d.get('context', 'N/A')}")
                    st.markdown(f"**Agent:** {d.get('agent_id', 'N/A')}")
        else:
            st.info("No decisions match the current filters.")

# ---- Tab 5: Cost Tracker --------------------------------------------------

with tab_cost:
    st.header("Cost Tracker")

    snapshots = get_quality_snapshots(conn)

    if not snapshots and total_ads == 0:
        st.info("No cost data yet — run the pipeline to start tracking spend.")
    else:
        # Summary metrics row
        total_spend = total_cost  # From sidebar calculation
        avg_cost_per_ad = total_spend / total_ads if total_ads > 0 else 0.0

        # Best quality-per-dollar from snapshots
        qpd_values = [
            s["quality_per_dollar"] for s in snapshots if s.get("quality_per_dollar") is not None
        ]
        best_qpd = max(qpd_values) if qpd_values else 0.0

        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Total Spend", f"${total_spend:.4f}")
        mcol2.metric("Avg Cost / Ad", f"${avg_cost_per_ad:.4f}")
        mcol3.metric("Best Quality/$", f"{best_qpd:.1f}")

        st.divider()

        # Cost breakdown by model
        st.subheader("Cost by Model")
        st.caption("Token spend breakdown across generation and evaluation models.")
        conn.row_factory = sqlite3.Row
        model_costs_rows = conn.execute(
            """
            SELECT model_id, SUM(cost_usd) as total_cost,
                   SUM(input_tokens) as total_input, SUM(output_tokens) as total_output,
                   COUNT(*) as call_count
            FROM (
                SELECT model_id, cost_usd, input_tokens, output_tokens FROM ads
                WHERE model_id IS NOT NULL
                UNION ALL
                SELECT evaluator_model as model_id, cost_usd, input_tokens, output_tokens
                FROM evaluations WHERE evaluator_model IS NOT NULL
            )
            GROUP BY model_id
            ORDER BY total_cost DESC
            """
        ).fetchall()

        if model_costs_rows:
            df_models = pd.DataFrame([dict(r) for r in model_costs_rows])
            df_models.columns = [
                "Model",
                "Total Cost ($)",
                "Input Tokens",
                "Output Tokens",
                "API Calls",
            ]
            st.dataframe(df_models, use_container_width=True, hide_index=True)
        else:
            st.caption("No per-model cost data available.")

        st.divider()

        # Charts from trends.py
        st.subheader("Quality per Dollar Over Time")
        st.caption(
            "Dual-axis chart: quality-per-dollar ratio (left axis) and "
            "cumulative token spend (right axis) across pipeline cycles."
        )
        st.plotly_chart(cost_efficiency_trend(conn), use_container_width=True, key="cost_efficiency_costs_tab")

# ---- Tab 6: Pipeline Runner -----------------------------------------------

with tab_runner:
    st.header("Pipeline Runner")
    st.caption(
        "Trigger the batch pipeline to generate, evaluate, and iterate on ad copy. "
        "Results appear in the other tabs after completion."
    )

    rcol1, rcol2 = st.columns([1, 3])
    with rcol1:
        batch_size = st.number_input(
            "Batch size (briefs)",
            min_value=1,
            max_value=50,
            value=3,
            step=1,
            key="runner_batch",
        )

    if st.button("Run Pipeline", type="primary", use_container_width=True):
        with st.status("Running pipeline...", expanded=True) as status:
            try:
                st.write("Initializing pipeline...")
                from src.pipeline.main import BatchPipeline

                pipeline = BatchPipeline(db_path=DB_PATH)

                st.write("Generating brief matrix...")
                progress = st.progress(0.0)
                briefs = pipeline.generate_brief_matrix()
                briefs = briefs[:batch_size]
                st.write(f"Processing {len(briefs)} briefs...")
                progress.progress(0.1)

                st.write("Running generation, evaluation, and iteration cycles...")
                progress.progress(0.2)

                result = pipeline.run(briefs)
                progress.progress(1.0)

                status.update(label="Pipeline complete!", state="complete")

                st.divider()
                st.subheader("Results")
                rcol1, rcol2, rcol3, rcol4 = st.columns(4)
                rcol1.metric("Processed", result.total_briefs)
                rcol2.metric("Passed", result.passed)
                rcol3.metric("Failed", result.failed)
                rcol4.metric(
                    "Avg Score",
                    f"{result.avg_score:.2f}" if result.avg_score else "N/A",
                )

                scol1, scol2, scol3 = st.columns(3)
                scol1.metric("Total Cycles", result.total_cycles)
                scol2.metric("Errors", result.errors)
                scol3.metric("Duration", f"{result.duration_seconds:.1f}s")

                st.success(
                    "Pipeline run complete. Switch to the other tabs "
                    "to explore the generated ads and quality metrics."
                )

            except Exception as exc:
                status.update(label="Pipeline failed", state="error")
                st.error(f"Pipeline error: {type(exc).__name__}: {exc}")

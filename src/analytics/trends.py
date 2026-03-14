"""Plotly chart generation for quality trends and cost analytics.

All functions return plotly.graph_objects.Figure instances and handle
empty data gracefully with an annotation instead of raising.
"""

from __future__ import annotations

import json
import sqlite3

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.db.queries import get_recent_calibration_runs
from src.evaluate.calibrate import ALPHA_THRESHOLD
from src.evaluate.rubrics import DIMENSION_WEIGHTS, DIMENSIONS, PASSING_THRESHOLD

# Nerdy brand cyan as primary, with complementary palette
PRIMARY_COLOR = "#17E2EA"
PASS_COLOR = "#17E2EA"
FAIL_COLOR = "#FF4B4B"
TEMPLATE = "plotly_white"

DIMENSION_COLORS = {
    "clarity": "#17E2EA",
    "learner_benefit": "#1B9E77",
    "cta_effectiveness": "#7570B3",
    "brand_voice": "#E7298A",
    "student_empathy": "#D95F02",
    "pedagogical_integrity": "#66A61E",
}


def _empty_figure(title: str) -> go.Figure:
    """Return a figure with a centered 'No data available' annotation."""
    fig = go.Figure()
    fig.update_layout(
        template=TEMPLATE,
        title=title,
        annotations=[
            {
                "text": "No data available",
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 18, "color": "gray"},
            }
        ],
    )
    return fig


def score_distribution(db_conn: sqlite3.Connection) -> go.Figure:
    """Histogram of weighted average scores across all evaluated ads.

    Vertical line at pass threshold. Bars colored by pass/fail.
    """
    db_conn.row_factory = sqlite3.Row
    rows = db_conn.execute(
        """
        SELECT e.ad_id,
               SUM(e.score * CASE e.dimension
                   WHEN 'clarity' THEN 0.20
                   WHEN 'learner_benefit' THEN 0.20
                   WHEN 'cta_effectiveness' THEN 0.16
                   WHEN 'brand_voice' THEN 0.12
                   WHEN 'student_empathy' THEN 0.12
                   WHEN 'pedagogical_integrity' THEN 0.20
                   ELSE 0 END) as weighted_avg
        FROM evaluations e
        WHERE e.eval_mode = 'final' OR e.eval_mode IS NULL
        GROUP BY e.ad_id
        """
    ).fetchall()

    if not rows:
        return _empty_figure("Score Distribution")

    scores = [r["weighted_avg"] for r in rows]

    fig = go.Figure()
    passing = [s for s in scores if s >= PASSING_THRESHOLD]
    failing = [s for s in scores if s < PASSING_THRESHOLD]

    if passing:
        fig.add_trace(go.Histogram(x=passing, name="Pass", marker_color=PASS_COLOR, opacity=0.8))
    if failing:
        fig.add_trace(go.Histogram(x=failing, name="Fail", marker_color=FAIL_COLOR, opacity=0.8))

    fig.add_vline(x=PASSING_THRESHOLD, line_dash="dash", line_color="gray", line_width=2)
    fig.add_annotation(
        x=PASSING_THRESHOLD,
        y=1,
        yref="paper",
        text=f"Threshold ({PASSING_THRESHOLD})",
        showarrow=False,
        yanchor="bottom",
    )

    fig.update_layout(
        template=TEMPLATE,
        title="Score Distribution",
        xaxis_title="Weighted Average Score",
        yaxis_title="Count",
        barmode="stack",
    )
    return fig


def convergence_curves(db_conn: sqlite3.Connection) -> go.Figure:
    """Line chart: avg weighted score per iteration cycle, one line per dimension."""
    db_conn.row_factory = sqlite3.Row
    rows = db_conn.execute(
        """
        SELECT qs.cycle_number, qs.avg_weighted_score, qs.dimension_averages
        FROM quality_snapshots qs
        ORDER BY qs.cycle_number
        """
    ).fetchall()

    if not rows:
        return _empty_figure("Convergence Curves")

    cycles = [r["cycle_number"] for r in rows]
    weighted_avgs = [r["avg_weighted_score"] for r in rows]

    fig = go.Figure()

    # Weighted average line
    fig.add_trace(
        go.Scatter(
            x=cycles,
            y=weighted_avgs,
            mode="lines+markers",
            name="Weighted Avg",
            line={"color": PRIMARY_COLOR, "width": 3},
        )
    )

    # Per-dimension lines
    for dim in DIMENSION_WEIGHTS:
        dim_values = []
        for r in rows:
            dim_avg_raw = r["dimension_averages"]
            if dim_avg_raw:
                dim_avgs = json.loads(dim_avg_raw) if isinstance(dim_avg_raw, str) else dim_avg_raw
                dim_values.append(dim_avgs.get(dim))
            else:
                dim_values.append(None)

        color = DIMENSION_COLORS.get(dim, "gray")
        fig.add_trace(
            go.Scatter(
                x=cycles,
                y=dim_values,
                mode="lines+markers",
                name=dim.replace("_", " ").title(),
                line={"color": color, "width": 1.5, "dash": "dot"},
            )
        )

    fig.update_layout(
        template=TEMPLATE,
        title="Convergence Curves",
        xaxis_title="Cycle Number",
        yaxis_title="Average Score",
    )
    return fig


def dimension_breakdown(db_conn: sqlite3.Connection) -> go.Figure:
    """Grouped bar chart of average score per dimension."""
    db_conn.row_factory = sqlite3.Row
    rows = db_conn.execute(
        """
        SELECT dimension, AVG(score) as avg_score
        FROM evaluations
        WHERE eval_mode = 'final' OR eval_mode IS NULL
        GROUP BY dimension
        ORDER BY dimension
        """
    ).fetchall()

    if not rows:
        return _empty_figure("Dimension Breakdown")

    dimensions = [r["dimension"] for r in rows]
    avg_scores = [r["avg_score"] for r in rows]
    colors = [DIMENSION_COLORS.get(d, PRIMARY_COLOR) for d in dimensions]

    fig = go.Figure(
        go.Bar(
            x=[d.replace("_", " ").title() for d in dimensions],
            y=avg_scores,
            marker_color=colors,
        )
    )
    fig.update_layout(
        template=TEMPLATE,
        title="Dimension Breakdown",
        xaxis_title="Dimension",
        yaxis_title="Average Score",
    )
    return fig


def before_after_comparison(db_conn: sqlite3.Connection, ad_id: str) -> go.Figure:
    """Side-by-side bar chart comparing dimension scores before and after iteration."""
    db_conn.row_factory = sqlite3.Row

    # Get iteration-mode scores (before)
    before_rows = db_conn.execute(
        "SELECT dimension, score FROM evaluations WHERE ad_id = ? AND eval_mode = 'iteration'",
        (ad_id,),
    ).fetchall()

    # Get final-mode scores (after)
    after_rows = db_conn.execute(
        "SELECT dimension, score FROM evaluations WHERE ad_id = ? AND eval_mode = 'final'",
        (ad_id,),
    ).fetchall()

    if not before_rows and not after_rows:
        return _empty_figure(f"Before/After — Ad {ad_id[:8]}")

    before_scores = {r["dimension"]: r["score"] for r in before_rows}
    after_scores = {r["dimension"]: r["score"] for r in after_rows}
    all_dims = sorted(set(before_scores) | set(after_scores))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Before (Iteration)",
            x=[d.replace("_", " ").title() for d in all_dims],
            y=[before_scores.get(d, 0) for d in all_dims],
            marker_color=FAIL_COLOR,
            opacity=0.7,
        )
    )
    fig.add_trace(
        go.Bar(
            name="After (Final)",
            x=[d.replace("_", " ").title() for d in all_dims],
            y=[after_scores.get(d, 0) for d in all_dims],
            marker_color=PASS_COLOR,
            opacity=0.7,
        )
    )

    fig.update_layout(
        template=TEMPLATE,
        title=f"Before/After — Ad {ad_id[:8]}",
        xaxis_title="Dimension",
        yaxis_title="Score",
        barmode="group",
    )
    return fig


def cost_efficiency_trend(db_conn: sqlite3.Connection) -> go.Figure:
    """Dual-axis line chart: quality_per_dollar (left) and token_spend_usd (right)."""
    db_conn.row_factory = sqlite3.Row
    rows = db_conn.execute(
        "SELECT cycle_number, quality_per_dollar, token_spend_usd "
        "FROM quality_snapshots ORDER BY cycle_number"
    ).fetchall()

    if not rows:
        return _empty_figure("Cost Efficiency Trend")

    cycles = [r["cycle_number"] for r in rows]
    qpd = [r["quality_per_dollar"] for r in rows]
    spend = [r["token_spend_usd"] for r in rows]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cycles,
            y=qpd,
            mode="lines+markers",
            name="Quality / Dollar",
            line={"color": PRIMARY_COLOR, "width": 2},
            yaxis="y",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cycles,
            y=spend,
            mode="lines+markers",
            name="Token Spend ($)",
            line={"color": "#666666", "width": 2, "dash": "dash"},
            yaxis="y2",
        )
    )

    fig.update_layout(
        template=TEMPLATE,
        title="Cost Efficiency Trend",
        xaxis_title="Cycle Number",
        yaxis={"title": "Quality per Dollar", "side": "left"},
        yaxis2={"title": "Token Spend (USD)", "side": "right", "overlaying": "y"},
    )
    return fig


def calibration_trend_chart(db_conn: sqlite3.Connection) -> go.Figure:
    """Dual-panel: Alpha over time (top), per-dimension MAE (bottom)."""
    runs = get_recent_calibration_runs(db_conn, limit=50)

    if not runs:
        return _empty_figure("Calibration Trends")

    # Reverse to chronological order (query returns DESC)
    runs = list(reversed(runs))
    timestamps = [r["timestamp"] for r in runs]

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Alpha Stability", "Per-Dimension MAE"),
        vertical_spacing=0.15,
    )

    # Top panel: Alpha over time
    alphas = [r["alpha_overall"] for r in runs]
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=alphas,
            mode="lines+markers",
            name="Alpha",
            line={"color": PRIMARY_COLOR, "width": 2},
        ),
        row=1,
        col=1,
    )
    fig.add_hline(
        y=ALPHA_THRESHOLD,
        line_dash="dash",
        line_color=FAIL_COLOR,
        line_width=1,
        row=1,
        col=1,
    )

    # Bottom panel: per-dimension MAE
    for dim in DIMENSIONS:
        mae_col = f"mae_{dim}"
        values = [r[mae_col] for r in runs]
        color = DIMENSION_COLORS.get(dim, "gray")
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=values,
                mode="lines+markers",
                name=dim.replace("_", " ").title(),
                line={"color": color, "width": 1.5},
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        template=TEMPLATE,
        title="Calibration Trends",
        height=600,
    )
    fig.update_yaxes(title_text="Alpha", row=1, col=1)
    fig.update_yaxes(title_text="MAE", row=2, col=1)

    return fig

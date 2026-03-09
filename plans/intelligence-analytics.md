# Phase 3 Intelligence, Analytics & Export

Story: story-585
Agent: architect

## Context

Phase 3 adds the intelligence and analytics layers on top of the working generation/evaluation/iteration pipeline (stories 581-584). Three distinct capabilities: (1) competitive intelligence — curated competitor ads feed real hook/CTA/emotional patterns into the generator's prompt context, (2) cost and quality analytics — LiteLLM-based performance-per-token tracking plus Plotly trend visualization, (3) export — ad libraries and decision logs to CSV/JSON for review and submission.

The competitor dataset is manually curated mock data structured to match what you'd find browsing the public Meta Ad Library website for Princeton Review, Kaplan, Khan Academy, and Chegg. We cannot browse Meta Ad Library from code — the curated.json contains realistic representative ads.

Performance-per-token is the north star metric: `quality_score / (tokens * price_per_token)`. This tells us which model/config produces the best quality per dollar of API spend.

## What changes

| File | Change |
|---|---|
| `data/competitor_ads/curated.json` | **Create.** 20-30 structured competitor ads with fields: brand, primary_text, headline, cta_button, hook_type, emotional_angle. Covers Princeton Review, Kaplan, Khan Academy, Chegg across SAT test prep campaigns. |
| `src/intel/__init__.py` | **Create.** Empty package init. |
| `src/intel/analyzer.py` | **Create.** Load curated.json, seed `competitor_ads` DB table, extract recurring hook/CTA/emotional patterns across brands, return structured `CompetitorPatterns` for injection into generation prompts. |
| `src/analytics/__init__.py` | **Create.** Empty package init. |
| `src/analytics/cost.py` | **Create.** LiteLLM-based token counting and cost calculation per API call. `record_api_cost()` writes to ads/evaluations tables. `compute_quality_snapshot()` calculates performance-per-token and writes to `quality_snapshots` table. |
| `src/analytics/trends.py` | **Create.** Plotly/Pandas chart generation: score distribution histograms, dimension-level convergence curves, before/after iteration comparisons. Returns Plotly figure objects for dashboard embedding. |
| `src/output/__init__.py` | **Create.** Empty package init. |
| `src/output/exporter.py` | **Create.** Export ad library (ads + scores) to CSV/JSON. Export decision log to JSON. Export summary statistics (pass rate, avg scores, cost totals, performance-per-token). |
| `src/db/queries.py` | **Modify.** Add query functions: `insert_competitor_ad()`, `get_competitor_ads()`, `insert_quality_snapshot()`, `get_quality_snapshots()`, `get_ads_with_scores()` (joins ads + evaluations for export), `get_decision_log()` (all decisions, ordered). |
| `src/generate/engine.py` | **Modify.** Accept optional `CompetitorPatterns` in generation context. Inject top hook patterns and CTA patterns into the system prompt as competitive context. |

## Read-only context

- `presearch/autonomous-ad-engine.md` — briefing with competitive intelligence approach, cost tracking model pricing, analytics requirements, data model schemas
- `src/models/evaluation.py` — `EvaluationResult` and `DimensionScore` models (needed by trends.py for type signatures)
- `src/models/ad.py` — `AdCopy` model (needed by exporter.py and cost.py)
- `src/models/decision.py` — `DecisionEntry` model (needed by exporter.py)
- `src/models/iteration.py` — `IterationRecord` model (needed by trends.py for convergence curves)
- `src/db/schema.sql` — table schemas for competitor_ads, quality_snapshots, ads, evaluations, iterations, decisions (already created in story-581)
- `src/db/init_db.py` — database connection helper
- All pipeline/iteration code from stories 582-584

## Tasks

1. **Create `data/competitor_ads/curated.json` with 20-30 competitor ads.** Structure each ad as: `{ "brand": str, "primary_text": str, "headline": str, "cta_button": str, "hook_type": str, "emotional_angle": str }`. Distribute across 4 brands (Princeton Review ~7, Kaplan ~7, Khan Academy ~5, Chegg ~5). Hook types should include: fear_of_missing_out, social_proof, question_hook, statistic_hook, urgency, aspiration, pain_point. Emotional angles should include: parental_anxiety, student_confidence, achievement, affordability, convenience, competitive_edge. CTA buttons should use real Meta CTA options: Learn More, Sign Up, Get Offer, Book Now. The ad copy should be realistic SAT test prep marketing — scores, timelines, outcomes, differentiators per brand.

2. **Implement `src/intel/analyzer.py`.** Three public functions:
   - `load_curated_ads(json_path: Path) -> list[dict]` — reads curated.json, validates structure.
   - `seed_competitor_ads(db_conn: Connection, ads: list[dict]) -> int` — upserts into `competitor_ads` table, returns count inserted. Idempotent (skip duplicates by primary_text hash).
   - `extract_patterns(ads: list[dict]) -> CompetitorPatterns` — counts hook_type and emotional_angle frequencies across all ads. Identifies top 3 hooks, top 3 emotional angles, all unique CTA buttons. Returns a `CompetitorPatterns` dataclass with `top_hooks: list[tuple[str, int]]`, `top_angles: list[tuple[str, int]]`, `cta_buttons: list[str]`, `sample_headlines: list[str]` (one strong headline per brand).
   - Define `CompetitorPatterns` as a `dataclass` (not Pydantic — it's internal, not a persistence boundary).
   - Log a decision when patterns are extracted: component="intel", action="extracted_patterns", rationale with pattern counts.

3. **Implement `src/analytics/cost.py`.** Functions:
   - `calculate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float` — uses LiteLLM's `litellm.completion_cost()` or manual pricing table as fallback. Pricing fallback: Gemini 2.5 Flash ($0.30/$2.50 per 1M), Gemini 2.5 Pro ($1.25/$10.00 per 1M), Claude Sonnet 4.6 ($3.00/$15.00 per 1M).
   - `record_api_cost(db_conn: Connection, table: str, row_id: str, model_id: str, input_tokens: int, output_tokens: int) -> float` — calculates cost, updates the token/cost columns on the specified table row, returns cost_usd.
   - `compute_quality_snapshot(db_conn: Connection, cycle_number: int) -> dict` — queries current cycle's ads and evaluations, calculates: avg_weighted_score, dimension_averages (JSON), ads_above_threshold, total_ads, token_spend_usd (sum of all costs this cycle), quality_per_dollar (avg_weighted_score / token_spend_usd). Inserts into `quality_snapshots` table. Returns the snapshot dict.
   - `get_performance_per_token(db_conn: Connection) -> list[dict]` — returns all snapshots ordered by cycle for trend display.
   - Handle edge case: zero token spend (first run, or free tier shows $0) — set quality_per_dollar to None rather than dividing by zero.

4. **Implement `src/analytics/trends.py`.** Functions returning `plotly.graph_objects.Figure`:
   - `score_distribution(db_conn: Connection) -> Figure` — histogram of weighted_average scores across all evaluated ads. Vertical line at pass threshold (7.0). Color by pass/fail.
   - `convergence_curves(db_conn: Connection) -> Figure` — line chart showing avg weighted score per iteration cycle. One line per dimension plus the weighted average. X-axis: cycle number. Shows whether quality improves over iterations.
   - `dimension_breakdown(db_conn: Connection) -> Figure` — grouped bar chart of average score per dimension, colored by dimension. Shows which dimensions are strong/weak.
   - `before_after_comparison(db_conn: Connection, ad_id: str) -> Figure` — side-by-side bar chart comparing dimension scores before and after iteration for a specific ad.
   - `cost_efficiency_trend(db_conn: Connection) -> Figure` — dual-axis line chart from quality_snapshots: quality_per_dollar on left axis, token_spend_usd on right axis, by cycle.
   - All functions handle empty data gracefully — return a figure with an annotation "No data available" rather than raising.
   - Use consistent Plotly theme: `plotly_white` template, Varsity Tutors orange (#FF6B35) as primary color.

5. **Implement `src/output/exporter.py`.** Functions:
   - `export_ad_library(db_conn: Connection, format: Literal["csv", "json"], output_path: Path) -> Path` — joins ads with their final evaluation scores (all dimensions + weighted average). CSV includes one row per ad with flattened dimension scores. JSON preserves nested structure. Returns the written file path.
   - `export_decision_log(db_conn: Connection, output_path: Path) -> Path` — all decisions ordered by timestamp as JSON array. Each entry includes timestamp, component, action, rationale, context.
   - `export_summary_stats(db_conn: Connection, output_path: Path) -> Path` — JSON with: total_ads, pass_rate, avg_weighted_score, dimension_averages, total_token_spend, avg_quality_per_dollar, iteration_stats (avg cycles per ad, improvement rate). Returns the written file path.
   - All exporters create parent directories if they don't exist.

6. **Add query functions to `src/db/queries.py`.** New functions (append to existing file):
   - `insert_competitor_ad(conn, ad: dict) -> str` — INSERT OR IGNORE into competitor_ads, returns id.
   - `get_competitor_ads(conn, brand: str | None = None) -> list[dict]` — SELECT from competitor_ads, optional brand filter.
   - `insert_quality_snapshot(conn, snapshot: dict) -> str` — INSERT into quality_snapshots.
   - `get_quality_snapshots(conn) -> list[dict]` — SELECT all snapshots ordered by cycle_number.
   - `get_ads_with_scores(conn) -> list[dict]` — JOIN ads with evaluations (final mode only), returns denormalized rows for export.
   - `get_all_decisions(conn) -> list[dict]` — SELECT all decisions ordered by timestamp.
   - `get_dimension_averages(conn, cycle_number: int | None = None) -> dict[str, float]` — AVG score grouped by dimension, optional cycle filter.
   - All queries use parameterized SQL. All return dicts (via `conn.row_factory = sqlite3.Row`).

7. **Update `src/generate/engine.py` to consume competitor patterns.** Modify the generation function signature to accept an optional `CompetitorPatterns` parameter. When provided, append a "Competitive Context" section to the system prompt that includes: top hook patterns with examples, common CTA buttons in the category, emotional angles that resonate. The generator should use these as inspiration, not copy them. Log a decision when competitor context is injected.

## Acceptance criteria

- `data/competitor_ads/curated.json` contains 20-30 ads across 4 brands, each with all 6 required fields, valid JSON
- `analyzer.py` loads curated.json, seeds the DB table (idempotent), and extracts patterns with correct frequency counts
- `cost.py` calculates correct costs for all three model IDs, handles zero-spend edge case, writes quality_snapshots
- `trends.py` returns valid Plotly figures for all 5 chart types, handles empty data without raising
- `exporter.py` writes valid CSV and JSON files, creates parent directories, includes all required fields
- `queries.py` has all 7 new query functions, all using parameterized SQL
- `engine.py` accepts CompetitorPatterns and injects competitive context into generation prompts when provided
- All new code passes `ruff check` with zero violations
- `uv run pytest` passes — new tests cover pattern extraction, cost calculation, export format validation

## Verification

- Run `uv run python -c "import json; ads = json.load(open('data/competitor_ads/curated.json')); assert 20 <= len(ads) <= 30; brands = set(a['brand'] for a in ads); assert len(brands) == 4; print(f'{len(ads)} ads across {brands}')"` — confirms curated.json structure
- Run `uv run python -c "from src.intel.analyzer import load_curated_ads, extract_patterns; ads = load_curated_ads('data/competitor_ads/curated.json'); p = extract_patterns(ads); print(f'Top hooks: {p.top_hooks}'); print(f'Top angles: {p.top_angles}')"` — confirms pattern extraction
- Run `uv run python -c "from src.analytics.cost import calculate_cost; c = calculate_cost('gemini-2.5-flash', 1000, 500); print(f'Flash cost: ${c:.6f}'); assert c > 0"` — confirms cost calculation
- Run `uv run python -c "from src.analytics.trends import score_distribution; from src.db.init_db import init_db; conn = init_db(':memory:'); fig = score_distribution(conn); print(fig.layout.title.text)"` — confirms empty-data handling
- Run `uv run ruff check src/intel/ src/analytics/ src/output/` — zero violations
- Run `uv run pytest tests/ -v -k 'intel or cost or trends or export'` — all Phase 3 tests pass
- Manually inspect exported CSV/JSON files for correct structure and completeness

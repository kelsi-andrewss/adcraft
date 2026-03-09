# Phase 3 Streamlit Dashboard

Story: story-586
Agent: quick-fixer

## Context

The Autonomous Ad Engine generates, evaluates, and iterates on FB/IG ad copy. All data lives in SQLite (ads, evaluations, iterations, decisions, quality_snapshots). Analytics modules (`src/analytics/trends.py`, `src/analytics/cost.py`) already produce Plotly figures. The pipeline entry point (`src/pipeline/main.py`) runs the batch process. This story builds the presentation layer that unifies everything into a single Streamlit app — the thing that gets screen-recorded for the demo video.

The dashboard writes zero data. It reads from the database via `src/db/queries.py` and embeds existing Plotly charts. The Pipeline Runner tab is the one exception: it invokes the existing pipeline entry point and displays progress, but the pipeline itself handles all writes.

## What changes

| File | Change |
|---|---|
| `src/dashboard/__init__.py` | Empty package init |
| `src/dashboard/app.py` | Multi-tab Streamlit app with six tabs: Ad Library, Quality Trends, Iteration Inspector, Decision Log, Cost Tracker, Pipeline Runner. Sidebar with database stats and refresh button. Professional styling for demo recording. |

## Read-only context

- `presearch/autonomous-ad-engine.md` — briefing with dashboard spec, data model, project structure
- `src/db/queries.py` — database query helpers (all reads go through here)
- `src/analytics/trends.py` — Plotly quality trend charts (score distributions, dimension improvement, convergence curves)
- `src/analytics/cost.py` — cost tracking and performance-per-dollar Plotly charts
- `src/pipeline/main.py` — pipeline entry point for the Runner tab
- `src/models/` — all Pydantic models for type-safe data display (AdCopy, EvaluationResult, IterationRecord, DecisionEntry)
- `src/db/init_db.py` — database initialization and connection

## Tasks

1. **Create `src/dashboard/__init__.py`.** Empty file, package init only.

2. **Create `src/dashboard/app.py` with page config and sidebar.** Set `st.set_page_config(page_title="Ad Engine Dashboard", layout="wide")`. Add a sidebar with:
   - Title and brief description
   - Database connection status (verify the SQLite file exists)
   - Summary stats: total ads, pass rate, total iterations, total cost
   - A refresh button (`st.button("Refresh")`) that calls `st.rerun()`
   - All sidebar stats queried via `src/db/queries.py` functions

3. **Tab 1 — Ad Library.** Searchable, filterable table of generated ads.
   - Use `st.tabs()` to create the six-tab layout. First tab: "Ad Library"
   - Filters in columns at the top: pass/fail toggle (`st.selectbox`), search box (`st.text_input`) for keyword search across primary_text/headline
   - Display results in `st.dataframe()` with columns: id (truncated), headline, primary_text (truncated), cta_button, weighted_score, passed (checkmark/x), created_at
   - Click-to-expand: use `st.expander()` per row for full ad copy details + all dimension scores with rationales
   - Query data via `queries.get_all_ads()` or equivalent, join with evaluation results

4. **Tab 2 — Quality Trends.** Embed existing Plotly charts from `src/analytics/trends.py`.
   - Import chart-building functions from `trends.py` (these return `plotly.graph_objects.Figure`)
   - Render each with `st.plotly_chart(fig, use_container_width=True)`
   - Expected charts: score distribution histogram, dimension-level improvement over cycles, before/after iteration comparison, convergence curve
   - Add a brief description above each chart explaining what it shows

5. **Tab 3 — Iteration Inspector.** Visual journey of a specific ad from original to final version.
   - `st.selectbox` to pick an ad by headline (query ads that have iteration history)
   - Once selected, query the iteration chain: source_ad -> iteration 1 -> iteration 2 -> ... -> final
   - Display as a vertical timeline using `st.container()` blocks with `st.columns()`:
     - Each step shows: cycle number, action_type (component_fix or full_regen), weak_dimension targeted, the ad copy at that stage, dimension scores, delta from previous
   - Use `st.metric()` for score deltas (shows green/red arrows natively)
   - Color-code pass/fail status per step

6. **Tab 4 — Decision Log.** Searchable viewer for the `decisions` table.
   - Search box (`st.text_input`) filtering across component, action, rationale
   - Filter by component (`st.multiselect` with distinct component values)
   - Display as `st.dataframe()` with columns: timestamp, component, action, rationale (truncated), agent_id
   - `st.expander()` per row for full rationale and context fields
   - Sort by timestamp descending (most recent first)

7. **Tab 5 — Cost Tracker.** Performance-per-dollar charts and cost breakdown.
   - Import chart-building functions from `src/analytics/cost.py` (these return Plotly figures)
   - Render with `st.plotly_chart(fig, use_container_width=True)`
   - Expected charts: quality-per-dollar over time, cost breakdown by model (Flash vs Pro vs Claude), cumulative spend curve
   - Add summary metrics row at the top using `st.metric()`: total spend, average cost per ad, best quality-per-dollar ratio

8. **Tab 6 — Pipeline Runner.** UI to trigger the batch pipeline and show progress.
   - `st.number_input` for batch size (default from pipeline config)
   - `st.button("Run Pipeline")` to trigger
   - On click, run the pipeline in a `st.status("Running pipeline...", expanded=True)` block
   - Inside the status block, use `st.write()` to log each stage as it completes (brief generation, ad generation, evaluation, iteration, final scoring)
   - Use `st.progress()` bar updated as stages complete
   - Import and call the pipeline's main function, passing a callback/logger that writes to the Streamlit status block
   - On completion, show summary stats and prompt user to switch to other tabs to see results
   - Handle errors gracefully: display the exception in `st.error()` without crashing the app

9. **Professional styling.** This is the demo video surface.
   - Use `st.markdown()` with custom CSS via `unsafe_allow_html=True` for minor tweaks only (card spacing, metric alignment)
   - Consistent color scheme: green for pass, red for fail, blue for neutral
   - All dataframes should have reasonable column widths and truncated long text
   - Charts should use `use_container_width=True` for responsive layout
   - No raw database IDs shown to user — truncate or hide UUIDs

## Acceptance criteria

- `uv run streamlit run src/dashboard/app.py` launches without errors on localhost:8501
- All six tabs render and are navigable
- Ad Library tab displays ads with pass/fail filtering and keyword search
- Quality Trends tab shows Plotly charts from `trends.py` with no import errors
- Iteration Inspector shows the full revision journey for a selected ad with score deltas
- Decision Log is searchable by component and keyword with full rationale expandable
- Cost Tracker shows performance-per-dollar charts and summary metrics
- Pipeline Runner triggers the pipeline and shows real-time progress in a status block
- Sidebar shows database stats (total ads, pass rate, cost) and refresh works
- No business logic in `app.py` — all data access goes through `queries.py`, all charts come from `trends.py`/`cost.py`
- App handles empty database gracefully (shows "No data yet — run the pipeline" messages, not crashes)
- Layout looks professional at 1920x1080 (demo recording resolution)

## Verification

- Run `uv run ruff check src/dashboard/` and confirm zero violations
- Run `uv run streamlit run src/dashboard/app.py` and confirm the app loads
- Verify each tab renders without errors (even with empty database — should show empty states, not tracebacks)
- With a populated database, verify:
  - Ad Library shows ads and filters work
  - Quality Trends shows charts
  - Iteration Inspector shows ad journey with deltas
  - Decision Log search returns matching entries
  - Cost Tracker shows spend charts
  - Pipeline Runner button triggers the pipeline and progress updates appear
- Screenshot at 1920x1080 for demo video quality check

# Phase 1 Data Layer

Story: story-582
Agent: architect

## Context

Implements the data layer that sits on top of story-581's Pydantic models and DB schema. Three concerns: seed ad briefs that drive generation, parameterized SQL helpers for all 6 tables, and a standalone decision logger that every future pipeline component imports. This is plumbing — nothing here calls an LLM.

## What changes

| File | Change |
|---|---|
| `src/briefs/seed_briefs.py` | Create seed `AdBrief` objects: parent-focused (SAT prep ROI, investment framing, score guarantees) and student-focused (score improvement, confidence, college readiness). ~6-8 briefs covering the audience x tone matrix. |
| `src/db/queries.py` | Parameterized raw SQL functions for all 6 tables: `insert_ad`/`get_ad`/`list_ads`, `insert_evaluation`/`get_evaluations_for_ad`, `insert_iteration`/`get_iterations_for_ad`, `insert_decision`/`list_decisions`, `insert_competitor_ad`/`list_competitor_ads`, `insert_quality_snapshot`/`list_quality_snapshots`. Each function takes a `sqlite3.Connection` as first arg. |
| `src/decisions/logger.py` | `log_decision(component, action, rationale, context, *, conn=None, agent_id="system")` — validates via `DecisionEntry` model, persists to `decisions` table. If `conn` is None, opens a connection using `DATABASE_PATH` env var (default `data/ads.db`). Standalone import: `from src.decisions.logger import log_decision`. |
| `src/briefs/__init__.py` | Empty init to make `src/briefs` a package. |
| `src/decisions/__init__.py` | Empty init to make `src/decisions` a package. |
| `tests/test_decisions.py` | Unit tests for decision logger: validates DecisionEntry fields, persists to DB, rejects missing required fields, generates UUID and timestamp automatically, works with both provided and default connections. |
| `tests/test_queries.py` | Unit tests for query helpers: round-trip insert/select for each table, parameterization prevents injection, `list_*` functions return correct types, foreign key references work. |

## Read-only context

- `presearch/autonomous-ad-engine.md` — briefing with model schemas, DB schema, patterns
- `src/models/brief.py` — AdBrief Pydantic model (created by story-581)
- `src/models/decision.py` — DecisionEntry model (created by story-581)
- `src/db/schema.sql` — table definitions (created by story-581)
- `src/db/init_db.py` — DB initialization (created by story-581)

## Tasks

1. **Create package inits.** Add empty `__init__.py` to `src/briefs/` and `src/decisions/`.

2. **Implement seed briefs** (`src/briefs/seed_briefs.py`). Define a `SEED_BRIEFS: list[AdBrief]` constant with 6-8 briefs across two audience segments:
   - Parent-focused (3-4 briefs): SAT prep ROI / investment framing, score guarantee / risk reduction, college admissions advantage, "don't let your kid fall behind" urgency. Tones: authoritative, reassuring, urgent.
   - Student-focused (3-4 briefs): score improvement / personal growth, confidence building, college readiness / independence, peer comparison. Tones: motivational, casual, aspirational.
   - Each brief populates all AdBrief fields: `audience_segment`, `product_offer`, `campaign_goal`, `tone`, `competitive_context`. Leave `competitive_context` as a generic placeholder (e.g., "Princeton Review, Kaplan, Khan Academy offer SAT prep") — story-582 doesn't depend on the competitive intelligence feature.
   - Include a `get_seed_briefs() -> list[AdBrief]` convenience function that returns a copy of `SEED_BRIEFS`.
   - Reference: briefing §Features item 1, §Shared Interfaces `AdBrief` fields.

3. **Implement query helpers** (`src/db/queries.py`). Raw SQL, parameterized queries, no ORM (per briefing §Patterns).
   - Every function takes `conn: sqlite3.Connection` as its first parameter. No module-level connection state.
   - Insert functions accept a Pydantic model instance and return the generated `id`. Generate UUIDs and timestamps inside the insert function, not in the caller.
   - Select functions return Pydantic model instances (or lists thereof), not raw tuples. Use `row_factory = sqlite3.Row` for dict-like access, then `Model(**dict(row))`.
   - Cover all 6 tables from `schema.sql`: ads, evaluations, iterations, decisions, competitor_ads, quality_snapshots.
   - Reference: briefing §Data Model for column names and types.

4. **Implement decision logger** (`src/decisions/logger.py`).
   - Signature: `log_decision(component: str, action: str, rationale: str, context: dict | None = None, *, conn: sqlite3.Connection | None = None, agent_id: str = "system") -> str` (returns the decision ID).
   - Validates by constructing a `DecisionEntry` (from `src/models/decision.py`). Let Pydantic raise on bad input.
   - If `conn` is None, open a connection using `get_db_path()` from `src/db/init_db.py` (or `DATABASE_PATH` env var with `data/ads.db` default). Close it after the write. If `conn` is provided, leave it open (caller manages lifecycle).
   - Serialize `context` as JSON string for the TEXT column.
   - Call `insert_decision` from `queries.py` — don't duplicate SQL.
   - This function must be importable standalone: `from src.decisions.logger import log_decision`. No circular imports, no heavy initialization on import.
   - Reference: briefing §Patterns "Decision logging" and §Features item 5.

5. **Write decision logger tests** (`tests/test_decisions.py`).
   - Use a fresh in-memory SQLite DB per test (fixture that runs `schema.sql` then yields the connection).
   - Test cases: basic round-trip (log + query back), required fields rejected when missing, context dict serialized to JSON, default agent_id is "system", custom agent_id preserved, returned ID matches persisted row.
   - Do NOT test the "conn=None opens default DB" path in unit tests — that touches the filesystem and belongs in integration tests.

6. **Write query helper tests** (`tests/test_queries.py`).
   - Same in-memory DB fixture as decision tests (extract to `tests/conftest.py`).
   - Test round-trip insert/select for at least: ads, evaluations, decisions. The other 3 tables can have minimal smoke tests.
   - Test that list functions return empty lists (not None) when no rows exist.
   - Test that insert functions generate unique IDs across calls.

## Acceptance criteria

- `from src.briefs.seed_briefs import get_seed_briefs` returns 6+ `AdBrief` instances covering both parent and student segments.
- `from src.decisions.logger import log_decision` works with only `component`, `action`, `rationale` args and a provided `conn`.
- Every query helper in `queries.py` has a corresponding test that does insert + select round-trip.
- `pytest tests/test_decisions.py tests/test_queries.py` passes with zero failures.
- No module-level DB connections or singletons in any file.
- No ORM usage anywhere — all SQL is raw and parameterized.

## Verification

- `pytest tests/test_decisions.py tests/test_queries.py -v` — all tests green.
- `ruff check src/briefs/ src/db/queries.py src/decisions/ tests/test_decisions.py tests/test_queries.py` — no lint errors.
- `python -c "from src.briefs.seed_briefs import get_seed_briefs; briefs = get_seed_briefs(); assert len(briefs) >= 6; print(f'{len(briefs)} seed briefs OK')"` — confirms briefs load.
- `python -c "from src.decisions.logger import log_decision; print('import OK')"` — confirms standalone import without side effects.

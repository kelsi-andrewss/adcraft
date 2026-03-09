# Bootstrap

Story: story-581
Agent: architect

## Context

Greenfield scaffold for the Autonomous Ad Engine — a system that generates, evaluates, and iterates on Facebook/Instagram ad copy for Varsity Tutors SAT test prep. This story lays the foundation: project init, dependency management, Pydantic data models, SQLite schema, CI, and project documentation. Everything downstream (evaluator, generator, iteration loops) depends on these models and the database being correct.

## What changes

| File | Change |
|---|---|
| `pyproject.toml` | Initialize project metadata, runtime deps (google-genai, anthropic, pydantic, litellm, plotly, pandas, python-dotenv, streamlit), dev deps (pytest, pytest-asyncio, ruff), tool config for ruff and pytest-asyncio |
| `.env.example` | Template with GEMINI_API_KEY, ANTHROPIC_API_KEY, DATABASE_PATH, LOG_LEVEL |
| `src/__init__.py` | Empty package init |
| `src/models/__init__.py` | Re-export all model classes for convenient imports |
| `src/models/brief.py` | `AdBrief` Pydantic model: audience_segment, product_offer, campaign_goal, tone, competitive_context |
| `src/models/ad.py` | `AdCopy` Pydantic model: id, primary_text, headline, description, cta_button, brief_id, model_id, generation_config, token_count |
| `src/models/evaluation.py` | `DimensionScore` and `EvaluationResult` Pydantic models with typed fields for dimension scoring, weighted averages, hard gate failures, evaluator metadata |
| `src/models/decision.py` | `DecisionEntry` Pydantic model: timestamp, component, action, rationale, context, agent_id |
| `src/models/iteration.py` | `IterationRecord` Pydantic model: source_ad_id, target_ad_id, cycle_number, action_type (Literal), weak_dimension, delta_scores, token_cost |
| `src/db/__init__.py` | Empty package init |
| `src/db/schema.sql` | CREATE TABLE statements for ads, evaluations, iterations, decisions, competitor_ads, quality_snapshots |
| `src/db/init_db.py` | `init_db(db_path)` function that reads schema.sql, executes it against a SQLite connection, handles idempotent creation |
| `.github/workflows/ci.yml` | GitHub Actions workflow: lint job (ruff check + ruff format --check), test job (pytest, ignore integration tests), triggers on push |
| `CLAUDE.md` | Build/test commands (uv sync, uv run pytest, uv run ruff check), architectural patterns (Procedural State Machine, Pydantic v2, Raw SQL, Cross-Model Evaluation), project structure overview |
| `README.md` | Project overview, setup instructions (uv sync, cp .env.example .env), run commands, architecture summary |
| `tests/__init__.py` | Empty package init |
| `tests/test_models.py` | Smoke tests: each Pydantic model round-trips through model_validate, schema.sql executes without error, init_db creates all expected tables |

## Read-only context

These files inform the implementation but should not be modified:
- `presearch/autonomous-ad-engine.md` — full technical briefing with Pydantic model schemas, SQLite table schemas, coding patterns, dependency list, environment variables

## Tasks

1. **Initialize project with uv.** Run `uv init` in the project root. Add runtime dependencies: google-genai, anthropic, `pydantic>=2.9`, litellm, plotly, pandas, python-dotenv, streamlit. Add dev dependencies: pytest, pytest-asyncio, ruff. (see briefing ## Dependencies for exact package names and version constraints)

2. **Configure pyproject.toml tooling.** Add `[tool.ruff]` section with target Python version and line length. Add `[tool.ruff.lint]` with select rules (E, F, W, I). Add `[tool.pytest.ini_options]` with `asyncio_mode = "auto"` and testpaths. (see briefing ## Patterns > Config and ## Project Structure for conventions)

3. **Create Pydantic models in src/models/.** Implement each model exactly as specified in the briefing's Shared Interfaces section:
   - `brief.py`: `AdBrief` — audience_segment, product_offer, campaign_goal, tone, competitive_context (see briefing ## Shared Interfaces > src/models/brief.py for field definitions)
   - `ad.py`: `AdCopy` — id, primary_text, headline, description, cta_button, brief_id, model_id, generation_config, token_count (see briefing ## Shared Interfaces > src/models/ad.py for field definitions)
   - `evaluation.py`: `DimensionScore` + `EvaluationResult` — dimension, score, rationale, confidence; ad_id, scores list, weighted_average, passed_threshold, hard_gate_failures, evaluator_model, token_count (see briefing ## Shared Interfaces > src/models/evaluation.py for field definitions)
   - `decision.py`: `DecisionEntry` — timestamp, component, action, rationale, context, agent_id (see briefing ## Shared Interfaces > src/models/decision.py for field definitions)
   - `iteration.py`: `IterationRecord` — source_ad_id, target_ad_id, cycle_number, action_type (component_fix|full_regen), weak_dimension, delta_scores, token_cost (see briefing ## Shared Interfaces > src/models/iteration.py for field definitions)
   - `__init__.py`: re-export all model classes

4. **Create SQLite schema.** Write `src/db/schema.sql` with CREATE TABLE IF NOT EXISTS for all six tables: ads, evaluations, iterations, decisions, competitor_ads, quality_snapshots. Match column names, types, and constraints exactly. (see briefing ## Data Model for complete table definitions)

5. **Implement database initialization.** Write `src/db/init_db.py` with an `init_db(db_path: str | Path)` function that: reads schema.sql relative to its own file location, opens a SQLite connection with WAL mode, executes the schema via `executescript()`, returns the connection. Include `ensure_data_dir()` helper that creates the parent directory of the database path. (see briefing ## Patterns > State management for SQLite conventions)

6. **Create .env.example.** Include GEMINI_API_KEY, ANTHROPIC_API_KEY, DATABASE_PATH (default data/ads.db), LOG_LEVEL (default INFO). Comment each variable with its purpose. (see briefing ## Environment for required variables)

7. **Set up GitHub Actions CI.** Create `.github/workflows/ci.yml` with two jobs: `lint` (runs ruff check and ruff format --check) and `test` (runs pytest ignoring integration tests). Both jobs use `uv sync` for dependency installation. Trigger on push to any branch. (see briefing ## Deployment > CI for workflow specification)

8. **Create CLAUDE.md.** Document: build command (uv sync), test command (uv run pytest), lint command (uv run ruff check), format command (uv run ruff format), run commands (pipeline + dashboard). Document architectural patterns: Procedural State Machine (not LangGraph), Pydantic v2 at all boundaries, raw SQL with parameterized queries, cross-model evaluation, decision logging at every branch point, snake_case everywhere except PascalCase for Pydantic models. (see briefing ## Patterns and ## Architecture for all conventions)

9. **Create README.md.** Project title, one-paragraph description, setup instructions (uv sync, cp .env.example .env, fill API keys), run commands, architecture overview, project structure tree.

10. **Write smoke tests.** Create `tests/test_models.py` with tests that: construct each Pydantic model with valid data and verify serialization round-trip, construct with invalid data and verify ValidationError, verify schema.sql executes on an in-memory SQLite database and creates all six tables, verify init_db creates tables and returns a usable connection.

## Acceptance criteria

- `uv sync` installs all dependencies without errors
- `uv run ruff check src/ tests/` passes clean
- `uv run pytest tests/test_models.py` passes — all models validate, schema creates tables
- All six Pydantic models importable from `src.models`
- `src/db/init_db.py` creates a SQLite database with all six tables when run
- `.env.example` contains all four environment variables
- `.github/workflows/ci.yml` is valid GitHub Actions YAML with lint and test jobs
- `CLAUDE.md` contains build, test, lint, and run commands plus architectural patterns
- No runtime imports fail (all packages in pyproject.toml)

## Verification

- Run `uv sync` and confirm zero errors
- Run `uv run ruff check src/ tests/` and confirm zero violations
- Run `uv run ruff format --check src/ tests/` and confirm zero reformats needed
- Run `uv run pytest tests/test_models.py -v` and confirm all tests pass
- Run `uv run python -c "from src.models import AdBrief, AdCopy, DimensionScore, EvaluationResult, DecisionEntry, IterationRecord; print('All models imported')"` and confirm output
- Run `uv run python -c "from src.db.init_db import init_db; conn = init_db(':memory:'); tables = [r[0] for r in conn.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall()]; assert len(tables) == 6; print(tables)"` and confirm six tables
- Push to GitHub and confirm CI workflow runs both lint and test jobs green

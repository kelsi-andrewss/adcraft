# AdCraft — Autonomous Ad Engine

## Commands

- **Install deps**: `uv sync`
- **Run tests**: `uv run pytest`
- **Run single test**: `uv run pytest tests/test_models.py -v`
- **Lint**: `uv run ruff check src/ tests/`
- **Format**: `uv run ruff format src/ tests/`
- **Format check**: `uv run ruff format --check src/ tests/`
- **Run pipeline**: `uv run python -m src.pipeline.main`
- **Run dashboard**: `uv run streamlit run src/dashboard/app.py`

## Architecture

**Procedural State Machine** — not LangGraph. Each pipeline stage is a function that takes state in and returns state out. A while loop with match/case handles iteration. No framework overhead for a linear pipeline with one feedback loop.

**Cross-Model Evaluation** — Generation uses Gemini 2.5 Flash (cheap, fast). Evaluation uses Gemini 2.5 Pro (better reasoning). Avoids self-bias where a model overrates its own output. Claude Sonnet reserved for calibration validation and tie-breaking only.

**Dual Evaluation Mode** — Iteration mode: single API call scoring all 5 dimensions (fast). Final mode: separate API call per dimension with specialized rubric (precise).

## Patterns

- **Pydantic v2 at all boundaries**. Use `model_validate()` and `model_dump()` — never v1 methods. Use `model_json_schema()` for structured output.
- **Raw SQL with parameterized queries**. No ORM. All DB access through `src/db/queries.py`. Context managers for connections.
- **Decision logging at every branch point**. Every function that makes a choice calls `log_decision()` before executing. This is not optional.
- **Exponential backoff for API retries** via tenacity. Handle rate limits (429), safety filter blocks, transient errors. Max 3 retries, 2^n second delays.
- **LiteLLM for token counting and cost tracking**. No manual token counting.
- **snake_case everywhere** except PascalCase for Pydantic models.
- **Google GenAI SDK** (`from google import genai`), not the deprecated `google-generativeai` package.

## Project Structure

```
src/
  models/      — Pydantic schemas (AdBrief, AdCopy, Evaluation, Decision, Iteration)
  generate/    — Gemini Flash ad copy generation, prompt templates
  evaluate/    — Cross-model LLM-as-judge, dimension rubrics, calibration
  iterate/     — Hybrid iteration controller, self-healing, coherence checker
  intel/       — Competitor pattern analysis, curated ad ingestion
  analytics/   — Cost tracking, quality trends, weight evolution
  pipeline/    — Batch orchestration, brief matrix, main entry point
  db/          — SQLite schema, init, query helpers
  decisions/   — Decision logger, queries, export
  dashboard/   — Streamlit app
  output/      — JSON/CSV export, report generation
tests/
  fixtures/    — Recorded LLM responses, sample briefs, reference ads
data/
  reference_ads/   — Varsity Tutors ads for calibration
  competitor_ads/  — Curated competitor patterns
  briefs/          — Seed ad brief configurations
```

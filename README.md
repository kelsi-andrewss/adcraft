# AdCraft — Autonomous Ad Engine for Varsity Tutors

Autonomous system that generates Facebook/Instagram ad copy for Varsity Tutors SAT test prep products. Uses Gemini 2.5 Flash for generation and Gemini 2.5 Pro for cross-model evaluation across 5 quality dimensions (clarity, value proposition, CTA strength, brand voice, emotional resonance). Self-healing iteration loops detect weak dimensions and auto-correct via hybrid component-fix or full regeneration. Tracks performance-per-token as the north star metric.

## Setup

```bash
uv sync
cp .env.example .env
# Fill in GEMINI_API_KEY and ANTHROPIC_API_KEY in .env
```

## Run

```bash
# Run the full generation/evaluation/iteration pipeline
uv run python -m src.pipeline.main

# Launch the Streamlit dashboard
uv run streamlit run src/dashboard/app.py
```

## Development

```bash
# Run tests
uv run pytest

# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/
```

## Architecture

- **Generation**: Gemini 2.5 Flash with structured JSON output and Pydantic validation
- **Evaluation**: Cross-model LLM-as-judge (Gemini 2.5 Pro) with 5 decomposed dimensions, weighted aggregation, and hard gates
- **Iteration**: Hybrid component-fix then full-regen with coherence checking, max 3 cycles
- **Persistence**: SQLite with WAL mode, raw SQL with parameterized queries
- **Decision Logging**: Every pipeline branch point records action, rationale, and context
- **Cost Tracking**: LiteLLM for unified token counting across Gemini and Claude
- **Dashboard**: Streamlit with Nerdy brand theme — ad library, quality trends, iteration inspector, decision log, cost tracker

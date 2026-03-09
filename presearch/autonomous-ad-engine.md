# AdCraft — Autonomous Ad Engine for Varsity Tutors

## Overview

An autonomous system that generates Facebook and Instagram ad copy for Varsity Tutors' SAT test prep products, evaluates quality across 5 decomposed dimensions using cross-model LLM-as-judge, iterates via self-healing feedback loops, and tracks performance-per-token as the north star metric. The system knows what good looks like, filters ruthlessly, and improves measurably over time.

The domain is tight: paid social ads for Meta platforms only. Primary text, headline, description, CTA button. One brand (Varsity Tutors), one audience family (SAT test prep — parents and students), one channel family (FB/IG).

## Summary

Autonomous ad copy engine for Varsity Tutors SAT test prep. Generates FB/IG ads using Gemini 2.5 Flash, evaluates with Gemini 2.5 Pro (cross-model to avoid self-bias), iterates via hybrid component-then-full regeneration. Five quality dimensions (clarity, value prop, CTA, brand voice, emotional resonance) scored independently with weighted aggregation and hard gates. Self-healing loops detect quality drops and auto-correct. Competitive intelligence via manually curated competitor ads from Meta Ad Library (public website) feeds real patterns into generation context. Performance-per-token tracking measures quality per dollar of API spend. SQLite persistence, Plotly visualization, comprehensive decision logging. Calibration-first build order: evaluator validated against labeled reference ads before any generation begins. Python, runs locally, one-command setup. Public GitHub repo with GitHub Actions CI.

## Features

### MVP (Phase 1 — Core Pipeline)
0. **Bootstrap**: `uv init && uv add google-genai anthropic pydantic litellm plotly pandas python-dotenv streamlit` + create all Pydantic models in `src/models/` + create `.env.example` + create `src/db/schema.sql` and init script + set up pytest config + create project CLAUDE.md from Patterns section + `uv add --dev pytest pytest-asyncio ruff` + init public GitHub repo + configure GitHub Actions CI (lint + test on push)
1. **Ad Brief System** — Define structured ad briefs with audience segment, product/offer, campaign goal, tone. Seed briefs covering parent-focused and student-focused angles. (`src/briefs/`, `src/models/brief.py`)
2. **Ad Copy Generator** — Gemini 2.5 Flash generation with structured JSON output. Pydantic-validated ad components (primary text, headline, description, CTA button). Few-shot prompting with brand voice examples. (`src/generate/`, `src/models/ad.py`)
3. **Evaluation Framework** — Cross-model LLM-as-judge using Gemini 2.5 Pro. Multi-trait specialization: separate prompt per dimension. Chain-of-thought rationale before scoring. Few-shot calibration with reference ads. Weighted aggregation with hard gates. (`src/evaluate/`, `src/models/evaluation.py`)
4. **SQLite Persistence** — Store all ads, evaluations, iterations, decisions. Enable quality trend queries and performance-per-token calculations. (`src/db/`)
5. **Decision Logger** — First-class system component, not an afterthought. Every conditional branch in the pipeline writes a human-readable entry before executing. Captures agent, action, rationale, context. (`src/decisions/`)

### Phase 2 — Autonomous Iteration
6. **Hybrid Iteration Controller** — Try component-level fix first (regenerate only the weak dimension's component). Run coherence check. Fall back to full regeneration if coherence drops. Max 3 cycles per ad before force-fail. (`src/iterate/`)
7. **Self-Healing Loops** — Quality ratchet: track running weighted average, detect regression. Auto-diagnose which dimension is dragging. Targeted intervention strategies per dimension. (`src/iterate/healing.py`)
8. **Weight Evolution** — Track per-dimension scores across all iterations. After 50+ ads, analyze whether initial weights match observed quality correlations. Document findings in decision log. (`src/analytics/weights.py`)
9. **Batch Pipeline** — Scale to 50+ ads efficiently. Brief variation generation (audience x offer x tone matrix). Parallel generation where rate limits allow. Progress tracking. (`src/pipeline/`)

### Phase 3 — Intelligence & Polish
10. **Competitive Intelligence** — Manually curate 20-30 competitor ads from the public Meta Ad Library website (Princeton Review, Kaplan, Khan Academy, Chegg). Structure into JSON dataset. Extract hook patterns, CTA patterns, emotional angles. Feed patterns into generation context. (`src/intel/`, `data/competitor_ads/`)
11. **Performance-Per-Token Tracking** — Track input/output tokens and cost for every API call. Calculate quality-per-dollar. Identify which model/config produces best ROI. (`src/analytics/cost.py`)
12. **Quality Trend Analytics** — Score distributions across iterations, dimension-level improvement tracking, before/after comparisons, convergence curves. (`src/analytics/trends.py`)
13. **Streamlit Dashboard** — Single-page product that unifies all system outputs. Tabs: Ad Library (browse ads with scores, filter by pass/fail), Quality Trends (embedded Plotly charts), Iteration Inspector (ad journey: original → feedback → revision → re-score), Decision Log (searchable/filterable), Cost Tracker (performance-per-dollar), Pipeline Runner (trigger batch runs, watch progress). (`src/dashboard/`)
14. **Evaluation Report & Export** — JSON/CSV export of all ads with scores. Summary statistics. Quality trend data. Decision log export. (`src/output/`)

### Stretch Goal
15. **Automated Competitor Scraping (Apify)** — Replace manual curation with automated scraping of the public Meta Ad Library UI via Apify. Enables larger dataset (hundreds of ads), refreshable data, temporal tracking of competitor ad changes. Only pursue after core pipeline, iteration, and intelligence features are solid. (`src/intel/scraper.py`) — Requires: `uv add apify-client`, `APIFY_API_TOKEN` env var.

### Cut
- Image generation (v2 scope — not targeting for this submission)
- Real-time Meta Ad Library monitoring (one-time collection is sufficient)

## Technical Research

### APIs & Services

**Gemini 2.5 Flash (Generation)**
- Model ID: `gemini-2.5-flash`
- Pricing: $0.30/1M input, $2.50/1M output (free tier available)
- Auth: API key via `GEMINI_API_KEY` (ai.google.dev)
- Client init: `genai.Client(api_key=os.environ["GEMINI_API_KEY"])`
- Structured output: `response_mime_type="application/json"`, `response_json_schema=AdCopy.model_json_schema()`
- SDK: `google-genai` — `from google import genai`
- Determinism: temperature=0.0, no guaranteed seed support
- Gotcha: safety filters may trigger on "anxiety"/"stress" language in parent-targeted ads

**Gemini 2.5 Pro (Evaluation)**
- Model ID: `gemini-2.5-pro`
- Pricing: $1.25/1M input, $10.00/1M output (free tier available)
- Same SDK and auth as Flash
- Used for multi-trait evaluation with structured rationale output

**Claude Sonnet 4.6 (Evaluation Fallback/Tie-breaker)**
- Model ID: `claude-sonnet-4-6`
- Pricing: $3.00/1M input, $15.00/1M output
- Auth: API key via `ANTHROPIC_API_KEY`, header `x-api-key`
- Structured output via tool use API (define evaluation tool, force tool_choice)
- Reserved for: tie-breaking disputed scores, final certification pass, calibration validation

**Meta Ad Library (Competitive Intelligence)**
- The public website (facebook.com/ads/library) is freely browsable — search any advertiser, see all active ads. This is our primary data source.
- Primary approach: manually browse and curate 20-30 competitor ads into structured JSON (Princeton Review, Kaplan, Khan Academy, Chegg). Categorize by hook type, emotional angle, CTA pattern.
- Stretch goal: automate collection via Apify Meta Ads Library Scraper (`apify/meta-ads-library-scraper`) for scale and refresh capability.
- Note: the official API (`graph.facebook.com/ads_archive`) is limited to EU/UK commercial ads and political ads globally — NOT usable for US commercial competitor research.

**LiteLLM (Cost Tracking)**
- Unified interface for token counting across Gemini and Claude
- Automatic cost calculation per call
- Not used for routing — direct SDK calls for generation/evaluation, LiteLLM for cost logging

### Build Order (Calibration First)

The evaluator is the foundation. If it can't tell an 8/10 from a 5/10, the feedback loop iterates toward a broken compass.

1. **Bootstrap** — scaffold, deps, models, DB, CI
2. **Curate reference ads** — collect from Slack (Varsity Tutors) + Meta Ad Library (competitors). Manually label 5-10 as "great" and 5-10 as "bad" with human scores per dimension.
3. **Build evaluator** — implement scoring, run against labeled reference ads. Validate: does it score "great" ads 7+ and "bad" ads <5? If not, fix rubric prompts until it does. This is the single most important step.
4. **Build generator** — only after evaluator is calibrated. Start with one audience, one offer.
5. **Wire iteration loop** — generate, evaluate, identify weakness, fix, re-evaluate.
6. **Scale** — brief matrix, batch pipeline, 50+ ads.
7. **Intelligence & analytics** — competitive patterns, cost tracking, visualizations.
8. **Dashboard** — Streamlit app unifying all outputs into a single product.
9. **Document** — decision log, limitations, technical writeup. Written continuously, not at the end.

### Execution Plan (One-Shot + Pressure Test)

This is designed to be built in one focused session, then refined in a pressure testing pass.

**Session 1 — Build (one-shot)**
Each phase feeds the next. No waiting between phases — the system's iteration cycles run in minutes.
- Phase 1 steps 1-5 above (bootstrap → evaluator calibrated → generator working → iteration wired)
- Phase 2 steps 6-7 (batch run → observe real data → document what happened)
- Phase 3 steps 8-9 (competitive patterns → dashboard → export)
- End state: working system with 50+ ads, scores, iteration history, dashboard, decision log

**Session 2 — Pressure Test & Tune**
Re-iteration plan using observed data from Session 1:
1. **Score distributions** — are they clustered in a narrow band? Too generous? Adjust rubric prompts, re-calibrate.
2. **Iteration deltas** — did weak dimensions actually improve? If not, tune feedback prompt templates. Try different intervention strategies per dimension.
3. **Coherence check** — are component-level fixes creating Frankenstein ads? Adjust the coherence threshold or bias toward full regeneration.
4. **Decision log review** — does the narrative read as a coherent engineering story? Fill gaps in reasoning.
5. **Weight evolution** — do the initial weights hold up against observed quality data? Document the finding either way.
6. **Run again** with tuned prompts. Compare before/after quality distributions.
7. **Stretch goal** — if confident, wire Apify for expanded competitive dataset, re-run pipeline.
8. **Final polish** — demo video, technical writeup, limitations doc.

### Architecture

**Procedural State Machine** (not LangGraph)
- Simple, testable, debuggable pipeline with explicit state transitions
- Each stage is a function that takes state in, returns state out
- Decision log writes happen at every branch point
- Rationale: LangGraph adds framework overhead without proportional benefit for a linear pipeline with one feedback loop. A while loop with match/case is clearer and produces better decision logs.

**Cross-Model Evaluation**
- Generation: Gemini 2.5 Flash (cheap, fast, good creative writing)
- Evaluation: Gemini 2.5 Pro (better reasoning, still affordable on free tier)
- Rationale: avoids self-bias where model overrates its own output. Claude reserved for calibration validation and tie-breaking only (10x cost premium not justified for every evaluation).

**Dual Evaluation Mode**
- **Iteration mode** (fast): single API call scoring all 5 dimensions in one structured response. Used during feedback loops where speed matters.
- **Final mode** (precise): multi-trait specialization — separate API call per dimension with dimension-specific rubric and few-shot examples. Used for final scoring and library admission.

### Patterns

- **HTTP client**: direct SDK clients (`genai.Client(api_key=...)`, `anthropic.Anthropic()`). No wrapper abstraction — two providers don't justify one.
- **Error handling**: retry with exponential backoff for rate limits and safety filter triggers. Log all failures to decision log. Never swallow errors silently.
- **Validation**: Pydantic v2 at all boundaries — ad briefs in, ad copy out, evaluation results. `model_validate_json()` for API responses.
- **State management**: dataclasses for pipeline state, SQLite for persistence. No ORM — raw SQL with parameterized queries.
- **Naming**: snake_case for everything (files, functions, variables). PascalCase for Pydantic models only.
- **Testing**: pytest with fixtures. Mock LLM responses via recorded snapshots for deterministic tests. Integration tests hit real APIs with temperature=0.
- **Decision logging**: every function that makes a choice calls `log_decision(component, action, rationale, context)` before executing. This is not optional.

### Project Structure
```
autonomous-ad-engine/
  src/
    models/          -- Pydantic schemas: AdBrief, AdCopy, Evaluation, Decision
    generate/        -- Gemini Flash ad copy generation, prompt templates
    evaluate/        -- Cross-model LLM-as-judge, dimension rubrics, calibration
    iterate/         -- Hybrid iteration controller, self-healing, coherence checker
    intel/           -- Competitor pattern analysis, curated ad ingestion
    analytics/       -- Cost tracking, quality trends, weight evolution, Plotly charts
    pipeline/        -- Batch orchestration, brief matrix generation, main entry point
    db/              -- SQLite schema, migrations, query helpers
    decisions/       -- Decision logger, decision log queries and export
    dashboard/       -- Streamlit app: ad library, trends, iteration inspector, decision log, cost tracker
    output/          -- JSON/CSV export, evaluation report generation
  tests/
    fixtures/        -- Recorded LLM responses, sample briefs, reference ads
    test_generate.py
    test_evaluate.py
    test_iterate.py
    test_pipeline.py
    test_decisions.py
  data/
    reference_ads/   -- Real Varsity Tutors ads (from Slack) for calibration
    competitor_ads/  -- Scraped/curated competitor ad patterns
    briefs/          -- Seed ad brief configurations
  docs/
    decision_log.md  -- Human-readable decision log (exported from DB)
    limitations.md   -- Honest limitations documentation
    technical_writeup.md -- 1-2 page technical writeup for submission
  .streamlit/
    config.toml      -- Nerdy brand theme (cyan #17E2EA, dark navy #0F0928)
  .github/
    workflows/
      ci.yml           -- GitHub Actions: lint (ruff) + test (pytest) on push
  pyproject.toml
  .env.example
  README.md
```

- Bootstrap command: `uv init adcraft && cd adcraft`
- Config: pyproject.toml with `[project]` metadata, `[tool.pytest.ini_options]` for test config, `[tool.ruff]` for lint config

### Nerdy Brand Theme (Streamlit)

Dashboard follows the Nerdy brand palette via `.streamlit/config.toml`:
- **Primary**: `#17E2EA` (Nerdy cyan — buttons, accents, interactive elements)
- **Background**: `#0F0928` (dark navy)
- **Secondary background**: `#161C2C` (card/panel backgrounds)
- **Text**: `#FFFFFF` (white)
- **Accent purple**: `#6C64C9` (secondary highlights, badges)

```toml
# .streamlit/config.toml
[theme]
primaryColor = "#17E2EA"
backgroundColor = "#0F0928"
secondaryBackgroundColor = "#161C2C"
textColor = "#FFFFFF"
font = "sans serif"
```

Bootstrap (story-581) creates this file. Dashboard inherits the theme automatically.

### Shared Interfaces

- `src/models/brief.py`: `AdBrief` — audience_segment, product_offer, campaign_goal, tone, competitive_context (used by features: 1, 2, 9)
- `src/models/ad.py`: `AdCopy` — id, primary_text, headline, description, cta_button, brief_id, model_id, generation_config, token_count (used by features: 2, 3, 4, 6, 13)
- `src/models/evaluation.py`: `DimensionScore` — dimension, score, rationale, confidence; `EvaluationResult` — ad_id, scores[], weighted_average, passed_threshold, hard_gate_failures[], evaluator_model, token_count (used by features: 3, 6, 7, 8, 11, 12)
- `src/models/decision.py`: `DecisionEntry` — timestamp, component, action, rationale, context, agent_id (used by features: 5, 6, 7, all)
- `src/models/iteration.py`: `IterationRecord` — source_ad_id, target_ad_id, cycle_number, action_type (component_fix|full_regen), weak_dimension, delta_scores, token_cost (used by features: 6, 7, 12)
- `src/db/queries.py`: shared query functions for ads, evaluations, iterations, decisions (used by features: 4, 11, 12, 13)

### Data Model

**ads** table:
- id TEXT PRIMARY KEY, brief_id TEXT, primary_text TEXT, headline TEXT, description TEXT, cta_button TEXT, model_id TEXT, temperature REAL, generation_seed TEXT, input_tokens INT, output_tokens INT, cost_usd REAL, created_at TIMESTAMP

**evaluations** table:
- id TEXT PRIMARY KEY, ad_id TEXT FK, dimension TEXT, score REAL, rationale TEXT, confidence REAL, evaluator_model TEXT, eval_mode TEXT (iteration|final), input_tokens INT, output_tokens INT, cost_usd REAL, created_at TIMESTAMP

**iterations** table:
- id TEXT PRIMARY KEY, source_ad_id TEXT FK, target_ad_id TEXT FK, cycle_number INT, action_type TEXT, weak_dimension TEXT, feedback_prompt TEXT, delta_weighted_avg REAL, token_cost REAL, created_at TIMESTAMP

**decisions** table:
- id TEXT PRIMARY KEY, timestamp TIMESTAMP, component TEXT, action TEXT, rationale TEXT, context TEXT, agent_id TEXT

**competitor_ads** table:
- id TEXT PRIMARY KEY, brand TEXT, primary_text TEXT, headline TEXT, cta_button TEXT, hook_type TEXT, emotional_angle TEXT, scraped_at TIMESTAMP

**quality_snapshots** table:
- id TEXT PRIMARY KEY, cycle_number INT, avg_weighted_score REAL, dimension_averages JSON, ads_above_threshold INT, total_ads INT, token_spend_usd REAL, quality_per_dollar REAL, created_at TIMESTAMP

### Dependencies

**Runtime:**
- `google-genai` — Gemini API SDK (generation + evaluation)
- `anthropic` — Claude API SDK (evaluation fallback)
- `pydantic>=2.9` — data validation and structured output schemas
- `litellm` — unified token counting and cost tracking
- `plotly` — interactive quality trend visualization
- `pandas` — data manipulation for analytics
- `streamlit` — dashboard UI (ad library, quality trends, iteration inspector, decision log, cost tracker)
- `python-dotenv` — environment variable management

**Development:**
- `pytest` — test framework
- `pytest-asyncio` — async test support
- `ruff` — linting and formatting

**Stretch goal only:**
- `apify-client` — automated Meta Ad Library scraping (not in initial deps)

### Gotchas

1. **Meta Ad Library API doesn't cover US commercial ads.** The official API only returns EU/UK commercial ads and political/social ads globally. The public website is freely browsable — manual curation from the website is the primary path. Apify automation is a stretch goal.
2. **Gemini safety filters on "anxiety"/"stress" language.** Parent-targeted SAT ads naturally use emotional language that can trigger content filters. Set `safety_settings` to `BLOCK_ONLY_HIGH`. Implement retry with softened language as fallback. Log all safety rejections.
3. **No deterministic seeds in Gemini API.** Temperature=0.0 reduces but doesn't eliminate variance. For "reproducibility" requirement, log exact prompts, model versions, and config so results can be approximately reproduced. Don't promise bit-perfect reproduction.
4. **LLM evaluator score inflation.** Even with cross-model evaluation, LLMs tend toward generous scoring. Calibrate by scoring known-bad reference ads first — if the worst reference ad scores above 5, the rubric needs tightening.
5. **Dimension score correlation.** When scoring all 5 dimensions in a single call (iteration mode), models tend to give similar scores across dimensions. Multi-trait specialization (final mode) mitigates this but costs 5x.
6. **Context window drift in iteration loops.** After 3 cycles of feedback, the model may lose sight of the original brief. Always include the original brief in every iteration prompt, not just the feedback.
7. **Free tier rate limits.** Gemini free tier is ~10-15 RPM, 500-1000 RPD. Generating and evaluating 50 ads with iterations could hit daily caps. Implement rate limiting with backoff. Paid tier costs <$5 total if needed.
8. **Gemini 2.0 Flash deprecated June 2026.** Use 2.5 Flash from the start. Don't reference 2.0 models.

### Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Curated competitor dataset too small for meaningful patterns | Low | Med | 20-30 ads across 4 competitors is sufficient for hook/CTA/angle pattern extraction. Stretch goal (Apify) expands to hundreds if needed. |
| Gemini safety filters reject valid ad copy | Med | Med | BLOCK_ONLY_HIGH setting. Retry with softened prompt. Log rejections. Have 60+ briefs to absorb some failures. |
| Free tier rate limits block batch generation | Med | Med | Implement rate limiting with exponential backoff. Spread generation across multiple runs. Paid tier costs <$5 total. |
| Evaluator scores don't correlate with human judgment | High | Med | Calibrate against reference ads before generating anything. If calibration fails, fix rubric first. This is the single most important thing to get right. |
| Quality doesn't measurably improve over iterations | High | Low | If hybrid iteration doesn't lift scores, document why honestly. The rubric rewards honesty about failures (20% Documentation). |
| Token costs exceed expectations | Low | Low | LiteLLM tracking catches this early. Estimated total cost: $2-10 for full 50+ ad pipeline. |

### Cost Estimate

**Development complexity:**
| Feature | Size | Notes |
|---------|------|-------|
| 0. Bootstrap | S | Scaffold, deps, models, DB schema |
| 1. Ad Brief System | S | Pydantic models + seed data |
| 2. Ad Copy Generator | M | Prompt engineering is the hard part |
| 3. Evaluation Framework | L | Calibration, multi-trait, hard gates — most critical feature |
| 4. SQLite Persistence | S | Schema + basic CRUD |
| 5. Decision Logger | S | But must be wired into everything |
| 6. Hybrid Iteration Controller | L | Component fix + coherence check + fallback logic |
| 7. Self-Healing Loops | M | Quality ratchet + regression detection |
| 8. Weight Evolution | S | Analytics on existing data |
| 9. Batch Pipeline | M | Rate limiting, progress, brief matrix |
| 10. Competitive Intelligence | S | Manual curation + pattern analysis (scraper is stretch) |
| 11. Cost Tracking | S | LiteLLM does the heavy lifting |
| 12. Quality Trends | M | Plotly charts from DB queries |
| 13. Streamlit Dashboard | M | 5-6 tabs unifying existing outputs — no new logic, just presentation |
| 14. Export & Report | S | JSON/CSV serialization |

**Operational costs per full pipeline run (50+ ads):**
| Component | Est. Cost per Run |
|-----------|-------------------|
| Gemini 2.5 Flash (generation) | $0 (free tier) or ~$0.50-1.00 |
| Gemini 2.5 Pro (evaluation) | $0 (free tier) or ~$1.50-3.00 |
| Claude Sonnet (calibration only) | ~$0.50-1.00 |
| Apify (stretch goal only) | ~$5-10/month |
| **Total per run** | **$0-5** |

Free tier should be sufficient for the submission. Paid tier is cheap insurance if rate limits bite.

### Deployment

Local CLI + Streamlit dashboard with public GitHub repo.
- **Platform**: Local Python environment via `uv`
- **Pipeline**: `uv run python -m src.pipeline.main` — runs the full generation/evaluation/iteration pipeline
- **Dashboard**: `uv run streamlit run src/dashboard/app.py` — opens the Nerdy-themed dashboard at localhost:8501
- **Setup**: `uv sync && cp .env.example .env` (user fills in API keys)
- **Demo**: Screen-record the Streamlit dashboard for the demo video. One page shows everything.
- **Repo**: Public GitHub repository (submission requirement)
- **CI**: GitHub Actions — runs `ruff check` and `pytest` on every push. Secrets (`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`) stored in GitHub Actions secrets for integration tests. Unit tests use mocked responses and run without API keys.
- **GitHub Actions workflow** (`.github/workflows/ci.yml`): lint job (ruff check + ruff format --check) and test job (pytest with --ignore for integration tests unless secrets are available)

## Environment

- `GEMINI_API_KEY` — Google AI Studio API key for Gemini models (required)
- `ANTHROPIC_API_KEY` — Anthropic API key for Claude evaluation (required — cross-model evaluation is a core architectural decision, not optional)
- `DATABASE_PATH` — SQLite database path, defaults to `data/ads.db` (optional)
- `APIFY_API_TOKEN` — Apify token for automated Meta Ad Library scraping (stretch goal only, optional)
- `LOG_LEVEL` — Logging verbosity, defaults to INFO (optional)

## Decisions

- **Scope**: v3 autonomous engine — maximizes bonus points (+24 possible), demonstrates systems thinking beyond basic pipeline
- **Generation model**: Gemini 2.5 Flash — cheapest current model with free tier, good creative writing, structured output support
- **Evaluation model**: Gemini 2.5 Pro (primary), Claude Sonnet 4.6 (calibration/tie-break) — cross-model avoids self-bias, Pro is 10x cheaper than Claude for routine evaluation
- **Orchestration**: Procedural state machine, not LangGraph — simpler, more testable, better decision logs, no framework overhead for a linear pipeline
- **Iteration strategy**: Hybrid component-then-full — try targeted fix first for token efficiency, fall back to full regen if coherence drops
- **Dimension weighting**: Weighted + hard gates, evolving — brand voice hard gate at <5, clarity and value prop weighted higher. Track correlations to validate/adjust. Documents independent judgment.
- **Evaluation mode**: Dual (iteration=single-call, final=multi-trait) — balances speed during iteration with precision for final scoring
- **Competitive intelligence**: Manual curation from public Meta Ad Library website, Apify automation as stretch goal — the +10 bonus rewards the intelligence layer, not the data collection method
- **Build order**: Calibration first — evaluator validated against labeled reference ads before any generation. If the judge can't tell good from bad, nothing else matters.
- **Persistence**: SQLite — lightweight, local, no server, sufficient for 50-100 ads with full history
- **Package manager**: uv — fast, modern, deterministic lockfile
- **No LangGraph**: overhead not justified. A while loop with explicit state is clearer and more testable than a graph framework for a pipeline with one feedback loop.

## Constraints

- Greenfield Python project (no existing codebase)
- Must run locally with API keys
- No real PII in generated content
- Gemini API free tier as primary (paid tier as fallback)
- Target: 50+ ads, 3+ iteration cycles, 10+ tests
- One-command setup
- Public GitHub repo with GitHub Actions CI
- Decision log is a graded deliverable (20% of rubric)
- Reference/competitor ad data: publicly available data or mock data (per Slack guidance)

## Reference

- [Gemini API Pricing](https://ai.google.dev/gemini-api/docs/pricing) — current model pricing and availability
- [Gemini Structured Output](https://ai.google.dev/gemini-api/docs/structured-output) — JSON mode with Pydantic schemas
- [Gemini Rate Limits](https://ai.google.dev/gemini-api/docs/rate-limits) — free tier constraints
- [Meta Ad Library API](https://www.facebook.com/ads/library/api) — official API (limited to EU/UK + political)
- [Apify Meta Ads Scraper](https://apify.com/leadsbrary/meta-ads-library-scraper/api) — third-party scraper for US commercial ads
- [LLM-as-Judge Guide (Evidently AI)](https://www.evidentlyai.com/llm-guide/llm-as-a-judge) — calibration and bias mitigation
- [LLM-as-Judge Guide (Langfuse)](https://langfuse.com/docs/evaluation/evaluation-methods/llm-as-a-judge) — implementation patterns
- [LLM-as-Judge Survey (arXiv)](https://arxiv.org/html/2412.05579v2) — comprehensive academic survey

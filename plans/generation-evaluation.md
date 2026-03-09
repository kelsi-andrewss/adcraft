# Phase 1 Core Generation + Evaluation

Story: story-583
Agent: architect

## Context

The evaluator is the foundation of the entire autonomous pipeline. If it can't distinguish an 8/10 from a 5/10, the feedback loop iterates toward a broken compass (briefing: "Build Order — Calibration First"). This plan enforces strict build order: reference ads first, evaluator second, calibration validation third, generator fourth. No generation code runs until the evaluator proves it can score known-good ads 7+ and known-bad ads <5.

Key architectural constraints from the briefing ("Architecture" and "Gotchas" sections):
- Cross-model evaluation: Gemini 2.5 Pro (`gemini-2.5-pro`) judges output from Gemini 2.5 Flash (`gemini-2.5-flash`) to avoid self-bias
- Dual evaluation mode: single-call for iteration speed, separate-call-per-dimension for final precision (briefing: "Dual Evaluation Mode")
- Chain-of-thought: rationale BEFORE score in every evaluation prompt to force reasoning (briefing: Gotcha #4 — score inflation)
- Hard gate: brand_voice < 5 = auto-reject regardless of weighted average (briefing: "Dimension weighting" decision)
- Safety settings: `BLOCK_ONLY_HIGH` for parent-targeted SAT ads that naturally use anxiety/stress language (briefing: Gotcha #2)
- SDK: `from google import genai`, structured output via `response_mime_type="application/json"` + `response_json_schema` (briefing: "APIs & Services")

## Prerequisites

This story assumes bootstrap (story for Phase 0) has completed and the following exist:
- `src/models/ad.py` — `AdCopy` Pydantic model
- `src/models/evaluation.py` — `DimensionScore`, `EvaluationResult` Pydantic models
- `src/db/queries.py` — DB persistence helpers (`save_ad`, `save_evaluation`, etc.)
- `src/decisions/logger.py` — `log_decision(component, action, rationale, context)` function
- `pyproject.toml` with all deps installed
- `.env` with `GEMINI_API_KEY`

If bootstrap hasn't run, these models are the first thing to build. The plan below references their interfaces but does not redefine them.

## What changes

| File | Change |
|---|---|
| `data/reference_ads/labeled_ads.json` | 10 manually labeled reference ads (5 "great" scoring 7-9 per dimension, 5 "bad" scoring 2-4) with human rationales per dimension. Covers parent-focused and student-focused angles for SAT test prep. Used as calibration ground truth. |
| `src/evaluate/rubrics.py` | Dimension definitions, scoring rubrics (1-10 scale), CoT prompt templates, few-shot examples, and dimension weights. Five dimensions: clarity (weight 0.25), value_prop (0.25), cta_effectiveness (0.20), brand_voice (0.15), emotional_resonance (0.15). Hard gate constant: `BRAND_VOICE_HARD_GATE = 5`. |
| `src/evaluate/engine.py` | `EvaluationEngine` with two public methods: `evaluate_iteration(ad_copy) -> EvaluationResult` (single Gemini 2.5 Pro call, all 5 dimensions) and `evaluate_final(ad_copy) -> EvaluationResult` (5 separate calls, one per dimension). Both enforce CoT (rationale before score in prompt/schema), apply weighted aggregation, enforce hard gate on brand_voice. Calls `log_decision` for every scoring decision and hard gate trigger. Retry with exponential backoff for safety filters and rate limits. |
| `src/evaluate/__init__.py` | Package init, exports `EvaluationEngine`. |
| `src/generate/engine.py` | `GenerationEngine` with `generate(brief) -> AdCopy` using Gemini 2.5 Flash. Structured output via `response_mime_type="application/json"` + `response_json_schema=AdCopy.model_json_schema()`. Few-shot brand voice examples in system prompt. Safety settings `BLOCK_ONLY_HIGH`. Retry with exponential backoff for safety filter blocks and rate limits. Calls `log_decision` for generation config choices and safety filter retries. |
| `src/generate/__init__.py` | Package init, exports `GenerationEngine`. |
| `tests/test_evaluate.py` | Unit tests with mocked Gemini responses: iteration mode scoring, final mode scoring, hard gate rejection (brand_voice < 5 forces fail even with high weighted avg), weighted average calculation, CoT rationale extraction. Tests calibration: mocked "great" ad scores 7+, mocked "bad" ad scores < 5. |
| `tests/test_generate.py` | Unit tests with mocked Gemini responses: successful generation returns valid AdCopy, safety filter retry triggers exponential backoff, structured output validation via Pydantic, log_decision called on every generation. |

## Read-only context

- `presearch/autonomous-ad-engine.md` — full briefing with API shapes, model IDs, pricing, structured output patterns, evaluation architecture, gotchas
- `src/models/ad.py` — `AdCopy` model (fields: id, primary_text, headline, description, cta_button, brief_id, model_id, generation_config, token_count)
- `src/models/evaluation.py` — `DimensionScore` (dimension, score, rationale, confidence), `EvaluationResult` (ad_id, scores[], weighted_average, passed_threshold, hard_gate_failures[], evaluator_model, token_count)
- `src/db/queries.py` — DB persistence helpers (save_evaluation, save_ad, etc.)
- `src/decisions/logger.py` — `log_decision(component, action, rationale, context)` function

## Tasks

**Order is non-negotiable. Each task gates the next.**

### 1. Create reference ads for calibration
**File:** `data/reference_ads/labeled_ads.json`

Create 10 reference ads with manual human scores per dimension (1-10). Structure:
```json
{
  "reference_ads": [
    {
      "id": "ref-great-001",
      "label": "great",
      "primary_text": "...",
      "headline": "...",
      "description": "...",
      "cta_button": "Learn More",
      "audience": "parent",
      "human_scores": {
        "clarity": 8,
        "value_prop": 9,
        "cta_effectiveness": 8,
        "brand_voice": 8,
        "emotional_resonance": 7
      },
      "human_rationale": "Clear benefit statement, specific score improvement claim..."
    }
  ]
}
```

- 5 "great" ads: clear value props, specific claims (e.g., "Average 160-point score improvement"), strong CTAs, authentic Varsity Tutors voice. Human scores 7-9 per dimension.
- 5 "bad" ads: vague claims, generic language, weak CTAs, off-brand voice. Human scores 2-4 per dimension.
- Mix of parent-focused ("Your child's SAT score...") and student-focused ("Crush the SAT...") angles.
- These serve as ground truth for calibration in Task 4 and as few-shot examples in rubric prompts.

### 2. Define dimension rubrics and prompts
**File:** `src/evaluate/rubrics.py`

For each of the 5 dimensions, define:
- **Rubric**: what scores 1-3 (poor), 4-6 (adequate), 7-8 (strong), 9-10 (exceptional) with concrete SAT ad examples
- **CoT prompt template**: instructs evaluator to produce rationale FIRST, then score (critical — briefing Gotcha #4 says LLMs inflate scores; forcing reasoning first anchors the score)
- **Few-shot examples**: 1 great + 1 bad reference ad excerpt per dimension with expected score and rationale
- **Dimension weights**: `DIMENSION_WEIGHTS = {"clarity": 0.25, "value_prop": 0.25, "cta_effectiveness": 0.20, "brand_voice": 0.15, "emotional_resonance": 0.15}`
- **Hard gate**: `BRAND_VOICE_HARD_GATE = 5` — below this, the ad auto-fails
- **Passing threshold**: `PASSING_THRESHOLD = 6.5` for weighted average

Prompt structure for single-dimension evaluation (final mode):
```
You are evaluating a Facebook/Instagram ad for Varsity Tutors SAT test prep.

Dimension: {dimension_name}
Rubric: {rubric_text}

Few-shot examples:
{examples}

Ad to evaluate:
Primary text: {primary_text}
Headline: {headline}
Description: {description}
CTA: {cta_button}

First, explain your reasoning about this ad's {dimension_name} in 2-3 sentences.
Then provide your score (1-10).
```

Prompt structure for all-dimensions evaluation (iteration mode):
```
Evaluate this ad across all 5 dimensions. For EACH dimension, first explain your reasoning, then score.
{combined rubrics and examples}
{ad text}
```

Response schema forces `{"rationale": "...", "score": N}` per dimension — rationale field comes first to encourage the model to reason before scoring.

### 3. Implement evaluation engine
**File:** `src/evaluate/engine.py`

```python
from google import genai
```

`EvaluationEngine` class:
- **Constructor**: initializes `genai.Client()`, stores model ID `gemini-2.5-pro`, loads rubrics from `rubrics.py`
- **`evaluate_iteration(ad_copy: AdCopy) -> EvaluationResult`**: single API call, all 5 dimensions in one structured response. Uses `response_mime_type="application/json"` with schema requiring rationale+score per dimension. Faster, used during iteration loops.
- **`evaluate_final(ad_copy: AdCopy) -> EvaluationResult`**: 5 separate API calls, one per dimension with dimension-specific rubric and few-shot examples. More precise, used for final scoring and library admission. (Briefing: "Dimension score correlation" gotcha — single-call produces correlated scores.)
- **`_compute_result(scores: list[DimensionScore]) -> EvaluationResult`**: applies weighted average from `DIMENSION_WEIGHTS`, checks hard gate (`brand_voice < BRAND_VOICE_HARD_GATE`), sets `passed_threshold` based on `PASSING_THRESHOLD`.
- **`_call_gemini(prompt, schema) -> dict`**: wrapper with retry logic — exponential backoff (base 2s, max 60s, 3 retries) for `429` rate limits and safety filter blocks. Logs every retry to `log_decision`.
- **Decision logging**: `log_decision("evaluator", ...)` called for: model call initiation, each dimension score, hard gate trigger/pass, final pass/fail determination, any retry.
- **Safety settings**: `BLOCK_ONLY_HIGH` on all categories.
- **Token tracking**: capture `usage_metadata` from response, pass `input_tokens` and `output_tokens` through to `EvaluationResult`.

### 4. Run calibration test (manual validation step)
Not a file write — this is the validation gate. After Tasks 1-3, run the evaluator against all 10 reference ads:

- **Pass criteria**: all 5 "great" ads score weighted average 7+, all 5 "bad" ads score weighted average < 5.
- **If calibration fails**: iterate on rubric prompts in `rubrics.py` — tighten language for inflated scores, add more concrete examples, adjust few-shot anchoring. Re-run until calibration passes.
- **Log calibration results**: `log_decision("calibration", "calibration_run", rationale, {scores_by_ad})` with full score breakdown.
- This is the most important step in the entire project. Do not proceed to Task 5 until calibration passes.

### 5. Implement generation engine
**File:** `src/generate/engine.py`

Only built AFTER calibration passes (Task 4).

`GenerationEngine` class:
- **Constructor**: initializes `genai.Client()`, stores model ID `gemini-2.5-flash`
- **`generate(brief: AdBrief) -> AdCopy`**: calls Gemini 2.5 Flash with structured output. System prompt includes brand voice guidelines and few-shot examples from reference ads. Uses `response_mime_type="application/json"` + `response_json_schema=AdCopy.model_json_schema()` for guaranteed valid output.
- **`_build_prompt(brief: AdBrief) -> str`**: constructs generation prompt with brief details, brand voice guidelines, platform constraints (FB/IG character limits), and few-shot examples.
- **`_call_gemini(prompt, schema) -> dict`**: same retry pattern as evaluator — exponential backoff for rate limits and safety filters. On safety filter block: log the rejection, retry with softened prompt (remove trigger words like "anxiety", "stress" and replace with "concern", "pressure"). Max 3 retries before raising.
- **Decision logging**: `log_decision("generator", ...)` for: generation initiation (with brief summary), model config (temperature, safety settings), safety filter triggers, successful generation.
- **Safety settings**: `BLOCK_ONLY_HIGH` on all harm categories.
- **Token tracking**: capture `usage_metadata`, populate `AdCopy.token_count` fields.
- **Temperature**: 0.7 for creative variety (briefing notes temperature=0.0 for determinism, but generation benefits from some variance — log this decision).

### 6. Wire log_decision into both engines
Both `GenerationEngine` and `EvaluationEngine` call `log_decision(component, action, rationale, context)` at every decision point:
- **Evaluator**: model call start, each dimension score with rationale, hard gate check result, weighted average computation, pass/fail determination, retries
- **Generator**: generation start with brief context, model config selection, safety filter events, successful output summary

This is not a separate file — it's woven into Tasks 3 and 5. Listed here to make the requirement explicit per the briefing: "Decision logging: every function that makes a choice calls log_decision before executing. This is not optional."

### 7. Write unit tests
**File:** `tests/test_evaluate.py`

Tests with mocked Gemini responses (no real API calls):
- `test_iteration_mode_returns_all_dimensions` — mock single-call response with 5 dimension scores, verify EvaluationResult has all 5
- `test_final_mode_makes_separate_calls` — mock 5 separate responses, verify 5 API calls made (one per dimension)
- `test_weighted_average_calculation` — known scores, verify weighted average matches manual calculation
- `test_hard_gate_brand_voice_below_5_fails` — brand_voice=4 with all other dimensions at 9, verify `passed_threshold=False` and `hard_gate_failures=["brand_voice"]`
- `test_hard_gate_brand_voice_at_5_passes` — brand_voice=5 with decent other scores, verify hard gate does not trigger
- `test_passing_threshold` — weighted average exactly at 6.5 passes, 6.4 fails
- `test_retry_on_rate_limit` — mock 429 response then success, verify retry with backoff
- `test_cot_rationale_extracted` — verify rationale field populated from model response
- `test_calibration_great_ad_scores_above_7` — mock evaluator response for a "great" reference ad, verify score > 7
- `test_calibration_bad_ad_scores_below_5` — mock evaluator response for a "bad" reference ad, verify score < 5
- `test_log_decision_called` — verify log_decision called for scoring decisions

**File:** `tests/test_generate.py`

Tests with mocked Gemini responses:
- `test_generate_returns_valid_adcopy` — mock structured JSON response, verify Pydantic validation passes
- `test_structured_output_schema_passed` — verify `response_json_schema` is passed to API call
- `test_safety_filter_retry` — mock safety block then success, verify retry logic fires
- `test_safety_filter_max_retries_raises` — mock 3 consecutive safety blocks, verify exception raised
- `test_log_decision_called_on_generate` — verify log_decision called with component="generator"
- `test_token_count_captured` — verify usage_metadata flows into AdCopy token fields
- `test_prompt_includes_brief_details` — verify brief audience/offer/tone appear in constructed prompt
- `test_prompt_includes_brand_voice_examples` — verify few-shot examples present in prompt

## Acceptance criteria

- Calibration passes: all 5 "great" reference ads score weighted average >= 7.0 in both iteration and final mode; all 5 "bad" reference ads score weighted average < 5.0
- Hard gate enforced: an ad with brand_voice < 5 is rejected regardless of other dimension scores
- Dual eval modes work: `evaluate_iteration` makes 1 API call, `evaluate_final` makes 5 API calls
- CoT ordering: evaluation prompts and response schemas place rationale before score
- Generator produces valid `AdCopy` instances validated by Pydantic
- Safety filter retry: both engines retry with exponential backoff on `BLOCK_ONLY_HIGH` triggers
- Decision logging: every model call and scoring decision produces a `log_decision` entry
- All unit tests pass with mocked responses (no API keys required)
- Correct model IDs: evaluator uses `gemini-2.5-pro`, generator uses `gemini-2.5-flash`

## Verification

1. **Calibration script** (run manually after implementation):
   ```bash
   uv run python -m src.evaluate.calibrate
   ```
   Loads `data/reference_ads/labeled_ads.json`, runs each through both eval modes, prints score table, asserts great >= 7.0 and bad < 5.0.

2. **Unit tests**:
   ```bash
   uv run pytest tests/test_evaluate.py tests/test_generate.py -v
   ```
   All pass with mocked responses, no API keys needed.

3. **Manual spot check**: generate one ad from a sample brief, evaluate it in both modes, inspect decision log entries for completeness.

## Risks and mitigations

- **Score inflation on "bad" reference ads** (briefing Gotcha #4): If the evaluator scores bad ads too generously, tighten rubric language with more explicit failure criteria and add more bad-ad few-shot examples. This is why calibration runs before anything else.
- **Single-call dimension correlation** (briefing Gotcha #5): Iteration mode scores may cluster. This is acceptable for iteration speed — final mode with separate calls is the precision mechanism.
- **Safety filter on anxiety/stress language** (briefing Gotcha #2): Both engines use `BLOCK_ONLY_HIGH` and retry with softened language. Parent-targeted ads are the highest risk.
- **Rubric iteration time**: Calibration may take multiple rubric revisions. Budget for this — it's not wasted time, it's the core deliverable.

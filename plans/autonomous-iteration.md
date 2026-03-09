# Phase 2 Autonomous Iteration

Story: story-584
Agent: architect

## Context

Phase 1 (story-581 bootstrap, story-583 gen+eval) delivers the generator, evaluator, persistence, and decision logger. This story closes the loop: generate an ad, evaluate it, identify the weakest dimension, fix that component, re-evaluate, and repeat until the ad passes or we hit the cycle cap. Then scale it to a batch of 50+ ads from a brief matrix.

The core architectural bet is hybrid iteration: component-level fix first (cheaper, preserves what's already good), full regeneration fallback (when the fix breaks coherence). A procedural state machine (while loop + match/case) drives the whole thing — no LangGraph.

Three subsystems:
1. **Iteration controller** — the state machine that orchestrates gen -> eval -> fix -> re-eval cycles
2. **Self-healing** — detects quality regressions and picks intervention strategies
3. **Batch pipeline** — takes a brief matrix, orchestrates parallel-ish loops within rate limits

Plus weight evolution analytics that tracks whether our initial dimension weights hold up against real data.

## What changes

| File | Change |
|---|---|
| `src/iterate/__init__.py` | Package init, re-export `IterationController` and `SelfHealer` |
| `src/iterate/controller.py` | `IterationController` class: procedural state machine that runs gen -> eval -> component fix -> coherence check -> re-eval cycles. Max 3 cycles. Falls back to full regen if coherence drops. Always includes original brief in every iteration prompt. Calls `log_decision` at every branch (component fix vs full regen, pass vs fail vs retry, force-fail after max cycles). |
| `src/iterate/healing.py` | `SelfHealer` class: maintains quality ratchet (running weighted average that only goes up). Detects regression (new score < running avg). Selects intervention strategy per dimension (e.g., clarity -> simplify language, CTA -> strengthen action verbs, brand voice -> inject brand guidelines, value prop -> sharpen differentiation, emotional resonance -> add parent/student specificity). Returns `InterventionPlan` with targeted feedback prompt. |
| `src/analytics/__init__.py` | Package init |
| `src/analytics/weights.py` | `WeightEvolver` class: after N ads (configurable, default 50), calculates Pearson correlation between each dimension score and overall human-perceived quality proxy (weighted average). Compares observed correlations to initial weight assignments. Logs findings via `log_decision`. Produces a weight recommendation dict. |
| `src/pipeline/__init__.py` | Package init |
| `src/pipeline/main.py` | `BatchPipeline` class + `main()` entry point. Takes a brief matrix (audience x offer x tone from `seed_briefs.py`). Orchestrates gen -> eval -> iterate loop per brief. Rate limiter with exponential backoff (10-15 RPM, token bucket). Progress tracking to stdout. Calls `log_decision` for pipeline-level decisions (batch start, brief skipped due to rate limit, batch complete with summary stats). Quality snapshot persistence after each batch. CLI entry point: `python -m src.pipeline.main`. |
| `tests/test_iterate.py` | Unit tests for `IterationController` and `SelfHealer` with mocked generator/evaluator. Tests: component fix path (weak dimension identified, fix applied, score improves), full regen fallback (coherence check fails after component fix), max cycle force-fail, quality ratchet (running avg only goes up), regression detection triggers intervention, original brief always present in iteration prompts. |
| `tests/test_pipeline.py` | Integration tests for `BatchPipeline` with mocked API calls. Tests: full loop (brief -> gen -> eval -> iterate -> pass), rate limiter respects RPM cap, force-fail after max cycles doesn't crash pipeline, progress tracking counts, quality snapshot written to DB, decision log entries present for every branch. |

## Read-only context

- `presearch/autonomous-ad-engine.md` — briefing with iteration architecture, gotchas, rate limits, cost model
- `src/generate/engine.py` — generator (created by story-583): takes AdBrief, returns AdCopy
- `src/evaluate/engine.py` — evaluator (created by story-583): takes AdCopy, returns EvaluationResult
- `src/db/queries.py` — persistence helpers for ads, evaluations, iterations, decisions
- `src/db/init_db.py` — database initialization
- `src/decisions/logger.py` — `log_decision(component, action, rationale, context)` function
- `src/models/iteration.py` — `IterationRecord` Pydantic model
- `src/models/ad.py` — `AdCopy` Pydantic model
- `src/models/evaluation.py` — `EvaluationResult`, `DimensionScore` Pydantic models
- `src/models/brief.py` — `AdBrief` Pydantic model
- `src/briefs/seed_briefs.py` — seed brief configurations and brief matrix generation

## Tasks

1. **Implement `src/iterate/controller.py` — the iteration state machine.** Define `IterationController` class that takes a generator, evaluator, healer, and db connection as constructor args. Core method: `iterate(brief: AdBrief) -> tuple[AdCopy | None, list[IterationRecord]]`. The loop:
   - Generate ad from brief via generator
   - Evaluate via evaluator
   - If passes threshold: `log_decision("iterate", "accept", ...)`, return ad
   - If cycle >= 3: `log_decision("iterate", "force_fail", ...)`, return None
   - Identify weakest dimension from evaluation scores
   - Attempt component-level fix: build a targeted feedback prompt that includes the **original brief** (not just the feedback — context window drift gotcha), the current ad, the weak dimension, and the healer's intervention strategy
   - Re-generate only the weak component via generator (pass `component_fix=True, target_dimension=weak_dimension` or equivalent)
   - Run coherence check: re-evaluate the patched ad. If weighted_average drops by more than 0.5 from pre-fix score, coherence has broken
   - If coherence fails: `log_decision("iterate", "coherence_fail_full_regen", ...)`, fall back to full regeneration from the original brief with accumulated feedback context
   - If coherence passes: continue loop with the new ad
   - Persist each `IterationRecord` to DB via `queries.py`
   - `log_decision` at every branch: accept, reject+retry, component_fix, coherence_pass, coherence_fail, full_regen, force_fail

   Use `match/case` on an enum or string state (`generate`, `evaluate`, `fix`, `coherence_check`, `regen`, `accept`, `fail`) for the state machine. NOT a graph framework. (see briefing ## Architecture > Procedural State Machine and ## Gotchas #6 for context drift)

2. **Implement `src/iterate/healing.py` — self-healing and quality ratchet.** Define `SelfHealer` class:
   - `__init__`: initialize running weighted average to 0.0
   - `update_ratchet(score: float)`: set running_avg = max(running_avg, score). The ratchet only goes up.
   - `detect_regression(current_score: float) -> bool`: returns True if current_score < running_avg
   - `diagnose(evaluation: EvaluationResult) -> str`: identify the weakest dimension (lowest score, with tie-breaking preferring hard-gated dimensions)
   - `select_intervention(dimension: str) -> InterventionPlan`: return a targeted feedback strategy. Strategies per dimension:
     - `clarity`: "Simplify sentence structure. Remove jargon. One idea per sentence."
     - `value_proposition`: "Sharpen the specific benefit. What does the student/parent get that competitors don't offer?"
     - `cta_strength`: "Use a stronger action verb. Create urgency without being pushy. Be specific about next step."
     - `brand_voice`: "Match Varsity Tutors' tone: confident, warm, expert but approachable. Avoid corporate stiffness."
     - `emotional_resonance`: "Connect to the parent's anxiety about their child's future or the student's desire to succeed. Be specific, not generic."
   - `build_feedback_prompt(brief: AdBrief, ad: AdCopy, evaluation: EvaluationResult) -> str`: combines original brief, current ad, weak dimension diagnosis, and intervention strategy into a structured feedback prompt for the generator
   - Log every intervention selection via `log_decision`

   Define `InterventionPlan` as a simple dataclass: dimension, strategy_text, severity (minor/major based on how far below threshold).

3. **Implement `src/analytics/weights.py` — weight evolution tracking.** Define `WeightEvolver` class:
   - `__init__(db_conn, min_sample_size: int = 50)`: store connection, load initial weights from evaluator config
   - `calculate_correlations() -> dict[str, float]`: query all evaluation scores from DB, compute Pearson correlation between each dimension's scores and the weighted averages. Return {dimension: correlation_coefficient}.
   - `compare_to_initial_weights(correlations: dict) -> dict`: compare observed correlations to initial weight assignments. Flag dimensions where weight and correlation diverge significantly (e.g., a dimension weighted at 0.25 but correlating at 0.05).
   - `recommend_weights(correlations: dict) -> dict[str, float]`: normalize correlations to sum to 1.0 as recommended weights. This is advisory — logged via `log_decision` but not auto-applied.
   - `evolve() -> dict`: main method that runs the full analysis if sample size is met. Logs decision with rationale whether weights should change or current weights are validated. Returns the analysis results.
   - All math uses `statistics` stdlib (pearson via manual calc or `statistics.correlation` in 3.12+). No numpy/scipy dependency.

4. **Implement `src/pipeline/main.py` — batch pipeline with rate limiting.** Define `BatchPipeline` class:
   - `__init__(db_path: str, rpm_limit: int = 12, rpd_limit: int = 800)`: init DB, create generator, evaluator, healer, controller instances. Set up rate limiter.
   - Rate limiter: token bucket algorithm. Track timestamps of recent API calls. Before each call, check if we'd exceed RPM. If so, sleep with exponential backoff (start 1s, max 60s, jitter). Also track daily call count against RPD limit. `log_decision` when throttling occurs.
   - `generate_brief_matrix() -> list[AdBrief]`: call `seed_briefs.py` to produce the audience x offer x tone cross-product. Log the matrix size.
   - `run(briefs: list[AdBrief] | None = None) -> BatchResult`: iterate over briefs, call `controller.iterate(brief)` for each. Track: passed, failed, total_cycles, total_tokens, total_cost. Print progress (`[12/60] Processing: parent_focused x sat_prep x urgent`). After batch completes, persist a quality_snapshot row. Log batch summary via `log_decision`.
   - `BatchResult` dataclass: total_briefs, passed, failed, avg_score, total_cycles, total_tokens, total_cost_usd, duration_seconds.
   - `main()` function: parse optional CLI args (--briefs N to limit brief count, --rpm N to override rate limit), run pipeline, print summary table.
   - Entry point: `if __name__ == "__main__"` block + `python -m src.pipeline.main` compatibility.
   - Wrap each brief's iteration loop in try/except so one failure doesn't crash the batch. Log the exception via `log_decision("pipeline", "brief_error", str(e), ...)`.

5. **Wire `log_decision` into every branch.** This is not a separate implementation step — it's a constraint on tasks 1-4. Every `if/elif/else`, every `match/case` arm, every fallback path must call `log_decision` before executing. The decision log is a graded deliverable (20% of rubric). Audit each file before marking complete: count conditional branches, count `log_decision` calls, they should be 1:1.

6. **Write `tests/test_iterate.py`.** Mock the generator and evaluator to return controlled responses. Test cases:
   - `test_accept_on_first_pass`: generator produces passing ad, controller returns it after 1 cycle with 0 iterations
   - `test_component_fix_improves_score`: first eval fails on clarity, component fix raises clarity, second eval passes
   - `test_coherence_fail_triggers_full_regen`: component fix drops weighted_avg by >0.5, controller falls back to full regen
   - `test_max_cycles_force_fail`: 3 consecutive failures -> controller returns None
   - `test_quality_ratchet_only_goes_up`: healer.update_ratchet with [5.0, 7.0, 6.0] -> running_avg stays 7.0
   - `test_regression_detected`: current score below running avg triggers regression flag
   - `test_intervention_selection_per_dimension`: each of 5 dimensions returns a distinct intervention strategy
   - `test_original_brief_in_every_prompt`: verify the feedback prompt always contains the original brief fields, not just iteration feedback
   - `test_decision_logging_at_every_branch`: mock log_decision, run a multi-cycle iteration, verify log_decision called for each state transition

7. **Write `tests/test_pipeline.py`.** Mock API calls end-to-end. Test cases:
   - `test_full_pipeline_loop`: single brief -> gen -> eval -> pass (happy path)
   - `test_pipeline_with_iteration`: brief that needs 2 cycles to pass
   - `test_rate_limiter_throttles`: fire 15 requests rapidly, verify delays inserted to stay under 12 RPM
   - `test_force_fail_doesnt_crash_batch`: one brief force-fails, next brief still runs
   - `test_quality_snapshot_persisted`: after batch, query DB for quality_snapshot row
   - `test_batch_result_accuracy`: verify passed/failed/total counts match actual outcomes
   - `test_decision_log_coverage`: verify log_decision called for batch_start, each brief's iteration decisions, and batch_complete
   - `test_brief_error_handled`: inject exception in one brief, verify pipeline continues and logs the error

## Acceptance criteria

- `IterationController.iterate(brief)` runs the full gen -> eval -> fix -> re-eval loop and returns either a passing `AdCopy` or `None` after max 3 cycles
- Component-level fix is attempted before full regeneration on every failing cycle
- Coherence check fires after every component fix; weighted_avg drop > 0.5 triggers full regen fallback
- Original brief is included in every iteration prompt (verified by test)
- Quality ratchet: `SelfHealer.running_avg` never decreases
- `SelfHealer.select_intervention` returns dimension-specific strategies for all 5 dimensions
- `WeightEvolver.evolve()` calculates correlations and logs weight recommendations after sufficient sample size
- `BatchPipeline.run()` processes a brief matrix sequentially with rate limiting (configurable RPM/RPD)
- Rate limiter enforces 10-15 RPM with exponential backoff and jitter
- One brief's failure (exception or force-fail) does not crash the batch
- `BatchResult` accurately reflects pass/fail/cycle counts
- Quality snapshot persisted to DB after each batch
- `log_decision` called at every conditional branch in controller, healer, and pipeline
- `uv run pytest tests/test_iterate.py tests/test_pipeline.py -v` passes all tests
- `uv run ruff check src/iterate/ src/analytics/ src/pipeline/ tests/test_iterate.py tests/test_pipeline.py` passes clean

## Verification

- Run `uv run ruff check src/iterate/ src/analytics/ src/pipeline/` — zero violations
- Run `uv run pytest tests/test_iterate.py -v` — all 9 test cases pass
- Run `uv run pytest tests/test_pipeline.py -v` — all 8 test cases pass
- Run `uv run python -c "from src.iterate.controller import IterationController; from src.iterate.healing import SelfHealer; from src.analytics.weights import WeightEvolver; from src.pipeline.main import BatchPipeline; print('All phase 2 imports OK')"` — confirms no import errors
- Grep `src/iterate/controller.py` for `log_decision` — count should be >= 7 (accept, reject+retry, component_fix, coherence_pass, coherence_fail, full_regen, force_fail)
- Grep `src/pipeline/main.py` for `log_decision` — count should be >= 4 (batch_start, throttle, brief_error, batch_complete)
- Run `uv run python -m src.pipeline.main --briefs 2 --rpm 5` with real API keys — produces 2 ads through the full loop (manual smoke test, not CI)

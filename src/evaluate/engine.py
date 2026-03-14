"""Evaluation engine for AdCraft.

Cross-model LLM-as-judge using Gemini 2.5 Pro or Claude Sonnet. Two modes:
- evaluate_iteration: single call, all 5 dimensions (fast, for iteration loops)
- evaluate_final: 5 separate calls, one per dimension (precise, for final scoring)

Both modes enforce CoT (rationale before score), apply weighted aggregation,
and enforce the brand_voice hard gate.
"""

from __future__ import annotations

import json
import os

import litellm
from google import genai
from google.genai import types
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.decisions.logger import log_decision
from src.evaluate.rubrics import (
    ALL_DIMENSIONS_SCHEMA,
    BRAND_VOICE_HARD_GATE,
    DIMENSION_WEIGHTS,
    DIMENSIONS,
    PASSING_THRESHOLD,
    SINGLE_DIMENSION_SCHEMA,
    build_all_dimensions_prompt,
    build_single_dimension_prompt,
)
from src.evaluate.utils import claude_retry
from src.models.ad import AdCopy
from src.models.evaluation import DimensionScore, EvaluationResult

EVALUATOR_MODEL = "gemini-2.5-pro"
CLAUDE_MODEL = "claude-sonnet-4-6"

SAFETY_SETTINGS = [
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_ONLY_HIGH"),
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_ONLY_HIGH"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_ONLY_HIGH"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_ONLY_HIGH"),
]


class EvaluationEngine:
    """Scores ad copy across 5 dimensions using Gemini 2.5 Pro or Claude."""

    def __init__(
        self,
        client: genai.Client | None = None,
        *,
        model_id: str | None = None,
    ) -> None:
        self._model = model_id or EVALUATOR_MODEL

        if self._is_claude:
            self._client = None
        elif client is not None:
            self._client = client
        else:
            api_key = os.environ.get("GEMINI_API_KEY", "")
            self._client = genai.Client(api_key=api_key)

        log_decision(
            "evaluator",
            "model_selected",
            f"Evaluator initialized with model '{self._model}'",
            {"model": self._model, "family": "claude" if self._is_claude else "gemini"},
        )

    @property
    def _is_claude(self) -> bool:
        return self._model.startswith("claude")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_iteration(self, ad_copy: AdCopy) -> EvaluationResult:
        """Score all 5 dimensions in a single API call (fast mode)."""
        log_decision(
            "evaluator",
            "evaluate_iteration_start",
            f"Evaluating ad '{ad_copy.id}' in iteration mode (single call, all dimensions)",
            {"ad_id": ad_copy.id, "mode": "iteration"},
        )

        prompt = build_all_dimensions_prompt(
            primary_text=ad_copy.primary_text,
            headline=ad_copy.headline,
            description=ad_copy.description,
            cta_button=ad_copy.cta_button,
        )

        raw, token_count = (
            self._call_claude(prompt, ALL_DIMENSIONS_SCHEMA)
            if self._is_claude
            else self._call_gemini(prompt, ALL_DIMENSIONS_SCHEMA)
        )

        scores: list[DimensionScore] = []
        for dim in DIMENSIONS:
            dim_data = raw[dim]
            score = DimensionScore(
                dimension=dim,
                score=float(dim_data["score"]),
                rationale=dim_data["rationale"],
                confidence=float(dim_data.get("confidence", 1.0)),
            )
            scores.append(score)
            log_decision(
                "evaluator",
                "dimension_score",
                f"{dim}={score.score:.1f} — {score.rationale[:100]}",
                {"ad_id": ad_copy.id, "dimension": dim, "score": score.score, "mode": "iteration"},
            )

        return self._compute_result(ad_copy.id, scores, token_count, mode="iteration")

    def evaluate_final(self, ad_copy: AdCopy) -> EvaluationResult:
        """Score each dimension in a separate API call (precise mode)."""
        log_decision(
            "evaluator",
            "evaluate_final_start",
            f"Evaluating ad '{ad_copy.id}' in final mode (separate call per dimension)",
            {"ad_id": ad_copy.id, "mode": "final"},
        )

        scores: list[DimensionScore] = []
        total_tokens = 0

        for dim in DIMENSIONS:
            prompt = build_single_dimension_prompt(
                dimension=dim,
                primary_text=ad_copy.primary_text,
                headline=ad_copy.headline,
                description=ad_copy.description,
                cta_button=ad_copy.cta_button,
            )

            raw, token_count = (
                self._call_claude(prompt, SINGLE_DIMENSION_SCHEMA)
                if self._is_claude
                else self._call_gemini(prompt, SINGLE_DIMENSION_SCHEMA)
            )
            total_tokens += token_count

            score = DimensionScore(
                dimension=dim,
                score=float(raw["score"]),
                rationale=raw["rationale"],
                confidence=float(raw.get("confidence", 1.0)),
            )
            scores.append(score)
            log_decision(
                "evaluator",
                "dimension_score",
                f"{dim}={score.score:.1f} — {score.rationale[:100]}",
                {"ad_id": ad_copy.id, "dimension": dim, "score": score.score, "mode": "final"},
            )

        return self._compute_result(ad_copy.id, scores, total_tokens, mode="final")

    def evaluate_visual(self, image, ad_copy, eval_mode="iteration"):
        """Evaluate visual content. Gemini-only (multimodal required)."""
        if self._is_claude:
            raise ValueError(
                f"Visual evaluation requires Gemini multimodal. "
                f"Current model '{self._model}' is not supported for visual eval."
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @claude_retry
    def _call_claude(self, prompt: str, schema: dict) -> tuple[dict, int]:
        """Call Claude via litellm.completion with JSON response_format."""
        schema_keys = list(schema.get("properties", {}).keys())
        schema_hint = f"\n\nYour response MUST be a JSON object with these keys: {schema_keys}"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert advertising evaluator. "
                    "Respond only with valid JSON matching the requested schema."
                ),
            },
            {
                "role": "user",
                "content": prompt + schema_hint,
            },
        ]

        try:
            response = litellm.completion(
                model=self._model,
                messages=messages,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            log_decision(
                "evaluator",
                "api_retry",
                f"Claude call failed, will retry: {type(exc).__name__}: {exc}",
                {"model": self._model, "error": str(exc)},
            )
            raise

        token_count = response.usage.total_tokens
        parsed = json.loads(response.choices[0].message.content)

        missing = [k for k in schema.get("required", []) if k not in parsed]
        if missing:
            raise ValueError(f"Claude response missing required keys: {missing}")

        return parsed, token_count

    def _compute_result(
        self,
        ad_id: str,
        scores: list[DimensionScore],
        token_count: int,
        *,
        mode: str,
    ) -> EvaluationResult:
        """Apply weighted average, hard gate, and threshold check."""
        weighted_avg = round(
            sum(s.score * DIMENSION_WEIGHTS[s.dimension] for s in scores),
            4,
        )

        hard_gate_failures: list[str] = []
        for s in scores:
            if s.dimension == "brand_voice" and s.score < BRAND_VOICE_HARD_GATE:
                hard_gate_failures.append("brand_voice")
                log_decision(
                    "evaluator",
                    "hard_gate_triggered",
                    f"brand_voice score {s.score:.1f} < hard gate {BRAND_VOICE_HARD_GATE}",
                    {"ad_id": ad_id, "score": s.score, "gate": BRAND_VOICE_HARD_GATE},
                )

        passed = weighted_avg >= PASSING_THRESHOLD and len(hard_gate_failures) == 0

        log_decision(
            "evaluator",
            "evaluation_complete",
            f"Ad '{ad_id}' {'PASSED' if passed else 'FAILED'}: "
            f"weighted_avg={weighted_avg:.2f}, hard_gate_failures={hard_gate_failures}",
            {
                "ad_id": ad_id,
                "weighted_average": weighted_avg,
                "passed": passed,
                "hard_gate_failures": hard_gate_failures,
                "mode": mode,
            },
        )

        return EvaluationResult(
            ad_id=ad_id,
            scores=scores,
            weighted_average=weighted_avg,
            passed_threshold=passed,
            hard_gate_failures=hard_gate_failures,
            evaluator_model=self._model,
            token_count=token_count,
        )

    def _call_gemini(self, prompt: str, schema: dict) -> tuple[dict, int]:
        """Call Gemini with structured output and retry logic.

        Returns (parsed_response_dict, total_token_count).
        """
        return self._call_gemini_with_retry(prompt, schema)

    @retry(
        retry=retry_if_exception_type((Exception,)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    )
    def _call_gemini_with_retry(self, prompt: str, schema: dict) -> tuple[dict, int]:
        """Inner call with tenacity retry decorator."""
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=schema,
                    safety_settings=SAFETY_SETTINGS,
                ),
            )
        except Exception as exc:
            log_decision(
                "evaluator",
                "api_retry",
                f"Gemini call failed, will retry: {type(exc).__name__}: {exc}",
                {"model": self._model, "error": str(exc)},
            )
            raise

        token_count = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            meta = response.usage_metadata
            token_count = getattr(meta, "total_token_count", 0) or 0

        parsed = json.loads(response.text)
        return parsed, token_count

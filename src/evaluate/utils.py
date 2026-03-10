"""Shared Gemini retry utilities for AdCraft.

Centralizes retry logic, retriable error detection, and safety settings
used by both evaluation and generation modules. Single source of truth
for transient HTTP error codes and tenacity configuration.
"""

from __future__ import annotations

from google.genai import types
from google.genai.errors import APIError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

# Transient HTTP codes worth retrying — mirrors the SDK's own _RETRY_HTTP_STATUS_CODES
RETRIABLE_STATUS_CODES: tuple[int, ...] = (408, 429, 500, 502, 503)


def is_retriable(exc: BaseException) -> bool:
    """Return True for transient API errors that may succeed on retry."""
    return isinstance(exc, APIError) and exc.code in RETRIABLE_STATUS_CODES


SAFETY_SETTINGS = [
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_ONLY_HIGH"),
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_ONLY_HIGH"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_ONLY_HIGH"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_ONLY_HIGH"),
]

gemini_retry = retry(
    retry=retry_if_exception(is_retriable),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    reraise=True,
)

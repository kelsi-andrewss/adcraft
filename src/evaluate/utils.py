"""Shared retry decorators for the evaluation engine."""

from __future__ import annotations

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

try:
    from litellm.exceptions import (
        RateLimitError,
        ServiceUnavailableError,
    )

    _LITELLM_RETRYABLE = (RateLimitError, ServiceUnavailableError)
except ImportError:
    _LITELLM_RETRYABLE = (Exception,)


claude_retry = retry(
    retry=retry_if_exception_type(_LITELLM_RETRYABLE),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    reraise=True,
)

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from mas_taxonomy.config import get_settings

VALID_PROVIDERS = ("openai", "anthropic", "gemini", "vertexai")

COST_PER_MILLION: dict[str, dict[str, float]] = {
    "gpt-5-mini": {"input": 0.25, "output": 2.0},
    "gpt-5.4": {"input": 2.50, "output": 15.0},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.0},
    "gpt-5.4-2026-03-05": {"input": 2.50, "output": 15.0},
}


def resolve_provider_and_model() -> tuple[str, str]:
    """Determine which provider/model to use for this run.

    Walks through ``LLM_PROVIDER_PRIORITY`` and returns the first
    provider whose API key is configured.  Falls back to ``"openai"``
    if the priority list is empty or contains no valid entries.

    Returns:
        (provider_name, model_id)

    Raises:
        RuntimeError: if no configured provider has a valid API key.
    """
    s = get_settings()

    raw = s.llm_provider_priority.strip()
    candidates = [p.strip().lower() for p in raw.split(",") if p.strip()] if raw else []
    candidates = [p for p in candidates if p in VALID_PROVIDERS]
    if not candidates:
        candidates = list(VALID_PROVIDERS)

    key_map = {
        "openai": s.openai_api_key,
        "anthropic": s.anthropic_api_key,
        "gemini": s.gemini_api_key,
        "vertexai": s.vertex_project,  # project ID acts as the "key" signal
    }
    model_map = {
        "openai": s.default_model_openai,
        "anthropic": s.default_model_anthropic,
        "gemini": s.default_model_gemini,
        "vertexai": s.default_model_vertexai,
    }

    for provider in candidates:
        if key_map.get(provider, "").strip():
            return provider, model_map[provider]

    raise RuntimeError(
        f"No valid API key found for any provider in priority list {candidates}. "
        "Set at least one of OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, or VERTEX_PROJECT."
    )


def _create_llm(
    provider: str,
    model: str,
    temperature: float = 0.0,
) -> ChatOpenAI | ChatAnthropic | ChatGoogleGenerativeAI:
    """Factory: create the right LangChain chat model for *provider*."""
    s = get_settings()

    if provider == "openai":
        return ChatOpenAI(
            model=model,
            api_key=s.openai_api_key,
            temperature=temperature,
        )
    if provider == "anthropic":
        return ChatAnthropic(
            model=model,
            api_key=s.anthropic_api_key,
            temperature=temperature,
        )
    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=s.gemini_api_key,
            temperature=temperature,
        )
    if provider == "vertexai":
        import os
        if s.vertex_credentials_file.strip():
            os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", s.vertex_credentials_file.strip())
        return ChatGoogleGenerativeAI(
            model=model,
            project=s.vertex_project,
            location=s.vertex_location,
            vertexai=True,
            temperature=temperature,
        )
    raise ValueError(f"Unknown LLM provider: {provider!r}")


# ---------------------------------------------------------------------------
# Token extraction – each wrapper stores usage differently
# ---------------------------------------------------------------------------

def _extract_token_usage(resp: Any) -> dict[str, int]:
    """Extract token usage from any supported LangChain chat-model response."""
    meta: dict = getattr(resp, "response_metadata", {}) or {}
    usage_info: dict = getattr(resp, "usage_metadata", {}) or {}

    prompt = 0
    completion = 0

    # --- OpenAI style (token_usage dict inside response_metadata) ---
    oai = meta.get("token_usage") or meta.get("usage") or {}
    if oai:
        prompt = oai.get("prompt_tokens", 0) or oai.get("input_tokens", 0)
        completion = oai.get("completion_tokens", 0) or oai.get("output_tokens", 0)

    # --- Anthropic style (usage dict inside response_metadata) ---
    if not (prompt or completion):
        anth = meta.get("usage", {})
        prompt = anth.get("input_tokens", 0)
        completion = anth.get("output_tokens", 0)

    # --- LangChain universal usage_metadata (works for Gemini & others) ---
    if not (prompt or completion) and usage_info:
        prompt = usage_info.get("input_tokens", 0) or usage_info.get("prompt_tokens", 0)
        completion = usage_info.get("output_tokens", 0) or usage_info.get("completion_tokens", 0)

    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

_retry_log = logging.getLogger("mas_taxonomy.llm_retry")


def log_to_file_handlers_only(logger: logging.Logger | None, level: int, msg: str, *args: Any) -> None:
    """
    Emit one log record only to ``FileHandler``s on *logger* (skip ``StreamHandler`` / console).
    Use for verbose per-attempt LLM / token lines so ``app.log`` stays complete while the CLI stays quiet.
    """
    if logger is None or not logger.handlers:
        return
    message = msg % args if args else msg
    record = logger.makeRecord(
        logger.name,
        level,
        __file__,
        0,
        message,
        (),
        None,
    )
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and record.levelno >= h.level:
            h.emit(record)


def _log_retry_warning(run_log: logging.Logger | None, msg: str, *args: Any) -> None:
    """Failures: console + file via the run logger (or stderr-only fallback)."""
    if run_log is not None:
        run_log.warning(msg, *args)
    else:
        _retry_log.warning(msg, *args)


def _log_retry_info(run_log: logging.Logger | None, msg: str, *args: Any) -> None:
    log_to_file_handlers_only(run_log, logging.INFO, msg, *args)


class LLMRetryExhausted(Exception):
    """All invoke attempts failed; ``token_usage`` sums every attempt that returned a billable response."""

    def __init__(self, message: str, *, token_usage: dict[str, int]):
        super().__init__(message)
        self.token_usage = token_usage


def call_llm_with_retry(
    invoke_fn: Callable[[], Any],
    max_retries: int = 2,
    backoff_seconds: float = 1.5,
    *,
    log: logging.Logger | None = None,
) -> tuple[Any, dict[str, int]]:
    """
    Calls an LLM invoke function with retry and exponential backoff.

    Handles two response shapes transparently:
      * Regular ``AIMessage`` (from plain ``llm.invoke()``)
      * Dict ``{"raw": AIMessage, "parsed": BaseModel, "parsing_error": …}``
        (from ``llm.with_structured_output(…, include_raw=True)``)

    Logs each failed attempt at WARNING on the run logger (console + ``app.log``).
    Logs each successful response / parse detail at INFO **only to ``app.log``** (file handlers), not the CLI.

    On success, the returned token dict is the **sum** over all attempts that returned billable usage
    (including failed parses that still had ``raw``).

    Raises:
        LLMRetryExhausted: If every attempt fails; carries accumulated ``token_usage``.
    """
    max_attempts = max_retries + 1
    accumulated: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for attempt in range(max_attempts):
        attempt_n = attempt + 1
        try:
            resp = invoke_fn()
        except Exception as e:
            _log_retry_warning(
                log,
                "LLM attempt %d/%d invoke failed: %s: %s",
                attempt_n,
                max_attempts,
                type(e).__name__,
                e,
            )
            if attempt < max_retries:
                time.sleep(backoff_seconds * (attempt + 1))
            else:
                raise LLMRetryExhausted(
                    f"LLM invoke failed after {max_attempts} attempts: {e}",
                    token_usage=dict(accumulated),
                ) from e
            continue

        raw_resp = resp["raw"] if isinstance(resp, dict) and "raw" in resp else resp
        usage = _extract_token_usage(raw_resp)
        accumulated = _accumulate_token_usage(accumulated, usage)

        _log_retry_info(
            log,
            "LLM attempt %d/%d: response received — this call tokens: prompt=%d, completion=%d, total=%d; "
            "cumulative so far: prompt=%d, completion=%d, total=%d",
            attempt_n,
            max_attempts,
            usage["prompt_tokens"],
            usage["completion_tokens"],
            usage["total_tokens"],
            accumulated["prompt_tokens"],
            accumulated["completion_tokens"],
            accumulated["total_tokens"],
        )

        if isinstance(resp, dict) and "raw" in resp and "parsed" in resp:
            parsing_error = resp.get("parsing_error")
            if parsing_error is not None:
                err = parsing_error if isinstance(parsing_error, Exception) else ValueError(str(parsing_error))
                # Log the raw content Claude returned so we can diagnose what fields were/weren't filled
                if log:
                    try:
                        raw_content = resp["raw"].content
                        _log_retry_warning(
                            log,
                            "LLM attempt %d/%d raw response content: %s",
                            attempt_n,
                            max_attempts,
                            raw_content,
                        )
                    except Exception:
                        pass
                _log_retry_warning(
                    log,
                    "LLM attempt %d/%d structured output failed: %s: %s",
                    attempt_n,
                    max_attempts,
                    type(err).__name__,
                    err,
                )
                if attempt < max_retries:
                    time.sleep(backoff_seconds * (attempt + 1))
                    continue
                raise LLMRetryExhausted(
                    f"Structured output failed after {max_attempts} attempts: {err}",
                    token_usage=dict(accumulated),
                ) from err
            if resp["parsed"] is None:
                ve = ValueError("Structured output parsing returned None")
                _log_retry_warning(
                    log,
                    "LLM attempt %d/%d structured output returned parsed=None",
                    attempt_n,
                    max_attempts,
                )
                if attempt < max_retries:
                    time.sleep(backoff_seconds * (attempt + 1))
                    continue
                raise LLMRetryExhausted(
                    f"Structured output returned None after {max_attempts} attempts",
                    token_usage=dict(accumulated),
                ) from ve
            _log_retry_info(
                log,
                "LLM attempt %d/%d: structured output OK — returning (cumulative tokens: prompt=%d, completion=%d, total=%d)",
                attempt_n,
                max_attempts,
                accumulated["prompt_tokens"],
                accumulated["completion_tokens"],
                accumulated["total_tokens"],
            )
            return resp["parsed"], dict(accumulated)

        _log_retry_info(
            log,
            "LLM attempt %d/%d: non-structured response OK — returning (cumulative tokens: prompt=%d, completion=%d, total=%d)",
            attempt_n,
            max_attempts,
            accumulated["prompt_tokens"],
            accumulated["completion_tokens"],
            accumulated["total_tokens"],
        )
        return resp, dict(accumulated)

    raise RuntimeError("Unexpected retry loop exit")


# ---------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------

def _accumulate_token_usage(current: dict[str, int], new: dict[str, int]) -> dict[str, int]:
    """Accumulate token usage across multiple LLM calls."""
    return {
        "prompt_tokens": current.get("prompt_tokens", 0) + new.get("prompt_tokens", 0),
        "completion_tokens": current.get("completion_tokens", 0) + new.get("completion_tokens", 0),
        "total_tokens": current.get("total_tokens", 0) + new.get("total_tokens", 0),
    }

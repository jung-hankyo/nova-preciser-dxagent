"""Thin wrapper around LangChain chat models.

All LLM calls in dxagent go through `LLM.structured` (returns a Pydantic object) or
`LLM.text` (returns a string). This keeps provider details in one place and lets
tests swap in `ScriptedLLM` without touching the graph.

Model specs use LangChain's "provider:model" form, optionally followed by "?option=value&...":
    anthropic:claude-sonnet-5-5
    openai:gpt-5-mini?effort=minimal          OpenAI reasoning model, reasoning effort minimal|low|medium|high
    deepseek:deepseek-v4-pro?thinking=low     DeepSeek, thinking off|low|high|max (default off)
    deepseek:deepseek-flash

DeepSeek (v0.4.2) is reached through its OpenAI-compatible endpoint (DEEPSEEK_BASE_URL, default
https://api.deepseek.com) with the langchain-openai client, so no extra package is needed. Thinking mode is switched
with `extra_body={"thinking": {"type": ...}}`; in thinking mode the API ignores temperature.

Structured output uses function calling for OpenAI and DeepSeek (supported by both, including DeepSeek thinking mode,
and tolerant of optional fields), and LangChain's default for Anthropic. DX_STRUCTURED_METHOD overrides it.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Callable, Protocol, TypeVar

from pydantic import BaseModel

from dxagent import usage  # noqa: E402

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMLike(Protocol):
    name: str

    def structured(self, schema: type[T], system: str, user: str) -> T: ...

    def text(self, system: str, user: str) -> str: ...


OPENAI_REASONING = re.compile(r"^(gpt-5|o\d)", re.I)
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def parse_spec(spec: str) -> tuple[str, str, dict[str, str]]:
    """'deepseek:deepseek-v4-pro?thinking=low' -> ('deepseek', 'deepseek-v4-pro', {'thinking': 'low'})."""
    if ":" not in spec:
        raise ValueError(f"Model spec must be 'provider:model', got {spec!r}")
    provider, rest = spec.split(":", 1)
    model, _, query = rest.partition("?")
    opts = dict(kv.split("=", 1) for kv in query.split("&") if "=" in kv)
    return provider.strip().lower(), model.strip(), opts


def build_chat_model(spec: str, temperature: float | None = None, max_tokens: int = 4096, http_client: Any = None):
    """(LangChain chat model, structured-output method or None for the provider default)."""
    provider, model, opts = parse_spec(spec)
    kwargs: dict[str, Any] = {"max_tokens": max_tokens}
    method = os.getenv("DX_STRUCTURED_METHOD") or None
    if provider == "deepseek":
        from langchain_openai import ChatOpenAI

        thinking = opts.get("thinking", "off").lower()
        extra: dict[str, Any] = {"thinking": {"type": "disabled" if thinking in ("off", "none", "disabled") else "enabled"}}
        if thinking not in ("off", "none", "disabled"):
            kwargs["reasoning_effort"] = thinking if thinking in ("low", "high", "max") else "high"
            kwargs["max_tokens"] = max(max_tokens, 16384)  # reasoning tokens count against the output budget
        elif temperature is not None:
            kwargs["temperature"] = temperature
        key = os.getenv("DEEPSEEK_API_KEY")
        if http_client is not None:
            kwargs["http_client"] = http_client
        chat = ChatOpenAI(model=model, base_url=os.getenv("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL), api_key=key,
                          extra_body=extra, **kwargs)
        return chat, method or "function_calling"
    from langchain.chat_models import init_chat_model

    if provider == "openai":
        if OPENAI_REASONING.match(model):
            # reasoning models reject a non-default temperature; their reasoning tokens use the output budget
            kwargs["reasoning_effort"] = opts.get("effort", "low")
            kwargs["max_tokens"] = max(max_tokens, 8192)
        elif temperature is not None:
            kwargs["temperature"] = temperature
        if http_client is not None:
            kwargs["http_client"] = http_client
        return init_chat_model(model, model_provider="openai", **kwargs), method or "function_calling"
    if temperature is not None:
        kwargs["temperature"] = temperature
    return init_chat_model(model, model_provider=provider, **kwargs), method


class LLM:
    """LangChain-backed model with retries."""

    def __init__(self, spec: str, temperature: float | None = None, max_tokens: int = 4096, retries: int = 3, role: str = "",
                 http_client: Any = None):
        self.name = spec
        self.role = role or spec  # v0.4.1: usage is recorded per role (doctor, knowledge, prior, patient, ...)
        self.retries = retries
        self._model, self._method = build_chat_model(spec, temperature, max_tokens, http_client)
        self._structured_cache: dict[type, Any] = {}

    def _with_retries(self, fn: Callable[[], Any]) -> Any:
        delay = 2.0
        for attempt in range(1, self.retries + 1):
            try:
                return fn()
            except Exception as exc:  # network, rate limit, or validation errors
                if attempt == self.retries:
                    raise
                log.warning("LLM call failed (%s), retry %d/%d in %.0fs", exc, attempt, self.retries, delay)
                time.sleep(delay)
                delay *= 2

    def structured(self, schema: type[T], system: str, user: str) -> T:
        from langchain_core.messages import HumanMessage, SystemMessage  # v0.5: LangChain only when this class is used

        if schema not in self._structured_cache:
            kw = {"method": self._method} if self._method else {}
            self._structured_cache[schema] = self._model.with_structured_output(schema, include_raw=True, **kw)
        runnable = self._structured_cache[schema]
        msgs = [SystemMessage(content=system), HumanMessage(content=user)]

        def call():
            t0 = time.time()
            try:
                out = runnable.invoke(msgs)
            except Exception:
                usage.record(self.role, self.name, schema.__name__, None, time.time() - t0, error=True)
                raise
            raw = out.get("raw") if isinstance(out, dict) else None
            usage.record(self.role, self.name, schema.__name__, getattr(raw, "usage_metadata", None), time.time() - t0)
            parsed = out.get("parsed") if isinstance(out, dict) else out
            if parsed is None:
                err = out.get("parsing_error") if isinstance(out, dict) else None
                raise RuntimeError(f"{self.name} returned no structured output for {schema.__name__}: {err}")
            return parsed

        return self._with_retries(call)

    def text(self, system: str, user: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        msgs = [SystemMessage(content=system), HumanMessage(content=user)]

        def call():
            t0 = time.time()
            try:
                out = self._model.invoke(msgs)
            except Exception:
                usage.record(self.role, self.name, "text", None, time.time() - t0, error=True)
                raise
            usage.record(self.role, self.name, "text", getattr(out, "usage_metadata", None), time.time() - t0)
            return out

        out = self._with_retries(call)
        content = out.content
        if isinstance(content, list):  # some providers return content blocks
            content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
        return str(content).strip()


class ScriptedLLM:
    """Deterministic stand-in for tests and dry runs.

    `handlers` maps a schema class (or the string "text") to a function
    `(system, user) -> result`. Unhandled calls raise, so tests fail loudly.
    """

    def __init__(self, handlers: dict[Any, Callable[[str, str], Any]], name: str = "scripted"):
        self.handlers = handlers
        self.name = name
        self.calls: list[tuple[str, str]] = []

    def structured(self, schema: type[T], system: str, user: str) -> T:
        self.calls.append((schema.__name__, user[:200]))
        if schema not in self.handlers:
            raise KeyError(f"ScriptedLLM has no handler for {schema.__name__}")
        return self.handlers[schema](system, user)

    def text(self, system: str, user: str) -> str:
        self.calls.append(("text", user[:200]))
        if "text" not in self.handlers:
            raise KeyError("ScriptedLLM has no text handler")
        return self.handlers["text"](system, user)

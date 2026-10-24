"""NovaLLM: the one model client of the competition runtime (OpenAI-compatible chat completions, stdlib only).

Preliminary round: the contest's hosted Doctor model, through the contest server. Final round: the team's own model served locally
(for example vLLM on a local GPU), same API. Nothing here depends on LangChain.

Robustness, in order of what usually goes wrong with a small open model behind an unknown gateway:
  * structured output is requested as `response_format` json_schema -> json_object -> none, downgraded automatically
    when the server rejects a mode (HTTP 400/422), and the JSON is always parsed leniently from the text
    (code fences, prose around it, trailing commas, Python literals) and validated with Pydantic;
  * one repair call when the JSON does not validate; `reasoning_effort` and `max_tokens` naming are dropped or
    switched if the server rejects them; an empty answer cut by the token limit is retried once with more room;
  * every call is charged to the per-case Budget, which refuses calls past a cap or too close to the deadline.

NovaLLM implements the core's LLMLike protocol (`structured`, `text`), so the unchanged graph nodes call it directly;
the role of a call (and so its reasoning effort and token limit) is inferred from the output schema.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from typing import Any, Optional, TypeVar

from pydantic import BaseModel, ValidationError

from dxagent import usage
from dxagent.nova.budget import CLOSING_ROLES, Budget, BudgetExhausted
from dxagent.nova.config import MAX_TOKENS_CEILING, NovaConfig

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

ROLE_BY_SCHEMA = {
    "ExtractionOutput": "perceive", "HypothesisOutput": "hypothesize", "CriticOutput": "critic",
    "ComplaintMatchOutput": "complaint", "PriorAssessment": "prior", "FreeQuestionOutput": "bank",
    "QuestionBank": "bank", "PhraseOutput": "phrase", "Assessment": "assess", "ChartOutput": "notes", "TopicPlan": "plan",
    "FocusPlan": "focus", "ReviewPlan": "review",
    "RubricScore": "judge", "JudgeOutput": "judge",
}
JSON_MODES = ("json_schema", "json_object", "none")


class LLMOutputError(RuntimeError):
    pass


class ConfigError(RuntimeError):
    """The endpoint is not configured: retrying cannot help."""


class HttpError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status
        self.body = body


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _balanced(text: str, start: int) -> Optional[str]:
    """The balanced {...} or [...] that opens at `start`, honoring strings."""
    open_c = text[start]
    close_c = "}" if open_c == "{" else "]"
    depth, i, in_str, esc = 0, start, False, False
    while i < len(text):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1] if ch == close_c else None
        i += 1
    return None


def _repairs(s: str) -> list[str]:
    out = [s]
    t = re.sub(r",\s*([}\]])", r"\1", s)  # trailing commas
    out.append(t)
    t2 = re.sub(r"\bTrue\b", "true", re.sub(r"\bFalse\b", "false", re.sub(r"\bNone\b", "null", t)))
    out.append(t2)
    out.append(t2.replace("“", '"').replace("”", '"'))
    return out


def extract_json(text: str) -> Any:
    """Parse the first JSON object (or array) in a model answer, leniently."""
    if not text:
        raise LLMOutputError("empty answer")
    s = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, flags=re.S)
    candidates = [fence.group(1).strip()] if fence else []
    candidates.append(s)
    for c in candidates:
        for cand in _repairs(c):
            try:
                return json.loads(cand)
            except Exception:
                pass
        for i, ch in enumerate(c):
            if ch in "{[":
                block = _balanced(c, i)
                if block:
                    for cand in _repairs(block):
                        try:
                            return json.loads(cand)
                        except Exception:
                            pass
                if ch == "{":
                    break
    raise LLMOutputError(f"no JSON object in answer: {text[:200]!r}")


def _parses(text: str) -> bool:
    try:
        extract_json(text)
        return True
    except Exception:
        return False


def schema_hint(model: type[BaseModel]) -> str:
    """A compact JSON skeleton of a Pydantic model: field names, types, enums and short descriptions."""
    js = model.model_json_schema()
    defs = js.get("$defs", {})

    def resolve(node: dict) -> dict:
        while "$ref" in node:
            node = defs.get(node["$ref"].split("/")[-1], {})
        return node

    def render(node: dict, depth: int) -> str:
        node = resolve(node)
        if "anyOf" in node:
            parts = [render(n, depth) for n in node["anyOf"] if resolve(n).get("type") != "null"]
            nullable = any(resolve(n).get("type") == "null" for n in node["anyOf"])
            return " | ".join(parts) + (" | null" if nullable else "")
        if "enum" in node:
            return "one of " + json.dumps(node["enum"], ensure_ascii=False)
        t = node.get("type")
        if t == "object" or "properties" in node:
            ind = "  " * depth
            props = list(node.get("properties", {}).items())
            rows = []
            for i, (k, v) in enumerate(props):
                desc = (v.get("description") or resolve(v).get("description") or "").replace("\n", " ")
                if len(desc) > 110:
                    desc = desc[:110].rsplit(" ", 1)[0] + " ..."
                line = f'{ind}  "{k}": {render(v, depth + 1)}' + ("," if i < len(props) - 1 else "")
                if desc:
                    line += f"   // {desc}"
                rows.append(line)
            return "{\n" + "\n".join(rows) + "\n" + ind + "}"
        if t == "array":
            return "[" + render(node.get("items", {}), depth) + ", ...]"
        if t == "number":
            lo, hi = node.get("minimum"), node.get("maximum")
            return f"number {lo}-{hi}" if lo is not None and hi is not None else "number"
        return {"string": "string", "integer": "integer", "boolean": "true|false"}.get(t, "any")

    return render(js, 0)


def validate(schema: type[T], obj: Any) -> T:
    if isinstance(obj, dict) and len(obj) == 1:  # {"FocusPlan": {...}}: a wrapper, not a field of the schema
        (k, inner), = obj.items()
        if isinstance(inner, dict) and k not in schema.model_fields:
            try:
                return schema.model_validate(inner)
            except ValidationError:
                pass
    try:
        return schema.model_validate(obj)
    except ValidationError:
        if isinstance(obj, dict) and len(obj) == 1:  # {"ExtractionOutput": {...}} and similar wrappers
            inner = next(iter(obj.values()))
            if isinstance(inner, dict):
                return schema.model_validate(inner)
        raise


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


def chat_url(base: str) -> str:
    """NOVA_API_BASE is the server root; the model is at /v1/chat/completions. A base that
    already ends in /v1 (OpenAI style) or in /chat/completions is used as given."""
    b = base.rstrip("/")
    if b.endswith("/chat/completions"):
        return b
    if re.search(r"/v\d+$", b):
        return b + "/chat/completions"
    return b + "/v1/chat/completions"


class HttpTransport:
    """POST {NOVA_API_BASE}/v1/chat/completions with a bearer token (development key or run token). In the contest
    harness the SDK's `client.chat()` is used instead (dxagent/nova/sdk.py SdkTransport)."""

    def __init__(self, base: str, key: str):
        self.base = base.rstrip("/")
        self.key = key

    def chat(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        if not self.base:
            raise ConfigError("NOVA_API_BASE is not set (use --offline for runs without a model)")
        url = chat_url(self.base)
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace") if e.fp else ""
            raise HttpError(e.code, body) from None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class NovaLLM:
    def __init__(self, cfg: NovaConfig, transport: Any = None, budget: Optional[Budget] = None, name: str = ""):
        self.cfg = cfg
        self.transport = transport or HttpTransport(cfg.api_base, cfg.api_key)
        self.budget = budget or Budget(cfg)
        self.name = name or cfg.model
        self.json_mode = cfg.json_mode if cfg.json_mode in JSON_MODES else "json_object"
        self.send_effort = True
        self.tokens_key = "max_tokens"
        self.failures = 0  # consecutive failed calls; past `breaker`, only the closing roles still try (once each)
        self.breaker = 4
        self.cut = False  # the last answer was cut by max_tokens even after the retry with more room
        self.trace: list[dict[str, Any]] = []  # one row per call (role, ok, tokens, seconds, error)

    # -- LLMLike --------------------------------------------------------------

    def structured(self, schema: type[T], system: str, user: str, role: Optional[str] = None) -> T:
        role = role or ROLE_BY_SCHEMA.get(schema.__name__, "other")
        sys_msg = (system + "\n\nReturn ONLY one JSON object, no markdown, no text before or after it. "
                   "Use this shape (the // comments explain fields; do not copy them):\n" + schema_hint(schema))
        messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]
        content = self._complete(messages, role, schema)
        try:
            return validate(schema, extract_json(content))
        except (LLMOutputError, ValidationError) as exc:
            if self.cut:  # cut by max_tokens even with twice the room: a repair call would be cut too
                raise LLMOutputError(f"{role}: answer cut by the token limit ({str(exc)[:160]})") from None
            log.info("%s output invalid (%s); one repair call", role, str(exc)[:160])
            repair = messages + [{"role": "assistant", "content": content[:6000]},
                                 {"role": "user", "content": f"That answer was not valid: {str(exc)[:600]}\n"
                                                             "Return ONLY the corrected JSON object."}]
            content2 = self._complete(repair, role, schema)
            try:
                return validate(schema, extract_json(content2))
            except (LLMOutputError, ValidationError) as exc2:
                raise LLMOutputError(f"{role}: invalid JSON after repair: {str(exc2)[:300]}") from None

    def text(self, system: str, user: str, role: Optional[str] = None) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self._complete(messages, role or "text", None).strip()

    # -- one completion -------------------------------------------------------

    def _payload(self, messages: list[dict], role: str, schema: Optional[type[BaseModel]], max_tokens: int) -> dict:
        effort = self.cfg.effort_for(role)
        if effort and not (self.send_effort and getattr(self.transport, "supports_reasoning_effort", True)):
            # without a reasoning_effort parameter (SDK 1.0.0) the level rides in the system message
            # ("Reasoning: low|medium|high"), which gpt-oss read; harmless for other models
            messages = [dict(messages[0], content=f"Reasoning: {effort}\n\n{messages[0]['content']}")] + messages[1:] \
                if messages and messages[0].get("role") == "system" else messages
        p: dict[str, Any] = {"model": self.cfg.model, "messages": messages, self.tokens_key: max_tokens}
        if self.cfg.temperature is not None:
            p["temperature"] = self.cfg.temperature
        if effort and self.send_effort:
            p["reasoning_effort"] = effort
        if schema is not None and self.json_mode == "json_object":
            p["response_format"] = {"type": "json_object"}
        elif schema is not None and self.json_mode == "json_schema":
            p["response_format"] = {"type": "json_schema", "json_schema": {
                "name": schema.__name__, "schema": schema.model_json_schema(), "strict": False}}
        return p

    def _complete(self, messages: list[dict], role: str, schema: Optional[type[BaseModel]]) -> str:
        max_tokens = self.cfg.max_tokens_for(role)
        attempt, grew = 0, False

        def roomy(n: int) -> int:  # never ask past the case's output-token budget (reasoning tokens included)
            return max(1024, min(n, MAX_TOKENS_CEILING, self.budget.output_room(role)))

        max_tokens = roomy(max_tokens)
        if self.failures >= self.breaker and role not in CLOSING_ROLES:
            raise BudgetExhausted(f"endpoint failing ({self.failures} consecutive errors): {role} skipped")
        while True:
            self.budget.check(role)
            payload = self._payload(messages, role, schema, max_tokens)
            t0 = time.time()
            try:
                resp = self.transport.chat(payload, timeout=self.budget.call_timeout(role))
            except HttpError as e:
                self._charge(role, 0, 0, False, time.time() - t0, str(e))
                if e.status == 504:  # the call's time is spent (the transport waited for it): no retry
                    raise
                if e.status in (400, 404, 422) and self._adapt(e.body, payload):
                    continue  # the server rejected a request option: retry without it
                if e.status == 503 and "empty answer" in e.body and not grew:
                    # the contest server's PROVIDER_RESPONSE_INVALID: reasoning used the whole budget -> more room
                    grew, max_tokens = True, roomy(max_tokens * 2)
                    continue
                if e.status in (408, 409, 429) or e.status >= 500:
                    attempt += 1
                    if attempt <= self.cfg.retries and self._sleep(2.0 * attempt):
                        continue
                raise
            except ConfigError as e:
                self._charge(role, 0, 0, False, time.time() - t0, str(e))
                raise
            except Exception as e:  # timeouts, connection resets
                self._charge(role, 0, 0, False, time.time() - t0, f"{type(e).__name__}: {e}")
                attempt += 1
                if attempt <= self.cfg.retries and self._sleep(2.0 * attempt):
                    continue
                raise
            content, finish, tin, tout = self._parse(resp, messages)
            self._charge(role, tin, tout, bool(content), time.time() - t0, "" if content else f"empty ({finish})")
            if content and finish == "length" and schema is not None and not grew and not _parses(content):
                grew, max_tokens = True, roomy(max_tokens * 2)  # the JSON was cut by the limit: once more with room
                continue
            self.cut = bool(content) and finish == "length" and grew
            if content:
                return content
            if finish == "length" and not grew:  # reasoning used the whole budget: once more with room
                grew, max_tokens = True, roomy(max_tokens * 2)
                continue
            raise LLMOutputError(f"{role}: empty answer (finish_reason={finish})")

    def _adapt(self, body: str, payload: dict) -> bool:
        b = body.lower()
        if "response_format" in payload and ("response_format" in b or "json" in b or "schema" in b or "guided" in b):
            i = JSON_MODES.index(self.json_mode)
            self.json_mode = JSON_MODES[min(i + 1, len(JSON_MODES) - 1)]
            log.warning("server rejected response_format; JSON mode now %s", self.json_mode)
            return True
        if "reasoning_effort" in payload and "reasoning" in b:
            self.send_effort = False
            return True
        if "max_tokens" in b and self.tokens_key == "max_tokens":
            self.tokens_key = "max_completion_tokens"
            return True
        if "response_format" in payload and self.json_mode != "none":  # unknown 400 with a JSON mode: try without it
            self.json_mode = "none"
            return True
        return False

    def _sleep(self, seconds: float) -> bool:
        if self.budget.remaining() - seconds < self.cfg.final_margin_s:
            return False
        time.sleep(seconds)
        return True

    @staticmethod
    def _parse(resp: dict, messages: list[dict]) -> tuple[str, str, int, int]:
        choice = (resp.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        if isinstance(content, list):  # content blocks
            content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
        u = resp.get("usage") or {}
        tin = int(u.get("prompt_tokens") or u.get("input_tokens") or sum(len(m["content"]) for m in messages) // 3)
        tout = int(u.get("completion_tokens") or u.get("output_tokens") or len(content) // 3)
        return str(content), str(choice.get("finish_reason") or ""), tin, tout

    def _charge(self, role: str, tin: int, tout: int, ok: bool, seconds: float, error: str = "") -> None:
        self.failures = 0 if ok else self.failures + 1
        self.budget.charge(role, tin, tout, ok, seconds)
        usage.record(role, self.name, role, {"input_tokens": tin, "output_tokens": tout}, seconds, error=not ok)
        self.trace.append({"role": role, "ok": ok, "in": tin, "out": tout, "s": round(seconds, 2),
                           **({"error": error[:200]} if error else {})})

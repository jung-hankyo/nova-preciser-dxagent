"""v0.4.2 providers: DeepSeek through its OpenAI-compatible API, OpenAI reasoning models. Mocked HTTP, no network."""

import json

import httpx
import pytest

from dxagent import usage
from dxagent.llm import LLM, parse_spec
from dxagent.schemas import JudgeOutput


def _reply(args: dict, reasoning: str = "") -> dict:
    msg = {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function",
                                                               "function": {"name": "JudgeOutput", "arguments": json.dumps(args)}}]}
    if reasoning:
        msg["reasoning_content"] = reasoning  # DeepSeek thinking mode returns this beside content
    return {"id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150,
                      "prompt_tokens_details": {"cached_tokens": 100}, "completion_tokens_details": {"reasoning_tokens": 12}}}


@pytest.fixture
def captured(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append((str(request.url), json.loads(request.content)))
        return httpx.Response(200, json=_reply({"top1_correct": True, "match_rank": 1, "rationale": "same disease"}, "thinking..."))

    return sent, httpx.Client(transport=httpx.MockTransport(handler))


def test_spec_options():
    assert parse_spec("deepseek:deepseek-v4-pro?thinking=low") == ("deepseek", "deepseek-v4-pro", {"thinking": "low"})
    assert parse_spec("openai:gpt-5-mini") == ("openai", "gpt-5-mini", {})


def test_deepseek_thinking_payload_and_structured_parse(captured):
    sent, client = captured
    usage.reset()
    usage.set_case("c1")
    llm = LLM("deepseek:deepseek-v4-pro?thinking=low", temperature=0.3, role="doctor", http_client=client)
    out = llm.structured(JudgeOutput, "system", "user")
    assert out.top1_correct is True and out.match_rank == 1
    url, body = sent[0]
    assert url.startswith("https://api.deepseek.com") and url.endswith("/chat/completions")
    assert body["model"] == "deepseek-v4-pro" and body["thinking"] == {"type": "enabled"}
    assert body["reasoning_effort"] == "low" and "temperature" not in body  # thinking mode ignores temperature
    assert body["tools"][0]["function"]["name"] == "JudgeOutput"  # function-calling structured output
    u = usage.case_usage("c1")["by_role"]["doctor"]
    assert u["calls"] == 1 and u["input_tokens"] == 120 and u["cache_read_tokens"] == 100 and u["reasoning_tokens"] == 12
    usage.set_case(None)


def test_deepseek_non_thinking_keeps_temperature(captured):
    sent, client = captured
    LLM("deepseek:deepseek-flash", temperature=0.3, http_client=client).structured(JudgeOutput, "s", "u")
    body = sent[0][1]
    assert body["thinking"] == {"type": "disabled"} and body["temperature"] == 0.3 and body["model"] == "deepseek-flash"


def test_openai_reasoning_model_drops_temperature(captured):
    sent, client = captured
    LLM("openai:gpt-5-mini?effort=minimal", temperature=0.3, max_tokens=1024, http_client=client).structured(JudgeOutput, "s", "u")
    url, body = sent[0]
    assert "api.openai.com" in url and body["model"] == "gpt-5-mini"
    assert "temperature" not in body and body["reasoning_effort"] == "minimal"
    assert body.get("max_completion_tokens", body.get("max_tokens")) >= 8192  # reasoning tokens need room


def test_price_table_covers_recommended_models():
    from dxagent.bench.prices import cost_usd

    for spec in ("deepseek:deepseek-v4-pro?thinking=low", "deepseek:deepseek-flash", "openai:gpt-5-mini?effort=minimal",
                 "anthropic:claude-sonnet-5-5"):
        assert cost_usd(spec, 1000, 100) is not None
    assert abs(cost_usd("deepseek:deepseek-flash", 1_000_000, 0, cache_read_tokens=1_000_000) - 0.006) < 1e-9

"""List prices for cost estimates in the benchmark report (v0.4.2). USD per 1M tokens: (input, cached input, output).

Checked 2026-10-02 against the providers' pricing pages (OpenAI's via third-party trackers citing its rate sheet).
Prices change: override with DX_PRICES='{"deepseek:deepseek-flash": [0.30, 0.006, 1.20], ...}' (keys are model specs
without "?options", or bare model names). DeepSeek rates are PEAK (01:00-04:00 and 06:00-10:00 UTC on weekdays);
off-peak is half, so the estimate is an upper bound for DeepSeek runs outside those hours.
"""

from __future__ import annotations

import json
import os
from typing import Optional

PRICES: dict[str, tuple[float, float, float]] = {
    # Anthropic (platform.claude.com/docs/en/about-claude/pricing); cached = cache hit
    "claude-fable-5-1": (10.0, 0.25, 50.0),
    "claude-opus-5-5": (4.0, 0.20, 20.0),
    "claude-sonnet-5-5": (2.0, 0.20, 10.0),
    "claude-sonnet-5": (2.0, 0.20, 10.0),
    "claude-haiku-4-5": (1.0, 0.10, 5.0),
    "claude-haiku-4-5-20251001": (1.0, 0.10, 5.0),
    # DeepSeek (api-docs.deepseek.com/quick_start/pricing), peak
    "deepseek-flash": (0.30, 0.006, 1.20),
    "deepseek-v4-flash": (0.30, 0.006, 1.20),  # legacy name, served and billed as deepseek-flash
    "deepseek-v4-pro": (1.32, 0.044, 3.96),
    # OpenAI (as cited from developers.openai.com/api/docs/pricing)
    "gpt-5-mini": (0.25, 0.025, 2.00),
}


def price_for(spec: str) -> Optional[tuple[float, float, float]]:
    table = dict(PRICES)
    try:
        table.update({k: tuple(v) for k, v in json.loads(os.getenv("DX_PRICES", "{}")).items()})
    except ValueError:
        pass
    base = spec.split("?", 1)[0]
    model = base.split(":", 1)[-1]
    return table.get(base) or table.get(model)


def cost_usd(spec: str, input_tokens: float, output_tokens: float, cache_read_tokens: float = 0.0) -> Optional[float]:
    p = price_for(spec)
    if p is None:
        return None
    fresh = max(input_tokens - cache_read_tokens, 0.0)
    return (fresh * p[0] + cache_read_tokens * p[1] + output_tokens * p[2]) / 1e6

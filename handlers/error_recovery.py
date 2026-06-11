import json
from random import random

import requests

from settings.constant import MODEL, BACKOFF_BASE_DELAY, BACKOFF_MAX_DELAY, OPENROUTER_API_KEY


def estimate_tokens(messages: list) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(json.dumps(messages, default=str)) // 4


def auto_compact(messages: list) -> list:
    conversation_text = json.dumps(messages, default=str)
    prompt = (
            "Summarize this conversation for continuity. Include:\n"
            "1) Task overview and success criteria\n"
            "2) Current state: completed work, files touched\n"
            "3) Key decisions and failed approaches\n"
            "4) Remaining next steps\n"
            "Be concise but preserve critical details.\n\n"
            + conversation_text
    )
    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        header = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 8000,
        }
        resp = requests.post(url, headers=header, json=payload, timeout=5)
        resp.raise_for_status()
    except Exception as e:
        summary = f"(compact failed: {e}). Previous context lost."

    continuation = (
        "This session continues from a previous conversation that was compacted. "
        f"Summary of prior context:\n\n{summary}\n\n"
        "Continue from where we left off without re-asking the user."
    )
    return [{"role": "user", "content": continuation}]


def backoff_delay(attempt: int) -> float:
    delay = min(BACKOFF_BASE_DELAY * (2 ** attempt), BACKOFF_MAX_DELAY)
    jitter = delay * random.uniform(0, 0.1)
    return delay + jitter
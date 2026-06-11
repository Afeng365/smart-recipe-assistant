import json
import time
import requests
from dataclasses import dataclass, field
from pathlib import Path

from settings.constant import TRANSCRIPT_DIR, MODEL, OPENROUTER_API_KEY


# === persisted_output ===
@dataclass
class CompactState:
    has_compacted: bool = False
    last_summary: str = ""
    recent_files: list[str] = field(default_factory=list)


def estimate_context_size(messages: list) -> int:
    return len(str(messages))


def write_transcript(messages: list) -> Path:
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with path.open("w") as handle:
        for message in messages:
            handle.write(json.dumps(message, default=str) + "\n")
    return path


def summarize_history(messages: list) -> str:
    conversation = json.dumps(messages, default=str)[:80000]
    prompt = (
        "Summarize this conversation so work can continue.\n"
        "Preserve:\n"
        "1. The current goal\n"
        "2. Important findings and decisions\n"
        "3. Files read or changed\n"
        "4. Remaining work\n"
        "5. User constraints and preferences\n"
        "Be compact but concrete.\n\n"
        f"{conversation}"
    )

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
    return resp.json()["choices"][0]["message"]["content"]


def compact_history(messages: list, state: CompactState, focus: str | None = None) -> list:
    transcript_path = write_transcript(messages)
    print(f"[transcript saved: {transcript_path}]")

    summary = summarize_history(messages)
    if focus:
        summary += f"\n\nFocus to preserve next: {focus}"
    if state.recent_files:
        recent_lines = "\n".join(f"- {path}" for path in state.recent_files)
        summary += f"\n\nRecent files to reopen if needed:\n{recent_lines}"

    state.has_compacted = True
    state.last_summary = summary

    return [{
        "role": "user",
        "content": (
            "This conversation was compacted so the agent can continue working.\n\n"
            f"{summary}"
        ),
    }]
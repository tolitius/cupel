"""cupel.board — leaderboard rules shared by the local server and the publisher.

Lives apart from `cupel.server` so `cupel.publish` can use the same rules without
importing the server (which imports the publisher for its endpoints).
"""

from urllib.parse import urlparse

# A run scored on only a handful of its prompts is not comparable to a full run.
# Scoring out of what was actually scored (rather than out of every prompt) is
# correct per run, but with no floor a run that errored on 21 of 23 prompts and
# happened to ace the other two would report 100% and sort to the top.
MIN_COVERAGE = 0.8

# cloud hosts we name; anything else that is not local is shown as its hostname
KNOWN_PROVIDERS = (("openrouter.ai", "OpenRouter"),
                   ("anthropic.com", "Anthropic"),
                   ("openai.com", "OpenAI"))


def provider_label(api_url: str, local_hw: dict) -> dict:
    """Hardware/provider badge for a run: local machine specs, or the cloud host."""
    if not api_url or "localhost" in api_url or "127.0.0.1" in api_url:
        return local_hw
    for host, name in KNOWN_PROVIDERS:
        if host in api_url:
            return {"name": name, "memory": ""}
    return {"name": urlparse(api_url).hostname or api_url, "memory": ""}


def coverage(results: list) -> float:
    scored = sum(1 for r in results if r.get("score") is not None)
    return scored / len(results) if results else 0.0

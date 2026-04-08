"""cupel.config — configuration loading, .env parsing, API config."""

import os
import sys
import yaml
from pathlib import Path


# ──────────────────────────────────────────────
# Prompt ID parsing (--prompts flag)
# ──────────────────────────────────────────────

def parse_prompt_ids(spec: str) -> set[int]:
    """Parse '1,18-22,5' into {1, 5, 18, 19, 20, 21, 22}."""
    ids = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.update(range(int(lo), int(hi) + 1))
        else:
            ids.add(int(part))
    return ids


# ──────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────

DEFAULTS = {
    "models": [],
    "eval_set": "eval-sets/eval-set.json",
    "image_filename": "what-am-i-looking-at.png",
    "output_dir": "./eval-results",
    "temperature": 0,
    "max_tokens": 16384,
    "generation_model": "",
}

IMAGE_PROMPT_ID = None

# Port → env-var mapping for local inference servers.
# Each provider gets its own key; LLM_API_KEY is the legacy fallback.
LOCAL_PROVIDER_KEYS = {
    8000:  "OMLX_API_KEY",
    11434: "OLLAMA_API_KEY",
    1234:  "LM_STUDIO_API_KEY",
    30000: "SGLANG_API_KEY",
}


def resolve_api_key_for_port(port: int) -> str:
    """Return the API key for a local provider port, falling back to LLM_API_KEY."""
    key_env = LOCAL_PROVIDER_KEYS.get(port)
    if key_env:
        key = os.environ.get(key_env, "")
        if key:
            return key
    return os.environ.get("LLM_API_KEY", "no-key")


def load_config(path: Path | None = None) -> tuple[dict, str | None]:
    candidates = [path] if path else [
        Path.cwd() / "config.yml",
        Path(__file__).parent.parent / "config.yml",
    ]
    for p in candidates:
        if p and p.exists():
            with open(p) as f:
                cfg = yaml.safe_load(f) or {}
            return {**DEFAULTS, **cfg}, str(p)
    return dict(DEFAULTS), None


def load_dotenv(path: Path = None) -> str | None:
    candidates = [path] if path else [
        Path.cwd() / ".env",
        Path.home() / ".cupel" / ".env",
        Path(__file__).parent.parent / ".env",
        Path.home() / ".env",
    ]
    for p in candidates:
        if p and p.exists():
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip("'\"")
                    if key not in os.environ:
                        os.environ[key] = value
            return str(p)
    return None


def reload_dotenv() -> str | None:
    """Re-read .env files, updating keys even if already set."""
    candidates = [
        Path.cwd() / ".env",
        Path.home() / ".cupel" / ".env",
        Path(__file__).parent.parent / ".env",
        Path.home() / ".env",
    ]
    loaded = None
    for p in candidates:
        if p and p.exists():
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip("'\"")
                    os.environ[key] = value
            if loaded is None:
                loaded = str(p)
    return loaded


def get_api_config() -> tuple[str, str]:
    """Get the eval/run endpoint config from env."""
    api_url = os.environ.get("LLM_API_URL", "http://localhost:8000/v1/chat/completions")
    api_key = os.environ.get("LLM_API_KEY", "no-key")
    return api_url, api_key


def get_judge_config(cfg: dict) -> tuple[str, str, str]:
    """
    Resolve judge model, url, and key from config.yml + env.
    Falls back to the eval endpoint if not specified.
    """
    judge_cfg = cfg.get("judge") or {}
    if isinstance(judge_cfg, str):
        # Simple form: judge: Qwen3.5-27B-8bit
        judge_cfg = {"model": judge_cfg}

    model = judge_cfg.get("model", "")
    api_url = judge_cfg.get("api_url") or os.environ.get("LLM_API_URL", "http://localhost:8000/v1/chat/completions")
    key_env = judge_cfg.get("api_key_env", "LLM_API_KEY")
    api_key = os.environ.get(key_env, "no-key")

    return model, api_url, api_key


def get_providers_config(cfg: dict) -> list[dict]:
    """Get external provider definitions from config."""
    return cfg.get("providers") or []

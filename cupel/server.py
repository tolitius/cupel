"""cupel.server — web UI for bench eval framework"""

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from cupel import __version__
from cupel.config import (
    load_config, load_dotenv, reload_dotenv, get_api_config, get_judge_config,
    get_providers_config, resolve_api_key_for_port,
)
from cupel.eval import (
    run_eval, judge_results, score_one, _prompt_text_for_judge, run_prompt,
    call_llm, find_image,
)
from cupel.discovery import detect_hardware, discover_providers

import yaml

# ── Logging ──
log = logging.getLogger("cupel")
log.setLevel(logging.INFO)
_log_dir = Path.home() / ".cupel"
_log_dir.mkdir(parents=True, exist_ok=True)
_log_path = _log_dir / "cupel.log"
_fh = logging.FileHandler(_log_path)
_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %(message)s", datefmt="%H:%M:%S"))
log.addHandler(_fh)

# Route uvicorn access & error logs to the same file
for _uv_name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
    _uv_log = logging.getLogger(_uv_name)
    _uv_log.addHandler(_fh)

BASE_DIR = Path(__file__).parent.parent   # repo root
PKG_DIR = Path(__file__).parent           # package dir (cupel/)
RESULTS_DIR = Path.cwd() / "eval-results"
DATA_DIR = PKG_DIR / "data"
UI_DIR = PKG_DIR / "ui"
TAGS_FILE = RESULTS_DIR / ".tags.json"
HIDDEN_FILE = RESULTS_DIR / ".hidden.json"

app = FastAPI(title="cupel", version=__version__)

# Load .env on import (needed for uvicorn --reload mode)
load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Job model
# ──────────────────────────────────────────────

@dataclass
class Job:
    id: str
    type: str               # "run" or "judge"
    status: str             # "running", "complete", "error", "cancelled"
    progress: list = field(default_factory=list)
    result_files: list = field(default_factory=list)
    created_at: str = ""
    error: str = ""
    cancelled: bool = False
    models: list = field(default_factory=list)
    prompt_ids: list = field(default_factory=list)
    live_results: dict = field(default_factory=dict)  # {model: {prompt_id: result_dict}}

jobs: dict[str, Job] = {}

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _config_path() -> Path:
    return Path.cwd() / "config.yml"

def _eval_set_path() -> Path:
    return Path.cwd() / "eval-sets" / "eval-set.json"

def _read_config() -> dict:
    """Read config.yml, returning defaults if missing."""
    cfg, _ = load_config(_config_path() if _config_path().exists() else None)
    return cfg

def _read_eval_set() -> dict:
    """Read eval-set.json with fallback chain:
    1. {cwd}/eval-sets/eval-set.json  (user's custom set)
    2. cupel/data/starter-eval-set.json (starter set, last resort)
    """
    p = _eval_set_path()
    if p.exists():
        with open(p) as f:
            return json.load(f)
    starter = DATA_DIR / "starter-eval-set.json"
    if starter.exists():
        with open(starter) as f:
            return json.load(f)
    return {"name": "empty", "prompts": []}

def _load_tags() -> dict:
    if TAGS_FILE.exists():
        with open(TAGS_FILE) as f:
            return json.load(f)
    return {}

def _save_tags(tags: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(TAGS_FILE, "w") as f:
        json.dump(tags, f, indent=2)

def _load_hidden() -> list:
    if HIDDEN_FILE.exists():
        with open(HIDDEN_FILE) as f:
            return json.load(f)
    return []

def _save_hidden(hidden: list):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(HIDDEN_FILE, "w") as f:
        json.dump(hidden, f, indent=2)

def _result_files() -> list[Path]:
    """List all eval result JSON files (exclude .tags.json)."""
    if not RESULTS_DIR.exists():
        return []
    return sorted(
        [p for p in RESULTS_DIR.glob("eval_*.json")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

def _summarize_result(path: Path, data: dict) -> dict:
    """Build summary metadata for a result file."""
    results = data.get("results", [])
    scored = [r for r in results if r.get("score") is not None]
    total_score = sum(r["score"] for r in scored)
    max_score = len(results) * 3
    total_elapsed = sum(r.get("elapsed_seconds", 0) for r in results)
    total_tokens = sum(r.get("completion_tokens", 0) for r in results)

    return {
        "filename": path.name,
        "model": data.get("model", "unknown"),
        "timestamp": data.get("timestamp", ""),
        "eval_set": data.get("eval_set", ""),
        "judge": data.get("judge", ""),
        "num_prompts": len(results),
        "num_scored": len(scored),
        "total_score": total_score,
        "max_score": max_score,
        "pct": round(total_score / max_score * 100, 1) if max_score > 0 else 0,
        "total_elapsed": round(total_elapsed, 1),
        "total_tokens": total_tokens,
    }

def _resolve_provider(cfg: dict, model: str, model_urls: dict | None = None) -> tuple[str, str]:
    """Resolve api_url and api_key for a model, checking config providers first."""
    for p in get_providers_config(cfg):
        if model in p.get("models", []):
            api_url = p.get("api_url", "")
            key_env = p.get("api_key_env", "LLM_API_KEY")
            return api_url, os.environ.get(key_env, "no-key")
    # URL passed from UI selection (provider the user picked in the dropdown)
    if model_urls and model in model_urls:
        base = model_urls[model]
        api_url = base.rstrip("/") + "/v1/chat/completions"
        # Resolve key from port (OMLX_API_KEY, OLLAMA_API_KEY, etc.)
        try:
            port = int(base.split(":")[-1].split("/")[0])
        except (ValueError, IndexError):
            port = 0
        return api_url, resolve_api_key_for_port(port)
    # Look up from discovered local providers (covers Author page, etc.)
    for p in discover_providers():
        if model in p.get("models", []):
            base = p.get("url", "")
            api_url = base.rstrip("/") + "/v1/chat/completions"
            port = p.get("port", 0)
            return api_url, resolve_api_key_for_port(port)
    return get_api_config()

# ──────────────────────────────────────────────
# Routes: state / init
# ──────────────────────────────────────────────

@app.get("/api/state")
async def get_state():
    has_config = _config_path().exists()
    has_eval_set = _eval_set_path().exists()
    has_results = len(_result_files()) > 0
    first_run = not has_results
    return {"first_run": first_run, "has_config": has_config, "has_eval_set": has_eval_set}

@app.post("/api/init")
async def init_project(request: Request):
    """Create config.yml + eval-set.json from starter templates."""
    cfg_path = _config_path()
    es_path = _eval_set_path()

    # Create config.yml with sensible defaults if missing
    if not cfg_path.exists():
        default_cfg = {
            "models": [],
            "eval_set": "eval-sets/eval-set.json",
            "image_filename": "what-am-i-looking-at.png",
            "output_dir": "./eval-results",
            "temperature": 0,
            "max_tokens": 16384,
            "thinking": None,
        }
        with open(cfg_path, "w") as f:
            yaml.dump(default_cfg, f, default_flow_style=False, sort_keys=False)

    # Copy full eval set if no eval-set.json exists
    if not es_path.exists():
        es_path.parent.mkdir(parents=True, exist_ok=True)
        full = DATA_DIR / "starter-eval-set.json"
        if full.exists():
            with open(full) as f:
                data = json.load(f)
            with open(es_path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            with open(es_path, "w") as f:
                json.dump({"name": "my eval set", "prompts": []}, f, indent=2)

    return {"status": "ok", "config": str(cfg_path), "eval_set": str(es_path)}

# ──────────────────────────────────────────────
# Routes: providers / hardware
# ──────────────────────────────────────────────

@app.get("/api/providers")
async def get_providers():
    reload_dotenv()
    local = await asyncio.to_thread(discover_providers)
    for p in local:
        p["source"] = "local"
    cfg = _read_config()
    for ep in get_providers_config(cfg):
        local.append({
            "name": ep.get("name", "external"),
            "url": ep.get("api_url", ""),
            "status": "configured",
            "models": ep.get("models", []),
            "source": "external",
            "api_key_env": ep.get("api_key_env", ""),
        })
    return local

@app.get("/api/providers/keys")
async def get_provider_keys():
    cfg = _read_config()
    keys = {}
    for p in get_providers_config(cfg):
        env_var = p.get("api_key_env", "")
        if env_var:
            keys[env_var] = bool(os.environ.get(env_var))
    judge_cfg = cfg.get("judge") or {}
    if isinstance(judge_cfg, dict):
        env_var = judge_cfg.get("api_key_env", "")
        if env_var and env_var not in keys:
            keys[env_var] = bool(os.environ.get(env_var))
    return keys

@app.get("/api/env-check")
async def check_env_var(key: str):
    reload_dotenv()
    return {"key": key, "set": bool(os.environ.get(key))}

@app.post("/api/providers/test")
async def test_provider(request: Request):
    """Test connectivity to a provider by fetching its model list."""
    import requests as req

    body = await request.json()
    api_url = body.get("api_url", "")
    api_key_env = body.get("api_key_env", "")

    api_key = os.environ.get(api_key_env, "") if api_key_env else ""

    # Derive /models URL
    if "anthropic.com" in api_url:
        # Anthropic has no /models endpoint — just check the key is set
        if api_key:
            return {"ok": True, "models": 3, "detail": "key set (3 known models)"}
        return {"ok": False, "detail": f"{api_key_env} not set"}

    base = api_url.split("/chat/completions")[0].rstrip("/")
    models_url = base + "/models"

    def _test():
        headers = {"HTTP-Referer": "https://github.com/tolitius/cupel", "X-OpenRouter-Title": "cupel", "User-Agent": "cupel/0.1"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        resp = req.get(models_url, headers=headers, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        count = len(data.get("data", []))
        return count

    try:
        count = await asyncio.to_thread(_test)
        return {"ok": True, "models": count, "detail": f"connected ({count} models)"}
    except req.ConnectionError:
        return {"ok": False, "detail": "connection refused \u2014 is the server running?"}
    except req.Timeout:
        return {"ok": False, "detail": "connection timed out"}
    except req.HTTPError as e:
        code = e.response.status_code if e.response is not None else "?"
        if code == 401 or code == 403:
            return {"ok": False, "detail": f"HTTP {code} \u2014 check API key"}
        return {"ok": False, "detail": f"HTTP {code}"}
    except Exception as e:
        return {"ok": False, "detail": str(e)[:100]}

@app.post("/api/providers/fetch-models")
async def fetch_provider_models(request: Request):
    import requests as req

    reload_dotenv()
    body = await request.json()
    api_url = body.get("api_url", "")
    api_key_env = body.get("api_key_env", "")

    api_key = os.environ.get(api_key_env, "") if api_key_env else ""
    is_local = "localhost" in api_url or "127.0.0.1" in api_url
    if not api_key and api_key_env and not is_local:
        raise HTTPException(400, detail=f"{api_key_env} not set in environment")

    # Anthropic — no listing API, return known models with pricing (per token, USD)
    if "anthropic.com" in api_url:
        return {"models": [
            {"id": "claude-opus-4-6",    "pricing": {"prompt": "0.000015",  "completion": "0.000075"}},
            {"id": "claude-sonnet-4-6",  "pricing": {"prompt": "0.000003",  "completion": "0.000015"}},
            {"id": "claude-haiku-4-5",   "pricing": {"prompt": "0.0000008", "completion": "0.000004"}},
        ]}

    # OpenAI-compatible APIs (OpenAI, OpenRouter, etc.) — call /models
    base = api_url.split("/chat/completions")[0].rstrip("/")
    models_url = base + "/models"

    def _fetch():
        headers = {"HTTP-Referer": "https://github.com/tolitius/cupel", "X-OpenRouter-Title": "cupel", "User-Agent": "cupel/0.1"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        resp = req.get(models_url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()

    try:
        data = await asyncio.to_thread(_fetch)

        models = []
        for m in data.get("data", []):
            mid = m.get("id", "")
            if not mid:
                continue
            entry = {"id": mid}
            # Extract pricing if available (OpenRouter provides this)
            pricing = m.get("pricing")
            if isinstance(pricing, dict):
                prompt_cost = pricing.get("prompt", "")
                completion_cost = pricing.get("completion", "")
                if prompt_cost or completion_cost:
                    # OpenRouter pricing is per-token; convert to per-million for display
                    entry["pricing"] = {"prompt": prompt_cost, "completion": completion_cost}
            models.append(entry)

        models.sort(key=lambda x: x["id"])
        return {"models": models}
    except Exception as e:
        raise HTTPException(502, detail=f"Failed to fetch models: {e}")

@app.get("/api/hardware")
async def get_hardware():
    hw = await asyncio.to_thread(detect_hardware)
    return hw

# ──────────────────────────────────────────────
# Routes: config
# ──────────────────────────────────────────────

@app.get("/api/config")
async def get_config():
    cfg = _read_config()
    return cfg

@app.put("/api/config")
async def put_config(request: Request):
    body = await request.json()
    cfg_path = _config_path()
    with open(cfg_path, "w") as f:
        yaml.dump(body, f, default_flow_style=False, sort_keys=False)
    return {"status": "ok"}

# ──────────────────────────────────────────────
# Routes: eval set
# ──────────────────────────────────────────────

@app.get("/api/eval-set")
async def get_eval_set(variant: str = None):
    if variant == "starter":
        starter = DATA_DIR / "starter-eval-set.json"
        if starter.exists():
            with open(starter) as f:
                return json.load(f)
        return {"name": "starter", "prompts": []}
    return _read_eval_set()

@app.put("/api/eval-set")
async def put_eval_set(request: Request):
    body = await request.json()
    es_path = _eval_set_path()
    es_path.parent.mkdir(parents=True, exist_ok=True)
    with open(es_path, "w") as f:
        json.dump(body, f, indent=2)
    return {"status": "ok"}

@app.post("/api/eval-set/prompts")
async def add_prompt(request: Request):
    prompt = await request.json()
    eval_set = _read_eval_set()

    # Auto-assign id if not provided
    existing_ids = {p["id"] for p in eval_set.get("prompts", [])}
    if "id" not in prompt:
        prompt["id"] = max(existing_ids, default=0) + 1

    eval_set.setdefault("prompts", []).append(prompt)
    es_path = _eval_set_path()
    es_path.parent.mkdir(parents=True, exist_ok=True)
    with open(es_path, "w") as f:
        json.dump(eval_set, f, indent=2)

    return {"status": "ok", "prompt": prompt}

# ──────────────────────────────────────────────
# Routes: results
# ──────────────────────────────────────────────

@app.get("/api/results")
async def list_results():
    tags = _load_tags()
    hidden = _load_hidden()
    results = []
    for path in _result_files():
        try:
            with open(path) as f:
                data = json.load(f)
            summary = _summarize_result(path, data)
            summary["tags"] = tags.get(path.name, [])
            summary["muted"] = path.name in hidden
            results.append(summary)
        except (json.JSONDecodeError, KeyError):
            continue
    return results

@app.get("/api/results/leaderboard")
async def get_leaderboard():
    hw = await asyncio.to_thread(detect_hardware)
    hidden = _load_hidden()

    # Collect all scored results — each result file is its own entry
    entries_list: list[dict] = []
    user_models: set[str] = set()

    # Load user results (skip muted)
    for path in _result_files():
        if path.name in hidden:
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, KeyError):
            continue

        model = data.get("model", "unknown")
        scored = [r for r in data.get("results", []) if r.get("score") is not None]
        if not scored:
            continue

        user_models.add(model)

        total_score = sum(r["score"] for r in scored)
        max_score = len(data["results"]) * 3
        total_elapsed = sum(r.get("elapsed_seconds", 0) for r in data["results"])
        total_tokens = sum(r.get("completion_tokens", 0) for r in data["results"])
        num_prompts = len(data["results"])

        # Determine if run was local or via external provider
        api_url = data.get("api_url", "")
        is_local = not api_url or "localhost" in api_url or "127.0.0.1" in api_url
        if is_local:
            entry_hw = hw
        else:
            # Derive provider name from URL
            if "openrouter.ai" in api_url:
                prov_name = "OpenRouter"
            elif "anthropic.com" in api_url:
                prov_name = "Anthropic"
            elif "openai.com" in api_url:
                prov_name = "OpenAI"
            else:
                from urllib.parse import urlparse
                prov_name = urlparse(api_url).hostname or api_url
            entry_hw = {"name": prov_name, "memory": ""}

        entries_list.append({
            "model": model,
            "total_score": total_score,
            "max_score": max_score,
            "pct": round(total_score / max_score * 100, 1) if max_score > 0 else 0,
            "scores_by_prompt": [
                {
                    "id": r["id"],
                    "score": r.get("score"),
                    "elapsed": r.get("elapsed_seconds", 0),
                    "tokens": r.get("completion_tokens", 0),
                    "category": r.get("category", ""),
                    "title": r.get("title", ""),
                }
                for r in data["results"]
            ],
            "hardware": entry_hw,
            "is_example": False,
            "filename": path.name,
            "judge_model": data.get("judge", ""),
            "self_judged": data.get("judge", "") == model,
            "timestamp": data.get("timestamp", ""),
            "tok_per_sec": round(total_tokens / total_elapsed, 1) if total_elapsed > 0 else 0,
            "avg_time": round(total_elapsed / num_prompts, 1) if num_prompts > 0 else 0,
        })

    # Load example data
    example_path = DATA_DIR / "example-run.json"
    if example_path.exists():
        try:
            with open(example_path) as f:
                example = json.load(f)
            for entry in example.get("models", []):
                model = entry["model"]
                if model in user_models:
                    continue  # user data takes precedence
                results = entry.get("results", [])
                api_url = entry.get("api_url", "")
                is_local = not api_url or "localhost" in api_url or "127.0.0.1" in api_url
                if is_local:
                    entry_hw = hw
                else:
                    if "openrouter.ai" in api_url:
                        prov_name = "OpenRouter"
                    elif "anthropic.com" in api_url:
                        prov_name = "Anthropic"
                    elif "openai.com" in api_url:
                        prov_name = "OpenAI"
                    else:
                        from urllib.parse import urlparse
                        prov_name = urlparse(api_url).hostname or api_url
                    entry_hw = {"name": prov_name, "memory": ""}
                scored = [r for r in results if r.get("score") is not None]
                if not scored:
                    continue
                total_score = sum(r["score"] for r in scored)
                max_score = len(results) * 3
                total_elapsed = sum(r.get("elapsed_seconds", 0) for r in results)
                total_tokens = sum(r.get("completion_tokens", 0) for r in results)
                num_prompts = len(results)

                entries_list.append({
                    "model": model,
                    "total_score": total_score,
                    "max_score": max_score,
                    "pct": round(total_score / max_score * 100, 1) if max_score > 0 else 0,
                    "scores_by_prompt": [
                        {
                            "id": r["id"],
                            "score": r.get("score"),
                            "elapsed": r.get("elapsed_seconds", 0),
                            "tokens": r.get("completion_tokens", 0),
                            "category": r.get("category", ""),
                            "title": r.get("title", ""),
                            "response": r.get("response", ""),
                            "judge_reason": r.get("judge_reason", ""),
                            "thinking": r.get("thinking", ""),
                        }
                        for r in results
                    ],
                    "hardware": entry_hw,
                    "is_example": True,
                    "filename": "data/example-run.json",
                    "judge_model": example.get("judge", ""),
                    "self_judged": False,
                    "timestamp": entry.get("timestamp", ""),
                    "tok_per_sec": round(total_tokens / total_elapsed, 1) if total_elapsed > 0 else 0,
                    "avg_time": round(total_elapsed / num_prompts, 1) if num_prompts > 0 else 0,
                })
        except (json.JSONDecodeError, KeyError):
            pass

    # Sort by percentage descending
    entries = sorted(entries_list, key=lambda x: x["pct"], reverse=True)

    # Collect unique prompts across all entries
    prompt_ids_seen = set()
    prompts_list = []
    for entry in entries:
        for sp in entry.get("scores_by_prompt", []):
            pid = sp["id"]
            if pid not in prompt_ids_seen:
                prompt_ids_seen.add(pid)
                prompts_list.append({"id": pid, "category": sp.get("category", ""), "title": sp.get("title", "")})

    max_score = entries[0]["max_score"] if entries else 0
    return {"entries": entries, "prompts": prompts_list, "max_score": max_score}

@app.get("/api/results/{filename}")
async def get_result(filename: str):
    path = RESULTS_DIR / filename
    if not path.exists() or not path.name.startswith("eval_"):
        raise HTTPException(status_code=404, detail="Result not found")
    with open(path) as f:
        return json.load(f)

@app.delete("/api/results/{filename}")
async def delete_result(filename: str):
    path = RESULTS_DIR / filename
    if not path.exists() or not path.name.startswith("eval_"):
        raise HTTPException(status_code=404, detail="Result not found")
    path.unlink()
    # Clean up tags
    tags = _load_tags()
    tags.pop(filename, None)
    _save_tags(tags)
    # Clean up hidden
    hidden = _load_hidden()
    if filename in hidden:
        hidden.remove(filename)
        _save_hidden(hidden)
    return {"status": "ok"}

@app.post("/api/results/{filename}/tag")
async def tag_result(filename: str, request: Request):
    path = RESULTS_DIR / filename
    if not path.exists() or not path.name.startswith("eval_"):
        raise HTTPException(status_code=404, detail="Result not found")
    body = await request.json()
    tag = body.get("tag", "").strip()
    if not tag:
        raise HTTPException(status_code=400, detail="Tag required")

    tags = _load_tags()
    file_tags = tags.get(filename, [])
    if tag not in file_tags:
        file_tags.append(tag)
    tags[filename] = file_tags
    _save_tags(tags)
    return {"status": "ok", "tags": file_tags}

@app.post("/api/results/{filename}/mute")
async def mute_result(filename: str):
    path = RESULTS_DIR / filename
    if not path.exists() or not path.name.startswith("eval_"):
        raise HTTPException(status_code=404, detail="Result not found")
    hidden = _load_hidden()
    if filename in hidden:
        hidden.remove(filename)
        muted = False
    else:
        hidden.append(filename)
        muted = True
    _save_hidden(hidden)
    return {"status": "ok", "muted": muted}

# ──────────────────────────────────────────────
# Routes: compare
# ──────────────────────────────────────────────

@app.get("/api/compare")
async def compare_responses(prompt_id: int):
    """Return all model responses for a given prompt across result files."""
    responses = []
    for path in _result_files():
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, KeyError):
            continue
        for r in data.get("results", []):
            if r.get("id") == prompt_id:
                responses.append({
                    "model": data.get("model", "unknown"),
                    "filename": path.name,
                    "timestamp": data.get("timestamp", ""),
                    "response": r.get("response", ""),
                    "responses": r.get("responses"),
                    "thinking": r.get("thinking", ""),
                    "score": r.get("score"),
                    "judge_reason": r.get("judge_reason", ""),
                    "judge_model": r.get("judge_model", ""),
                    "elapsed_seconds": r.get("elapsed_seconds", 0),
                    "completion_tokens": r.get("completion_tokens", 0),
                })
                break
    # Also include example data
    example_path = DATA_DIR / "example-run.json"
    if example_path.exists():
        try:
            with open(example_path) as f:
                example = json.load(f)
            for entry in example.get("models", []):
                for r in entry.get("results", []):
                    if r.get("id") == prompt_id:
                        responses.append({
                            "model": entry["model"],
                            "filename": "data/example-run.json",
                            "timestamp": entry.get("timestamp", ""),
                            "response": r.get("response", ""),
                            "responses": r.get("responses"),
                            "thinking": r.get("thinking", ""),
                            "score": r.get("score"),
                            "judge_reason": r.get("judge_reason", ""),
                            "judge_model": r.get("judge_model", ""),
                            "elapsed_seconds": r.get("elapsed_seconds", 0),
                            "completion_tokens": r.get("completion_tokens", 0),
                            "is_example": True,
                        })
                        break
        except (json.JSONDecodeError, KeyError):
            pass

    # Find prompt info from eval set
    eval_set = _read_eval_set()
    prompt_info = next((p for p in eval_set.get("prompts", []) if p.get("id") == prompt_id), {})

    return {
        "prompt_id": prompt_id,
        "title": prompt_info.get("title", f"Prompt {prompt_id}"),
        "category": prompt_info.get("category", ""),
        "rubric": prompt_info.get("rubric", {}),
        "responses": responses,
    }

# ──────────────────────────────────────────────
# Routes: jobs
# ──────────────────────────────────────────────

@app.post("/api/jobs")
async def create_job(request: Request):
    body = await request.json()
    job_type = body.get("type", "run")

    job = Job(
        id=str(uuid.uuid4())[:8],
        type=job_type,
        status="running",
        created_at=datetime.now().isoformat(),
    )
    jobs[job.id] = job

    if job_type == "run":
        asyncio.create_task(_run_job(job, body))
    elif job_type == "judge":
        asyncio.create_task(_judge_job(job, body))
    else:
        job.status = "error"
        job.error = f"Unknown job type: {job_type}"

    return {"id": job.id, "status": job.status}

async def _run_job(job: Job, body: dict):
    """Execute an eval run, checking cancellation between each prompt."""
    try:
        models = body.get("models", [])
        model_urls_map = body.get("model_urls") or {}
        eval_set_key = body.get("eval_set", "full")
        log.info("job %s started  type=run models=%s eval_set=%s", job.id, models, eval_set_key)
        prompt_ids = body.get("prompts")
        thinking = body.get("thinking")

        # Load eval set (with fallback to packaged data)
        if eval_set_key == "starter":
            es_path = DATA_DIR / "starter-eval-set.json"
            with open(es_path) as f:
                eval_set = json.load(f)
        else:
            eval_set = _read_eval_set()
        if not eval_set.get("prompts"):
            raise FileNotFoundError("No eval set found \u2014 create one from the Author page or run 'cupel init'")

        prompts = eval_set["prompts"]
        if prompt_ids:
            id_set = set(prompt_ids)
            prompts = [p for p in prompts if p["id"] in id_set]

        job.models = models
        job.prompt_ids = [p["id"] for p in prompts]

        cfg = _read_config()
        if thinking is not None:
            cfg["_thinking_budget"] = thinking
        elif cfg.get("thinking") is not None:
            cfg["_thinking_budget"] = int(cfg["thinking"])

        rubric_by_id = {p["id"]: p.get("rubric", {}) for p in eval_set["prompts"]}
        prompt_by_id = {p["id"]: _prompt_text_for_judge(p) for p in eval_set["prompts"]}

        def _emit(model, prompt_id, status, elapsed=0):
            job.progress.append({
                "model": model, "prompt_id": prompt_id,
                "status": status, "elapsed": elapsed,
                "ts": datetime.now().isoformat(),
            })

        # ── Load image for multimodal prompt ──
        image_b64 = await asyncio.to_thread(
            find_image, cfg.get("image_filename", "what-am-i-looking-at.png"), None
        )

        # ── Setup for saving + judging ──
        output_dir = Path(cfg.get("output_dir", "./eval-results"))
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        thinking_budget = cfg.get("_thinking_budget")
        t_label = f"_think{thinking_budget}" if thinking_budget is not None else ""
        saved_files = []

        # ── Resolve judge config once ──
        judge_model_override = body.get("judge_model")
        if judge_model_override:
            judge_model = judge_model_override
            judge_url, judge_key = _resolve_provider(cfg, judge_model, model_urls_map)
        else:
            judge_model, judge_url, judge_key = get_judge_config(cfg)
        if not judge_model and models:
            judge_model = models[0]
            judge_url, judge_key = _resolve_provider(cfg, judge_model, model_urls_map)

        # ── Run prompts, save, and judge each model ──
        model_urls = {}
        all_results = {m: [] for m in models}
        for model in models:
            model_url, model_key = _resolve_provider(cfg, model, model_urls_map)
            model_urls[model] = model_url
            for p in prompts:
                if job.cancelled:
                    break
                _emit(model, p["id"], "running")
                result, status = await asyncio.to_thread(
                    run_prompt, model_url, model_key, model, p, cfg, image_b64
                )
                all_results[model].append(result)
                job.live_results.setdefault(model, {})[p["id"]] = result
                _emit(model, p["id"], status, result.get("elapsed_seconds", 0))
            if job.cancelled:
                break

            # Save this model's results
            if all_results[model]:
                safe = model.replace("/", "_").replace(" ", "_")
                out = output_dir / f"eval_{safe}{t_label}_{timestamp}.json"
                data = {
                    "model": model, "api_url": model_urls.get(model, ""),
                    "thinking_budget": thinking_budget,
                    "timestamp": timestamp,
                    "results": all_results[model],
                }
                with open(out, "w") as f:
                    json.dump(data, f, indent=2)
                saved_files.append(str(out))

                # Judge this model's results
                if judge_model:
                    for result in data["results"]:
                        if job.cancelled:
                            break
                        pid = result["id"]
                        has_response = result.get("response") or any(r for r in result.get("responses", []))
                        if result.get("skipped") or result.get("error") or not has_response:
                            _emit(model, pid, "skip")
                            continue
                        _emit(model, pid, "judging")
                        try:
                            score, reason = await asyncio.to_thread(
                                score_one, judge_url, judge_key, judge_model,
                                prompt_by_id.get(pid, ""), rubric_by_id.get(pid, {}),
                                result["response"], result.get("responses"),
                            )
                            if score is not None:
                                result["score"] = score
                                result["judge_reason"] = reason
                                result["judge_model"] = judge_model
                                _emit(model, pid, f"scored:{score}", result.get("elapsed_seconds", 0))
                            else:
                                log.warning("judge returned no score  model=%s prompt=#%d: %s", model, pid, reason[:100])
                                result["judge_reason"] = reason
                                _emit(model, pid, "error")
                        except Exception as e:
                            log.error("judge error  model=%s prompt=#%d: %s", model, pid, e)
                            result["judge_reason"] = f"judge error: {e}"
                            _emit(model, pid, "error")

                    # Save scores back
                    data["judge"] = judge_model
                    data["judge_url"] = judge_url
                    with open(out, "w") as f:
                        json.dump(data, f, indent=2)

            if job.cancelled:
                break

        if job.cancelled:
            job.result_files = saved_files
            return

        job.result_files = saved_files
        job.status = "complete"
        log.info("job %s complete  files=%s", job.id, [Path(f).name for f in saved_files])

    except Exception as e:
        if not job.cancelled:
            job.status = "error"
            job.error = str(e)
            log.error("job %s failed: %s", job.id, e)

async def _judge_job(job: Job, body: dict):
    """Execute judging on existing result files."""
    try:
        files = body.get("files", [])
        judge_model_override = body.get("judge_model")

        cfg = _read_config()
        eval_set = _read_eval_set()
        rubric_by_id = {p["id"]: p.get("rubric", {}) for p in eval_set.get("prompts", [])}
        prompt_by_id = {p["id"]: _prompt_text_for_judge(p) for p in eval_set.get("prompts", [])}

        if judge_model_override:
            judge_model = judge_model_override
            # Try to get URL/key for this model from config
            _, judge_url, judge_key = get_judge_config(cfg)
        else:
            judge_model, judge_url, judge_key = get_judge_config(cfg)

        if not judge_model:
            raise ValueError("No judge model configured. Set judge.model in config.yml.")

        # Load data files
        data_files = []
        for fpath in files:
            p = Path(fpath)
            if not p.is_absolute():
                p = RESULTS_DIR / Path(fpath).name
            if not p.exists():
                raise FileNotFoundError(f"Result file not found: {fpath}")
            with open(p) as f:
                data_files.append((str(p), json.load(f)))

        def on_progress(model, prompt_id, status, elapsed):
            job.progress.append({
                "model": model,
                "prompt_id": prompt_id,
                "status": status,
                "elapsed": elapsed,
                "ts": datetime.now().isoformat(),
            })

        await asyncio.to_thread(
            judge_results, data_files, judge_model, judge_url, judge_key,
            rubric_by_id, prompt_by_id, on_progress,
        )

        job.result_files = files
        job.status = "complete"

    except Exception as e:
        job.status = "error"
        job.error = str(e)

@app.get("/api/jobs")
async def list_jobs():
    return [
        {
            "id": j.id,
            "type": j.type,
            "status": j.status,
            "created_at": j.created_at,
            "progress_count": len(j.progress),
            "result_files": j.result_files,
            "error": j.error,
            "models": j.models,
            "prompt_ids": j.prompt_ids,
        }
        for j in jobs.values()
    ]

@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "id": job.id,
        "type": job.type,
        "status": job.status,
        "progress": job.progress,
        "result_files": job.result_files,
        "created_at": job.created_at,
        "error": job.error,
        "models": job.models,
        "prompt_ids": job.prompt_ids,
    }

@app.get("/api/jobs/{job_id}/prompt-detail/{prompt_id}")
async def get_prompt_detail(job_id: str, prompt_id: int):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    details = {}
    for model in job.models:
        result = job.live_results.get(model, {}).get(prompt_id)
        if result:
            details[model] = result
    return {"prompt_id": prompt_id, "models": details}

@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status == "running":
        job.cancelled = True
        job.status = "cancelled"
        job.error = "Cancelled by user"
    return {"id": job.id, "status": job.status}

@app.get("/api/jobs/{job_id}/stream")
async def stream_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        sent = 0
        while True:
            # Send any new progress events
            while sent < len(job.progress):
                event = job.progress[sent]
                yield f"data: {json.dumps(event)}\n\n"
                sent += 1

            # Check if job is done
            if job.status in ("complete", "error", "cancelled"):
                if job.status == "complete":
                    yield f"data: {json.dumps({'type': 'complete', 'result_files': job.result_files})}\n\n"
                elif job.status == "cancelled":
                    yield f"data: {json.dumps({'type': 'cancelled', 'result_files': job.result_files})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'error', 'error': job.error})}\n\n"
                break

            await asyncio.sleep(0.3)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

# ──────────────────────────────────────────────
# Routes: generate prompt (LLM-assisted)
# ──────────────────────────────────────────────

@app.post("/api/generate-prompt")
async def generate_prompt(request: Request):
    """Use LLM to help generate an eval prompt from a description."""
    import re
    body = await request.json()
    description = body.get("description", "")
    category = body.get("category", "general")

    if not description:
        raise HTTPException(status_code=400, detail="Description required")

    cfg = _read_config()

    # Pick a model: use explicitly provided, first configured, or error
    model = body.get("model")
    if not model:
        models = cfg.get("models", [])
        if not models:
            raise HTTPException(status_code=400, detail="No models configured")
        model = models[0]
    model_url, model_key = _resolve_provider(cfg, model)

    try:
        prompt_text = (
            "You are helping create evaluation prompts for benchmarking LLMs. "
            "Given a topic description, generate a well-crafted prompt and a 0-3 scoring rubric.\n\n"
            "Respond with ONLY a JSON object (no markdown fences, no explanation):\n"
            '{"title": "short title", "category": "category_name", "prompt": "the full prompt text", '
            '"rubric": {"3": "criteria for 3", "2": "criteria for 2", "1": "criteria for 1", "0": "criteria for 0"}}\n\n'
            f"Topic: {description}\nCategory: {category}\n\nGenerate an eval prompt with rubric."
        )
        resp = await asyncio.to_thread(
            call_llm, model_url, model_key, model,
            prompt_text,
            temperature=0.7, max_tokens=8192,
        )
        content = resp["content"]
        stats = {
            "elapsed": resp.get("elapsed_seconds", 0),
            "prompt_tokens": resp.get("prompt_tokens", 0),
            "completion_tokens": resp.get("completion_tokens", 0),
        }
        log.info("generate-prompt raw content (%d chars): %s", len(content), content[:300])

        def _try_parse_json(text):
            """Extract JSON object — handles thinking-text prefixes and truncated output."""
            # 1. Search backwards from last } (skips stray { in thinking text)
            last_brace = text.rfind('}')
            if last_brace >= 0:
                for i in range(last_brace, -1, -1):
                    if text[i] == '{':
                        try:
                            return json.loads(text[i:last_brace + 1])
                        except (json.JSONDecodeError, ValueError):
                            continue

            # 2. Truncated JSON — model ran out of tokens mid-object.
            #    Find the first { and try to close open strings/braces.
            first = text.find('{')
            if first < 0:
                return None
            fragment = text[first:]
            in_str, esc, stack = False, False, []
            for c in fragment:
                if esc:
                    esc = False
                    continue
                if c == '\\' and in_str:
                    esc = True
                    continue
                if c == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if c in '{[':
                    stack.append('}' if c == '{' else ']')
                elif c in '}]' and stack:
                    stack.pop()
            repaired = fragment
            if in_str:
                repaired += '"'
            while stack:
                repaired += stack.pop()
            try:
                result = json.loads(repaired)
                log.info("generate-prompt: repaired truncated JSON (%d open braces closed)", len(stack) + (1 if in_str else 0))
                return result
            except (json.JSONDecodeError, ValueError):
                return None

        parsed = _try_parse_json(content)

        if parsed and isinstance(parsed, dict):
            # Unwrap: if no recognized keys at top level, look one level deeper
            if "title" not in parsed and "prompt" not in parsed:
                for v in parsed.values():
                    if isinstance(v, dict):
                        parsed = v
                        break

            # Normalize field names — LLMs use many variants
            def _pick(d, *keys):
                for k in keys:
                    if k in d and d[k]:
                        return d[k]
                return None

            rubric = _pick(parsed, "rubric", "Rubric", "scoring_rubric", "scoring", "criteria") or {}
            if isinstance(rubric, str):
                try:
                    rubric = json.loads(rubric)
                except (json.JSONDecodeError, TypeError):
                    rubric = {"3": rubric}

            generated = {
                "title": _pick(parsed, "title", "Title", "name", "Name") or "",
                "prompt": _pick(parsed, "prompt", "Prompt", "prompt_text", "question", "Question", "text", "Text") or "",
                "category": _pick(parsed, "category", "Category") or category,
                "rubric": rubric,
            }
            log.info("generate-prompt normalized: title=%r prompt=%d chars rubric_keys=%s",
                      generated["title"], len(generated["prompt"]),
                      list(generated["rubric"].keys()) if isinstance(generated["rubric"], dict) else "?")
            return {"status": "ok", "prompt": generated, "model": model, **stats}
        else:
            log.warning("generate-prompt: no valid JSON found in %d chars of content", len(content))
            return {"status": "ok", "raw": content, "model": model, **stats}
    except Exception as e:
        log.error("generate-prompt failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

# ──────────────────────────────────────────────
# Mount static files LAST (catch-all)
# ──────────────────────────────────────────────

if UI_DIR.exists():
    app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")

# ──────────────────────────────────────────────
# Start UI helper
# ──────────────────────────────────────────────

def _start_ui(port=8042):
    """Start the web UI server."""
    import uvicorn
    import webbrowser
    import threading

    providers = discover_providers()
    online = [p for p in providers if p["status"] == "online"]

    print(f"\n  cupel {__version__}")
    print(f"  opening dashboard \u2192 http://localhost:{port}\n")
    for p in online:
        models = ", ".join(p["models"][:3])
        extra = f" +{len(p['models'])-3} more" if len(p["models"]) > 3 else ""
        print(f"  \u25cf {p['url']} \u2014 {len(p['models'])} models")
    for p in providers:
        if p["status"] == "offline":
            print(f"  \u25cb {p['url']} \u2014 offline")
    print()

    threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

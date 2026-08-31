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
    get_providers_config, resolve_api_key_for_port, resolve_path,
)
from cupel.eval import (
    judge_results, score_one, judge_one, rubrics_for_run,
    _prompt_text_for_judge, run_prompt, call_llm, find_image,
)
from cupel.discovery import detect_hardware, discover_providers, detect_thermal
from cupel.stats import (
    score_pct_ci, check_pct_ci, run_group_key,
    measured_noise_floor, rank_within_noise,
)
from cupel.schema import (
    eval_set_meta, normalize_run, refresh_judges, record_judge_error,
)

import yaml

# ── Logging ──
log = logging.getLogger("cupel")
log.setLevel(logging.INFO)
_log_dir = Path.home() / ".cupel"
_log_dir.mkdir(parents=True, exist_ok=True)
_log_path = _log_dir / "cupel.log"
_fh = logging.FileHandler(_log_path)
_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
log.addHandler(_fh)

# Route uvicorn access & error logs to the same file
for _uv_name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
    _uv_log = logging.getLogger(_uv_name)
    _uv_log.addHandler(_fh)

class _ThermalFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "/api/thermal" in msg and "200" in msg:
            return False
        return True

logging.getLogger("uvicorn.access").addFilter(_ThermalFilter())

BASE_DIR = Path(__file__).parent.parent   # repo root
PKG_DIR = Path(__file__).parent           # package dir (cupel/)
RESULTS_DIR = Path.home() / ".cupel" / "eval-results"
DATA_DIR = PKG_DIR / "data"
UI_DIR = PKG_DIR / "ui"
TAGS_FILE = RESULTS_DIR / ".tags.json"
HIDDEN_FILE = RESULTS_DIR / ".hidden.json"

app = FastAPI(title="cupel", version=__version__)

# Load .env on import (needed for uvicorn --reload mode)
load_dotenv()

# Only the local UI may call the API. A wildcard let any page the user happened to
# be browsing reach this server — which can start jobs, rewrite config.yml, and
# write files. (A wildcard origin with credentials is also rejected by browsers.)
# CUPEL_ALLOWED_ORIGINS takes a comma-separated list for remote/proxied setups.
_allowed_origins = [
    o.strip() for o in os.environ.get("CUPEL_ALLOWED_ORIGINS", "").split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    # any port on loopback, so --port works without extra configuration
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_origins=_allowed_origins,
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

# background-polled thermal state — avoids blocking requests during swift cold compile
_thermal_cache = {"state": None}

async def _thermal_loop():
    """poll thermal state every 30s in the background."""
    while True:
        try:
            _thermal_cache["state"] = await asyncio.to_thread(detect_thermal)
        except Exception:
            pass
        await asyncio.sleep(30)

@app.on_event("startup")
async def _start_thermal_poller():
    asyncio.create_task(_thermal_loop())

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _config_path() -> Path:
    return Path.home() / ".cupel" / "config.yml"

def _eval_set_path() -> Path:
    cfg = _read_config()
    return resolve_path(cfg.get("eval_set", "eval-sets/eval-set.json"))

def _read_config() -> dict:
    """Read config.yml, returning defaults if missing."""
    cfg, _ = load_config()
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

def _eval_set_label(data: dict) -> str:
    """Name of the eval set a run used.

    Older runs stored a bare name string (or nothing at all); newer ones store an
    object carrying the path and content hash.
    """
    es = data.get("eval_set")
    if isinstance(es, dict):
        return es.get("name", "")
    return es or ""


def _provider_label(api_url: str, local_hw: dict) -> dict:
    """Hardware/provider badge for a run: local machine specs, or the cloud host."""
    if not api_url or "localhost" in api_url or "127.0.0.1" in api_url:
        return local_hw
    for host, name in (("openrouter.ai", "OpenRouter"),
                       ("anthropic.com", "Anthropic"),
                       ("openai.com", "OpenAI")):
        if host in api_url:
            return {"name": name, "memory": ""}
    from urllib.parse import urlparse
    return {"name": urlparse(api_url).hostname or api_url, "memory": ""}


# A run scored on only a handful of its prompts is not comparable to a full run.
# Scoring out of what was actually scored (rather than out of every prompt) is
# correct per run, but with no floor a run that errored on 21 of 23 prompts and
# happened to ace the other two would report 100% and sort to the top.
MIN_COVERAGE = 0.8


def _coverage(results: list) -> float:
    scored = sum(1 for r in results if r.get("score") is not None)
    return scored / len(results) if results else 0.0


_hw_cache: dict = {}


def _cached_hardware() -> dict:
    """Machine specs, detected once per process.

    `detect_hardware()` shells out to system_profiler / nvidia-smi with multi-second
    timeouts. The hardware does not change while the server runs, so paying for it
    on every leaderboard request is pure latency.
    """
    if "hw" not in _hw_cache:
        _hw_cache["hw"] = detect_hardware()
    return _hw_cache["hw"]


# Leaderboard responses, keyed by the metric and the mtimes of the files they were
# built from. The bootstrap intervals cost ~0.26s per request (110k resamples over
# 55 runs) and recomputing them for an unchanged set of files buys nothing.
_lb_cache: dict = {}


def _results_signature() -> tuple:
    """Identity of the current result set — any write invalidates the cache."""
    try:
        return tuple(sorted((p.name, p.stat().st_mtime_ns) for p in _result_files()))
    except OSError:
        return ()


def _group_entries(entries: list) -> list:
    """Collapse repeat runs of the same configuration into one row.

    Two runs of the same model, eval set, judge and settings are two samples of one
    thing, not two competitors. Listing them separately is how the leaderboard came
    to show `Qwen3.6-27B-8bit` twice and `Qwen3.6-35B-A3B-bf16` three times, ranked
    against each other on differences smaller than their own run-to-run spread.
    """
    groups: dict = {}
    for e in entries:
        groups.setdefault(e["_group_key"], []).append(e)

    out = []
    for members in groups.values():
        members.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
        latest = members[0]

        pooled = [s for e in members for s in e["_scores"]]
        if not pooled:
            continue
        per_run_pct = [e["pct"] for e in members]
        pct = 100.0 * sum(pooled) / (len(pooled) * 3)
        ci_lo, ci_hi = score_pct_ci(pooled)

        # the criteria vector, where the eval set provides one. A run has it only if
        # every scored prompt does — a partial vector is not comparable to a full one.
        pooled_check = [c for e in members for c in e.get("_check_scores", [])]
        has_check = bool(pooled_check) and len(pooled_check) == len(pooled)
        check_pct = check_lo = check_hi = None
        if has_check:
            check_pct = round(100.0 * sum(pooled_check) / len(pooled_check), 1)
            lo, hi = check_pct_ci(pooled_check)
            check_lo, check_hi = round(lo, 1), round(hi, 1)

        # average each prompt's score across the runs, so the per-prompt ticks and
        # the category breakdown describe the group rather than one arbitrary run
        by_prompt: dict = {}
        for e in members:
            for sp in e.get("scores_by_prompt", []):
                acc = by_prompt.setdefault(sp["id"], {
                    "id": sp["id"], "category": sp.get("category", ""),
                    "title": sp.get("title", ""), "_vals": [], "_el": [], "_tok": [],
                })
                if sp.get("score") is not None:
                    acc["_vals"].append(sp["score"])
                acc["_el"].append(sp.get("elapsed", 0))
                acc["_tok"].append(sp.get("tokens", 0))

        scores_by_prompt = []
        for pid in sorted(by_prompt):
            acc = by_prompt[pid]
            vals = acc.pop("_vals"); el = acc.pop("_el"); tok = acc.pop("_tok")
            acc["score"] = round(sum(vals) / len(vals), 2) if vals else None
            acc["elapsed"] = round(sum(el) / len(el), 1) if el else 0
            acc["tokens"] = round(sum(tok) / len(tok)) if tok else 0
            scores_by_prompt.append(acc)

        entry = dict(latest)
        entry.pop("_group_key", None)
        entry.pop("_scores", None)
        entry.update({
            "pct": round(pct, 1),
            "ci_lo": round(ci_lo, 1),
            "ci_hi": round(ci_hi, 1),
            "has_check": has_check,
            "check_pct": check_pct,
            "check_ci_lo": check_lo,
            "check_ci_hi": check_hi,
            "total_score": sum(pooled),
            "max_score": len(pooled) * 3,
            "n_runs": len(members),
            "per_run_pct": [round(p, 1) for p in per_run_pct],
            "spread": round(max(per_run_pct) - min(per_run_pct), 1) if len(members) > 1 else 0.0,
            "scores_by_prompt": scores_by_prompt,
            "runs": [{"filename": m["filename"], "pct": m["pct"],
                      "timestamp": m.get("timestamp", "")} for m in members],
        })
        out.append(entry)
    return out


def _summarize_result(path: Path, data: dict) -> dict:
    """Build summary metadata for a result file."""
    results = data.get("results", [])
    scored = [r for r in results if r.get("score") is not None]
    total_score = sum(r["score"] for r in scored)
    # Denominator counts only what was actually scored. An errored or skipped
    # prompt is missing data, not a zero — folding it in penalises a model for
    # its provider's rate limiter.
    max_score = len(scored) * 3
    total_elapsed = sum(r.get("elapsed_seconds", 0) for r in results)
    total_tokens = sum(r.get("completion_tokens", 0) for r in results)

    return {
        "filename": path.name,
        "model": data.get("model", "unknown"),
        "timestamp": data.get("timestamp", ""),
        "eval_set": _eval_set_label(data),
        "judge": data.get("judge", ""),
        "judges": [j.get("model", "") for j in data.get("judges", []) if j.get("model")],
        "notes": data.get("notes", ""),
        "num_prompts": len(results),
        "num_scored": len(scored),
        "n_errors": sum(1 for r in results if r.get("error")),
        "n_skipped": sum(1 for r in results if r.get("skipped")),
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
            "max_tokens": 16384,
            "thinking": None,
        }
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
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

def _read_pyproject_version():
    """Read version from pyproject.toml so it stays current even without reinstall."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if pyproject.exists():
        for line in pyproject.read_text().splitlines():
            if line.startswith("version"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return __version__

@app.get("/api/version")
async def get_version():
    return {"version": _read_pyproject_version()}

@app.get("/api/hardware")
async def get_hardware():
    hw = await asyncio.to_thread(_cached_hardware)
    return hw

@app.get("/api/thermal")
async def get_thermal():
    return _thermal_cache

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
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w") as f:
        yaml.dump(body, f, default_flow_style=False, sort_keys=False)
    return {"status": "ok"}

# ──────────────────────────────────────────────
# Routes: eval set
# ──────────────────────────────────────────────

@app.get("/api/eval-sets")
async def list_eval_sets():
    es_dir = Path.home() / ".cupel" / "eval-sets"
    if not es_dir.is_dir():
        return []
    files = sorted(p.name for p in es_dir.glob("*.json"))
    return [f"eval-sets/{f}" for f in files]

@app.post("/api/import-eval-set")
async def import_eval_set(request: Request):
    body = await request.json()
    content = body.get("content")
    filename = body.get("filename", "")
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        raise HTTPException(400, detail="File is not valid JSON")
    if not isinstance(data, dict):
        raise HTTPException(400, detail="Eval set must be a JSON object, got " + type(data).__name__)
    if "name" not in data or not isinstance(data.get("name"), str) or not data["name"].strip():
        raise HTTPException(400, detail='Eval set must have a non-empty "name" field')
    if "prompts" not in data or not isinstance(data.get("prompts"), list):
        raise HTTPException(400, detail='Eval set must have a "prompts" array')
    # Take only the basename. `es_dir / "../../x.json"` resolves outside the eval-set
    # directory and writes anywhere the process can reach — verified exploitable.
    safe_name = Path(filename).name
    if not safe_name or not safe_name.endswith(".json"):
        raise HTTPException(400, detail="Filename must be a plain *.json name")

    es_dir = Path.home() / ".cupel" / "eval-sets"
    es_dir.mkdir(parents=True, exist_ok=True)
    dest = (es_dir / safe_name).resolve()
    # belt and braces: refuse anything that still escapes (symlinks, odd names)
    if es_dir.resolve() not in dest.parents:
        raise HTTPException(400, detail="Refusing to write outside the eval-sets directory")

    with open(dest, "w") as f:
        f.write(content)
    return {"path": f"eval-sets/{safe_name}"}

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
                data = normalize_run(json.load(f))
            summary = _summarize_result(path, data)
            summary["tags"] = tags.get(path.name, [])
            summary["muted"] = path.name in hidden
            results.append(summary)
        except (json.JSONDecodeError, KeyError) as e:
            log.warning("skipping corrupt result file %s: %s", path.name, e)
            continue
    return results

@app.get("/api/results/leaderboard")
async def get_leaderboard(metric: str = "auto"):
    sig = (metric, _results_signature(), tuple(sorted(_load_hidden())))
    if _lb_cache.get("sig") == sig:
        return _lb_cache["payload"]

    hw = await asyncio.to_thread(_cached_hardware)
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
        except (json.JSONDecodeError, KeyError) as e:
            log.warning("skipping corrupt result file %s: %s", path.name, e)
            continue

        model = data.get("model", "unknown")
        scored = [r for r in data.get("results", []) if r.get("score") is not None]
        if not scored:
            continue

        user_models.add(model)

        total_score = sum(r["score"] for r in scored)
        # Only scored prompts count toward the denominator — see _summarize_result.
        max_score = len(scored) * 3
        total_elapsed = sum(r.get("elapsed_seconds", 0) for r in data["results"])
        total_tokens = sum(r.get("completion_tokens", 0) for r in data["results"])
        num_prompts = len(data["results"])

        # Use hardware stored at run time; fall back to detection for old files
        api_url = data.get("api_url", "")
        entry_hw = data.get("hardware") or _provider_label(api_url, hw)

        ci_lo, ci_hi = score_pct_ci([r["score"] for r in scored])
        coverage = _coverage(data["results"])

        entries_list.append({
            "_group_key": run_group_key(data),
            "_scores": [r["score"] for r in scored],
            "_check_scores": [r["check_score"] for r in scored if r.get("check_score") is not None],
            "model": model,
            "total_score": total_score,
            "max_score": max_score,
            "pct": round(total_score / max_score * 100, 1) if max_score > 0 else 0,
            "ci_lo": round(ci_lo, 1),
            "ci_hi": round(ci_hi, 1),
            "coverage": round(coverage, 3),
            "low_coverage": coverage < MIN_COVERAGE,
            "n_errors": sum(1 for r in data["results"] if r.get("error")),
            "n_skipped": sum(1 for r in data["results"] if r.get("skipped")),
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
            "notes": data.get("notes", ""),
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
                entry_hw = _provider_label(api_url, hw)
                scored = [r for r in results if r.get("score") is not None]
                if not scored:
                    continue
                total_score = sum(r["score"] for r in scored)
                max_score = len(scored) * 3
                total_elapsed = sum(r.get("elapsed_seconds", 0) for r in results)
                total_tokens = sum(r.get("completion_tokens", 0) for r in results)
                num_prompts = len(results)
                ci_lo, ci_hi = score_pct_ci([r["score"] for r in scored])

                entries_list.append({
                    "_group_key": ("example", model),
                    "_scores": [r["score"] for r in scored],
                    "_check_scores": [r["check_score"] for r in scored if r.get("check_score") is not None],
                    "coverage": round(_coverage(results), 3),
                    "low_coverage": _coverage(results) < MIN_COVERAGE,
                    "model": model,
                    "total_score": total_score,
                    "max_score": max_score,
                    "pct": round(total_score / max_score * 100, 1) if max_score > 0 else 0,
                    "ci_lo": round(ci_lo, 1),
                    "ci_hi": round(ci_hi, 1),
                    "n_errors": sum(1 for r in results if r.get("error")),
                    "n_skipped": sum(1 for r in results if r.get("skipped")),
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
        except (json.JSONDecodeError, KeyError) as e:
            log.warning("skipping corrupt example file: %s", e)

    # Sort by percentage descending
    entries = _group_entries(entries_list)

    # Rank on the criteria vector when every entry has one. Ranking a mix of
    # criteria-scored and level-scored runs on one axis would compare two different
    # measurements, so it falls back to 0-3 unless the whole board can use it.
    use_check = bool(entries) and all(e.get("has_check") for e in entries)
    if metric == "score":
        use_check = False
    elif metric == "check" and not use_check:
        raise HTTPException(400, detail="not every run has criteria scores to rank on")

    if use_check:
        for e in entries:
            e["pct"] = e["check_pct"]
            e["ci_lo"], e["ci_hi"] = e["check_ci_lo"], e["check_ci_hi"]

    # incomplete runs sort below complete ones regardless of percentage
    entries.sort(key=lambda x: (not x.get("low_coverage", False), x["pct"]), reverse=True)

    # Ties come from measured run-to-run variation, not a modelled interval: the same
    # config run twice landed this far apart, so smaller gaps are not orderings.
    noise = measured_noise_floor(e.get("per_run_pct", []) for e in entries)
    rank_within_noise(entries, noise["floor"] if noise else None)

    # Collect unique prompts across all entries
    prompt_ids_seen = set()
    prompts_list = []
    for entry in entries:
        for sp in entry.get("scores_by_prompt", []):
            pid = sp["id"]
            if pid not in prompt_ids_seen:
                prompt_ids_seen.add(pid)
                prompts_list.append({"id": pid, "category": sp.get("category", ""), "title": sp.get("title", "")})

    # Denominators now vary per entry (a run with errors has fewer scored prompts),
    # so the headline figure is the most any entry could have attained.
    max_score = max((e["max_score"] for e in entries), default=0)
    payload = {"entries": entries, "prompts": prompts_list, "max_score": max_score,
               "metric": "check_score" if use_check else "score",
               "check_available": bool(entries) and all(e.get("has_check") for e in entries),
               "noise": noise}
    _lb_cache["sig"], _lb_cache["payload"] = sig, payload
    return payload

@app.get("/api/results/{filename}")
async def get_result(filename: str):
    path = RESULTS_DIR / filename
    if not path.exists() or not path.name.startswith("eval_"):
        raise HTTPException(status_code=404, detail="Result not found")
    with open(path) as f:
        # normalise on read so the detail view sees `judgments` even for runs
        # saved before judgments existed — the file itself is left alone
        return normalize_run(json.load(f))

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
        except (json.JSONDecodeError, KeyError) as e:
            log.warning("skipping corrupt result file %s: %s", path.name, e)
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
        except (json.JSONDecodeError, KeyError) as e:
            log.warning("skipping corrupt example file: %s", e)

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
        run_notes = body.get("notes", "")

        # Load eval set (with fallback to packaged data)
        if eval_set_key == "starter":
            es_path = DATA_DIR / "starter-eval-set.json"
            with open(es_path) as f:
                eval_set = json.load(f)
        else:
            es_path = _eval_set_path()
            eval_set = _read_eval_set()
        es_meta = eval_set_meta(es_path, eval_set)
        if not eval_set.get("prompts"):
            raise FileNotFoundError("No eval set found \u2014 create one from the Author page or run 'cupel init'")

        if not run_notes:
            if eval_set_key == "starter":
                run_notes = "starter"
            else:
                run_notes = _eval_set_path().stem

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
        output_dir = resolve_path(cfg.get("output_dir", "./eval-results"))
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hw = await asyncio.to_thread(_cached_hardware)
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
                if result.get("error"):
                    log.warning("prompt #%d skipped  model=%s: %s", p["id"], model, result["error"])
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
                    # identity of what was run, so re-judging and grouping can
                    # match this run to the exact prompts and rubrics it used
                    "eval_set": es_meta,
                    "temperature": cfg.get("temperature"),
                    "notes": run_notes,
                    "hardware": hw,
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
                        _emit(model, pid, "judging")
                        status, score = await asyncio.to_thread(
                            judge_one, result, rubric_by_id.get(pid),
                            prompt_by_id.get(pid, ""), judge_url, judge_key, judge_model,
                        )
                        if status == "scored":
                            _emit(model, pid, f"scored:{score}", result.get("elapsed_seconds", 0))
                        else:
                            _emit(model, pid, status)

                    # Save scores back
                    refresh_judges(data)
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
    """Re-score existing result files, appending judgments rather than replacing."""
    try:
        files = body.get("files", [])
        model_urls_map = body.get("model_urls") or {}
        replace = bool(body.get("replace"))

        # accept a single judge or a list — scoring with several judges makes
        # their disagreement measurable, which is the real diagnostic
        judge_list = body.get("judges") or []
        if not judge_list and body.get("judge_model"):
            judge_list = [body["judge_model"]]

        cfg = _read_config()
        if not judge_list:
            configured, _, _ = get_judge_config(cfg)
            if configured:
                judge_list = [configured]
        if not judge_list:
            raise ValueError("No judge model configured. Set judge.model in config.yml.")

        def _emit(model, prompt_id, status, elapsed=0):
            job.progress.append({
                "model": model, "prompt_id": prompt_id,
                "status": status, "elapsed": elapsed,
                "ts": datetime.now().isoformat(),
            })

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

        job.models = [d.get("model", "") for _, d in data_files]

        for idx, judge_model in enumerate(judge_list):
            # `replace` clears prior judgments once, on the first judge — later
            # judges in the same request must append or they would wipe each other
            replace_this = replace and idx == 0
            judge_url, judge_key = _resolve_provider(cfg, judge_model, model_urls_map)
            log.info("job %s judging  judge=%s url=%s files=%d",
                     job.id, judge_model, judge_url, len(data_files))

            # Rubrics come from each run's own eval set, so a mixed selection of
            # files scored against different sets is still judged correctly.
            for fpath, data in data_files:
                if job.cancelled:
                    break
                # no forced set: use what the run recorded, else identify it.
                # Never silently substitute the currently-configured eval set.
                rubric_by_id, prompt_by_id, _, warning = rubrics_for_run(data)
                if not rubric_by_id:
                    log.error("job %s  %s: %s", job.id, Path(fpath).name, warning)
                    _emit(data.get("model", ""), 0, f"error:{warning}")
                    continue
                if warning:
                    log.warning("job %s  %s: %s", job.id, Path(fpath).name, warning)
                    _emit(data.get("model", ""), 0, f"warning:{warning}")

                await asyncio.to_thread(
                    judge_results, [(fpath, data)], judge_model, judge_url, judge_key,
                    rubric_by_id, prompt_by_id, _emit, replace_this,
                )
            if job.cancelled:
                break

        job.result_files = files
        job.status = "cancelled" if job.cancelled else "complete"

    except Exception as e:
        log.error("job %s judge failed: %s", job.id, e)
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

def _start_ui(port=8042, host="127.0.0.1"):
    """Start the web UI server.

    Binds loopback by default. The server has no authentication and its API can
    start jobs, rewrite config.yml, and read which API keys are set — on 0.0.0.0
    that is reachable by anything on the network. Pass --host to opt out.
    """
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

    if host not in ("127.0.0.1", "localhost"):
        print(f"  \u26a0 listening on {host} \u2014 the API is unauthenticated and reachable")
        print(f"    by anything that can route to this machine\n")

    threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    uvicorn.run(app, host=host, port=port, log_level="info")

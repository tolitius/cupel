"""cupel.publish — build the public leaderboard snapshot for cupel.run/bench.

The snapshot is committed to a public repository, so every byte of it is treated
as permanent. Two rules keep it safe:

  * allowlist construction — every output object is a literal dict of named
    fields; nothing is copied from a result file and trimmed;
  * validate the output — `validate_snapshot` checks the finished object against
    the schema and scans it for any text taken from the source files.

The exporter does all the math. The public page only selects a precomputed
variant, so no ranking or interval logic lives in JavaScript.
"""

import json
import os
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from cupel.board import KNOWN_PROVIDERS, MIN_COVERAGE, coverage, provider_label
from cupel.schema import eval_set_meta, infer_eval_set
from cupel.stats import (
    check_pct_ci, measured_noise_floor, rank_within_noise, run_group_key, score_pct_ci,
)

SCHEMA_VERSION = 1
SUBSETS = ("all", "text")
METRICS = ("score", "check")
TEXT_EXCLUDED = "multimodal"
SNAPSHOT_RELPATH = Path("site") / "bench" / "bench.json"

# the publisher's live view, used as the public page's initial state
VIEW_ENUMS = {
    "text": (True, False),
    "metric": ("score", "check"),
    "chart_tab": (None, "scatter", "bar", "radar"),
    "sort_col": ("rank", "model", "score", "pts", "pct", "delta", "speed", "avg"),
    "sort_dir": ("asc", "desc"),
}
VIEW_DEFAULTS = {"text": False, "metric": "score", "chart_tab": None,
                 "sort_col": "score", "sort_dir": "desc"}
FROZEN_KEYS = ("local", "notes")

PUBLIC_PROVIDER_NAMES = tuple(name for _, name in KNOWN_PROVIDERS)
UNKNOWN_PROVIDER = "remote"


class PublishError(Exception):
    """The snapshot cannot be built from the current results."""


# ──────────────────────────────────────────────
# request validation
# ──────────────────────────────────────────────

def check_view(view: dict | None) -> dict:
    view = {} if view is None else view
    if not isinstance(view, dict):
        raise PublishError("view must be an object")
    unknown = set(view) - set(VIEW_ENUMS)
    if unknown:
        raise PublishError(f"unknown view keys: {', '.join(sorted(map(str, unknown)))}")
    out = {}
    for key, allowed in VIEW_ENUMS.items():
        val = view.get(key, VIEW_DEFAULTS[key])
        # bools only match bools: `1 in (True, False)` is true in Python
        if key == "text":
            if not isinstance(val, bool):
                raise PublishError("view.text must be true or false")
        elif isinstance(val, bool) or val not in allowed:
            raise PublishError(f"view.{key} must be one of {list(allowed)}")
        out[key] = val
    return out


def check_frozen(frozen: dict | None) -> dict:
    frozen = {} if frozen is None else frozen
    if not isinstance(frozen, dict):
        raise PublishError("frozen must be an object")
    unknown = set(frozen) - set(FROZEN_KEYS)
    if unknown:
        raise PublishError(f"unknown frozen keys: {', '.join(sorted(map(str, unknown)))}")
    out = {}
    for key in FROZEN_KEYS:
        val = frozen.get(key, False)
        if not isinstance(val, bool):
            raise PublishError(f"frozen.{key} must be true or false")
        out[key] = val
    return out


# ──────────────────────────────────────────────
# small public-field helpers
# ──────────────────────────────────────────────

def public_hardware(hw) -> dict:
    """Name and memory only. A cloud row is a known provider label or `remote`.

    `provider_label` falls back to the raw hostname of `api_url`, which can be a
    LAN address or an internal domain, so anything unrecognised becomes `remote`.
    """
    hw = hw if isinstance(hw, dict) else {}
    name = str(hw.get("name") or "").strip()
    memory = str(hw.get("memory") or "").strip()
    if memory:
        return {"name": name, "memory": memory}
    return {"name": name if name in PUBLIC_PROVIDER_NAMES else UNKNOWN_PROVIDER, "memory": ""}


def _eval_set_names(es) -> set[str]:
    """Names a run's eval set goes by: its display name and its file stem."""
    out = set()
    if isinstance(es, dict):
        if es.get("name"):
            out.add(str(es["name"]))
        if es.get("path"):
            out.add(Path(str(es["path"])).stem)
    elif isinstance(es, str) and es:
        out.add(es)
        out.add(Path(es).stem)
    return {n for n in out if n.strip()}


def _candidate_names(eval_sets) -> set[str]:
    out = set()
    for path, data in eval_sets:
        out.add(Path(path).stem)
        if isinstance(data, dict) and data.get("name"):
            out.add(str(data["name"]))
    return {n for n in out if n.strip()}


def _public_notes(run: dict, name_like: set[str]) -> str:
    """Run-level notes, or "" when they are just the eval set's name.

    `cupel run` defaults notes to the eval set's file stem, so most notes are not
    notes at all — they would publish the private set's name.
    """
    notes = run.get("notes")
    if not isinstance(notes, str):
        return ""
    notes = notes.strip()
    names = {n.strip().casefold() for n in name_like | _eval_set_names(run.get("eval_set"))}
    if not notes or notes.casefold() in names:
        return ""
    return notes


def _half_up(x: float) -> int:
    return int(x + 0.5)


def _id_key(pid):
    return (0, pid, "") if isinstance(pid, (int, float)) else (1, 0, str(pid))


# ──────────────────────────────────────────────
# loading and selection
# ──────────────────────────────────────────────

def _is_sample(path: Path) -> bool:
    return path.name == "example-run.json"


def _scored(results: list) -> list:
    return [r for r in results if r.get("score") is not None]


def _run_hash(run: dict, candidates: list) -> str | None:
    es = run.get("eval_set")
    if isinstance(es, dict) and es.get("hash"):
        return str(es["hash"])
    path, data, _ = infer_eval_set(run, [(p, d) for p, d, _h in candidates])
    if path is None:
        return None
    for p, _d, h in candidates:
        if p == path:
            return h
    return None


# ──────────────────────────────────────────────
# per-group variants
# ──────────────────────────────────────────────

def _subset(results: list, subset: str) -> list:
    if subset == "text":
        return [r for r in results if r.get("category", "") != TEXT_EXCLUDED]
    return results


def _variant(members: list, subset: str, metric: str, categories: list):
    """One (subset, metric) cell of a group. Returns (variant, per_run_pcts) or (None, [])."""
    subs = [_subset(m["results"], subset) for m in members]
    scored_by_member = [_scored(s) for s in subs]
    pooled_results = [r for sc in scored_by_member for r in sc]
    if not pooled_results:
        return None, []

    cats = [c for c in categories if not (subset == "text" and c == TEXT_EXCLUDED)]
    latest_cov = coverage(subs[0])

    if metric == "score":
        pooled = [r["score"] for r in pooled_results]
        per_run = []
        for sc in scored_by_member:
            if sc:
                total, mx = sum(r["score"] for r in sc), len(sc) * 3
                per_run.append(round(total / mx * 100, 1))
        pct = 100.0 * sum(pooled) / (len(pooled) * 3)
        lo, hi = score_pct_ci(pooled)
        by_cat = {}
        for c in cats:
            vals = [r["score"] for r in pooled_results if r.get("category", "") == c]
            by_cat[c] = _half_up(100.0 * sum(vals) / (len(vals) * 3)) if vals else None
        pts = sum(pooled)
        variant = {
            "pts": round(pts, 2) if isinstance(pts, float) else pts,
            "max": len(pooled) * 3,
            "pct": round(pct, 1),
            "ci_lo": round(lo, 1),
            "ci_hi": round(hi, 1),
        }
    else:
        # the criteria vector exists only when every scored prompt carries one —
        # a partial vector is not comparable to a full one
        if any(r.get("check_score") is None for r in pooled_results):
            return None, []
        pooled = [r["check_score"] for r in pooled_results]
        per_run = [round(100.0 * sum(r["check_score"] for r in sc) / len(sc), 1)
                   for sc in scored_by_member if sc]
        lo, hi = check_pct_ci(pooled)
        by_cat = {}
        for c in cats:
            vals = [r["check_score"] for r in pooled_results if r.get("category", "") == c]
            by_cat[c] = _half_up(100.0 * sum(vals) / len(vals)) if vals else None
        variant = {
            "pct": round(100.0 * sum(pooled) / len(pooled), 1),
            "ci_lo": round(lo, 1),
            "ci_hi": round(hi, 1),
        }

    variant.update({
        "rank": None,
        "tied": False,
        "coverage": round(latest_cov, 3),
        "low_coverage": latest_cov < MIN_COVERAGE,
        "by_category": by_cat,
    })
    return variant, per_run


def _speed(run: dict) -> tuple[float, float]:
    """tok/s and average seconds per prompt, over every prompt, as the server does."""
    results = run["results"]
    elapsed = sum(r.get("elapsed_seconds", 0) or 0 for r in results)
    tokens = sum(r.get("completion_tokens", 0) or 0 for r in results)
    tps = round(tokens / elapsed, 1) if elapsed > 0 else 0
    avg = round(elapsed / len(results), 1) if results else 0
    return tps, avg


def _per_prompt(members: list) -> list:
    acc: dict = {}
    for m in members:
        for r in m["results"]:
            if "id" not in r:
                continue
            a = acc.setdefault(r["id"], {"category": str(r.get("category", "") or ""), "vals": []})
            if r.get("score") is not None:
                a["vals"].append(r["score"])
    out = []
    for pid in sorted(acc, key=_id_key):
        vals = acc[pid]["vals"]
        out.append({
            "id": pid,
            "category": acc[pid]["category"],
            "score": round(sum(vals) / len(vals), 2) if vals else None,
        })
    return out


# ──────────────────────────────────────────────
# build
# ──────────────────────────────────────────────

def build_snapshot(result_paths: list[Path], hidden, view: dict, frozen: dict,
                   handle: str, cupel_version: str, now: datetime,
                   eval_sets: list | None = None, local_hw: dict | None = None):
    """Build the public snapshot. Returns (snapshot, summary).

    `summary` is for the publisher's preview only and is never written.
    `eval_sets` is a list of (path, eval_set_dict) used to identify runs that
    predate stored eval-set hashes. `local_hw` labels old local runs that did not
    store their hardware, as the leaderboard does.
    """
    view = check_view(view)
    frozen = check_frozen(frozen)
    hidden = set(hidden or ())
    eval_sets = list(eval_sets or [])
    local_hw = local_hw or {}
    candidates = [(Path(p), d, eval_set_meta(p, d)["hash"]) for p, d in eval_sets]
    name_like = _candidate_names(eval_sets)

    dropped = {"muted": 0, "local": 0, "other_eval_set": 0, "unidentified_eval_set": 0}

    # 1-2. load, skip muted / unscored / sample, apply frozen filters at run level
    runs = []
    for path in result_paths:
        path = Path(path)
        if _is_sample(path):
            continue
        if path.name in hidden:
            dropped["muted"] += 1
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            continue
        if not _scored(data["results"]):
            continue
        hw = data.get("hardware") or provider_label(data.get("api_url", ""), local_hw)
        pub_hw = public_hardware(hw)
        if frozen["local"] and not pub_hw["memory"]:
            dropped["local"] += 1
            continue
        runs.append({"data": data, "hardware": pub_hw})

    # 3. one board, one eval set: identify each run's set and keep the dominant one
    by_hash: dict = {}
    for run in runs:
        h = _run_hash(run["data"], candidates)
        if h is None:
            dropped["unidentified_eval_set"] += 1
            continue
        by_hash.setdefault(h, []).append(run)
    if not by_hash:
        raise PublishError("no publishable runs: nothing scored on an identifiable eval set")
    chosen = max(by_hash, key=lambda h: (
        len(by_hash[h]), max(str(r["data"].get("timestamp", "")) for r in by_hash[h]), h))
    for h, rs in by_hash.items():
        if h != chosen:
            dropped["other_eval_set"] += len(rs)
    runs = by_hash[chosen]

    # prompts on the board: from the eval set file when we have it, else from the runs
    chosen_file = next((d for _p, d, h in candidates if h == chosen), None)
    prompt_cats: dict = {}
    if chosen_file is not None:
        for p in chosen_file.get("prompts", []):
            if "id" in p:
                prompt_cats.setdefault(p["id"], str(p.get("category", "") or ""))
    else:
        for run in runs:
            for r in run["data"]["results"]:
                if "id" in r:
                    prompt_cats.setdefault(r["id"], str(r.get("category", "") or ""))
    prompts = [{"id": pid, "category": prompt_cats[pid]} for pid in sorted(prompt_cats, key=_id_key)]
    categories = sorted({p["category"] for p in prompts if p["category"]})
    # categories the runs actually report, for the per-category breakdown
    for run in runs:
        for r in run["data"]["results"]:
            c = str(r.get("category", "") or "")
            if c and c not in categories:
                categories.append(c)
    categories.sort()

    # 4. group repeat runs exactly as the leaderboard does
    groups: dict = {}
    for run in runs:
        groups.setdefault(run_group_key(run["data"]), []).append(run)

    entries = []
    for members in groups.values():
        members.sort(key=lambda r: r["data"].get("timestamp", ""), reverse=True)
        latest = members[0]
        datas = [m["data"] for m in members]
        variants, per_run = {}, {}
        for subset in SUBSETS:
            variants[subset] = {}
            for metric in METRICS:
                v, pr = _variant(datas, subset, metric, categories)
                variants[subset][metric] = v
                per_run[(subset, metric)] = pr
        if variants["all"]["score"] is None:
            continue
        tps, avg = _speed(latest["data"])
        judge = str(latest["data"].get("judge", "") or "")
        model = str(latest["data"].get("model", "unknown"))
        entry = {
            "model": model,
            "judge_model": judge,
            "self_judged": judge == model,
            "hardware": dict(latest["hardware"]),
            "is_local": bool(latest["hardware"]["memory"]),
            "n_runs": len(members),
            "tok_per_sec": tps,
            "avg_time": avg,
        }
        if frozen["notes"]:
            notes = _public_notes(latest["data"], name_like)
            if notes:
                entry["notes"] = notes
        entry["per_prompt"] = _per_prompt(datas)
        entry["variants"] = variants
        entries.append({"entry": entry, "per_run": per_run})

    if not entries:
        raise PublishError("no publishable runs")

    # 5. rank each variant within its own measured noise
    variants_meta = {}
    for subset in SUBSETS:
        variants_meta[subset] = {}
        for metric in METRICS:
            cells = [(e["entry"], e["entry"]["variants"][subset][metric], e["per_run"][(subset, metric)])
                     for e in entries if e["entry"]["variants"][subset][metric] is not None]
            noise = measured_noise_floor(pr for _e, _v, pr in cells)
            floor = noise["floor"] if noise else None
            cells.sort(key=lambda c: (not c[1]["low_coverage"], c[1]["pct"], c[0]["tok_per_sec"] or 0),
                       reverse=True)
            rank_within_noise([v for _e, v, _pr in cells], floor)
            meta = {"noise_floor": floor}
            if metric == "score":
                meta["max_pts"] = max((v["max"] for _e, v, _pr in cells), default=0)
            variants_meta[subset][metric] = meta

    # 6. the criteria metric is public only if every entry has it in both subsets
    check_available = all(e["entry"]["variants"][s]["check"] is not None
                          for e in entries for s in SUBSETS)
    if not check_available:
        for e in entries:
            for s in SUBSETS:
                e["entry"]["variants"][s]["check"] = None
        for s in SUBSETS:
            variants_meta[s]["check"] = None
        view["metric"] = "score"

    out_entries = [e["entry"] for e in entries]
    active = "text" if view["text"] else "all"

    def order(e):
        v = e["variants"][active][view["metric"]]
        if v is None:
            return (True, 0, 0, 0, e["model"])
        # the same order ranks were assigned in: tied rows keep their pct order
        return (False, v["rank"], -v["pct"], -(e["tok_per_sec"] or 0), e["model"])
    out_entries.sort(key=order)

    judges = {e["judge_model"] for e in out_entries}
    n_text = sum(1 for p in prompts if p["category"] != TEXT_EXCLUDED)

    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "handle": str(handle or ""),
        "published_at": now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cupel_version": str(cupel_version or ""),
        "eval_set": {"hash": chosen, "n_prompts": len(prompts), "n_text_prompts": n_text},
        "judge": judges.pop() if len(judges) == 1 else "mixed",
        "view": view,
        "frozen": frozen,
        "check_available": check_available,
        "categories": categories,
        "prompts": prompts,
        "variants_meta": variants_meta,
        "entries": out_entries,
    }
    summary = {
        "n_entries": len(out_entries),
        "n_prompts": len(prompts),
        "frozen": dict(frozen),
        "check_available": check_available,
        "dropped": dropped,
        "eval_set": {"hash": chosen, "n_runs": len(runs)},
    }
    return snapshot, summary


# ──────────────────────────────────────────────
# validation
# ──────────────────────────────────────────────

FORBIDDEN_KEYS = {
    "title", "prompt", "response", "thinking", "judge_reason", "criteria_results",
    "filename", "runs", "api_url", "judge_url", "path", "spec", "per_run_pct",
    "error", "judgments",
}

TOP_KEYS = {"schema_version", "handle", "published_at", "cupel_version", "eval_set",
            "judge", "view", "frozen", "check_available", "categories", "prompts",
            "variants_meta", "entries"}
ENTRY_KEYS = {"model", "judge_model", "self_judged", "hardware", "is_local", "n_runs",
              "tok_per_sec", "avg_time", "per_prompt", "variants"}
ENTRY_OPTIONAL = {"notes"}
SCORE_KEYS = {"pts", "max", "pct", "ci_lo", "ci_hi", "rank", "tied", "coverage",
              "low_coverage", "by_category"}
CHECK_KEYS = SCORE_KEYS - {"pts", "max"}

PRIVATE_NET = re.compile(
    r"localhost|127\.0\.0\.1"
    r"|(?<![\d.])10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    r"|(?<![\d.])172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}"
    r"|(?<![\d.])192\.168\.\d{1,3}\.\d{1,3}")

MIN_LEN = 12        # long free text: short strings would hit common words
MIN_TITLE_LEN = 4   # titles and names are short but identifying


def _walk(obj, path="$"):
    """Yield (path, key_or_None, value) for every node."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield path, k, v
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


def _keys_exact(obj, required: set, label: str, errors: list, optional: set = frozenset()):
    if not isinstance(obj, dict):
        errors.append(f"{label}: expected an object")
        return False
    extra = set(obj) - required - optional
    missing = required - set(obj)
    if extra:
        errors.append(f"{label}: unexpected keys {sorted(map(str, extra))}")
    if missing:
        errors.append(f"{label}: missing keys {sorted(missing)}")
    return not extra and not missing


def _check_structure(snap, errors):
    if not _keys_exact(snap, TOP_KEYS, "snapshot", errors):
        return
    if snap["schema_version"] != SCHEMA_VERSION:
        errors.append("snapshot: wrong schema_version")
    _keys_exact(snap["eval_set"], {"hash", "n_prompts", "n_text_prompts"}, "eval_set", errors)
    _keys_exact(snap["view"], set(VIEW_ENUMS), "view", errors)
    _keys_exact(snap["frozen"], set(FROZEN_KEYS), "frozen", errors)
    for i, p in enumerate(snap["prompts"] if isinstance(snap["prompts"], list) else []):
        _keys_exact(p, {"id", "category"}, f"prompts[{i}]", errors)
    vm = snap["variants_meta"]
    if _keys_exact(vm, set(SUBSETS), "variants_meta", errors):
        for s in SUBSETS:
            if _keys_exact(vm[s], set(METRICS), f"variants_meta.{s}", errors):
                _keys_exact(vm[s]["score"], {"noise_floor", "max_pts"}, f"variants_meta.{s}.score", errors)
                if vm[s]["check"] is not None:
                    _keys_exact(vm[s]["check"], {"noise_floor"}, f"variants_meta.{s}.check", errors)
    for i, e in enumerate(snap["entries"] if isinstance(snap["entries"], list) else []):
        label = f"entries[{i}]"
        if not _keys_exact(e, ENTRY_KEYS, label, errors, ENTRY_OPTIONAL):
            continue
        if "notes" in e and not snap["frozen"].get("notes"):
            errors.append(f"{label}: notes present while frozen.notes is false")
        hw = e["hardware"]
        if _keys_exact(hw, {"name", "memory"}, f"{label}.hardware", errors):
            if not hw["memory"] and hw["name"] not in PUBLIC_PROVIDER_NAMES + (UNKNOWN_PROVIDER,):
                errors.append(f"{label}.hardware: cloud name is not a known provider label")
        for j, p in enumerate(e["per_prompt"] if isinstance(e["per_prompt"], list) else []):
            _keys_exact(p, {"id", "category", "score"}, f"{label}.per_prompt[{j}]", errors)
        if _keys_exact(e["variants"], set(SUBSETS), f"{label}.variants", errors):
            for s in SUBSETS:
                if _keys_exact(e["variants"][s], set(METRICS), f"{label}.variants.{s}", errors):
                    v = e["variants"][s]["score"]
                    if v is not None:
                        _keys_exact(v, SCORE_KEYS, f"{label}.variants.{s}.score", errors)
                    c = e["variants"][s]["check"]
                    if c is not None:
                        if not snap["check_available"]:
                            errors.append(f"{label}.variants.{s}.check: present while check_available is false")
                        _keys_exact(c, CHECK_KEYS, f"{label}.variants.{s}.check", errors)


def _collect_forbidden(result_paths, eval_sets) -> list[tuple[str, str, str]]:
    """(value, source, field) for every string from the sources that must not appear."""
    out = []

    def add(val, source, field, min_len):
        if isinstance(val, str) and len(val.strip()) >= min_len:
            out.append((val.strip(), source, field))

    for path in result_paths:
        path = Path(path)
        if _is_sample(path):
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        src = path.name
        for name in _eval_set_names(data.get("eval_set")):
            add(name, src, "eval_set.name", MIN_TITLE_LEN)
        es = data.get("eval_set")
        if isinstance(es, dict):
            add(es.get("path"), src, "eval_set.path", MIN_TITLE_LEN)
        for key in ("api_url", "judge_url"):
            url = data.get(key)
            add(url, src, key, MIN_LEN)
            host = urlparse(url).hostname if isinstance(url, str) and url else None
            if host and not any(h in host for h, _n in KNOWN_PROVIDERS):
                add(host, src, f"{key} host", MIN_TITLE_LEN)
        hw = data.get("hardware")
        if isinstance(hw, dict):
            add(hw.get("spec"), src, "hardware.spec", MIN_LEN)
        for r in data.get("results", []) if isinstance(data.get("results"), list) else []:
            if not isinstance(r, dict):
                continue
            add(r.get("title"), src, "title", MIN_TITLE_LEN)
            for field in ("prompt", "response", "thinking", "judge_reason", "notes"):
                add(r.get(field), src, field, MIN_LEN)
            for j in r.get("judgments", []) or []:
                if isinstance(j, dict):
                    add(j.get("reason"), src, "judgments.reason", MIN_LEN)

    for path, data in eval_sets:
        src = Path(path).name
        add(Path(path).stem, src, "eval set file name", MIN_TITLE_LEN)
        if not isinstance(data, dict):
            continue
        add(data.get("name"), src, "eval set name", MIN_TITLE_LEN)
        for p in data.get("prompts", []) or []:
            if not isinstance(p, dict):
                continue
            add(p.get("title"), src, "eval set prompt title", MIN_TITLE_LEN)
            for k, v in p.items():
                if k in ("id", "category", "title"):
                    continue
                for s in _strings(v):
                    add(s, src, f"eval set prompt {k}", MIN_LEN)
    return out


def validate_snapshot(snapshot: dict, result_paths: list[Path], home: Path,
                      eval_sets: list | None = None) -> list[str]:
    """Every reason this snapshot must not be published. Empty means valid."""
    errors: list[str] = []
    eval_sets = list(eval_sets or [])

    # keys: the forbidden list, anywhere
    for path, key, _v in _walk(snapshot):
        k = str(key)
        if k in FORBIDDEN_KEYS or k.startswith("_"):
            errors.append(f"forbidden key {k!r} at {path}")

    # structure: only schema fields
    _check_structure(snapshot, errors)

    # values: scan every string in the output (keys included)
    blob = "\n".join(_strings(snapshot))
    home_s = str(home)
    if home_s and home_s not in ("/", ".") and home_s in blob:
        errors.append("output contains the home directory path")
    m = PRIVATE_NET.search(blob)
    if m:
        errors.append(f"output contains a local or private network address ({m.group(0)})")

    seen = set()
    for val, src, field in _collect_forbidden(result_paths, eval_sets):
        if val in blob and (src, field) not in seen:
            seen.add((src, field))
            errors.append(f"output contains text from {src} field {field}")
    return errors


# ──────────────────────────────────────────────
# write
# ──────────────────────────────────────────────

def serialize(snapshot: dict) -> str:
    return json.dumps(snapshot, indent=2, sort_keys=False, ensure_ascii=False) + "\n"


def write_snapshot(snapshot: dict, repo: Path) -> Path:
    """Write site/bench/bench.json atomically. Returns the path written."""
    target = Path(repo) / SNAPSHOT_RELPATH
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".bench.", suffix=".json", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(serialize(snapshot))
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target

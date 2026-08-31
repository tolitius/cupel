"""cupel.schema — run-file shape, judgment records, and legacy migration.

A result used to carry exactly one judgment, written directly onto the record as
`score` / `judge_reason` / `judge_model`. Re-judging overwrote it, so a previous
judgment could never be recovered and two judges could never be compared.

Judgments now live in a list. The flat fields remain as a mirror of the consensus
so every existing reader — the dashboard, the results page, `cupel results`, the
bundled example data — keeps working untouched.

Migration happens on read, not by rewriting files: a run saved before this change
is normalised into the new shape when it is loaded, and only written back in the
new shape if something re-judges it.

This module deliberately imports nothing from cupel.eval — the judge prompt is
passed in — so that eval.py can depend on it without a cycle.
"""

import hashlib
import json
from datetime import datetime

# scoring_version for judgments produced before versions were recorded
LEGACY_VERSION = "legacy"


# ──────────────────────────────────────────────
# Versioning
# ──────────────────────────────────────────────

def _canonical(obj) -> str:
    """Stable JSON — key order and whitespace can't change the hash."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _as_iso(ts: str) -> str:
    """Normalise a run timestamp to ISO.

    Run files name themselves with `20260101_120000`, but judgments are stamped in
    ISO. Judgments get sorted by this field to find the most recent judge, and
    comparing the two formats as strings silently orders them wrong — `_` sorts
    above `-`, so a 2026 legacy stamp beats a 2026 ISO one.
    """
    if not ts:
        return ""
    try:
        return datetime.strptime(ts, "%Y%m%d_%H%M%S").isoformat(timespec="seconds")
    except (ValueError, TypeError):
        return ts


def scoring_version(rubric, judge_system: str = "") -> str:
    """Identity of the scoring rules that produced a judgment.

    Both halves matter: the rubric says what to look for, the judge system prompt
    says how strictly to grade. cupel's judge prompt has been materially rewritten
    before, which silently put old and new scores on different scales in the same
    leaderboard. Recording this makes that visible instead.
    """
    payload = _canonical({"judge_system": judge_system, "rubric": rubric})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def eval_set_meta(path, data: dict) -> dict:
    """Identity of the eval set a run used.

    The hash covers the prompts only, so editing a set's display name does not
    invalidate comparability, but editing a prompt or a rubric does.
    """
    prompts = data.get("prompts", [])
    return {
        "name": data.get("name", ""),
        "path": str(path) if path else "",
        "hash": "sha256:" + hashlib.sha256(
            _canonical(prompts).encode("utf-8")).hexdigest()[:16],
        "n_prompts": len(prompts),
    }


# ──────────────────────────────────────────────
# Judgments
# ──────────────────────────────────────────────

def match_eval_set(run: dict, eval_set: dict) -> float:
    """How well an eval set explains a run's results, as a 0.0-1.0 fraction.

    Matches on (id, title) pairs. Prompt ids alone are not enough — two different
    eval sets both number from 1, which is exactly how a cupel-og run came to be
    scored against coding-bench-v4's rubrics. Titles are recorded on every result,
    so the pair identifies a set essentially unambiguously.
    """
    results = run.get("results", [])
    if not results:
        return 0.0
    prompts = {(p.get("id"), p.get("title", "")) for p in eval_set.get("prompts", [])}
    if not prompts:
        return 0.0
    hits = sum(1 for r in results if (r.get("id"), r.get("title", "")) in prompts)
    return hits / len(results)


def infer_eval_set(run: dict, candidates, threshold: float = 0.9):
    """Pick the eval set a run was scored against.

    `candidates` is an iterable of (path, eval_set_dict). Returns
    (path, eval_set, score) for the best match at or above `threshold`, else
    (None, None, best_score) — the caller should refuse rather than guess, because
    guessing wrong silently grades answers against unrelated rubrics.
    """
    best = (None, None, 0.0)
    for path, eval_set in candidates:
        score = match_eval_set(run, eval_set)
        if score > best[2]:
            best = (path, eval_set, score)
    if best[2] >= threshold:
        return best
    return (None, None, best[2])


def make_judgment(judge_model, judge_url, score, reason,
                  criteria_results=None, check_score=None,
                  version=LEGACY_VERSION, judged_at=None) -> dict:
    j = {
        "judge_model": judge_model or "",
        "judge_url": judge_url or "",
        "score": score,
        "reason": reason or "",
        "scoring_version": version,
        "judged_at": judged_at or datetime.now().isoformat(timespec="seconds"),
    }
    if criteria_results is not None:
        j["criteria_results"] = criteria_results
    if check_score is not None:
        j["check_score"] = check_score
    return j


def consensus(judgments: list[dict]) -> dict:
    """Combine judgments into the values mirrored onto the result record.

    The consensus score is the lower median. Lower, not upper, because the judge
    prompt itself instructs "when in doubt, mark lower" — the aggregate should not
    be more generous than the individual judgments it summarises.
    """
    scored = [j for j in judgments if j.get("score") is not None]
    if not scored:
        return {"score": None, "judge_reason": "", "judge_model": "",
                "judge_agreement": 0, "n_judgments": len(judgments)}

    scores = sorted(j["score"] for j in scored)
    # lower median: index (n-1)//2 picks the lower of the two middles when even
    median = scores[(len(scores) - 1) // 2]

    # carry the reasoning from a judgment that actually gave the consensus score,
    # so the displayed explanation matches the displayed number
    representative = next(j for j in scored if j["score"] == median)

    out = {
        "score": median,
        "judge_reason": representative.get("reason", ""),
        "judge_model": representative.get("judge_model", ""),
        "judge_agreement": scores[-1] - scores[0],
        "n_judgments": len(scored),
    }
    if "criteria_results" in representative:
        out["criteria_results"] = representative["criteria_results"]
    if "check_score" in representative:
        out["check_score"] = representative["check_score"]
    if len(scored) > 1:
        out["judge_consensus"] = "median"
    return out


def drop_judge(result: dict, judge_model: str) -> bool:
    """Remove one judge's judgments from a result and recompute the mirror.

    The undo for a bad re-judge. It works because judgments accumulate: the
    judgments from every *other* judge — including their reasoning — are still
    there to restore from.

    Returns True if anything was removed.
    """
    judgments = result.get("judgments") or []
    keep = [j for j in judgments if j.get("judge_model") != judge_model]
    if len(keep) == len(judgments):
        return False

    result["judgments"] = keep
    # these are recomputed from what remains; stale values would outlive their source
    for field in ("judge_agreement", "judge_consensus", "check_score", "criteria_results"):
        result.pop(field, None)
    apply_consensus(result)
    return True


def record_judge_error(result: dict, judge_model: str, message: str) -> dict:
    """Note that a judge failed on this result, without touching the mirror.

    A failed judgment produces no judgment, so it must not write to `score` or
    `judge_reason` — those mirror the consensus of the judgments that did succeed.
    Writing the failure text into `judge_reason` is what left results displaying
    one judge's score beside another judge's error message.
    """
    result.setdefault("judge_errors", []).append({
        "judge_model": judge_model or "",
        "error": message,
        "at": datetime.now().isoformat(timespec="seconds"),
    })
    return result


def repair_mirror(data: dict) -> int:
    """Recompute every result's flat fields from its judgments.

    The flat `score` / `judge_reason` pair is only ever a mirror, so anything that
    wrote to it directly can be undone by recomputing. Returns the number of
    results whose visible score or reason changed.
    """
    fixed = 0
    for result in data.get("results", []):
        before = (result.get("score"), result.get("judge_reason"))
        apply_consensus(result)
        if (result.get("score"), result.get("judge_reason")) != before:
            fixed += 1
    return fixed


def clear_judgments(result: dict) -> dict:
    """Reset a result to unscored. Used when the last judgment is removed."""
    result["judgments"] = []
    result["score"] = None
    result["judge_reason"] = ""
    for field in ("judge_model", "judge_agreement", "judge_consensus",
                  "check_score", "criteria_results"):
        result.pop(field, None)
    return result


def apply_consensus(result: dict) -> dict:
    """Recompute the mirrored flat fields from `result["judgments"]`."""
    judgments = result.get("judgments") or []
    if not judgments:
        # nothing left to mirror — the result is unscored again, and leaving the
        # old score behind would show a score with no judgment backing it
        if "judgments" in result:
            clear_judgments(result)
        return result

    c = consensus(judgments)
    result["score"] = c["score"]
    result["judge_reason"] = c["judge_reason"]
    result["judge_model"] = c["judge_model"]
    if "criteria_results" in c:
        result["criteria_results"] = c["criteria_results"]
    if "check_score" in c:
        result["check_score"] = c["check_score"]
    if c["n_judgments"] > 1:
        result["judge_agreement"] = c["judge_agreement"]
        result["judge_consensus"] = c["judge_consensus"]
    return result


def add_judgment(result: dict, judgment: dict, replace: bool = False) -> dict:
    """Append a judgment to a result and refresh the mirror.

    `replace` restores the old destructive behaviour for callers that explicitly
    ask for it — re-scoring with the same judge after fixing a rubric, say.
    """
    # Safety net: a caller that skipped normalise_run would otherwise silently
    # drop the existing judgment, which is the exact bug this module removes.
    if "judgments" not in result:
        normalize_result(result, {})
    if replace:
        result["judgments"] = [judgment]
    else:
        result["judgments"].append(judgment)
    return apply_consensus(result)


# ──────────────────────────────────────────────
# Legacy migration (read-time)
# ──────────────────────────────────────────────

def normalize_result(result: dict, run: dict) -> dict:
    """Give a pre-judgments result record a `judgments` list.

    Idempotent: a result that already has judgments is returned unchanged.
    An unscored result gets an empty list, not a fabricated judgment.
    """
    if "judgments" in result:
        return result
    if result.get("score") is None:
        result["judgments"] = []
        return result

    result["judgments"] = [make_judgment(
        judge_model=result.get("judge_model") or run.get("judge", ""),
        judge_url=run.get("judge_url", ""),
        score=result["score"],
        reason=result.get("judge_reason", ""),
        criteria_results=result.get("criteria_results"),
        check_score=result.get("check_score"),
        version=LEGACY_VERSION,
        judged_at=_as_iso(run.get("timestamp", "")),
    )]
    return result


def normalize_run(data: dict) -> dict:
    """Normalise a whole run file in place. Safe to call repeatedly."""
    for result in data.get("results", []):
        normalize_result(result, data)

    if "judges" not in data:
        # One entry per judge model. Each prompt has its own rubric and therefore
        # its own per-judgment scoring_version, so those are rolled up into a
        # single run-level hash rather than producing one entry per prompt.
        seen: dict[str, dict] = {}
        versions: dict[str, set] = {}
        for result in data.get("results", []):
            for j in result.get("judgments", []):
                model = j.get("judge_model", "")
                entry = seen.setdefault(model, {
                    "model": model,
                    "url": j.get("judge_url", ""),
                    "judged_at": j.get("judged_at", ""),
                })
                # keep the most recent time this judge ran
                if j.get("judged_at", "") > entry["judged_at"]:
                    entry["judged_at"] = j["judged_at"]
                versions.setdefault(model, set()).add(
                    j.get("scoring_version", LEGACY_VERSION))

        for model, entry in seen.items():
            vs = versions[model]
            # a single version passes through unchanged so it stays readable;
            # a mix is hashed so any rubric change still shows up as a change
            entry["scoring_version"] = (
                vs.pop() if len(vs) == 1
                else hashlib.sha256(
                    _canonical(sorted(vs)).encode("utf-8")).hexdigest()[:12]
            )
        data["judges"] = list(seen.values())

    return data


def refresh_judges(data: dict) -> dict:
    """Rebuild the run-level judge list after judging. Forces recomputation."""
    data.pop("judges", None)
    normalize_run(data)
    # keep the legacy single-judge fields pointing at the most recent judge
    if data["judges"]:
        latest = max(data["judges"], key=lambda j: j.get("judged_at", ""))
        data["judge"] = latest["model"]
        data["judge_url"] = latest["url"]
    return data


def judge_models(data: dict) -> list[str]:
    """Every judge that has scored this run."""
    return [j["model"] for j in data.get("judges", []) if j.get("model")]

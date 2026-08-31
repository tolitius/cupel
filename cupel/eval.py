"""cupel.eval — core eval engine: LLM calls, judging, run/judge orchestration."""

import json
import logging
import os
import re
import time
import base64
import requests
from pathlib import Path
from datetime import datetime

from cupel import __version__
from cupel.config import resolve_path
from cupel.stats import criteria_score
from cupel.schema import (
    make_judgment, add_judgment, scoring_version, normalize_run, refresh_judges,
    infer_eval_set, record_judge_error,
)

log = logging.getLogger("cupel")


# ──────────────────────────────────────────────
# Image handling
# ──────────────────────────────────────────────

def find_image(image_filename: str, image_dir: Path | None,
               config_path: str | None = None) -> str | None:
    # Try resolve_path first (handles relative paths anchored to config dir)
    resolved = resolve_path(image_filename, config_path)
    if resolved.exists():
        with open(resolved, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        print(f"  ✓ Image: {resolved} ({len(b64) // 1024}KB)")
        return b64

    # Fall back to basename search for bare filenames
    basename = Path(image_filename).name
    candidates = []
    if image_dir:
        candidates.append(image_dir / basename)
    candidates.extend([
        Path.home() / ".cupel" / "eval-sets" / basename,
        Path(__file__).parent / "data" / basename,
        Path.cwd() / basename,
        Path.cwd() / "eval-sets" / basename,
    ])
    for path in candidates:
        if path.exists():
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            print(f"  ✓ Image: {path} ({len(b64) // 1024}KB)")
            return b64
    print(f"  ⚠ '{image_filename}' not found — prompt #1 will be skipped")
    return None


def build_vision_content(prompt_text: str, image_b64: str) -> list[dict]:
    return [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        {"type": "text", "text": prompt_text},
    ]


# ──────────────────────────────────────────────
# API call — standard OpenAI chat completions
# ──────────────────────────────────────────────

def call_llm(
    api_url: str, api_key: str, model: str, prompt: str,
    temperature: float | None = None, max_tokens: int = 16384,
    thinking_budget: int | None = None, image_b64: str | None = None,
) -> dict:
    content = build_vision_content(prompt, image_b64) if image_b64 else prompt
    messages = [{"role": "user", "content": content}]
    return _call_llm_raw(api_url, api_key, model, messages, temperature, max_tokens, thinking_budget)


def call_llm_multi(
    api_url: str, api_key: str, model: str, messages: list[dict],
    temperature: float | None = None, max_tokens: int = 16384,
    thinking_budget: int | None = None,
) -> dict:
    return _call_llm_raw(api_url, api_key, model, messages, temperature, max_tokens, thinking_budget)


def _call_llm_raw(
    api_url: str, api_key: str, model: str, messages: list[dict],
    temperature: float | None = None, max_tokens: int = 16384,
    thinking_budget: int | None = None,
) -> dict:

    is_anthropic = "api.anthropic.com" in api_url
    is_openrouter = "openrouter.ai" in api_url

    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        body["temperature"] = temperature

    if thinking_budget is not None:
        body["thinking_budget"] = thinking_budget  # oMLX
        body["think"] = thinking_budget > 0         # Ollama

    headers = {
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/tolitius/cupel",
        "X-OpenRouter-Title": "cupel",
        "User-Agent": f"cupel/{__version__}",
    }

    if is_anthropic:
        # Anthropic uses x-api-key, not Authorization: Bearer
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
        # Anthropic doesn't accept these fields
        body.pop("thinking_budget", None)
        body.pop("think", None)
        # Anthropic requires system messages as a top-level field, not in messages
        if body["messages"] and body["messages"][0].get("role") == "system":
            body["system"] = body["messages"][0]["content"]
            body["messages"] = body["messages"][1:]
    elif is_openrouter:
        if api_key and api_key != "no-key":
            headers["Authorization"] = f"Bearer {api_key}"
        # OpenRouter doesn't use these fields
        body.pop("thinking_budget", None)
        body.pop("think", None)
        # OpenRouter reasoning support
        if thinking_budget is None:
            body["reasoning"] = {"effort": "high", "exclude": True}
        elif thinking_budget > 0:
            body["reasoning"] = {"max_tokens": thinking_budget, "exclude": True}
        # thinking_budget == 0 → omit reasoning block (no thinking)
    else:
        if api_key and api_key != "no-key":
            headers["Authorization"] = f"Bearer {api_key}"

    log.info("llm call  model=%s url=%s tokens=%d", model, api_url, max_tokens)
    start = time.time()
    try:
        resp = requests.post(api_url, headers=headers, json=body, timeout=1800)
    except requests.exceptions.Timeout:
        elapsed = time.time() - start
        log.error("llm timeout after %.0fs  model=%s url=%s", elapsed, model, api_url)
        raise RuntimeError(f"LLM request timed out after {elapsed:.0f}s")
    except requests.exceptions.ConnectionError as e:
        elapsed = time.time() - start
        log.error("llm connection error after %.1fs  model=%s url=%s: %s", elapsed, model, api_url, e)
        raise
    elapsed = time.time() - start

    if resp.status_code >= 400:
        try:
            err_body = resp.json()
            err_msg = err_body.get("error", {})
            if isinstance(err_msg, dict):
                err_msg = err_msg.get("message", resp.text[:200])
            log.error("llm HTTP %d  model=%s: %s", resp.status_code, model, err_msg)
            raise RuntimeError(f"HTTP {resp.status_code}: {err_msg}")
        except (ValueError, KeyError):
            log.error("llm HTTP %d  model=%s: %s", resp.status_code, model, resp.text[:200])
            resp.raise_for_status()

    log.info("llm done  model=%s elapsed=%.1fs", model, elapsed)

    try:
        data = resp.json()
    except json.JSONDecodeError:
        log.error("llm response not JSON  model=%s: %s", model, resp.text[:300])
        raise RuntimeError(f"LLM returned non-JSON response ({len(resp.text)} chars)")
    usage = data.get("usage", {})

    if is_anthropic:
        # Anthropic response: {"content": [{"type": "text", "text": "..."}], ...}
        content_text = ""
        thinking = ""
        for block in data.get("content", []):
            if block.get("type") == "thinking":
                thinking += block.get("thinking", "")
            elif block.get("type") == "text":
                content_text += block.get("text", "")
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
        finish_reason = data.get("stop_reason", "")
    else:
        # OpenAI-compatible response
        choices = data.get("choices")
        if not choices:
            log.error("llm response missing choices  model=%s: %s", model, json.dumps(data)[:300])
            raise RuntimeError("LLM response has no 'choices' field")
        choice = choices[0].get("message") or choices[0].get("delta", {})
        content_text = choice.get("content") or ""   # content can be null for tool_calls
        thinking = choice.get("thinking") or ""
        # fix: oMLX, DeepSeek, vLLM, SGLang, llama.cpp, Ollama use reasoning_content
        if not thinking and choice.get("reasoning_content"):
            thinking = choice["reasoning_content"]
        # OpenRouter returns reasoning in a separate field
        if not thinking and choice.get("reasoning"):
            thinking = choice["reasoning"]
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        finish_reason = choices[0].get("finish_reason", "")

        # Capture native tool_calls — serialize into content so judge can see them
        tool_calls = choice.get("tool_calls")
        if tool_calls and not content_text.strip():
            calls = []
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args_str = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_str)
                except (json.JSONDecodeError, TypeError):
                    args = args_str
                calls.append({"tool": name, "args": args})
            content_text = json.dumps(calls, indent=2)

    # Strip leaked <think> tags from content
    if "<think>" in content_text:
        think_match = re.search(r"<think>(.*?)</think>", content_text, re.DOTALL)
        if think_match:
            if not thinking:
                thinking = think_match.group(1).strip()
            content_text = content_text[:think_match.start()] + content_text[think_match.end():]
            content_text = content_text.strip()
        elif content_text.startswith("<think>"):
            if not thinking:
                thinking = content_text.replace("<think>", "").strip()
            content_text = ""

    if finish_reason == "length":
        log.warning("llm response truncated (hit max_tokens=%d)  model=%s elapsed=%.1fs tokens=%d",
                    max_tokens, model, elapsed, completion_tokens)

    # estimate thinking tokens from text — no tokenizer dependency
    thinking_tokens = len(thinking) // 4 if thinking else 0

    return {
        "content": content_text,
        "thinking": thinking,
        "elapsed_seconds": round(elapsed, 2),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "thinking_tokens": thinking_tokens,
        "total_tokens": usage.get("total_tokens", prompt_tokens + completion_tokens),
        "finish_reason": finish_reason,
    }


# ──────────────────────────────────────────────
# Judge — auto-score using rubrics
# ──────────────────────────────────────────────

JUDGE_SYSTEM = """You are a strict evaluator scoring LLM responses against per-prompt rubrics.

Shared rules for all rubric formats:
- Be evidence-based: cite verbatim quotes from the response to justify each judgment.
- Be conservative: when in doubt, mark lower / UNMET.
- Do not infer intent — judge only what is explicitly stated in the response.

## Format A — Level rubric (keys "0","1","2","3")
Score 0-3 per the level descriptions with these calibration rules:
- The rubric's level-3 description is the FLOOR for a 2, not the target for a 3.
- Score 3 ONLY when the response goes beyond the rubric: adds correct caveats, identifies version-specific gotchas, notes when the textbook answer doesn't apply, or distinguishes nuances a senior engineer would catch.
- Score 2 when the response matches the rubric's level-3 description accurately.
- Score 1 when the response is partially correct.
- Score 0 when the response is wrong or hallucinates key facts.
- Confident claims without version sensitivity cap at 2.
- Naming a pattern correctly but misdescribing its mechanism caps at 2.
- Template-pattern code without demonstrating why each element is needed caps at 2.
Reply with ONLY: {"score": <0-3>, "reason": "<one sentence>"}

## Format B — Criteria rubric (list of criteria with id/type/weight/check)
Evaluate each criterion independently as MET or UNMET.
- For each criterion, provide a verbatim substring quote from the response. The quote MUST appear verbatim in the response text — no paraphrasing, no summarization, no fabrication.
- For UNMET criteria, the quote may be an empty string.
- A criterion is MET only if the response clearly and correctly satisfies the check description AND you can cite a verbatim quote that demonstrates this.
- When the response is ambiguous or partially addresses a criterion, mark UNMET.
Reply with ONLY: {"criteria": [{"id": "<id>", "met": true/false, "quote": "<verbatim substring or empty string>"}], "reason": "<one sentence overall>"}

You MUST respond with ONLY a JSON object, no other text."""


def _prompt_text_for_judge(p: dict) -> str:
    """Extract a flat prompt string for the judge, handling both single and multi-turn."""
    if "prompt" in p:
        return p["prompt"]
    # Multi-turn: reconstruct a readable transcript of all turn messages
    parts = []
    for i, turn in enumerate(p.get("turns", []), 1):
        parts.append(f"--- Turn {i} ---")
        for msg in turn.get("messages", []):
            parts.append(f"[{msg['role']}]: {msg['content']}")
        for msg in turn.get("inject_after", []):
            parts.append(f"[{msg['role']}]: {msg['content']}")
    return "\n\n".join(parts)


def _is_criteria_rubric(rubric):
    """Return True if rubric uses the criteria-based format."""
    return isinstance(rubric, dict) and "criteria" in rubric


def _build_criteria_judge_prompt(prompt_text, rubric, response_text, responses=None):
    """Build judge prompt for criteria-based rubrics."""
    parts = [f"## Prompt given to the model\n\n{prompt_text}"]

    if rubric.get("context"):
        parts.append(f"## Context\n\n{rubric['context']}")

    criteria_lines = ["## Criteria to evaluate\n"]
    for c in rubric["criteria"]:
        criteria_lines.append(
            f"- {c['id']} ({c['type']}, weight={c.get('weight', 1)}): {c['check']}"
        )
    parts.append("\n".join(criteria_lines))

    if responses and len(responses) > 1:
        resp_section = "## Full conversation responses (judge all turns)\n\n"
        for i, r in enumerate(responses, 1):
            resp_section += f"### Turn {i} response\n\n{r}\n\n"
        parts.append(resp_section)
    else:
        parts.append(f"## Response to score\n\n{response_text}")

    parts.append(
        'For each criterion, decide MET or UNMET and provide a verbatim substring quote from the response. '
        'The quote MUST appear verbatim in the response text — no paraphrasing. For UNMET, the quote may be an empty string. '
        'Respond with ONLY JSON:\n'
        '{"criteria": [{"id": "<criterion_id>", "met": true/false, '
        '"quote": "<verbatim substring or empty string>"}], '
        '"reason": "<one sentence overall>"}'
    )

    return "\n\n".join(parts)


def build_judge_prompt(prompt_text: str, rubric: dict, response_text: str,
                       responses: list[str] | None = None) -> str:
    if _is_criteria_rubric(rubric):
        return _build_criteria_judge_prompt(prompt_text, rubric, response_text, responses)

    rubric_str = "\n".join(f"  {k}: {v}" for k, v in sorted(rubric.items()))

    # For multi-turn: show the full conversation transcript
    if responses and len(responses) > 1:
        resp_section = "## Full conversation responses (judge all turns)\n\n"
        for i, r in enumerate(responses, 1):
            resp_section += f"### Turn {i} response\n\n{r}\n\n"
    else:
        resp_section = f"## Response to score\n\n{response_text}"

    return f"""## Prompt given to the model

{prompt_text}

## Scoring rubric

{rubric_str}

{resp_section}

Score this response 0-3 per the rubric. Respond with ONLY JSON: {{"score": <0-3>, "reason": "<one sentence>"}}"""


# ── Criteria aggregation ──

DEFAULT_SCORE_MAP = [(7, 3), (5, 2), (3, 1), (0, 0)]


def _parse_score_map(score_map_dict):
    """Parse score_map dict with range-string keys into [(threshold, score)] sorted desc."""
    if not score_map_dict:
        return DEFAULT_SCORE_MAP
    try:
        entries = []
        for key, score in score_map_dict.items():
            key = str(key).strip()
            if key.endswith("+"):
                threshold = int(key[:-1])
            elif "-" in key:
                threshold = int(key.split("-")[0])
            else:
                threshold = int(key)
            entries.append((threshold, int(score)))
        entries.sort(key=lambda x: x[0], reverse=True)
        return entries
    except (ValueError, TypeError):
        return DEFAULT_SCORE_MAP


def _aggregate_criteria(criteria_results, rubric):
    """Aggregate criteria results into a 0-3 score.

    Returns (final_score, total) where total = positive_met - negative_met, floored at 0.
    """
    criteria_lookup = {c["id"]: (c["type"], c.get("weight", 1)) for c in rubric["criteria"]}

    positive_met = 0
    negative_met = 0
    for cr in criteria_results:
        cid = cr["id"]
        if cid not in criteria_lookup:
            continue
        ctype, weight = criteria_lookup[cid]
        if cr.get("met"):
            if ctype == "positive":
                positive_met += weight
            elif ctype == "negative":
                negative_met += weight

    total = max(0, positive_met - negative_met)

    score_map = _parse_score_map(rubric.get("score_map"))
    final_score = 0
    for threshold, score in score_map:
        if total >= threshold:
            final_score = score
            break

    return final_score, total


def parse_judge_response(text: str, rubric=None) -> tuple[int | None, str, list | None]:
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
    try:
        obj = json.loads(cleaned)

        # Criteria-based response
        if "criteria" in obj and rubric is not None and _is_criteria_rubric(rubric):
            criteria_results = obj["criteria"]
            judge_reason = obj.get("reason", "")
            final_score, total = _aggregate_criteria(criteria_results, rubric)
            max_total = sum(c.get("weight", 1) for c in rubric["criteria"] if c["type"] == "positive")
            marks = " ".join(
                f"{cr['id']}{'✓' if cr.get('met') else '✗'}"
                for cr in criteria_results
            )
            reason = f"{judge_reason} | criteria: {marks} | total: {total}/{max_total} → {final_score}"
            return final_score, reason, criteria_results

        # Legacy level-based response
        score = int(obj.get("score", -1))
        reason = obj.get("reason", "")
        if 0 <= score <= 3:
            return score, reason, None
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    m = re.search(r'"score"\s*:\s*(\d)', text)
    if m:
        score = int(m.group(1))
        if 0 <= score <= 3:
            r = re.search(r'"reason"\s*:\s*"([^"]*)"', text)
            return score, r.group(1) if r else "", None
    return None, f"unparseable judge response: {text[:100]}", None


def rubrics_for_run(data, forced_eval_set=None):
    """Load the rubrics and prompts a run was actually scored against.

    Resolution order: an explicitly forced set, then the set recorded in the run,
    then identification from the run's own (id, title) pairs. If none of those
    succeed it gives up rather than guessing — judging against whatever eval set
    config.yml happens to name today is how a run of one benchmark got graded
    against another's rubrics, silently rewriting every score.

    Returns (rubric_by_id, prompt_by_id, prompts, warning or None).
    """
    from cupel.schema import eval_set_meta

    if forced_eval_set is not None:
        return _rubric_maps(forced_eval_set) + (None,)

    meta = data.get("eval_set")
    warning = None

    if isinstance(meta, dict) and meta.get("path"):
        p = Path(meta["path"])
        if not p.is_absolute():
            p = resolve_path(str(p))
        if p.exists():
            try:
                with open(p) as f:
                    eval_set = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                eval_set, warning = None, f"could not read {p.name}: {e}"
            if eval_set is not None:
                if meta.get("hash") and eval_set_meta(p, eval_set)["hash"] != meta["hash"]:
                    warning = (f"{p.name} has changed since this run — rubrics may "
                               f"not match the prompts that were scored")
                return _rubric_maps(eval_set) + (warning,)
        else:
            warning = f"eval set {meta['path']} no longer exists"
    else:
        warning = "run predates eval-set tracking"

    # No recorded eval set. Identify it from the run's own (id, title) pairs before
    # considering anything else — falling back to whatever config names today is how
    # a cupel-og run got graded against coding-bench-v4's Python rubrics.
    path, inferred, confidence = infer_eval_set(data, _available_eval_sets())
    if inferred is not None:
        return _rubric_maps(inferred) + (
            f"{warning}; identified as {Path(path).name} "
            f"({confidence:.0%} of prompts match)",)

    # Refuse rather than guess. A wrong eval set silently rewrites every score.
    detail = (f"best match was only {confidence:.0%}"
              if confidence else "no eval set in ~/.cupel/eval-sets/ matches its prompts")
    return {}, {}, [], (
        f"{warning}; cannot identify which eval set this run used — {detail}. "
        f"Pass --eval-set to say explicitly.")


def _available_eval_sets():
    """(path, eval_set) for every eval set on disk, for identification."""
    out = []
    seen = set()
    for d in (Path.home() / ".cupel" / "eval-sets", Path(__file__).parent / "data"):
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.json")):
            if p.resolve() in seen:
                continue
            seen.add(p.resolve())
            try:
                with open(p) as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict) and data.get("prompts"):
                out.append((p, data))
    return out


def _rubric_maps(eval_set):
    prompts = eval_set.get("prompts", [])
    return (
        {p["id"]: p.get("rubric", {}) for p in prompts},
        {p["id"]: _prompt_text_for_judge(p) for p in prompts},
        prompts,
    )


def apply_judgment(result, score, reason, criteria_results, judge_model, rubric=None,
                   judge_url="", replace=False):
    """Record a judgment on a result.

    The single place a score lands on a result, so the CLI, the server, and the
    importable orchestrator all behave identically — they used to do this inline
    in four places and had already drifted.

    Judgments accumulate rather than overwrite: re-judging with a second judge
    keeps the first, which is what makes judges comparable and disagreement
    measurable. `score` and friends remain on the record as a mirror of the
    consensus so existing readers are unaffected.

    For criteria rubrics this also stores `check_score`: the continuous 0.0-1.0
    value from the criteria vector. That is the number to compare models on;
    collapsing it to 0-3 costs most of the power to tell two models apart
    (see cupel.stats).
    """
    check = None
    if criteria_results is not None and rubric:
        cs = criteria_score(criteria_results, rubric)
        if cs is not None:
            check = round(cs, 4)

    judgment = make_judgment(
        judge_model=judge_model,
        judge_url=judge_url,
        score=score,
        reason=reason,
        criteria_results=criteria_results,
        check_score=check,
        version=scoring_version(rubric or {}, JUDGE_SYSTEM),
    )
    return add_judgment(result, judgment, replace=replace)


# Judges that reason before answering can spend the whole budget thinking and
# return no JSON at all. 1024 was too tight for local thinking models — observed
# failing ~26% of prompts, reported only as "unparseable".
JUDGE_MAX_TOKENS = int(os.environ.get("CUPEL_JUDGE_MAX_TOKENS", "3072"))


def score_one(api_url, api_key, judge_model, prompt_text, rubric, response_text,
              responses=None):
    """Score a single response using the judge model. Returns (score, reason, criteria_results)."""
    judge_prompt = build_judge_prompt(prompt_text, rubric, response_text, responses)
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": judge_prompt},
    ]
    resp = call_llm_multi(
        api_url, api_key, judge_model, messages,
        temperature=0, max_tokens=JUDGE_MAX_TOKENS,
        thinking_budget=0,
    )
    score, reason, criteria_results = parse_judge_response(resp["content"], rubric=rubric)

    if score is None and resp.get("finish_reason") == "length":
        # not a malformed answer — the judge never finished one
        return None, (f"judge hit the {JUDGE_MAX_TOKENS}-token limit before answering "
                      f"(thinking used {resp.get('thinking_tokens', 0)} est. tokens); "
                      f"raise CUPEL_JUDGE_MAX_TOKENS or use a judge that answers directly"), None

    # One-shot retry on parse failure. Only worth spending a second round-trip when
    # the judge at least attempted JSON: a reply with no brace at all is a model that
    # answered in prose, and asking it to "re-emit as valid JSON" just produces more
    # prose at double the cost — observed on a small local judge in the field.
    if score is None and "{" in resp["content"]:
        messages.append({"role": "assistant", "content": resp["content"]})
        messages.append({"role": "user", "content":
            "your previous response was not valid JSON. re-emit the same evaluation "
            "as a single valid JSON object only — no prose, no markdown fences. "
            "all string values must be properly JSON-escaped (escape double quotes as \\\", "
            "newlines as \\n, backslashes as \\\\). quote fields should be SHORT — "
            "pick the most diagnostic phrase, not a paragraph."})
        resp = call_llm_multi(
            api_url, api_key, judge_model, messages,
            temperature=0, max_tokens=JUDGE_MAX_TOKENS,
            thinking_budget=0,
        )
        score, reason, criteria_results = parse_judge_response(resp["content"], rubric=rubric)

    return score, reason, criteria_results


# ──────────────────────────────────────────────
# Command: run
# ──────────────────────────────────────────────

def run_prompt(api_url, api_key, model, p, cfg, image_b64):
    pid = p["id"]

    if p.get("category") == "multimodal" and not image_b64:
        return {
            "id": pid, "title": p["title"], "category": p["category"],
            "skipped": True, "reason": "image not provided",
        }, "skip"

    # ── multi-turn prompts ──
    if "turns" in p:
        return _run_multi_turn(api_url, api_key, model, p, cfg)

    try:
        use_image = image_b64 if p.get("category") == "multimodal" else None
        resp = call_llm(
            api_url, api_key, model, p["prompt"],
            temperature=cfg.get("temperature"),
            max_tokens=cfg.get("max_tokens", 16384),
            thinking_budget=cfg.get("_thinking_budget"),
            image_b64=use_image,
        )

        content = resp["content"]
        thinking = resp.get("thinking", "")
        finish_reason = resp.get("finish_reason", "")

        if not content.strip() and thinking:
            return {
                "id": pid, "title": p["title"], "category": p["category"],
                "prompt": p["prompt"], "response": "", "thinking": thinking,
                "elapsed_seconds": resp["elapsed_seconds"],
                "completion_tokens": resp["completion_tokens"],
                "thinking_tokens": resp.get("thinking_tokens", 0),
                "error": "truncated: thinking consumed all tokens, no answer produced",
                "score": None,
            }, "error"

        result = {
            "id": pid, "title": p["title"], "category": p["category"],
            "prompt": p["prompt"], "response": content, "thinking": thinking,
            "elapsed_seconds": resp["elapsed_seconds"],
            "completion_tokens": resp["completion_tokens"],
            "thinking_tokens": resp.get("thinking_tokens", 0),
            "score": None, "judge_reason": "", "notes": "",
        }
        if finish_reason == "length":
            result["notes"] = f"truncated: hit max_tokens ({resp['completion_tokens']} tokens)"
        return result, f"{resp['elapsed_seconds']}s"
    except Exception as e:
        log.error("run_prompt failed  prompt=#%d title=%s: %s", pid, p.get("title", ""), e)
        return {
            "id": pid, "title": p["title"], "category": p["category"],
            "error": str(e), "score": None,
        }, "error"


def _call_and_record(api_url, api_key, model, history, cfg, all_responses,
                     all_thinking, stats):
    """Call LLM with current history, record response, update history in place."""
    resp = call_llm_multi(
        api_url, api_key, model, list(history),
        temperature=cfg.get("temperature"),
        max_tokens=cfg.get("max_tokens", 16384),
        thinking_budget=cfg.get("_thinking_budget"),
    )
    content = resp["content"]
    thinking = resp.get("thinking", "")
    stats["elapsed"] += resp["elapsed_seconds"]
    stats["tokens"] += resp.get("completion_tokens", 0)
    if thinking:
        all_thinking.append(thinking)
    history.append({"role": "assistant", "content": content})
    all_responses.append(content)


def _run_multi_turn(api_url, api_key, model, p, cfg):
    """Run a multi-turn prompt: accumulate messages, call model per turn.

    Rule: call the LLM whenever the last message in history is from "user".
    This handles both explicit user messages and injected tool results.
    """
    pid = p["id"]
    turns = p["turns"]
    history = []          # full message history sent to model
    all_responses = []    # model response per turn
    all_thinking = []
    stats = {"elapsed": 0, "tokens": 0}

    try:
        for turn in turns:
            # Add this turn's messages to history
            for msg in turn.get("messages", []):
                history.append(msg)

            # Call LLM if last message is from user (needs a response)
            if history and history[-1].get("role") == "user":
                _call_and_record(api_url, api_key, model, history, cfg,
                                 all_responses, all_thinking, stats)

            # Inject post-response messages (e.g. simulated tool results)
            for msg in turn.get("inject_after", []):
                history.append(msg)

            # If inject ended with a user message, call LLM again
            if turn.get("inject_after") and history[-1].get("role") == "user":
                _call_and_record(api_url, api_key, model, history, cfg,
                                 all_responses, all_thinking, stats)

        if not all_responses[-1].strip() and all_thinking:
            return {
                "id": pid, "title": p["title"], "category": p["category"],
                "turns": turns, "response": "", "responses": all_responses,
                "thinking": "\n---\n".join(all_thinking),
                "elapsed_seconds": round(stats["elapsed"], 2),
                "completion_tokens": stats["tokens"],
                "error": "truncated: thinking consumed all tokens, no answer produced",
                "score": None,
            }, "error"

        return {
            "id": pid, "title": p["title"], "category": p["category"],
            "turns": turns,
            "response": all_responses[-1],       # final response for backward compat
            "responses": all_responses,           # all turn responses
            "thinking": "\n---\n".join(all_thinking),
            "elapsed_seconds": round(stats["elapsed"], 2),
            "completion_tokens": stats["tokens"],
            "score": None, "judge_reason": "", "notes": "",
        }, f"{round(stats['elapsed'], 2)}s"
    except Exception as e:
        log.error("run_prompt (multi-turn) failed  prompt=#%d title=%s: %s", pid, p.get("title", ""), e)
        return {
            "id": pid, "title": p["title"], "category": p["category"],
            "error": str(e), "score": None,
        }, "error"


# ──────────────────────────────────────────────
# Extracted functions for server.py import
# ──────────────────────────────────────────────

def run_eval(models, prompts, cfg, api_url, api_key, image_b64=None, on_progress=None):
    """Run eval prompts against models. Importable from server.py.

    Args:
        models: list of model name strings
        prompts: list of prompt dicts from eval set
        cfg: config dict (temperature, max_tokens, _thinking_budget, etc.)
        api_url: inference endpoint URL
        api_key: API key
        image_b64: optional base64 image for vision prompts
        on_progress: callback(model, prompt_id, status, elapsed) for SSE

    Returns:
        (all_results, saved_files) where all_results = {model: [result_dicts]}
    """
    output_dir = resolve_path(cfg.get("output_dir", "./eval-results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    thinking_budget = cfg.get("_thinking_budget")

    all_results = {m: [] for m in models}

    for model in models:
        for p in prompts:
            if on_progress:
                on_progress(model, p["id"], "running", 0)
            result, status = run_prompt(api_url, api_key, model, p, cfg, image_b64)
            all_results[model].append(result)
            elapsed = result.get("elapsed_seconds", 0)
            if on_progress:
                on_progress(model, p["id"], status, elapsed)

    # Save per-model JSONs
    t_label = f"_think{thinking_budget}" if thinking_budget is not None else ""
    saved_files = []
    for model in models:
        safe = model.replace("/", "_").replace(" ", "_")
        out = output_dir / f"eval_{safe}{t_label}_{timestamp}.json"
        with open(out, "w") as f:
            json.dump({
                "model": model, "api_url": api_url,
                "thinking_budget": thinking_budget,
                "timestamp": timestamp,
                "results": all_results[model],
            }, f, indent=2)
        saved_files.append(str(out))

    return all_results, saved_files


def rubric_is_empty(rubric) -> bool:
    """True when a rubric carries nothing to grade against.

    Covers the absent case and the subtler one: a criteria rubric whose `criteria`
    list is empty is a non-empty dict, so a plain truthiness check waves it through
    and the judge scores every criterion unmet — a 0.
    """
    if not rubric:
        return True
    if _is_criteria_rubric(rubric):
        return not rubric.get("criteria")
    # a level rubric needs at least one level description
    return not any(str(v).strip() for v in rubric.values() if not isinstance(v, (dict, list)))


def judge_one(result, rubric, prompt_text, judge_url, judge_key, judge_model,
              replace=False):
    """Judge a single result in place. Returns (status, score).

    status is "skip", "scored", or "error". This is the one place the decision to
    skip, score, or record a failure lives — the CLI, the server, and
    `judge_results` each used to carry their own copy of it, and every fix to one
    had to be remembered in the other three. It wasn't, three times over: the
    empty-rubric guard and the failure-path desync each shipped to some copies and
    not others.
    """
    has_response = result.get("response") or any(r for r in result.get("responses", []))
    if result.get("skipped") or result.get("error") or not has_response:
        return "skip", None

    # An absent rubric means this prompt isn't in the eval set being used. Judging
    # anyway hands the judge an empty rubric, which it reports as "nothing met" and
    # scores 0 — turning a missing rubric into a failing grade.
    if rubric_is_empty(rubric):
        record_judge_error(result, judge_model,
                           "no rubric for this prompt in the eval set — not scored")
        return "skip", None

    try:
        score, reason, criteria_results = score_one(
            judge_url, judge_key, judge_model, prompt_text, rubric,
            result["response"], responses=result.get("responses"),
        )
    except Exception as e:
        log.warning("judge scoring failed  prompt=#%s: %s", result.get("id"), e)
        record_judge_error(result, judge_model, f"judge error: {e}")
        return "error", None

    if score is None:
        # a failed judgment adds no judgment, so it must not touch the mirrored
        # score/reason — those belong to the judgments that did stand
        record_judge_error(result, judge_model, reason)
        return "error", None

    apply_judgment(result, score, reason, criteria_results, judge_model, rubric,
                   judge_url=judge_url, replace=replace)
    return "scored", score


def judge_results(data_files, judge_model, judge_url, judge_key, rubric_by_id,
                  prompt_by_id, on_progress=None, replace=False):
    """Score existing result files. Importable from server.py.

    Judgments are appended, so re-judging with a different judge keeps the earlier
    one and the two become comparable. Pass replace=True for the old destructive
    behaviour.

    Args:
        data_files: list of (filepath, data_dict) tuples
        judge_model: model name for judging
        judge_url: judge API endpoint
        judge_key: judge API key
        rubric_by_id: {prompt_id: rubric_dict}
        prompt_by_id: {prompt_id: prompt_text}
        on_progress: callback(model, prompt_id, status, elapsed) for SSE
        replace: discard existing judgments instead of appending

    Returns:
        list of updated (filepath, data_dict) tuples
    """
    for fpath, data in data_files:
        model = data["model"]
        # bring pre-judgments files into the current shape before adding to them
        normalize_run(data)
        for result in data["results"]:
            pid = result["id"]
            if on_progress:
                on_progress(model, pid, "judging", 0)

            status, score = judge_one(
                result, rubric_by_id.get(pid), prompt_by_id.get(pid, ""),
                judge_url, judge_key, judge_model, replace=replace,
            )
            if on_progress:
                if status == "scored":
                    on_progress(model, pid, f"scored:{score}", result.get("elapsed_seconds", ""))
                else:
                    on_progress(model, pid, status, 0)

        # Save scores back — refresh_judges rebuilds the run-level judge list and
        # points the legacy judge/judge_url fields at the most recent judge.
        refresh_judges(data)
        with open(fpath, "w") as f:
            json.dump(data, f, indent=2)

    return data_files

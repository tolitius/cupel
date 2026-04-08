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
from cupel.config import IMAGE_PROMPT_ID

log = logging.getLogger("cupel")


# ──────────────────────────────────────────────
# Image handling
# ──────────────────────────────────────────────

def find_image(image_filename: str, image_dir: Path | None) -> str | None:
    candidates = []
    if image_dir:
        candidates.append(image_dir / image_filename)
    candidates.extend([
        Path.cwd() / image_filename,
        Path.cwd() / "eval-sets" / image_filename,
        Path(__file__).parent / "data" / image_filename,
        Path(__file__).parent.parent / image_filename,
        Path.home() / image_filename,
        Path.home() / "eval-images" / image_filename,
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
    temperature: float = 0, max_tokens: int = 16384,
    thinking_budget: int | None = None, image_b64: str | None = None,
) -> dict:
    content = build_vision_content(prompt, image_b64) if image_b64 else prompt
    messages = [{"role": "user", "content": content}]
    return _call_llm_raw(api_url, api_key, model, messages, temperature, max_tokens, thinking_budget)


def call_llm_multi(
    api_url: str, api_key: str, model: str, messages: list[dict],
    temperature: float = 0, max_tokens: int = 16384,
    thinking_budget: int | None = None,
) -> dict:
    return _call_llm_raw(api_url, api_key, model, messages, temperature, max_tokens, thinking_budget)


def _call_llm_raw(
    api_url: str, api_key: str, model: str, messages: list[dict],
    temperature: float = 0, max_tokens: int = 16384,
    thinking_budget: int | None = None,
) -> dict:

    is_anthropic = "api.anthropic.com" in api_url
    is_openrouter = "openrouter.ai" in api_url

    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

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

    data = resp.json()
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
    else:
        # OpenAI-compatible response
        choice = data["choices"][0]["message"]
        content_text = choice.get("content") or ""   # content can be null for tool_calls
        thinking = choice.get("thinking") or ""
        # OpenRouter returns reasoning in a separate field
        if not thinking and choice.get("reasoning"):
            thinking = choice["reasoning"]
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

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

    return {
        "content": content_text,
        "thinking": thinking,
        "elapsed_seconds": round(elapsed, 2),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": usage.get("total_tokens", prompt_tokens + completion_tokens),
    }


# ──────────────────────────────────────────────
# Judge — auto-score using rubrics
# ──────────────────────────────────────────────

JUDGE_SYSTEM = """You are a strict, fair evaluator scoring LLM responses on a 0-3 scale.

0 = wrong or hallucinated
1 = partially correct
2 = correct but shallow
3 = correct and insightful

You MUST respond with ONLY a JSON object, no other text:
{"score": <0-3>, "reason": "<one sentence>"}"""


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


def build_judge_prompt(prompt_text: str, rubric: dict, response_text: str,
                       responses: list[str] | None = None) -> str:
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


def parse_judge_response(text: str) -> tuple[int | None, str]:
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
    try:
        obj = json.loads(cleaned)
        score = int(obj.get("score", -1))
        reason = obj.get("reason", "")
        if 0 <= score <= 3:
            return score, reason
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    m = re.search(r'"score"\s*:\s*(\d)', text)
    if m:
        score = int(m.group(1))
        if 0 <= score <= 3:
            r = re.search(r'"reason"\s*:\s*"([^"]*)"', text)
            return score, r.group(1) if r else ""
    return None, f"unparseable judge response: {text[:100]}"


def score_one(api_url, api_key, judge_model, prompt_text, rubric, response_text,
              responses=None):
    """Score a single response using the judge model."""
    judge_prompt = build_judge_prompt(prompt_text, rubric, response_text, responses)
    resp = call_llm(
        api_url, api_key, judge_model, judge_prompt,
        temperature=0, max_tokens=256,
        thinking_budget=0,
    )
    return parse_judge_response(resp["content"])


# ──────────────────────────────────────────────
# Command: run
# ──────────────────────────────────────────────

def run_prompt(api_url, api_key, model, p, cfg, image_b64):
    pid = p["id"]

    if pid == IMAGE_PROMPT_ID and not image_b64:
        return {
            "id": pid, "title": p["title"], "category": p["category"],
            "skipped": True, "reason": "image not provided",
        }, "skip"

    # ── multi-turn prompts ──
    if "turns" in p:
        return _run_multi_turn(api_url, api_key, model, p, cfg)

    try:
        use_image = image_b64 if pid == IMAGE_PROMPT_ID else None
        resp = call_llm(
            api_url, api_key, model, p["prompt"],
            temperature=cfg.get("temperature", 0),
            max_tokens=cfg.get("max_tokens", 16384),
            thinking_budget=cfg.get("_thinking_budget"),
            image_b64=use_image,
        )

        content = resp["content"]
        thinking = resp.get("thinking", "")

        if not content.strip() and thinking:
            return {
                "id": pid, "title": p["title"], "category": p["category"],
                "prompt": p["prompt"], "response": "", "thinking": thinking,
                "elapsed_seconds": resp["elapsed_seconds"],
                "completion_tokens": resp["completion_tokens"],
                "error": "truncated: thinking consumed all tokens, no answer produced",
                "score": None,
            }, "error"

        return {
            "id": pid, "title": p["title"], "category": p["category"],
            "prompt": p["prompt"], "response": content, "thinking": thinking,
            "elapsed_seconds": resp["elapsed_seconds"],
            "completion_tokens": resp["completion_tokens"],
            "score": None, "judge_reason": "", "notes": "",
        }, f"{resp['elapsed_seconds']}s"
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
        temperature=cfg.get("temperature", 0),
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
    output_dir = Path(cfg.get("output_dir", "./eval-results"))
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


def judge_results(data_files, judge_model, judge_url, judge_key, rubric_by_id,
                  prompt_by_id, on_progress=None):
    """Score existing result files. Importable from server.py.

    Args:
        data_files: list of (filepath, data_dict) tuples
        judge_model: model name for judging
        judge_url: judge API endpoint
        judge_key: judge API key
        rubric_by_id: {prompt_id: rubric_dict}
        prompt_by_id: {prompt_id: prompt_text}
        on_progress: callback(model, prompt_id, status, elapsed) for SSE

    Returns:
        list of updated (filepath, data_dict) tuples
    """
    for fpath, data in data_files:
        model = data["model"]
        for result in data["results"]:
            pid = result["id"]

            has_response = result.get("response") or any(r for r in result.get("responses", []))
            if result.get("skipped") or result.get("error") or not has_response:
                if on_progress:
                    on_progress(model, pid, "skip", 0)
                continue

            if on_progress:
                on_progress(model, pid, "judging", 0)

            try:
                score, reason = score_one(
                    judge_url, judge_key, judge_model,
                    prompt_by_id.get(pid, ""),
                    rubric_by_id.get(pid, {}),
                    result["response"],
                    responses=result.get("responses"),
                )
                if score is not None:
                    result["score"] = score
                    result["judge_reason"] = reason
                    result["judge_model"] = judge_model
                    elapsed = result.get("elapsed_seconds", "")
                    if on_progress:
                        on_progress(model, pid, f"scored:{score}", elapsed)
                else:
                    result["judge_reason"] = reason
                    if on_progress:
                        on_progress(model, pid, "error", 0)
            except Exception as e:
                result["judge_reason"] = f"judge error: {e}"
                if on_progress:
                    on_progress(model, pid, "error", 0)

        # Save scores back
        data["judge"] = judge_model
        data["judge_url"] = judge_url
        with open(fpath, "w") as f:
            json.dump(data, f, indent=2)

    return data_files

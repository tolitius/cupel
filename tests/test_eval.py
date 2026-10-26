"""Smoke tests for cupel.eval helpers and bundled data files."""

import json
from pathlib import Path

from cupel.eval import build_vision_content

DATA_DIR = Path(__file__).resolve().parent.parent / "cupel" / "data"


# ── build_vision_content ──

def test_build_vision_content_structure():
    result = build_vision_content("describe this", "AAAA")
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["type"] == "image_url"
    assert "AAAA" in result[0]["image_url"]["url"]
    assert result[1]["type"] == "text"
    assert result[1]["text"] == "describe this"


# ── starter-eval-set.json validity ──

def test_starter_eval_set_parses():
    path = DATA_DIR / "starter-eval-set.json"
    data = json.loads(path.read_text())
    prompts = data["prompts"]
    assert isinstance(prompts, list) and len(prompts) > 0

def test_starter_eval_set_required_keys():
    path = DATA_DIR / "starter-eval-set.json"
    data = json.loads(path.read_text())
    for p in data["prompts"]:
        assert "id" in p
        assert "title" in p
        assert "rubric" in p

def test_starter_eval_set_rubric_keys():
    path = DATA_DIR / "starter-eval-set.json"
    data = json.loads(path.read_text())
    for p in data["prompts"]:
        rubric = p["rubric"]
        for k in ("0", "1", "2", "3"):
            assert k in rubric, f"prompt {p['id']} missing rubric key {k}"

def test_starter_eval_set_unique_ids():
    path = DATA_DIR / "starter-eval-set.json"
    data = json.loads(path.read_text())
    ids = [p["id"] for p in data["prompts"]]
    assert len(ids) == len(set(ids)), "duplicate prompt ids"


# ── example-run.json validity ──

def test_example_run_parses():
    path = DATA_DIR / "example-run.json"
    data = json.loads(path.read_text())
    assert isinstance(data, dict)
    assert "models" in data
    assert isinstance(data["models"], list) and len(data["models"]) > 0

def test_example_run_model_results():
    path = DATA_DIR / "example-run.json"
    data = json.loads(path.read_text())
    for model in data["models"]:
        assert "model" in model
        assert "results" in model
        for r in model["results"]:
            assert "id" in r
            assert "score" in r
            assert "response" in r

def test_example_run_scores_in_range():
    path = DATA_DIR / "example-run.json"
    data = json.loads(path.read_text())
    for model in data["models"]:
        for r in model["results"]:
            # an unscored prompt is missing data, not a zero — skip, don't coerce
            if r["score"] is None:
                continue
            assert 0 <= r["score"] <= 3, f"score {r['score']} out of range for model {model['model']} prompt {r['id']}"


# ── sampling params on Anthropic models ──

import pytest

from cupel import eval as ev

ANTHROPIC = "https://api.anthropic.com/v1/messages"


class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        raise AssertionError("raise_for_status should not be reached")


_OK = {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
       "usage": {"input_tokens": 1, "output_tokens": 1}}


@pytest.fixture
def posts(monkeypatch):
    """Capture every request body; answer with queued responses, then 200 OK."""
    sent, queue = [], []

    def _post(url, headers=None, json=None, timeout=None):
        sent.append(json)
        return queue.pop(0) if queue else _Resp(200, _OK)

    monkeypatch.setattr(ev.requests, "post", _post)
    monkeypatch.setattr(ev, "_NO_SAMPLING_MODELS", set())
    return sent, queue


def test_opus_5_5_gets_no_temperature(posts):
    sent, _ = posts
    ev.call_llm(ANTHROPIC, "k", "claude-opus-5-5", "hi", temperature=0)
    assert "temperature" not in sent[0]


def test_opus_4_6_keeps_temperature(posts):
    sent, _ = posts
    ev.call_llm(ANTHROPIC, "k", "claude-opus-4-6", "hi", temperature=0)
    assert sent[0]["temperature"] == 0


def test_non_anthropic_keeps_temperature(posts):
    sent, _ = posts
    posts[1].append(_Resp(200, {"choices": [{"message": {"content": "ok"}}], "usage": {}}))
    ev.call_llm("http://localhost:8000/v1/chat/completions", "k", "claude-opus-5-5",
                "hi", temperature=0)
    assert sent[0]["temperature"] == 0


def test_unknown_model_learns_to_drop_temperature(posts):
    sent, queue = posts
    queue.append(_Resp(400, {"error": {"message": "`temperature` is deprecated for this model."}}))
    out = ev.call_llm(ANTHROPIC, "k", "claude-foo-9", "hi", temperature=0)
    assert out["content"] == "ok"
    assert sent[0]["temperature"] == 0 and "temperature" not in sent[1]
    ev.call_llm(ANTHROPIC, "k", "claude-foo-9", "hi", temperature=0)
    assert len(sent) == 3 and "temperature" not in sent[2]


def test_other_400_is_not_retried(posts):
    sent, queue = posts
    queue.append(_Resp(400, {"error": {"message": "max_tokens too large"}}))
    with pytest.raises(RuntimeError, match="HTTP 400"):
        ev.call_llm(ANTHROPIC, "k", "claude-foo-9", "hi", temperature=0)
    assert len(sent) == 1


# ── judge circuit ──

def test_judge_pass_stops_when_judge_rejects_every_request(tmp_path, monkeypatch):
    calls = []

    def _reject(*a, **kw):
        calls.append(1)
        raise RuntimeError("HTTP 400: `temperature` is deprecated for this model.")

    monkeypatch.setattr(ev, "score_one", _reject)
    data = {"model": "m", "results": [
        {"id": i, "title": str(i), "category": "c", "response": f"answer {i}"}
        for i in range(1, 11)]}
    path = tmp_path / "eval_m.json"
    progress = []
    ev.judge_results([(str(path), data)], "claude-opus-5-5", ANTHROPIC, "k",
                     {i: {"3": "good"} for i in range(1, 11)}, {},
                     on_progress=lambda m, pid, st, el: progress.append(st))

    assert len(calls) == 3
    assert any(s.startswith("error:judge claude-opus-5-5 rejects requests") for s in progress)
    saved = json.loads(path.read_text())
    assert all(r.get("score") is None for r in saved["results"])
    assert sum(1 for r in saved["results"] if r.get("judge_errors")) == 3


def test_judge_circuit_disarms_after_a_success():
    c = ev.JudgeCircuit()
    err = {"judge_errors": [{"error": "judge error: HTTP 429: slow down"}]}
    assert c.record({}, "scored") is None
    for _ in range(5):
        assert c.record(err, "error") is None


# ── results summary flags runs whose judging failed ──

def test_summary_flags_judge_failed_runs(tmp_path):
    from cupel.server import _summarize_result
    data = {"model": "m", "results": [
        {"id": 1, "response": "a", "score": None, "judgments": [],
         "judge_errors": [{"judge_model": "j", "error": "judge error: HTTP 400: nope",
                           "at": "2026-10-25T17:29:01"}]},
        {"id": 2, "response": "b", "score": 2,
         "judgments": [{"judge_model": "j", "score": 2}]},
    ]}
    s = _summarize_result(tmp_path / "eval_m.json", data)
    assert s["n_judge_failed"] == 1
    assert "HTTP 400" in s["last_judge_error"]

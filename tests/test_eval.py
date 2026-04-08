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
            assert 0 <= r["score"] <= 3, f"score {r['score']} out of range for model {model['model']} prompt {r['id']}"

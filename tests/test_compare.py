"""Tests for cupel.tools.compare — paired run comparison."""

import json

import pytest

from cupel.tools.compare import (
    load_run, pick_metric, scores_by_id, criteria_rows, format_comparison,
)


def _run(model, results, **kw):
    return {"model": model, "results": results, **kw}


def _scored(i, score, check=None, criteria=None):
    r = {"id": i, "score": score}
    if check is not None:
        r["check_score"] = check
    if criteria is not None:
        r["criteria_results"] = criteria
    return r


# ── pick_metric ──

def test_pick_metric_prefers_check_score_when_present():
    runs = [_run("a", [_scored(1, 3, 1.0), _scored(2, 2, 0.5)]),
            _run("b", [_scored(1, 2, 0.6), _scored(2, 1, 0.2)])]
    assert pick_metric(runs) == "check_score"

def test_pick_metric_falls_back_when_any_run_lacks_it():
    runs = [_run("a", [_scored(1, 3, 1.0)]),
            _run("b", [_scored(1, 2)])]
    assert pick_metric(runs) == "score"

def test_pick_metric_falls_back_when_partially_present():
    """A run where only some tasks carry check_score cannot be compared on it."""
    runs = [_run("a", [_scored(1, 3, 1.0), _scored(2, 2)])]
    assert pick_metric(runs) == "score"

def test_pick_metric_respects_explicit_request():
    runs = [_run("a", [_scored(1, 3, 1.0)])]
    assert pick_metric(runs, "score") == "score"

def test_pick_metric_no_scored_results():
    assert pick_metric([_run("a", [{"id": 1, "score": None}])]) == "score"


# ── scores_by_id ──

def test_scores_by_id_skips_unscored():
    run = _run("a", [_scored(1, 3), {"id": 2, "score": None}, _scored(3, 1)])
    assert scores_by_id(run, "score") == {1: 3, 3: 1}

def test_scores_by_id_uses_requested_metric():
    run = _run("a", [_scored(1, 3, 0.9)])
    assert scores_by_id(run, "check_score") == {1: 0.9}

def test_scores_by_id_skips_error_rows():
    run = _run("a", [_scored(1, 3), {"id": 2, "error": "boom", "score": None}])
    assert 2 not in scores_by_id(run, "score")


# ── criteria_rows ──

def test_criteria_rows_extracts_vectors():
    run = _run("a", [_scored(1, 3, 1.0, [{"id": "P1", "met": True},
                                         {"id": "P2", "met": False}])])
    assert criteria_rows(run) == [{"P1": True, "P2": False}]

def test_criteria_rows_empty_for_level_rubrics():
    assert criteria_rows(_run("a", [_scored(1, 3)])) == []

def test_criteria_rows_ignores_malformed_entries():
    run = _run("a", [_scored(1, 3, 1.0, [{"met": True}, {"id": "P1", "met": True}])])
    assert criteria_rows(run) == [{"P1": True}]


# ── format_comparison ──

def test_same_model_is_labelled_noise_floor_not_a_win():
    """The failure this guards against: reporting run-to-run drift as a model win."""
    a = _run("same", [_scored(i, 3) for i in range(20)])
    b = _run("same", [_scored(i, 3 if i > 3 else 2) for i in range(20)])
    out = format_comparison(a, b, "score")
    assert "NOISE FLOOR" in out
    assert "is ahead" not in out

def test_different_models_can_resolve():
    a = _run("strong", [_scored(i, 3) for i in range(20)])
    b = _run("weak", [_scored(i, 1) for i in range(20)])
    out = format_comparison(a, b, "score")
    assert "strong is ahead" in out
    assert "NOISE FLOOR" not in out

def test_no_difference_is_reported_unresolved():
    a = _run("a", [_scored(i, 2) for i in range(20)])
    b = _run("b", [_scored(i, 2) for i in range(20)])
    out = format_comparison(a, b, "score")
    assert "NOT RESOLVED" in out

def test_single_run_caveat_shown_for_real_comparisons():
    a = _run("a", [_scored(i, 3) for i in range(10)])
    b = _run("b", [_scored(i, 1) for i in range(10)])
    assert "run-to-run variance" in format_comparison(a, b, "score")

def test_no_shared_tasks():
    a = _run("a", [_scored(1, 3)])
    b = _run("b", [_scored(99, 3)])
    assert "nothing to pair" in format_comparison(a, b, "score")

def test_overlap_is_reported():
    a = _run("a", [_scored(i, 2) for i in range(10)])
    b = _run("b", [_scored(i, 2) for i in range(10)])
    assert "OVERLAP" in format_comparison(a, b, "score")

def test_criteria_breakdown_flags_dead_weight():
    """A criterion every run passes carries no information and should be called out."""
    crit = [{"id": "P1", "met": True}, {"id": "P2", "met": True}]
    crit_b = [{"id": "P1", "met": True}, {"id": "P2", "met": False}]
    a = _run("a", [_scored(i, 3, 1.0, crit) for i in range(6)])
    b = _run("b", [_scored(i, 2, 0.5, crit_b) for i in range(6)])
    out = format_comparison(a, b, "check_score")
    assert "dead weight" in out and "P1" in out
    assert "P2" in out


# ── load_run ──

def test_load_run_rejects_file_without_results(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"model": "m"}))
    with pytest.raises(ValueError):
        load_run(p)

def test_load_run_accepts_valid(tmp_path):
    p = tmp_path / "ok.json"
    p.write_text(json.dumps(_run("m", [_scored(1, 3)])))
    assert load_run(p)["model"] == "m"

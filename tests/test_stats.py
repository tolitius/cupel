"""Tests for cupel.stats — intervals, paired comparison, criteria scoring."""

import pytest

from cupel.stats import (
    bootstrap_ci, score_pct_ci, overlaps, paired_bootstrap,
    criteria_score, criterion_discrimination,
    run_group_key, aggregate_runs, rank_with_ties,
)


# ── bootstrap_ci ──

def test_bootstrap_ci_empty():
    assert bootstrap_ci([]) == (0.0, 0.0)

def test_bootstrap_ci_single_value_has_no_spread():
    assert bootstrap_ci([2]) == (2.0, 2.0)

def test_bootstrap_ci_brackets_the_mean():
    vals = [0, 1, 2, 3, 3, 2, 1, 3, 2, 2]
    lo, hi = bootstrap_ci(vals)
    mean = sum(vals) / len(vals)
    assert lo <= mean <= hi

def test_bootstrap_ci_is_deterministic():
    vals = [0, 1, 2, 3, 3, 2]
    assert bootstrap_ci(vals) == bootstrap_ci(vals)

def test_bootstrap_ci_narrows_with_more_data():
    few = bootstrap_ci([0, 3] * 5)
    many = bootstrap_ci([0, 3] * 100)
    assert (many[1] - many[0]) < (few[1] - few[0])

def test_bootstrap_ci_zero_width_when_all_identical():
    lo, hi = bootstrap_ci([2] * 20)
    assert lo == hi == 2.0

def test_bootstrap_ci_ignores_none():
    assert bootstrap_ci([2, None, 2, None]) == (2.0, 2.0)


# ── score_pct_ci ──

def test_score_pct_ci_perfect_scores():
    lo, hi = score_pct_ci([3] * 10)
    assert lo == hi == 100.0

def test_score_pct_ci_empty():
    assert score_pct_ci([]) == (0.0, 0.0)

def test_score_pct_ci_scales_to_max_per_item():
    lo, hi = score_pct_ci([1] * 10, max_per_item=2)
    assert lo == hi == 50.0


# ── overlaps ──

@pytest.mark.parametrize("a,b,expected", [
    ((0, 10), (5, 15), True),
    ((0, 10), (10, 20), True),    # touching counts as unresolved
    ((0, 10), (11, 20), False),
    ((5, 15), (0, 10), True),
])
def test_overlaps(a, b, expected):
    assert overlaps(a, b) is expected


# ── paired_bootstrap ──

def test_paired_bootstrap_no_shared_tasks():
    r = paired_bootstrap({1: 3}, {2: 3})
    assert r["n"] == 0
    assert r["significant"] is False

def test_paired_bootstrap_identical_models_not_significant():
    a = {i: 2 for i in range(20)}
    r = paired_bootstrap(a, dict(a))
    assert r["mean_diff"] == 0
    assert r["significant"] is False

def test_paired_bootstrap_consistent_winner_is_significant():
    a = {i: 3 for i in range(20)}
    b = {i: 1 for i in range(20)}
    r = paired_bootstrap(a, b)
    assert r["mean_diff"] == 2
    assert r["a_wins"] == 20 and r["b_wins"] == 0
    assert r["significant"] is True

def test_paired_bootstrap_only_uses_shared_ids():
    r = paired_bootstrap({1: 3, 2: 3, 9: 0}, {1: 1, 2: 1})
    assert r["n"] == 2

def test_paired_bootstrap_counts_ties():
    r = paired_bootstrap({1: 2, 2: 3}, {1: 2, 2: 1})
    assert r["ties"] == 1 and r["a_wins"] == 1

def test_paired_bootstrap_detects_gap_independent_cis_cannot():
    """The reason paired comparison exists: task difficulty cancels.

    A small consistent edge — A beats B on 5 of 15 tasks and ties the rest — is
    invisible to two independent intervals but resolved by the paired test.
    """
    base = [0, 1, 2, 2, 1, 0, 2, 1, 2, 0, 1, 2, 1, 0, 2]
    b = {i: v for i, v in enumerate(base)}
    a = {i: (min(3, v + 1) if i < 5 else v) for i, v in enumerate(base)}

    paired = paired_bootstrap(a, b)
    assert paired["significant"] is True
    assert paired["a_wins"] == 5 and paired["b_wins"] == 0

    # the same data as two independent intervals cannot resolve it
    assert overlaps(score_pct_ci(list(a.values())), score_pct_ci(list(b.values())))


# ── criteria_score ──

RUBRIC = {
    "criteria": [
        {"id": "P1", "type": "positive", "weight": 4, "check": "..."},
        {"id": "P2", "type": "positive", "weight": 3, "check": "..."},
        {"id": "P3", "type": "positive", "weight": 3, "check": "..."},
        {"id": "N1", "type": "negative", "weight": 4, "check": "..."},
    ]
}

def test_criteria_score_all_positives_met():
    cr = [{"id": "P1", "met": True}, {"id": "P2", "met": True}, {"id": "P3", "met": True}]
    assert criteria_score(cr, RUBRIC) == 1.0

def test_criteria_score_none_met():
    cr = [{"id": "P1", "met": False}, {"id": "P2", "met": False}]
    assert criteria_score(cr, RUBRIC) == 0.0

def test_criteria_score_partial_is_weighted():
    cr = [{"id": "P1", "met": True}, {"id": "P2", "met": False}, {"id": "P3", "met": False}]
    assert criteria_score(cr, RUBRIC) == pytest.approx(4 / 10)

def test_criteria_score_negative_subtracts():
    cr = [{"id": "P1", "met": True}, {"id": "P2", "met": True}, {"id": "N1", "met": True}]
    assert criteria_score(cr, RUBRIC) == pytest.approx(3 / 10)

def test_criteria_score_clamps_at_zero():
    cr = [{"id": "P1", "met": False}, {"id": "N1", "met": True}]
    assert criteria_score(cr, RUBRIC) == 0.0

def test_criteria_score_ignores_unknown_ids():
    cr = [{"id": "P1", "met": True}, {"id": "ZZ", "met": True}]
    assert criteria_score(cr, RUBRIC) == pytest.approx(4 / 10)

def test_criteria_score_rejects_level_rubric():
    assert criteria_score([{"id": "P1", "met": True}], {"0": "a", "3": "b"}) is None

def test_criteria_score_is_finer_grained_than_collapse():
    """The whole point: the vector distinguishes runs the 0-3 collapse merges."""
    good = [{"id": "P1", "met": True}, {"id": "P2", "met": True}, {"id": "P3", "met": False}]
    better = [{"id": "P1", "met": True}, {"id": "P2", "met": True}, {"id": "P3", "met": True}]
    assert criteria_score(good, RUBRIC) != criteria_score(better, RUBRIC)


# ── criterion_discrimination ──

def test_discrimination_zero_when_unanimous():
    rows = [{"P1": True}] * 10
    assert criterion_discrimination(rows)["P1"]["discrimination"] == 0.0

def test_discrimination_max_at_even_split():
    rows = [{"P1": True}] * 5 + [{"P1": False}] * 5
    assert criterion_discrimination(rows)["P1"]["discrimination"] == pytest.approx(1.0)

def test_discrimination_reports_counts():
    rows = [{"P1": True}] * 3 + [{"P1": False}]
    d = criterion_discrimination(rows)["P1"]
    assert d["n"] == 4 and d["met"] == 3 and d["pass_rate"] == 0.75


# ── run_group_key ──

def _run(**kw):
    base = {"model": "m", "results": [], "judge": "j", "thinking_budget": None}
    base.update(kw)
    return base

def test_group_key_same_config_matches():
    assert run_group_key(_run()) == run_group_key(_run())

def test_group_key_separates_thinking_budget():
    assert run_group_key(_run(thinking_budget=4096)) != run_group_key(_run())

def test_group_key_separates_judge():
    assert run_group_key(_run(judge="a")) != run_group_key(_run(judge="b"))

def test_group_key_separates_eval_set_by_hash():
    a = _run(eval_set={"name": "x", "hash": "aaa"})
    b = _run(eval_set={"name": "x", "hash": "bbb"})
    assert run_group_key(a) != run_group_key(b)

def test_group_key_separates_scoring_version():
    a = _run(judges=[{"model": "j", "scoring_version": "v1"}])
    b = _run(judges=[{"model": "j", "scoring_version": "v2"}])
    assert run_group_key(a) != run_group_key(b)

def test_group_key_legacy_run_marked_legacy():
    assert run_group_key(_run())[3] == "legacy"


# ── aggregate_runs ──

def _scored_run(model, scores, **kw):
    return _run(model=model,
                results=[{"id": i, "score": s} for i, s in enumerate(scores)],
                **kw)

def test_aggregate_collapses_repeat_runs():
    runs = [_scored_run("m", [3, 3, 2]), _scored_run("m", [3, 2, 2])]
    groups = aggregate_runs(runs)
    assert len(groups) == 1
    assert groups[0]["n_runs"] == 2
    assert groups[0]["n_scored"] == 6

def test_aggregate_reports_spread_across_runs():
    runs = [_scored_run("m", [3, 3, 3]), _scored_run("m", [0, 0, 0])]
    assert aggregate_runs(runs)[0]["spread"] == 100.0

def test_aggregate_excludes_unscored_from_denominator():
    """An errored prompt is missing data, not a zero."""
    run = _run(model="m", results=[{"id": 1, "score": 3}, {"id": 2, "score": None}])
    g = aggregate_runs([run])[0]
    assert g["n_scored"] == 1
    assert g["pct"] == 100.0

def test_aggregate_skips_runs_with_no_scores():
    assert aggregate_runs([_scored_run("m", [])]) == []

def test_aggregate_sorted_descending():
    runs = [_scored_run("lo", [1, 1]), _scored_run("hi", [3, 3])]
    assert [g["model"] for g in aggregate_runs(runs)] == ["hi", "lo"]


# ── rank_with_ties ──

def test_rank_ties_when_intervals_overlap():
    entries = [
        {"pct": 90.0, "ci_lo": 80.0, "ci_hi": 95.0},
        {"pct": 88.0, "ci_lo": 78.0, "ci_hi": 93.0},
    ]
    ranked = rank_with_ties(entries)
    assert ranked[0]["rank"] == ranked[1]["rank"] == 1
    assert ranked[1]["tied"] is True

def test_rank_separates_when_intervals_clear():
    entries = [
        {"pct": 90.0, "ci_lo": 85.0, "ci_hi": 95.0},
        {"pct": 50.0, "ci_lo": 40.0, "ci_hi": 60.0},
    ]
    ranked = rank_with_ties(entries)
    assert ranked[0]["rank"] == 1 and ranked[1]["rank"] == 2
    assert ranked[1]["tied"] is False

def test_rank_empty():
    assert rank_with_ties([]) == []

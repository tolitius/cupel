"""Tests for cupel.schema — judgments, consensus, and legacy migration."""

import pytest

from cupel.schema import (
    drop_judge, repair_mirror, record_judge_error, match_eval_set, infer_eval_set,
    LEGACY_VERSION, scoring_version, eval_set_meta, make_judgment, consensus,
    apply_consensus, add_judgment, normalize_result, normalize_run,
    refresh_judges, judge_models,
)


# ── scoring_version ──

def test_scoring_version_is_stable():
    r = {"3": "good", "0": "bad"}
    assert scoring_version(r, "sys") == scoring_version(r, "sys")

def test_scoring_version_ignores_key_order():
    a = scoring_version({"0": "bad", "3": "good"}, "sys")
    b = scoring_version({"3": "good", "0": "bad"}, "sys")
    assert a == b

def test_scoring_version_changes_with_rubric():
    assert scoring_version({"3": "a"}, "sys") != scoring_version({"3": "b"}, "sys")

def test_scoring_version_changes_with_judge_prompt():
    """The judge prompt was rewritten once already and silently rescaled scores."""
    r = {"3": "good"}
    assert scoring_version(r, "lenient") != scoring_version(r, "strict")

def test_scoring_version_is_short():
    assert len(scoring_version({}, "")) == 12


# ── eval_set_meta ──

def test_eval_set_meta_hashes_prompts():
    a = eval_set_meta("p.json", {"name": "x", "prompts": [{"id": 1}]})
    b = eval_set_meta("p.json", {"name": "x", "prompts": [{"id": 2}]})
    assert a["hash"] != b["hash"]

def test_eval_set_meta_renaming_does_not_break_comparability():
    a = eval_set_meta("p.json", {"name": "old", "prompts": [{"id": 1}]})
    b = eval_set_meta("p.json", {"name": "new", "prompts": [{"id": 1}]})
    assert a["hash"] == b["hash"]

def test_eval_set_meta_counts_prompts():
    m = eval_set_meta("p.json", {"name": "x", "prompts": [{"id": 1}, {"id": 2}]})
    assert m["n_prompts"] == 2


# ── consensus ──

def _j(score, model="j", reason="r", **kw):
    return make_judgment(model, "url", score, reason, **kw)

def test_consensus_single_judgment_passes_through():
    c = consensus([_j(2)])
    assert c["score"] == 2 and c["judge_agreement"] == 0

def test_consensus_takes_lower_median_on_even_counts():
    """Conservative by design — the judge prompt says 'when in doubt, mark lower'."""
    assert consensus([_j(2), _j(3)])["score"] == 2

def test_consensus_median_of_three():
    assert consensus([_j(0), _j(2), _j(3)])["score"] == 2

def test_consensus_median_of_four():
    assert consensus([_j(0), _j(1), _j(2), _j(3)])["score"] == 1

def test_consensus_reports_agreement_spread():
    assert consensus([_j(0), _j(3)])["judge_agreement"] == 3

def test_consensus_unanimous_has_no_spread():
    assert consensus([_j(2), _j(2), _j(2)])["judge_agreement"] == 0

def test_consensus_reason_matches_reported_score():
    c = consensus([_j(1, "a", "low"), _j(3, "b", "high")])
    assert c["score"] == 1 and c["judge_reason"] == "low" and c["judge_model"] == "a"

def test_consensus_no_judgments():
    assert consensus([])["score"] is None

def test_consensus_ignores_unscored_judgments():
    c = consensus([_j(2), _j(None)])
    assert c["score"] == 2 and c["n_judgments"] == 1

def test_consensus_carries_criteria_from_representative():
    crit = [{"id": "P1", "met": True}]
    c = consensus([_j(2, criteria_results=crit, check_score=0.5)])
    assert c["criteria_results"] == crit and c["check_score"] == 0.5


# ── add_judgment / apply_consensus ──

def test_add_judgment_appends_and_preserves_prior():
    r = {"score": 3, "judgments": [_j(3, "first")]}
    add_judgment(r, _j(1, "second"))
    assert len(r["judgments"]) == 2
    assert r["judgments"][0]["judge_model"] == "first"

def test_add_judgment_updates_mirror_to_consensus():
    r = {"judgments": [_j(3, "a")]}
    add_judgment(r, _j(1, "b"))
    assert r["score"] == 1          # lower median of [1, 3]
    assert r["judge_agreement"] == 2

def test_add_judgment_replace_discards_prior():
    r = {"score": 3, "judgments": [_j(3, "first")]}
    add_judgment(r, _j(0, "second"), replace=True)
    assert len(r["judgments"]) == 1 and r["score"] == 0

def test_add_judgment_to_fresh_result():
    r = {}
    add_judgment(r, _j(2))
    assert r["score"] == 2 and len(r["judgments"]) == 1

def test_single_judgment_sets_no_consensus_marker():
    """A single-judge run must look exactly like it did before this change."""
    r = {}
    add_judgment(r, _j(2))
    assert "judge_agreement" not in r and "judge_consensus" not in r

def test_apply_consensus_noop_without_judgments():
    r = {"score": 3}
    assert apply_consensus(r)["score"] == 3


# ── normalize_result ──

def test_normalize_synthesizes_judgment_from_legacy_fields():
    run = {"judge": "opus", "judge_url": "u", "timestamp": "20260101_000000"}
    r = {"id": 1, "score": 2, "judge_reason": "ok"}
    normalize_result(r, run)
    assert len(r["judgments"]) == 1
    j = r["judgments"][0]
    assert j["judge_model"] == "opus" and j["score"] == 2
    assert j["scoring_version"] == LEGACY_VERSION

def test_normalize_prefers_per_result_judge_model():
    run = {"judge": "run-level"}
    r = {"id": 1, "score": 2, "judge_model": "result-level"}
    normalize_result(r, run)
    assert r["judgments"][0]["judge_model"] == "result-level"

def test_normalize_unscored_gets_empty_list_not_fake_judgment():
    r = {"id": 1, "score": None}
    normalize_result(r, {})
    assert r["judgments"] == []

def test_normalize_is_idempotent():
    r = {"id": 1, "score": 2, "judge_reason": "ok"}
    normalize_result(r, {"judge": "j"})
    normalize_result(r, {"judge": "j"})
    assert len(r["judgments"]) == 1

def test_normalize_does_not_change_visible_score():
    r = {"id": 1, "score": 2, "judge_reason": "ok", "judge_model": "j"}
    before = r["score"]
    normalize_result(r, {})
    assert r["score"] == before

def test_normalize_carries_criteria_results():
    crit = [{"id": "P1", "met": True}]
    r = {"id": 1, "score": 3, "criteria_results": crit, "check_score": 0.9}
    normalize_result(r, {})
    assert r["judgments"][0]["criteria_results"] == crit
    assert r["judgments"][0]["check_score"] == 0.9


# ── normalize_run ──

def test_normalize_run_builds_judge_list():
    data = {"judge": "opus", "judge_url": "u",
            "results": [{"id": 1, "score": 2}, {"id": 2, "score": 3}]}
    normalize_run(data)
    assert judge_models(data) == ["opus"]

def test_normalize_run_dedupes_judges():
    data = {"judge": "opus", "results": [{"id": i, "score": 2} for i in range(5)]}
    normalize_run(data)
    assert len(data["judges"]) == 1

def test_normalize_run_handles_empty():
    data = {"results": []}
    normalize_run(data)
    assert data["judges"] == []

def test_normalize_run_is_idempotent():
    data = {"judge": "opus", "results": [{"id": 1, "score": 2}]}
    normalize_run(data)
    normalize_run(data)
    assert len(data["judges"]) == 1
    assert len(data["results"][0]["judgments"]) == 1


# ── refresh_judges ──

def test_refresh_judges_lists_multiple_judges():
    data = {"results": [{"id": 1, "judgments": [
        make_judgment("a", "u", 2, "r", judged_at="2026-01-01T00:00:00"),
        make_judgment("b", "u", 3, "r", judged_at="2026-02-01T00:00:00"),
    ]}]}
    refresh_judges(data)
    assert sorted(judge_models(data)) == ["a", "b"]

def test_refresh_judges_points_legacy_fields_at_latest():
    data = {"results": [{"id": 1, "judgments": [
        make_judgment("old", "u1", 2, "r", judged_at="2026-01-01T00:00:00"),
        make_judgment("new", "u2", 3, "r", judged_at="2026-02-01T00:00:00"),
    ]}]}
    refresh_judges(data)
    assert data["judge"] == "new" and data["judge_url"] == "u2"


# ── end to end on a realistic legacy run ──

def test_legacy_run_survives_rejudging_with_a_second_judge():
    """The whole point of Phase 1: re-judging must not destroy the first judgment."""
    data = {
        "model": "m", "judge": "opus", "judge_url": "u", "timestamp": "20260101_000000",
        "results": [{"id": 1, "score": 3, "judge_reason": "great", "judge_model": "opus"}],
    }
    normalize_run(data)
    result = data["results"][0]

    add_judgment(result, make_judgment("local-judge", "u2", 1, "actually weak"))
    refresh_judges(data)

    assert len(result["judgments"]) == 2
    assert result["judgments"][0]["judge_model"] == "opus"
    assert result["judgments"][0]["score"] == 3        # original preserved
    assert result["score"] == 1                        # lower median of [3, 1]
    assert result["judge_agreement"] == 2
    assert sorted(judge_models(data)) == ["local-judge", "opus"]


# ── drop_judge / repair_mirror / record_judge_error ──

def test_drop_judge_removes_and_recomputes():
    r = {"judgments": [_j(3, "keep"), _j(0, "remove")]}
    apply_consensus(r)
    assert r["score"] == 0
    assert drop_judge(r, "remove") is True
    assert r["score"] == 3 and len(r["judgments"]) == 1

def test_drop_judge_noop_for_unknown_judge():
    r = {"judgments": [_j(3, "keep")]}
    apply_consensus(r)
    assert drop_judge(r, "nobody") is False
    assert r["score"] == 3

def test_drop_last_judgment_returns_to_unscored():
    r = {"judgments": [_j(3, "only")]}
    apply_consensus(r)
    drop_judge(r, "only")
    assert r["score"] is None and r["judge_reason"] == ""

def test_drop_judge_clears_stale_derived_fields():
    r = {"judgments": [_j(3, "a"), _j(1, "b")]}
    apply_consensus(r)
    assert "judge_agreement" in r
    drop_judge(r, "b")
    assert "judge_agreement" not in r

def test_record_judge_error_does_not_touch_the_mirror():
    """The bug that showed one judge's score beside another judge's error."""
    r = {"judgments": [_j(3, "good")]}
    apply_consensus(r)
    record_judge_error(r, "bad-judge", "unparseable judge response: thought...")
    assert r["score"] == 3
    assert "unparseable" not in r["judge_reason"]
    assert r["judge_errors"][0]["judge_model"] == "bad-judge"

def test_repair_mirror_restores_a_clobbered_reason():
    r = {"judgments": [_j(3, "orig", "the real reason")]}
    apply_consensus(r)
    r["judge_reason"] = "clobbered by a failed judge"
    data = {"results": [r]}
    assert repair_mirror(data) == 1
    assert r["judge_reason"] == "the real reason"

def test_repair_mirror_is_noop_when_consistent():
    r = {"judgments": [_j(2, "a")]}
    apply_consensus(r)
    assert repair_mirror({"results": [r]}) == 0


# ── eval-set identification ──

SET_A = {"name": "a", "prompts": [
    {"id": 1, "title": "alpha", "rubric": {"3": "x"}},
    {"id": 2, "title": "beta", "rubric": {"3": "x"}},
]}
SET_B = {"name": "b", "prompts": [
    {"id": 1, "title": "gamma", "rubric": {"3": "y"}},
    {"id": 2, "title": "delta", "rubric": {"3": "y"}},
]}
RUN_A = {"results": [{"id": 1, "title": "alpha"}, {"id": 2, "title": "beta"}]}

def test_match_eval_set_perfect():
    assert match_eval_set(RUN_A, SET_A) == 1.0

def test_match_eval_set_ignores_id_only_collisions():
    """Both sets number from 1 — ids alone would match either."""
    assert match_eval_set(RUN_A, SET_B) == 0.0

def test_match_eval_set_empty_run():
    assert match_eval_set({"results": []}, SET_A) == 0.0

def test_infer_picks_the_matching_set():
    path, es, score = infer_eval_set(RUN_A, [("b.json", SET_B), ("a.json", SET_A)])
    assert path == "a.json" and score == 1.0

def test_infer_refuses_below_threshold():
    path, es, score = infer_eval_set(RUN_A, [("b.json", SET_B)])
    assert path is None and es is None and score == 0.0

def test_infer_partial_match_below_threshold_refuses():
    half = {"name": "h", "prompts": [{"id": 1, "title": "alpha"}]}
    path, _, score = infer_eval_set(RUN_A, [("h.json", half)])
    assert path is None and score == 0.5

"""Integration tests for re-judging.

The behaviour these lock down is the point of the whole change: scoring an
existing run with a second judge must keep the first judgment, and every existing
reader must keep working against the mirrored flat fields.
"""

import json
from pathlib import Path

import pytest

from cupel import eval as ev
from cupel.schema import normalize_run, judge_models


@pytest.fixture
def legacy_run(tmp_path):
    """A run file in the pre-judgments shape, as all 55 stored runs are."""
    data = {
        "model": "test-model",
        "api_url": "http://localhost:8000/v1/chat/completions",
        "timestamp": "20260101_120000",
        "judge": "original-judge",
        "judge_url": "https://api.anthropic.com/v1/messages",
        "results": [
            {"id": 1, "title": "A", "category": "c", "response": "answer one",
             "score": 3, "judge_reason": "great", "judge_model": "original-judge"},
            {"id": 2, "title": "B", "category": "c", "response": "answer two",
             "score": 1, "judge_reason": "weak", "judge_model": "original-judge"},
        ],
    }
    p = tmp_path / "eval_test-model_20260101_120000.json"
    p.write_text(json.dumps(data, indent=2))
    return p


def _fake_judge(scores):
    """Stand in for score_one so no model is called."""
    def _score(api_url, api_key, judge_model, prompt_text, rubric, response_text,
               responses=None):
        return scores[judge_model], f"{judge_model} says so", None
    return _score


RUBRICS = {1: {"3": "good", "0": "bad"}, 2: {"3": "good", "0": "bad"}}
PROMPTS = {1: "prompt one", 2: "prompt two"}


def test_rejudging_preserves_the_original_judgment(legacy_run, monkeypatch):
    monkeypatch.setattr(ev, "score_one", _fake_judge({"second-judge": 1}))

    data = normalize_run(json.loads(legacy_run.read_text()))
    ev.judge_results([(str(legacy_run), data)], "second-judge", "u", "k",
                     RUBRICS, PROMPTS)

    saved = json.loads(legacy_run.read_text())
    r1 = saved["results"][0]
    assert len(r1["judgments"]) == 2
    assert r1["judgments"][0]["judge_model"] == "original-judge"
    assert r1["judgments"][0]["score"] == 3     # original intact
    assert r1["judgments"][1]["judge_model"] == "second-judge"
    assert r1["judgments"][1]["score"] == 1


def test_mirror_reflects_consensus_after_rejudging(legacy_run, monkeypatch):
    monkeypatch.setattr(ev, "score_one", _fake_judge({"second-judge": 1}))

    data = normalize_run(json.loads(legacy_run.read_text()))
    ev.judge_results([(str(legacy_run), data)], "second-judge", "u", "k",
                     RUBRICS, PROMPTS)

    r1 = json.loads(legacy_run.read_text())["results"][0]
    assert r1["score"] == 1                 # lower median of [3, 1]
    assert r1["judge_agreement"] == 2
    assert r1["judge_consensus"] == "median"


def test_run_level_judge_list_tracks_both(legacy_run, monkeypatch):
    monkeypatch.setattr(ev, "score_one", _fake_judge({"second-judge": 2}))

    data = normalize_run(json.loads(legacy_run.read_text()))
    ev.judge_results([(str(legacy_run), data)], "second-judge", "u", "k",
                     RUBRICS, PROMPTS)

    saved = json.loads(legacy_run.read_text())
    assert sorted(judge_models(saved)) == ["original-judge", "second-judge"]
    assert saved["judge"] == "second-judge"   # legacy field points at the newest


def test_replace_discards_prior_judgment(legacy_run, monkeypatch):
    monkeypatch.setattr(ev, "score_one", _fake_judge({"second-judge": 0}))

    data = normalize_run(json.loads(legacy_run.read_text()))
    ev.judge_results([(str(legacy_run), data)], "second-judge", "u", "k",
                     RUBRICS, PROMPTS, replace=True)

    r1 = json.loads(legacy_run.read_text())["results"][0]
    assert len(r1["judgments"]) == 1
    assert r1["score"] == 0


def test_three_judges_accumulate(legacy_run, monkeypatch):
    for judge, score in [("j2", 2), ("j3", 0)]:
        monkeypatch.setattr(ev, "score_one", _fake_judge({judge: score}))
        data = normalize_run(json.loads(legacy_run.read_text()))
        ev.judge_results([(str(legacy_run), data)], judge, "u", "k", RUBRICS, PROMPTS)

    r1 = json.loads(legacy_run.read_text())["results"][0]
    assert [j["score"] for j in r1["judgments"]] == [3, 2, 0]
    assert r1["score"] == 2                  # median of [3, 2, 0]
    assert r1["judge_agreement"] == 3


def test_scoring_version_recorded_on_new_judgments(legacy_run, monkeypatch):
    monkeypatch.setattr(ev, "score_one", _fake_judge({"j2": 2}))

    data = normalize_run(json.loads(legacy_run.read_text()))
    ev.judge_results([(str(legacy_run), data)], "j2", "u", "k", RUBRICS, PROMPTS)

    js = json.loads(legacy_run.read_text())["results"][0]["judgments"]
    assert js[0]["scoring_version"] == "legacy"
    assert js[1]["scoring_version"] != "legacy" and len(js[1]["scoring_version"]) == 12


def test_different_rubric_yields_different_scoring_version(legacy_run, monkeypatch):
    monkeypatch.setattr(ev, "score_one", _fake_judge({"j2": 2}))
    data = normalize_run(json.loads(legacy_run.read_text()))
    ev.judge_results([(str(legacy_run), data)], "j2", "u", "k", RUBRICS, PROMPTS)
    v_a = json.loads(legacy_run.read_text())["results"][0]["judgments"][1]["scoring_version"]

    monkeypatch.setattr(ev, "score_one", _fake_judge({"j3": 2}))
    other = {1: {"3": "DIFFERENT", "0": "bad"}, 2: RUBRICS[2]}
    data = normalize_run(json.loads(legacy_run.read_text()))
    ev.judge_results([(str(legacy_run), data)], "j3", "u", "k", other, PROMPTS)
    v_b = json.loads(legacy_run.read_text())["results"][0]["judgments"][-1]["scoring_version"]

    assert v_a != v_b


def test_unscored_results_are_left_alone(tmp_path, monkeypatch):
    """A prompt with no response must not acquire a judgment."""
    data = {
        "model": "m", "timestamp": "t", "judge": "j",
        "results": [{"id": 1, "title": "A", "error": "timeout", "score": None}],
    }
    p = tmp_path / "eval_m_t.json"
    p.write_text(json.dumps(data))

    monkeypatch.setattr(ev, "score_one", _fake_judge({"j2": 3}))
    ev.judge_results([(str(p), normalize_run(json.loads(p.read_text())))],
                     "j2", "u", "k", RUBRICS, PROMPTS)

    r = json.loads(p.read_text())["results"][0]
    assert r["judgments"] == []
    assert r["score"] is None


def test_criteria_rubric_records_check_score(tmp_path, monkeypatch):
    data = {"model": "m", "timestamp": "t",
            "results": [{"id": 1, "title": "A", "response": "x", "score": None}]}
    p = tmp_path / "eval_m_t.json"
    p.write_text(json.dumps(data))

    rubric = {"criteria": [
        {"id": "P1", "type": "positive", "weight": 3},
        {"id": "P2", "type": "positive", "weight": 1},
    ]}
    crit = [{"id": "P1", "met": True}, {"id": "P2", "met": False}]

    def _score(*a, **kw):
        return 2, "partial", crit
    monkeypatch.setattr(ev, "score_one", _score)

    ev.judge_results([(str(p), normalize_run(json.loads(p.read_text())))],
                     "j", "u", "k", {1: rubric}, {1: "prompt"})

    r = json.loads(p.read_text())["results"][0]
    assert r["check_score"] == 0.75          # 3 of 4 positive weight
    assert r["judgments"][0]["check_score"] == 0.75


# ── rubrics_for_run: judge against the eval set the run actually used ──

def _eval_set_file(tmp_path, name, prompts):
    from cupel.schema import eval_set_meta
    p = tmp_path / name
    data = {"name": name, "prompts": prompts}
    p.write_text(json.dumps(data))
    return p, eval_set_meta(p, data)


def test_rubrics_come_from_the_runs_own_eval_set(tmp_path):
    """The recorded eval set wins over anything configured elsewhere."""
    used, meta = _eval_set_file(tmp_path, "used.json", [
        {"id": 1, "title": "correct prompt", "prompt": "P", "rubric": {"3": "right"}},
    ])

    r_by_id, _, prompts, warning = ev.rubrics_for_run({"eval_set": meta, "results": []})

    assert r_by_id[1] == {"3": "right"}
    assert prompts[0]["title"] == "correct prompt"
    assert warning is None


def test_forced_eval_set_overrides_everything(tmp_path):
    """--eval-set is an explicit override, not a fallback."""
    used, meta = _eval_set_file(tmp_path, "used.json", [
        {"id": 1, "title": "recorded", "prompt": "P", "rubric": {"3": "recorded"}},
    ])
    forced = {"name": "forced", "prompts": [
        {"id": 1, "title": "forced", "prompt": "Q", "rubric": {"3": "forced"}},
    ]}

    r_by_id, _, _, warning = ev.rubrics_for_run({"eval_set": meta, "results": []}, forced)
    assert r_by_id[1] == {"3": "forced"}
    assert warning is None


def test_rubrics_warn_when_eval_set_changed_since_the_run(tmp_path):
    used, meta = _eval_set_file(tmp_path, "used.json", [
        {"id": 1, "title": "t", "prompt": "P", "rubric": {"3": "right"}},
    ])
    # edit the set on disk after the run recorded its hash
    used.write_text(json.dumps({"name": "used.json", "prompts": [
        {"id": 1, "title": "t", "prompt": "P", "rubric": {"3": "EDITED"}},
    ]}))

    _, _, _, warning = ev.rubrics_for_run({"eval_set": meta, "results": []}, None)
    assert warning and "has changed" in warning


def test_rubrics_warn_when_eval_set_is_gone(tmp_path):
    used, meta = _eval_set_file(tmp_path, "gone.json", [{"id": 1, "rubric": {}}])
    used.unlink()
    _, _, _, warning = ev.rubrics_for_run({"eval_set": meta, "results": []})
    assert warning and "no longer exists" in warning


def test_legacy_run_is_identified_by_its_prompt_titles(tmp_path, monkeypatch):
    """A run with no recorded eval set is identified, not guessed at."""
    prompts = [{"id": i, "title": f"prompt {i}", "prompt": "P",
                "rubric": {"3": f"r{i}"}} for i in range(1, 6)]
    _, _ = _eval_set_file(tmp_path, "real.json", prompts)
    decoy = tmp_path / "decoy.json"
    decoy.write_text(json.dumps({"name": "decoy", "prompts": [
        {"id": i, "title": f"UNRELATED {i}", "rubric": {"3": "wrong"}} for i in range(1, 6)]}))

    candidates = [(p, json.loads(p.read_text())) for p in sorted(tmp_path.glob("*.json"))]
    monkeypatch.setattr(ev, "_available_eval_sets", lambda: candidates)

    run = {"results": [{"id": p["id"], "title": p["title"]} for p in prompts]}
    r_by_id, _, _, warning = ev.rubrics_for_run(run)

    assert r_by_id[1] == {"3": "r1"}          # the real set, not the decoy
    assert "identified as real.json" in warning


def test_legacy_run_refuses_when_nothing_matches(monkeypatch):
    """Refusing beats guessing — a wrong eval set silently rewrites every score."""
    monkeypatch.setattr(ev, "_available_eval_sets", lambda: [
        (Path("other.json"), {"name": "other",
                              "prompts": [{"id": 1, "title": "different", "rubric": {"3": "x"}}]}),
    ])
    run = {"results": [{"id": 1, "title": "mine"}]}
    r_by_id, _, prompts, warning = ev.rubrics_for_run(run)

    assert r_by_id == {} and prompts == []
    assert "cannot identify which eval set" in warning
    assert "--eval-set" in warning


# ── judge_one: the single judging decision, shared by CLI/server/library ──

def _res(**kw):
    r = {"id": 1, "title": "t", "response": "an answer"}
    r.update(kw)
    return r


def test_judge_one_skips_missing_rubric_without_calling_the_judge(monkeypatch):
    """The bug that scored 8 prompts as 0 against an eval set that lacked them."""
    called = []
    monkeypatch.setattr(ev, "score_one", lambda *a, **k: called.append(1) or (0, "x", None))
    r = _res()
    status, score = ev.judge_one(r, None, "p", "u", "k", "j")
    assert status == "skip" and score is None
    assert not called                       # the judge was never asked
    assert "no rubric" in r["judge_errors"][0]["error"]


def test_judge_one_skips_empty_criteria_list(monkeypatch):
    """A criteria rubric with no criteria is a non-empty dict — truthiness misses it."""
    monkeypatch.setattr(ev, "score_one", lambda *a, **k: (0, "all unmet", []))
    status, _ = ev.judge_one(_res(), {"criteria": []}, "p", "u", "k", "j")
    assert status == "skip"


def test_judge_one_skips_blank_level_rubric(monkeypatch):
    monkeypatch.setattr(ev, "score_one", lambda *a, **k: (0, "x", None))
    status, _ = ev.judge_one(_res(), {"3": "", "0": ""}, "p", "u", "k", "j")
    assert status == "skip"


def test_judge_one_scores_with_a_real_rubric(monkeypatch):
    monkeypatch.setattr(ev, "score_one", lambda *a, **k: (2, "ok", None))
    r = _res()
    status, score = ev.judge_one(r, {"3": "good", "0": "bad"}, "p", "u", "k", "j")
    assert status == "scored" and score == 2
    assert r["judgments"][0]["score"] == 2


def test_judge_one_skips_results_with_no_response(monkeypatch):
    monkeypatch.setattr(ev, "score_one", lambda *a, **k: (3, "x", None))
    status, _ = ev.judge_one(_res(response="", error="timeout"), {"3": "g"}, "p", "u", "k", "j")
    assert status == "skip"


def test_judge_one_failure_leaves_the_mirror_alone(monkeypatch):
    """A failed judgment must not overwrite a standing judge's score or reason."""
    monkeypatch.setattr(ev, "score_one", lambda *a, **k: (None, "unparseable ...", None))
    r = _res()
    ev.apply_judgment(r, 3, "the good reason", None, "good-judge", {"3": "g"})
    status, _ = ev.judge_one(r, {"3": "g"}, "p", "u", "k", "bad-judge")
    assert status == "error"
    assert r["score"] == 3
    assert r["judge_reason"] == "the good reason"
    assert r["judge_errors"][-1]["judge_model"] == "bad-judge"


def test_judge_one_exception_is_recorded_not_raised(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("connection refused")
    monkeypatch.setattr(ev, "score_one", boom)
    r = _res()
    status, _ = ev.judge_one(r, {"3": "g"}, "p", "u", "k", "j")
    assert status == "error"
    assert "connection refused" in r["judge_errors"][-1]["error"]


# ── rubric_is_empty ──

@pytest.mark.parametrize("rubric,expected", [
    (None, True), ({}, True), ({"criteria": []}, True), ({"3": "", "0": ""}, True),
    ({"3": "good", "0": "bad"}, False),
    ({"criteria": [{"id": "P1", "type": "positive"}]}, False),
])
def test_rubric_is_empty(rubric, expected):
    assert ev.rubric_is_empty(rubric) is expected

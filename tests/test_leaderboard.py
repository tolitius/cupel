"""Tests for leaderboard grouping, intervals, and the coverage floor.

These cover the arithmetic that decides what the dashboard shows. The behaviour
they lock down is that repeat runs are samples, not competitors, and that a
difference smaller than its own interval is not reported as a ranking.
"""

import pytest

from cupel.server import _group_entries, _coverage, MIN_COVERAGE
from cupel.stats import measured_noise_floor, rank_within_noise


def _entry(key, pct, scores, filename="f.json", ts="2026-01-01", **kw):
    e = {
        "_group_key": key, "_scores": scores, "model": key[0] if isinstance(key, tuple) else key,
        "pct": pct, "total_score": sum(scores), "max_score": len(scores) * 3,
        "filename": filename, "timestamp": ts,
        "scores_by_prompt": [{"id": i + 1, "score": s, "category": "c", "title": f"p{i+1}",
                              "elapsed": 1, "tokens": 10} for i, s in enumerate(scores)],
    }
    e.update(kw)
    return e


# ── grouping ──

def test_repeat_runs_collapse_into_one_row():
    a = _entry(("m", "es", "j", "v", None, None, ""), 100.0, [3, 3], "a.json", "2026-01-01")
    b = _entry(("m", "es", "j", "v", None, None, ""), 50.0, [1, 2], "b.json", "2026-01-02")
    out = _group_entries([a, b])
    assert len(out) == 1
    assert out[0]["n_runs"] == 2

def test_different_configs_stay_separate():
    a = _entry(("m", "es", "j", "v", None, None, ""), 100.0, [3])
    b = _entry(("m", "es", "j", "v", 4096, None, ""), 100.0, [3])   # different thinking budget
    assert len(_group_entries([a, b])) == 2

def test_grouped_pct_pools_all_scores():
    a = _entry(("m", "e", "j", "v", None, None, ""), 100.0, [3, 3], "a.json")
    b = _entry(("m", "e", "j", "v", None, None, ""), 0.0, [0, 0], "b.json")
    g = _group_entries([a, b])[0]
    assert g["pct"] == 50.0
    assert g["total_score"] == 6 and g["max_score"] == 12

def test_grouped_row_reports_spread_between_runs():
    a = _entry(("m", "e", "j", "v", None, None, ""), 100.0, [3, 3], "a.json")
    b = _entry(("m", "e", "j", "v", None, None, ""), 0.0, [0, 0], "b.json")
    assert _group_entries([a, b])[0]["spread"] == 100.0

def test_single_run_has_no_spread_and_n_of_one():
    g = _group_entries([_entry(("m", "e", "j", "v", None, None, ""), 100.0, [3])])[0]
    assert g["n_runs"] == 1 and g["spread"] == 0.0

def test_grouped_row_lists_its_runs_for_expansion():
    a = _entry(("m", "e", "j", "v", None, None, ""), 100.0, [3], "a.json", "2026-01-01")
    b = _entry(("m", "e", "j", "v", None, None, ""), 66.7, [2], "b.json", "2026-01-02")
    runs = _group_entries([a, b])[0]["runs"]
    assert {r["filename"] for r in runs} == {"a.json", "b.json"}

def test_grouped_row_keeps_the_latest_runs_metadata():
    a = _entry(("m", "e", "j", "v", None, None, ""), 100.0, [3], "old.json", "2026-01-01")
    b = _entry(("m", "e", "j", "v", None, None, ""), 100.0, [3], "new.json", "2026-06-01")
    assert _group_entries([a, b])[0]["filename"] == "new.json"

def test_per_prompt_scores_are_averaged_across_runs():
    a = _entry(("m", "e", "j", "v", None, None, ""), 100.0, [3, 3], "a.json")
    b = _entry(("m", "e", "j", "v", None, None, ""), 33.3, [1, 1], "b.json")
    sbp = _group_entries([a, b])[0]["scores_by_prompt"]
    assert [sp["score"] for sp in sbp] == [2.0, 2.0]

def test_grouping_drops_internal_keys():
    g = _group_entries([_entry(("m", "e", "j", "v", None, None, ""), 100.0, [3])])[0]
    assert "_group_key" not in g and "_scores" not in g

def test_group_with_no_scores_is_dropped():
    assert _group_entries([_entry(("m", "e", "j", "v", None, None, ""), 0.0, [])]) == []


# ── intervals attached to grouped rows ──

def test_grouped_row_carries_an_interval():
    a = _entry(("m", "e", "j", "v", None, None, ""), 66.7, [0, 1, 2, 3, 2, 1], "a.json")
    g = _group_entries([a])[0]
    assert g["ci_lo"] <= g["pct"] <= g["ci_hi"]

def test_more_runs_narrow_the_interval():
    key = ("m", "e", "j", "v", None, None, "")
    one = _group_entries([_entry(key, 50.0, [0, 3] * 4, "a.json")])[0]
    many = _group_entries([_entry(key, 50.0, [0, 3] * 4, f"{i}.json") for i in range(8)])[0]
    assert (many["ci_hi"] - many["ci_lo"]) < (one["ci_hi"] - one["ci_lo"])


# ── coverage floor ──

def test_coverage_counts_scored_fraction():
    assert _coverage([{"score": 3}, {"score": None}]) == 0.5

def test_coverage_of_empty_run():
    assert _coverage([]) == 0.0

def test_full_coverage():
    assert _coverage([{"score": 1}, {"score": 2}]) == 1.0

def test_min_coverage_is_a_real_threshold():
    assert 0 < MIN_COVERAGE <= 1


# ── ranking against measured run-to-run noise ──

def test_noise_floor_is_the_widest_observed_repeat_gap():
    n = measured_noise_floor([[92.8, 87.0], [91.3, 91.3]])
    assert n["floor"] == 5.8          # the widest gap actually seen
    assert n["n_pairs"] == 2 and n["n_configs"] == 2


def test_noise_floor_ignores_configs_run_only_once():
    n = measured_noise_floor([[90.0], [80.0, 78.0]])
    assert n["n_configs"] == 1 and n["floor"] == 2.0


def test_no_repeat_runs_means_no_floor():
    """Better to make no claim than to invent a band from an assumption."""
    assert measured_noise_floor([[90.0], [80.0]]) is None


def test_three_runs_yield_three_pairs():
    n = measured_noise_floor([[88.4, 88.4, 89.9]])
    assert n["n_pairs"] == 3


def test_gaps_below_the_floor_share_a_rank():
    e = [{"pct": 92.8}, {"pct": 91.3}]
    rank_within_noise(e, 5.8)
    assert e[0]["rank"] == e[1]["rank"] == 1
    assert e[1]["tied"] is True


def test_gaps_above_the_floor_rank_distinctly():
    e = [{"pct": 92.8}, {"pct": 60.0}]
    rank_within_noise(e, 5.8)
    assert (e[0]["rank"], e[1]["rank"]) == (1, 2)
    assert e[1]["tied"] is False


def test_without_a_floor_nothing_is_marked_tied():
    e = [{"pct": 92.8}, {"pct": 92.7}]
    rank_within_noise(e, None)
    assert (e[0]["rank"], e[1]["rank"]) == (1, 2)
    assert not any(x["tied"] for x in e)


def test_ranking_bands_are_measured_from_the_band_leader():
    """A long chain of small steps must not collapse into one rank."""
    e = [{"pct": 100.0}, {"pct": 96.0}, {"pct": 92.0}, {"pct": 88.0}]
    rank_within_noise(e, 5.0)
    # 96 and 92 are within 5 of 100 and 96 respectively, but 92 is 8 below the
    # band leader (100), so it starts a new band
    assert e[0]["rank"] == e[1]["rank"] == 1
    assert e[2]["rank"] == 3


# ── criteria-vector ranking ──

def _check_entry(model, scores, checks, ts="2026-01-01"):
    e = _entry(("m" + model, "es", "j", "v", None, None, ""), 0.0, scores,
               f"{model}.json", ts)
    e["model"] = model
    e["_check_scores"] = checks
    return e


def test_criteria_scores_pool_into_the_group():
    e = _check_entry("A", [3, 3], [1.0, 0.5])
    g = _group_entries([e])[0]
    assert g["has_check"] is True
    assert g["check_pct"] == 75.0


def test_partial_criteria_coverage_is_not_usable():
    """A run where only some prompts have a vector cannot be compared on it."""
    e = _check_entry("A", [3, 3], [1.0])          # 2 scores, 1 check
    assert _group_entries([e])[0]["has_check"] is False


def test_no_criteria_scores_means_no_check_metric():
    e = _check_entry("A", [3, 3], [])
    g = _group_entries([e])[0]
    assert g["has_check"] is False and g["check_pct"] is None


def test_criteria_vector_separates_what_the_collapse_merges():
    """The reason this metric exists — verified on the shape of real data."""
    a = _check_entry("A", [2, 2], [0.9, 0.9])
    b = _check_entry("B", [2, 2], [0.4, 0.4])
    ga, gb = _group_entries([a])[0], _group_entries([b])[0]
    assert ga["pct"] == gb["pct"]                  # identical on the 0-3 collapse
    assert ga["check_pct"] != gb["check_pct"]      # distinguishable on the vector


def test_criteria_interval_is_computed_on_the_vector():
    g = _group_entries([_check_entry("A", [3] * 6, [1.0] * 6)])[0]
    assert g["check_ci_lo"] == g["check_ci_hi"] == 100.0


# ── caching ──

def test_leaderboard_cache_is_keyed_on_file_mtimes(tmp_path, monkeypatch):
    """A changed result file must invalidate the cached response."""
    import cupel.server as srv
    monkeypatch.setattr(srv, "RESULTS_DIR", tmp_path)
    f = tmp_path / "eval_m_1.json"
    f.write_text("{}")
    sig1 = srv._results_signature()
    assert sig1 == srv._results_signature()          # stable while untouched

    import os, time
    time.sleep(0.01)
    os.utime(f, None)
    assert srv._results_signature() != sig1          # invalidated by a write


def test_results_signature_survives_a_missing_dir(tmp_path, monkeypatch):
    import cupel.server as srv
    monkeypatch.setattr(srv, "RESULTS_DIR", tmp_path / "gone")
    assert srv._results_signature() == ()


def test_hardware_is_detected_once(monkeypatch):
    import cupel.server as srv
    srv._hw_cache.clear()
    calls = []
    monkeypatch.setattr(srv, "detect_hardware", lambda: calls.append(1) or {"name": "x"})
    srv._cached_hardware(); srv._cached_hardware(); srv._cached_hardware()
    assert len(calls) == 1
    srv._hw_cache.clear()

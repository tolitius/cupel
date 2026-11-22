"""Tests for the public leaderboard exporter.

The snapshot is committed to a public repo, so the leak tests come first: every
piece of eval content is planted as a unique canary and must not appear anywhere
in the output.
"""

import asyncio
import builtins
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from cupel import publish
from cupel.publish import PublishError, build_snapshot, validate_snapshot

NOW = datetime(2026, 11, 22, 18, 4, 11, tzinfo=timezone.utc)
HASH = "sha256:0123456789abcdef"
LOCAL_HW = {"name": "Apple M4 Max", "memory": "128 GB", "spec": "CANARY_SPEC_detected_hw"}
VIEW = {"text": False, "metric": "score", "chart_tab": None, "sort_col": "score", "sort_dir": "desc"}
FROZEN = {"local": False, "notes": False}


def _results(scores, cats=None, check=None, canary=""):
    cats = cats or ["code"] * len(scores)
    out = []
    for i, s in enumerate(scores):
        r = {
            "id": i + 1, "category": cats[i], "score": s,
            "title": f"CANARY_TITLE_{canary}{i}",
            "prompt": f"CANARY_PROMPT_{canary}_{i}_full_prompt_text",
            "response": f"CANARY_RESPONSE_{canary}_{i}_model_answer",
            "thinking": f"CANARY_THINKING_{canary}_{i}_chain_of_thought",
            "judge_reason": f"CANARY_REASON_{canary}_{i}_judge_explanation",
            "notes": f"CANARY_RNOTES_{canary}_{i}_per_result_note",
            "criteria_results": [{"criterion": f"CANARY_CRIT_{canary}_{i}_xxxxxxxx", "met": True}],
            "elapsed_seconds": 10.0, "completion_tokens": 100,
        }
        if s is None:
            r["error"] = f"CANARY_ERROR_{canary}_{i}_provider_failure"
        if check is not None:
            r["check_score"] = check[i]
        out.append(r)
    return out


def _run(model, scores, *, cats=None, check=None, ts="2026-11-01T10:00:00", hw=LOCAL_HW,
         api_url="http://localhost:8000/v1", eval_hash=HASH, notes="", judge="judge-x",
         thinking_budget=None, canary=None):
    data = {
        "model": model, "api_url": api_url, "judge": judge,
        "judge_url": "http://10.1.2.3:9000/CANARY_JUDGE_URL",
        "timestamp": ts, "thinking_budget": thinking_budget, "temperature": 0,
        "notes": notes,
        "results": _results(scores, cats, check, canary if canary is not None else model),
    }
    if eval_hash is not None:
        data["eval_set"] = {"name": "CANARY_ESNAME_set", "path": "/Users/someone/private/CANARY_ESPATH.json",
                            "hash": eval_hash, "n_prompts": len(scores)}
    if hw is not None:
        data["hardware"] = hw
    return data


def _write(tmp_path, runs):
    paths = []
    for i, data in enumerate(runs):
        p = tmp_path / f"eval_{i:03d}.json"
        p.write_text(json.dumps(data))
        paths.append(p)
    return paths


def _build(paths, *, hidden=(), view=VIEW, frozen=FROZEN, eval_sets=None, local_hw=LOCAL_HW):
    return build_snapshot(paths, set(hidden), dict(view), dict(frozen), "tester", "0.1.94",
                          NOW, eval_sets or [], local_hw)


def _keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _keys(v)


# ──────────────────────────────────────────────
# leak tests
# ──────────────────────────────────────────────

@pytest.fixture
def canary_paths(tmp_path):
    runs = [
        _run("model-a", [3, 2, 1, None], cats=["code", "code", "reasoning", "multimodal"],
             notes="CANARY_RUNNOTES_model_a"),
        _run("model-b", [2, 2, 2, 2], cats=["code", "code", "reasoning", "multimodal"],
             api_url="http://192.168.1.20:8000/v1", hw=None),
        _run("model-c", [1, 1, 3, 3], cats=["code", "code", "reasoning", "multimodal"],
             api_url="https://openrouter.ai/api/v1", hw=None),
    ]
    return _write(tmp_path, runs)


def test_no_canary_appears_anywhere(canary_paths):
    snap, _ = _build(canary_paths)
    blob = json.dumps(snap, ensure_ascii=False)
    assert "CANARY" not in blob


def test_run_notes_absent_when_notes_frozen_off(canary_paths):
    snap, _ = _build(canary_paths, frozen={"local": False, "notes": False})
    assert "CANARY_RUNNOTES" not in json.dumps(snap)
    assert all("notes" not in e for e in snap["entries"])


def test_run_notes_present_only_where_allowed_when_on(canary_paths):
    snap, _ = _build(canary_paths, frozen={"local": False, "notes": True})
    blob = json.dumps(snap)
    assert blob.count("CANARY_RUNNOTES_model_a") == 1
    a = next(e for e in snap["entries"] if e["model"] == "model-a")
    assert a["notes"] == "CANARY_RUNNOTES_model_a"
    # nothing else leaks with notes on
    assert "CANARY" not in blob.replace("CANARY_RUNNOTES_model_a", "")


def test_no_forbidden_keys(canary_paths):
    snap, _ = _build(canary_paths, frozen={"local": False, "notes": True})
    keys = set(_keys(snap))
    assert not keys & publish.FORBIDDEN_KEYS
    assert not [k for k in keys if k.startswith("_")]


def test_lan_cloud_run_is_remote_and_openrouter_is_named(canary_paths):
    snap, _ = _build(canary_paths)
    by_model = {e["model"]: e for e in snap["entries"]}
    assert by_model["model-b"]["hardware"] == {"name": "remote", "memory": ""}
    assert by_model["model-c"]["hardware"] == {"name": "OpenRouter", "memory": ""}
    assert by_model["model-a"]["hardware"] == {"name": "Apple M4 Max", "memory": "128 GB"}
    assert not by_model["model-b"]["is_local"] and by_model["model-a"]["is_local"]


def test_home_directory_never_appears(canary_paths):
    snap, _ = _build(canary_paths, frozen={"local": False, "notes": True})
    assert str(Path.home()) not in json.dumps(snap)


def test_builder_output_passes_validation(canary_paths):
    snap, _ = _build(canary_paths, frozen={"local": False, "notes": True})
    assert validate_snapshot(snap, canary_paths, Path.home()) == []


@pytest.mark.parametrize("key", sorted(publish.FORBIDDEN_KEYS) + ["_check_scores"])
def test_validator_catches_each_forbidden_key(canary_paths, key):
    snap, _ = _build(canary_paths)
    snap["entries"][0]["per_prompt"][0][key] = "x"
    errs = validate_snapshot(snap, canary_paths, Path.home())
    assert any(repr(key) in e for e in errs)


@pytest.mark.parametrize("canary", [
    "CANARY_TITLE_model-a0", "CANARY_PROMPT_model-a_0_full_prompt_text",
    "CANARY_RESPONSE_model-a_1_model_answer", "CANARY_THINKING_model-a_2_chain_of_thought",
    "CANARY_REASON_model-a_0_judge_explanation", "CANARY_RNOTES_model-a_1_per_result_note",
    "CANARY_ESNAME_set", "/Users/someone/private/CANARY_ESPATH.json",
])
def test_validator_catches_each_canary_value(canary_paths, canary):
    snap, _ = _build(canary_paths)
    snap["entries"][0]["model"] = f"prefix {canary} suffix"
    assert validate_snapshot(snap, canary_paths, Path.home()) != []


def test_validator_catches_home_dir_and_private_addresses(canary_paths, tmp_path):
    snap, _ = _build(canary_paths)
    snap["entries"][0]["model"] = str(tmp_path / "models" / "x")
    assert any("home directory" in e for e in validate_snapshot(snap, canary_paths, tmp_path))
    for addr in ("localhost", "127.0.0.1", "10.0.0.5", "172.20.1.1", "192.168.0.9"):
        snap["entries"][0]["model"] = f"m@{addr}"
        assert any("private network" in e for e in validate_snapshot(snap, canary_paths, Path.home()))


def test_validator_rejects_unknown_fields_and_raw_hostnames(canary_paths):
    snap, _ = _build(canary_paths)
    snap["entries"][0]["extra"] = 1
    assert any("unexpected keys" in e for e in validate_snapshot(snap, canary_paths, Path.home()))
    snap, _ = _build(canary_paths)
    snap["entries"][1]["hardware"] = {"name": "gpu-box.corp", "memory": ""}
    assert any("known provider" in e for e in validate_snapshot(snap, canary_paths, Path.home()))
    snap, _ = _build(canary_paths)
    snap["entries"][0]["hardware"]["spec"] = "x"
    assert validate_snapshot(snap, canary_paths, Path.home()) != []


def test_validator_catches_eval_set_names_and_prompt_text_from_eval_set_files(canary_paths, tmp_path):
    es_path = tmp_path / "secret-bench.json"
    es = {"name": "My Secret Bench", "prompts": [
        {"id": 1, "title": "Rate limiter", "category": "code", "prompt": "Design a rate limiter for ..."}]}
    snap, _ = _build(canary_paths)
    for leak in ("secret-bench", "My Secret Bench", "Rate limiter", "Design a rate limiter for ..."):
        snap["entries"][0]["model"] = leak
        assert validate_snapshot(snap, canary_paths, Path.home(), [(es_path, es)]) != [], leak


def test_sample_file_is_never_opened(tmp_path, monkeypatch):
    sample = Path(publish.__file__).parent / "data" / "example-run.json"
    paths = _write(tmp_path, [_run("m", [3, 2])]) + [sample]
    real_open = builtins.open

    def guarded(file, *a, **kw):
        assert Path(file).name != "example-run.json", "exporter opened the sample file"
        return real_open(file, *a, **kw)

    monkeypatch.setattr(builtins, "open", guarded)
    snap, _ = _build(paths)
    assert validate_snapshot(snap, paths, Path.home()) == []
    assert [e["model"] for e in snap["entries"]] == ["m"]


# ──────────────────────────────────────────────
# correctness
# ──────────────────────────────────────────────

def test_all_score_matches_the_leaderboard_endpoint(tmp_path, monkeypatch):
    import cupel.server as srv
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    runs = [
        _run("alpha", [3, 3, 2, 1, 0, 3], ts="2026-11-01T10:00:00"),
        _run("alpha", [3, 2, 2, 1, 1, 3], ts="2026-11-02T10:00:00"),
        _run("beta", [2, 2, 2, 2, None, 2], ts="2026-11-03T10:00:00"),
        _run("gamma", [1, 0, 3, 3, 2, 1], ts="2026-11-04T10:00:00"),
        _run("muted", [3, 3, 3, 3, 3, 3], ts="2026-11-05T10:00:00"),
    ]
    paths = _write(results_dir, runs)
    hidden = [paths[-1].name]
    (results_dir / ".hidden.json").write_text(json.dumps(hidden))
    monkeypatch.setattr(srv, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(srv, "HIDDEN_FILE", results_dir / ".hidden.json")
    monkeypatch.setattr(srv, "DATA_DIR", tmp_path / "no-samples")
    monkeypatch.setattr(srv, "_cached_hardware", lambda: LOCAL_HW)
    srv._lb_cache.clear()

    board = asyncio.run(srv.get_leaderboard(metric="score"))
    snap, _ = _build(srv._result_files(), hidden=hidden)
    pub = {e["model"]: e for e in snap["entries"]}
    assert set(pub) == {e["model"] for e in board["entries"]} == {"alpha", "beta", "gamma"}
    for e in board["entries"]:
        v = pub[e["model"]]["variants"]["all"]["score"]
        assert (v["pct"], v["ci_lo"], v["ci_hi"]) == (e["pct"], e["ci_lo"], e["ci_hi"])
        assert pub[e["model"]]["n_runs"] == e["n_runs"]
        assert (v["pts"], v["max"]) == (e["total_score"], e["max_score"])
        assert v["rank"] == e["rank"] and v["tied"] == e["tied"]
    assert snap["variants_meta"]["all"]["score"]["noise_floor"] == board["noise"]["floor"]
    srv._lb_cache.clear()


def test_text_variant_excludes_multimodal(tmp_path):
    cats = ["code", "code", "multimodal", "multimodal"]
    paths = _write(tmp_path, [_run("m", [3, 3, 0, 0], cats=cats)])
    snap, _ = _build(paths)
    e = snap["entries"][0]
    a, t = e["variants"]["all"]["score"], e["variants"]["text"]["score"]
    assert (a["pts"], a["max"], a["pct"]) == (6, 12, 50.0)
    assert (t["pts"], t["max"], t["pct"]) == (6, 6, 100.0)
    assert t["ci_lo"] == t["ci_hi"] == 100.0
    assert "multimodal" not in t["by_category"] and a["by_category"]["multimodal"] == 0
    assert snap["eval_set"]["n_text_prompts"] == 2 and snap["eval_set"]["n_prompts"] == 4
    assert snap["variants_meta"]["text"]["score"]["max_pts"] == 6


def test_text_coverage_is_over_text_prompts(tmp_path):
    cats = ["code", "code", "code", "code", "multimodal"]
    paths = _write(tmp_path, [_run("m", [3, 3, 3, 3, None], cats=cats)])
    e = _build(paths)[0]["entries"][0]
    assert e["variants"]["all"]["score"]["coverage"] == 0.8
    assert e["variants"]["text"]["score"]["coverage"] == 1.0


def test_repeat_runs_collapse_and_measure_noise(tmp_path):
    paths = _write(tmp_path, [
        _run("m", [3, 3, 3], ts="2026-11-01"), _run("m", [3, 3, 0], ts="2026-11-02"),
        _run("other", [1, 1, 1]),
    ])
    snap, _ = _build(paths)
    m = next(e for e in snap["entries"] if e["model"] == "m")
    assert m["n_runs"] == 2
    assert m["per_prompt"][2]["score"] == 1.5
    assert snap["variants_meta"]["all"]["score"]["noise_floor"] == pytest.approx(33.3)


def test_no_repeats_rank_one_to_n_without_ties(tmp_path):
    paths = _write(tmp_path, [_run(f"m{i}", [i % 4, 2, 1]) for i in range(4)])
    snap, _ = _build(paths)
    for s in ("all", "text"):
        ranks = sorted(e["variants"][s]["score"]["rank"] for e in snap["entries"])
        assert ranks == [1, 2, 3, 4]
        assert not any(e["variants"][s]["score"]["tied"] for e in snap["entries"])
    assert snap["variants_meta"]["all"]["score"]["noise_floor"] is None


def test_low_coverage_ranks_below_complete(tmp_path):
    paths = _write(tmp_path, [
        _run("partial", [3, None, None, None, None]),
        _run("full", [1, 1, 1, 1, 1]),
    ])
    snap, _ = _build(paths)
    by = {e["model"]: e for e in snap["entries"]}
    for s in ("all", "text"):
        assert by["partial"]["variants"][s]["score"]["low_coverage"]
        assert by["partial"]["variants"][s]["score"]["rank"] > by["full"]["variants"][s]["score"]["rank"]


def test_muted_runs_are_absent(tmp_path):
    paths = _write(tmp_path, [_run("keep", [3]), _run("mute-me", [3])])
    snap, summary = _build(paths, hidden=[paths[1].name])
    assert [e["model"] for e in snap["entries"]] == ["keep"]
    assert "mute-me" not in json.dumps(snap)
    assert summary["dropped"]["muted"] == 1


def test_frozen_local_drops_cloud_runs(canary_paths):
    snap, summary = _build(canary_paths, frozen={"local": True, "notes": False})
    assert [e["model"] for e in snap["entries"]] == ["model-a"]
    assert summary["dropped"]["local"] == 2
    assert snap["frozen"]["local"] is True


def test_dominant_eval_set_is_published_and_others_dropped(tmp_path):
    other = "sha256:fedcba9876543210"
    paths = _write(tmp_path, [_run("a", [3]), _run("b", [2]), _run("c", [1], eval_hash=other)])
    snap, summary = _build(paths)
    assert snap["eval_set"]["hash"] == HASH
    assert {e["model"] for e in snap["entries"]} == {"a", "b"}
    assert summary["dropped"]["other_eval_set"] == 1


def test_legacy_runs_are_identified_from_eval_set_files(tmp_path):
    es_path = tmp_path / "sets" / "my-set.json"
    es_path.parent.mkdir()
    es = {"name": "My Set", "prompts": [
        {"id": i + 1, "title": f"CANARY_TITLE_legacy{i}", "category": c, "prompt": "p" * 20}
        for i, c in enumerate(["code", "code", "multimodal"])]}
    es_path.write_text(json.dumps(es))
    from cupel.schema import eval_set_meta
    es_hash = eval_set_meta(es_path, es)["hash"]
    paths = _write(tmp_path, [
        _run("legacy1", [3, 2, 1], eval_hash=None, canary="legacy"),
        _run("legacy2", [1, 2, 3], eval_hash=None, canary="legacy"),
        _run("hashed", [3, 3, 3], eval_hash=es_hash, canary="legacy"),
        _run("stranger", [3, 3, 3], eval_hash=None, canary="unknown"),
    ])
    snap, summary = _build(paths, eval_sets=[(es_path, es)])
    assert snap["eval_set"] == {"hash": es_hash, "n_prompts": 3, "n_text_prompts": 2}
    assert {e["model"] for e in snap["entries"]} == {"legacy1", "legacy2", "hashed"}
    assert summary["dropped"]["unidentified_eval_set"] == 1
    assert validate_snapshot(snap, paths, Path.home(), [(es_path, es)]) == []


def test_mixed_sets_report_hashes_not_names(tmp_path):
    paths = _write(tmp_path, [_run("x", [3], eval_hash=None, canary="nomatch")])
    with pytest.raises(PublishError) as exc:
        _build(paths)
    assert "CANARY" not in str(exc.value)


def test_notes_equal_to_eval_set_name_are_dropped(tmp_path):
    es_path = tmp_path / "coding-bench-v4.json"
    paths = _write(tmp_path, [
        _run("a", [3], notes="coding-bench-v4"),
        _run("b", [3], notes="CANARY_ESNAME_set"),
        _run("c", [3], notes="  real note about quant  "),
    ])
    snap, _ = _build(paths, frozen={"local": False, "notes": True},
                     eval_sets=[(es_path, {"name": "x", "prompts": [{"id": 99, "title": "zzzz"}]})])
    notes = {e["model"]: e.get("notes") for e in snap["entries"]}
    assert notes == {"a": None, "b": None, "c": "real note about quant"}


def test_check_unavailable_nulls_every_check_and_forces_score(tmp_path):
    paths = _write(tmp_path, [
        _run("has", [3, 2], check=[1.0, 0.5]),
        _run("lacks", [3, 2]),
    ])
    view = dict(VIEW, metric="check")
    snap, _ = _build(paths, view=view)
    assert snap["check_available"] is False
    assert snap["view"]["metric"] == "score"
    assert all(e["variants"][s]["check"] is None for e in snap["entries"] for s in ("all", "text"))
    assert snap["variants_meta"]["all"]["check"] is None


def test_check_unavailable_when_missing_in_text_subset_only(tmp_path):
    cats = ["code", "multimodal"]
    paths = _write(tmp_path, [_run("a", [3, 2], cats=cats, check=[1.0, 0.5]),
                              _run("b", [1, 2], cats=["multimodal", "multimodal"], check=[0.2, 0.5])])
    snap, _ = _build(paths)
    # b has no text prompts at all, so it has no text variant and the board has no check
    assert snap["check_available"] is False


def test_check_available_builds_check_variants(tmp_path):
    paths = _write(tmp_path, [
        _run("a", [3, 2], check=[1.0, 0.5]),
        _run("b", [3, 3], check=[0.9, 0.9]),
    ])
    snap, _ = _build(paths, view=dict(VIEW, metric="check"))
    assert snap["check_available"] is True and snap["view"]["metric"] == "check"
    a = next(e for e in snap["entries"] if e["model"] == "a")["variants"]["all"]["check"]
    assert a["pct"] == 75.0 and "pts" not in a and "max" not in a
    assert snap["entries"][0]["model"] == "b"
    assert validate_snapshot(snap, paths, Path.home()) == []


def test_output_is_deterministic(canary_paths):
    one = publish.serialize(_build(canary_paths)[0])
    two = publish.serialize(_build(canary_paths)[0])
    assert one == two


def test_view_and_frozen_are_validated(canary_paths):
    with pytest.raises(PublishError):
        _build(canary_paths, view=dict(VIEW, bogus=1))
    with pytest.raises(PublishError):
        _build(canary_paths, view=dict(VIEW, metric="auto"))
    with pytest.raises(PublishError):
        _build(canary_paths, frozen={"local": False, "notes": False, "sample": True})
    with pytest.raises(PublishError):
        _build(canary_paths, frozen={"local": "yes", "notes": False})


def test_write_snapshot_is_atomic_and_newline_terminated(canary_paths, tmp_path):
    snap, _ = _build(canary_paths)
    out = publish.write_snapshot(snap, tmp_path / "repo")
    assert out == tmp_path / "repo" / "site" / "bench" / "bench.json"
    text = out.read_text()
    assert text.endswith("}\n") and json.loads(text) == snap
    assert [p.name for p in out.parent.iterdir()] == ["bench.json"]


def test_entries_follow_rank_then_pct_within_a_tie_band(tmp_path):
    paths = _write(tmp_path, [
        _run("rep", [3, 3, 3, 3], ts="2026-11-01"), _run("rep", [3, 3, 0, 0], ts="2026-11-02"),
        _run("fast-lower", [3, 3, 3, 2]), _run("slow-higher", [3, 3, 3, 3]),
    ])
    # make fast-lower the fastest so a speed-first order would be wrong
    data = json.loads(paths[2].read_text())
    for r in data["results"]:
        r["completion_tokens"] = 10_000
    paths[2].write_text(json.dumps(data))
    snap, _ = _build(paths)
    vs = [e["variants"]["all"]["score"] for e in snap["entries"]]
    assert [v["pct"] for v in vs] == sorted((v["pct"] for v in vs), reverse=True)
    assert snap["entries"][0]["model"] == "slow-higher"


# ──────────────────────────────────────────────
# endpoints
# ──────────────────────────────────────────────

import re
import subprocess

from fastapi.testclient import TestClient


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def pub(tmp_path, monkeypatch):
    """A fake home with results, a publish config, and a cupel checkout to commit into."""
    import cupel.server as srv
    home = tmp_path / "home"
    results = home / ".cupel" / "eval-results"
    results.mkdir(parents=True)
    _write(results, [_run("a", [3, 2, 1]), _run("b", [1, 1, 1])])
    repo = tmp_path / "cupel-repo"
    (repo / "site").mkdir(parents=True)
    (repo / "site" / "index.html").write_text("landing\n")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "commit.gpgsign", "false")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    (home / ".cupel" / "config.yml").write_text(f"publish:\n  repo: {repo}\n  handle: tester\n")

    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    monkeypatch.setattr(srv, "RESULTS_DIR", results)
    monkeypatch.setattr(srv, "HIDDEN_FILE", results / ".hidden.json")
    monkeypatch.setattr(srv, "_cached_hardware", lambda: LOCAL_HW)
    srv._publish_tokens.clear()
    return {"client": TestClient(srv.app), "repo": repo, "srv": srv}


BODY = {"view": VIEW, "frozen": FROZEN}


def _preview(c):
    r = c.post("/api/publish/preview", json=BODY)
    assert r.status_code == 200, r.text
    return r.json()


def test_status_reports_enabled(pub):
    r = pub["client"].get("/api/publish/status").json()
    assert r["enabled"] and r["handle"] == "tester" and r["repo"] == str(pub["repo"])


def test_status_disabled_without_config(pub, tmp_path):
    (Path.home() / ".cupel" / "config.yml").write_text("models: []\n")
    r = pub["client"].get("/api/publish/status").json()
    assert r["enabled"] is False and "publish.repo" in r["reason"]
    assert pub["client"].post("/api/publish/preview", json=BODY).status_code == 404


def test_status_disabled_when_repo_is_not_git(pub, tmp_path):
    plain = tmp_path / "plain"
    (plain / "site").mkdir(parents=True)
    (Path.home() / ".cupel" / "config.yml").write_text(f"publish:\n  repo: {plain}\n")
    assert pub["client"].get("/api/publish/status").json()["enabled"] is False


@pytest.mark.parametrize("url", ["/api/publish/preview", "/api/publish/commit"])
def test_foreign_origin_is_refused(pub, url):
    r = pub["client"].post(url, json={}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403


@pytest.mark.parametrize("url", ["/api/publish/preview", "/api/publish/commit"])
def test_non_json_is_refused(pub, url):
    r = pub["client"].post(url, content="{}", headers={"Content-Type": "text/plain"})
    assert r.status_code == 415


def test_loopback_origin_is_allowed(pub):
    r = pub["client"].post("/api/publish/preview", json=BODY, headers={"Origin": "http://localhost:8042"})
    assert r.status_code == 200


def test_preview_returns_snapshot_and_summary(pub):
    out = _preview(pub["client"])
    assert out["token"] and out["snapshot"]["handle"] == "tester"
    assert out["summary"]["n_entries"] == 2
    assert "CANARY" not in json.dumps(out["snapshot"])


def test_preview_rejects_bad_view(pub):
    r = pub["client"].post("/api/publish/preview", json={"view": {"bogus": 1}, "frozen": FROZEN})
    assert r.status_code == 422


def test_unknown_or_expired_token_is_gone(pub):
    c = pub["client"]
    assert c.post("/api/publish/commit", json={"token": "nope"}).status_code == 410
    token = _preview(c)["token"]
    pub["srv"]._publish_tokens[token]["expires"] = 0
    assert c.post("/api/publish/commit", json={"token": token}).status_code == 410


def test_commit_makes_one_commit_touching_only_bench_json(pub):
    c, repo = pub["client"], pub["repo"]
    before = _git(repo, "rev-parse", "HEAD")
    (repo / "site" / "untracked-wip.txt").write_text("leave me alone\n")
    out = c.post("/api/publish/commit", json={"token": _preview(c)["token"]}).json()
    assert out["committed"] is True
    assert _git(repo, "rev-parse", "HEAD~1") == before
    assert out["sha"] == _git(repo, "rev-parse", "HEAD")
    assert _git(repo, "show", "--name-only", "--format=", "HEAD") == "site/bench/bench.json"
    assert out["message"] == "bench: publish 2 models, 3 prompts"
    assert _git(repo, "log", "-1", "--format=%s") == out["message"]
    assert "untracked-wip.txt" in _git(repo, "status", "--porcelain")


def test_staged_unrelated_file_blocks_the_commit(pub):
    c, repo = pub["client"], pub["repo"]
    before = _git(repo, "rev-parse", "HEAD")
    (repo / "site" / "index.html").write_text("edited\n")
    _git(repo, "add", "site/index.html")
    r = c.post("/api/publish/commit", json={"token": _preview(c)["token"]})
    assert r.status_code == 409 and "staged" in r.json()["detail"]
    assert _git(repo, "rev-parse", "HEAD") == before
    assert not (repo / "site" / "bench" / "bench.json").exists()


def test_dirty_bench_json_blocks_the_commit(pub):
    c, repo = pub["client"], pub["repo"]
    assert c.post("/api/publish/commit", json={"token": _preview(c)["token"]}).json()["committed"]
    (repo / "site" / "bench" / "bench.json").write_text("{}\n")
    r = c.post("/api/publish/commit", json={"token": _preview(c)["token"]})
    assert r.status_code == 409 and "uncommitted" in r.json()["detail"]


def test_unchanged_snapshot_is_not_committed(pub, monkeypatch):
    import cupel.server as srv
    c, repo = pub["client"], pub["repo"]
    fixed = datetime(2026, 11, 22, 12, 0, 0, tzinfo=timezone.utc)

    class FixedDT(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed
    monkeypatch.setattr(srv, "datetime", FixedDT)
    assert c.post("/api/publish/commit", json={"token": _preview(c)["token"]}).json()["committed"]
    head = _git(repo, "rev-parse", "HEAD")
    out = c.post("/api/publish/commit", json={"token": _preview(c)["token"]}).json()
    assert out == {"committed": False, "reason": "no changes", "path": out["path"]}
    assert _git(repo, "rev-parse", "HEAD") == head


def test_nothing_ever_pushes():
    root = Path(__file__).resolve().parent.parent / "cupel"
    for name in ("publish.py", "server.py", "board.py"):
        src = (root / name).read_text()
        assert not re.search(r"""["']push["']""", src), name
        assert "git push" not in src.replace("git -C <repo> push", ""), name

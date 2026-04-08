"""Smoke tests for cupel.config pure functions."""

import os
from pathlib import Path

from cupel.config import parse_prompt_ids, load_config, resolve_api_key_for_port, DEFAULTS


# ── parse_prompt_ids ──

def test_parse_single():
    assert parse_prompt_ids("3") == {3}

def test_parse_csv():
    assert parse_prompt_ids("1,5") == {1, 5}

def test_parse_range():
    assert parse_prompt_ids("18-22") == {18, 19, 20, 21, 22}

def test_parse_mixed():
    assert parse_prompt_ids("1,18-22,5") == {1, 5, 18, 19, 20, 21, 22}

def test_parse_single_range():
    assert parse_prompt_ids("3-3") == {3}


# ── load_config ──

def test_load_config_nonexistent():
    cfg, path = load_config(Path("/nonexistent/path"))
    assert path is None
    assert cfg == dict(DEFAULTS)
    for key in ("models", "eval_set", "temperature", "max_tokens"):
        assert key in cfg


# ── resolve_api_key_for_port ──

def test_resolve_api_key_env_cleared(monkeypatch):
    monkeypatch.delenv("OMLX_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert resolve_api_key_for_port(8000) == "no-key"

def test_resolve_api_key_env_set(monkeypatch):
    monkeypatch.setenv("OMLX_API_KEY", "test123")
    assert resolve_api_key_for_port(8000) == "test123"

def test_resolve_api_key_unknown_port(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert resolve_api_key_for_port(9999) == "no-key"


# ── version consistency ──

def test_version_is_string():
    from cupel import __version__
    assert isinstance(__version__, str)
    assert len(__version__) > 0

def test_version_matches_pyproject():
    from cupel import __version__
    if __version__ == "dev":
        return  # not installed, skip match check
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    text = pyproject.read_text()
    for line in text.splitlines():
        if line.strip().startswith("version"):
            # version = "0.1.53"
            v = line.split("=", 1)[1].strip().strip('"')
            assert __version__ == v
            return
    raise AssertionError("version not found in pyproject.toml")

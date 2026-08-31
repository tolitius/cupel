"""Security regression tests.

Each of these covers a hole that was confirmed exploitable against the running
server, not a hypothetical one.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import cupel.server as srv


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    return TestClient(srv.app)


VALID = json.dumps({"name": "x", "prompts": []})


# ── import-eval-set path traversal ──

@pytest.mark.parametrize("filename", [
    "../../ESCAPE.json", "../ESCAPE.json", "/tmp/ABS.json",
    "..%2F..%2FESCAPE.json", "subdir/../../ESCAPE.json",
])
def test_import_cannot_escape_the_eval_sets_directory(client, tmp_path, filename):
    r = client.post("/api/import-eval-set", json={"content": VALID, "filename": filename})
    assert r.status_code in (200, 400)
    if r.status_code == 200:
        # whatever it wrote must live directly in eval-sets/
        written = tmp_path / ".cupel" / "eval-sets"
        assert all(p.parent == written for p in written.rglob("*.json"))
    # nothing anywhere else in the fake home
    strays = [p for p in tmp_path.rglob("ESCAPE.json")
              if p.parent != tmp_path / ".cupel" / "eval-sets"]
    assert strays == []


def test_import_rejects_non_json_filenames(client):
    r = client.post("/api/import-eval-set", json={"content": VALID, "filename": "evil.sh"})
    assert r.status_code == 400


def test_import_rejects_empty_filename(client):
    r = client.post("/api/import-eval-set", json={"content": VALID, "filename": ""})
    assert r.status_code == 400


def test_import_still_accepts_a_normal_name(client, tmp_path):
    r = client.post("/api/import-eval-set", json={"content": VALID, "filename": "mine.json"})
    assert r.status_code == 200
    assert r.json()["path"] == "eval-sets/mine.json"
    assert (tmp_path / ".cupel" / "eval-sets" / "mine.json").exists()


def test_import_still_validates_content(client):
    bad = client.post("/api/import-eval-set", json={"content": "not json", "filename": "a.json"})
    assert bad.status_code == 400


# ── CORS ──

@pytest.mark.parametrize("origin", [
    "https://evil.example.com", "http://evil.com",
    "http://localhost.evil.com", "http://notlocalhost",
])
def test_cors_blocks_foreign_origins(client, origin):
    r = client.options("/api/jobs", headers={
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    })
    assert r.headers.get("access-control-allow-origin") is None


@pytest.mark.parametrize("origin", [
    "http://localhost:8042", "http://127.0.0.1:8042", "http://localhost:9999",
])
def test_cors_allows_the_local_ui_on_any_port(client, origin):
    r = client.options("/api/jobs", headers={
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    })
    assert r.headers.get("access-control-allow-origin") == origin


# ── bind address ──

def test_server_binds_loopback_by_default():
    """The API has no auth; 0.0.0.0 exposed it to the whole network."""
    import inspect
    assert inspect.signature(srv._start_ui).parameters["host"].default == "127.0.0.1"

import io
import json
import urllib.error

import pytest

from cupel import discovery

LLAMA_CPP_MODELS = {
    "models": [{"name": "nemotron-3.5-super-vl-nvfp4", "model": "nemotron-3.5-super-vl-nvfp4",
                "capabilities": ["completion", "multimodal"]}],
    "object": "list",
    "data": [{"id": "nemotron-3.5-super-vl-nvfp4", "aliases": ["nemotron-3.5-super-vl-nvfp4"],
              "object": "model", "owned_by": "llamacpp", "meta": {"n_ctx": 32768}}],
}


class _Resp(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def _http_error(url, code, body):
    return urllib.error.HTTPError(url, code, "err", {}, io.BytesIO(json.dumps(body).encode()))


@pytest.fixture
def server(monkeypatch):
    """Fake urlopen. `routes` maps host -> handler(req) returning (code, body) or raising."""
    routes, calls = {}, []
    monkeypatch.delenv("LLM_API_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLAMA_CPP_API_KEY", raising=False)

    def urlopen(req, timeout=None):
        host = req.full_url.split("//")[1].split(":")[0]
        calls.append((req.full_url, req.get_header("Authorization")))
        handler = routes.get(host)
        if handler is None:
            raise urllib.error.URLError("connection refused")
        code, body = handler(req)
        if code != 200:
            raise _http_error(req.full_url, code, body)
        return _Resp(json.dumps(body).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    return routes, calls


def test_parse_llama_cpp_response():
    assert discovery._parse_models(LLAMA_CPP_MODELS) == ["nemotron-3.5-super-vl-nvfp4"]


def test_parse_ollama_shape_only():
    assert discovery._parse_models({"models": [{"name": "a"}, {"model": "b"}, {"name": "a"}]}) == ["a", "b"]


def test_parse_garbage():
    assert discovery._parse_models([]) == []
    assert discovery._parse_models({"data": [{"object": "model"}]}) == []


def test_llama_cpp_discovered(server):
    routes, _ = server
    routes["localhost"] = lambda req: (200, LLAMA_CPP_MODELS)
    p = discovery._probe_port("localhost", 8080)
    assert p == {"url": "http://localhost:8080", "port": 8080, "status": "online",
                 "models": ["nemotron-3.5-super-vl-nvfp4"]}


def test_loading_model(server):
    routes, _ = server
    routes["localhost"] = lambda req: (503, {"error": {"code": 503, "message": "Loading model"}})
    p = discovery._probe_port("localhost", 8080)
    assert p["status"] == "loading"
    assert p["detail"] == "Loading model"
    assert p["models"] == []


def test_not_found_reports_detail(server):
    routes, _ = server
    routes["localhost"] = lambda req: (404, {"error": "not found"})
    p = discovery._probe_port("localhost", 8080)
    assert p["status"] == "online"
    assert p["models"] == []
    assert p["detail"] == "HTTP 404 from /v1/models: not found"


def test_falls_back_to_ipv4_when_localhost_has_no_models(server):
    routes, calls = server
    routes["localhost"] = lambda req: (404, {})              # e.g. another service on ::1
    routes["127.0.0.1"] = lambda req: (200, LLAMA_CPP_MODELS)
    p = discovery._probe_port("localhost", 8080)
    assert p["url"] == "http://127.0.0.1:8080"
    assert p["models"] == ["nemotron-3.5-super-vl-nvfp4"]
    assert [c[0].split("/")[2] for c in calls] == ["localhost:8080", "127.0.0.1:8080"]


def test_falls_back_to_ipv4_when_localhost_offline(server):
    routes, _ = server
    routes["127.0.0.1"] = lambda req: (200, LLAMA_CPP_MODELS)
    assert discovery._probe_port("localhost", 8080)["url"] == "http://127.0.0.1:8080"


def test_offline(server):
    p = discovery._probe_port("localhost", 8080)
    assert p == {"url": "http://localhost:8080", "port": 8080, "status": "offline", "models": []}


def test_generic_key_not_sent_unless_asked(server, monkeypatch):
    routes, calls = server
    monkeypatch.setenv("LLM_API_KEY", "omlx-secret")
    routes["localhost"] = lambda req: (200, LLAMA_CPP_MODELS)
    discovery._probe_port("localhost", 8080)
    assert calls == [("http://localhost:8080/v1/models", None)]


def test_generic_key_retried_on_401(server, monkeypatch):
    routes, calls = server
    monkeypatch.setenv("LLM_API_KEY", "shared")
    routes["localhost"] = lambda req: ((200, LLAMA_CPP_MODELS) if req.get_header("Authorization") == "Bearer shared"
                                       else (401, {"error": "unauthorized"}))
    p = discovery._probe_port("localhost", 8080)
    assert p["models"] == ["nemotron-3.5-super-vl-nvfp4"]
    assert [c[1] for c in calls] == [None, "Bearer shared"]


def test_port_key_sent_and_rejected(server, monkeypatch):
    routes, calls = server
    monkeypatch.setenv("LLAMA_CPP_API_KEY", "wrong")
    routes["localhost"] = lambda req: (401, {"error": "unauthorized"})
    routes["127.0.0.1"] = lambda req: (401, {"error": "unauthorized"})
    p = discovery._probe_port("localhost", 8080)
    assert p["auth_error"] == "auth_failed"
    assert calls[0] == ("http://localhost:8080/v1/models", "Bearer wrong")


def test_auth_required_without_key(server):
    routes, _ = server
    routes["localhost"] = lambda req: (401, {"error": "unauthorized"})
    routes["127.0.0.1"] = lambda req: (401, {"error": "unauthorized"})
    assert discovery._probe_port("localhost", 8080)["auth_error"] == "auth_required"


def test_discover_providers_probes_all_ports(server):
    routes, _ = server
    routes["localhost"] = lambda req: ((200, LLAMA_CPP_MODELS) if ":8080/" in req.full_url else (404, {}))
    ps = {p["port"]: p for p in discovery.discover_providers()}
    assert sorted(ps) == sorted(discovery.PROBE_PORTS)
    assert ps[8080]["models"] == ["nemotron-3.5-super-vl-nvfp4"]

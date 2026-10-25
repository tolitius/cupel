"""cupel.discovery — hardware detection and inference provider discovery."""

import json
import logging
import os
import platform
import subprocess

log = logging.getLogger("cupel")


def detect_hardware():
    """Detect machine hardware. Returns dict with name, memory, spec."""

    info = {
        "name": platform.machine(),
        "memory": "unknown",
        "spec": "",
    }

    if platform.system() == "Darwin":
        try:
            chip = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL, timeout=5
            ).decode().strip()
            info["name"] = chip
        except Exception as e:
            log.debug("sysctl brand_string failed: %s", e)
        # Try to get Apple Silicon chip name
        try:
            chip_info = subprocess.check_output(
                ["system_profiler", "SPHardwareDataType"],
                stderr=subprocess.DEVNULL, timeout=10
            ).decode()
            for line in chip_info.splitlines():
                line = line.strip()
                if "Chip" in line and ":" in line:
                    info["name"] = line.split(":", 1)[1].strip()
                elif "Memory" in line and ":" in line:
                    info["memory"] = line.split(":", 1)[1].strip()
                elif "Total Number of Cores" in line and ":" in line:
                    info["spec"] = line.split(":", 1)[1].strip() + " cores"
        except Exception as e:
            log.debug("system_profiler failed: %s", e)
    else:
        # Linux — try nvidia-smi
        try:
            gpu_output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL, timeout=5
            ).decode().strip()
            if gpu_output:
                from collections import Counter
                gpu_names = []
                total_mem_mib = 0
                for line in gpu_output.splitlines():
                    parts = line.split(",")
                    gpu_names.append(parts[0].strip())
                    if len(parts) > 1:
                        mem_str = parts[1].strip()  # e.g. "24576 MiB"
                        try:
                            total_mem_mib += int(mem_str.split()[0])
                        except (ValueError, IndexError):
                            pass
                counts = Counter(gpu_names)
                if len(counts) == 1 and len(gpu_names) == 1:
                    info["name"] = gpu_names[0]
                else:
                    info["name"] = ", ".join(
                        f"{n}x {name}" if n > 1 else name
                        for name, n in counts.items()
                    )
                if total_mem_mib:
                    info["memory"] = f"{total_mem_mib} MiB"
                if len(gpu_names) > 1:
                    info["spec"] = f"{len(gpu_names)} GPUs"
        except Exception as e:
            log.debug("nvidia-smi failed: %s", e)
        # Memory
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        info["memory"] = f"{kb // (1024*1024)} GB"
                        break
        except Exception as e:
            log.debug("/proc/meminfo read failed: %s", e)

    return info


def detect_thermal():
    """Read macOS thermal state via ProcessInfo.thermalState. Returns int 0-3 or None."""
    if platform.system() != "Darwin":
        return None
    try:
        # timeout=60: swift cold-compiles Foundation module cache (~20s) on first run
        out = subprocess.check_output(
            ["swift", "-e", "import Foundation; print(ProcessInfo.processInfo.thermalState.rawValue)"],
            stderr=subprocess.DEVNULL,
            timeout=60
        ).decode().strip()
        return int(out)
    except subprocess.TimeoutExpired:
        # cold compile can exceed even 60s on slow machines — not an error
        log.debug("thermal detection timed out (swift module cache cold compile)")
        return None
    except Exception as e:
        log.warning("thermal detection failed: %s", e)
        return None


PROBE_PORTS = [8000, 11434, 1234, 30000, 8080]
MODELS_PATH = "/v1/models"


def _parse_models(data):
    """Model ids from a /v1/models body: OpenAI-style data[].id, else Ollama-style models[].model/name."""
    if not isinstance(data, dict):
        return []
    ids = [m.get("id") for m in data.get("data") or [] if isinstance(m, dict)]
    if not any(ids):
        ids = [m.get("model") or m.get("name") for m in data.get("models") or [] if isinstance(m, dict)]
    return list(dict.fromkeys(i for i in ids if i))


def _error_message(body):
    """Pull a human-readable message out of an error body (JSON or text)."""
    try:
        data = json.loads(body)
        err = data.get("error", data) if isinstance(data, dict) else data
        if isinstance(err, dict):
            err = err.get("message") or err.get("detail") or ""
        return str(err)[:120]
    except Exception:
        return body.decode("utf-8", "replace").strip()[:120] if isinstance(body, bytes) else ""


def _request(url, api_key):
    """GET url. Returns (http_code, body); raises on connection errors."""
    import urllib.request
    import urllib.error

    req = urllib.request.Request(url, method="GET")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        try:
            body = e.read()
        except Exception:
            body = b""
        return e.code, body


def _probe(host, port):
    """Probe one endpoint. Returns a dict with status, models and, when relevant, auth_error / detail."""
    from cupel.config import port_api_key

    url = f"http://{host}:{port}{MODELS_PATH}"
    # Only send the key meant for this port; LLM_API_KEY is tried only if the server asks for auth
    port_key = port_api_key(port)
    fallback_key = os.environ.get("LLM_API_KEY", "")
    if fallback_key == "no-key":
        fallback_key = ""
    try:
        code, body = _request(url, port_key)
        used_key = bool(port_key)
        if code in (401, 403) and not port_key and fallback_key:
            code, body = _request(url, fallback_key)
            used_key = True
    except Exception:
        return {"status": "offline", "models": []}

    try:
        models = _parse_models(json.loads(body))
    except Exception:
        models = []
    if models:
        return {"status": "online", "models": models}

    if code == 503:
        return {"status": "loading", "models": [],
                "detail": _error_message(body) or "server is loading"}
    if code in (401, 403):
        return {"status": "online", "models": [],
                "auth_error": "auth_failed" if used_key else "auth_required",
                "detail": f"HTTP {code} from {MODELS_PATH}"}
    if code == 200:
        return {"status": "online", "models": [], "detail": "no models in response"}
    msg = _error_message(body)
    return {"status": "online", "models": [],
            "detail": f"HTTP {code} from {MODELS_PATH}" + (f": {msg}" if msg else "")}


def _probe_port(host, port):
    """Probe host:port; for localhost also try 127.0.0.1, since localhost may resolve to ::1
    where a different (or no) server listens. Returns a provider dict."""
    hosts = [host] + (["127.0.0.1"] if host == "localhost" else [])
    best = None
    for h in hosts:
        entry = {"url": f"http://{h}:{port}", "port": port, **_probe(h, port)}
        if entry["models"]:
            return entry
        if best is None or (best["status"] == "offline" and entry["status"] != "offline"):
            best = entry
    return best


def discover_providers():
    """Probe known ports for inference servers. Returns list of provider dicts."""
    from urllib.parse import urlparse

    results = []
    seen_ports = set()

    # Check LLM_API_URL env first
    env_url = os.environ.get("LLM_API_URL", "")
    if env_url:
        try:
            parsed = urlparse(env_url)
            port = parsed.port or 80
            host = parsed.hostname or "localhost"
            seen_ports.add(port)
            results.append(_probe_port(host, port))
        except Exception:
            pass

    for port in PROBE_PORTS:
        if port in seen_ports:
            continue
        seen_ports.add(port)
        results.append(_probe_port("localhost", port))

    return results

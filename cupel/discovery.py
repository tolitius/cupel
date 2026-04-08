"""cupel.discovery — hardware detection and inference provider discovery."""

import json
import os
import platform
import subprocess


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
        except Exception:
            pass
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
        except Exception:
            pass
    else:
        # Linux — try nvidia-smi
        try:
            gpu = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL, timeout=5
            ).decode().strip()
            if gpu:
                info["name"] = gpu.split(",")[0].strip()
                info["memory"] = gpu.split(",")[1].strip() if "," in gpu else "unknown"
        except Exception:
            pass
        # Memory
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        info["memory"] = f"{kb // (1024*1024)} GB"
                        break
        except Exception:
            pass

    return info


def discover_providers():
    """Probe known ports for inference servers. Returns list of provider dicts."""
    import urllib.request
    import urllib.error
    import socket
    from cupel.config import resolve_api_key_for_port

    PROBE_PORTS = [
        {"port": 8000,  "models_path": "/v1/models"},
        {"port": 11434, "models_path": "/v1/models"},
        {"port": 1234,  "models_path": "/v1/models"},
        {"port": 30000, "models_path": "/v1/models"},
    ]

    def _probe(host, port, models_path):
        """Probe a single endpoint. Returns (status, models, auth_error)."""
        api_key = resolve_api_key_for_port(port)
        url = f"http://{host}:{port}{models_path}"
        req = urllib.request.Request(url, method="GET")
        has_key = bool(api_key and api_key != "no-key")
        if has_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        try:
            resp = urllib.request.urlopen(req, timeout=2)
            data = json.loads(resp.read())
            models = [m["id"] for m in data.get("data", [])]
            return "online", models, None
        except urllib.error.HTTPError as e:
            # Server responded (even with 401/403) — it's online
            if e.code in (401, 403):
                # Try to parse body for model list anyway
                try:
                    body = e.read()
                    data = json.loads(body)
                    models = [m["id"] for m in data.get("data", [])]
                    if models:
                        return "online", models, None
                except Exception:
                    pass
                # No models from auth error
                if has_key:
                    return "online", [], "auth_failed"
                else:
                    return "online", [], "auth_required"
            # Other HTTP errors — server is online
            try:
                body = e.read()
                data = json.loads(body)
                models = [m["id"] for m in data.get("data", [])]
                return "online", models, None
            except Exception:
                return "online", [], None
        except (urllib.error.URLError, socket.timeout, OSError):
            return "offline", [], None
        except Exception:
            return "offline", [], None

    results = []
    seen_ports = set()

    # Check LLM_API_URL env first
    env_url = os.environ.get("LLM_API_URL", "")
    if env_url:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(env_url)
            port = parsed.port or 80
            host = parsed.hostname or "localhost"
            if port not in seen_ports:
                seen_ports.add(port)
                status, models, auth_error = _probe(host, port, "/v1/models")
                entry = {
                    "url": f"http://{host}:{port}",
                    "port": port,
                    "status": status,
                    "models": models,
                }
                if auth_error:
                    entry["auth_error"] = auth_error
                results.append(entry)
        except Exception:
            pass

    for probe in PROBE_PORTS:
        port = probe["port"]
        if port in seen_ports:
            continue
        seen_ports.add(port)
        status, models, auth_error = _probe("localhost", port, probe["models_path"])
        entry = {
            "url": f"http://localhost:{port}",
            "port": port,
            "status": status,
            "models": models,
        }
        if auth_error:
            entry["auth_error"] = auth_error
        results.append(entry)

    return results

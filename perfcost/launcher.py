"""Start / stop the dashboard (a Streamlit server) as a background process, and show its link."""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from .store import DEFAULT_ROOT

APP = Path(__file__).with_name("app.py")
STATE = ".perfcost_server.json"


def parse_cpus(spec):
    """'32-63' or '0,2,4-7' -> {32, ..., 63}."""
    cpus = set()
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            cpus.update(range(int(a), int(b) + 1))
        elif part:
            cpus.add(int(part))
    return cpus


def _port_open(port, host="127.0.0.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) == 0


def _alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def server_url(port=8501):
    return f"http://localhost:{port}"


def serve(root=None, port=8501, cpus=None, wait=20.0):
    """Start the dashboard for `root` unless it is already running; returns its URL.

    The server is a separate process that keeps running after the notebook cell (and the kernel) end,
    so one dashboard shows every run under root. Stop it with perfcost.stop_server().
    """
    root = Path(root or DEFAULT_ROOT).resolve()
    root.mkdir(parents=True, exist_ok=True)
    state_file = root / STATE
    if state_file.exists():
        st = json.loads(state_file.read_text())
        if _alive(st.get("pid")) and _port_open(st.get("port", port)):
            return server_url(st["port"])
    while _port_open(port):              # port taken by something else (or another root's dashboard)
        port += 1
    cmd = [sys.executable, "-m", "streamlit", "run", str(APP), "--server.port", str(port),
           "--server.headless", "true", "--browser.gatherUsageStats", "false",
           "--server.runOnSave", "false", "--", "--root", str(root)]
    preexec = None
    if cpus is not None and hasattr(os, "sched_setaffinity"):
        cores = parse_cpus(cpus) & set(range(os.cpu_count() or 1))
        if cores:
            preexec = lambda: os.sched_setaffinity(0, cores)   # noqa: E731
        else:
            import warnings
            warnings.warn(f"cpus={cpus!r} does not exist on this machine ({os.cpu_count()} cores); dashboard not pinned")
    log = open(root / ".perfcost_server.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, preexec_fn=preexec, start_new_session=True)
    state_file.write_text(json.dumps({"pid": proc.pid, "port": port, "cpus": cpus, "started": time.time()}))
    t0 = time.time()
    while time.time() - t0 < wait:
        if _port_open(port):
            break
        if proc.poll() is not None:
            raise RuntimeError(f"dashboard failed to start; see {root / '.perfcost_server.log'}")
        time.sleep(0.3)
    return server_url(port)


def stop_server(root=None):
    root = Path(root or DEFAULT_ROOT).resolve()
    state_file = root / STATE
    if not state_file.exists():
        return False
    st = json.loads(state_file.read_text())
    try:
        os.killpg(os.getpgid(int(st["pid"])), 15)
    except (OSError, KeyError, ValueError):
        pass
    state_file.unlink(missing_ok=True)
    return True


def show_link(url, text="perfcost dashboard"):
    """Clickable link in Jupyter / VS Code notebooks, plain text elsewhere."""
    hint = ("(on a remote VM through VS Code, the port is forwarded automatically; if the link does not open, "
            "add the port in the PORTS tab)")
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display
        if get_ipython() is not None:
            display(HTML(f'<a href="{url}" target="_blank" style="font-size:1.1em">&#128202; {text} &rarr; {url}</a>'
                         f'<br><small>{hint}</small>'))
            return
    except ImportError:
        pass
    print(f"{text}: {url}\n{hint}")


def browse(root=None, port=8501, cpus=None):
    """Open the dashboard for finished runs (starts it if needed)."""
    url = serve(root=root, port=port, cpus=cpus)
    show_link(url)
    return url

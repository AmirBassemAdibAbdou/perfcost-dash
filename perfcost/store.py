"""On-disk format of a run, and helpers to read it back.

A run is one folder  <root>/<run_id>/  with three files:
  meta.json      who/what: name, dataset, model, labels, hardware, status, timestamps
  events.jsonl   one line per prediction: {"i", "t", "y", "p", "c", "l"}
                 (index, unix time, true label, predicted label, confidence, latency in ms)
  costs.jsonl    one line per cost record: {"t", "kind", "energy_kwh", "n", "wall_time_s", "mode", ...}

Everything is append-only text, so a crashed run is still readable and the dashboard can read the files
while they are being written.
"""
import json
import os
import time
from pathlib import Path

import pandas as pd

DEFAULT_ROOT = Path(os.environ.get("PERFCOST_ROOT", "perfcost_runs"))
EVENT_COLUMNS = {"i": "index", "t": "time", "y": "y_true", "p": "y_pred", "c": "confidence", "l": "latency_ms"}


def write_json_atomic(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def read_jsonl(path):
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:      # a half-written last line while the run is going
                        pass
    except FileNotFoundError:
        pass
    return rows


def list_runs(root=DEFAULT_ROOT):
    """All runs under root, newest first, as a list of meta dicts (with 'run_id' and 'path')."""
    root = Path(root)
    out = []
    if not root.exists():
        return out
    for d in root.iterdir():
        mf = d / "meta.json"
        if d.is_dir() and mf.exists():
            try:
                meta = json.loads(mf.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            meta["run_id"], meta["path"] = d.name, str(d)
            if meta.get("status") == "running" and not _alive(meta.get("pid")):
                meta["status"] = "stopped"            # process is gone without finish()
            out.append(meta)
    return sorted(out, key=lambda m: m.get("started_at", 0), reverse=True)


def _alive(pid):
    if not pid:
        return True
    if pid == os.getpid():
        return True
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def load_events(run_dir):
    rows = read_jsonl(Path(run_dir) / "events.jsonl")
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=list(EVENT_COLUMNS.values()))
    df = df.rename(columns=EVENT_COLUMNS)
    for c in EVENT_COLUMNS.values():
        if c not in df:
            df[c] = None
    df["y_true"] = df["y_true"].astype(str)
    df["y_pred"] = df["y_pred"].map(lambda v: "__invalid__" if v is None or (isinstance(v, float) and v != v) else str(v))
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
    return df.sort_values("index").reset_index(drop=True)


def load_costs(run_dir):
    return pd.DataFrame(read_jsonl(Path(run_dir) / "costs.jsonl"))


def now():
    return time.time()

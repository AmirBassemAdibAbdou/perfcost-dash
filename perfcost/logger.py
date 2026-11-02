"""Recording a run: perfcost.start(...) -> Run; run.log(...); run.log_cost(...); run.finish()."""
import json
import os
import platform
import re
import threading
import time
from pathlib import Path

from .store import DEFAULT_ROOT, write_json_atomic


def _slug(text):
    return re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()[:60] or "run"


def _clean(v):
    """JSON-friendly scalar (numpy types, NaN -> None)."""
    if v is None:
        return None
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and v != v:
        return None
    return v


def hardware_info():
    info = {"platform": platform.platform(), "python": platform.python_version(), "cpu_count": os.cpu_count()}
    try:
        info["usable_cores"] = len(os.sched_getaffinity(0))
    except AttributeError:
        pass
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                info["cpu_model"] = line.split(":", 1)[1].strip()
                break
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                info["ram_gb"] = round(int(line.split()[1]) / 1024 ** 2, 1)
                break
    except OSError:
        pass
    return info


class Run:
    """One run of one model on one dataset. Thread-safe; writes are buffered and append-only."""

    def __init__(self, path, meta, flush_every=50, flush_seconds=2.0):
        self.path = Path(path)
        self.meta = meta
        self._buf, self._lock = [], threading.Lock()
        self._i = 0
        self._last_flush = time.time()
        self._flush_every, self._flush_seconds = flush_every, flush_seconds
        self._tracker = None
        self.path.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self.path / "meta.json", self.meta)

    # ------------------------------------------------------------ predictions
    def log(self, y_true, y_pred, confidence=None, latency_ms=None, t=None):
        """Record one prediction. y_pred=None means an invalid / unparseable answer."""
        with self._lock:
            self._buf.append({"i": self._i, "t": t or time.time(), "y": _clean(y_true), "p": _clean(y_pred),
                              "c": _clean(confidence), "l": _clean(latency_ms)})
            self._i += 1
            if len(self._buf) >= self._flush_every or time.time() - self._last_flush > self._flush_seconds:
                self._flush()

    def log_batch(self, y_true, y_pred, confidence=None, latency_ms=None):
        """Record many predictions at once (e.g. after a batch predict). Lists of equal length."""
        n = len(y_true)
        confidence = confidence if confidence is not None else [None] * n
        latency_ms = latency_ms if latency_ms is not None else [None] * n
        t = time.time()
        for y, p, c, l in zip(y_true, y_pred, confidence, latency_ms):
            self.log(y, p, c, l, t=t)
        self.flush()

    def _flush(self):
        if self._buf:
            with open(self.path / "events.jsonl", "a") as f:
                f.write("".join(json.dumps(r) + "\n" for r in self._buf))
            self._buf = []
        self._last_flush = time.time()

    def flush(self):
        with self._lock:
            self._flush()

    # ------------------------------------------------------------ costs
    def log_cost(self, kind="inference", energy_kwh=None, n=None, wall_time_s=None, mode=None, **extra):
        """Record a cost measurement.

        kind: "inference" (energy for n predictions; mode "online" or "batch"), "training", "cold_start",
              or anything else you want to keep. Extra keyword values are stored as they are
              (e.g. throughput_per_s=..., latency_ms_mean=..., co2_kg=...).
        """
        row = {"t": time.time(), "kind": kind, "energy_kwh": _clean(energy_kwh), "n": _clean(n),
               "wall_time_s": _clean(wall_time_s), "mode": mode}
        row.update({k: _clean(v) for k, v in extra.items()})
        with self._lock, open(self.path / "costs.jsonl", "a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def update_meta(self, **kw):
        self.meta.update(kw)
        write_json_atomic(self.path / "meta.json", self.meta)

    # ------------------------------------------------------------ end
    def finish(self, status="finished"):
        self.flush()
        if self._tracker:
            self._tracker.stop(self)
            self._tracker = None
        self.update_meta(status=status, finished_at=time.time(), n_logged=self._i)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.finish("failed" if exc_type else "finished")
        return False

    def __repr__(self):
        return f"<perfcost.Run {self.meta.get('name')!r} n={self._i} at {self.path}>"


class _EnergyTracker:
    """Optional: CodeCarbon in a background thread, recording cumulative energy every few seconds."""

    def __init__(self, run, interval=5.0, country_iso_code="USA", tracking_mode="machine"):
        from codecarbon import OfflineEmissionsTracker   # optional dependency: pip install perfcost-dash[energy]
        self.tracker = OfflineEmissionsTracker(country_iso_code=country_iso_code, tracking_mode=tracking_mode,
                                               measure_power_secs=1, save_to_file=False, log_level="error")
        self.interval, self.t0 = interval, time.time()
        self._stop = threading.Event()
        self.tracker.start()
        self.thread = threading.Thread(target=self._loop, args=(run,), daemon=True)
        self.thread.start()

    def _energy_now(self):
        e = getattr(self.tracker, "_total_energy", None)
        return getattr(e, "kWh", None)

    def _loop(self, run):
        while not self._stop.wait(self.interval):
            e = self._energy_now()
            if e is not None:
                run.log_cost(kind="live_energy", energy_kwh_cum=float(e))

    def stop(self, run):
        self._stop.set()
        self.tracker.stop()
        d = self.tracker.final_emissions_data
        run.log_cost(kind="live_energy", energy_kwh_cum=float(d.energy_consumed))
        run.log_cost(kind="inference", energy_kwh=float(d.energy_consumed), n=run._i,
                     wall_time_s=time.time() - self.t0, mode="tracked", co2_kg=float(d.emissions))


def start(name, dataset=None, model=None, labels=None, expected_n=None, preset="balanced", root=None,
          launch=True, port=8501, cpus=None, track_energy=False, country_iso_code="USA", show=True, tags=None):
    """Start recording a run and (by default) make sure the dashboard is running.

    name        free text shown in the dashboard, e.g. "adult / llama3.1-8b few-shot"
    dataset     dict, e.g. {"name": "Adult Income", "modality": "tabular", "n_train": 32561, "n_test": 16281}
                (runs with the same dataset name are compared with each other)
    model       dict, e.g. {"name": "Llama 3.1 8B", "params": "8B", "quantization": "Q4_K_M", "strategy": "few-shot"}
    labels      the closed label set (optional; otherwise taken from the true labels)
    expected_n  how many predictions this run will make (for progress and time-left)
    launch      start the dashboard server if it is not running yet
    cpus        pin the dashboard process to these cores, e.g. "32-63" (keeps it off your measured cores)
    track_energy  also measure energy with CodeCarbon (needs `pip install perfcost-dash[energy]`);
                leave False if you measure energy yourself and pass it with run.log_cost()
    """
    root = Path(root or DEFAULT_ROOT)
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = root / f"{ts}_{_slug(name)}"
    k = 1
    while path.exists():
        k += 1
        path = root / f"{ts}_{_slug(name)}-{k}"
    meta = dict(name=name, dataset=dataset or {}, model=model or {}, labels=[str(x) for x in labels] if labels else None,
                expected_n=expected_n or (dataset or {}).get("n_test"), preset=preset, tags=tags or [],
                hardware=hardware_info(), started_at=time.time(), status="running", pid=os.getpid())
    run = Run(path, meta)
    if track_energy:
        run._tracker = _EnergyTracker(run, country_iso_code=country_iso_code)
    if launch:
        from .launcher import serve, show_link
        url = serve(root=root, port=port, cpus=cpus)
        if show:
            show_link(url, f"perfcost dashboard: {name}")
    return run


def import_run(name, y_true, y_pred, confidence=None, latency_ms=None, dataset=None, model=None, labels=None,
               costs=None, root=None, started_at=None, tags=None):
    """Store an already finished run (e.g. from a predictions CSV) so the dashboard can compare it.

    costs: list of dicts passed to Run.log_cost, e.g.
           [{"kind": "training", "energy_kwh": 0.03, "wall_time_s": 700},
            {"kind": "inference", "mode": "online", "energy_kwh": 0.0004, "n": 1000, "throughput_per_s": 98}]
    """
    run = start(name, dataset=dataset, model=model, labels=labels, expected_n=len(y_true), root=root,
                launch=False, tags=tags)
    t = started_at or time.time()
    n = len(y_true)
    confidence = list(confidence) if confidence is not None else [None] * n
    latency_ms = list(latency_ms) if latency_ms is not None else [None] * n
    for y, p, c, l in zip(y_true, y_pred, confidence, latency_ms):
        t += (_clean(l) or 0) / 1000
        run.log(y, p, c, l, t=t)
    for c in costs or []:
        run.log_cost(**c)
    run.update_meta(pid=None, imported=True)
    run.finish()
    return run

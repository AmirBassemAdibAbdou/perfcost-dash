"""`perfcost demo`: simulated runs so you can see the dashboard without training anything.

Two finished runs are imported instantly (a fast "traditional" model and a slow "LLM"), then a third run is
logged live for about a minute. All numbers are SIMULATED and labelled as such.
"""
import time

import numpy as np

from .logger import import_run, start

LABELS = ["cat", "dog", "bird", "fish"]
DATASET = {"name": "Demo (simulated)", "modality": "synthetic", "n_train": 8000, "n_test": 2000, "classes": LABELS}


def _simulate(n, acc, conf_ok, conf_bad, latency_ms, rng):
    y = rng.choice(LABELS, n, p=[0.4, 0.3, 0.2, 0.1])
    right = rng.random(n) < acc
    p = np.where(right, y, [rng.choice([l for l in LABELS if l != t]) for t in y])
    c = np.where(right, rng.uniform(*conf_ok, n), rng.uniform(*conf_bad, n))
    lat = rng.lognormal(np.log(latency_ms), 0.25, n)
    return y, p, c, lat


def run_demo(root=None, port=8501, cpus=None, n_live=2000, seconds=60):
    rng = np.random.default_rng(0)
    y, p, c, lat = _simulate(2000, 0.91, (0.7, 1.0), (0.4, 0.8), 2.0, rng)
    import_run("demo / gradient boosting (simulated)", y, p, c, lat, dataset=DATASET,
               model={"name": "Gradient boosting", "strategy": "trained", "params": "~1 MB"},
               labels=LABELS, root=root,
               costs=[{"kind": "training", "energy_kwh": 0.002, "wall_time_s": 60},
                      {"kind": "inference", "mode": "online", "energy_kwh": 0.0002, "n": 1000},
                      {"kind": "inference", "mode": "batch", "energy_kwh": 0.00001, "n": 2000, "throughput_per_s": 50000}])
    y, p, c, lat = _simulate(2000, 0.86, (0.9, 1.0), (0.85, 1.0), 900.0, rng)
    import_run("demo / 8B LLM few-shot (simulated)", y, p, c, lat, dataset=DATASET,
               model={"name": "8B LLM", "strategy": "few-shot", "params": "8B", "quantization": "Q4_K_M"},
               labels=LABELS, root=root,
               costs=[{"kind": "inference", "mode": "online", "energy_kwh": 0.03, "n": 1000},
                      {"kind": "cold_start", "wall_time_s": 6.5}])

    run = start("demo / 1B LLM LoRA (simulated, live)", dataset=DATASET,
                model={"name": "1B LLM + LoRA", "strategy": "LoRA", "params": "1B", "quantization": "Q4_K_M"},
                labels=LABELS, expected_n=n_live, root=root, port=port, cpus=cpus)
    run.log_cost(kind="training", energy_kwh=1.5, wall_time_s=6 * 3600)
    y, p, c, lat = _simulate(n_live, 0.89, (0.6, 1.0), (0.5, 0.9), 150.0, rng)
    pause = seconds / n_live
    for i in range(n_live):
        time.sleep(pause)
        run.log(y[i], p[i], c[i], lat[i])
        if i == n_live // 2:
            run.log_cost(kind="inference", mode="online", energy_kwh=0.006, n=1000)
    run.finish()
    print("demo finished; the dashboard keeps running (perfcost stop to end it)")

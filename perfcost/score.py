"""The cost-performance score.

Five components, each scored from 0 to 1 RELATIVE TO THE OTHER RUNS ON THE SAME DATASET (the reference set):

  performance   macro-F1 / best macro-F1 in the reference set
  calibration   1 - ECE                                   (how far confidence can be trusted)
  latency       1 / (1 + log10(latency / lowest latency))
  energy        1 / (1 + log10(E / lowest E)),   E = inference energy per request + training energy / N
  throughput    1 / (1 + log10(highest throughput / throughput))

Costs are compared on a log scale because they span orders of magnitude: every TENFOLD gap to the best run
has the same effect (1x -> 1.00, 10x -> 0.50, 100x -> 0.33, 1000x -> 0.25), and no run ever scores 0 just
for being the most expensive one. N is the expected number of requests over the model's life: training
energy is spread over N requests, so at small N training-free models look cheaper and at large N trained
models do (the same trade-off as the break-even point).

The total is a WEIGHTED GEOMETRIC MEAN of the components, so a very bad component cannot be hidden by good
ones (a weighted sum would let it). Missing components (e.g. no energy logged yet) are left out and the
remaining weights are renormalised.

Weights are a value judgement, so the dashboard offers presets plus sliders; report rankings under more
than one preset.
"""
import math

COMPONENTS = ["performance", "calibration", "latency", "energy", "throughput"]
PRESETS = {
    "accuracy-first": {"performance": 0.60, "calibration": 0.10, "latency": 0.10, "energy": 0.10, "throughput": 0.10},
    "balanced":       {"performance": 0.30, "calibration": 0.10, "latency": 0.20, "energy": 0.30, "throughput": 0.10},
    "green":          {"performance": 0.15, "calibration": 0.05, "latency": 0.15, "energy": 0.50, "throughput": 0.15},
}
DEFAULT_VOLUME = 1_000_000
FLOOR = 0.01            # keeps the geometric mean defined when a component is ~0


def _ok(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def _log_ratio_score(value, best, lower_is_better=True):
    if not (_ok(value) and _ok(best)) or value <= 0 or best <= 0:
        return None
    ratio = value / best if lower_is_better else best / value
    return 1.0 / (1.0 + math.log10(max(ratio, 1.0)))


def energy_per_request(m, volume=DEFAULT_VOLUME):
    e = m.get("energy_per_1000_kwh")
    if not _ok(e):
        return None
    train = m.get("train_energy_kwh") or 0.0
    return e / 1000 + (train / volume if volume else 0.0)


def components(m, reference, volume=DEFAULT_VOLUME):
    """Component scores of metrics dict m against a list of metrics dicts (m may be in the list)."""
    ref = list(reference) + [m]
    best_f1 = max((r["macro_f1"] for r in ref if _ok(r.get("macro_f1"))), default=None)
    lats = [r["latency_ms_mean"] for r in ref if _ok(r.get("latency_ms_mean")) and r["latency_ms_mean"] > 0]
    ens = [x for x in (energy_per_request(r, volume) for r in ref) if _ok(x) and x > 0]
    thr = [r["throughput_per_s"] for r in ref if _ok(r.get("throughput_per_s")) and r["throughput_per_s"] > 0]
    c = {}
    if _ok(m.get("macro_f1")) and best_f1:
        c["performance"] = max(0.0, min(1.0, m["macro_f1"] / best_f1))
    if _ok(m.get("ece")):
        c["calibration"] = max(0.0, min(1.0, 1 - m["ece"]))
    if lats:
        c["latency"] = _log_ratio_score(m.get("latency_ms_mean"), min(lats))
    if ens:
        c["energy"] = _log_ratio_score(energy_per_request(m, volume), min(ens))
    if thr:
        c["throughput"] = _log_ratio_score(m.get("throughput_per_s"), max(thr), lower_is_better=False)
    return {k: v for k, v in c.items() if v is not None}


def total(comps, weights):
    """Weighted geometric mean of the available components (0-1); None if nothing to score."""
    w = {k: weights.get(k, 0.0) for k in comps if weights.get(k, 0.0) > 0}
    s = sum(w.values())
    if not s:
        return None
    return math.exp(sum(w[k] / s * math.log(max(comps[k], FLOOR)) for k in w))


def contributions(comps, weights):
    """How many 'score points' each component costs the run: log-share of the gap to a perfect 1.0."""
    w = {k: weights.get(k, 0.0) for k in comps if weights.get(k, 0.0) > 0}
    s = sum(w.values()) or 1.0
    return {k: w[k] / s * -math.log(max(comps[k], FLOOR)) for k in w}


def rank(runs_metrics, weights, volume=DEFAULT_VOLUME):
    """[(run_id, score, components)] for a dict {run_id: metrics}, best first."""
    ms = list(runs_metrics.values())
    out = []
    for rid, m in runs_metrics.items():
        comps = components(m, ms, volume)
        out.append((rid, total(comps, weights), comps))
    return sorted(out, key=lambda r: -1 if r[1] is None else r[1], reverse=True)

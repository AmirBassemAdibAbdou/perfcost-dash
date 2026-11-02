"""Performance and cost metrics computed from a run's events and cost records (numpy/pandas only)."""
import math

import numpy as np
import pandas as pd

INVALID = "__invalid__"


def per_class_f1(y_true, y_pred, labels):
    """F1 per class; a predicted label outside `labels` (e.g. an invalid answer) only counts as a miss."""
    y_true, y_pred = np.asarray(y_true, dtype=object), np.asarray(y_pred, dtype=object)
    out = {}
    for c in labels:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        out[c] = 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
    return out


def macro_f1(y_true, y_pred, labels):
    f = per_class_f1(y_true, y_pred, labels)
    return float(np.mean(list(f.values()))) if f else float("nan")


def ece(confidence, correct, n_bins=15):
    """Top-label expected calibration error (0 = confidence matches accuracy exactly)."""
    conf = np.asarray(confidence, dtype=float)
    corr = np.asarray(correct, dtype=bool)
    ok = ~np.isnan(conf)
    if ok.sum() == 0:
        return float("nan")
    conf, corr = conf[ok], corr[ok]
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return float(total)


def reliability(confidence, correct, n_bins=10):
    conf = np.asarray(confidence, dtype=float)
    corr = np.asarray(correct, dtype=bool)
    ok = ~np.isnan(conf)
    conf, corr = conf[ok], corr[ok]
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            rows.append(dict(bin_mid=(lo + hi) / 2, confidence=conf[m].mean(), accuracy=corr[m].mean(), n=int(m.sum())))
    return pd.DataFrame(rows)


def labels_of(events, declared=None):
    if declared:
        return [str(x) for x in declared]
    return sorted(set(events["y_true"]))


def performance(events, labels=None):
    """Summary performance metrics of all predictions logged so far."""
    if events.empty:
        return {}
    labels = labels_of(events, labels)
    correct = (events["y_pred"] == events["y_true"]).to_numpy()
    return dict(
        n=len(events),
        accuracy=float(correct.mean()),
        macro_f1=macro_f1(events["y_true"], events["y_pred"], labels),
        ece=ece(events["confidence"], correct),
        invalid_rate=float((events["y_pred"] == INVALID).mean() + (~events["y_pred"].isin(labels + [INVALID])).mean()),
        majority_baseline=float(events["y_true"].value_counts(normalize=True).iloc[0]),
    )


def running_curves(events, labels=None, points=120):
    """Accuracy and macro-F1 after every k-th prediction (for the live curves)."""
    if events.empty:
        return pd.DataFrame(columns=["n", "accuracy", "macro_f1"])
    labels = labels_of(events, labels)
    n = len(events)
    marks = sorted(set(np.linspace(1, n, min(points, n)).astype(int)))
    y, p = events["y_true"].to_numpy(object), events["y_pred"].to_numpy(object)
    corr = np.cumsum(y == p)
    rows = [dict(n=k, accuracy=corr[k - 1] / k, macro_f1=macro_f1(y[:k], p[:k], labels)) for k in marks]
    return pd.DataFrame(rows)


def confusion(events, labels=None):
    labels = labels_of(events, labels)
    cols = labels + ([INVALID] if (events["y_pred"] == INVALID).any() else [])
    m = pd.crosstab(pd.Categorical(events["y_true"], categories=labels),
                    pd.Categorical(events["y_pred"], categories=cols), dropna=False)
    m.index.name, m.columns.name = "true", "predicted"
    return m


def timing(events):
    """Latency statistics and throughput from the per-prediction latencies and timestamps."""
    if events.empty:
        return {}
    lat = events["latency_ms"].dropna()
    t = events["time"].astype(float)
    span = float(t.max() - t.min()) if len(t) > 1 else float("nan")
    out = dict(
        latency_ms_mean=float(lat.mean()) if len(lat) else float("nan"),
        latency_ms_p50=float(lat.quantile(0.5)) if len(lat) else float("nan"),
        latency_ms_p95=float(lat.quantile(0.95)) if len(lat) else float("nan"),
        elapsed_s=span,
    )
    if len(t) > 1 and span > 0:
        out["throughput_per_s"] = (len(t) - 1) / span
    elif len(lat):
        out["throughput_per_s"] = 1000.0 / lat.mean() if lat.mean() > 0 else float("nan")
    else:
        out["throughput_per_s"] = float("nan")
    return out


def costs(cost_df):
    """Aggregate the cost records: energy per 1,000 predictions, training cost, best measured throughput."""
    out = {}
    if cost_df is None or cost_df.empty:
        return out
    df = cost_df.copy()
    for c in ("energy_kwh", "n", "wall_time_s", "throughput_per_s", "latency_ms_mean", "power_w"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    kind = df["kind"] if "kind" in df else pd.Series(["inference"] * len(df))
    inf = df[(kind == "inference") & df.get("energy_kwh", pd.Series(dtype=float)).notna()] if "energy_kwh" in df else df.iloc[0:0]
    if "n" in inf:
        inf = inf[inf["n"] > 0]
    if len(inf):
        mode = inf["mode"] if "mode" in inf else pd.Series([None] * len(inf), index=inf.index)
        use = inf[mode == "online"] if (mode == "online").any() else inf
        out["energy_per_1000_kwh"] = float((use["energy_kwh"] / use["n"] * 1000).mean())
        if (mode == "batch").any():
            b = inf[mode == "batch"]
            out["batch_energy_per_1000_kwh"] = float((b["energy_kwh"] / b["n"] * 1000).mean())
    if "throughput_per_s" in df and df["throughput_per_s"].notna().any():
        out["best_throughput_per_s"] = float(df["throughput_per_s"].max())
    if "latency_ms_mean" in df:
        lm = df[(kind == "inference") & df["latency_ms_mean"].notna()]
        if len(lm):
            out["cost_latency_ms_mean"] = float(lm["latency_ms_mean"].mean())
    tr = df[kind == "training"]
    if len(tr):
        if "energy_kwh" in tr:
            out["train_energy_kwh"] = float(tr["energy_kwh"].mean())
        if "wall_time_s" in tr:
            out["train_time_s"] = float(tr["wall_time_s"].mean())
    cs = df[kind == "cold_start"]
    if len(cs) and "wall_time_s" in cs:
        out["cold_start_s"] = float(cs["wall_time_s"].mean())
    return out


def live_energy(cost_df):
    """Cumulative energy over time from a built-in tracker (kind == 'live_energy')."""
    if cost_df is None or cost_df.empty or "kind" not in cost_df:
        return pd.DataFrame(columns=["t", "energy_kwh_cum", "power_w"])
    d = cost_df[cost_df["kind"] == "live_energy"].copy()
    if d.empty:
        return pd.DataFrame(columns=["t", "energy_kwh_cum", "power_w"])
    d = d.sort_values("t")
    d["energy_kwh_cum"] = pd.to_numeric(d["energy_kwh_cum"], errors="coerce")
    dt = d["t"].diff()
    d["power_w"] = d["energy_kwh_cum"].diff() * 3.6e6 / dt
    return d[["t", "energy_kwh_cum", "power_w"]]


def all_metrics(events, cost_df, labels=None):
    """Everything the score needs, in one flat dict."""
    m = {}
    m.update(performance(events, labels))
    m.update(timing(events))
    m.update(costs(cost_df))
    lat = m.get("latency_ms_mean")
    if (lat is None or (isinstance(lat, float) and math.isnan(lat))) and "cost_latency_ms_mean" in m:
        m["latency_ms_mean"] = m["cost_latency_ms_mean"]          # latency measured outside the logged predictions
    if "best_throughput_per_s" in m and not math.isnan(m["best_throughput_per_s"]):
        m["throughput_per_s"] = max(m.get("throughput_per_s") or 0, m["best_throughput_per_s"])
    return m

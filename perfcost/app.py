"""The perfcost dashboard (Streamlit). Started by perfcost.start() / perfcost.serve(); not imported directly.

    streamlit run app.py -- --root perfcost_runs
"""
import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # works from a source checkout too

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from perfcost import metrics as M  # noqa: E402
from perfcost import score as S  # noqa: E402
from perfcost import store  # noqa: E402

# categorical palette (fixed order) and neutral inks
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                            "#e87ba4", "#008300", "#6250d6", "#e34948")
GREY, INK2 = "#8a8984", "#52514e"
COMP_LABEL = {"performance": "Performance", "calibration": "Calibration", "latency": "Latency",
              "energy": "Energy", "throughput": "Throughput"}
COMP_HELP = {"performance": "macro-F1 / best macro-F1", "calibration": "1 - ECE",
             "latency": "mean latency vs fastest run (log)", "energy": "energy per request incl. training/N vs cheapest run (log)",
             "throughput": "predictions per second vs fastest run (log)"}
VOLUMES = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000, 1_000_000_000]


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(store.DEFAULT_ROOT))
    a, _ = p.parse_known_args(sys.argv[1:])
    return Path(a.root)


ROOT = args()
st.set_page_config(page_title="perfcost", page_icon="📊", layout="wide")


# ----------------------------------------------------------------------------- data (cached by file size)
def _sig(path):
    p = Path(path)
    return tuple((p / f).stat().st_size if (p / f).exists() else 0 for f in ("events.jsonl", "costs.jsonl", "meta.json"))


@st.cache_data(show_spinner=False, max_entries=256)
def load(path, sig):
    ev, co = store.load_events(path), store.load_costs(path)
    return ev, co


@st.cache_data(show_spinner=False, max_entries=256)
def run_metrics(path, sig, labels):
    ev, co = load(path, sig)
    return M.all_metrics(ev, co, list(labels) if labels else None)


def metrics_of(meta):
    return run_metrics(meta["path"], _sig(meta["path"]), tuple(meta.get("labels") or ()))


def fmt(v, unit="", digits=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if isinstance(v, float):
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
            return f"{v:.2e}{unit}"
        return f"{v:.{digits}g}{unit}"
    return f"{v}{unit}"


def fmt_dur(s):
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return "–"
    s = int(s)
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m:02d}m" if h else (f"{m}m {s:02d}s" if m else f"{s}s")


def dataset_key(meta):
    return (meta.get("dataset") or {}).get("name") or "(no dataset name)"


def card(title, d, keys=None):
    with st.container(border=True):
        st.markdown(f"**{title}**")
        items = [(k, d[k]) for k in (keys or d) if k in d and d[k] not in (None, "", [])]
        if not items:
            st.caption("not given")
        lines = []
        for k, v in items:
            if isinstance(v, (list, tuple)):
                v = f"{len(v)} classes" if k in ("classes", "labels") and len(v) > 6 else ", ".join(map(str, v))
            elif isinstance(v, int) and v >= 10000:
                v = f"{v:,}"
            lines.append(f"<span style='color:{INK2}'>{k.replace('_', ' ')}</span>&nbsp; {v}")
        st.markdown("<div style='line-height:1.9'>" + "<br>".join(lines) + "</div>", unsafe_allow_html=True)


def axis_clean(chart):
    return chart.configure_axis(grid=True, gridOpacity=0.25, domainColor=GREY, tickColor=GREY,
                                labelColor=INK2, titleColor=INK2).configure_view(strokeWidth=0)


# ----------------------------------------------------------------------------- sidebar
runs = store.list_runs(ROOT)
st.sidebar.title("📊 perfcost")
st.sidebar.caption(f"runs folder: `{ROOT}`")
if not runs:
    st.info("No runs yet. In a notebook: `run = perfcost.start(name=..., dataset=..., model=...)`, then "
            "`run.log(y_true, y_pred, confidence, latency_ms)` for every prediction.")
    time.sleep(3)
    st.rerun()

follow = st.sidebar.toggle("Follow the newest running run", value=True,
                           help="Automatically switch to a run when it starts.")
labels_for = {m["run_id"]: f"{'🟢' if m.get('status') == 'running' else '⚪' if m.get('status') == 'finished' else '🔴'} "
                           f"{m.get('name')}  ·  {datetime.fromtimestamp(m.get('started_at', 0)):%d %b %H:%M}"
              for m in runs}
ids = [m["run_id"] for m in runs]
status_at_load = {m["run_id"]: m.get("status") for m in runs}
running = [m["run_id"] for m in runs if m.get("status") == "running"]
if follow and running:
    st.session_state["run_id"] = running[0]
if st.session_state.get("run_id") not in ids:
    st.session_state["run_id"] = running[0] if running else ids[0]
run_id = st.sidebar.selectbox("Run", ids, format_func=labels_for.get, key="run_id")

st.sidebar.markdown("### Score weights")
preset = st.sidebar.selectbox("Preset", list(S.PRESETS), index=list(S.PRESETS).index("balanced"),
                              help="accuracy-first: a mistake is expensive; balanced; green: high volume, energy matters most")
weights = {}
with st.sidebar.expander("Adjust weights", expanded=False):
    for c in S.COMPONENTS:
        weights[c] = st.slider(COMP_LABEL[c], 0.0, 1.0, float(S.PRESETS[preset][c]), 0.05, key=f"w_{preset}_{c}")
volume = st.sidebar.select_slider("Expected requests over the model's life (N)", VOLUMES, value=S.DEFAULT_VOLUME,
                                  format_func=lambda v: f"{v:,}",
                                  help="Training energy is spread over N requests in the energy score.")
refresh = st.sidebar.select_slider("Refresh every", [2, 5, 10, 30, 60], value=5, format_func=lambda v: f"{v} s")
paused = st.sidebar.toggle("Pause refreshing", value=False)
st.sidebar.caption("Score = weighted geometric mean of 0-1 component scores, relative to the other runs on the "
                   "same dataset. Costs on a log scale: each 10x gap to the best run halves, thirds, ... the "
                   "component (1x → 1.00, 10x → 0.50, 100x → 0.33).")


# ----------------------------------------------------------------------------- main body
@st.fragment(run_every=None if paused else refresh)
def body():
    all_runs = store.list_runs(ROOT)
    metas = {m["run_id"]: m for m in all_runs}
    newest_running = next((m["run_id"] for m in all_runs if m.get("status") == "running"), None)
    if follow and newest_running and newest_running != run_id:
        st.rerun(scope="app")                       # a new run started: switch to it
    if set(metas) != set(ids) or any(mm.get("status") != status_at_load.get(rid) for rid, mm in metas.items()):
        st.rerun(scope="app")                       # runs added/removed: refresh the sidebar list
    meta = metas.get(run_id)
    if meta is None:
        st.warning("This run no longer exists.")
        return
    sig = _sig(meta["path"])
    ev, co = load(meta["path"], sig)
    labels = meta.get("labels")
    m = metrics_of(meta)
    peers = {rid: mm for rid, mm in metas.items() if dataset_key(mm) == dataset_key(meta)}
    peer_metrics = {rid: metrics_of(mm) for rid, mm in peers.items()}

    # ------------------------------------------------ header
    status = meta.get("status", "?")
    badge = {"running": "🟢 running", "finished": "⚪ finished", "failed": "🔴 failed", "stopped": "🔴 stopped"}.get(status, status)
    st.markdown(f"## {meta.get('name')}  <span style='font-size:0.55em;color:{INK2}'>{badge}</span>",
                unsafe_allow_html=True)
    n, total_n = len(ev), meta.get("expected_n")
    elapsed = (meta.get("finished_at") or time.time()) - meta.get("started_at", time.time())
    if total_n:
        frac = min(1.0, n / total_n)
        eta = (elapsed / n * (total_n - n)) if n and status == "running" else None
        st.progress(frac, text=f"{n:,} / {total_n:,} predictions · elapsed {fmt_dur(elapsed)}"
                               + (f" · about {fmt_dur(eta)} left" if eta else ""))
    else:
        st.caption(f"{n:,} predictions · elapsed {fmt_dur(elapsed)}")

    c1, c2, c3 = st.columns(3)
    with c1:
        card("Dataset", meta.get("dataset") or {})
    with c2:
        card("Model", meta.get("model") or {})
    with c3:
        card("Hardware", meta.get("hardware") or {}, ["cpu_model", "usable_cores", "cpu_count", "ram_gb", "platform"])

    # ------------------------------------------------ score
    comps = S.components(m, list(peer_metrics.values()), volume)
    sc = S.total(comps, weights)
    ranking = S.rank(peer_metrics, weights, volume)
    pos = next((i + 1 for i, r in enumerate(ranking) if r[0] == run_id), None)
    st.markdown("### Cost–performance score")
    s1, s2, s3 = st.columns([1, 2, 2])
    with s1:
        st.metric("Score (0–100)", "–" if sc is None else f"{sc * 100:.0f}")
        st.caption(f"Rank {pos} of {len(ranking)} runs on **{dataset_key(meta)}** · preset **{preset}** · "
                   f"N = {volume:,}")
        if len(ranking) == 1:
            st.caption("Only one run on this dataset so far: every component is relative to the best run, "
                       "so a single run scores 100 on the relative parts.")
    with s2:
        if comps:
            cd = pd.DataFrame([{"component": COMP_LABEL[k], "score": v, "weight": weights.get(k, 0),
                                "meaning": COMP_HELP[k]} for k, v in comps.items()])
            bars = alt.Chart(cd).mark_bar(cornerRadiusEnd=4, height=16, color=BLUE).encode(
                x=alt.X("score:Q", scale=alt.Scale(domain=[0, 1]), title="component score (1 = best run)"),
                y=alt.Y("component:N", sort=[COMP_LABEL[c] for c in S.COMPONENTS], title=None,
                        axis=alt.Axis(labelLimit=200, labelOverlap=False)),
                tooltip=["component", "meaning", alt.Tooltip("score:Q", format=".2f"), alt.Tooltip("weight:Q", format=".2f")])
            text = bars.mark_text(align="left", dx=4, color=INK2).encode(text=alt.Text("score:Q", format=".2f"))
            st.altair_chart(axis_clean((bars + text).properties(height=34 * len(cd) + 20)), use_container_width=True)
        else:
            st.caption("No components yet: log predictions (and costs) first.")
    with s3:
        rt = pd.DataFrame([{"": "▶" if rid == run_id else "", "run": metas[rid].get("name"),
                            "score": None if s is None else round(s * 100), "status": metas[rid].get("status")}
                           for rid, s, _ in ranking])
        st.dataframe(rt, hide_index=True, use_container_width=True, height=min(35 * (len(rt) + 1) + 3, 250))

    # ------------------------------------------------ tabs
    t_perf, t_cost, t_cmp, t_raw = st.tabs(["Performance", "Cost", "Compare runs", "Raw data"])

    with t_perf:
        if ev.empty:
            st.info("Waiting for the first predictions…")
        else:
            k = st.columns(5)
            k[0].metric("Accuracy", fmt(m.get("accuracy")))
            k[1].metric("Macro-F1", fmt(m.get("macro_f1")))
            k[2].metric("ECE", fmt(m.get("ece")), help="Expected calibration error: 0 = confidence matches accuracy")
            k[3].metric("Invalid answers", f"{m.get('invalid_rate', 0) * 100:.1f}%")
            k[4].metric("Majority-class baseline", fmt(m.get("majority_baseline")),
                        help="Accuracy of always predicting the most common class")
            curves = M.running_curves(ev, labels).melt("n", var_name="metric", value_name="value")
            line = alt.Chart(curves).mark_line(strokeWidth=2).encode(
                x=alt.X("n:Q", title="predictions so far"), y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)),
                color=alt.Color("metric:N", scale=alt.Scale(domain=["accuracy", "macro_f1"], range=[BLUE, ORANGE]),
                                legend=alt.Legend(orient="top", title=None)),
                tooltip=["n", "metric", alt.Tooltip("value:Q", format=".3f")])
            base = alt.Chart(pd.DataFrame({"y": [m.get("majority_baseline")]})).mark_rule(
                strokeDash=[4, 4], color=GREY).encode(y="y:Q")
            a, b = st.columns(2)
            with a:
                st.markdown("**Running accuracy and macro-F1** (dashed: majority baseline)")
                st.altair_chart(axis_clean((line + base).properties(height=260)), use_container_width=True)
            with b:
                rel = M.reliability(ev["confidence"], (ev["y_pred"] == ev["y_true"]).to_numpy())
                st.markdown("**Reliability diagram** (on the diagonal = well calibrated)")
                if rel.empty:
                    st.caption("No confidence scores logged.")
                else:
                    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
                        color=GREY, strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
                    pts = alt.Chart(rel).mark_line(point=alt.OverlayMarkDef(size=60), color=AQUA, strokeWidth=2).encode(
                        x=alt.X("confidence:Q", scale=alt.Scale(domain=[0, 1]), title="mean confidence"),
                        y=alt.Y("accuracy:Q", scale=alt.Scale(domain=[0, 1]), title="accuracy"),
                        tooltip=[alt.Tooltip("confidence:Q", format=".2f"), alt.Tooltip("accuracy:Q", format=".2f"), "n"])
                    st.altair_chart(axis_clean((diag + pts).properties(height=260)), use_container_width=True)
            labs = M.labels_of(ev, labels)
            f1s = pd.DataFrame({"class": list(M.per_class_f1(ev["y_true"], ev["y_pred"], labs).keys()),
                                "f1": list(M.per_class_f1(ev["y_true"], ev["y_pred"], labs).values())})
            f1s["support"] = f1s["class"].map(ev["y_true"].value_counts()).fillna(0).astype(int)
            a, b = st.columns(2)
            with a:
                shown = f1s.sort_values("f1")
                title = "**F1 per class**"
                if len(shown) > 30:
                    shown = pd.concat([shown.head(15), shown.tail(15)])
                    title += " (15 weakest and 15 strongest)"
                st.markdown(title)
                st.altair_chart(axis_clean(alt.Chart(shown).mark_bar(cornerRadiusEnd=3, color=BLUE).encode(
                    x=alt.X("f1:Q", scale=alt.Scale(domain=[0, 1]), title="F1"),
                    y=alt.Y("class:N", sort="x", title=None, axis=alt.Axis(labelOverlap=False, labelLimit=220)),
                    tooltip=["class", alt.Tooltip("f1:Q", format=".3f"), "support"]).properties(
                    height=max(200, 26 * len(shown)))), use_container_width=True)
            with b:
                if len(labs) <= 20:
                    cm = M.confusion(ev, labels)
                    cm.index, cm.columns = cm.index.astype(str), cm.columns.astype(str)
                    cm = cm.reset_index().melt("true", var_name="predicted", value_name="count")
                    cm["count"] = cm["count"].astype(int)
                    st.markdown("**Confusion matrix**")
                    heat = alt.Chart(cm).mark_rect(stroke="white", strokeWidth=2).encode(
                        x=alt.X("predicted:N", title="predicted"), y=alt.Y("true:N", title="true"),
                        color=alt.Color("count:Q", scale=alt.Scale(type="linear", range=["#f0efec", BLUE], interpolate="rgb"), legend=None),
                        tooltip=["true", "predicted", "count"])
                    cm["dark"] = cm["count"] > cm["count"].max() / 2
                    txt = alt.Chart(cm).mark_text().encode(x="predicted:N", y="true:N", text="count:Q",
                                                           color=alt.condition("datum.dark", alt.value("white"), alt.value(INK2)))
                    st.altair_chart(axis_clean((heat + txt).properties(height=max(200, 34 * len(labs)))),
                                    use_container_width=True)
                else:
                    conf = ev[ev["y_pred"] != ev["y_true"]].groupby(["y_true", "y_pred"]).size().nlargest(15)
                    st.markdown("**Most frequent confusions**")
                    st.dataframe(conf.rename("count").reset_index().rename(columns={"y_true": "true", "y_pred": "predicted"}),
                                 hide_index=True, use_container_width=True)

    with t_cost:
        k = st.columns(6)
        k[0].metric("Latency, mean", fmt(m.get("latency_ms_mean"), " ms"))
        k[1].metric("Latency, p95", fmt(m.get("latency_ms_p95"), " ms"))
        k[2].metric("Throughput", fmt(m.get("throughput_per_s"), "/s"))
        k[3].metric("Energy / 1k pred. (kWh)", fmt(m.get("energy_per_1000_kwh")))
        k[4].metric("Training energy (kWh)", fmt(m.get("train_energy_kwh")),
                    help=f"training time {fmt_dur(m.get('train_time_s'))}" if m.get("train_time_s") else None)
        k[5].metric("Cold start", fmt_dur(m.get("cold_start_s")) if m.get("cold_start_s") else "–")
        if ev.empty:
            st.info("Waiting for the first predictions…")
        else:
            lat = ev.dropna(subset=["latency_ms"])
            a, b = st.columns(2)
            with a:
                st.markdown("**Latency over the run** (points: each prediction; line: rolling median)")
                if lat.empty:
                    st.caption("No latencies logged.")
                else:
                    d = lat[["index", "latency_ms"]].copy()
                    d["rolling_median"] = d["latency_ms"].rolling(max(5, len(d) // 50), min_periods=1).median()
                    if len(d) > 3000:
                        d = d.iloc[:: len(d) // 3000]
                    pts = alt.Chart(d).mark_circle(size=10, opacity=0.25, color=BLUE).encode(
                        x=alt.X("index:Q", title="prediction #"), y=alt.Y("latency_ms:Q", title="ms"))
                    rl = alt.Chart(d).mark_line(color=ORANGE, strokeWidth=2).encode(x="index:Q", y="rolling_median:Q")
                    st.altair_chart(axis_clean((pts + rl).properties(height=260)), use_container_width=True)
            with b:
                st.markdown("**Latency distribution**")
                if not lat.empty:
                    st.altair_chart(axis_clean(alt.Chart(lat).mark_bar(color=BLUE, cornerRadiusTopLeft=3,
                                                                        cornerRadiusTopRight=3).encode(
                        x=alt.X("latency_ms:Q", bin=alt.Bin(maxbins=40), title="ms"),
                        y=alt.Y("count():Q", title="predictions")).properties(height=260)), use_container_width=True)
        le = M.live_energy(co)
        if len(le) > 1:
            le = le.assign(minutes=(le["t"] - le["t"].min()) / 60)
            a, b = st.columns(2)
            with a:
                st.markdown("**Cumulative energy** (built-in tracker)")
                st.altair_chart(axis_clean(alt.Chart(le).mark_line(color=AQUA, strokeWidth=2).encode(
                    x=alt.X("minutes:Q"), y=alt.Y("energy_kwh_cum:Q", title="kWh"))).properties(height=220),
                                use_container_width=True)
            with b:
                st.markdown("**Power draw**")
                st.altair_chart(axis_clean(alt.Chart(le.dropna()).mark_line(color=YELLOW, strokeWidth=2).encode(
                    x=alt.X("minutes:Q"), y=alt.Y("power_w:Q", title="W"))).properties(height=220),
                                use_container_width=True)
        if not co.empty:
            st.markdown("**Cost records**")
            shown = co[co.get("kind", pd.Series(dtype=str)) != "live_energy"] if "kind" in co else co
            st.dataframe(shown.drop(columns=["t"], errors="ignore"), hide_index=True, use_container_width=True)

    with t_cmp:
        rows = []
        for rid, mm in peer_metrics.items():
            e = S.energy_per_request(mm, volume)
            rows.append(dict(run=metas[rid].get("name"), current=rid == run_id, macro_f1=mm.get("macro_f1"),
                             energy_per_request_kwh=e, latency_ms=mm.get("latency_ms_mean"),
                             strategy=(metas[rid].get("model") or {}).get("strategy", "–")))
        cmp = pd.DataFrame(rows)
        xcol, xtitle = ("energy_per_request_kwh", f"energy per request incl. training / N (kWh, log), N = {volume:,}") \
            if cmp["energy_per_request_kwh"].notna().any() else ("latency_ms", "mean latency (ms, log)")
        d = cmp.dropna(subset=[xcol, "macro_f1"])
        d = d[d[xcol] > 0]
        st.markdown(f"**Cost vs performance on {dataset_key(meta)}** (current run highlighted)")
        if d.empty:
            st.caption("Not enough data yet.")
        else:
            pts = alt.Chart(d).mark_point(filled=True, size=90, stroke="white", strokeWidth=1.5).encode(
                x=alt.X(f"{xcol}:Q", scale=alt.Scale(type="log"), title=xtitle,
                        axis=alt.Axis(format=".0e", tickCount=6)),
                y=alt.Y("macro_f1:Q", title="macro-F1", scale=alt.Scale(zero=False)),
                color=alt.condition("datum.current", alt.value(ORANGE), alt.value(BLUE)),
                size=alt.condition("datum.current", alt.value(260), alt.value(90)),
                tooltip=["run", "strategy", alt.Tooltip("macro_f1:Q", format=".3f"),
                         alt.Tooltip(f"{xcol}:Q", format=".2e")])
            lab = alt.Chart(d).mark_text(align="left", dx=8, dy=-6, fontSize=11, color=INK2).encode(
                x=f"{xcol}:Q", y="macro_f1:Q", text="run:N")
            st.altair_chart(axis_clean((pts + lab).properties(height=380)), use_container_width=True)
        st.markdown("**Ranking under every preset** (does the conclusion depend on the weights?)")
        tab = {}
        for p, w in S.PRESETS.items():
            for rid, s, _ in S.rank(peer_metrics, w, volume):
                tab.setdefault(rid, {"run": metas[rid].get("name")})[p] = None if s is None else round(s * 100)
        for rid, s, _ in ranking:
            tab[rid]["your weights"] = None if s is None else round(s * 100)
        st.dataframe(pd.DataFrame(tab.values()).sort_values("your weights", ascending=False),
                     hide_index=True, use_container_width=True)

    with t_raw:
        st.markdown("**Latest predictions**")
        st.dataframe(ev.tail(25).iloc[::-1], hide_index=True, use_container_width=True)
        st.markdown("**All metrics**")
        st.json({k: v for k, v in m.items()}, expanded=False)
        st.markdown("**Run metadata**")
        st.json({k: v for k, v in meta.items() if k != "path"}, expanded=False)
        st.caption(f"folder: `{meta['path']}`")


body()

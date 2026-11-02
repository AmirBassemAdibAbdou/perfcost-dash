import numpy as np

import perfcost
from perfcost import metrics as M
from perfcost import store


def test_macro_f1_and_invalid():
    y = ["a", "a", "b", "b"]
    p = ["a", "__invalid__", "b", "a"]
    f = M.per_class_f1(y, p, ["a", "b"])
    assert np.isclose(f["a"], 0.5) and np.isclose(f["b"], 2 / 3)


def test_ece_perfect_and_bad():
    assert M.ece([1.0, 1.0], [True, True]) == 0.0
    assert np.isclose(M.ece([1.0, 1.0], [False, False]), 1.0)


def test_roundtrip(tmp_path):
    run = perfcost.start("t", dataset={"name": "D", "n_test": 4}, model={"name": "M"}, labels=["x", "y"],
                         root=tmp_path, launch=False)
    for yt, yp, c, l in [("x", "x", 0.9, 10), ("y", "x", 0.6, 12), ("y", "y", 0.8, 11), ("x", None, None, 50)]:
        run.log(yt, yp, c, l)
    run.log_cost(kind="inference", mode="online", energy_kwh=0.002, n=4)
    run.log_cost(kind="training", energy_kwh=1.0, wall_time_s=60)
    run.finish()
    meta = store.list_runs(tmp_path)[0]
    assert meta["status"] == "finished"
    ev, co = store.load_events(meta["path"]), store.load_costs(meta["path"])
    m = M.all_metrics(ev, co, ["x", "y"])
    assert m["n"] == 4 and np.isclose(m["accuracy"], 0.5) and np.isclose(m["invalid_rate"], 0.25)
    assert np.isclose(m["energy_per_1000_kwh"], 0.5) and m["train_energy_kwh"] == 1.0


def test_import_run(tmp_path):
    perfcost.import_run("imp", ["a", "b"], ["a", "a"], [0.9, 0.7], [5, 5], dataset={"name": "D"}, root=tmp_path)
    assert store.list_runs(tmp_path)[0]["status"] == "finished"

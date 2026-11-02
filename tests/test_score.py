import math

from perfcost import score as S


def m(f1, ece, lat, e1000, thr, train=0.0):
    return dict(macro_f1=f1, ece=ece, latency_ms_mean=lat, energy_per_1000_kwh=e1000, throughput_per_s=thr,
                train_energy_kwh=train)


def test_tenfold_cost_halves_component():
    a, b = m(0.9, 0.0, 1.0, 1.0, 100), m(0.9, 0.0, 10.0, 10.0, 10)
    c = S.components(b, [a, b])
    assert math.isclose(c["latency"], 0.5) and math.isclose(c["energy"], 0.5) and math.isclose(c["throughput"], 0.5)
    assert S.components(a, [a, b])["latency"] == 1.0


def test_best_on_everything_scores_100():
    a, b = m(0.9, 0.0, 1.0, 1.0, 100), m(0.8, 0.1, 10.0, 10.0, 10)
    assert math.isclose(S.total(S.components(a, [a, b]), S.PRESETS["balanced"]), 1.0)
    assert S.total(S.components(b, [a, b]), S.PRESETS["balanced"]) < 1.0


def test_training_amortised_over_volume():
    trained, prompted = m(0.9, 0.0, 1.0, 0.001, 100, train=10.0), m(0.9, 0.0, 1.0, 0.01, 100)
    small = S.components(trained, [trained, prompted], volume=1_000)["energy"]
    large = S.components(trained, [trained, prompted], volume=1_000_000_000)["energy"]
    assert small < large and math.isclose(large, 1.0, rel_tol=1e-3)


def test_missing_components_renormalise():
    c = {"performance": 1.0}
    assert S.total(c, S.PRESETS["green"]) == 1.0
    assert S.total({}, S.PRESETS["green"]) is None


def test_geometric_mean_punishes_one_bad_component():
    good = {"performance": 1.0, "energy": 1.0}
    bad = {"performance": 1.0, "energy": 0.01}
    w = {"performance": 0.5, "energy": 0.5}
    assert S.total(bad, w) < 0.5 * (1.0 + 0.01) / 1.0          # below the arithmetic mean
    assert S.total(good, w) == 1.0

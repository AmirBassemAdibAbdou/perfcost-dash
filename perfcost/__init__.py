"""perfcost: rate a classifier run on performance AND cost, live, in a dashboard.

    import perfcost
    run = perfcost.start("adult / xgboost", dataset={"name": "Adult Income", "n_test": 16281},
                         model={"name": "XGBoost", "strategy": "trained"}, labels=["<=50K", ">50K"])
    for x, y in test_set:
        ...
        run.log(y_true=y, y_pred=pred, confidence=conf, latency_ms=ms)
    run.log_cost(kind="inference", mode="online", energy_kwh=0.0004, n=1000)
    run.finish()
"""
from .launcher import browse, serve, stop_server
from .logger import Run, import_run, start
from .score import PRESETS

__version__ = "0.1.0"
__all__ = ["start", "import_run", "Run", "serve", "browse", "stop_server", "PRESETS"]

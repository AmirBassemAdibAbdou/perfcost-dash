"""Command line:  perfcost serve | stop | demo  [--root DIR] [--port 8501] [--cpus 32-63]"""
import argparse
import sys

from .launcher import serve, show_link, stop_server


def main(argv=None):
    p = argparse.ArgumentParser(prog="perfcost", description="cost-performance dashboard for classifier runs")
    p.add_argument("command", choices=["serve", "stop", "demo"])
    p.add_argument("--root", default=None, help="folder with the runs (default: ./perfcost_runs or $PERFCOST_ROOT)")
    p.add_argument("--port", type=int, default=8501)
    p.add_argument("--cpus", default=None, help="pin the dashboard to these cores, e.g. 32-63")
    a = p.parse_args(argv)
    if a.command == "serve":
        show_link(serve(root=a.root, port=a.port, cpus=a.cpus))
    elif a.command == "stop":
        print("stopped" if stop_server(a.root) else "no dashboard running for this folder")
    elif a.command == "demo":
        from .demo import run_demo
        run_demo(root=a.root, port=a.port, cpus=a.cpus)


if __name__ == "__main__":
    sys.exit(main())

"""Command line entry point: `dsverify run CONFIG.yaml`."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .pipeline import run


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="dsverify", description="Verify downscaled wind against station observations.")
    p.add_argument("--version", action="version", version=f"dsverify {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a verification defined in a YAML config")
    r.add_argument("config")
    args = p.parse_args(argv)
    if args.cmd == "run":
        out = run(args.config)
        print(f"done: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

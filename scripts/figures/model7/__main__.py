"""Unified command line entry point for Model7 architecture figures."""

from __future__ import annotations

import argparse
import importlib
import sys


FIGURES = {
    "architecture": "scripts.figures.model7.architecture",
    "architecture-3d": "scripts.figures.model7.architecture_3d",
    "multiview": "scripts.figures.model7.multiview_overview",
    "overall": "scripts.figures.model7.overall_process",
    "overall-pptx": "scripts.figures.model7.overall_process_pptx",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        add_help=False,
        description="Generate a Model7 architecture or RADE processing figure."
    )
    parser.add_argument("figure", choices=FIGURES)
    if len(sys.argv) == 1 or sys.argv[1] in {"-h", "--help"}:
        parser = argparse.ArgumentParser(
            description="Generate a Model7 architecture or RADE processing figure."
        )
        parser.add_argument("figure", choices=FIGURES)
        parser.print_help()
        return
    args, forwarded = parser.parse_known_args()
    module = importlib.import_module(FIGURES[args.figure])
    previous_argv = sys.argv
    try:
        sys.argv = [f"{previous_argv[0]} {args.figure}", *forwarded]
        module.main()
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()

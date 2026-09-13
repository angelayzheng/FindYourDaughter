"""Required Branchseed entrypoint and local visualization launcher."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    ui_parser = argparse.ArgumentParser(add_help=False)
    ui_parser.add_argument("--gui", action="store_true")
    ui_parser.add_argument("--webui", action="store_true")
    ui_flags, remaining = ui_parser.parse_known_args(arguments)
    if ui_flags.gui and ui_flags.webui:
        ui_parser.error("--gui and --webui cannot be used together")

    if ui_flags.webui:
        if remaining:
            ui_parser.error("--webui does not take evaluator or viewer arguments")
        return subprocess.call(
            [sys.executable, "-m", "streamlit", "run", str(ROOT / "frontend" / "app.py")],
            cwd=ROOT,
        )

    if ui_flags.gui:
        from scripts.view_nifti_3d import main as gui_main

        gui_main(remaining)
        return 0

    from backend.cli import main as backend_main
    if any(argument in ("-h", "--help") for argument in arguments):
        from backend.cli import build_parser

        parser = build_parser()
        parser.epilog = (
            "Developer UI modes: 'python run.py --gui --subject dataset/subject001' "
            "or 'python run.py --webui'."
        )
        parser.print_help()
        return 0

    return backend_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())

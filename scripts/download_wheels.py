"""Populate the local wheelhouse on an internet-connected preparation machine."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import sysconfig
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WHEELHOUSE = ROOT / "vendor" / "wheels"
REQUIREMENTS = (ROOT / "requirements-backend.txt", ROOT / "requirements-frontend.txt")


def write_manifest() -> None:
    wheels = sorted(WHEELHOUSE.glob("*.whl"))
    manifest = {
        "implementation": platform.python_implementation(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": sysconfig.get_platform(),
        "wheels": {
            wheel.name: hashlib.sha256(wheel.read_bytes()).hexdigest()
            for wheel in wheels
        },
    }
    (WHEELHOUSE / "wheelhouse.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Refresh wheelhouse metadata without downloading packages",
    )
    args = parser.parse_args()
    WHEELHOUSE.mkdir(parents=True, exist_ok=True)
    if not args.manifest_only:
        command = [sys.executable, "-m", "pip", "download", "--only-binary=:all:", "--dest", str(WHEELHOUSE)]
        for requirements in REQUIREMENTS:
            command.extend(("--requirement", str(requirements)))
        subprocess.run(command, check=True)
    write_manifest()
    print(f"Wheelhouse ready at {WHEELHOUSE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

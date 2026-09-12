"""Install all runtime dependencies without contacting a package index."""

from __future__ import annotations

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


def main() -> int:
    manifest_path = WHEELHOUSE / "wheelhouse.json"
    if not manifest_path.is_file():
        raise SystemExit(
            "The local wheelhouse is missing its manifest. On an internet-connected machine, run "
            "`python scripts/download_wheels.py` before packaging the submission."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime = {
        "implementation": platform.python_implementation(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": sysconfig.get_platform(),
    }
    expected_runtime = {key: manifest[key] for key in runtime}
    if runtime != expected_runtime:
        raise SystemExit(
            "This wheelhouse targets "
            f"{expected_runtime}, but the current interpreter is {runtime}. "
            "Rebuild it on a matching online machine before judging."
        )
    for filename, expected_hash in manifest["wheels"].items():
        wheel = WHEELHOUSE / filename
        if not wheel.is_file():
            raise SystemExit(f"Wheel listed in the manifest is missing: {filename}")
        actual_hash = hashlib.sha256(wheel.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise SystemExit(f"Wheel failed its SHA-256 check: {filename}")

    command = [sys.executable, "-m", "pip", "install", "--no-index", "--find-links", str(WHEELHOUSE)]
    for requirements in REQUIREMENTS:
        command.extend(("--requirement", str(requirements)))
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

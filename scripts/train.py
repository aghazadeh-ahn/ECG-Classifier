"""Train the experiment named on the command line."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecgcls.train_signal import load_experiment, run_experiment


def main() -> None:
    config_path = Path(sys.argv[1])
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    run_experiment(ROOT, load_experiment(config_path))


if __name__ == "__main__":
    main()

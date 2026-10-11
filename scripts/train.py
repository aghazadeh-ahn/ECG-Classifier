"""Train the experiment named on the command line."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecgcls.train_atrial import run_atrial_experiment
from ecgcls.train_beats import run_beat_experiment
from ecgcls.train_rhythm import run_rhythm_experiment
from ecgcls.train_signal import load_experiment, run_experiment


def main() -> None:
    config_path = Path(sys.argv[1])
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    cfg = load_experiment(config_path)
    name = cfg["model"]["name"]
    if name == "beat_cnn":
        run_beat_experiment(ROOT, cfg)
    elif name == "atrial_cnn_lstm":
        run_atrial_experiment(ROOT, cfg)
    elif name == "dilated_resnet_rhythm":
        run_rhythm_experiment(ROOT, cfg)
    else:
        run_experiment(ROOT, cfg)


if __name__ == "__main__":
    main()

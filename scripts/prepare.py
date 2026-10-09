"""Build the shared prepared records from the raw challenge files."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecgcls.prepare import prepare_dataset


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "prepare.yaml").read_text(encoding="utf-8"))
    raw_dir = ROOT / cfg["raw_dir"]
    processed_dir = ROOT / cfg["processed_dir"]
    prepare_dataset(raw_dir, processed_dir, cfg)


if __name__ == "__main__":
    main()

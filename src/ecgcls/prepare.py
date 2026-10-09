"""One shared preparation for every later model.

The stored arrays are filtered, inversion-corrected, normalized, and padded.
Training-time augmentation is intentionally not applied here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt
from sklearn.model_selection import StratifiedKFold, train_test_split

from ecgcls.records import Record, load_records, load_signal

CLASSES = ("N", "A", "O", "~")


def prepare_dataset(raw_dir: Path, processed_dir: Path, cfg: dict) -> dict:
    sampling_rate = int(cfg["sampling_rate"])
    target = int(round(float(cfg["length_seconds"]) * sampling_rate))
    beat_samples = int(round(float(cfg["beat_seconds"]) * sampling_rate))
    pre_samples = int(round(float(cfg["pre_seconds"]) * sampling_rate))
    if pre_samples >= beat_samples:
        raise ValueError("The atrial window must sit inside the beat window")
    records = load_records(raw_dir)
    record_dir = processed_dir / "records"
    record_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for index, record in enumerate(records):
        prepared = prepare_record(record, sampling_rate, target, beat_samples, pre_samples, cfg["bandpass"])
        stem = Path(record.record_id).name
        np.savez_compressed(record_dir / f"{stem}.npz", **prepared)
        rows.append(
            {
                "record_id": record.record_id,
                "stem": stem,
                "label": record.label,
                "duration_s": float(prepared["duration_s"]),
                "n_beats": int(prepared["n_beats"]),
                "baseline_std": float(prepared["baseline_std"]),
                "inverted": bool(prepared["inverted"]),
            }
        )
        if index % 500 == 0:
            print(f"prepared {index}/{len(records)}", flush=True)
    splits = make_splits([row["label"] for row in rows], cfg["split"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "index.json").write_text(json.dumps(rows), encoding="utf-8")
    (processed_dir / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
    summary = {
        "n_records": len(rows),
        "target_samples": target,
        "beat_samples": beat_samples,
        "atrial_samples": pre_samples,
        "sampling_rate": sampling_rate,
        "n_inverted": int(sum(row["inverted"] for row in rows)),
        "labels": {label: int(sum(row["label"] == label for row in rows)) for label in CLASSES},
        "split": {key: cfg["split"][key] for key in ("seed", "test_size", "n_folds")},
        "n_test": len(splits["test"]),
        "n_trainval": len(splits["trainval"]),
    }
    (processed_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"done records {summary['n_records']} inverted {summary['n_inverted']} "
        f"test {summary['n_test']}",
        flush=True,
    )
    return summary


def prepare_record(
    record: Record,
    sampling_rate: int,
    target: int,
    beat_samples: int,
    pre_samples: int,
    band_cfg: dict,
) -> dict:
    original, file_rate = load_signal(record)
    if file_rate != sampling_rate:
        raise ValueError(f"{record.record_id} is sampled at {file_rate} Hz")
    if original.size > target:
        raise ValueError(f"{record.record_id} is longer than the prepared length")
    baseline_std = _baseline_std(original, sampling_rate, float(band_cfg["low_hz"]))
    filtered = _bandpass(
        original,
        sampling_rate,
        low_hz=float(band_cfg["low_hz"]),
        high_hz=float(band_cfg["high_hz"]),
        order=int(band_cfg["order"]),
    )
    corrected, inverted = _correct_inversion(filtered, sampling_rate)
    normalized = _zscore(corrected)
    signal = np.zeros(target, dtype=np.float32)
    mask = np.zeros(target, dtype=np.uint8)
    signal[: normalized.size] = normalized
    mask[: normalized.size] = 1
    peaks = _r_peaks(signal[: normalized.size], sampling_rate)
    beats, atrial, rr_before = _beat_views(signal, peaks, beat_samples, pre_samples, sampling_rate)
    return {
        "signal": signal,
        "mask": mask,
        "peaks": peaks.astype(np.int32),
        "beats": beats,
        "atrial": atrial,
        "rr_before": rr_before,
        "duration_s": np.float32(original.size / sampling_rate),
        "n_beats": np.int32(peaks.size),
        "baseline_std": np.float32(baseline_std),
        "inverted": np.bool_(inverted),
        "label": np.array(record.label),
    }


def make_splits(labels: list[str], split_cfg: dict) -> dict:
    labels_array = np.asarray(labels)
    indices = np.arange(len(labels))
    trainval, test = train_test_split(
        indices,
        test_size=float(split_cfg["test_size"]),
        random_state=int(split_cfg["seed"]),
        stratify=labels_array,
    )
    folder = StratifiedKFold(
        n_splits=int(split_cfg["n_folds"]),
        shuffle=True,
        random_state=int(split_cfg["seed"]),
    )
    folds = []
    for train_pos, val_pos in folder.split(trainval, labels_array[trainval]):
        folds.append(
            {
                "train": trainval[train_pos].astype(int).tolist(),
                "val": trainval[val_pos].astype(int).tolist(),
            }
        )
    test_set = set(test.astype(int).tolist())
    trainval_set = set(trainval.astype(int).tolist())
    if test_set & trainval_set:
        raise RuntimeError("A record fell on both sides of the locked split")
    return {
        "seed": int(split_cfg["seed"]),
        "test_size": float(split_cfg["test_size"]),
        "n_folds": int(split_cfg["n_folds"]),
        "trainval": trainval.astype(int).tolist(),
        "test": test.astype(int).tolist(),
        "folds": folds,
    }


def _bandpass(signal: np.ndarray, sampling_rate: int, low_hz: float, high_hz: float, order: int) -> np.ndarray:
    nyquist = sampling_rate / 2
    low = max(low_hz, 0.05)
    high = min(high_hz, nyquist * 0.95)
    sos = butter(order, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    padlen = 3 * (2 * sos.shape[0])
    if signal.size <= padlen:
        return np.asarray(signal, dtype=np.float32)
    return sosfiltfilt(sos, signal).astype(np.float32)


def _baseline_std(signal: np.ndarray, sampling_rate: int, low_hz: float) -> float:
    nyquist = sampling_rate / 2
    sos = butter(2, max(low_hz, 0.05) / nyquist, btype="lowpass", output="sos")
    padlen = 3 * (2 * sos.shape[0])
    if signal.size <= padlen:
        return float(np.std(signal))
    baseline = sosfiltfilt(sos, signal)
    return float(np.std(baseline))


def _correct_inversion(signal: np.ndarray, sampling_rate: int) -> tuple[np.ndarray, bool]:
    if signal.size < sampling_rate:
        return signal.astype(np.float32), False
    deviation = float(np.std(signal))
    if deviation == 0:
        return signal.astype(np.float32), False
    distance = max(1, int(0.3 * sampling_rate))
    peaks, _ = find_peaks(np.abs(signal), distance=distance, prominence=0.5 * deviation)
    if peaks.size < 3:
        return signal.astype(np.float32), False
    if float(np.median(signal[peaks])) < 0:
        return (-signal).astype(np.float32), True
    return signal.astype(np.float32), False


def _zscore(signal: np.ndarray) -> np.ndarray:
    mean = float(signal.mean()) if signal.size else 0.0
    deviation = float(signal.std()) if signal.size else 1.0
    if deviation < 1e-6:
        deviation = 1.0
    return ((signal - mean) / deviation).astype(np.float32)


def _r_peaks(signal: np.ndarray, sampling_rate: int) -> np.ndarray:
    if signal.size < sampling_rate:
        return np.empty(0, dtype=np.int32)
    deviation = float(np.std(signal))
    distance = max(1, int(0.25 * sampling_rate))
    prominence = max(0.5 * deviation, 1e-6)
    peaks, _ = find_peaks(signal, distance=distance, prominence=prominence)
    return peaks.astype(np.int32)


def _beat_views(
    signal: np.ndarray,
    peaks: np.ndarray,
    beat_samples: int,
    pre_samples: int,
    sampling_rate: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    beats = np.zeros((peaks.size, beat_samples), dtype=np.float32)
    atrial = np.zeros((peaks.size, pre_samples), dtype=np.float32)
    rr_before = np.zeros(peaks.size, dtype=np.float32)
    if peaks.size > 1:
        rr_before[1:] = np.diff(peaks).astype(np.float32) / float(sampling_rate)
    for index, peak in enumerate(peaks):
        start = int(peak) - pre_samples
        stop = start + beat_samples
        valid_start = max(start, 0)
        valid_stop = min(stop, signal.size)
        destination = valid_start - start
        beats[index, destination : destination + (valid_stop - valid_start)] = signal[valid_start:valid_stop]
        atrial[index] = beats[index, :pre_samples]
    return beats, atrial, rr_before

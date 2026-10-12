"""Local RR drops: a short interval against the surrounding baseline, not a global premature count."""

from __future__ import annotations

import numpy as np

from ecgcls.prepare import _r_peaks

SAMPLING_RATE = 300
N_FEATURES = 4


def local_rr_features(signal: np.ndarray, mask: np.ndarray) -> np.ndarray:
    features = np.zeros(N_FEATURES, dtype=np.float32)
    valid = np.flatnonzero(mask)
    if valid.size < SAMPLING_RATE:
        return features
    peaks = _r_peaks(np.asarray(signal[: int(valid.max()) + 1], dtype=np.float32), SAMPLING_RATE)
    if peaks.size < 6:
        return features
    intervals = np.diff(peaks).astype(np.float64) / float(SAMPLING_RATE)
    baseline = np.empty(intervals.size, dtype=np.float64)
    window = 5
    for index in range(intervals.size):
        start = max(0, index - window)
        stop = min(intervals.size, index + window + 1)
        baseline[index] = float(np.median(intervals[start:stop]))
    drop = np.clip((baseline - intervals) / np.maximum(baseline, 1e-6), 0.0, 1.0)
    median = float(np.median(intervals))
    stable = np.abs(intervals - median) < 0.12 * median
    spikes = drop > 0.40
    alternation = np.zeros(0, dtype=bool)
    if intervals.size > 1:
        side = intervals - median
        alternation = (side[1:] * side[:-1] < 0) & (np.abs(side[1:]) > 0.15 * median) & (np.abs(side[:-1]) > 0.15 * median)
    features[0] = float(drop.max())
    features[1] = float(spikes.mean())
    features[2] = float(alternation.mean()) if alternation.size else 0.0
    features[3] = float(spikes.sum() * stable.mean() / intervals.size)
    return features

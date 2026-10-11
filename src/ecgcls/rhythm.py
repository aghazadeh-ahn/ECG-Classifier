"""Extrasystole counts and relative-RR spread, computed from the same waveform the network sees."""

from __future__ import annotations

import numpy as np

from ecgcls.prepare import _r_peaks

SAMPLING_RATE = 300
N_FEATURES = 7


def rhythm_features(signal: np.ndarray, mask: np.ndarray) -> np.ndarray:
    features = np.zeros(N_FEATURES, dtype=np.float32)
    valid = np.flatnonzero(mask)
    if valid.size < SAMPLING_RATE:
        return features
    peaks = _r_peaks(np.asarray(signal[: int(valid.max()) + 1], dtype=np.float32), SAMPLING_RATE)
    if peaks.size < 4:
        return features
    intervals = np.diff(peaks).astype(np.float64) / float(SAMPLING_RATE)
    if float(intervals.mean()) < 1e-6:
        return features
    premature = np.zeros(intervals.size, dtype=bool)
    premature[1:] = intervals[1:] < 0.8 * intervals[:-1]
    compensatory = np.zeros(intervals.size, dtype=bool)
    compensatory[1:-1] = premature[1:-1] & (intervals[2:] > 1.2 * intervals[:-2])
    doublet = premature[1:] & premature[:-1]
    triplet = premature[2:] & premature[1:-1] & premature[:-2]
    relative = np.diff(intervals) / float(intervals.mean())
    clean = relative[~(premature[1:] | premature[:-1])]
    premature_count = int(premature.sum())
    features[0] = premature_count / intervals.size
    features[1] = float(compensatory.sum() / premature_count) if premature_count else 0.0
    features[2] = float(doublet.mean()) if doublet.size else 0.0
    features[3] = float(triplet.mean()) if triplet.size else 0.0
    features[4] = float(relative.std()) if relative.size else 0.0
    features[5] = float(clean.std()) if clean.size else 0.0
    features[6] = float(intervals.std() / intervals.mean())
    return features

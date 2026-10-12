"""Boundary features for Normal/Other: artifact, QRS width, short-long RR, and local RR drops."""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

from ecgcls.prepare import _beat_views, _r_peaks

SAMPLING_RATE = 300
BEAT_SAMPLES = 180
PRE_SAMPLES = 90
N_FEATURES = 7


def boundary_features(signal: np.ndarray, mask: np.ndarray) -> np.ndarray:
    features = np.zeros(N_FEATURES, dtype=np.float32)
    valid = np.flatnonzero(mask)
    if valid.size < SAMPLING_RATE:
        return features
    prefix = np.asarray(signal[: int(valid.max()) + 1], dtype=np.float32)
    peaks = _r_peaks(prefix, SAMPLING_RATE)
    if peaks.size < 4:
        return features
    intervals = np.diff(peaks).astype(np.float64) / float(SAMPLING_RATE)
    median = float(np.median(intervals))

    # 0 artifact ratio: peak outside QRS / typical QRS amplitude
    protect = np.zeros(prefix.size, dtype=bool)
    for peak in peaks:
        protect[max(0, int(peak) - 40) : min(prefix.size, int(peak) + 40)] = True
    outside = np.abs(prefix[~protect]) if (~protect).any() else np.array([0.0], dtype=np.float32)
    inside = np.abs(prefix[protect]) if protect.any() else np.array([1.0], dtype=np.float32)
    features[0] = float(outside.max() / (np.percentile(inside, 95) + 1e-6))

    # 1 median QRS width (seconds)
    beats, _atrial, _rr = _beat_views(prefix, peaks, BEAT_SAMPLES, PRE_SAMPLES, SAMPLING_RATE)
    widths = []
    for window in beats:
        local = np.abs(window)
        thr = 0.3 * float(local.max() + 1e-6)
        above = np.where(local > thr)[0]
        if above.size:
            widths.append((above.max() - above.min()) / float(SAMPLING_RATE))
    features[1] = float(np.median(widths)) if widths else 0.0

    # 2 short-long pair fraction
    if intervals.size >= 2:
        pairs = 0
        for index in range(intervals.size - 1):
            if intervals[index] < 0.85 * median and intervals[index + 1] > 1.15 * median:
                pairs += 1
        features[2] = pairs / float(intervals.size - 1)

    # 3 high-frequency ratio in 5-15 Hz
    freqs, power = welch(prefix, fs=SAMPLING_RATE, nperseg=min(1024, max(8, prefix.size)))
    band = (freqs >= 0.5) & (freqs <= 40)
    features[3] = float(power[(freqs >= 5) & (freqs <= 15)].sum() / (power[band].sum() + 1e-12))

    # 4-6 local RR drop stats (same spirit as before, tightened)
    if peaks.size >= 6:
        baseline = np.empty(intervals.size, dtype=np.float64)
        window = 5
        for index in range(intervals.size):
            start = max(0, index - window)
            stop = min(intervals.size, index + window + 1)
            baseline[index] = float(np.median(intervals[start:stop]))
        drop = np.clip((baseline - intervals) / np.maximum(baseline, 1e-6), 0.0, 1.0)
        spikes = drop > 0.40
        features[4] = float(drop.max())
        features[5] = float(spikes.mean())
        side = intervals - median
        if intervals.size > 1:
            alternation = (side[1:] * side[:-1] < 0) & (np.abs(side[1:]) > 0.15 * median) & (np.abs(side[:-1]) > 0.15 * median)
            features[6] = float(alternation.mean())
    return features

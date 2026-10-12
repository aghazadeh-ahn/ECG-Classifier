"""The beat least like the record's own median beat."""

from __future__ import annotations

import numpy as np

from ecgcls.prepare import _beat_views, _r_peaks

SAMPLING_RATE = 300
BEAT_SAMPLES = 180
PRE_SAMPLES = 90


def outlier_beat(signal: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    beat = np.zeros(BEAT_SAMPLES, dtype=np.float32)
    valid = np.flatnonzero(mask)
    if valid.size < SAMPLING_RATE:
        return beat, 0.0
    peaks = _r_peaks(np.asarray(signal[: int(valid.max()) + 1], dtype=np.float32), SAMPLING_RATE)
    if peaks.size < 3:
        return beat, 0.0
    beats, _atrial, _rr = _beat_views(signal[: int(valid.max()) + 1], peaks, BEAT_SAMPLES, PRE_SAMPLES, SAMPLING_RATE)
    template = np.median(beats, axis=0)
    template_norm = float(np.linalg.norm(template)) + 1e-6
    correlations = []
    for window in beats:
        correlations.append(float(np.dot(window, template) / ((np.linalg.norm(window) + 1e-6) * template_norm)))
    index = int(np.argmin(correlations))
    oddness = float(1.0 - correlations[index])
    return np.ascontiguousarray(beats[index]), oddness

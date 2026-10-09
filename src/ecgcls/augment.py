"""Training-only waveform changes. Stored preparations are never rewritten."""

from __future__ import annotations

import numpy as np

SAMPLING_RATE = 300


def augment_signal(
    signal: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    noise_std: float,
    cutout_seconds: float,
    stretch: float,
    scale_min: float,
    scale_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    stretched, stretched_mask = _stretch(signal, mask, float(rng.uniform(1.0 - stretch, 1.0 + stretch)))
    scale = float(rng.uniform(scale_min, scale_max))
    noisy = stretched.copy()
    valid = stretched_mask.astype(bool)
    noisy[valid] *= scale
    if noise_std > 0 and int(valid.sum()):
        noisy[valid] += rng.normal(0.0, noise_std, size=int(valid.sum())).astype(np.float32)
    return _cutout(noisy, stretched_mask, rng, cutout_seconds)


def _stretch(signal: np.ndarray, mask: np.ndarray, factor: float) -> tuple[np.ndarray, np.ndarray]:
    valid = int(mask.sum())
    if valid < 8:
        return signal.copy(), mask.copy()
    new_length = max(8, int(round(valid * factor)))
    source = np.linspace(0.0, 1.0, valid, endpoint=False)
    target = np.linspace(0.0, 1.0, new_length, endpoint=False)
    stretched = np.interp(target, source, signal[:valid]).astype(np.float32)
    out_signal = np.zeros_like(signal)
    out_mask = np.zeros_like(mask)
    keep = min(new_length, signal.size)
    out_signal[:keep] = stretched[:keep]
    out_mask[:keep] = 1
    return out_signal, out_mask


def _cutout(
    signal: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    cutout_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    width = int(round(cutout_seconds * SAMPLING_RATE))
    valid = int(mask.sum())
    if width <= 0 or valid <= width:
        return signal, mask
    start = int(rng.integers(0, valid - width))
    signal = signal.copy()
    mask = mask.copy()
    signal[start : start + width] = 0
    mask[start : start + width] = 0
    return signal, mask

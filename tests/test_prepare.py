"""The shared preparation keeps the whole recording and both beat views."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ecgcls.prepare import _beat_views, _correct_inversion, _zscore, make_splits, prepare_record
from ecgcls.records import Record


def test_preparation_pads_without_dropping_samples_and_splits_atrial_window(tmp_path: Path):
    sampling_rate = 300
    original = np.zeros(sampling_rate * 10, dtype=np.float64)
    peaks = np.arange(300, 2700, 300)
    original[peaks] = 8.0
    original[peaks + 5] = -2.0
    mat = tmp_path / "A00"
    mat.mkdir()
    import scipy.io

    scipy.io.savemat(mat / "A00001.mat", {"val": original.reshape(1, -1) * 1000.0})
    (mat / "A00001.hea").write_text(
        "A00001 1 300 3000 05:05:15 1/05/2000\nA00001.mat 16+24 1000/mV 16 0 -127 0 0 ECG\n",
        encoding="utf-8",
    )
    record = Record("A00/A00001", mat / "A00001.mat", mat / "A00001.hea", "N")
    prepared = prepare_record(
        record,
        sampling_rate,
        target=sampling_rate * 61,
        beat_samples=180,
        pre_samples=90,
        band_cfg={"low_hz": 0.5, "high_hz": 40.0, "order": 4},
    )
    assert prepared["signal"].shape == (sampling_rate * 61,)
    assert int(prepared["mask"].sum()) == original.size
    assert prepared["beats"].shape[1] == 180
    assert prepared["atrial"].shape == (prepared["beats"].shape[0], 90)
    assert np.allclose(prepared["atrial"], prepared["beats"][:, :90])
    assert prepared["rr_before"][0] == 0
    if prepared["rr_before"].size > 1:
        assert prepared["rr_before"][1] > 0


def test_negative_spikes_are_flipped_and_splits_do_not_overlap():
    sampling_rate = 300
    signal = np.zeros(sampling_rate * 10, dtype=np.float32)
    signal[np.arange(300, 2700, 300)] = -8.0
    corrected, inverted = _correct_inversion(signal, sampling_rate)
    assert inverted
    assert float(corrected.max()) > 0
    labels = ["N", "N", "A", "O", "~"] * 20
    splits = make_splits(labels, {"seed": 42, "test_size": 0.2, "n_folds": 5})
    assert set(splits["test"]).isdisjoint(splits["trainval"])
    assert len(splits["test"]) + len(splits["trainval"]) == len(labels)
    normalized = _zscore(np.array([1.0, 3.0], dtype=np.float32))
    assert abs(float(normalized.mean())) < 1e-5


def test_beat_window_is_centered_on_the_peak():
    signal = np.zeros(400, dtype=np.float32)
    signal[200] = 5.0
    beats, atrial, rr_before = _beat_views(signal, np.array([200], dtype=np.int32), 180, 90, 300)
    assert beats.shape == (1, 180)
    assert atrial.shape == (1, 90)
    assert float(beats[0, 90]) == 5.0
    assert rr_before[0] == 0

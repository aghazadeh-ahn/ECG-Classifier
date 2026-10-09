"""The light convolutional baseline stays near one hundred thousand parameters."""

from __future__ import annotations

import numpy as np
import torch

from ecgcls.augment import augment_signal
from ecgcls.metrics import challenge_f1
from ecgcls.models.cnn import Cnn1d, parameter_count


def test_light_cnn_has_about_one_hundred_thousand_parameters():
    model = Cnn1d(channels=[32, 64, 128, 128], kernels=[7, 5, 5, 3], dropout=0.2)
    assert 80_000 <= parameter_count(model) <= 150_000


def test_forward_returns_four_class_logits():
    model = Cnn1d(channels=[32, 64, 128, 128], kernels=[7, 5, 5, 3], dropout=0.0)
    model.eval()
    signal = torch.randn(2, 18300)
    mask = torch.ones(2, 18300)
    mask[:, 9000:] = 0
    with torch.inference_mode():
        logits = model(signal, mask)
    assert logits.shape == (2, 4)
    assert torch.isfinite(logits).all()


def test_correct_noise_stays_out_of_the_average_and_a_wrong_call_moves_it():
    truth = np.array(["N", "A", "O", "~"])
    assert challenge_f1(truth, np.array(["N", "A", "O", "~"])) == 1.0
    assert challenge_f1(truth, np.array(["N", "A", "O", "N"])) < 1.0


def test_augmentation_keeps_the_stored_length():
    signal = np.zeros(18300, dtype=np.float32)
    mask = np.zeros(18300, dtype=np.uint8)
    signal[:9000] = 1
    mask[:9000] = 1
    augmented, augmented_mask = augment_signal(
        signal,
        mask,
        np.random.default_rng(0),
        noise_std=0.05,
        cutout_seconds=0.5,
        stretch=0.05,
        scale_min=0.9,
        scale_max=1.1,
    )
    assert augmented.shape == signal.shape
    assert augmented_mask.shape == mask.shape
    assert 0 < int(augmented_mask.sum()) <= signal.size

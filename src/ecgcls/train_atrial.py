"""Train the atrial model with the same folds, early stopping, and one locked test."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ecgcls.augment import augment_signal
from ecgcls.metrics import CLASSES, challenge_f1, confusion, per_class_f1, report_rows
from ecgcls.models.atrial import AtrialCnnLstm
from ecgcls.models.cnn import parameter_count
from ecgcls.plots import save_classification_report, save_confusion, save_curve
from ecgcls.prepare import _beat_views, _r_peaks
from ecgcls.train_signal import class_weights, load_arrays

BEAT_SAMPLES = 180
PRE_SAMPLES = 90
SAMPLING_RATE = 300


class AtrialDataset(Dataset):
    def __init__(
        self,
        signals: np.ndarray,
        masks: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        augment_cfg: dict | None,
        seed: int,
    ) -> None:
        self.signals = signals
        self.masks = masks
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.augment_cfg = augment_cfg
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> tuple[np.ndarray, np.ndarray, int]:
        index = int(self.indices[item])
        signal = np.array(self.signals[index], copy=True)
        mask = np.array(self.masks[index], copy=True)
        if self.augment_cfg is not None:
            signal, mask = augment_signal(
                signal,
                mask,
                np.random.default_rng(self.seed + self.epoch * 1_000_003 + item),
                **self.augment_cfg,
            )
        atrial, rr_before = _windows(signal, mask)
        return atrial, rr_before, int(self.labels[index])


def run_atrial_experiment(root: Path, cfg: dict) -> dict:
    seed = int(cfg["seed"])
    _seed_everything(seed)
    processed_dir = root / cfg["processed_dir"]
    result_dir = root / "results" / cfg["name"]
    result_dir.mkdir(parents=True, exist_ok=True)
    signals, masks, labels, splits = load_arrays(processed_dir, root / "runs" / "signal_cache")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = parameter_count(_build(cfg["model"]))
    print(f"device {device} params {params}", flush=True)
    train_cfg = cfg["train"]
    augment_cfg = {
        "noise_std": float(cfg["augment"]["noise_std"]),
        "cutout_seconds": float(cfg["augment"]["cutout_seconds"]),
        "stretch": float(cfg["augment"]["stretch"]),
        "scale_min": float(cfg["augment"]["scale_min"]),
        "scale_max": float(cfg["augment"]["scale_max"]),
    }
    histories: list[list[dict]] = []
    best_epochs: list[int] = []
    fold_val: list[float] = []
    oof = np.zeros((labels.shape[0], len(CLASSES)), dtype=np.float32)
    oof_filled = np.zeros(labels.shape[0], dtype=bool)
    fold_states: list[dict] = []
    test_index = np.asarray(splits["test"], dtype=np.int64)

    for fold_id, fold in enumerate(splits["folds"]):
        print(f"fold {fold_id}", flush=True)
        _seed_everything(seed + fold_id)
        fold_model = _build(cfg["model"]).to(device)
        train_index = np.asarray(fold["train"], dtype=np.int64)
        val_index = np.asarray(fold["val"], dtype=np.int64)
        train_loader = _loader(signals, masks, labels, train_index, augment_cfg, seed + fold_id, True, train_cfg, device)
        val_loader = _loader(signals, masks, labels, val_index, None, seed, False, train_cfg, device)
        history, best_epoch = _fit(
            fold_model,
            train_loader,
            val_loader,
            train_cfg,
            class_weights(labels, train_index).to(device),
            device,
        )
        probabilities, predicted, truth = _predict(fold_model, val_loader, device)
        score = challenge_f1(np.asarray(CLASSES)[truth], np.asarray(CLASSES)[predicted])
        oof[val_index] = probabilities
        oof_filled[val_index] = True
        fold_states.append({key: value.detach().cpu().clone() for key, value in fold_model.state_dict().items()})
        histories.append(history)
        best_epochs.append(best_epoch)
        fold_val.append(score)
        print(f"fold {fold_id} best epoch {best_epoch} challenge F1 {score:.4f}", flush=True)

    print("score locked test with the five fold models", flush=True)
    test_loader = _loader(signals, masks, labels, test_index, None, seed, False, train_cfg, device)
    probabilities = _ensemble_probabilities(_build(cfg["model"]), fold_states, test_loader, device)
    truth_labels = np.asarray(CLASSES)[labels[test_index]]
    predicted_labels = np.asarray(CLASSES)[probabilities.argmax(axis=1)]
    test_score = challenge_f1(truth_labels, predicted_labels)
    matrix = confusion(truth_labels, predicted_labels)
    rows = report_rows(truth_labels, predicted_labels)
    save_curve(result_dir / "train_curve.png", histories, "loss", "Training loss", "Training loss")
    save_curve(result_dir / "val_curve.png", histories, "val_f1", "Validation challenge F1", "Challenge F1")
    save_confusion(result_dir / "confusion_matrix.png", matrix)
    save_classification_report(result_dir / "classification_report.png", rows, test_score)
    oof_truth = np.asarray(CLASSES)[labels[oof_filled]]
    oof_predicted = np.asarray(CLASSES)[oof[oof_filled].argmax(axis=1)]
    metrics = {
        "name": cfg["name"],
        "params": params,
        "device": str(device),
        "cv_f1": float(np.mean(fold_val)),
        "cv_f1_std": float(np.std(fold_val)),
        "cv_f1_per_fold": [float(score) for score in fold_val],
        "best_epochs": best_epochs,
        "test_f1": test_score,
        "test_f1_per_class": per_class_f1(truth_labels, predicted_labels),
        "test_confusion": matrix,
        "classification_report": rows,
        "oof_f1": challenge_f1(oof_truth, oof_predicted),
        "score_test": "fold_ensemble",
    }
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (result_dir / "history.json").write_text(json.dumps(histories), encoding="utf-8")
    np.savez_compressed(
        result_dir / "predictions.npz",
        oof=oof,
        oof_filled=oof_filled,
        test_index=test_index,
        test_probs=probabilities,
        test_true=truth_labels,
        test_pred=predicted_labels,
    )
    print(f"test challenge F1 {test_score:.4f} cv {metrics['cv_f1']:.4f} params {params}", flush=True)
    return metrics


def _windows(signal: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.flatnonzero(mask)
    if valid.size == 0:
        return np.zeros((1, PRE_SAMPLES), dtype=np.float32), np.zeros(1, dtype=np.float32)
    length = int(valid.max()) + 1
    peaks = _r_peaks(signal[:length], SAMPLING_RATE)
    if peaks.size == 0:
        return np.zeros((1, PRE_SAMPLES), dtype=np.float32), np.zeros(1, dtype=np.float32)
    _beats, atrial, rr_before = _beat_views(signal[:length], peaks, BEAT_SAMPLES, PRE_SAMPLES, SAMPLING_RATE)
    return np.ascontiguousarray(atrial), np.ascontiguousarray(rr_before)


def _build(model_cfg: dict) -> AtrialCnnLstm:
    return AtrialCnnLstm(
        channels=[int(channel) for channel in model_cfg["channels"]],
        kernels=[int(kernel) for kernel in model_cfg["kernels"]],
        lstm_hidden=int(model_cfg["lstm_hidden"]),
        lstm_layers=int(model_cfg["lstm_layers"]),
        dropout=float(model_cfg["dropout"]),
    )


def _fit(model, train_loader, val_loader, train_cfg, weights, device) -> tuple[list[dict], int]:
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    epochs = int(train_cfg["epochs"])
    patience = int(train_cfg["patience"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(train_cfg["min_lr"]),
    )
    best_score = -1.0
    best_epoch = 1
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    stale = 0
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        loss = _train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step()
        _probabilities, predicted, truth = _predict(model, val_loader, device)
        score = challenge_f1(np.asarray(CLASSES)[truth], np.asarray(CLASSES)[predicted])
        history.append({"epoch": epoch, "loss": loss, "val_f1": score})
        print(f"epoch {epoch} loss {loss:.4f} val F1 {score:.4f} lr {learning_rate:.6f}", flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return history, best_epoch


def _ensemble_probabilities(template, states, loader, device) -> np.ndarray:
    template = template.to(device)
    accumulated = None
    for state in states:
        template.load_state_dict(state)
        probabilities, _predicted, _truth = _predict(template, loader, device)
        accumulated = probabilities if accumulated is None else accumulated + probabilities
    return accumulated / len(states)


def _loader(signals, masks, labels, indices, augment_cfg, seed, shuffle, train_cfg, device) -> DataLoader:
    dataset = AtrialDataset(signals, masks, labels, np.asarray(indices), augment_cfg, seed)
    batch_size = int(train_cfg["batch_size"])
    drop_last = shuffle and len(dataset) > batch_size
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=drop_last,
        pin_memory=device.type == "cuda",
        collate_fn=_collate,
    )


def _collate(batch: list[tuple[np.ndarray, np.ndarray, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(atrial.shape[0] for atrial, _rr, _label in batch)
    atrial_batch = torch.zeros(len(batch), width, PRE_SAMPLES, dtype=torch.float32)
    rr_batch = torch.zeros(len(batch), width, dtype=torch.float32)
    beat_mask = torch.zeros(len(batch), width, dtype=torch.float32)
    labels = torch.empty(len(batch), dtype=torch.long)
    for index, (atrial, rr_before, label) in enumerate(batch):
        count = atrial.shape[0]
        atrial_batch[index, :count] = torch.from_numpy(atrial)
        rr_batch[index, :count] = torch.from_numpy(rr_before)
        beat_mask[index, :count] = 1
        labels[index] = label
    return atrial_batch, rr_batch, beat_mask, labels


def _train_one_epoch(model, loader, optimizer, criterion, device, epoch: int) -> float:
    model.train()
    loader.dataset.set_epoch(epoch)
    total = 0.0
    seen = 0
    for atrial, rr_before, beat_mask, label in loader:
        atrial = atrial.to(device, non_blocking=True)
        rr_before = rr_before.to(device, non_blocking=True)
        beat_mask = beat_mask.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(atrial, rr_before, beat_mask), label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.item()) * atrial.shape[0]
        seen += atrial.shape[0]
    return total / max(seen, 1)


def _predict(model, loader, device):
    model.eval()
    probabilities = []
    truths = []
    with torch.inference_mode():
        for atrial, rr_before, beat_mask, label in loader:
            logits = model(
                atrial.to(device, non_blocking=True),
                rr_before.to(device, non_blocking=True),
                beat_mask.to(device, non_blocking=True),
            )
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            truths.append(label.numpy())
    stacked = np.concatenate(probabilities, axis=0)
    truth = np.concatenate(truths)
    return stacked, stacked.argmax(axis=1), truth


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

"""
train.py
=========
Reproducible training pipeline for the ExoplanetCNN classifier.

Usage:
    python train.py --n-per-class 600 --epochs 40 --seed 42

Produces (under ml/checkpoints/<run_id>/):
  - model.pt              : trained weights + architecture config
  - scaler.json            : aux-feature standardization stats (mean/std)
  - temperature.json       : calibration temperature
  - metrics.json           : train/val/test metrics, confusion matrix
  - training_log.csv       : per-epoch loss/accuracy history
  - config.json             : full reproducibility record (seed, hyperparams, data sizes)

Model versioning: each run gets a timestamped run_id; `latest` symlink/pointer
file is updated so the backend can always load the most recent good model
without hardcoding a path.
"""

from __future__ import annotations
import argparse
import json
import time
import os
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from dataset_builder import DatasetBuilder, to_tensors, AUX_FEATURE_NAMES
from model_cnn import ExoplanetCNN, TemperatureScaler, CLASS_NAMES, N_CLASSES


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def standardize_fit(x: torch.Tensor):
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-6)
    return mean, std


def standardize_apply(x: torch.Tensor, mean, std):
    return (x - mean) / std


def train_one_run(n_per_class=600, epochs=40, batch_size=32, lr=1e-3, seed=42,
                   val_frac=0.15, test_frac=0.15, baseline_days_range=(40, 120),
                   patience=8, out_root="checkpoints"):
    set_seed(seed)
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_root, run_id)
    os.makedirs(run_dir, exist_ok=True)

    print(f"=== Run {run_id} ===")
    print(f"Building dataset: {n_per_class} samples/class x 5 classes...")
    t0 = time.time()
    builder = DatasetBuilder(seed=seed)
    samples = builder.build_dataset(n_per_class=n_per_class, baseline_days_range=baseline_days_range)
    print(f"Dataset built: {len(samples)} samples in {time.time()-t0:.1f}s")

    g, l, aux, y = to_tensors(samples)

    # ---- split: stratified by class ----
    n = len(y)
    idx_by_class = {c: np.where(y.numpy() == c)[0] for c in range(N_CLASSES)}
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for c, idxs in idx_by_class.items():
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_frac))
        n_test = max(1, int(len(idxs) * test_frac))
        val_idx.extend(idxs[:n_val])
        test_idx.extend(idxs[n_val:n_val + n_test])
        train_idx.extend(idxs[n_val + n_test:])
    train_idx, val_idx, test_idx = np.array(train_idx), np.array(val_idx), np.array(test_idx)
    rng.shuffle(train_idx)

    # ---- standardize aux features (fit on train only) ----
    aux_mean, aux_std = standardize_fit(aux[train_idx])
    aux_std_t = standardize_apply(aux, aux_mean, aux_std)

    def subset(idxs):
        return TensorDataset(g[idxs], l[idxs], aux_std_t[idxs], y[idxs])

    train_ds, val_ds, test_ds = subset(train_idx), subset(val_idx), subset(test_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"Split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    # ---- class weights (handles any imbalance from generation failures) ----
    class_counts = torch.bincount(y[train_idx], minlength=N_CLASSES).float()
    class_weights = (class_counts.sum() / (N_CLASSES * class_counts.clamp_min(1)))
    print("Class counts (train):", class_counts.tolist())

    model = ExoplanetCNN(global_len=g.shape[1], local_len=l.shape[1], n_aux_features=aux.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_n = 0.0, 0, 0
        for gb, lb, ab, yb in train_loader:
            optimizer.zero_grad()
            logits = model(gb, lb, ab)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(yb)
            train_correct += (logits.argmax(1) == yb).sum().item()
            train_n += len(yb)

        model.eval()
        val_loss, val_correct, val_n = 0.0, 0, 0
        with torch.no_grad():
            for gb, lb, ab, yb in val_loader:
                logits = model(gb, lb, ab)
                loss = criterion(logits, yb)
                val_loss += loss.item() * len(yb)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_n += len(yb)

        train_loss /= train_n
        val_loss /= val_n
        train_acc = train_correct / train_n
        val_acc = val_correct / val_n
        scheduler.step(val_loss)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                         "train_acc": train_acc, "val_acc": val_acc, "lr": optimizer.param_groups[0]["lr"]})
        print(f"Epoch {epoch:3d}/{epochs} | train_loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.3f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    model.load_state_dict(best_state)

    # ---- calibration (temperature scaling) on val set ----
    model.eval()
    with torch.no_grad():
        val_logits = torch.cat([model(gb, lb, ab) for gb, lb, ab, _ in val_loader])
        val_labels = torch.cat([yb for _, _, _, yb in val_loader])
    scaler = TemperatureScaler()
    temperature = scaler.fit(val_logits, val_labels)
    print(f"Calibration temperature: {temperature:.3f}")

    # ---- test set evaluation ----
    with torch.no_grad():
        test_logits = torch.cat([model(gb, lb, ab) for gb, lb, ab, _ in test_loader])
        test_labels = torch.cat([yb for _, _, _, yb in test_loader])
    test_probs = torch.softmax(test_logits / temperature, dim=1)
    test_preds = test_probs.argmax(1)
    test_acc = (test_preds == test_labels).float().mean().item()

    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for t, p in zip(test_labels.numpy(), test_preds.numpy()):
        confusion[t, p] += 1

    per_class_precision, per_class_recall, per_class_f1 = [], [], []
    for c in range(N_CLASSES):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        per_class_precision.append(prec)
        per_class_recall.append(rec)
        per_class_f1.append(f1)

    metrics = {
        "test_accuracy": test_acc,
        "confusion_matrix": confusion.tolist(),
        "class_names": CLASS_NAMES,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "macro_f1": float(np.mean(per_class_f1)),
        "calibration_temperature": temperature,
        "best_val_loss": best_val_loss,
        "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
    }
    print("Test accuracy:", test_acc, "Macro F1:", metrics["macro_f1"])
    print("Confusion matrix (rows=true, cols=pred):")
    print(CLASS_NAMES)
    print(confusion)

    # ---- save artifacts ----
    torch.save({
        "state_dict": model.state_dict(),
        "global_len": g.shape[1], "local_len": l.shape[1], "n_aux_features": aux.shape[1],
        "aux_feature_names": AUX_FEATURE_NAMES,
    }, os.path.join(run_dir, "model.pt"))

    with open(os.path.join(run_dir, "scaler.json"), "w") as f:
        json.dump({"aux_mean": aux_mean.tolist(), "aux_std": aux_std.tolist(),
                   "aux_feature_names": AUX_FEATURE_NAMES}, f, indent=2)

    with open(os.path.join(run_dir, "temperature.json"), "w") as f:
        json.dump({"temperature": temperature}, f, indent=2)

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    import csv
    with open(os.path.join(run_dir, "training_log.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    config = {
        "run_id": run_id, "seed": seed, "n_per_class": n_per_class, "epochs_requested": epochs,
        "epochs_run": len(history), "batch_size": batch_size, "lr": lr,
        "val_frac": val_frac, "test_frac": test_frac, "baseline_days_range": list(baseline_days_range),
        "model_params": sum(p.numel() for p in model.parameters()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # update "latest" pointer
    with open(os.path.join(out_root, "latest.json"), "w") as f:
        json.dump({"run_id": run_id, "run_dir": run_dir}, f, indent=2)

    print(f"\nSaved all artifacts to {run_dir}")
    return run_dir, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-class", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    train_one_run(n_per_class=args.n_per_class, epochs=args.epochs, batch_size=args.batch_size,
                  lr=args.lr, seed=args.seed, patience=args.patience)

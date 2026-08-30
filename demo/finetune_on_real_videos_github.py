"""
Fine-Tune BiLSTM on Real Videos
Adapts the model to MotionBERT-derived skeletons from recorded footage
Authors: Kornel Lipka, Yu-Cian Huang, Ssu-Cheng Chen
Course: Deep Learning and Decision Making, TUM SS2026
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motionbert_joint_mapping import h36m_to_hit605

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Config
# change your path
MOTIONBERT_OUTPUTS = r"path to MotionBERT outputs (from run_all_motionbert.ps1)"
# change your path
STARTING_MODEL      = r"path to starting model checkpoint"
# change your path
SAVE_PATH            = r"path to save fine-tuned model"

WINDOW_SIZE  = 64
STRIDE       = 32
NUM_CLASSES  = 10
INPUT_SIZE   = 51
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT      = 0.3
BATCH_SIZE   = 32
EPOCHS       = 100   # increased - loss hadn't converged at 30
LR           = 1e-4   # lower than original training - fine-tuning, not training from scratch
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTION_NAMES = [
    "Preparation", "Grasp Bird's Tail", "Single Whip", "Lift up Hand",
    "White Crane Spreads Wings", "Brush Knee and Twist Step",
    "Hold the Lute", "Pulling Blocking and Pounding",
    "Apparent Close Up", "Cross Hands"
]

VIDEO_LABELS = {
    "01_preparation":        0,  # Preparation
    "02_grasp_birds_tail":   1,  # Grasp Bird's Tail
    "03_single_whip":        2,  # Single Whip
    "04_lift_hand":          3,  # Lift up Hand
    "05_white_crane":        4,  # White Crane Spreads Wings
    "06_brush_knee":         5,  # Brush Knee and Twist Step
    "07_hold_lute":          6,  # Hold the Lute
    "08_pulling_blocking":   7,  # Pulling Blocking and Pounding
    "09_apparent_close":     8,  # Apparent Close Up
    "10_cross_hands":        9,  # Cross Hands
}


def normalize_joints(joints):
    pelvis = joints[0].copy()
    joints = joints - pelvis
    std    = joints.std(axis=0) + 1e-6
    joints = joints / std
    return joints


# Dataset: builds windows from MotionBERT outputs + known labels
class RealVideoDataset(Dataset):
    def __init__(self, video_names, is_train=True, val_fraction=0.2):
        self.windows = []

        for name in video_names:
            x3d_path = os.path.join(MOTIONBERT_OUTPUTS, name, "X3D.npy")
            if not os.path.exists(x3d_path):
                print(f"  MISSING: {x3d_path} - skipping {name}")
                continue

            label = VIDEO_LABELS[name]
            x3d = np.load(x3d_path)                       # (T, 17, 3) H36M order
            hit605 = h36m_to_hit605(x3d)                    # (T, 17, 3) HIT605 order
            T = hit605.shape[0]

            normalized = np.array([normalize_joints(hit605[t]) for t in range(T)])
            flat = normalized.reshape(T, -1)                # (T, 51)

            # Build all windows for this video
            video_windows = []
            for start in range(0, T - WINDOW_SIZE + 1, STRIDE):
                end = start + WINDOW_SIZE
                video_windows.append(flat[start:end])

            # Split: last val_fraction of each video's windows held out
            # for a quick sanity-check validation split (temporal split,
            # not random, to avoid near-duplicate overlapping windows
            # leaking between train/val).
            split_idx = int(len(video_windows) * (1 - val_fraction))
            if is_train:
                selected = video_windows[:split_idx]
            else:
                selected = video_windows[split_idx:]

            for w in selected:
                self.windows.append((w, label))

            print(f"  {name}: {T} frames -> {len(video_windows)} windows "
                  f"({'train' if is_train else 'val'}: {len(selected)})")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x, y = self.windows[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# Model (identical architecture)
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_out):
        scores  = self.attn(lstm_out).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = (lstm_out * weights.unsqueeze(-1)).sum(dim=1)
        return context, weights


class TaiChiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super().__init__()
        self.lstm      = nn.LSTM(input_size, hidden_size, num_layers,
                                  batch_first=True, bidirectional=True,
                                  dropout=dropout if num_layers > 1 else 0.0)
        self.attention = AttentionLayer(hidden_size)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out, _           = self.lstm(x)
        context, weights = self.attention(out)
        context          = self.dropout(context)
        return self.fc(context), weights


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += x.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits, _ = model(x)
            preds = logits.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    if not all_labels:
        return 0.0, 0.0, [], []
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, f1, all_preds, all_labels


def main():
    print(f"Device: {DEVICE}\n")

    video_names = list(VIDEO_LABELS.keys())

    print("Building training set (80% of each video's windows):")
    train_dataset = RealVideoDataset(video_names, is_train=True, val_fraction=0.2)
    print("\nBuilding validation set (last 20% of each video's windows):")
    val_dataset = RealVideoDataset(video_names, is_train=False, val_fraction=0.2)

    print(f"\nTotal train windows: {len(train_dataset)}")
    print(f"Total val windows: {len(val_dataset)}\n")

    if len(train_dataset) == 0:
        print("No training data found - check that MotionBERT outputs exist "
              "at the configured paths.")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Compute class weights to counter the imbalance we saw (e.g.
    # 04_lift_hand had 343 train windows, 07_hold_lute only 32 - without
    # weighting, the model barely learns the sparse classes).
    train_labels = [label for _, label in train_dataset.windows]
    class_counts = np.zeros(NUM_CLASSES)
    for label in train_labels:
        class_counts[label] += 1
    print("\nTrain windows per class:")
    for i, name in enumerate(ACTION_NAMES):
        print(f"    {name:<28s} {int(class_counts[i]):4d}")
    # Inverse-frequency weights, normalized so they average to 1
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.mean()
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    print(f"\nClass weights: {class_weights.cpu().numpy().round(2)}\n")

    # Start from the already-trained (augmented) model, not from scratch
    print(f"Loading starting model: {STARTING_MODEL}")
    model = TaiChiLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
                        NUM_CLASSES, DROPOUT).to(DEVICE)
    model.load_state_dict(torch.load(STARTING_MODEL, map_location=DEVICE))
    print("Loaded. Starting fine-tuning...\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1 = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_acc, val_f1, _, _ = evaluate(model, val_loader)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  Saved best model (Val F1={best_val_f1:.4f}) -> {SAVE_PATH}")

    # Final report
    model.load_state_dict(torch.load(SAVE_PATH))
    val_acc, val_f1, preds, labels = evaluate(model, val_loader)
    print("\n" + "-" * 60)
    print("Final Report (held-out 20% of each video's windows)")
    print("-" * 60)
    print(f"Val Accuracy: {val_acc:.4f}")
    print(f"Val Macro F1: {val_f1:.4f}")
    if labels:
        print("\nPer-class report:")
        print(classification_report(labels, preds, target_names=ACTION_NAMES,
                                     zero_division=0))
    print(f"\nModel saved to: {SAVE_PATH}")


if __name__ == "__main__":
    main()

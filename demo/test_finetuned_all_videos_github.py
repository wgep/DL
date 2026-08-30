"""
Test Fine-Tuned Model on Real Videos
Reports final per-video accuracy across all 9 recordings
Authors: Kornel Lipka, Yu-Cian Huang, Ssu-Cheng Chen
Course: Deep Learning and Decision Making, TUM SS2026
"""

import numpy as np
import torch
import torch.nn as nn
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motionbert_joint_mapping import h36m_to_hit605

# Config
# change your path
MOTIONBERT_OUTPUTS = r"path to MotionBERT outputs (from run_all_motionbert.ps1)"
# change your path
MODEL_PATH          = r"path to fine-tuned model checkpoint"

WINDOW_SIZE  = 64
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT      = 0.3
NUM_CLASSES  = 10
INPUT_SIZE   = 51
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTION_NAMES = [
    "Preparation", "Grasp Bird's Tail", "Single Whip", "Lift up Hand",
    "White Crane Spreads Wings", "Brush Knee and Twist Step",
    "Hold the Lute", "Pulling Blocking and Pounding",
    "Apparent Close Up", "Cross Hands"
]

VIDEO_LABELS = {
    "01_preparation":        "Preparation",
    "02_grasp_birds_tail":   "Grasp Bird's Tail",
    "03_single_whip":        "Single Whip",
    "04_lift_hand":          "Lift up Hand",
    "05_white_crane":        "White Crane Spreads Wings",
    "06_brush_knee":         "Brush Knee and Twist Step",
    "07_hold_lute":          "Hold the Lute",
    "08_pulling_blocking":   "Pulling Blocking and Pounding",
    "09_apparent_close":     "Apparent Close Up",
    "10_cross_hands":        "Cross Hands",
}


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


def normalize_joints(joints):
    pelvis = joints[0].copy()
    joints = joints - pelvis
    std    = joints.std(axis=0) + 1e-6
    joints = joints / std
    return joints


def predict_video(x3d, model):
    hit605 = h36m_to_hit605(x3d)
    T = hit605.shape[0]
    normalized = np.array([normalize_joints(hit605[t]) for t in range(T)])
    flat = normalized.reshape(T, -1)

    if T < WINDOW_SIZE:
        return None, None

    windows = np.lib.stride_tricks.sliding_window_view(flat, WINDOW_SIZE, axis=0)
    windows = windows.transpose(0, 2, 1)
    x = torch.tensor(windows, dtype=torch.float32).to(DEVICE)

    all_probs = []
    BATCH = 512
    with torch.no_grad():
        for i in range(0, x.shape[0], BATCH):
            chunk = x[i:i+BATCH]
            logits, _ = model(chunk)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
    all_probs = np.concatenate(all_probs, axis=0)

    start = len(all_probs) // 4
    end   = 3 * len(all_probs) // 4
    middle_probs = all_probs[start:end] if len(all_probs) > 4 else all_probs
    avg_probs = middle_probs.mean(axis=0)
    return ACTION_NAMES[avg_probs.argmax()], avg_probs


def main():
    print(f"Device: {DEVICE}\n")
    print(f"Loading model: {MODEL_PATH}")
    model = TaiChiLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
                        NUM_CLASSES, DROPOUT).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded!\n")

    correct_count = 0
    total_count = 0
    results = []

    for name, expected in VIDEO_LABELS.items():
        x3d_path = os.path.join(MOTIONBERT_OUTPUTS, name, "X3D.npy")
        if not os.path.exists(x3d_path):
            print(f"MISSING: {x3d_path} - skipping {name}")
            continue

        x3d = np.load(x3d_path)
        pred, probs = predict_video(x3d, model)
        correct = pred == expected
        if correct:
            correct_count += 1
        total_count += 1

        status = "CORRECT" if correct else "WRONG"
        print(f"{name:<24s} expected={expected:<28s} predicted={pred!s:<28s} [{status}]")
        results.append((name, expected, pred, correct))

    print("\n" + "=" * 60)
    print(f"FINAL ACCURACY: {correct_count}/{total_count} = "
          f"{correct_count/total_count:.1%}" if total_count else "No videos tested.")
    print("=" * 60)


if __name__ == "__main__":
    main()

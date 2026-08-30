"""
Build Final Demo Video
Animates skeleton + live predictions for all 9 real videos
Authors: Kornel Lipka, Yu-Cian Huang, Ssu-Cheng Chen
Course: Deep Learning and Decision Making, TUM SS2026
"""

import numpy as np
import torch
import torch.nn as nn
import cv2
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motionbert_joint_mapping import h36m_to_hit605

# Config

MOTIONBERT_OUTPUTS = r"path to MotionBERT outputs (from run_all_motionbert.ps1)"

MODEL_PATH          = r"path to fine-tuned model checkpoint"

OUTPUT_PATH          = r"path to save final demo video"

WINDOW_SIZE  = 64
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT      = 0.3
NUM_CLASSES  = 10
INPUT_SIZE   = 51
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WIDTH, HEIGHT = 800, 900
FPS = 30
BG_COLOR = (20, 20, 20)

ACTION_NAMES = [
    "Preparation", "Grasp Bird's Tail", "Single Whip", "Lift up Hand",
    "White Crane Spreads Wings", "Brush Knee and Twist Step",
    "Hold the Lute", "Pulling Blocking and Pounding",
    "Apparent Close Up", "Cross Hands"
]

ACTION_COLORS = [
    (255, 200, 0), (0, 255, 100), (0, 150, 255), (255, 100, 0),
    (200, 0, 255), (0, 255, 255), (255, 0, 150), (150, 255, 0),
    (255, 150, 150), (100, 200, 255),
]

VIDEO_ORDER = [
    ("01_preparation", "Preparation"),
    ("02_grasp_birds_tail", "Grasp Bird's Tail"),
    ("03_single_whip", "Single Whip"),
    ("04_lift_hand", "Lift up Hand"),
    ("05_white_crane", "White Crane Spreads Wings"),
    ("06_brush_knee", "Brush Knee and Twist Step"),
    ("07_hold_lute", "Hold the Lute"),
    ("08_pulling_blocking", "Pulling Blocking and Pounding"),
    ("09_apparent_close", "Apparent Close Up"),
    ("10_cross_hands", "Cross Hands"),
]

# HIT605 skeleton connections (same as skeleton_demo.py)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 4), (0, 7), (1, 2), (2, 3), (4, 5), (5, 6),
    (7, 8), (7, 16), (8, 9), (8, 10), (8, 13), (10, 11), (11, 12),
    (13, 14), (14, 15), (16, 10), (16, 13),
]


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


def project_joints(joints_3d, width, height):
    x_coords = joints_3d[:, 0]
    y_coords = joints_3d[:, 1]   # FIXED: no longer negated - MotionBERT's
                                  # H36M output uses the opposite Y
                                  # convention from HIT605 mocap data,
                                  # which caused the upside-down rendering.
    margin = 120
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)
    scale = min((width - 2*margin) / x_range, (height - 2*margin - 100) / y_range)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    screen_x = ((x_coords - x_center) * scale + width  / 2).astype(int)
    screen_y = ((y_coords - y_center) * scale + height / 2 + 50).astype(int)
    return screen_x, screen_y


def draw_skeleton(frame, joints_3d, color):
    sx, sy = project_joints(joints_3d, frame.shape[1], frame.shape[0])
    for i, j in SKELETON_CONNECTIONS:
        cv2.line(frame, (sx[i], sy[i]), (sx[j], sy[j]), color, 2, cv2.LINE_AA)
    for i in range(len(sx)):
        size = 8 if i == 9 else 5
        cv2.circle(frame, (sx[i], sy[i]), size, (255, 255, 255), -1)
        cv2.circle(frame, (sx[i], sy[i]), size, color, 2)


def draw_info(frame, true_label, pred_label, confidence, frame_num, total_frames, color):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 100), (30, 30, 30), -1)
    cv2.putText(frame, f"True Action:      {true_label}",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    pred_color = (0, 255, 0) if pred_label == true_label else (0, 100, 255)
    cv2.putText(frame, f"Predicted Action: {pred_label} ({confidence:.1%})",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pred_color, 2)
    cv2.rectangle(frame, (0, frame.shape[0]-8), (frame.shape[1], frame.shape[0]), (50, 50, 50), -1)
    progress = int(frame_num / total_frames * frame.shape[1])
    cv2.rectangle(frame, (0, frame.shape[0]-8), (progress, frame.shape[0]), color, -1)


def main():
    print(f"Device: {DEVICE}\n")
    print("Loading fine-tuned model...")
    model = TaiChiLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
                        NUM_CLASSES, DROPOUT).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded!\n")

    writer = cv2.VideoWriter(
        OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (WIDTH, HEIGHT)
    )

    for name, true_label in VIDEO_ORDER:
        x3d_path = os.path.join(MOTIONBERT_OUTPUTS, name, "X3D.npy")
        if not os.path.exists(x3d_path):
            print(f"MISSING: {x3d_path} - skipping {name}")
            continue

        print(f"Rendering: {name} (true label: {true_label})")
        x3d = np.load(x3d_path)
        hit605 = h36m_to_hit605(x3d)   # (T, 17, 3)
        T = hit605.shape[0]

        color = ACTION_COLORS[ACTION_NAMES.index(true_label)]
        frame_buffer = []
        current_pred = "Waiting..."
        current_conf = 0.0

        # Smoothing buffer for the DRAWN skeleton only - MotionBERT's raw
        # per-frame 3D estimates can be jittery frame to frame, which
        # looks like "crazy fast rotating". This is purely visual - the
        # model's predictions above still use the raw (unsmoothed) data,
        # unchanged from what already achieved 100% test accuracy.
        SMOOTH_WINDOW = 5
        draw_buffer = []

        for t in range(T):
            frame = np.full((HEIGHT, WIDTH, 3), BG_COLOR, dtype=np.uint8)
            joints_3d = hit605[t]

            draw_buffer.append(joints_3d)
            if len(draw_buffer) > SMOOTH_WINDOW:
                draw_buffer.pop(0)
            smoothed_joints = np.mean(draw_buffer, axis=0)
            draw_skeleton(frame, smoothed_joints, color)

            normalized = normalize_joints(joints_3d)
            frame_buffer.append(normalized.flatten())

            if len(frame_buffer) >= WINDOW_SIZE:
                window = np.array(frame_buffer[-WINDOW_SIZE:])
                x_t = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    logits, _ = model(x_t)
                    probs = torch.softmax(logits, dim=1)[0]
                    idx = probs.argmax().item()
                    current_pred = ACTION_NAMES[idx]
                    current_conf = probs[idx].item()

            draw_info(frame, true_label, current_pred, current_conf, t, T, color)
            cv2.putText(frame, f"{name} | Frame {t+1}/{T}",
                        (20, HEIGHT-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            writer.write(frame)

        # 1 second pause between actions
        for _ in range(FPS):
            pause = np.full((HEIGHT, WIDTH, 3), (10, 10, 10), dtype=np.uint8)
            cv2.putText(pause, "Next action...", (WIDTH//2 - 100, HEIGHT//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
            writer.write(pause)

    writer.release()
    print(f"\nDone. Final demo video saved to:\n  {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

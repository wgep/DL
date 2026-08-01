"""
HIT TaiChi Action Recognition — STGCN Model
Dataset: HIT605 TaiChi (Xu et al., 2020)
Authors: Kornel Lipka, Yu-Cian Huang, Ssu-Cheng Chen
Course: Deep Learning and Decision Making, TUM SS2026
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import os
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import random

# Seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



# Config

BASE_DIR    = r"C:\Users\lkorn\Documents\Deep Learning\Project\TaiChi.tar\TaiChi"
FEATURE_DIR = os.path.join(BASE_DIR, "feature")
LABEL_DIR   = os.path.join(BASE_DIR, "label")
TRAIN_LIST  = os.path.join(BASE_DIR, "split1", "train.txt")
TEST_LIST   = os.path.join(BASE_DIR, "split1", "test.txt")

WINDOW_SIZE  = 64    # frames per window
STRIDE       = 32    # overlap
NUM_CLASSES  = 10
NUM_JOINTS   = 17
NUM_COORDS   = 3     # x, y, z
BATCH_SIZE   = 32
EPOCHS       = 50
LR           = 1e-3
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTION_NAMES = [
    "Preparation", "Grasp Bird's Tail", "Single Whip",
    "Lift up Hand", "White Crane Spread Its Wings",
    "Brush Knee and Twist Step", "Hold the Lute",
    "Pulling Blocking and Pounding", "Apparent Close Up", "Cross Hands"
]


# Adjacency matrix - 17 joints

def build_adjacency():
    """
    17-joint human skeleton adjacency matrix.
    Joints: Pelvis(0) RHip(1) RKnee(2) RAnkle(3) LHip(4) LKnee(5)
            LAnkle(6) Spine(7) Neck(8) Head(9) LShoulder(10) LElbow(11)
            LWrist(12) RShoulder(13) RElbow(14) RWrist(15) Chest(16)
    """
    edges = [
        (0, 1), (0, 4), (0, 7),   # Pelvis connections
        (1, 2), (2, 3),            # Right leg
        (4, 5), (5, 6),            # Left leg
        (7, 8), (7, 16),           # Spine connections
        (8, 9),                    # Neck - Head
        (8, 10), (8, 13),          # Neck - Shoulders
        (10, 11), (11, 12),        # Left arm
        (13, 14), (14, 15),        # Right arm
        (16, 10), (16, 13),        # Chest - Shoulders
    ]
    adj = torch.zeros(NUM_JOINTS, NUM_JOINTS)
    for i, j in edges:
        adj[i, j] = 1
        adj[j, i] = 1
    return adj


def normalized_adjacency(adj):
    """D^{-1/2} A_tilde D^{-1/2} normalization."""
    A_tilde = adj + torch.eye(adj.size(0), device=adj.device)
    D_vec = A_tilde.sum(dim=1)
    D_inv_sqrt = torch.diag(D_vec.pow(-0.5))
    return D_inv_sqrt @ A_tilde @ D_inv_sqrt



# Dataset

class TaiChiSTGCNDataset(Dataset):
    """
    Loads HIT TaiChi skeleton data for STGCN.
    Feature shape per file: (17, 3, T)
    STGCN input shape:      (3, 17, window_size) = (C, N, T)
    """
    def __init__(self, file_list_path, feature_dir, label_dir,
                 window_size=64, stride=32, override_files=None):
        self.windows = []

        if override_files is not None:
            filenames = override_files
        else:
            with open(file_list_path, "r") as f:
                filenames = [line.strip() for line in f if line.strip()]

        for fname in filenames:
            feat_path  = os.path.join(feature_dir, fname)
            label_path = os.path.join(label_dir,   fname)

            if not os.path.exists(feat_path) or not os.path.exists(label_path):
                continue

            feat     = np.load(feat_path).astype(np.float32)   # (17, 3, T)
            labels   = np.load(label_path).astype(np.int64)    # (1, T)
            T        = feat.shape[2]
            labels_T = labels[0]                                # (T,)

            for start in range(0, T - window_size + 1, stride):
                end  = start + window_size
                # Reshape: (17, 3, W) → (3, 17, W) = (C, N, T)
                x    = feat[:, :, start:end].transpose(1, 0, 2)  # (3, 17, W)
                y    = int(np.bincount(labels_T[start:end]).argmax())
                self.windows.append((x, y))

        print(f"{len(self.windows)} windows loaded")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x, y = self.windows[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)



# STGCN model

class TemporalConvBlock(nn.Module):
    """1D causal conv over time for each node independently."""
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,
                               kernel_size=kernel_size, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        B, C, N, T = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(B * N, C, T)
        x = self.conv(x)
        T_out = x.size(-1)
        x = x.view(B, N, -1, T_out).permute(0, 2, 1, 3).contiguous()
        return self.relu(self.bn(x))


class GraphConvBlock(nn.Module):
    """Spatial graph convolution: H = σ(A_hat X Θ)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.theta = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn    = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x, A_hat):
        out = torch.einsum('nm, bcmt -> bcnt', A_hat, x)
        return self.relu(self.bn(self.theta(out)))


class STBlock(nn.Module):
    """TCN → GCN → TCN with residual connection."""
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.tcn1 = TemporalConvBlock(in_channels,    hidden_channels, kernel_size)
        self.gcn  = GraphConvBlock(hidden_channels,   hidden_channels)
        self.tcn2 = TemporalConvBlock(hidden_channels, out_channels,   kernel_size)

        self.residual_conv = nn.Sequential()
        if in_channels != out_channels:
            self.residual_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A_hat):
        residual      = self.residual_conv(x)
        out           = self.tcn1(x)
        out           = self.gcn(out, A_hat)
        out           = self.tcn2(out)
        time_cut      = 2 * (self.kernel_size - 1)
        residual      = residual[:, :, :, time_cut:]
        return self.relu(out + residual)


class TaiChiSTGCN(nn.Module):
    """
    STGCN adapted for Tai Chi action classification.
    Input:  (B, 3, 17, T)  — coords × joints × frames
    Output: (B, num_classes)
    """
    def __init__(self, num_classes, adj, window_size=64,
                 n_blocks=2, channels=None, kernel_size=3):
        super().__init__()

        if channels is None:
            channels = [(3, 16, 64), (64, 16, 128)]

        self.register_buffer("A_hat", normalized_adjacency(adj))

        self.blocks = nn.ModuleList()
        T = window_size
        for (c_in, c_hid, c_out) in channels:
            self.blocks.append(STBlock(c_in, c_hid, c_out, kernel_size))
            T = T - 2 * (kernel_size - 1)

        final_channels = channels[-1][2]

        # Classification head — global average pool then FC
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(final_channels, num_classes)

    def forward(self, x):
        # x: (B, 3, 17, T)
        for block in self.blocks:
            x = block(x, self.A_hat)
        # x: (B, C, 17, T_remaining)
        x = self.pool(x)          # (B, C, 1, 1)
        x = x.flatten(1)          # (B, C)
        x = self.dropout(x)
        return self.fc(x)         # (B, num_classes)



# Training and evaluation

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
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
            x, y  = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, f1, all_preds, all_labels


def run_training(model, train_loader, test_loader, model_name="stgcn"):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "test_acc": [], "test_f1": []}
    best_f1  = 0
    save_path = os.path.join(r"C:\Users\lkorn\Documents\Deep Learning\Project", f"best_{model_name}.pth")

    print(f"\nTraining: {model_name}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc   = train_epoch(model, train_loader, optimizer, criterion)
        test_acc, test_f1, _, _ = evaluate(model, test_loader)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["test_f1"].append(test_f1)

        print(f"  Epoch {epoch:02d}/{EPOCHS} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Test Acc: {test_acc:.4f} | "
              f"Test F1: {test_f1:.4f}")

        if test_f1 > best_f1:
            best_f1 = test_f1
            torch.save(model.state_dict(), save_path)
            print(f"    Saved best model (F1={best_f1:.4f})")

    model.load_state_dict(torch.load(save_path))
    return history, best_f1



# Visualizations

def plot_training_curve(history, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history["train_loss"], color="steelblue")
    axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss"); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["test_acc"], color="steelblue")
    axes[1].set_title("Test Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy"); axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Training Curves — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"training_curves_{model_name}.png", dpi=150)
    plt.show()
    print(f"Saved: training_curves_{model_name}.png")


def plot_confusion_matrix(labels, preds, title="Confusion Matrix — STGCN"):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=ACTION_NAMES, yticklabels=ACTION_NAMES)
    plt.title(title)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig("confusion_matrix_stgcn.png", dpi=150)
    plt.show()
    print("Saved: confusion_matrix_stgcn.png")


def plot_skeleton_graph():
    """Visualize the 17-joint skeleton graph used in STGCN."""
    joints = ['Pelvis','RHip','RKnee','RAnkle','LHip','LKnee','LAnkle',
              'Spine','Neck','Head','LShoulder','LElbow','LWrist',
              'RShoulder','RElbow','RWrist','Chest']

    # Approximate 2D positions for visualization
    pos = {
        0: (0, 0),    # Pelvis
        1: (1, -1),   # RHip
        2: (1, -2),   # RKnee
        3: (1, -3),   # RAnkle
        4: (-1, -1),  # LHip
        5: (-1, -2),  # LKnee
        6: (-1, -3),  # LAnkle
        7: (0, 1),    # Spine
        8: (0, 2),    # Neck
        9: (0, 3),    # Head
        10: (-1, 2),  # LShoulder
        11: (-2, 1),  # LElbow
        12: (-3, 0),  # LWrist
        13: (1, 2),   # RShoulder
        14: (2, 1),   # RElbow
        15: (3, 0),   # RWrist
        16: (0, 1.5), # Chest
    }
    edges = [(0,1),(0,4),(0,7),(1,2),(2,3),(4,5),(5,6),(7,8),(7,16),
             (8,9),(8,10),(8,13),(10,11),(11,12),(13,14),(14,15),(16,10),(16,13)]

    fig, ax = plt.subplots(figsize=(8, 10))
    for i, j in edges:
        x = [pos[i][0], pos[j][0]]
        y = [pos[i][1], pos[j][1]]
        ax.plot(x, y, 'b-', linewidth=2)
    for idx, (x, y) in pos.items():
        ax.scatter(x, y, s=200, c='steelblue', zorder=5)
        ax.annotate(f"{idx}:{joints[idx]}", (x, y),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.set_title("17-Joint Skeleton Graph (STGCN Adjacency)", fontsize=13)
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("skeleton_graph.png", dpi=150)
    plt.show()
    print("Saved: skeleton_graph.png")


def ablation_summary(results):
    """Compare all models."""
    print("\n" + "-"*60)
    print("Full Ablation Study Results")
    print("-"*60)
    print(f"{'Model':<30} {'Top-1 Acc':>10} {'Macro F1':>10}")
    print("-"*60)
    for name, acc, f1 in results:
        print(f"{name:<30} {acc:>10.4f} {f1:>10.4f}")
    print("-"*60)

    models = [r[0] for r in results]
    accs   = [r[1] for r in results]
    f1s    = [r[2] for r in results]
    x      = np.arange(len(models))
    colors = ["tomato", "steelblue", "green"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(x, accs, color=colors, edgecolor="black")
    axes[0].set_xticks(x); axes[0].set_xticklabels(models, rotation=10)
    axes[0].set_ylabel("Top-1 Accuracy"); axes[0].set_title("Accuracy — Ablation Study")
    axes[0].set_ylim(0, 1); axes[0].grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.01, f"{v:.4f}", ha="center", fontweight="bold")

    axes[1].bar(x, f1s, color=colors, edgecolor="black")
    axes[1].set_xticks(x); axes[1].set_xticklabels(models, rotation=10)
    axes[1].set_ylabel("Macro F1"); axes[1].set_title("F1 Score — Ablation Study")
    axes[1].set_ylim(0, 1); axes[1].grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(f1s):
        axes[1].text(i, v + 0.01, f"{v:.4f}", ha="center", fontweight="bold")

    plt.suptitle("Full Ablation Study — BiLSTM vs BiLSTM+Attention vs STGCN",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("ablation_study_full.png", dpi=150)
    plt.show()
    print("Saved: ablation_study_full.png")



# Main

def main():
    print(f"Device: {DEVICE}\n")

    # Build adjacency matrix
    print("Building 17-joint skeleton graph...")
    adj = build_adjacency()
    print(f"   Adjacency matrix: {adj.shape}, {int(adj.sum()//2)} edges\n")

    # Visualize skeleton
    plot_skeleton_graph()

    # Load datasets
    print("Loading datasets...")
    test_files = sorted(
        [f for f in os.listdir(FEATURE_DIR) if f.startswith("test")],
        key=lambda x: int(x.replace("test_s", "").replace(".npy", ""))
    )
    print(f"  Auto-detected {len(test_files)} test files")

    train_dataset = TaiChiSTGCNDataset(TRAIN_LIST, FEATURE_DIR, LABEL_DIR,
                                        WINDOW_SIZE, STRIDE)
    test_dataset  = TaiChiSTGCNDataset(TEST_LIST, FEATURE_DIR, LABEL_DIR,
                                        WINDOW_SIZE, STRIDE,
                                        override_files=test_files)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=0)
    test_loader   = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=0)
    print(f"  Train: {len(train_dataset)} | Test: {len(test_dataset)}\n")

    # Train STGCN
    model = TaiChiSTGCN(
        num_classes=NUM_CLASSES,
        adj=adj,
        window_size=WINDOW_SIZE,
        n_blocks=2,
        channels=[(3, 16, 64), (64, 16, 128)],
        kernel_size=3
    ).to(DEVICE)

    history, best_f1 = run_training(model, train_loader, test_loader, "stgcn")
    acc, f1, preds, labels = evaluate(model, test_loader)

    # Visualizations
    print("\nGenerating visualizations...")
    plot_training_curve(history, "STGCN")
    plot_confusion_matrix(labels, preds)

    # Final results — include previous BiLSTM results for comparison
    results = [
        ("BiLSTM (No Attention)", 0.8925, 0.8825),
        ("BiLSTM + Attention",    0.8863, 0.8702),
        ("STGCN",                 acc,    f1),
    ]
    ablation_summary(results)

    # Final report
    print("\n" + "-"*60)
    print("Final Classification Report — STGCN")
    print("-"*60)
    print(f"Top-1 Accuracy : {acc:.4f}")
    print(f"Macro F1 Score : {f1:.4f}")
    print("\nPer-class Report:")
    print(classification_report(labels, preds,
                                  target_names=ACTION_NAMES, zero_division=0))


if __name__ == "__main__":
    main()

"""
HIT TaiChi Action Recognition — BiLSTM + Attention Model
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

# ─────────────────────────────────────────
# CONFIG — only change BASE_DIR!
# ─────────────────────────────────────────
BASE_DIR    = r"C:\Users\lkorn\Documents\Deep Learning\Project\TaiChi.tar\TaiChi"
FEATURE_DIR = os.path.join(BASE_DIR, "feature")
LABEL_DIR   = os.path.join(BASE_DIR, "label")
TRAIN_LIST  = os.path.join(BASE_DIR, "split1", "train.txt")
TEST_LIST   = os.path.join(BASE_DIR, "split1", "test.txt")   # original (10 files)

WINDOW_SIZE  = 64    # frames per segment fed into LSTM
STRIDE       = 32    # overlap between windows
NUM_CLASSES  = 10    # 10 Tai Chi actions
INPUT_SIZE   = 17*3  # 17 joints × 3 coords = 51
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT      = 0.3
BATCH_SIZE   = 32
EPOCHS       = 50
LR           = 1e-3
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTION_NAMES = [
    "Preparation",
    "Grasp Bird's Tail",
    "Single Whip",
    "Lift up Hand",
    "White Crane Spread Its Wings",
    "Brush Knee and Twist Step",
    "Hold the Lute",
    "Pulling Blocking and Pounding",
    "Apparent Close Up",
    "Cross Hands"
]


# ─────────────────────────────────────────
# 0. SHAPE VERIFICATION
# ─────────────────────────────────────────
def verify_shapes():
    """Verify feature and label shapes before training."""
    print("🔍 Verifying data shapes...")
    with open(TRAIN_LIST, "r") as f:
        first_file = f.readline().strip()

    feat  = np.load(os.path.join(FEATURE_DIR, first_file))
    label = np.load(os.path.join(LABEL_DIR,   first_file))
    print(f"  Feature shape : {feat.shape}  (expected: 17, 3, T)")
    print(f"  Label shape   : {label.shape}  (expected: 1, T)")
    assert feat.shape[0] == 17 and feat.shape[1] == 3, \
        f"❌ Unexpected feature shape: {feat.shape}"
    print("  ✅ Shapes confirmed!\n")


# ─────────────────────────────────────────
# 1. CLASS DISTRIBUTION ANALYSIS
# ─────────────────────────────────────────
def plot_class_distribution():
    """Plot class distribution to check for imbalance."""
    print("📊 Analysing class distribution...")
    counter = collections.Counter()

    with open(TRAIN_LIST, "r") as f:
        filenames = [l.strip() for l in f if l.strip()]

    for fname in filenames:
        label_path = os.path.join(LABEL_DIR, fname)
        if not os.path.exists(label_path):
            continue
        labels = np.load(label_path)[0].astype(int)
        counter.update(labels.tolist())

    classes = [ACTION_NAMES[i] for i in range(NUM_CLASSES)]
    counts  = [counter.get(i, 0) for i in range(NUM_CLASSES)]

    plt.figure(figsize=(12, 5))
    bars = plt.bar(classes, counts, color="steelblue", edgecolor="black")
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.ylabel("Number of Frames")
    plt.title("Class Distribution in Training Set (Frame-Level)")
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 50, str(count),
                 ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=150)
    plt.show()
    print("  ✅ Saved: class_distribution.png\n")
    return counts


# ─────────────────────────────────────────
# 2. DATASET
# ─────────────────────────────────────────
class TaiChiDataset(Dataset):
    """
    Loads HIT TaiChi skeleton data.
    Feature shape per file: (17, 3, T)  — joints × coords × frames
    Label shape per file:   (1, T)      — frame-level class labels
    Slices into fixed-length windows for LSTM input.
    """
    def __init__(self, file_list_path, feature_dir, label_dir,
                 window_size=64, stride=32, override_files=None):
        self.windows = []
        self.window_size = window_size

        # Use override_files if provided, else read from txt
        if override_files is not None:
            filenames = override_files
        else:
            with open(file_list_path, "r") as f:
                filenames = [line.strip() for line in f if line.strip()]

        for fname in filenames:
            feat_path  = os.path.join(feature_dir, fname)
            label_path = os.path.join(label_dir,   fname)

            if not os.path.exists(feat_path) or not os.path.exists(label_path):
                print(f"⚠️  Missing: {fname} — skipping")
                continue

            feat     = np.load(feat_path).astype(np.float32)   # (17, 3, T)
            labels   = np.load(label_path).astype(np.int64)    # (1, T)
            T        = feat.shape[2]
            feat_T   = feat.reshape(-1, T).T                    # (T, 51)
            labels_T = labels[0]                                # (T,)

            for start in range(0, T - window_size + 1, stride):
                end = start + window_size
                x   = feat_T[start:end]
                y   = int(np.bincount(labels_T[start:end]).argmax())
                self.windows.append((x, y))

        print(f"  ✅ {len(self.windows)} windows from {os.path.basename(file_list_path)}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x, y = self.windows[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# ─────────────────────────────────────────
# 3. MODELS
# ─────────────────────────────────────────
class AttentionLayer(nn.Module):
    """Temporal attention over BiLSTM outputs (L7 — Attention Mechanism)."""
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_out):
        # lstm_out: (batch, seq, hidden*2)
        scores  = self.attn(lstm_out).squeeze(-1)              # (batch, seq)
        weights = torch.softmax(scores, dim=1)                 # (batch, seq)
        context = (lstm_out * weights.unsqueeze(-1)).sum(dim=1) # (batch, hidden*2)
        return context, weights


class TaiChiLSTM(nn.Module):
    """BiLSTM + Temporal Attention classifier (main model)."""
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super().__init__()
        self.lstm      = nn.LSTM(input_size, hidden_size, num_layers,
                                  batch_first=True, bidirectional=True,
                                  dropout=dropout if num_layers > 1 else 0.0)
        self.attention = AttentionLayer(hidden_size)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out, _          = self.lstm(x)
        context, weights = self.attention(out)
        context         = self.dropout(context)
        return self.fc(context), weights


class TaiChiLSTM_NoAttention(nn.Module):
    """BiLSTM baseline WITHOUT attention — for ablation study."""
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, bidirectional=True,
                                dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out, (hn, _) = self.lstm(x)
        last_hidden  = torch.cat((hn[-2], hn[-1]), dim=1)
        last_hidden  = self.dropout(last_hidden)
        return self.fc(last_hidden), None


# ─────────────────────────────────────────
# 4. TRAINING & EVALUATION
# ─────────────────────────────────────────
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
            x, y  = x.to(DEVICE), y.to(DEVICE)
            logits, _ = model(x)
            preds = logits.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, f1, all_preds, all_labels


def run_training(model, train_loader, test_loader, model_name="model"):
    """Full training loop — returns history and best results."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    history   = {"train_loss": [], "train_acc": [], "test_acc": [], "test_f1": []}
    best_f1   = 0
    save_path = f"best_{model_name}.pth"

    print(f"\n🚀 Training: {model_name}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
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
            print(f"    💾 Saved best model (F1={best_f1:.4f})")

    model.load_state_dict(torch.load(save_path))
    return history, best_f1


# ─────────────────────────────────────────
# 5. VISUALIZATIONS
# ─────────────────────────────────────────
def plot_training_curves(history_with, history_without):
    """Plot training loss curves for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    axes[0].plot(history_with["train_loss"],    label="With Attention",    color="steelblue")
    axes[0].plot(history_without["train_loss"], label="Without Attention", color="tomato", linestyle="--")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Test accuracy curve
    axes[1].plot(history_with["test_acc"],    label="With Attention",    color="steelblue")
    axes[1].plot(history_without["test_acc"], label="Without Attention", color="tomato", linestyle="--")
    axes[1].set_title("Test Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Training Curves — Ablation Study", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()
    print("✅ Saved: training_curves.png")


def plot_confusion_matrix(labels, preds, title="Confusion Matrix"):
    """Plot confusion matrix."""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=ACTION_NAMES, yticklabels=ACTION_NAMES)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fname = title.lower().replace(" ", "_") + ".png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"✅ Saved: {fname}")


def visualize_attention(model, test_loader):
    """Visualize temporal attention weights for 3 test samples."""
    model.eval()
    x_batch, y_batch = next(iter(test_loader))
    x_batch = x_batch.to(DEVICE)

    with torch.no_grad():
        _, weights = model(x_batch)   # weights: (batch, seq)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    for i in range(3):
        attn = weights[i].cpu().numpy()
        axes[i].bar(range(len(attn)), attn, color="steelblue", alpha=0.8)
        axes[i].set_title(f"Sample {i+1} — True Action: {ACTION_NAMES[y_batch[i].item()]}")
        axes[i].set_xlabel("Frame (Time Step)")
        axes[i].set_ylabel("Attention Weight")
        axes[i].grid(True, alpha=0.3)

    plt.suptitle("Temporal Attention Weights — Which Frames Matter?",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("attention_weights.png", dpi=150)
    plt.show()
    print("✅ Saved: attention_weights.png")


def ablation_summary(acc_with, f1_with, acc_without, f1_without):
    """Print and plot ablation study summary table."""
    print("\n" + "="*55)
    print("🔬 ABLATION STUDY RESULTS")
    print("="*55)
    print(f"{'Model':<30} {'Top-1 Acc':>10} {'Macro F1':>10}")
    print("-"*55)
    print(f"{'BiLSTM + Attention':<30} {acc_with:>10.4f} {f1_with:>10.4f}")
    print(f"{'BiLSTM (No Attention)':<30} {acc_without:>10.4f} {f1_without:>10.4f}")
    print(f"{'Improvement':<30} {acc_with-acc_without:>+10.4f} {f1_with-f1_without:>+10.4f}")
    print("="*55)

    # Bar chart
    models  = ["BiLSTM\n(No Attention)", "BiLSTM\n+ Attention"]
    accs    = [acc_without, acc_with]
    f1s     = [f1_without,  f1_with]
    x       = np.arange(2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].bar(x, accs, color=["tomato", "steelblue"], edgecolor="black")
    axes[0].set_xticks(x); axes[0].set_xticklabels(models)
    axes[0].set_ylabel("Top-1 Accuracy"); axes[0].set_title("Accuracy — Ablation Study")
    axes[0].set_ylim(0, 1); axes[0].grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.01, f"{v:.4f}", ha="center", fontweight="bold")

    axes[1].bar(x, f1s, color=["tomato", "steelblue"], edgecolor="black")
    axes[1].set_xticks(x); axes[1].set_xticklabels(models)
    axes[1].set_ylabel("Macro F1 Score"); axes[1].set_title("F1 Score — Ablation Study")
    axes[1].set_ylim(0, 1); axes[1].grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(f1s):
        axes[1].text(i, v + 0.01, f"{v:.4f}", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig("ablation_study.png", dpi=150)
    plt.show()
    print("✅ Saved: ablation_study.png")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    print(f"🖥️  Device: {DEVICE}\n")

    # 0. Verify shapes
    verify_shapes()

    # 1. Class distribution
    plot_class_distribution()

    # 2. Load datasets
    print("📦 Loading datasets...")
    # Auto-detect ALL test files from folder (ignores incomplete test.txt)
    test_files = sorted(
        [f for f in os.listdir(FEATURE_DIR) if f.startswith("test")],
        key=lambda x: int(x.replace("test_s", "").replace(".npy", ""))
    )
    print(f"  Auto-detected {len(test_files)} test files from folder")
    train_dataset = TaiChiDataset(TRAIN_LIST,  FEATURE_DIR, LABEL_DIR, WINDOW_SIZE, STRIDE)
    test_dataset  = TaiChiDataset(TEST_LIST,   FEATURE_DIR, LABEL_DIR, WINDOW_SIZE, STRIDE,
                                   override_files=test_files)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"  Train windows: {len(train_dataset)} | Test windows: {len(test_dataset)}\n")

    # 3. Train WITH attention (main model)
    model_with = TaiChiLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
                              NUM_CLASSES, DROPOUT).to(DEVICE)
    history_with, best_f1_with = run_training(model_with, train_loader,
                                               test_loader, "with_attention")
    acc_with, f1_with, preds_with, labels_with = evaluate(model_with, test_loader)

    # 4. Train WITHOUT attention (ablation baseline)
    model_without = TaiChiLSTM_NoAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
                                            NUM_CLASSES, DROPOUT).to(DEVICE)
    history_without, _ = run_training(model_without, train_loader,
                                       test_loader, "without_attention")
    acc_without, f1_without, _, _ = evaluate(model_without, test_loader)

    # 5. Visualizations
    print("\n📊 Generating visualizations...")
    plot_training_curves(history_with, history_without)
    plot_confusion_matrix(labels_with, preds_with,
                          "Confusion Matrix — BiLSTM + Attention")
    visualize_attention(model_with, test_loader)
    ablation_summary(acc_with, f1_with, acc_without, f1_without)

    # 6. Final report
    print("\n" + "="*60)
    print("📈 FINAL CLASSIFICATION REPORT — BiLSTM + Attention")
    print("="*60)
    print(f"Top-1 Accuracy : {acc_with:.4f}")
    print(f"Macro F1 Score : {f1_with:.4f}")
    print("\nPer-class Report:")
    print(classification_report(labels_with, preds_with,
                                  target_names=ACTION_NAMES, zero_division=0))

    print("\n📁 Output files saved:")
    for f in ["class_distribution.png", "training_curves.png",
              "attention_weights.png", "ablation_study.png",
              "confusion_matrix_-_bilstm_+_attention.png"]:
        print(f"   • {f}")


if __name__ == "__main__":
    main()
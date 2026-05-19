"""
Standalone analysis script for the fine-tuned CLIP model on the RSICD dataset.

Produces:
  - Terminal output: pre-trained and fine-tuned top-1 accuracy
  - classification_report.txt: per-class precision / recall / F1
  - confusion_matrix.png: normalized confusion matrix heatmap
  - Terminal summary: top-5 and bottom-5 classes by F1, confusion info for worst 5

Dependencies (install if needed):
    pip install scikit-learn matplotlib
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
import open_clip
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

from eval_utils import load_class_map, TEMPLATES

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT    = Path(".")
IMAGE_DIR    = DATA_ROOT / "RSICD_images"
JSON_PATH    = DATA_ROOT / "dataset_rsicd.json"
CLASSES_DIR  = DATA_ROOT / "txtclasses_rsicd"
WEIGHTS_PATH = DATA_ROOT / "finetuned_clip.pt"

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_test_filenames(json_path):
    """Return filenames for all entries whose split == 'test'."""
    with open(json_path) as f:
        data = json.load(f)
    return [img["filename"] for img in data["images"] if img["split"] == "test"]


def build_text_embeddings(model, tokenizer, class_names):
    """Average and re-normalize embeddings over the three prompt templates."""
    embeddings = []
    with torch.no_grad():
        for cls in class_names:
            prompts = [t.format(cls.replace("_", " ")) for t in TEMPLATES]
            tokens = tokenizer(prompts).to(device)
            embs = F.normalize(model.encode_text(tokens), dim=-1).mean(dim=0)
            embeddings.append(F.normalize(embs, dim=0))
    return torch.stack(embeddings)  # (num_classes, embed_dim)


def run_eval(model, tokenizer, preprocess, test_filenames,
             class_names, filename_to_class, collect_preds=False):
    """
    Zero-shot classify the full test set.
    Returns (accuracy, y_true, y_pred).  y_true/y_pred are empty when
    collect_preds=False.
    """
    model.eval()
    text_embs = build_text_embeddings(model, tokenizer, class_names)

    correct = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for fname in test_filenames:
            true_cls = filename_to_class.get(fname, "unknown")
            img = preprocess(
                Image.open(IMAGE_DIR / fname).convert("RGB")
            ).unsqueeze(0).to(device)

            img_emb   = F.normalize(model.encode_image(img), dim=-1)
            pred_idx  = (img_emb @ text_embs.T).squeeze(0).argmax().item()
            pred_cls  = class_names[pred_idx]

            correct += pred_cls == true_cls
            if collect_preds:
                y_true.append(true_cls)
                y_pred.append(pred_cls)

    acc = correct / len(test_filenames) if test_filenames else 0.0
    return acc, y_true, y_pred


def main():
    print(f"Device: {device}")

    class_names, filename_to_class = load_class_map(CLASSES_DIR)
    test_filenames = load_test_filenames(JSON_PATH)
    print(f"Test images : {len(test_filenames)}")
    print(f"Classes     : {len(class_names)}")

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    # ── 1. Pre-trained baseline ────────────────────────────────────────────────
    print("\n[1/2] Pre-trained baseline (no fine-tuning)...")
    model_base, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model_base = model_base.to(device)

    base_acc, _, _ = run_eval(
        model_base, tokenizer, preprocess,
        test_filenames, class_names, filename_to_class,
        collect_preds=False,
    )
    print(f"Pre-trained baseline top-1 accuracy : {base_acc:.4f}  ({base_acc:.1%})")

    del model_base
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── 2. Fine-tuned eval ────────────────────────────────────────────────────
    print("\n[2/2] Fine-tuned model...")
    model_ft, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    state_dict = torch.load(WEIGHTS_PATH, map_location=device)
    model_ft.load_state_dict(state_dict)
    model_ft = model_ft.to(device)

    ft_acc, y_true, y_pred = run_eval(
        model_ft, tokenizer, preprocess,
        test_filenames, class_names, filename_to_class,
        collect_preds=True,
    )
    print(f"Fine-tuned model top-1 accuracy     : {ft_acc:.4f}  ({ft_acc:.1%})")

    # ── 3. Classification report ──────────────────────────────────────────────
    print("\n--- Classification Report ---")
    report_str = classification_report(y_true, y_pred, digits=3)
    print(report_str)

    with open("classification_report.txt", "w") as f:
        f.write(report_str)
    print("Saved: classification_report.txt")

    # ── 4. Confusion matrix figure ────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=class_names, normalize="true")

    fig, ax = plt.subplots(figsize=(12, 12), dpi=200)
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_title("Normalized Confusion Matrix")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    fig.savefig("confusion_matrix.png")
    plt.close(fig)
    print("Saved: confusion_matrix.png")

    # ── 5. Best / worst class summary ─────────────────────────────────────────
    report_dict = classification_report(y_true, y_pred, digits=3, output_dict=True)

    # Keep only entries that correspond to actual class names
    class_f1 = {
        cls: report_dict[cls]["f1-score"]
        for cls in class_names
        if cls in report_dict
    }
    sorted_f1 = sorted(class_f1.items(), key=lambda x: x[1], reverse=True)

    best5  = sorted_f1[:5]
    worst5 = sorted_f1[-5:]

    print("\n--- Top 5 classes by F1 ---")
    for cls, f1 in best5:
        print(f"  {cls:<30}  F1 = {f1:.3f}")

    print("\n--- Bottom 5 classes by F1 ---")
    for cls, f1 in worst5:
        print(f"  {cls:<30}  F1 = {f1:.3f}")

    print("\n--- Confusion analysis for bottom 5 classes ---")
    for cls, _ in worst5:
        row_idx = class_names.index(cls)
        row = cm[row_idx].copy()
        row[row_idx] = 0.0          # exclude the diagonal (correct predictions)
        confused_idx = int(np.argmax(row))
        confused_cls = class_names[confused_idx]
        confused_pct = row[confused_idx] * 100
        print(f"  {cls} -> most often confused with {confused_cls} ({confused_pct:.1f}% of cases)")


if __name__ == "__main__":
    main()

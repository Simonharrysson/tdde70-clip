import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import (ImageDataset, get_clip_transform,
                     load_class_mapping, get_rsicd_class_names,
                     get_test_filenames, get_test_data)


# ── Shared helper ────────────────────────────────────────────────────────────

def build_text_embeddings(model, tokenizer, class_names, templates, device):
    """Average text embeddings over multiple prompt templates per class."""
    model.eval()
    embeddings = []
    with torch.no_grad():
        for cls in class_names:
            prompts = [t.format(cls.replace('_', ' ')) for t in templates]
            tokens = tokenizer(prompts).to(device)
            embs = model.encode_text(tokens)
            embs = F.normalize(embs, dim=-1).mean(dim=0)
            embeddings.append(F.normalize(embs, dim=0))
    return torch.stack(embeddings)   # (n_classes, D)


def encode_images(model, loader, device):
    """Encode all images in a DataLoader, returns (embeddings, filenames)."""
    model.eval()
    all_embs, all_fnames = [], []
    with torch.no_grad():
        for images, fnames in loader:
            images = images.to(device)
            embs = F.normalize(model.encode_image(images), dim=-1)
            all_embs.append(embs.cpu())
            all_fnames.extend(fnames)
    return torch.cat(all_embs), all_fnames


# ── Task 1: Zero-shot classification ────────────────────────────────────────

def zero_shot_accuracy(model, tokenizer, cfg, device, split='test'):
    """Classify Split C images into RSICD classes via cosine similarity."""
    class_names = get_rsicd_class_names(cfg.classes_path)
    text_embs = build_text_embeddings(
        model, tokenizer, class_names, cfg.prompt_templates, device
    )
    class_map = load_class_mapping(cfg.classes_path)
    filenames = get_test_filenames(cfg.json_path)

    transform = get_clip_transform(cfg.image_size)
    dataset = ImageDataset(cfg.image_path, filenames, transform)
    loader = DataLoader(dataset, batch_size=128, num_workers=2)

    img_embs, fnames = encode_images(model, loader, device)
    logits = img_embs @ text_embs.T.cpu()
    preds = logits.argmax(dim=-1)

    correct = sum(
        class_map.get(fname, '') == class_names[preds[i].item()]
        for i, fname in enumerate(fnames)
        if fname in class_map
    )
    total = sum(1 for fname in fnames if fname in class_map)
    return correct / total if total > 0 else 0.0


def zero_shot_detailed(model, tokenizer, cfg, device):
    """Returns per-class accuracy, confusion matrix data, and overall metrics."""
    from collections import defaultdict
    import numpy as np

    class_names = get_rsicd_class_names(cfg.classes_path)
    text_embs = build_text_embeddings(
        model, tokenizer, class_names, cfg.prompt_templates, device
    )
    class_map = load_class_mapping(cfg.classes_path)
    filenames = get_test_filenames(cfg.json_path)
    n_cls = len(class_names)

    transform = get_clip_transform(cfg.image_size)
    dataset = ImageDataset(cfg.image_path, filenames, transform)
    loader = DataLoader(dataset, batch_size=128, num_workers=2)
    img_embs, fnames = encode_images(model, loader, device)
    logits = img_embs @ text_embs.T.cpu()
    preds = logits.argmax(dim=-1)

    confusion = torch.zeros(n_cls, n_cls, dtype=torch.long)
    cls_idx = {c: i for i, c in enumerate(class_names)}

    for i, fname in enumerate(fnames):
        if fname not in class_map:
            continue
        true_i = cls_idx.get(class_map[fname])
        pred_i = preds[i].item()
        if true_i is not None:
            confusion[true_i, pred_i] += 1

    per_class_acc = confusion.diagonal().float() / confusion.sum(dim=1).clamp(min=1).float()
    overall_acc = confusion.diagonal().sum().item() / confusion.sum().item()

    # Macro F1
    tp = confusion.diagonal().float()
    fp = confusion.sum(dim=0).float() - tp
    fn = confusion.sum(dim=1).float() - tp
    precision = tp / (tp + fp).clamp(min=1e-8)
    recall = tp / (tp + fn).clamp(min=1e-8)
    f1_per_class = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
    macro_f1 = f1_per_class.mean().item()

    return {
        'accuracy': overall_acc,
        'macro_f1': macro_f1,
        'per_class_acc': {c: per_class_acc[i].item() for i, c in enumerate(class_names)},
        'confusion_matrix': confusion.numpy(),
        'class_names': class_names,
    }


# ── Task 2 & 3: Retrieval ────────────────────────────────────────────────────

def retrieval_eval(model, tokenizer, cfg, device, ks=(1, 5, 10)):
    """
    Image-to-text and text-to-image retrieval on Split C.
    Ground truth: each image has 5 captions; captions belong to exactly one image.
    Returns Recall@k for both directions.
    """
    model.eval()
    test_data = get_test_data(cfg.json_path)     # [(filename, [cap1..cap5]), ...]
    filenames = [d[0] for d in test_data]
    all_captions = [cap for _, caps in test_data for cap in caps]  # 5 per image

    transform = get_clip_transform(cfg.image_size)
    dataset = ImageDataset(cfg.image_path, filenames, transform)
    loader = DataLoader(dataset, batch_size=128, num_workers=2)
    img_embs, _ = encode_images(model, loader, device)

    # Encode captions
    txt_embs = []
    with torch.no_grad():
        for i in range(0, len(all_captions), 128):
            batch = all_captions[i:i + 128]
            tokens = tokenizer(batch).to(device)
            embs = F.normalize(model.encode_text(tokens), dim=-1)
            txt_embs.append(embs.cpu())
    txt_embs = torch.cat(txt_embs)       # (N_img * 5, D)

    sim = img_embs @ txt_embs.T          # (N_img, N_txt)
    n_imgs = len(filenames)
    n_caps = 5

    def recall_at_k(scores, gt_indices, k):
        topk = scores.topk(k).indices.tolist()
        return int(any(j in gt_indices for j in topk))

    results = {}
    max_k = max(ks)

    # Image → text
    i2t = {k: 0 for k in ks}
    for i in range(n_imgs):
        gt = set(range(i * n_caps, (i + 1) * n_caps))
        topk = sim[i].topk(max_k).indices.tolist()
        for k in ks:
            if any(j in gt for j in topk[:k]):
                i2t[k] += 1
    results['i2t'] = {f'R@{k}': i2t[k] / n_imgs for k in ks}

    # Text → image
    sim_t2i = sim.T
    t2i = {k: 0 for k in ks}
    for j in range(len(all_captions)):
        gt_img = j // n_caps
        topk = sim_t2i[j].topk(max_k).indices.tolist()
        for k in ks:
            if gt_img in topk[:k]:
                t2i[k] += 1
    n_txt = len(all_captions)
    results['t2i'] = {f'R@{k}': t2i[k] / n_txt for k in ks}

    return results


# ── Run all evaluations ──────────────────────────────────────────────────────

def run_full_eval(model, tokenizer, cfg, device):
    print("\n── Evaluation ──────────────────────────────────────────")

    details = zero_shot_detailed(model, tokenizer, cfg, device)
    print(f"Zero-shot accuracy : {details['accuracy']:.4f}")
    print(f"Macro F1           : {details['macro_f1']:.4f}")

    retrieval = retrieval_eval(model, tokenizer, cfg, device, ks=cfg.retrieval_ks)
    print("\nImage → Text retrieval:")
    for k, v in retrieval['i2t'].items():
        print(f"  {k}: {v:.4f}")
    print("Text → Image retrieval:")
    for k, v in retrieval['t2i'].items():
        print(f"  {k}: {v:.4f}")

    return {**details, 'retrieval': retrieval}


if __name__ == '__main__':
    import open_clip
    from config import Config

    cfg = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _, _ = open_clip.create_model_and_transforms(cfg.model_name, pretrained=cfg.pretrained)

    ckpt_path = 'checkpoints/best_model.pt'
    if __import__('os').path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        print(f"Loaded checkpoint (epoch {ckpt['epoch']}, acc={ckpt['acc']:.4f})")
    else:
        print("No checkpoint found — evaluating pretrained CLIP (baseline)")

    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(cfg.model_name)
    run_full_eval(model, tokenizer, cfg, device)

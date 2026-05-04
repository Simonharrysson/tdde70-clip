import os
import json
import torch
import open_clip
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import Config
from dataset import (LabeledDataset, UnlabeledDataset,
                     get_clip_transform, get_augmentation_transform)
from losses import clip_loss, simclr_loss
from evaluate import zero_shot_accuracy


def collate_labeled(batch):
    images = torch.stack([b[0] for b in batch])
    captions = [b[1] for b in batch]
    return images, captions


def train(cfg: Config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)

    # ── Model ──────────────────────────────────────────────────────────
    model, _, _ = open_clip.create_model_and_transforms(
        cfg.model_name, pretrained=cfg.pretrained
    )
    tokenizer = open_clip.get_tokenizer(cfg.model_name)
    model = model.to(device)

    # ── Data ───────────────────────────────────────────────────────────
    clip_tf = get_clip_transform(cfg.image_size)
    aug_tf = get_augmentation_transform(cfg.image_size)

    labeled_full = LabeledDataset(cfg.json_path, cfg.image_path, transform=clip_tf)
    n_val = max(1, int(cfg.val_fraction * len(labeled_full)))
    n_train = len(labeled_full) - n_val
    labeled_train, labeled_val = random_split(
        labeled_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Split B  → train: {n_train}, val: {n_val}")

    labeled_loader = DataLoader(
        labeled_train, batch_size=cfg.batch_size_labeled,
        shuffle=True, num_workers=2, pin_memory=True,
        collate_fn=collate_labeled, drop_last=True,
    )

    if cfg.semi_supervised:
        unlabeled_ds = UnlabeledDataset(cfg.json_path, cfg.image_path, augment=aug_tf)
        print(f"Split A (unlabeled): {len(unlabeled_ds)}")
        unlabeled_loader = DataLoader(
            unlabeled_ds, batch_size=cfg.batch_size_unlabeled,
            shuffle=True, num_workers=2, pin_memory=True, drop_last=True,
        )
        unlabeled_iter = iter(unlabeled_loader)

    # ── Optimizer & scheduler ──────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(labeled_loader)
    warmup_steps = int(cfg.warmup_epochs * len(labeled_loader))
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=cfg.lr * 0.01
    )

    # ── Training loop ──────────────────────────────────────────────────
    best_acc = 0.0
    log_rows = []
    step = 0

    for epoch in range(cfg.epochs):
        model.train()
        sup_total, unsup_total = 0.0, 0.0

        for images, captions in labeled_loader:
            images = images.to(device)
            tokens = tokenizer(captions).to(device)

            img_feat = model.encode_image(images)
            txt_feat = model.encode_text(tokens)
            loss_sup = clip_loss(img_feat, txt_feat, model.logit_scale)

            loss_unsup = torch.tensor(0.0, device=device)
            if cfg.semi_supervised:
                try:
                    v1, v2 = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    v1, v2 = next(unlabeled_iter)
                v1, v2 = v1.to(device), v2.to(device)
                z1 = model.encode_image(v1)
                z2 = model.encode_image(v2)
                loss_unsup = simclr_loss(z1, z2, cfg.simclr_temperature)

            loss = loss_sup + cfg.lambda_unsup * loss_unsup

            # Linear warmup
            if step < warmup_steps:
                for g in optimizer.param_groups:
                    g['lr'] = cfg.lr * (step + 1) / warmup_steps

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step >= warmup_steps:
                scheduler.step()

            sup_total += loss_sup.item()
            unsup_total += loss_unsup.item()
            step += 1

        n = len(labeled_loader)
        print(f"Epoch {epoch+1:02d}/{cfg.epochs} | "
              f"sup={sup_total/n:.4f}  unsup={unsup_total/n:.4f}")

        if (epoch + 1) % cfg.eval_every == 0:
            acc = zero_shot_accuracy(model, tokenizer, cfg, device, split='test')
            print(f"           zero-shot acc (test): {acc:.4f}")
            log_rows.append({'epoch': epoch + 1, 'sup_loss': sup_total/n,
                              'unsup_loss': unsup_total/n, 'test_acc': acc})

            if acc > best_acc:
                best_acc = acc
                ckpt_path = os.path.join(cfg.checkpoint_dir, 'best_model.pt')
                torch.save({'model_state': model.state_dict(),
                            'epoch': epoch + 1, 'acc': acc, 'cfg': cfg}, ckpt_path)
                print(f"           ✓ new best saved ({acc:.4f})")

    # Save training log
    log_path = os.path.join(cfg.results_dir, 'train_log.json')
    with open(log_path, 'w') as f:
        json.dump(log_rows, f, indent=2)
    print(f"\nBest test accuracy: {best_acc:.4f}")
    print(f"Training log saved to {log_path}")
    return model


if __name__ == '__main__':
    cfg = Config()
    train(cfg)

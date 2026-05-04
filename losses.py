import torch
import torch.nn.functional as F


def clip_loss(image_features, text_features, logit_scale):
    """Symmetric CLIP contrastive loss (standard InfoNCE)."""
    img = F.normalize(image_features, dim=-1)
    txt = F.normalize(text_features, dim=-1)
    logits = logit_scale.exp() * img @ txt.T
    labels = torch.arange(len(logits), device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2


def simclr_loss(z1, z2, temperature=0.1):
    """NT-Xent loss for two augmented views of the same image (SimCLR).

    For a batch of N unlabeled images with views z1, z2:
    - Positive pair for image i: (z1_i, z2_i)
    - All other 2N-2 samples are negatives
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    N = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)           # (2N, D)
    sim = (z @ z.T) / temperature             # (2N, 2N)

    # Exclude self-similarity from softmax denominator
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, float('-inf'))

    # Positive for sample i is at position i+N (and vice versa)
    labels = torch.cat([
        torch.arange(N, 2 * N),
        torch.arange(N),
    ]).to(z.device)

    return F.cross_entropy(sim, labels)

import torch
import torch.nn.functional as F
import open_clip
from pathlib import Path
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
import random
from dataset import make_loader
from eval_utils import load_class_map, zero_shot_accuracy

def load_split_b(json_path, val_ratio=0.1, seed=42):
    with open(json_path) as f:
        data = json.load(f)
    images = [img for img in data["images"] if img["split"] == "val"]
    random.seed(seed)
    random.shuffle(images)
    split = int((1 - val_ratio) * len(images))
    train_images = images[:split]
    val_images   = images[split:]
    train_samples = [
        (img["filename"], sentence["raw"])
        for img in train_images
        for sentence in img["sentences"]
    ]
    val_filenames = [img["filename"] for img in val_images]
    return train_samples, val_filenames

# ── Settings ─────────────────────────────────────────────────────────────────
DATA_ROOT   = Path(".")
IMAGE_DIR   = DATA_ROOT / "RSICD_images"
JSON_PATH   = DATA_ROOT / "dataset_rsicd.json"
CLASSES_DIR = DATA_ROOT / "txtclasses_rsicd"

EPOCHS      = 3
BATCH_SIZE  = 32
LR          = 1e-6

#Standard normalization values, don't change!
normalize = transforms.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)

train_preprocess = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=90),  # Satellite images are rotationally invariant
    transforms.ToTensor(),
    normalize,
])

# ── Load model ────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# Freeze everything first
for param in model.parameters():
    param.requires_grad = False

# Only unfreeze the last 3 transformer blocks in the image encoder
# and the last 3 in the text encoder
for block in list(model.visual.transformer.resblocks)[-3:]:
    for param in block.parameters():
        param.requires_grad = True

for block in list(model.transformer.resblocks)[-3:]:
    for param in block.parameters():
        param.requires_grad = True

# Unfreeze the visual projection
if model.visual.proj is not None:
    model.visual.proj.requires_grad = True

# Unfreeze the text projection
if hasattr(model, 'text_projection') and model.text_projection is not None:
    model.text_projection.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Training {trainable:,} / {total:,} parameters")

model = model.to(device)

# ── Load data ─────────────────────────────────────────────────────────────────
train_samples, val_images = load_split_b(JSON_PATH)
print(f"Train images: {len(train_samples) // 5}  Val images: {len(val_images)}")

loader = make_loader(train_samples, IMAGE_DIR, tokenizer, train_preprocess, BATCH_SIZE)
class_names, filename_to_class = load_class_map(CLASSES_DIR)

# ── Training loop ─────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS * len(loader))

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, tokens in loader:
        images = images.to(device)
        tokens = tokens.to(device)

        image_embeddings = F.normalize(model.encode_image(images), dim=-1)
        text_embeddings  = F.normalize(model.encode_text(tokens), dim=-1)

        similarity = model.logit_scale.exp() * image_embeddings @ text_embeddings.T
        labels     = torch.arange(images.shape[0]).to(device)
        loss       = (F.cross_entropy(similarity, labels) + F.cross_entropy(similarity.T, labels)) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    val_acc  = zero_shot_accuracy(model, tokenizer, val_images, IMAGE_DIR, preprocess,
                                  class_names, filename_to_class, device)
    print(f"Epoch {epoch + 1}/{EPOCHS}  loss: {avg_loss:.4f}  val_acc: {val_acc:.1%}")

# ── Save the fine-tuned model ─────────────────────────────────────────────────
torch.save(model.state_dict(), "finetuned_clip.pt")
print("Saved finetuned_clip.pt")

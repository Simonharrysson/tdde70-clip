import json
import torch
import torch.nn.functional as F
import open_clip
from PIL import Image
from pathlib import Path
from torchvision import transforms
from dataset import make_loader
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── Settings ─────────────────────────────────────────────────────────────────
DATA_ROOT   = Path(".")
IMAGE_DIR   = DATA_ROOT / "RSICD_images"
JSON_PATH   = DATA_ROOT / "dataset_rsicd.json"

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
    transforms.RandomRotation(degrees=90), # Satellite images are rotationally invariant
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

# ── Load Split B (labeled images + captions) ──────────────────────────────────
with open(JSON_PATH) as f:
    data = json.load(f)

# Split B is called "val" in the json — use all 5 captions per image
samples = [
    (img["filename"], sentence["raw"])
    for img in data["images"]
    if img["split"] == "val"
    for sentence in img["sentences"]
]
print(f"Training on {len(samples)} labeled images")

loader = make_loader(samples, IMAGE_DIR, tokenizer, train_preprocess, BATCH_SIZE)

# ── Training loop ─────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay = 0.1)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, tokens in loader:
        images = images.to(device)
        tokens = tokens.to(device)

        # Get embeddings from the model
        image_embeddings = model.encode_image(images)
        text_embeddings  = model.encode_text(tokens)

        # Normalize so we can use cosine similarity
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings  = F.normalize(text_embeddings, dim=-1)

        # Similarity matrix: how similar is each image to each caption?
        # Shape: (batch_size, batch_size)
        similarity = model.logit_scale.exp() * image_embeddings @ text_embeddings.T

        # The correct match for image i is caption i (diagonal of the matrix)
        labels = torch.arange(images.shape[0]).to(device)

        # Loss: push correct pairs together, wrong pairs apart
        loss_images   = F.cross_entropy(similarity, labels)
        loss_captions = F.cross_entropy(similarity.T, labels)
        loss = (loss_images + loss_captions) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch + 1}/{EPOCHS}  loss: {avg_loss:.4f}")

# ── Save the fine-tuned model ─────────────────────────────────────────────────
torch.save(model.state_dict(), "finetuned_clip.pt")
print("Saved finetuned_clip.pt")

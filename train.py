import json
import torch
import torch.nn.functional as F
import open_clip
from PIL import Image
from pathlib import Path

# ── Settings ─────────────────────────────────────────────────────────────────
DATA_ROOT   = Path(".")
IMAGE_DIR   = DATA_ROOT / "RSICD_images"
JSON_PATH   = DATA_ROOT / "dataset_rsicd.json"

EPOCHS      = 5
BATCH_SIZE  = 32
LR          = 1e-5

# ── Load model ────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model = model.to(device)

# ── Load Split B (labeled images + captions) ──────────────────────────────────
with open(JSON_PATH) as f:
    data = json.load(f)

# Split B is called "val" in the json
samples = [
    (img["filename"], img["sentences"][0]["raw"])
    for img in data["images"]
    if img["split"] == "val"
]
print(f"Training on {len(samples)} labeled images")

# ── Training loop ─────────────────────────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    # Go through the data in batches
    for i in range(0, len(samples), BATCH_SIZE):
        batch = samples[i : i + BATCH_SIZE]
        filenames = [s[0] for s in batch]
        captions  = [s[1] for s in batch]

        # Load and preprocess images
        images = torch.stack([
            preprocess(Image.open(IMAGE_DIR / f).convert("RGB"))
            for f in filenames
        ]).to(device)

        # Tokenize captions
        tokens = tokenizer(captions).to(device)

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
        labels = torch.arange(len(batch)).to(device)

        # Loss: push correct pairs together, wrong pairs apart
        loss_images   = F.cross_entropy(similarity, labels)
        loss_captions = F.cross_entropy(similarity.T, labels)
        loss = (loss_images + loss_captions) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / (len(samples) / BATCH_SIZE)
    print(f"Epoch {epoch + 1}/{EPOCHS}  loss: {avg_loss:.4f}")

# ── Save the fine-tuned model ─────────────────────────────────────────────────
torch.save(model.state_dict(), "finetuned_clip.pt")
print("Saved finetuned_clip.pt")

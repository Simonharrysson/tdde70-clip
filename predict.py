import torch
import torch.nn.functional as F
import open_clip
from PIL import Image
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
IMAGE_DIR = Path("leaderboard_images")

CLASS_NAMES = [
    "agricultural", "airplane", "baseballdiamond", "beach", "buildings",
    "chaparral", "denseresidential", "forest", "freeway", "golfcourse",
    "harbor", "intersection", "mediumresidential", "mobilehomepark", "overpass",
    "parkinglot", "river", "runway", "sparseresidential", "storagetanks", "tenniscourt"
]

# ── Load model ────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model.load_state_dict(torch.load("finetuned_clip.pt", map_location=device))
model = model.to(device).eval()

# ── Build text embeddings for each class ──────────────────────────────────────
templates = [
    "a satellite image of {}",
    "an aerial image of {}",
    "a remote sensing image of {}",
]

with torch.no_grad():
    text_embeddings = []
    for cls in CLASS_NAMES:
        prompts = [t.format(cls.replace("_", " ")) for t in templates]
        tokens = tokenizer(prompts).to(device)
        embs = model.encode_text(tokens)
        embs = F.normalize(embs, dim=-1).mean(dim=0)
        text_embeddings.append(F.normalize(embs, dim=0))
    text_embeddings = torch.stack(text_embeddings)  # (21, D)

# ── Predict ───────────────────────────────────────────────────────────────────
image_paths = sorted(IMAGE_DIR.glob("*.jpg")) + sorted(IMAGE_DIR.glob("*.png"))
print(f"Found {len(image_paths)} images")

predictions = []
for i, path in enumerate(image_paths):
    if i % 100 == 0:
        print(f"  {i}/{len(image_paths)}...")

    image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        img_emb = F.normalize(model.encode_image(image), dim=-1)
        scores = (img_emb @ text_embeddings.T).squeeze(0)

    pred_class = CLASS_NAMES[scores.argmax().item()]
    predictions.append(f"{path.name} {pred_class}")

# ── Write predictions.txt ─────────────────────────────────────────────────────
with open("predictions.txt", "w") as f:
    f.write("\n".join(predictions) + "\n")

print(f"Saved predictions.txt with {len(predictions)} entries")

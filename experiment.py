import json
import torch
import torch.nn.functional as F
import open_clip
from PIL import Image
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
DATA_ROOT    = Path(".")
IMAGE_DIR    = DATA_ROOT / "RSICD_images"
JSON_PATH    = DATA_ROOT / "dataset_rsicd.json"
CLASSES_DIR  = DATA_ROOT / "txtclasses_rsicd"
N_SAMPLES    = 20   # how many test images to run

# ── Load class names and filename→class mapping ──────────────────────────────
class_names = sorted(p.stem.lower() for p in CLASSES_DIR.glob("*.txt"))

filename_to_class = {}
for txt_file in CLASSES_DIR.glob("*.txt"):
    cls = txt_file.stem.lower()
    for line in txt_file.read_text().splitlines():
        fname = line.strip()
        if fname:
            filename_to_class[fname] = cls

# ── Load model ───────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model = model.to(device).eval()

# ── Build text embeddings for each class ─────────────────────────────────────
templates = [
    "a satellite image of {}",
    "an aerial image of {}",
    "a remote sensing image of {}",
]

with torch.no_grad():
    text_embeddings = []
    for cls in class_names:
        prompts = [t.format(cls.replace("_", " ")) for t in templates]
        tokens = tokenizer(prompts).to(device)
        embs = model.encode_text(tokens)
        embs = F.normalize(embs, dim=-1).mean(dim=0)
        text_embeddings.append(F.normalize(embs, dim=0))
    text_embeddings = torch.stack(text_embeddings)  # (n_classes, D)

print(f"Classes ({len(class_names)}): {class_names}\n")

# ── Run on N_SAMPLES test images ─────────────────────────────────────────────
with open(JSON_PATH) as f:
    data = json.load(f)

test_images = [img for img in data["images"] if img["split"] == "test"][:N_SAMPLES]

correct = 0
for img_data in test_images:
    fname = img_data["filename"]
    true_class = filename_to_class.get(fname, "unknown")

    image = preprocess(Image.open(IMAGE_DIR / fname).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        img_emb = F.normalize(model.encode_image(image), dim=-1)
        scores = (img_emb @ text_embeddings.T).squeeze(0)

    pred_idx = scores.argmax().item()
    pred_class = class_names[pred_idx]
    top3 = scores.topk(3)

    hit = pred_class == true_class
    correct += hit
    mark = "✓" if hit else "✗"
    print(f"{mark} {fname}")
    print(f"  true: {true_class}  |  pred: {pred_class}")
    top3_str = "  |  ".join(
        f"{class_names[i]} ({scores[i]:.2f})" for i in top3.indices
    )
    print(f"  top3: {top3_str}\n")

print(f"Accuracy on {N_SAMPLES} samples: {correct}/{N_SAMPLES} = {correct/N_SAMPLES:.0%}")

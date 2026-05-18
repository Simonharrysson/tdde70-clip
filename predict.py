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

# Generic templates applied to every class
GENERIC_TEMPLATES = [
    "a satellite image of {}",
    "an aerial image of {}",
    "a remote sensing image of {}",
    "an aerial photograph of {}",
    "an overhead view of {}",
]

# Class-specific prompts. These are tailored to RSICD's actual caption style
# (short, declarative, plain English) and add discriminating cues for the
# classes the baseline model confuses.
CLASS_PROMPTS = {
    "agricultural": [
        "a satellite image of agricultural land",
        "an aerial image of farmland with crop fields",
        "an aerial view of a patchwork of green and brown crop fields",
        "a remote sensing image of cultivated farmland",
    ],
    "airplane": [
        "a satellite image of an airplane",
        "an aerial image of airplanes parked at an airport",
        "many planes are parked in an airport",
        "an aerial view of aircraft on a tarmac",
    ],
    "baseballdiamond": [
        "a satellite image of a baseball diamond",
        "an aerial view of a fan-shaped baseball field",
        "an aerial image of a baseball field with a diamond-shaped infield",
    ],
    "beach": [
        "a satellite image of a beach",
        "an aerial image of a sandy beach next to the ocean",
        "white waves in an ocean are near a sandy beach",
        "a coastline with a strip of sand between water and land",
    ],
    "buildings": [
        "a satellite image of buildings",
        "an aerial image of a cluster of urban buildings",
        "several large buildings in an urban area",
        "an overhead view of commercial and office buildings",
    ],
    "chaparral": [
        "a satellite image of chaparral",
        "an aerial image of arid land covered in shrubs and bushes",
        "dry scrubland with low bushes and sparse vegetation",
        "an overhead view of dense shrubs covering rolling hills",
        "a remote sensing image of bushy shrubland",
    ],
    "denseresidential": [
        "a satellite image of a dense residential area",
        "an aerial image of densely packed houses with small yards",
        "a densely populated residential neighborhood",
        "rows of closely spaced houses with narrow streets between them",
        "a tightly packed suburban neighborhood seen from above",
    ],
    "forest": [
        "a satellite image of a forest",
        "an aerial image of a dense green forest",
        "a continuous canopy of trees covering the ground",
        "an overhead view of woodland",
    ],
    "freeway": [
        "a satellite image of a freeway",
        "an aerial image of a multi-lane highway",
        "a long highway with several lanes of traffic",
        "an overhead view of a highway cutting through the landscape",
    ],
    "golfcourse": [
        "a satellite image of a golf course",
        "an aerial image of a golf course with manicured green fairways and sand bunkers",
        "an overhead view of a golf course with greens, fairways, and sand traps",
    ],
    "harbor": [
        "a satellite image of a harbor",
        "an aerial image of boats and ships docked at a harbor",
        "ships and boats moored at a port",
        "an overhead view of a harbor with docks and piers",
    ],
    "intersection": [
        "a satellite image of a road intersection",
        "an aerial image of a crossroads where two roads meet",
        "an overhead view of an intersection of roads",
    ],
    "mediumresidential": [
        "a satellite image of a medium density residential area",
        "an aerial image of a suburban neighborhood with moderately spaced houses",
        "some buildings and green trees are in a medium residential area",
        "houses with medium-sized yards arranged in a suburban pattern",
        "a suburban neighborhood with houses and trees, neither tightly packed nor widely spaced",
    ],
    "mobilehomepark": [
        "a satellite image of a mobile home park",
        "an aerial image of rows of mobile homes in a trailer park",
        "neatly arranged rows of rectangular mobile homes",
        "a residential area of small rectangular trailer homes lined up in rows",
        "an overhead view of a trailer park with uniform rectangular homes",
    ],
    "overpass": [
        "a satellite image of an overpass",
        "an aerial image of a highway overpass crossing another road",
        "a road passing over another road on a bridge",
        "an overhead view of a multi-level road interchange",
    ],
    "parkinglot": [
        "a satellite image of a parking lot",
        "an aerial image of a parking lot filled with cars",
        "an overhead view of rows of parked cars in a parking lot",
    ],
    "river": [
        "a satellite image of a river",
        "an aerial image of a winding river through the landscape",
        "a curving river flowing through land",
    ],
    "runway": [
        "a satellite image of an airport runway",
        "an aerial image of a long straight runway at an airport",
        "a long straight paved strip used by aircraft for takeoff and landing",
        "an overhead view of an empty airport runway",
    ],
    "sparseresidential": [
        "a satellite image of a sparse residential area",
        "an aerial image of widely spaced houses with large yards and lots of greenery",
        "a residential area with houses sparsely distributed across the land",
        "a rural neighborhood where houses are far apart, surrounded by trees and open space",
        "an overhead view of a low-density residential area with large lots",
    ],
    "storagetanks": [
        "a satellite image of storage tanks",
        "an aerial image of large cylindrical storage tanks",
        "an overhead view of round industrial storage tanks",
        "a cluster of circular oil or fuel storage tanks",
    ],
    "tenniscourt": [
        "a satellite image of a tennis court",
        "an aerial image of a rectangular tennis court",
        "an overhead view of a tennis court with court lines",
    ],
}


# ── Load model ────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model.load_state_dict(torch.load("finetuned_clip.pt", map_location=device))
model = model.to(device).eval()

# ── Build text embeddings for each class ──────────────────────────────────────
with torch.no_grad():
    text_embeddings = []
    for cls in CLASS_NAMES:
        # Combine generic templates (using readable display name) with
        # class-specific descriptive prompts.
        display = cls.replace("_", " ")
        generic = [t.format(display) for t in GENERIC_TEMPLATES]
        specific = CLASS_PROMPTS.get(cls, [])
        prompts = generic + specific

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
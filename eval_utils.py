import torch
import torch.nn.functional as F
from PIL import Image

TEMPLATES = [
    "a satellite image of {}",
    "an aerial image of {}",
    "a remote sensing image of {}",
]

def load_class_map(classes_dir):
    class_names = sorted(p.stem.lower() for p in classes_dir.glob("*.txt"))
    filename_to_class = {}
    for txt_file in classes_dir.glob("*.txt"):
        cls = txt_file.stem.lower()
        for line in txt_file.read_text().splitlines():
            fname = line.strip()
            if fname:
                filename_to_class[fname] = cls
    return class_names, filename_to_class

def zero_shot_accuracy(model, tokenizer, val_filenames, image_dir, preprocess,
                       class_names, filename_to_class, device):
    model.eval()
    with torch.no_grad():
        text_embeddings = []
        for cls in class_names:
            prompts = [t.format(cls.replace("_", " ")) for t in TEMPLATES]
            tokens = tokenizer(prompts).to(device)
            embs = F.normalize(model.encode_text(tokens), dim=-1).mean(dim=0)
            text_embeddings.append(F.normalize(embs, dim=0))
        text_embeddings = torch.stack(text_embeddings)

        correct = 0
        for fname in val_filenames:
            true_class = filename_to_class.get(fname, "unknown")
            image = preprocess(Image.open(image_dir / fname).convert("RGB")).unsqueeze(0).to(device)
            img_emb = F.normalize(model.encode_image(image), dim=-1)
            pred_class = class_names[(img_emb @ text_embeddings.T).squeeze(0).argmax().item()]
            correct += pred_class == true_class

    return correct / len(val_filenames) if val_filenames else 0.0

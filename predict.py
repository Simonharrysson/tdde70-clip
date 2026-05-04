"""Generate predictions.txt for the CodaBench leaderboard."""
import os
import zipfile
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import open_clip

from config import Config
from dataset import ImageDataset, get_clip_transform
from evaluate import build_text_embeddings, encode_images


def predict(cfg: Config, checkpoint_path: str = None, output_path: str = "predictions.txt"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, _, _ = open_clip.create_model_and_transforms(cfg.model_name, pretrained=cfg.pretrained)
    tokenizer = open_clip.get_tokenizer(cfg.model_name)

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("No checkpoint — using pretrained weights only")

    model = model.to(device)
    model.eval()

    # Unzip leaderboard images if needed
    lb_dir = os.path.join(cfg.data_root, "leaderboard_images")
    if not os.path.exists(lb_dir):
        print("Unzipping leaderboard data...")
        with zipfile.ZipFile(cfg.leaderboard_zip_path) as z:
            z.extractall(lb_dir)

    # Find the actual image directory (may be nested inside the zip)
    image_dir = lb_dir
    for entry in os.listdir(lb_dir):
        candidate = os.path.join(lb_dir, entry)
        if os.path.isdir(candidate):
            image_dir = candidate
            break

    filenames = sorted(f for f in os.listdir(image_dir) if f.lower().endswith('.jpg'))
    print(f"Found {len(filenames)} leaderboard images")

    transform = get_clip_transform(cfg.image_size)
    dataset = ImageDataset(image_dir, filenames, transform)
    loader = DataLoader(dataset, batch_size=128, num_workers=2)

    # Build class embeddings for the 21 leaderboard classes
    class_names = list(cfg.leaderboard_classes)
    text_embs = build_text_embeddings(
        model, tokenizer, class_names, cfg.prompt_templates, device
    )

    img_embs, fnames = encode_images(model, loader, device)
    logits = img_embs @ text_embs.T.cpu()
    preds = logits.argmax(dim=-1)

    lines = [f"{fname} {class_names[preds[i].item()]}" for i, fname in enumerate(fnames)]
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Saved {len(lines)} predictions → {output_path}")
    print("Next: zip it with  zip predictions.zip predictions.txt  and upload to CodaBench")


if __name__ == '__main__':
    cfg = Config()
    predict(cfg, checkpoint_path='checkpoints/best_model.pt')

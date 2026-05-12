import json
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class RSICDDataset(Dataset):
    def __init__(self, samples, image_dir, tokenizer, preprocess):
        self.samples = samples
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.preprocess = preprocess

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f, cap = self.samples[idx]
        img = self.preprocess(Image.open(self.image_dir / f).convert("RGB"))
        return img, self.tokenizer([cap])[0]

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

def make_loader(samples, image_dir, tokenizer, preprocess, batch_size):
    return DataLoader(RSICDDataset(samples, image_dir, tokenizer, preprocess),
                      batch_size=batch_size, shuffle=True)

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


def make_loader(samples, image_dir, tokenizer, preprocess, batch_size):
    return DataLoader(RSICDDataset(samples, image_dir, tokenizer, preprocess),
                      batch_size=batch_size, shuffle=True)

import json
import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


# ── Transforms ──────────────────────────────────────────────────────────────

def get_clip_transform(image_size=224):
    """Standard CLIP preprocessing — no augmentation."""
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]),
    ])


def get_augmentation_transform(image_size=224):
    """Strong augmentation for self-supervised learning on satellite imagery.
    Random rotations and vertical flips are added because overhead imagery
    has no canonical orientation.
    """
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.2, 1.0),
                            interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=90),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]),
    ])


# ── Datasets ────────────────────────────────────────────────────────────────

class LabeledDataset(Dataset):
    """Split B (json split='val'): 1094 images with 5 captions each."""

    def __init__(self, json_path, image_dir, transform=None):
        with open(json_path) as f:
            data = json.load(f)

        self.image_dir = Path(image_dir)
        self.transform = transform
        self.samples = [
            (img['filename'], [s['raw'] for s in img['sentences']])
            for img in data['images']
            if img['split'] == 'val'
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, captions = self.samples[idx]
        image = Image.open(self.image_dir / filename).convert('RGB')
        if self.transform:
            image = self.transform(image)
        caption = random.choice(captions)
        return image, caption, captions       # (image, sampled caption, all 5 captions)


class UnlabeledDataset(Dataset):
    """Split A (json split='train'): 8734 images, captions intentionally ignored.
    Returns two independently augmented views of each image for SimCLR.
    """

    def __init__(self, json_path, image_dir, augment=None):
        with open(json_path) as f:
            data = json.load(f)

        self.image_dir = Path(image_dir)
        self.augment = augment
        self.filenames = [
            img['filename'] for img in data['images']
            if img['split'] == 'train'
        ]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.image_dir / self.filenames[idx]).convert('RGB')
        v1 = self.augment(image) if self.augment else image
        v2 = self.augment(image) if self.augment else image
        return v1, v2


class ImageDataset(Dataset):
    """Generic image-only dataset used for evaluation and leaderboard prediction."""

    def __init__(self, image_dir, filenames, transform=None):
        self.image_dir = Path(image_dir)
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        image = Image.open(self.image_dir / fname).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, fname


# ── Helpers ─────────────────────────────────────────────────────────────────

def load_class_mapping(classes_dir):
    """Returns dict: filename -> class_name (lowercase) from txtclasses_rsicd."""
    mapping = {}
    for txt_file in Path(classes_dir).glob('*.txt'):
        class_name = txt_file.stem.lower()
        for line in txt_file.read_text().splitlines():
            fname = line.strip()
            if fname:
                mapping[fname] = class_name
    return mapping


def get_rsicd_class_names(classes_dir):
    """Sorted list of RSICD class names (lowercase)."""
    return sorted(p.stem.lower() for p in Path(classes_dir).glob('*.txt'))


def get_test_filenames(json_path):
    """Returns filenames for Split C (json split='test')."""
    with open(json_path) as f:
        data = json.load(f)
    return [img['filename'] for img in data['images'] if img['split'] == 'test']


def get_test_data(json_path):
    """Returns list of (filename, captions) for Split C."""
    with open(json_path) as f:
        data = json.load(f)
    return [
        (img['filename'], [s['raw'] for s in img['sentences']])
        for img in data['images']
        if img['split'] == 'test'
    ]

import os
from dataclasses import dataclass


@dataclass
class Config:
    # ── Data paths ───────────────────────────────────────────────────────
    data_root: str = "."
    image_dir: str = "RSICD_images"
    json_file: str = "dataset_rsicd.json"
    classes_dir: str = "txtclasses_rsicd"
    leaderboard_zip: str = "Leaderboard_data.zip"

    # ── Model ────────────────────────────────────────────────────────────
    model_name: str = "ViT-B-32"
    pretrained: str = "openai"
    image_size: int = 224

    # ── Training ─────────────────────────────────────────────────────────
    epochs: int = 15
    batch_size_labeled: int = 64
    lr: float = 5e-6
    weight_decay: float = 0.1
    warmup_epochs: float = 1.0
    val_fraction: float = 0.15      # fraction of Split B held out for validation

    # ── Semi-supervised ──────────────────────────────────────────────────
    semi_supervised: bool = True
    batch_size_unlabeled: int = 128
    lambda_unsup: float = 0.5       # weight for SimCLR loss term
    simclr_temperature: float = 0.1

    # ── Evaluation ───────────────────────────────────────────────────────
    eval_every: int = 1
    retrieval_ks: tuple = (1, 5, 10)

    # ── Checkpoints / output ─────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"

    # ── Prompt templates for zero-shot classification ────────────────────
    prompt_templates: tuple = (
        "a satellite image of {}",
        "an aerial image of {}",
        "a remote sensing image of {}",
        "an overhead image of {}",
    )

    # ── Leaderboard class names (21 UCMerced classes) ────────────────────
    leaderboard_classes: tuple = (
        "agricultural", "airplane", "baseballdiamond", "beach", "buildings",
        "chaparral", "denseresidential", "forest", "freeway", "golfcourse",
        "harbor", "intersection", "mediumresidential", "mobilehomepark", "overpass",
        "parkinglot", "river", "runway", "sparseresidential", "storagetanks", "tenniscourt",
    )

    @property
    def image_path(self):
        return os.path.join(self.data_root, self.image_dir)

    @property
    def json_path(self):
        return os.path.join(self.data_root, self.json_file)

    @property
    def classes_path(self):
        return os.path.join(self.data_root, self.classes_dir)

    @property
    def leaderboard_zip_path(self):
        return os.path.join(self.data_root, self.leaderboard_zip)

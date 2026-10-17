import hashlib
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


DATASET_CLASSES = {
    "mvtec": [
        "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
        "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
        "transistor", "wood", "zipper",
    ],
    "visa": [
        "candle", "capsules", "cashew", "chewinggum", "fryum",
        "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4",
        "pipe_fryum",
    ],
    "btad": ["01", "02", "03"],
    "mpdd": [
        "bracket_black", "bracket_brown", "bracket_white", "connector",
        "metal_plate", "tubes",
    ],
    "beltad": ["belt"],
}

DATASET_DIR_NAMES = {
    "mvtec": "mvtec_anomaly_detection",
    "visa": "VisA",
    "btad": "BTAD",
    "mpdd": "MPDD",
    "beltad": "BeltAD",
}

NORMAL_DIRS = {"good", "ok"}

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def get_classes(dataset):
    try:
        return DATASET_CLASSES[dataset.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset: {dataset}") from exc


def dataset_dir_name(dataset):
    return DATASET_DIR_NAMES[dataset.lower()]


def normal_train_path(root, class_name):
    for name in ("good", "ok"):
        path = Path(root) / class_name / "train" / name
        if path.is_dir():
            return path
    return None


def list_images(path):
    return sorted(
        (item for item in Path(path).iterdir()
         if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda item: item.name.lower(),
    )


def validate_dataset(root, dataset):
    root = Path(root).expanduser().resolve()
    missing = []
    for class_name in get_classes(dataset):
        if normal_train_path(root, class_name) is None:
            missing.append(str(Path(class_name) / "train" / "good-or-ok"))
        if not (root / class_name / "test").is_dir():
            missing.append(str(Path(class_name) / "test"))
    if missing:
        raise FileNotFoundError(f"Incomplete {dataset} dataset at {root}: {missing[:5]}")
    return root


def image_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def mask_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])


class IndustrialTestDataset(Dataset):
    def __init__(self, root, dataset, class_name, image_size=368):
        self.root = validate_dataset(root, dataset)
        self.dataset = dataset.lower()
        self.class_name = class_name
        self.image_transform = image_transform(image_size)
        self.mask_transform = mask_transform(image_size)
        self.image_size = image_size
        self.samples = self._collect_samples()

    def _mask_path(self, image_path, defect_type):
        mask_root = self.root / self.class_name / "ground_truth" / defect_type
        stem = image_path.stem
        candidates = [
            f"{stem}_mask.png", f"{stem}.png",
            f"{stem}_mask.bmp", f"{stem}.bmp",
            f"{stem}_mask.jpg", f"{stem}.jpg",
        ]
        for name in candidates:
            path = mask_root / name
            if path.is_file():
                return path
        raise FileNotFoundError(f"No ground-truth mask for {image_path}")

    def _collect_samples(self):
        samples = []
        test_root = self.root / self.class_name / "test"
        for defect_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
            label = int(defect_dir.name.lower() not in NORMAL_DIRS)
            for image_path in list_images(defect_dir):
                mask_path = self._mask_path(image_path, defect_dir.name) if label else None
                samples.append((image_path, mask_path, label))
        if not samples:
            raise ValueError(f"No test images found in {test_root}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = self.image_transform(image.convert("RGB"))
        if mask_path is None:
            mask = torch.zeros(1, self.image_size, self.image_size)
        else:
            with Image.open(mask_path) as image_mask:
                mask = self.mask_transform(image_mask.convert("L"))
        return image, mask, label, str(image_path)


def make_support_manifest(root, dataset, class_name, shot, rounds=10, seed=668):
    root = validate_dataset(root, dataset)
    train_dir = normal_train_path(root, class_name)
    candidates = list_images(train_dir)
    if len(candidates) < shot:
        raise ValueError(f"{class_name} only has {len(candidates)} support candidates")
    token = f"fastrecon:{dataset}:{class_name}:{shot}:{seed}".encode("utf-8")
    class_seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
    rng = random.Random(class_seed)
    return {
        "dataset": dataset,
        "class": class_name,
        "shot": shot,
        "rounds": rounds,
        "seed": seed,
        "samples": [
            [str(path.relative_to(root)) for path in rng.sample(candidates, shot)]
            for _ in range(rounds)
        ],
    }


def save_support_manifest(path, root, dataset, class_name, shot, rounds=10, seed=668):
    payload = make_support_manifest(root, dataset, class_name, shot, rounds, seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def load_support_images(root, manifest, image_size=368):
    root = Path(root).expanduser().resolve()
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    transform = image_transform(image_size)
    result = []
    for paths in payload["samples"]:
        tensors = []
        for relative in paths:
            with Image.open(root / relative) as image:
                tensors.append(transform(image.convert("RGB")))
        result.append(torch.stack(tensors))
    return result


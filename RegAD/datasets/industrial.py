import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor


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
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {sorted(DATASET_CLASSES)}") from exc


def dataset_dir_name(dataset):
    return DATASET_DIR_NAMES[dataset.lower()]


def list_images(path):
    path = Path(path)
    return sorted(
        (item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda item: item.name.lower(),
    )


def normal_dirs(class_name_root):
    for name in ("good", "ok"):
        path = class_name_root / "train" / name
        if path.is_dir():
            return path
    return None


def validate_dataset(root, dataset):
    root = Path(root).expanduser().resolve()
    missing = []
    for class_name in get_classes(dataset):
        if normal_dirs(root / class_name) is None:
            missing.append(str(Path(class_name) / "train" / "good-or-ok"))
        if not (root / class_name / "test").is_dir():
            missing.append(str(Path(class_name) / "test"))
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"{root} is not a complete {dataset} root; missing: {preview}")
    return root


def image_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.LANCZOS),
        transforms.ToTensor(),
    ])


def mask_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])


class IndustrialTrainDataset(Dataset):
    """Official RegAD leave-one-class-out registration episodes."""

    def __init__(self, root, dataset, target_class, shot=2, batch_size=32, image_size=224, seed=668,
                 cache_images=True):
        self.root = validate_dataset(root, dataset)
        self.dataset = dataset.lower()
        self.target_class = target_class
        self.shot = shot
        self.batch_size = batch_size
        self.transform = image_transform(image_size)
        self.image_size = image_size
        self.rng = random.Random(seed)
        self.seed = seed
        if target_class not in get_classes(self.dataset):
            raise ValueError(f"Unknown {self.dataset} class: {target_class}")
        self.class_images = {
            class_name: list_images(normal_dirs(self.root / class_name))
            for class_name in get_classes(self.dataset)
            if class_name != target_class
        }
        if any(len(paths) < 2 for paths in self.class_images.values()):
            raise ValueError("Each source class needs at least two normal training images")
        self.image_cache = self._build_cache() if cache_images else None
        self.shuffle_dataset()

    def _build_cache(self):
        paths = [path for class_paths in self.class_images.values() for path in class_paths]
        with ThreadPoolExecutor(max_workers=16) as executor:
            tensors = executor.map(self._load_uint8, paths)
            return dict(zip(paths, tensors))

    def _load_uint8(self, path):
        with Image.open(path) as image:
            image = image.convert("RGB").resize(
                (self.image_size, self.image_size), resample=Image.Resampling.LANCZOS
            )
            return pil_to_tensor(image)

    def shuffle_dataset(self, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
        self.episodes = []
        for paths in self.class_images.values():
            paths = list(paths)
            self.rng.shuffle(paths)
            for start in range(0, len(paths), self.batch_size):
                queries = paths[start:start + self.batch_size]
                supports = []
                for query in queries:
                    pool = [path for path in paths if path != query]
                    # Match the author's randint loop: independently sample each
                    # support after excluding the query, so duplicates are allowed.
                    supports.append(self.rng.choices(pool, k=self.shot))
                self.episodes.append((queries, supports))

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        query_paths, support_paths = self.episodes[index]
        queries = torch.stack([self._load(path) for path in query_paths])
        supports = torch.stack([
            torch.stack([self._load(path) for path in paths]) for paths in support_paths
        ])
        return queries, supports

    def _load(self, path):
        if self.image_cache is not None:
            return self.image_cache[path].float().div_(255)
        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))


class IndustrialTestDataset(Dataset):
    def __init__(self, root, dataset, target_class, image_size=224):
        self.root = validate_dataset(root, dataset)
        self.dataset = dataset.lower()
        self.target_class = target_class
        self.image_size = image_size
        self.transform = image_transform(image_size)
        self.mask_transform = mask_transform(image_size)
        self.samples = self._samples()

    def _samples(self):
        samples = []
        test_root = self.root / self.target_class / "test"
        for anomaly_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
            is_anomaly = anomaly_dir.name.lower() not in NORMAL_DIRS
            for image_path in list_images(anomaly_dir):
                mask_path = self._mask_path(image_path, anomaly_dir.name) if is_anomaly else None
                samples.append((image_path, mask_path, int(is_anomaly)))
        if not samples:
            raise ValueError(f"No test images found in {test_root}")
        return samples

    def _mask_path(self, image_path, anomaly_type):
        mask_dir = self.root / self.target_class / "ground_truth" / anomaly_type
        stem = image_path.stem
        names = [
            f"{stem}_mask.png", f"{stem}.png",
            f"{stem}_mask.bmp", f"{stem}.bmp",
            f"{stem}_mask.jpg", f"{stem}.jpg",
        ]
        for name in names:
            candidate = mask_dir / name
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"No mask found for {image_path} in {mask_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image_tensor = self.transform(image.convert("RGB"))
        if mask_path is None:
            mask_tensor = torch.zeros(1, self.image_size, self.image_size)
        else:
            with Image.open(mask_path) as mask:
                mask_tensor = self.mask_transform(mask.convert("L"))
        return image_tensor, mask_tensor, label, str(image_path)


def support_manifest(root, dataset, target_class, shot, rounds=10, seed=668):
    root = validate_dataset(root, dataset)
    train_dir = normal_dirs(root / target_class)
    normal_paths = list_images(train_dir)
    if len(normal_paths) < shot:
        raise ValueError(f"{target_class} has {len(normal_paths)} normal images, fewer than shot={shot}")
    key = f"{dataset}:{target_class}:{shot}:{seed}".encode("utf-8")
    class_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    rng = random.Random(class_seed)
    return [
        [str(path.relative_to(root)) for path in rng.sample(normal_paths, shot)]
        for _ in range(rounds)
    ]


def save_support_manifest(path, root, dataset, target_class, shot, rounds=10, seed=668):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "class": target_class,
        "shot": shot,
        "rounds": rounds,
        "seed": seed,
        "samples": support_manifest(root, dataset, target_class, shot, rounds, seed),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def load_support_rounds(root, manifest, image_size=224):
    manifest = Path(manifest)
    if manifest.suffix == ".pt":
        rounds = torch.load(manifest, map_location="cpu", weights_only=False)
        if not isinstance(rounds, (list, tuple)) or not all(torch.is_tensor(item) for item in rounds):
            raise ValueError(f"Unsupported official support tensor format: {manifest}")
        return list(rounds)

    root = Path(root).expanduser().resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    transform = image_transform(image_size)
    rounds = []
    for relative_paths in payload["samples"]:
        images = []
        for relative_path in relative_paths:
            with Image.open(root / relative_path) as image:
                images.append(transform(image.convert("RGB")))
        rounds.append(torch.stack(images))
    return rounds

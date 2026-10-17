"""RegAD medical adaptation.

Self-source protocol: the registration model is trained on the *same* dataset's
full normal training split (there is only one class per medical dataset), and the
canonical few-shot support (from _shared/medical_data.py) is used only at
inference time.

The episode sampler mirrors datasets/industrial.py (RegAD's official
leave-one-class-out episodes) but draws queries and supports from a single
normal pool.
"""

import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor

_SHARED = Path(__file__).resolve().parents[2] / "_shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

from medical_data import (  # noqa: E402
    canonical_support_paths,
    collect_test_samples,
    normal_train_paths,
)


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


class MedicalTrainDataset(Dataset):
    """Self-source registration episodes over the full normal training split."""

    def __init__(self, dataset, shot=2, batch_size=32, image_size=224, seed=668,
                 cache_images=True):
        self.dataset = dataset
        self.shot = shot
        self.batch_size = batch_size
        self.image_size = image_size
        self.transform = image_transform(image_size)
        self.rng = random.Random(seed)
        self.seed = seed
        self.paths = normal_train_paths(dataset)
        if len(self.paths) < 2:
            raise ValueError(f"{dataset} needs at least two normal training images")
        self.image_cache = self._build_cache() if cache_images else None
        self.shuffle_dataset(seed)

    def _build_cache(self):
        with ThreadPoolExecutor(max_workers=16) as executor:
            tensors = executor.map(self._load_uint8, self.paths)
            return dict(zip(self.paths, tensors))

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
        paths = list(self.paths)
        self.rng.shuffle(paths)
        for start in range(0, len(paths), self.batch_size):
            queries = paths[start:start + self.batch_size]
            supports = []
            for query in queries:
                pool = [path for path in paths if path != query]
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


class MedicalTestDataset(Dataset):
    """Yields (image, mask[1,S,S], label, path) for RegAD's evaluator."""

    def __init__(self, dataset, image_size=224):
        self.dataset = dataset
        self.image_size = image_size
        self.transform = image_transform(image_size)
        self.mask_transform = mask_transform(image_size)
        self.samples = collect_test_samples(dataset)

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
                mask_tensor = (self.mask_transform(mask.convert("L")) > 0.5).float()
        return image_tensor, mask_tensor, label, str(image_path)


def load_canonical_support(dataset, shot, image_size=224):
    """Single inference round built from the canonical support selection."""
    transform = image_transform(image_size)
    images = []
    for path in canonical_support_paths(dataset, shot):
        with Image.open(path) as image:
            images.append(transform(image.convert("RGB")))
    return [torch.stack(images)]

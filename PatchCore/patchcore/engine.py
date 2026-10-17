"""PatchCore inference engine: memory bank + kNN anomaly scoring."""

import numpy as np
import torch

from .coreset import coreset_indices
from .models import FeatureExtractor


class PatchCoreEngine:
    def __init__(self, device="cuda", backbone="layer23", coreset_ratio=0.1,
                 neighbor_count=3, image_size=224):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.extractor = FeatureExtractor(layer=backbone).to(self.device)
        self.coreset_ratio = coreset_ratio
        self.neighbor_count = neighbor_count
        self.memory = None

    @torch.no_grad()
    def fit(self, support_tensors):
        """support_tensors: list of [T,3,H,W] tensors (T = shots this round)."""
        patches = []
        for group in support_tensors:
            features = self.extractor(group.to(self.device))  # [T, G, C]
            patches.append(features.cpu().numpy().reshape(-1, features.shape[-1]))
        bank = np.concatenate(patches, axis=0) if patches else None
        if bank is None or len(bank) == 0:
            raise ValueError("Empty support set")
        indices = coreset_indices(bank, self.coreset_ratio, seed=668)
        self.memory = bank[indices]
        del bank
        torch.cuda.empty_cache()

    def _map_from_scores(self, indices_and_scores, height, width):
        order = np.argsort(indices_and_scores[:, 0], kind="stable")
        scores = indices_and_scores[order, 1]
        return scores.reshape(height, width)

    @torch.no_grad()
    def score(self, images, image_size=None):
        """images: [B,3,H,W]; returns (image_scores, list of 2D anomaly maps [B,H,W])."""
        image_size = image_size or self.image_size
        features = self.extractor(images.to(self.device))  # [B, G, C]
        memory = torch.from_numpy(self.memory).to(self.device).float()  # [M, C]
        batch = features.cpu().numpy()
        memory_num = memory.cpu().numpy()
        image_scores = []
        maps = []
        for sample in batch:
            distances = self._distances(sample, memory_num)  # [G]
            image_scores.append(float(distances.max()))
            height = width = int(round(image_size / 16))
            score_map = distances.reshape(height, width)
            score_map = self._upsample(score_map, image_size)
            maps.append(score_map)
        return np.asarray(image_scores), maps

    def _distances(self, sample, memory):
        """Mean of the k smallest squared euclidean distances to memory patches."""
        diff = sample[:, None, :] - memory[None, :, :]
        squared = np.einsum("ijk,ijk->ij", diff, diff)
        k = min(self.neighbor_count, squared.shape[1])
        top_k = np.partition(squared, k - 1, axis=1)[:, :k]
        return np.mean(top_k, axis=1)

    def _upsample(self, score_map, target):
        import cv2
        if score_map.shape[0] != target:
            map_float = score_map.astype(np.float32)
            resized = cv2.resize(map_float, (target, target),
                                 interpolation=cv2.INTER_LINEAR)
            return resized
        return score_map
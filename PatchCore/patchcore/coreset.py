"""Self-contained k-center-greedy coreset subsetting for PatchCore."""

import numpy as np


def greedy_coresetindices(features, ratio, seed=None, max_candidates=30000):
    """Select indices covering the point set via farthest-first sampling.

    features: (N, D) float array. Returns integer array of length max(1, round(N*ratio)).
    Large inputs (> max_candidates) are pre-subsampled, as in the reference PatchCore.
    """
    if features.shape[0] > max_candidates and ratio < 1.0:
        if seed is not None:
            np.random.seed(seed)
        chosen_rows = np.random.choice(features.shape[0], size=max_candidates,
                                       replace=False)
        subset = features[chosen_rows]
    else:
        chosen_rows = np.arange(features.shape[0])
        subset = features
    num = max(1, int(round(subset.shape[0] * ratio)))
    if num >= subset.shape[0]:
        return chosen_rows
    if seed is not None:
        np.random.seed(seed)
    selected = [int(np.random.randint(subset.shape[0]))]
    min_dist = np.full(subset.shape[0], np.inf)
    while len(selected) < num:
        center = subset[selected[-1]]
        dist = np.linalg.norm(subset - center, axis=1)
        min_dist = np.minimum(min_dist, dist)
        selected.append(int(np.argmax(min_dist)))
    return chosen_rows[np.asarray(selected, dtype=int)]


def coreset_indices(features, ratio, seed=None):
    features = np.asarray(features, dtype=np.float32)
    indices = greedy_coresetindices(features, ratio, seed=seed)
    return indices
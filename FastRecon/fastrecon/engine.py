import csv
import json
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.random_projection import SparseRandomProjection
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

_SHARED = Path(__file__).resolve().parent.parent.parent / "_shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

from eval_common import compute_metrics as compute_full_metrics  # noqa: E402
from eval_common import save_visual  # noqa: E402

from fastrecon.data import IndustrialTestDataset, load_support_images


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


class FeatureExtractor(nn.Module):
    """The public code's ImageNet WRN50 layer2/layer3 feature extractor."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = wide_resnet50_2(weights=weights)
        self.features = []
        self.backbone.layer2[-1].register_forward_hook(self._capture)
        self.backbone.layer3[-1].register_forward_hook(self._capture)
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    def _capture(self, _module, _inputs, output):
        self.features.append(output)

    @torch.inference_mode()
    def forward(self, images):
        self.features = []
        self.backbone(images)
        layer2, layer3 = (F.avg_pool2d(item, 3, stride=1, padding=1) for item in self.features)
        layer3 = F.interpolate(layer3, size=layer2.shape[-2:], mode="nearest")
        return torch.cat((layer2, layer3), dim=1)


def flatten_features(features):
    return features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])


@torch.inference_mode()
def extract_support(extractor, images, device):
    features = extractor(images.to(device, non_blocking=True))
    batch, channels, height, width = features.shape
    locations = features.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
    return locations.reshape(-1, channels), locations.mean(dim=0)


def greedy_coreset(features, ratio, seed, projection_eps=0.9):
    count = max(1, int(features.shape[0] * ratio))
    cpu_features = features.detach().float().cpu().numpy()
    projector = SparseRandomProjection(
        n_components="auto", eps=projection_eps, random_state=seed,
    )
    projected = torch.from_numpy(projector.fit_transform(cpu_features)).to(
        features.device, dtype=torch.float32
    )
    selected = []
    minimum = None
    index = 0
    for _ in range(count):
        selected.append(index)
        distance = (projected - projected[index]).square().sum(dim=1)
        minimum = distance if minimum is None else torch.minimum(minimum, distance)
        minimum[selected] = -1
        index = int(torch.argmax(minimum).item())
    indices = torch.tensor(selected, device=features.device)
    return features.index_select(0, indices), selected


def reconstruction_basis(coreset):
    # S^T (S S^T)^-1 S is the projector onto the row space of S. QR is the
    # stable equivalent and avoids the singular inverse in the public code.
    basis, _ = torch.linalg.qr(coreset.T.float(), mode="reduced")
    return basis


@torch.inference_mode()
def reconstruct_scores(query, support_mean, basis, regularization):
    target = (query.float() + regularization * support_mean.float()) / (1 + regularization)
    reconstruction = (target @ basis) @ basis.T
    return torch.linalg.vector_norm(query.float() - reconstruction, dim=-1)


@torch.inference_mode()
def extract_queries(extractor, loader, device):
    features, masks, labels, paths = [], [], [], []
    spatial_shape = None
    for images, batch_masks, batch_labels, batch_paths in loader:
        output = extractor(images.to(device, non_blocking=True))
        spatial_shape = output.shape[-2:]
        output = output.permute(0, 2, 3, 1).flatten(1, 2)
        features.append(output.cpu().to(torch.float16))
        masks.append(batch_masks[:, 0].numpy())
        labels.extend(batch_labels.numpy().tolist())
        paths.extend(batch_paths)
    return (
        torch.cat(features), np.concatenate(masks), np.asarray(labels), paths, spatial_shape,
    )


def compute_aurocs(score_maps, image_scores, masks, labels):
    score_maps = np.asarray(score_maps, dtype=np.float64)
    masks_bin = (np.asarray(masks) > 0.5).astype(np.uint8)
    return compute_full_metrics(
        labels,
        np.asarray(image_scores, dtype=np.float64),
        [mask for mask in masks_bin],
        [map_ for map_ in score_maps],
    )


METRIC_KEYS = ("i_auroc", "i_aupr", "p_auroc", "p_aupr", "p_aupro")


def evaluate_round(query_features, masks, labels, support, extractor, device, args, round_seed,
                   spatial_shape):
    support_pool, support_mean = extract_support(extractor, support, device)
    coreset, indices = greedy_coreset(
        support_pool, args.coreset_ratio, round_seed, args.projection_eps
    )
    basis = reconstruction_basis(coreset)
    height, width = spatial_shape
    maps = []
    image_scores = []
    for start in range(0, len(query_features), args.reconstruction_batch_size):
        query = query_features[start:start + args.reconstruction_batch_size].to(
            device, dtype=torch.float32
        )
        scores = reconstruct_scores(query, support_mean, basis, args.regularization)
        image_scores.extend(scores.amax(dim=1).cpu().numpy().tolist())
        scores = scores.reshape(-1, 1, height, width)
        scores = F.interpolate(
            scores, size=(args.image_size, args.image_size), mode="bilinear", align_corners=False
        )[:, 0].cpu().numpy()
        maps.extend(gaussian_filter(item, sigma=args.gaussian_sigma) for item in scores)
    metrics = compute_aurocs(maps, image_scores, masks, labels)
    metrics.update({"coreset_size": len(indices), "feature_grid": [height, width]})
    return metrics, maps


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def evaluate_category(args, data_root, dataset, class_name, shot, manifest, output, device):
    set_seed(args.seed)
    started = time.time()
    extractor = FeatureExtractor(pretrained=True).to(device).eval()
    test_data = IndustrialTestDataset(data_root, dataset, class_name, args.image_size)
    loader = DataLoader(
        test_data, batch_size=args.feature_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    query_features, masks, labels, paths, spatial_shape = extract_queries(extractor, loader, device)
    support_rounds = load_support_images(data_root, manifest, args.image_size)
    rounds = []
    round_maps = None
    for index, support in enumerate(support_rounds):
        metrics, maps = evaluate_round(
            query_features, masks, labels, support, extractor, device, args,
            args.seed + index, spatial_shape,
        )
        metrics["round"] = index
        rounds.append(metrics)
        if round_maps is None:
            round_maps = maps
        print(
            f"[{dataset}/{class_name}/k={shot}] round {index + 1}/{len(support_rounds)} "
            f"I-AUROC={metrics['i_auroc']:.4f} P-AUROC={metrics['p_auroc']:.4f}",
            flush=True,
        )
    payload = {
        "dataset": dataset,
        "class": class_name,
        "shot": shot,
    }
    for key in METRIC_KEYS:
        values = np.asarray([item[key] for item in rounds], dtype=float)
        payload[key] = float(np.mean(values))
        payload[f"{key}_std"] = float(np.std(values))
    payload.update({
        "rounds": rounds,
        "protocol": {
            "backbone": "wide_resnet50_2_imagenet1k_v1",
            "layers": ["layer2", "layer3"],
            "image_size": args.image_size,
            "coreset_ratio": args.coreset_ratio,
            "regularization": args.regularization,
            "gaussian_sigma": args.gaussian_sigma,
            "support_manifest": str(Path(manifest).resolve()),
        },
        "elapsed_seconds": time.time() - started,
    })
    if not getattr(args, "no_visual", False) and round_maps is not None:
        from PIL import Image
        visual_dir = Path(output).parent / "visual"
        masks_bin = [(np.asarray(mask) > 0.5).astype(np.uint8) for mask in masks]
        for score_map, mask, path in zip(round_maps, masks_bin, paths):
            unique = Path(path).relative_to(test_data.root / class_name / "test")
            original = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
            save_visual(visual_dir / str(unique).replace("/", "__"), original, mask, score_map)
        payload["visual_dir"] = str(visual_dir)
    save_json(output, payload)
    return payload


def summarize_results(result_files, output_csv, output_json):
    rows = [json.loads(Path(path).read_text(encoding="utf-8")) for path in result_files]
    grouped = OrderedDict()
    for row in sorted(rows, key=lambda item: (item["dataset"], item["shot"], item["class"])):
        grouped.setdefault((row["dataset"], row["shot"]), []).append(row)
    summary = []
    for (dataset, shot), items in grouped.items():
        summary.extend(items)
        mean = {"dataset": dataset, "class": "mean", "shot": shot}
        for key in METRIC_KEYS:
            values = np.asarray([item[key] for item in items if key in item], dtype=float)
            values = values[~np.isnan(values)]
            mean[key] = float(np.mean(values)) if len(values) else float("nan")
        summary.append(mean)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["dataset", "class", "shot", *METRIC_KEYS]
        )
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in writer.fieldnames} for row in summary)
    save_json(output_json, summary)
    return summary


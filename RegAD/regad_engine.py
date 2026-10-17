import csv
import json
import math
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

_SHARED = Path(__file__).resolve().parent.parent / "_shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

from eval_common import compute_metrics as compute_full_metrics  # noqa: E402
from eval_common import save_visual  # noqa: E402

from datasets.industrial import IndustrialTestDataset, IndustrialTrainDataset, load_support_rounds
from losses.norm_loss import CosLoss
from models.siamese import Encoder, Predictor
from models.stn import stn_net
from utils.funcs import (
    embedding_concat,
    grey_img,
    hflip_img,
    rot90_img,
    rot_img,
    translation_img,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def make_models(args, device, pretrained=True):
    models = [stn_net(args, pretrained=pretrained), Encoder(), Predictor()]
    return [model.to(device) for model in models]


def model_state(models):
    return {"STN": models[0].state_dict(), "ENC": models[1].state_dict(), "PRED": models[2].state_dict()}


def load_model_state(models, checkpoint, device):
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    models[0].load_state_dict(state["STN"])
    models[1].load_state_dict(state["ENC"])
    models[2].load_state_dict(state["PRED"])
    return state


def augment_support(support):
    augmented = [support]
    for angle in [-np.pi / 4, -3 * np.pi / 16, -np.pi / 8, -np.pi / 16,
                  np.pi / 16, np.pi / 8, 3 * np.pi / 16, np.pi / 4]:
        augmented.append(rot_img(support, angle))
    for x_shift, y_shift in [(0.2, 0.2), (-0.2, 0.2), (-0.2, -0.2), (0.2, -0.2),
                             (0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1), (0.1, -0.1)]:
        augmented.append(translation_img(support, x_shift, y_shift))
    augmented.extend([hflip_img(support), grey_img(support)])
    augmented.extend(rot90_img(support, angle) for angle in [1, 2, 3])
    return torch.cat(augmented, dim=0)


def _feature_embedding(stn):
    embedding = stn.stn1_output
    embedding = embedding_concat(embedding, stn.stn2_output)
    return embedding_concat(embedding, stn.stn3_output)


@torch.no_grad()
def extract_query_embeddings(stn, test_loader, device):
    embeddings = []
    labels = []
    masks = []
    paths = []
    stn.eval()
    for query, mask, label, path in test_loader:
        stn(query.to(device, non_blocking=True))
        embeddings.append(_feature_embedding(stn).cpu())
        labels.extend(label.numpy().tolist())
        masks.append(mask.numpy())
        paths.extend(path)
    return (
        torch.cat(embeddings, dim=0),
        np.asarray(labels),
        np.concatenate(masks, axis=0)[:, 0],
        paths,
    )


@torch.no_grad()
def infer_anomaly_maps(stn, support, query_embeddings, device, image_size, seed, location_chunk=32):
    stn.eval()

    support = augment_support(support.to(device))
    generator = torch.Generator(device=support.device)
    generator.manual_seed(seed)
    support = support[torch.randperm(support.size(0), generator=generator, device=support.device)]

    stn(support)
    support_embedding = _feature_embedding(stn)

    sample_count, channels, height, width = support_embedding.shape
    flattened_support = support_embedding.reshape(sample_count, channels, height * width)
    mean = flattened_support.mean(dim=0)
    centered = flattened_support - mean.unsqueeze(0)
    centered /= math.sqrt(max(sample_count - 1, 1))

    queries = query_embeddings.to(device)
    flattened_queries = queries.reshape(queries.size(0), channels, height * width)
    squared_distances = torch.empty(
        flattened_queries.size(0), height * width, device=device, dtype=queries.dtype
    )
    regularization = 0.01
    identity = torch.eye(sample_count, device=device, dtype=queries.dtype)
    for start in range(0, height * width, location_chunk):
        stop = min(start + location_chunk, height * width)
        support_chunk = centered[:, :, start:stop]
        delta = flattened_queries[:, :, start:stop] - mean[:, start:stop]
        middle = torch.einsum("ncl,mcl->lnm", support_chunk, support_chunk)
        middle = middle / regularization + identity.unsqueeze(0)
        projection = torch.einsum("ncl,bcl->lnb", support_chunk, delta)
        solved = torch.linalg.solve(middle, projection)
        correction = (projection * solved).sum(dim=1).transpose(0, 1)
        squared = delta.square().sum(dim=1) / regularization
        squared -= correction / (regularization ** 2)
        squared_distances[:, start:stop] = torch.clamp_min(squared, 0)
    distances = torch.sqrt(squared_distances).reshape(-1, 1, height, width)
    score_maps = F.interpolate(
        distances, size=(image_size, image_size), mode="bilinear", align_corners=False
    ).squeeze(1).cpu().numpy()
    score_maps = np.stack([gaussian_filter(score, sigma=4) for score in score_maps])
    return score_maps


def compute_aurocs(score_maps, labels, masks):
    score_maps = np.asarray(score_maps, dtype=np.float64)
    minimum, maximum = score_maps.min(), score_maps.max()
    if maximum > minimum:
        score_maps = (score_maps - minimum) / (maximum - minimum)
    image_scores = score_maps.reshape(score_maps.shape[0], -1).max(axis=1)
    masks_bin = (np.asarray(masks) > 0.5).astype(np.uint8)
    metrics = compute_full_metrics(
        labels, image_scores, [mask for mask in masks_bin], [map_ for map_ in score_maps]
    )
    return metrics


METRIC_KEYS = ("i_auroc", "i_aupr", "p_auroc", "p_aupr", "p_aupro")


def evaluate(models, support_rounds, test_loader, device, image_size, seed, visual_dir=None,
             query_chunk=None):
    query_embeddings, labels, masks, paths = extract_query_embeddings(models[0], test_loader, device)
    rounds = []
    for index, support in enumerate(support_rounds):
        if query_chunk:
            chunks = []
            for start in range(0, query_embeddings.size(0), query_chunk):
                chunks.append(infer_anomaly_maps(
                    models[0], support, query_embeddings[start:start + query_chunk],
                    device, image_size, seed + index,
                ))
            scores = np.concatenate(chunks, axis=0)
        else:
            scores = infer_anomaly_maps(
                models[0], support, query_embeddings, device, image_size, seed + index
            )
        result = compute_aurocs(scores, labels, masks)
        result["round"] = index
        rounds.append(result)
        if visual_dir is not None and index == 0:
            from PIL import Image
            save_dir = Path(visual_dir)
            for score_map, mask, path in zip(scores, masks, paths):
                unique = Path(path).relative_to(
                    test_loader.dataset.root / test_loader.dataset.target_class / "test"
                )
                original = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
                save_visual(
                    save_dir / (str(unique).replace("/", "__")),
                    original, mask, score_map,
                )
    summary = {}
    for key in METRIC_KEYS:
        values = np.asarray([item[key] for item in rounds], dtype=float)
        summary[key] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    summary.update({"rounds": rounds})
    return summary


def train_epoch(models, loader, optimizers, device):
    for model in models:
        model.train()
    losses = []
    for queries, supports in loader:
        queries = queries.squeeze(0).to(device)
        supports = supports.squeeze(0).to(device)
        batch, shot, channels, height, width = supports.shape
        for optimizer in optimizers:
            optimizer.zero_grad()
        query_feature = models[0](queries)
        support_feature = models[0](supports.reshape(batch * shot, channels, height, width)) / shot
        support_feature = support_feature.reshape(batch, shot, *support_feature.shape[1:]).sum(dim=1)
        z_query = models[1](query_feature)
        z_support = models[1](support_feature)
        loss = (CosLoss(models[2](z_query), z_support) + CosLoss(models[2](z_support), z_query)) / 2
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def cosine_learning_rate(optimizers, initial_lr, epoch, epochs):
    learning_rate = initial_lr * 0.5 * (1 + math.cos(math.pi * epoch / epochs))
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
    return learning_rate


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def train_category(args, data_root, dataset, target_class, shot, output_dir, support_path, device):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    models = make_models(args, device, pretrained=not args.no_pretrained)
    optimizers = [
        torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum) for model in models
    ]
    train_data = IndustrialTrainDataset(
        data_root, dataset, target_class, shot, args.batch_size, args.img_size, args.seed
    )
    test_data = IndustrialTestDataset(data_root, dataset, target_class, args.img_size)
    loader_options = {
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, **loader_options)
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, **loader_options)
    support_rounds = load_support_rounds(data_root, support_path, args.img_size)

    best_score = -float("inf")
    history = []
    start_epoch = 0
    last_checkpoint = output_dir / "last.pt"
    if args.resume and last_checkpoint.is_file():
        state = load_model_state(models, last_checkpoint, device)
        for optimizer, optimizer_state in zip(optimizers, state["optimizers"]):
            optimizer.load_state_dict(optimizer_state)
        start_epoch = state["epoch"] + 1
        history_path = output_dir / "history.json"
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))
            history = [record for record in history if record["epoch"] <= state["epoch"]]
        train_data.shuffle_dataset(args.seed + start_epoch)
    started = time.time()
    for epoch in range(start_epoch, args.epochs):
        metrics = None
        if args.eval_every > 0 and epoch % args.eval_every == 0:
            metrics = evaluate(models, support_rounds, test_loader, device, args.img_size, args.seed)
            score = metrics["i_auroc"] + metrics["p_auroc"]
            if score > best_score:
                best_score = score
                torch.save({
                    **model_state(models),
                    "epoch": epoch,
                    "metrics": metrics,
                    "dataset": dataset,
                    "class": target_class,
                    "shot": shot,
                }, output_dir / "best.pt")
                save_json(output_dir / "best_metrics.json", metrics)
        learning_rate = cosine_learning_rate(optimizers, args.lr, epoch + 1, args.epochs)
        loss = train_epoch(models, train_loader, optimizers, device)
        record = {
            "epoch": epoch,
            "train_loss": loss,
            "learning_rate": learning_rate,
            "elapsed_seconds": time.time() - started,
            "metrics": metrics,
        }
        history.append(record)
        save_json(output_dir / "history.json", history)
        train_data.shuffle_dataset(args.seed + epoch + 1)
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            torch.save({
                **model_state(models),
                "optimizers": [optimizer.state_dict() for optimizer in optimizers],
                "epoch": epoch,
            }, last_checkpoint)
        print(
            f"[{dataset}/{target_class}/k={shot}] epoch {epoch + 1}/{args.epochs} "
            f"loss={loss:.6f}"
            + (f" I-AUROC={metrics['i_auroc']:.4f} P-AUROC={metrics['p_auroc']:.4f}" if metrics else ""),
            flush=True,
        )
    if args.eval_every == 0:
        torch.save({
            **model_state(models),
            "epoch": args.epochs - 1,
            "selection": "fixed_final_epoch",
            "dataset": dataset,
            "class": target_class,
            "shot": shot,
        }, output_dir / "best.pt")
    return output_dir / "best.pt"


def evaluate_checkpoint(args, data_root, dataset, target_class, shot, checkpoint, support_path, output, device):
    models = make_models(args, device, pretrained=False)
    state = load_model_state(models, checkpoint, device)
    test_data = IndustrialTestDataset(data_root, dataset, target_class, args.img_size)
    test_loader = DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    support_rounds = load_support_rounds(data_root, support_path, args.img_size)
    metrics = evaluate(
        models, support_rounds, test_loader, device, args.img_size, args.seed,
        visual_dir=Path(output).parent / "visual" if not getattr(args, "no_visual", False) else None,
    )
    metrics.update({
        "dataset": dataset,
        "class": target_class,
        "shot": shot,
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_epoch": state.get("epoch"),
    })
    save_json(output, metrics)
    return metrics


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
        writer = csv.DictWriter(handle, fieldnames=["dataset", "class", "shot", *METRIC_KEYS])
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in writer.fieldnames} for row in summary)
    save_json(output_json, summary)
    return summary

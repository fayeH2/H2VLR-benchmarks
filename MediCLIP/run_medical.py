import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, "/path/to/H2VLR")

import open_clip  # noqa: E402  (local bundled version)
from models.Adapter import Adapter  # noqa: E402
from models.Necker import Necker  # noqa: E402
from models.MapMaker import MapMaker  # noqa: E402
from datasets.dataset import TrainDataset  # noqa: E402
from utils.losses import FocalLoss, BinaryDiceLoss  # noqa: E402
from utils.misc_helper import AverageMeter, create_logger, set_seed  # noqa: E402
from _shared.eval_common import compute_metrics  # noqa: E402, E501

try:
    import cv2
    from PIL import Image
except Exception:  # pragma: no cover
    cv2 = None


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class IndexedTrainDataset(TrainDataset):
    """TrainDataset restricted to an explicit list of support indices."""

    def __init__(self, args, source, preprocess, support_indices, **kwargs):
        self.args = args
        self.source = source
        self.transform_img = preprocess
        self.support_indices = list(support_indices)
        self.data_to_iterate = self.get_image_data()
        self.augs, self.augs_pro = self.load_anomaly_syn()
        assert sum(self.augs_pro) == 1.0

    def get_image_data(self):
        data = []
        with open(os.path.join(self.source, "samples", "train.json"), "r") as fh:
            for line in fh:
                data.append(json.loads(line))
        return [data[i] for i in self.support_indices]


class MedicalTestDataset(torch.utils.data.Dataset):
    def __init__(self, args, source, preprocess):
        super().__init__()
        self.args = args
        self.source = source
        self.transform_img = preprocess
        self.data_to_iterate = self.get_image_data()

    def get_image_data(self):
        data = []
        with open(os.path.join(self.source, "samples", "test.json"), "r") as fh:
            for line in fh:
                data.append(json.loads(line))
        return data

    def __len__(self):
        return len(self.data_to_iterate)

    def __getitem__(self, idx):
        info = self.data_to_iterate[idx]
        image_path = os.path.join(self.source, "images", info["filename"])
        image = Image.open(image_path).convert("RGB").resize(
            (self.args.image_size, self.args.image_size), Image.BILINEAR
        )
        mask_rel = info.get("mask", "")
        if mask_rel:
            mask_path = os.path.join(self.source, "images", mask_rel)
            mask = Image.open(mask_path).convert("L").resize(
                (self.args.image_size, self.args.image_size), Image.NEAREST
            )
            mask = (np.array(mask).astype(np.float32) / 255.0)
            mask[mask != 0.0] = 1.0
        else:
            mask = np.zeros((self.args.image_size, self.args.image_size), dtype=np.float32)
        image = self.transform_img(image)
        return {
            "image": image,
            "mask": torch.from_numpy(mask),
            "is_anomaly": int(info["label"]),
            "image_path": image_path,
            "clsname": info["clsname"],
        }


# ---------------------------------------------------------------------------
# Model assembly (mirrors train.py)
# ---------------------------------------------------------------------------
def build_components(cfg, device):
    model, preprocess, model_cfg = open_clip.create_model_and_transforms(
        cfg.model_name, cfg.image_size, device=device
    )
    for param in model.parameters():
        param.requires_grad_(False)
    cfg.model_cfg = model_cfg

    img = torch.ones((1, 3, model_cfg["vision_cfg"]["image_size"],
                      model_cfg["vision_cfg"]["image_size"]), device=model.device)
    with torch.no_grad():
        _, tokens = model.encode_image(img, cfg.layers_out)
    import math
    if len(tokens[0].shape) == 3:
        model.token_size = [int(math.sqrt(t.shape[1] - 1)) for t in tokens]
        model.token_c = [t.shape[-1] for t in tokens]
    else:
        model.token_size = [t.shape[2] for t in tokens]
        model.token_c = [t.shape[1] for t in tokens]
    model.embed_dim = model_cfg["embed_dim"]

    necker = Necker(clip_model=model).to(model.device)
    adapter = Adapter(clip_model=model, target=model_cfg["embed_dim"]).to(model.device)

    from models.CoOp import PromptMaker
    prompt_maker = PromptMaker(
        prompts=cfg.prompts,
        clip_model=model,
        n_ctx=cfg.n_learnable_token,
        CSC=cfg.CSC,
        class_token_position=cfg.class_token_positions,
    ).to(model.device)
    map_maker = MapMaker(image_size=cfg.image_size).to(model.device)
    return model, preprocess, necker, adapter, prompt_maker, map_maker


# ---------------------------------------------------------------------------
# Author's pixel normalisation (from test.py)
# ---------------------------------------------------------------------------
def pixel_normalization(segmentations, image_size, avgpool_size=128):
    segmentations = torch.tensor(segmentations[:, None, ...]).cuda()
    segmentations = torch.nn.functional.interpolate(
        segmentations, (image_size, image_size), mode="bilinear", align_corners=True
    )
    seg_pooled = torch.nn.functional.avg_pool2d(
        segmentations, (avgpool_size, avgpool_size), stride=1
    ).cpu().numpy()
    min_scores = seg_pooled.reshape(-1).min(axis=-1).reshape(1)
    max_scores = seg_pooled.reshape(-1).max(axis=-1).reshape(1)
    segmentations = segmentations.squeeze(1).cpu().numpy()
    segmentations = (segmentations - min_scores) / (max_scores - min_scores)
    segmentations = np.clip(segmentations, a_min=0, a_max=1)
    for i in range(segmentations.shape[0]):
        segmentations[i] = cv2.GaussianBlur(segmentations[i], (5, 5), 0)
    return segmentations


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_inference(test_dataset, model, necker, adapter, prompt_maker,
                  map_maker, cfg, batch_size, desc="test"):
    loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    image_scores, image_labels = [], []
    raw_maps, masks = [], []
    image_paths = []
    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"].to(model.device)
        _, image_tokens = model.encode_image(images, out_layers=cfg.layers_out)
        image_features = necker(image_tokens)
        vision_adapter_features = adapter(image_features)
        prompt_adapter_features = prompt_maker(vision_adapter_features)
        anomaly_map = map_maker(vision_adapter_features, prompt_adapter_features)

        B, _, H, W = anomaly_map.shape
        anomaly_map = anomaly_map[:, 1, :, :]
        score, _ = torch.max(anomaly_map.view(B, H * W), dim=-1)
        image_scores.extend(score.cpu().numpy().tolist())
        image_labels.extend(batch["is_anomaly"].numpy().tolist())
        raw_maps.append(anomaly_map.cpu().numpy())
        masks.append(batch["mask"].numpy())
        image_paths.extend(batch["image_path"])

    raw_maps = np.concatenate(raw_maps, axis=0)          # N,H,W
    masks = np.concatenate(masks, axis=0)                # N,H,W
    return image_scores, image_labels, raw_maps, masks, image_paths


def evaluate(test_dataset, *components, cfg, batch_size):
    image_scores, image_labels, raw_maps, masks, paths = run_inference(
        test_dataset, *components, cfg=cfg, batch_size=batch_size
    )
    norm_maps = pixel_normalization(raw_maps, cfg.image_size)
    metrics = compute_metrics(image_labels, image_scores, masks, norm_maps, aupro=True)
    return metrics, image_scores, image_labels, norm_maps, masks, paths


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_one_epoch(train_loader, optimizer, model, necker, adapter,
                    prompt_maker, map_maker, cfg, logger, epoch):
    focal = FocalLoss()
    dice = BinaryDiceLoss()
    adapter.train()
    prompt_maker.train()
    meter = AverageMeter(cfg.print_freq_step)
    for i, batch in enumerate(train_loader):
        images = batch["image"].to(model.device)
        gt_mask = batch["mask"].to(model.device).float()
        with torch.no_grad():
            _, image_tokens = model.encode_image(images, out_layers=cfg.layers_out)
            image_features = necker(image_tokens)
        vision_adapter_features = adapter(image_features)
        prompt_adapter_features = prompt_maker(vision_adapter_features)
        anomaly_map = map_maker(vision_adapter_features, prompt_adapter_features)
        loss = focal(anomaly_map, gt_mask) + dice(anomaly_map[:, 1, :, :], gt_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        meter.update(loss.item())
        if (i + 1) % cfg.print_freq_step == 0:
            logger.info("Epoch [{}/{}] iter {}/{} loss {:.5f} ({:.5f})".format(
                epoch, cfg.epoch, i + 1, len(train_loader), meter.val, meter.avg))
    return meter.avg


def main(args):
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    with open(args.config_path) as fh:
        cfg = EasyDict(yaml.load(fh, Loader=yaml.FullLoader))
    cfg.data_root = str(SCRIPT_DIR / "data")
    cfg.epoch = args.epoch
    cfg.train_dataset = args.dataset
    cfg.test_datasets = [args.dataset]

    data_dir = os.path.join(cfg.data_root, args.dataset)
    support_indices = [int(x) for x in args.support_indices.replace(",", " ").split()]

    out_dir = Path(args.output_root) / args.dataset / f"k{args.k_shot}" / args.clsname
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "result.json").exists() and not args.overwrite:
        print(f"[skip] {out_dir/'result.json'} exists")
        return

    logger = create_logger("med", str(out_dir / "train.log"))
    logger.info(f"args: {vars(args)}")
    logger.info(f"support_indices: {support_indices}")

    model, preprocess, necker, adapter, prompt_maker, map_maker = build_components(cfg, device)

    train_dataset = IndexedTrainDataset(
        args=cfg, source=data_dir, preprocess=preprocess,
        support_indices=support_indices,
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, drop_last=False)
    test_dataset = MedicalTestDataset(args=cfg, source=data_dir, preprocess=preprocess)
    logger.info(f"train={len(train_dataset)} test={len(test_dataset)}")

    optimizer = torch.optim.Adam([
        {"params": prompt_maker.prompt_learner.parameters(), "lr": 0.001},
        {"params": adapter.parameters(), "lr": 0.001},
    ], lr=0.001, betas=(0.5, 0.999))

    best_score = -1.0
    best_epoch = -1
    for epoch in range(1, cfg.epoch + 1):
        train_one_epoch(train_loader, optimizer, model, necker, adapter,
                        prompt_maker, map_maker, cfg, logger, epoch)
        if epoch % args.val_freq == 0 or epoch == cfg.epoch:
            img_scores, img_labels, _, _, _ = run_inference(
                test_dataset, model, necker, adapter, prompt_maker, map_maker,
                cfg=cfg, batch_size=cfg.batch_size, desc=f"val e{epoch}")
            from sklearn.metrics import roc_auc_score
            img_auroc = roc_auc_score(img_labels, img_scores)
            logger.info(f"[val] epoch {epoch} image-auroc {img_auroc:.4f}")
            if img_auroc > best_score:
                best_score = img_auroc
                best_epoch = epoch
                torch.save({
                    "adapter_state_dict": adapter.state_dict(),
                    "prompt_state_dict": prompt_maker.prompt_learner.state_dict(),
                }, out_dir / "checkpoint.pkl")

    # final evaluation with the best checkpoint
    ckpt = torch.load(out_dir / "checkpoint.pkl", map_location="cpu")
    adapter.load_state_dict(ckpt["adapter_state_dict"])
    prompt_maker.prompt_learner.load_state_dict(ckpt["prompt_state_dict"])
    adapter.eval()
    prompt_maker.eval()

    metrics, image_scores, image_labels, norm_maps, masks, paths = evaluate(
        test_dataset, model, necker, adapter, prompt_maker, map_maker,
        cfg=cfg, batch_size=cfg.batch_size,
    )
    payload = {
        "method": "MediCLIP",
        "dataset": args.dataset,
        "shot": args.k_shot,
        "class": args.clsname,
        "best_epoch": best_epoch,
        "best_val_image_auroc": float(best_score),
        "i_auroc": metrics["i_auroc"],
        "i_aupr": metrics["i_aupr"],
        "p_auroc": metrics["p_auroc"],
        "p_aupr": metrics.get("p_aupr", float("nan")),
        "p_aupro": metrics.get("p_aupro", float("nan")),
    }
    with (out_dir / "result.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("RESULT " + json.dumps(payload))
    print("RESULT", json.dumps(payload))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediCLIP medical benchmark")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config_path", default=str(SCRIPT_DIR / "config" / "brainmri.yaml"))
    parser.add_argument("--clsname", default=None)
    parser.add_argument("--k_shot", type=int, default=1)
    parser.add_argument("--support_indices", type=str, default="0")
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--val_freq", type=int, default=50)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-root", default="outputs_medical")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.clsname is None:
        args.clsname = {"brainmri": "brain", "liverct": "liver", "busi": "breast"}[args.dataset]
    main(args)

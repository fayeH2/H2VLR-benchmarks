#!/usr/bin/env python3
"""INP-Former few-shot training + evaluation on BTAD / MPDD / BeltAD.

Faithful port of the official INP_Former_Few_Shot.py pipeline (same model
architecture, augmentation, optimizer and 200-epoch schedule) adapted to the
generic industrial dataset layout, reporting the four shared metrics
(I-AUROC, I-AUPR, P-AUROC, P-AUPRO) and per-image visualisation.

One model is trained per dataset on the union of the per-class one-shot
supports (exactly like the official MVTec/VisA runs).
"""

import argparse
import os
import random
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import warnings
from tqdm import tqdm
from torch.nn.init import trunc_normal_
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from optimizers import StableAdamW  # noqa: E402
from utils import (  # noqa: E402
    WarmCosineScheduler, augmentation, evaluation_maps, get_logger,
    global_cosine_hm_adaptive, setup_seed,
)
from dataset import MVTecDataset, get_data_transforms  # noqa: E402
from models import vit_encoder  # noqa: E402
from models.uad import INP_Former  # noqa: E402
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block  # noqa: E402

_SHARED = Path(__file__).resolve().parent.parent / "_shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

from eval_common import compute_metrics, save_json, save_visual  # noqa: E402
from industrial_data import (  # noqa: E402
    dataset_dir_name, get_classes, validate_dataset,
)

DATASET_META = {
    "btad": {"name": "BTAD", "classes": None},
    "mpdd": {"name": "MPDD", "classes": None},
    "beltad": {"name": "BeltAD", "classes": None},
}


def mean_metrics(per_class):
    summed = {}
    count = 0
    for payload, _visual_dir in per_class:
        for key in ("i_auroc", "i_aupr", "p_auroc", "p_aupro"):
            summed[key] = summed.get(key, 0.0) + payload[key]
        count += 1
    return {key: value / max(count, 1) for key, value in summed.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["btad", "mpdd", "beltad"], required=True)
    parser.add_argument("--data-root", default="/path/to/datasets/industrial")
    parser.add_argument("--output-root", default="outputs_industrial")
    parser.add_argument("--save-dir", default="saved_results")
    parser.add_argument("--encoder", default="dinov2reg_vit_base_14")
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--crop-size", type=int, default=392)
    parser.add_argument("--INP-num", type=int, default=6)
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--total-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-visual", action="store_true")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training and load model.pth from --save-dir")
    args = parser.parse_args()

    setup_seed(args.seed)
    data_root = Path(args.data_root) / dataset_dir_name(args.dataset)
    validate_dataset(data_root, args.dataset)
    item_list = get_classes(args.dataset)
    logger = get_logger(
        f"INP-Former-Industrial-{args.dataset}-{args.shot}",
        Path(args.save_dir) / f"INP-Former-Industrial-{args.dataset}-{args.shot}",
    )
    print_fn = logger.info

    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size)
    device = args.device if torch.cuda.is_available() else "cpu"

    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    encoder = vit_encoder.load(args.encoder)
    embed_dim, num_heads = 768, 12

    Bottleneck = nn.ModuleList([Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.)])
    INP = nn.ParameterList(
        [nn.Parameter(torch.randn(args.INP_num, embed_dim)) for _ in range(1)]
    )
    INP_Extractor = nn.ModuleList([
        Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                          qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        for _ in range(1)
    ])
    INP_Guided_Decoder = nn.ModuleList([
        Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        for _ in range(8)
    ])

    model = INP_Former(
        encoder=encoder, bottleneck=Bottleneck, aggregation=INP_Extractor,
        decoder=INP_Guided_Decoder, target_layers=target_layers,
        remove_class_token=True, fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder, prototype_token=INP,
    ).to(device)

    trainable = nn.ModuleList([Bottleneck, INP_Guided_Decoder, INP_Extractor, INP])
    for module in trainable.modules():
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.01, a=-0.03, b=0.03)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    train_data_list = []
    test_data_list = []
    for i, item in enumerate(item_list):
        train_path = data_root / item / "train"
        train_data = ImageFolder(root=str(train_path), transform=data_transform)
        train_data.classes = item
        train_data.class_to_idx = {item: i}
        token = f"inp:{args.dataset}:{item}:{args.shot}:{args.seed}"
        rng = random.Random(int.from_bytes(token.encode("utf-8")[:8], "big"))
        chosen = rng.sample(train_data.samples, args.shot)
        train_data.samples = chosen
        print_fn(train_data.samples[:3])
        augmented_data = []
        for img, label in train_data:
            augmented_imgs = augmentation(img)
            for aug_img in augmented_imgs:
                augmented_data.append((aug_img, label))
        augmented_dataset = torch.utils.data.TensorDataset(
            torch.stack([item[0] for item in augmented_data]),
            torch.tensor([item[1] for item in augmented_data]),
        )
        test_data = MVTecDataset(root=str(data_root / item), transform=data_transform,
                                 gt_transform=gt_transform, phase="test")
        train_data_list.append(augmented_dataset)
        test_data_list.append(test_data)
    train_data = ConcatDataset(train_data_list)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, drop_last=True)

    optimizer = StableAdamW([{'params': trainable.parameters()}],
                            lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4,
                            amsgrad=True, eps=1e-10)
    lr_scheduler = WarmCosineScheduler(
        optimizer, base_value=1e-3, final_value=1e-4,
        total_iters=args.total_epochs * len(train_dataloader), warmup_iters=100,
    )
    print_fn(f"train image number:{len(train_data)}")
    started = time.time()

    save_dir = Path(args.save_dir) / f"INP-Former-Industrial-{args.dataset}-{args.shot}"
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "model.pth"
    if args.eval_only:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        print_fn(f"loaded {model_path}")
    else:
        for epoch in range(args.total_epochs):
            model.train()
            loss_list = []
            for img, _ in tqdm(train_dataloader, ncols=80):
                img = img.to(device)
                en, de, g_loss = model(img)
                loss = global_cosine_hm_adaptive(en, de, y=3)
                loss = loss + 0.2 * g_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)
                optimizer.step()
                lr_scheduler.step()
                loss_list.append(loss.item())
            print_fn(f"[{args.dataset}] epoch [{epoch + 1}/{args.total_epochs}] "
                     f"loss:{np.mean(loss_list):.4f} elapsed:{time.time() - started:.0f}s")
        torch.save(model.state_dict(), model_path)

    per_class = []
    for item, test_data in zip(item_list, test_data_list):
        test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.workers)
        gt_px, pr_px, gt_sp, pr_sp, img_paths = evaluation_maps(
            model, test_dataloader, device, max_ratio=0.01, resize_mask=256
        )
        metrics = compute_metrics(
            gt_sp, pr_sp, [row for row in gt_px], [row for row in pr_px]
        )
        payload = {
            "dataset": args.dataset,
            "class": item,
            "shot": args.shot,
            **metrics,
            "train_image_number": len(train_data),
            "model": str((save_dir / "model.pth").resolve()),
        }
        result_path = (Path(args.output_root) / args.dataset / f"k{args.shot}"
                       / item / "result.json")
        save_json(result_path, payload)
        visual_dir = None
        if not args.no_visual:
            from PIL import Image
            visual_dir = (Path(args.output_root) / args.dataset / f"k{args.shot}"
                          / item / "visual")
            gt_masks = [(row > 0.5).astype(np.uint8) for row in gt_px]
            for score_map, mask, path in zip(pr_px, gt_masks, img_paths):
                unique = Path(path).relative_to(data_root / item / "test")
                original = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
                save_visual(visual_dir / str(unique).replace("/", "__"),
                            original, mask, score_map)
            payload["visual_dir"] = str(visual_dir)
        per_class.append((payload, visual_dir))
        print_fn(f"{item}: I-AUROC:{metrics['i_auroc']:.4f} I-AUPR:{metrics['i_aupr']:.4f} "
                 f"P-AUROC:{metrics['p_auroc']:.4f} P-AUPRO:{metrics['p_aupro']:.4f}")

    mean = mean_metrics(per_class)
    summary_dir = Path(args.output_root) / args.dataset / f"k{args.shot}"
    save_json(summary_dir / "summary.json", {
        "dataset": args.dataset, "shot": args.shot, "mean": mean,
        "classes": [item[0] for item in per_class],
    })
    print(f"[{args.dataset}] mean " + " ".join(f"{k}:{v:.4f}" for k, v in mean.items()))


if __name__ == "__main__":
    main()
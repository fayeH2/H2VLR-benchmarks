#!/usr/bin/env python3
"""INP-Former few-shot training + evaluation on BrainMRI / LiverCT / BUSI.

Direct port of INP_Former_Industrial.py to the shared medical protocol:
the few-shot support is the canonical selection (from _shared/medical_data.py),
the test split / masks come from the same shared module, and the four shared
metrics (I-AUROC, I-AUPR, P-AUROC, P-AUPRO) are reported.
"""

import argparse
import os
import sys
import time
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torch.nn.init import trunc_normal_
from torch.utils.data import DataLoader, Dataset, TensorDataset

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from optimizers import StableAdamW  # noqa: E402
from utils import (  # noqa: E402
    WarmCosineScheduler, augmentation, evaluation_maps, get_logger,
    global_cosine_hm_adaptive, setup_seed,
)
from dataset import get_data_transforms  # noqa: E402
from models import vit_encoder  # noqa: E402
from models.uad import INP_Former  # noqa: E402
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block  # noqa: E402

_SHARED = Path(__file__).resolve().parent.parent / "_shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

from eval_common import compute_metrics, rounded_metrics, save_json  # noqa: E402
from medical_data import canonical_support_paths, clsname, collect_test_samples  # noqa: E402


class MedicalSupportImages(Dataset):
    def __init__(self, paths, data_transform):
        self.paths = paths
        self.data_transform = data_transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.data_transform(image.convert("RGB")), 0


class MedicalTestDataset(Dataset):
    """Yields (image, gt[1,h,w], label, path) for INP-Former's evaluator."""

    def __init__(self, dataset, data_transform, gt_transform):
        self.samples = collect_test_samples(dataset)
        self.data_transform = data_transform
        self.gt_transform = gt_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path, label = self.samples[index]
        with Image.open(image_path) as image:
            img = self.data_transform(image.convert("RGB"))
        if label == 0 or mask_path is None:
            gt = torch.zeros(1, img.shape[-2], img.shape[-1])
        else:
            with Image.open(mask_path) as mask:
                gt = self.gt_transform(mask.convert("L"))
        return img, gt, label, image_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["brainmri", "liverct", "busi"], required=True)
    parser.add_argument("--output-root", default="outputs_medical")
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
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training and load model.pth from --save-dir")
    args = parser.parse_args()

    setup_seed(args.seed)
    dataset = args.dataset
    target_class = clsname(dataset)
    logger = get_logger(
        f"INP-Former-Medical-{dataset}-{args.shot}",
        Path(args.save_dir) / f"INP-Former-Medical-{dataset}-{args.shot}",
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

    support_paths = [str(p) for p in canonical_support_paths(dataset, args.shot)]
    print_fn(f"canonical support ({len(support_paths)}): {support_paths}")
    augmented = []
    support_dataset = MedicalSupportImages(support_paths, data_transform)
    for img, _ in support_dataset:
        for aug_img in augmentation(img):
            augmented.append((aug_img, 0))
    train_data = TensorDataset(
        torch.stack([item[0] for item in augmented]),
        torch.tensor([item[1] for item in augmented]),
    )
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, drop_last=True)

    optimizer = StableAdamW([{'params': trainable.parameters()}],
                            lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4,
                            amsgrad=True, eps=1e-10)
    lr_scheduler = WarmCosineScheduler(
        optimizer, base_value=1e-3, final_value=1e-4,
        total_iters=args.total_epochs * max(len(train_dataloader), 1), warmup_iters=100,
    )
    print_fn(f"train image number:{len(train_data)}")
    started = time.time()

    save_dir = Path(args.save_dir) / f"INP-Former-Medical-{dataset}-{args.shot}"
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "model.pth"
    if args.eval_only:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        print_fn(f"loaded {model_path}")
    else:
        for epoch in range(args.total_epochs):
            model.train()
            loss_list = []
            for img, _ in tqdm(train_dataloader, ncols=80, disable=True):
                img = img.to(device)
                en, de, g_loss = model(img)
                loss = global_cosine_hm_adaptive(en, de, y=3) + 0.2 * g_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)
                optimizer.step()
                lr_scheduler.step()
                loss_list.append(loss.item())
            if (epoch + 1) % 20 == 0 or epoch + 1 == args.total_epochs:
                print_fn(f"[{dataset}] epoch [{epoch + 1}/{args.total_epochs}] "
                         f"loss:{np.mean(loss_list):.4f} elapsed:{time.time() - started:.0f}s")
        torch.save(model.state_dict(), model_path)

    test_data = MedicalTestDataset(dataset, data_transform, gt_transform)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers)
    gt_px, pr_px, gt_sp, pr_sp, img_paths = evaluation_maps(
        model, test_dataloader, device, max_ratio=0.01, resize_mask=256
    )
    metrics = compute_metrics(gt_sp, pr_sp, [row for row in gt_px], [row for row in pr_px])
    payload = {
        "method": "INP-Former",
        "dataset": dataset,
        "class": target_class,
        "shot": args.shot,
        **rounded_metrics(metrics),
        "train_image_number": len(train_data),
        "model": str(model_path.resolve()),
    }
    result_path = Path(args.output_root) / dataset / f"k{args.shot}" / target_class / "result.json"
    save_json(result_path, payload)
    save_json(Path(args.output_root) / dataset / f"k{args.shot}" / "summary.json", {
        "dataset": dataset, "shot": args.shot, "class": target_class, "mean": metrics,
    })
    print_fn(f"{target_class}: I-AUROC:{metrics['i_auroc']:.4f} I-AUPR:{metrics['i_aupr']:.4f} "
             f"P-AUROC:{metrics['p_auroc']:.4f} P-AUPRO:{metrics['p_aupro']:.4f}")


if __name__ == "__main__":
    main()

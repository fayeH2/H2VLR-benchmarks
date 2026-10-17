#!/usr/bin/env python3
"""PromptAD few-shot training + evaluation on BTAD / MPDD / BeltAD.

Single-run adaptation of the official train_seg.py pipeline (prompt-learner
SGD training, pixel-level v2t + triplet + match losses, 100 epochs) that, at
every epoch, computes the four shared metrics (I-AUROC, I-AUPR, P-AUROC,
P-AUPRO), keeps the epoch with the best P-AUROC, writes per-class result.json
and saves per-image visualisations (original image, GT mask, prediction map).

Because classification and segmentation share one prompt-learner, the image
score is the per-image max of the pixel anomaly map (the training signal is
the same SEG objective as the official reproduction).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from loguru import logger
import torch.optim.lr_scheduler
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

_shared = Path(__file__).resolve().parent.parent / "_shared"
if str(_shared) not in sys.path:
    sys.path.insert(0, str(_shared))
from eval_common import compute_metrics, save_json, save_visual  # noqa: E402

from datasets import get_dataloader_from_args, dataset_classes  # noqa: E402
from datasets.industrial import load_btad, load_mpdd, load_beltad  # noqa: E402
from PromptAD import PromptAD, TripletLoss  # noqa: E402
from utils.training_utils import setup_seed  # noqa: E402
from utils.eval_utils import specify_resolution  # noqa: E402

TASK = 'SEG'

_LOADERS = {"btad": load_btad, "mpdd": load_mpdd, "beltad": load_beltad}

gt_labels = []


def evaluate_epoch(model, dataloader, device, resolution, names_from_index):
    score_maps = []
    gt_mask_list = []
    indices = []
    for (data, mask, label, name, img_type) in dataloader:
        data = [model.transform(Image.fromarray(f.numpy())) for f in data]
        data = torch.stack(data, dim=0)
        for _d, _n, _l, m in zip(data, name, label, mask):
            m = m.numpy()
            m[m > 0] = 1
            gt_mask_list += [m]
            indices.append(names_from_index.pop(0))
        data = data.to(device)
        score_map = model(data, 'seg')
        score_maps += score_map
    resized_maps = [
        cv2.resize(np.asarray(sm, dtype=np.float32), (resolution, resolution),
                   interpolation=cv2.INTER_CUBIC)
        for sm in score_maps
    ]
    resized_masks = [
        cv2.resize(np.asarray(gm, dtype=np.uint8), (resolution, resolution),
                   interpolation=cv2.INTER_NEAREST)
        for gm in gt_mask_list
    ]
    return resized_maps, resized_masks, indices


def fit(model, args, dataloader, device, check_path, train_data, output_dir,
        original_paths):
    model.eval_mode()
    features1 = []
    features2 = []
    for (data, mask, label, name, img_type) in train_data:
        data = [model.transform(Image.fromarray(cv2.cvtColor(f.numpy(), cv2.COLOR_BGR2RGB))) for f in data]
        data = torch.stack(data, dim=0).to(device)
        _, _, feature_map1, feature_map2 = model.encode_image(data)
        features1.append(feature_map1)
        features2.append(feature_map2)
    features1 = torch.cat(features1, dim=0)
    features2 = torch.cat(features2, dim=0)
    model.build_image_feature_gallery(features1, features2)

    optimizer = torch.optim.SGD(model.prompt_learner.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.Epoch, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss().to(device)
    criterion_tip = TripletLoss(margin=0.0)

    best = None
    for epoch in range(args.Epoch):
        for (data, mask, label, name, img_type) in train_data:
            data = [model.transform(Image.fromarray(cv2.cvtColor(f.numpy(), cv2.COLOR_BGR2RGB))) for f in data]
            data = torch.stack(data, dim=0).to(device)
            data = data.to(device)

            normal_text_prompt, abnormal_text_prompt_handle, abnormal_text_prompt_learned = model.prompt_learner()
            optimizer.zero_grad()

            normal_text_features = model.encode_text_embedding(
                normal_text_prompt, model.tokenized_normal_prompts)
            abnormal_text_features_handle = model.encode_text_embedding(
                abnormal_text_prompt_handle, model.tokenized_abnormal_prompts_handle)
            abnormal_text_features_learned = model.encode_text_embedding(
                abnormal_text_prompt_learned, model.tokenized_abnormal_prompts_learned)
            abnormal_text_features = torch.cat([abnormal_text_features_handle,
                                                abnormal_text_features_learned], dim=0)

            mean_ad_handle = torch.mean(F.normalize(abnormal_text_features_handle, dim=-1), dim=0)
            mean_ad_learned = torch.mean(F.normalize(abnormal_text_features_learned, dim=-1), dim=0)
            loss_match_abnormal = (mean_ad_handle - mean_ad_learned).norm(dim=0) ** 2.0

            _, feature_map, _, _ = model.encode_image(data)

            normal_text_features_anchor = normal_text_features.mean(dim=0).unsqueeze(0)
            normal_text_features_anchor = normal_text_features_anchor / normal_text_features_anchor.norm(dim=-1, keepdim=True)
            abnormal_text_features_anchor = abnormal_text_features.mean(dim=0).unsqueeze(0)
            abnormal_text_features_anchor = abnormal_text_features_anchor / abnormal_text_features_anchor.norm(dim=-1, keepdim=True)
            abnormal_text_features = abnormal_text_features / abnormal_text_features.norm(dim=-1, keepdim=True)

            l_pos = torch.einsum('nic,cj->nij', feature_map, normal_text_features_anchor.transpose(0, 1))
            l_neg_v2t = torch.einsum('nic,cj->nij', feature_map, abnormal_text_features.transpose(0, 1))

            if model.precision == 'fp16':
                logit_scale = model.model.logit_scale.half()
            else:
                logit_scale = model.model.logit_scale

            logits_v2t = torch.cat([l_pos, l_neg_v2t], dim=-1) * logit_scale
            target_v2t = torch.zeros([logits_v2t.shape[0], logits_v2t.shape[1]], dtype=torch.long).to(device)

            loss_v2t = criterion(logits_v2t.transpose(1, 2), target_v2t)
            trip_loss = criterion_tip(feature_map, normal_text_features_anchor,
                                      abnormal_text_features_anchor)
            loss = loss_v2t + trip_loss + loss_match_abnormal * args.lambda1

            loss.backward()
            optimizer.step()

        scheduler.step()
        model.build_text_feature_gallery()

        score_maps, gt_mask_list, indices = evaluate_epoch(
            model, dataloader, device, args.resolution,
            [i for i in range(len(original_paths))])
        gt_sp = []
        for i in indices:
            gt_sp.append(gt_labels[i])
        score_imgs = [float(np.max(am)) for am in score_maps]
        metrics = compute_metrics(
            np.asarray(gt_sp, dtype=int), np.asarray(score_imgs, dtype=float),
            [row for row in gt_mask_list], [row for row in score_maps],
        )

        if best is None or best["p_auroc"] < metrics["p_auroc"]:
            best = dict(metrics)
            best["epoch"] = epoch + 1
            torch.save({
                "metrics": metrics,
                "state_dict": model.state_dict(),
            }, check_path)
            if args.vis:
                visual_dir = output_dir / "visual"
                visual_dir.mkdir(parents=True, exist_ok=True)
                class_test_root = Path(original_paths[0]).parent.parent
                for index, score_map, gt_mask in zip(indices, score_maps, gt_mask_list):
                    original = np.asarray(
                        Image.open(original_paths[index]).convert("RGB"), dtype=np.uint8)
                    unique = str(Path(original_paths[index]).relative_to(
                        class_test_root)).replace(os.sep, "__")
                    save_visual(visual_dir / unique, original,
                                (gt_mask > 0.5).astype(np.uint8), score_map)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"[{args.class_name}] epoch {epoch + 1}/{args.Epoch} "
                        f"i_auroc:{metrics['i_auroc']:.4f} p_auroc:{metrics['p_auroc']:.4f} "
                        f"p_aupro:{metrics['p_aupro']:.4f}")
    return best


def main(args):
    kwargs = vars(args)
    if kwargs['seed'] is None:
        kwargs['seed'] = 111
    setup_seed(kwargs['seed'])
    kwargs['device'] = 'cuda:0'

    train_dataloader, train_dataset_inst = get_dataloader_from_args(
        phase='train', perturbed=False, **kwargs)
    test_dataloader, test_dataset_inst = get_dataloader_from_args(
        phase='test', perturbed=False, **kwargs)

    kwargs['out_size_h'] = kwargs['resolution']
    kwargs['out_size_w'] = kwargs['resolution']

    model = PromptAD(**kwargs)
    model = model.to(kwargs['device'])

    loader = _LOADERS[kwargs['dataset']]
    _train, (test_paths, test_masks, test_labels, test_types) = loader(
        kwargs['class_name'], kwargs['k_shot'])
    global gt_labels
    gt_labels = test_labels

    output_dir = Path(args.output_root) / args.dataset / f"k{args.k_shot}" / args.class_name
    output_dir.mkdir(parents=True, exist_ok=True)
    check_path = output_dir / "check_point.pt"

    best = fit(model, args, test_dataloader, kwargs['device'], check_path,
               train_dataloader, output_dir, test_paths)
    payload = {
        "dataset": args.dataset,
        "class": args.class_name,
        "shot": args.k_shot,
        "seed": kwargs['seed'],
        "epoch": best["epoch"],
        "i_auroc": round(float(best["i_auroc"]), 4),
        "i_aupr": round(float(best["i_aupr"]), 4),
        "p_auroc": round(float(best["p_auroc"]), 4),
        "p_aupr": round(float(best["p_aupr"]), 4),
        "p_aupro": round(float(best["p_aupro"]), 4),
        "loss": args.lambda1,
        "resolution": args.resolution,
    }
    save_json(output_dir / "result.json", payload)
    print(f"Object:{args.class_name} ================ "
          f"I-AUROC:{payload['i_auroc']} I-AUPR:{payload['i_aupr']} "
          f"P-AUROC:{payload['p_auroc']} P-AUPRO:{payload['p_aupro']}")


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_args():
    parser = argparse.ArgumentParser(description='PromptAD industrial')
    parser.add_argument('--dataset', type=str, default='btad',
                        choices=['btad', 'mpdd', 'beltad'])
    parser.add_argument('--class_name', type=str, default='01')

    parser.add_argument('--img-resize', type=int, default=240)
    parser.add_argument('--img-cropsize', type=int, default=240)
    parser.add_argument('--resolution', type=int, default=400)

    parser.add_argument('--batch-size', type=int, default=400)
    parser.add_argument('--vis', type=str2bool, choices=[True, False], default=True)
    parser.add_argument("--root-dir", type=str, default="./result")
    parser.add_argument("--output-root", type=str, default="outputs_industrial")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--gpu-id", type=int, default=0)

    parser.add_argument('--k-shot', type=int, default=1)
    parser.add_argument("--backbone", type=str, default="ViT-B-16-plus-240",
                        choices=['ViT-B-16-plus-240', 'ViT-B-16'])
    parser.add_argument("--pretrained_dataset", type=str, default="laion400m_e32")

    parser.add_argument("--n_ctx", type=int, default=4)
    parser.add_argument("--n_ctx_ab", type=int, default=1)
    parser.add_argument("--n_pro", type=int, default=1)
    parser.add_argument("--n_pro_ab", type=int, default=4)
    parser.add_argument("--Epoch", type=int, default=100)

    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--lambda1", type=float, default=0.001)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu_id}"
    main(args)
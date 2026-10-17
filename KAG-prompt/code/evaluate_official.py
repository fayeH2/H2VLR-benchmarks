#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision import transforms
from tqdm import tqdm

# pytorchvideo 0.1.5 imports the pre-0.17 torchvision module name.
try:
    import torchvision.transforms.functional_tensor  # noqa: F401
except ModuleNotFoundError:
    import torchvision.transforms._functional_tensor as functional_tensor
    sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor

from metrics import cal_pro_score
from model.openllama import OpenLLAMAPEFTModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
from render_visual import read_rgb, save_triplet  # noqa: E402


def _render_visual(visual_root, dataset, data_root, class_name, k_shot,
                   image_path, anomaly_map, mask):
    class_test = Path(data_root) / class_name / "test"
    try:
        rel = Path(image_path).relative_to(class_test)
    except ValueError:
        rel = Path(os.path.basename(str(image_path)))
    stem = f"{class_name}__{os.path.splitext(str(rel))[0]}".replace(os.sep, "__")
    original = read_rgb(image_path)
    save_triplet(Path(visual_root) / dataset / f"k{k_shot}" / class_name,
                 stem, original, mask, anomaly_map)


CLASSES = {
    "mvtec": [
        "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
        "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
    ],
    "visa": [
        "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
        "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
    ],
}

DESCRIPTIONS = {
    "mvtec": {name: name.replace("_", " ") for name in CLASSES["mvtec"]},
    "visa": {
        "candle": "candle", "capsules": "capsule", "cashew": "cashew",
        "chewinggum": "chewinggom", "fryum": "fryum", "macaroni1": "macaroni",
        "macaroni2": "macaroni", "pcb1": "pcb", "pcb2": "pcb", "pcb3": "pcb",
        "pcb4": "pcb", "pipe_fryum": "pipe fryum",
    },
}

OFFICIAL_ROUNDS = {
    "mvtec": {1: 195, 2: 195, 4: 194},
    "visa": {1: 14, 2: 57, 4: 78},
}

PAPER_RESULTS = {
    "mvtec": {1: (95.8, 96.2), 2: (96.6, 96.5), 4: (97.1, 96.7)},
    "visa": {1: (91.6, 97.0), 2: (92.7, 97.4), 4: (93.3, 97.7)},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate official KAG-prompt checkpoints")
    parser.add_argument("--dataset", choices=("mvtec", "visa"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--k-shot", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--imagebind-checkpoint", type=Path, required=True)
    parser.add_argument("--method-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--class-name", choices=CLASSES["mvtec"] + CLASSES["visa"])
    parser.add_argument("--cal-pro", action="store_true")
    parser.add_argument("--visual-root", type=Path, default=None,
                        help="If set, dump query/gt-overlay/heatmap visualisations here.")
    return parser.parse_args()


def get_mvtec_support(root, class_name, shot):
    train_dir = root / class_name / "train" / "good"
    start = OFFICIAL_ROUNDS["mvtec"][shot]
    selected = []
    for index in range(start, start + shot):
        candidate = train_dir / f"{index:03d}.png"
        if not candidate.is_file():
            break
        selected.append(candidate)
    if not selected:
        selected = sorted(train_dir.glob("*.png"))[-shot:]
    return selected


def read_visa_split(root):
    rows = []
    with (root / "split_csv" / "1cls.csv").open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) >= 4 and row[0] in CLASSES["visa"]:
                rows.append(row)
    return rows


def get_visa_support(root, class_name, shot, rows):
    offset = OFFICIAL_ROUNDS["visa"][shot] * 4
    normal = [root / row[3] for row in rows if row[0] == class_name and row[1] == "train"]
    return normal[offset:offset + shot]


def iter_samples(dataset, root, class_name, visa_rows):
    if dataset == "mvtec":
        for image_path in sorted((root / class_name / "test").glob("*/*.png")):
            defect = image_path.parent.name
            mask = None if defect == "good" else (
                root / class_name / "ground_truth" / defect / f"{image_path.stem}_mask.png"
            )
            yield image_path, mask, defect == "good"
        return

    for row in visa_rows:
        if row[0] != class_name or row[1] != "test":
            continue
        image_path = root / row[3]
        is_normal = row[2] in ("good", "normal")
        mask = None
        if not is_normal:
            mask = root / class_name / "ground_truth" / "bad" / f"{image_path.stem}.png"
        yield image_path, mask, is_normal


def load_mask(dataset, image_path, mask_path):
    if mask_path is None:
        mask = Image.new("L", Image.open(image_path).size, 0)
    else:
        mask = Image.open(mask_path).convert("L")
    if dataset == "mvtec":
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()])
    tensor = transform(mask)
    threshold = 0.1 if dataset == "mvtec" else tensor.max() / 100
    return (tensor.squeeze() > threshold).numpy().astype(np.uint8)


def evaluate_class(model, args, class_name, support, visa_rows):
    pixel_predictions, pixel_labels = [], []
    image_predictions, image_labels = [], []
    samples = list(iter_samples(args.dataset, args.data_root, class_name, visa_rows))
    for image_path, mask_path, is_normal in tqdm(samples, desc=class_name, leave=False):
        with torch.no_grad():
            anomaly_map, global_score = model.generate({
                "prompt": DESCRIPTIONS[args.dataset][class_name],
                "image_paths": [str(image_path)],
                "normal_img_paths": [str(path) for path in support],
                "r": 0.1,
            })
        anomaly_map = anomaly_map.reshape(224, 224).detach().cpu().numpy()
        mask = load_mask(args.dataset, image_path, mask_path)
        if args.visual_root is not None:
            _render_visual(args.visual_root, args.dataset, args.data_root,
                           class_name, args.k_shot, image_path, anomaly_map, mask)
        top_values = np.partition(anomaly_map.ravel(), -30)[-30:]
        image_score = 0.1 * float(global_score.detach().cpu()) + 0.9 * float(top_values.mean())
        pixel_predictions.append(anomaly_map)
        pixel_labels.append(mask)
        image_predictions.append(image_score)
        image_labels.append(0 if is_normal else 1)

    pixel_predictions = np.asarray(pixel_predictions)
    pixel_labels = np.asarray(pixel_labels)
    image_predictions = np.asarray(image_predictions)
    image_labels = np.asarray(image_labels)
    result = {
        "class": class_name,
        "i_auroc": roc_auc_score(image_labels, image_predictions) * 100,
        "p_auroc": roc_auc_score(pixel_labels.ravel(), pixel_predictions.ravel()) * 100,
        "i_ap": average_precision_score(image_labels, image_predictions) * 100,
    }
    if args.cal_pro:
        result["p_pro"] = cal_pro_score(pixel_labels, pixel_predictions) * 100
    return result


def main():
    args = parse_args()
    visa_rows = read_visa_split(args.data_root) if args.dataset == "visa" else []
    model_args = {
        "model": "openllama_peft",
        "imagebind_ckpt_path": str(args.imagebind_checkpoint),
        "anomalygpt_ckpt_path": str(args.method_checkpoint),
        "stage": 2,
        "features_list": [6, 12, 18, 24],
    }
    model = OpenLLAMAPEFTModel(**model_args)
    checkpoint = torch.load(args.method_checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    model = model.to(args.device).eval()

    output_dir = args.output_root / args.dataset / f"k_{args.k_shot}"
    output_dir.mkdir(parents=True, exist_ok=True)
    classes = [args.class_name] if args.class_name else CLASSES[args.dataset]
    rows, support_rows = [], []
    for class_name in classes:
        if args.dataset == "mvtec":
            support = get_mvtec_support(args.data_root, class_name, args.k_shot)
        else:
            support = get_visa_support(args.data_root, class_name, args.k_shot, visa_rows)
        if len(support) != args.k_shot:
            raise RuntimeError(f"Expected {args.k_shot} support images for {class_name}, got {len(support)}")
        support_rows.extend(
            {"dataset": args.dataset, "shot": args.k_shot, "class": class_name, "image": str(path.resolve())}
            for path in support
        )
        result = evaluate_class(model, args, class_name, support, visa_rows)
        rows.append(result)
        print(json.dumps(result, ensure_ascii=True))

    fieldnames = ["class", "i_auroc", "p_auroc", "i_ap"] + (["p_pro"] if args.cal_pro else [])
    with (output_dir / "per_class.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "support_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "shot", "class", "image"])
        writer.writeheader()
        writer.writerows(support_rows)

    paper_i, paper_p = PAPER_RESULTS[args.dataset][args.k_shot]
    summary = {
        "dataset": args.dataset,
        "shot": args.k_shot,
        "classes": len(rows),
        "i_auroc": float(np.mean([row["i_auroc"] for row in rows])),
        "p_auroc": float(np.mean([row["p_auroc"] for row in rows])),
        "paper_i_auroc": paper_i,
        "paper_p_auroc": paper_p,
    }
    summary["i_delta"] = summary["i_auroc"] - paper_i
    summary["p_delta"] = summary["p_auroc"] - paper_p
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

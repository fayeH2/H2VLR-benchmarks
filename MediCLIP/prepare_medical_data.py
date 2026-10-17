"""Convert the H2VLR_compare medical datasets into the MediCLIP data layout.

Source (UniVAD / H2VLR_V2 shared pool):
    /path/to/datasets/medical/{BrainMRI,BUSI,LiverCT}/<cls>/

Target (MediCLIP expects `<root>/images/<filename>` + `<root>/samples/{train,test}.json`):
    MediCLIP/data/<name>/images/...      (symlinks to the source images)
    MediCLIP/data/<name>/samples/{train,test}.json

`test.json` lines carry {filename, clsname, label, mask}; `train.json` carries
{filename, clsname, label, mask}. Filenames are relative to `images/`.
The training list is written in sorted filename order so that the fixed seed
indices used across H2VLR_compare (datasets/seeds_*) map to the same support
images.
"""

import json
import os
import shutil
from pathlib import Path

SRC = Path("/path/to/datasets/medical")
OUT = Path(__file__).resolve().parent / "data"

SPECS = {
    "brainmri": {
        "src": ("BrainMRI", "brain"),
        "clsname": "brain",
        "normal": {"train": "train/good", "test": "test/good"},
        "anomaly": {"test": ["test/ungood"]},
        "masks": {"test": ["ground_truth/ungood"]},
    },
    "liverct": {
        "src": ("LiverCT", "liver"),
        "clsname": "liver",
        "normal": {"train": "train/good", "test": "test/good"},
        "anomaly": {"test": ["test/ungood"]},
        "masks": {"test": ["ground_truth/ungood"]},
    },
    "busi": {
        "src": ("BUSI", "breast"),
        "clsname": "breast",
        "normal": {"train": "train/good", "test": "test/good"},
        "anomaly": {"test": ["test/benign", "test/malignant"]},
        "masks": {"test": ["ground_truth/benign", "ground_truth/malignant"]},
    },
}

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def _list_images(directory):
    if not directory.is_dir():
        return []
    files = [p for p in directory.iterdir() if p.suffix.lower() in IMG_EXTS]
    return sorted(files, key=lambda p: p.name)


def _link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(src.resolve(), dst)


def prepare(name, spec):
    src_root = SRC / spec["src"][0] / spec["src"][1]
    dst_root = OUT / name
    if dst_root.exists():
        shutil.rmtree(dst_root)
    (dst_root / "images").mkdir(parents=True, exist_ok=True)
    (dst_root / "samples").mkdir(parents=True, exist_ok=True)

    clsname = spec["clsname"]
    train_lines = []
    test_lines = []

    # ---- train (normal only) -------------------------------------------------
    train_dir = src_root / spec["normal"]["train"]
    for img in _list_images(train_dir):
        rel = f"train/good/{img.name}"
        _link(img, dst_root / "images" / rel)
        train_lines.append({"filename": rel, "clsname": clsname, "label": 0, "mask": ""})
    train_lines.sort(key=lambda d: d["filename"])

    # ---- test (normal) -------------------------------------------------------
    test_norm_dir = src_root / spec["normal"]["test"]
    for img in _list_images(test_norm_dir):
        rel = f"test/good/{img.name}"
        _link(img, dst_root / "images" / rel)
        test_lines.append({"filename": rel, "clsname": clsname, "label": 0, "mask": ""})
    n_normal = len(test_lines)

    # ---- test (anomaly) + masks ---------------------------------------------
    mask_dirs = spec["masks"]["test"]
    for anom_dir in spec["anomaly"]["test"]:
        anom_path = src_root / anom_dir
        sub = anom_path.name
        for img in _list_images(anom_path):
            rel = f"test/{sub}/{img.name}"
            _link(img, dst_root / "images" / rel)
            mask_rel = ""
            for mdir in mask_dirs:
                cand = src_root / mdir / img.name
                if cand.exists():
                    mask_rel = f"test/mask/{mdir.split('/')[-1]}/{img.name}"
                    _link(cand, dst_root / "images" / mask_rel)
                    break
            test_lines.append(
                {"filename": rel, "clsname": clsname, "label": 1, "mask": mask_rel}
            )

    with (dst_root / "samples" / "train.json").open("w", encoding="utf-8") as fh:
        for line in train_lines:
            fh.write(json.dumps(line) + "\n")
    with (dst_root / "samples" / "test.json").open("w", encoding="utf-8") as fh:
        for line in test_lines:
            fh.write(json.dumps(line) + "\n")

    n_anom = len(test_lines) - n_normal
    print(
        f"[{name}] train={len(train_lines)}  test normal={n_normal} "
        f"anomaly={n_anom}  -> {dst_root}"
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, spec in SPECS.items():
        prepare(name, spec)


if __name__ == "__main__":
    main()

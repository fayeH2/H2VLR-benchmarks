import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_ROOT = SCRIPT_DIR / "outputs_medical"

SEED_FILES = {
    "brainmri": Path("/path/to/H2VLR/datasets/seeds_BrainAD/brain/selected_samples_per_run.txt"),
    "liverct": Path("/path/to/H2VLR/datasets/seeds_liverAD/liver/selected_samples_per_run.txt"),
    "busi": Path("/path/to/H2VLR/datasets/seeds_busi/breast/selected_samples_per_run.txt"),
}
CONFIGS = {
    "brainmri": SCRIPT_DIR / "config" / "brainmri.yaml",
    "liverct": SCRIPT_DIR / "config" / "brainmri.yaml",
    "busi": SCRIPT_DIR / "config" / "brainmri.yaml",
}
CLSNAMES = {"brainmri": "brain", "liverct": "liver", "busi": "breast"}
DATASET_ORDER = ["busi", "liverct", "brainmri"] 

# Paper (arXiv 2405.11315) reference numbers for the available datasets.
PAPER = {
    "brainmri": {"i_auroc": 94.1},
    "busi": {"i_auroc": 88.5, "p_auroc": 88.0},
}


def parse_seeds(path):
    """Parse `#k: a b c` lines into {k: [int, ...]} (busi ids are digit strings)."""
    seeds = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"#(\d+):\s*(.*)", line)
        if not m:
            continue
        k = int(m.group(1))
        seeds[k] = [int(tok) for tok in m.group(2).split()]
    return seeds


def run_one(dataset, shot, indices, args):
    cmd = [
        sys.executable, str(SCRIPT_DIR / "run_medical.py"),
        "--dataset", dataset,
        "--config_path", str(CONFIGS[dataset]),
        "--k_shot", str(shot),
        "--support_indices", " ".join(str(i) for i in indices),
        "--epoch", str(args.epoch),
        "--val_freq", str(args.val_freq),
        "--gpu", str(args.gpu),
        "--output-root", str(OUT_ROOT),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(SCRIPT_DIR), check=True)


def aggregate():
    rows = []
    for dataset in DATASET_ORDER:
        for shot in (1, 2, 4):
            rj = OUT_ROOT / dataset / f"k{shot}" / CLSNAMES[dataset] / "result.json"
            if rj.exists():
                rows.append(json.loads(rj.read_text()))

    if not rows:
        print("no results found")
        return

    csv_path = SCRIPT_DIR / "medical_results_summary.csv"
    fields = ["method", "dataset", "shot", "class", "i_auroc", "i_aupr",
              "p_auroc", "p_aupr", "p_aupro", "best_epoch"]
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    json_path = SCRIPT_DIR / "medical_results_summary.json"
    json_path.write_text(json.dumps(rows, indent=2))

    print("\n==================== SUMMARY ====================")
    header = f"{'dataset':<10}{'shot':>5}{'I-AUROC':>10}{'I-AUPR':>10}{'P-AUROC':>10}{'P-AUPRO':>10}"
    print(header)
    for r in rows:
        print(f"{r['dataset']:<10}{r['shot']:>5}"
              f"{r['i_auroc']:>10.4f}{r['i_aupr']:>10.4f}"
              f"{r['p_auroc']:>10.4f}{r['p_aupro']:>10.4f}")
    print(f"\nwrote {csv_path}\nwrote {json_path}")

    print("\n==== paper reference (k=4) ====")
    for ds, ref in PAPER.items():
        k4 = next((r for r in rows if r["dataset"] == ds and r["shot"] == 4), None)
        if not k4:
            continue
        parts = []
        if "i_auroc" in ref:
            parts.append(f"I-AUROC paper {ref['i_auroc']:.1f} vs ours {k4['i_auroc']*100:.1f}")
        if "p_auroc" in ref:
            parts.append(f"P-AUROC paper {ref['p_auroc']:.1f} vs ours {k4['p_auroc']*100:.1f}")
        print(f"  {ds}: " + "; ".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epoch", type=int, default=200)
    ap.add_argument("--val_freq", type=int, default=25)
    ap.add_argument("--datasets", default=",".join(DATASET_ORDER))
    ap.add_argument("--shots", default="1,2,4")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--aggregate_only", action="store_true")
    args = ap.parse_args()

    if args.aggregate_only:
        aggregate()
        return

    want_ds = [d for d in args.datasets.split(",") if d]
    want_shots = [int(s) for s in args.shots.split(",") if s]

    for dataset in want_ds:
        seeds = parse_seeds(SEED_FILES[dataset])
        for shot in want_shots:
            run_one(dataset, shot, seeds[shot], args)
    aggregate()


if __name__ == "__main__":
    main()

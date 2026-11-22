#!/usr/bin/env python3
"""
Leave-one-site-out evaluation (reviewers 7 and 8.3) on the three site groups in loso/site_map.csv.

For each held-out group the test set is every frame from that group. The remaining videos are
split into train and validation by whole video (the WhatsApp clips stay together), choosing,
from 3,000 random assignments, the one whose per-class validation share is closest to 15%,
as in the paper. Datasets are symlinks into TV_R-final-filtered (the masked labels the paper
reports). Each model is trained with the paper's settings (seed 0), selected on validation,
and scored once on the held-out site.

Run with ~/fydp_env (Ultralytics 8.4.62), the environment the paper used.
  ~/fydp_env/bin/python loso/run_loso.py folds     # build loso/data/fold_*, write loso/folds.md
  ~/fydp_env/bin/python loso/run_loso.py all       # build if needed, train and score everything
  ~/fydp_env/bin/python loso/run_loso.py status
Safe to rerun: finished runs are skipped, an interrupted one resumes.
"""
import csv
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/georgedessa/FYDP")
SRC = ROOT / "TV_R-final-filtered"
OUT = ROOT / "loso"
DATA = OUT / "data"
RUNS = ROOT / "runs"
NAMES = ["3_person_violation", "footpath_violation", "helmet_violation", "license_plate", "wrong_side_violation"]
VIDEO_RE = re.compile(r"^(.*?)_(?:MOV|mp4)-\d+")
CONFIGS = {
    "s_aug": dict(model="yolo26s.pt", batch=8, hsv_v=0.5, scale=0.7, mixup=0.1),
    "n_aug": dict(model="yolo26n.pt", batch=16, hsv_v=0.5, scale=0.7, mixup=0.1),
}
COMMON = dict(epochs=150, patience=30, imgsz=1280, cache=True, workers=16, optimizer="auto",
              deterministic=True, amp=True, seed=0)
VAL_SHARE = 0.15


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(OUT / "loso.log", "a") as f:
        f.write(line + "\n")


def load_frames():
    site = {r["video"]: r["site_group"] for r in csv.DictReader(open(OUT / "site_map.csv"))}
    frames = []  # (image path, video, split group, site, class counts)
    for img in sorted(SRC.glob("*/images/*.jpg")):
        v = VIDEO_RE.match(img.name).group(1)
        lab = img.parent.parent / "labels" / (img.stem + ".txt")
        c = Counter(int(l.split()[0]) for l in open(lab) if l.strip()) if lab.exists() else Counter()
        group = "whatsapp" if v.startswith("WhatsApp") else v
        frames.append((img, v, group, site[v], c))
    return frames


def build_folds():
    frames = load_frames()
    sites = sorted({f[3] for f in frames})
    summary = []
    rng = np.random.default_rng(0)
    for held in sites:
        test = [f for f in frames if f[3] == held]
        rest = [f for f in frames if f[3] != held]
        groups = sorted({f[2] for f in rest})
        gcount = {g: np.zeros(len(NAMES)) for g in groups}
        for f in rest:
            for k, n in f[4].items():
                gcount[f[2]][k] += n
        total = sum(gcount.values())
        present = total > 0
        best, best_err = None, 1e9
        for _ in range(3000):
            pick = [g for g in groups if rng.random() < VAL_SHARE]
            if not pick or len(pick) == len(groups):
                continue
            share = sum(gcount[g] for g in pick)[present] / total[present]
            err = float(np.abs(share - VAL_SHARE).sum())
            if err < best_err:
                best, best_err = set(pick), err
        fold = DATA / f"fold_{held}"
        split_of = lambda f: "test" if f[3] == held else ("valid" if f[2] in best else "train")
        for sp in ("train", "valid", "test"):
            for sub in ("images", "labels"):
                (fold / sp / sub).mkdir(parents=True, exist_ok=True)
        counts = defaultdict(lambda: np.zeros(len(NAMES), int))
        nimg, vids = Counter(), defaultdict(set)
        for f in frames:
            sp = split_of(f)
            img, lab = f[0], f[0].parent.parent / "labels" / (f[0].stem + ".txt")
            for src, dst in ((img, fold / sp / "images" / img.name), (lab, fold / sp / "labels" / lab.name)):
                if src.exists() and not dst.exists():
                    dst.symlink_to(src)
            nimg[sp] += 1
            vids[sp].add(f[1])
            for k, n in f[4].items():
                counts[sp][k] += n
        (fold / "data.yaml").write_text(
            f"path: {fold}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
            f"nc: {len(NAMES)}\nnames:\n" + "".join(f"- {n}\n" for n in NAMES))
        for sp in ("train", "valid", "test"):
            summary.append(dict(fold=held, split=sp, videos=len(vids[sp]), images=nimg[sp],
                                **{NAMES[k]: int(counts[sp][k]) for k in range(len(NAMES))},
                                video_list=" ".join(sorted(vids[sp])) if sp != "train" else ""))
    short = ["3P", "FP", "HV", "LP", "WS"]
    md = ["| Held-out site | Split | Videos | Images | " + " | ".join(short) + " | Videos (val/test) |",
          "|---|---|---|---|" + "---|" * len(short) + "---|"]
    for r in summary:
        md.append(f"| {r['fold']} | {r['split']} | {r['videos']} | {r['images']} | "
                  + " | ".join(str(r[n]) for n in NAMES) + f" | {r['video_list']} |")
    (OUT / "folds.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def run_name(cfg, held):
    return f"loso_{held}_{cfg}"


def train_one(cfg, held):
    from ultralytics import YOLO
    rd = RUNS / run_name(cfg, held)
    data = DATA / f"fold_{held}" / "data.yaml"
    last = rd / "weights" / "last.pt"
    if last.exists() and not (rd / "TRAINED").exists():
        log(f"{rd.name}: resuming from last.pt")
        try:
            YOLO(str(last)).train(resume=True)
        except Exception as e:
            if "nothing to resume" not in str(e):
                raise
    elif not (rd / "TRAINED").exists():
        c = dict(CONFIGS[cfg])
        model = str(ROOT / c.pop("model"))
        log(f"{rd.name}: training {Path(model).name} {c}")
        YOLO(model).train(data=str(data), project=str(RUNS), name=rd.name, exist_ok=True, **COMMON, **c)
    (rd / "TRAINED").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))

    best = YOLO(str(rd / "weights" / "best.pt"))
    row = {"held_out_site": held, "config": cfg}
    for split in ("val", "test"):
        m = best.val(data=str(data), split=split, imgsz=1280, batch=16, plots=False,
                     project=str(OUT / "eval"), name=f"{rd.name}_{split}", exist_ok=True, verbose=False)
        row[f"{split}_mAP50"] = round(float(m.box.map50), 4)
        row[f"{split}_mAP50_95"] = round(float(m.box.map), 4)
        row[f"{split}_P"] = round(float(m.box.mp), 4)
        row[f"{split}_R"] = round(float(m.box.mr), 4)
        for i, c in enumerate(m.box.ap_class_index):
            row[f"{split}_AP50_{NAMES[int(c)]}"] = round(float(m.box.ap50[i]), 4)
    (rd / "metrics.json").write_text(json.dumps(row, indent=1))
    log(f"{rd.name}: val mAP50 {row['val_mAP50']}  held-out-site mAP50 {row['test_mAP50']}  "
        f"mAP50-95 {row['test_mAP50_95']}")
    (rd / "LOSO_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))


def collect():
    rows = [json.loads(p.read_text()) for p in sorted(RUNS.glob("loso_*/metrics.json"))]
    if not rows:
        return
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("held_out_site", "config"), k))
    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    sites = sorted({r["site_group"] for r in csv.DictReader(open(OUT / "site_map.csv"))})
    if cmd == "folds":
        build_folds()
    elif cmd == "one":
        train_one(sys.argv[2], sys.argv[3])
    elif cmd == "status":
        for cfg in CONFIGS:
            for s in sites:
                done = (RUNS / run_name(cfg, s) / "LOSO_DONE").exists()
                print(f"{cfg:6s} hold out {s}: {'done' if done else 'pending'}")
    elif cmd == "all":
        if not (DATA / f"fold_{sites[-1]}" / "data.yaml").exists():
            build_folds()
        log("starting LOSO runs")
        for cfg in CONFIGS:
            for s in sites:
                if (RUNS / run_name(cfg, s) / "LOSO_DONE").exists():
                    continue
                r = subprocess.run([sys.executable, __file__, "one", cfg, s])
                if r.returncode != 0:
                    log(f"{run_name(cfg, s)}: FAILED (exit {r.returncode}); continuing")
                collect()
        collect()
        log("all LOSO runs finished")


if __name__ == "__main__":
    main()

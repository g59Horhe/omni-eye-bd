#!/usr/bin/env python3
"""
Cross-audit setup for missing plate labels.

Splits videos by plate density (plates per image):
  >= --good : well-labelled  -> TV_R-auditor/train + valid  (train the auditor on these)
  <  --good : sparse         -> TV_R-auditor/audit/train    (audit these; auditor never saw them)
Uses symlinks, so no images are copied and the source is untouched.

Usage:
  python make_plate_auditor_data.py --src ~/FYDP/TV_R-raw-fixed --dst ~/FYDP/TV_R-auditor --good 0.3
Then:
  yolo detect train model=$HOME/FYDP/yolo26n.pt data=$HOME/FYDP/TV_R-auditor/data.yaml \
      imgsz=1280 epochs=80 patience=20 batch=16 project=$HOME/FYDP/runs name=plate_auditor
  python audit_missing_plates.py --data ~/FYDP/TV_R-auditor/audit \
      --weights ~/FYDP/runs/plate_auditor/weights/best.pt --out ~/FYDP/plate_audit_cross --conf 0.25
"""
import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NAME_RE = re.compile(r"^(.+?)_([A-Za-z0-9]+-\d+)$")


def video_of(p):
    b = re.sub(r"_(jpg|jpeg|png)\.rf\..*$", "", p.name)
    m = NAME_RE.match(b)
    return m.group(1) if m else "?"


def link(img, dst_split):
    (dst_split / "images").mkdir(parents=True, exist_ok=True)
    (dst_split / "labels").mkdir(parents=True, exist_ok=True)
    (dst_split / "images" / img.name).symlink_to(img.resolve())
    lbl = img.parent.parent / "labels" / (img.stem + ".txt")
    if lbl.exists():
        (dst_split / "labels" / lbl.name).symlink_to(lbl.resolve())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--good", type=float, default=0.3, help="min plates/image for a well-labelled video")
    ap.add_argument("--cls", type=int, default=3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    src, dst = Path(a.src).expanduser(), Path(a.dst).expanduser()
    if dst.exists():
        raise SystemExit(f"{dst} exists; delete it first")
    data = yaml.safe_load(open(src / "data.yaml"))

    by_vid = defaultdict(list)
    plates = defaultdict(int)
    for s in ("train", "valid", "test"):
        d = src / s / "images"
        if not d.exists():
            continue
        for p in d.iterdir():
            if p.suffix.lower() not in IMG_EXTS:
                continue
            v = video_of(p)
            by_vid[v].append(p)
            lbl = p.parent.parent / "labels" / (p.stem + ".txt")
            if lbl.exists():
                plates[v] += sum(1 for ln in lbl.read_text().splitlines()
                                 if ln.split() and int(float(ln.split()[0])) == a.cls)

    good = sorted(v for v in by_vid if plates[v] / len(by_vid[v]) >= a.good)
    sparse = sorted(v for v in by_vid if v not in good)

    rng = random.Random(a.seed)
    good_imgs = [p for v in good for p in by_vid[v]]
    rng.shuffle(good_imgs)
    n_val = int(len(good_imgs) * a.val_frac)
    for i, p in enumerate(good_imgs):
        link(p, dst / ("valid" if i < n_val else "train"))
    for v in sparse:
        for p in by_vid[v]:
            link(p, dst / "audit" / "train")

    new = dict(data)
    new.update(path=str(dst), train="train/images", val="valid/images")
    new.pop("test", None)
    yaml.safe_dump(new, open(dst / "data.yaml", "w"), sort_keys=False)

    print(f"well-labelled videos ({len(good)}, {len(good_imgs)} imgs -> auditor train/valid):")
    print("   " + ", ".join(f"{v[:24]}[{plates[v]/len(by_vid[v]):.2f}]" for v in good))
    print(f"sparse videos to audit ({len(sparse)}, {sum(len(by_vid[v]) for v in sparse)} imgs):")
    print("   " + ", ".join(f"{v[:24]}[{plates[v]/len(by_vid[v]):.2f}]" for v in sparse))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()

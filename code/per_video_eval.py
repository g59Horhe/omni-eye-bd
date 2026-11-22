#!/usr/bin/env python3
"""
Per-video evaluation: mAP50 per class for every video in a split.
Shows whether poor results come from specific videos (night, portrait,
labelling differences) or are spread evenly.

Usage:
  python per_video_eval.py --data ~/FYDP/TV_R-4-filtered \
      --weights ~/FYDP/runs/tv_r_v5_clean/weights/best.pt --splits valid train
Never pass 'test' here while you are still making decisions.
"""
import argparse
import csv
import re
import shutil
from collections import defaultdict
from pathlib import Path

import yaml
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NAME_RE = re.compile(r"^(.+?)_([A-Za-z0-9]+-\d+)$")


def video_of(p):
    base = re.sub(r"_(jpg|jpeg|png)\.rf\..*$", "", p.name)
    m = NAME_RE.match(base)
    return m.group(1) if m else "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--splits", nargs="+", default=["valid"])
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--tmp", default="/tmp/per_video_eval")
    ap.add_argument("--out", default="per_video_results.csv")
    a = ap.parse_args()
    if "test" in a.splits:
        print("WARNING: evaluating on test. Only do this once, for the final model.")

    root = Path(a.data).expanduser()
    names = yaml.safe_load(open(root / "data.yaml"))["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    short = [n.replace("_violation", "")[:10] for n in names]
    model = YOLO(a.weights)
    tmp = Path(a.tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    rows = []

    for split in a.splits:
        by_vid = defaultdict(list)
        for p in (root / split / "images").iterdir():
            if p.suffix.lower() in IMG_EXTS:
                by_vid[video_of(p)].append(p)
        print(f"\n=== {split}: {len(by_vid)} videos  (cells: mAP50 (instances), '-' = class absent)")
        print(f"{'video':>42} {'imgs':>5} | " + " | ".join(f"{s:>12}" for s in short))
        for v in sorted(by_vid):
            d = tmp / split / re.sub(r"[^A-Za-z0-9_-]", "_", v)
            (d / "images").mkdir(parents=True)
            (d / "labels").mkdir(parents=True)
            counts = [0] * len(names)
            for p in by_vid[v]:
                (d / "images" / p.name).symlink_to(p.resolve())
                lbl = root / split / "labels" / (p.stem + ".txt")
                if lbl.exists():
                    (d / "labels" / lbl.name).symlink_to(lbl.resolve())
                    for ln in lbl.read_text().splitlines():
                        if ln.split():
                            counts[int(float(ln.split()[0]))] += 1
            yml = d / "data.yaml"
            yaml.safe_dump({"path": str(d), "train": "images", "val": "images", "names": names}, open(yml, "w"))
            res = model.val(data=str(yml), imgsz=a.imgsz, batch=16, plots=False, verbose=False)
            ap50 = dict(zip([int(i) for i in res.box.ap_class_index], res.box.ap50))
            cells = [f"{ap50.get(c, 0):.2f} ({counts[c]:>3})" if counts[c] else f"{'-':>12}" for c in range(len(names))]
            print(f"{v[:42]:>42} {len(by_vid[v]):5d} | " + " | ".join(f"{c:>12}" for c in cells), flush=True)
            rows.append([split, v, len(by_vid[v])] + [f"{ap50.get(c, 0):.3f}" if counts[c] else "" for c in range(len(names))]
                        + counts)

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "video", "images"] + [f"mAP50_{n}" for n in names] + [f"n_{n}" for n in names])
        w.writerows(rows)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()

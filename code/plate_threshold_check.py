#!/usr/bin/env python3
"""
Pick a small-plate threshold from evidence, not just a histogram.

Outputs:
  1. Image resolution summary (so fractions -> pixels is correct)
  2. Plate width/height percentiles in px (original and @1280)
  3. % of plates removed for several candidate thresholds
  4. crops/width_XX-YY.jpg : grids of real plate crops per width bucket
     -> open these and find where plates stop being readable.

Usage:
  python plate_threshold_check.py --data ~/FYDP/TV_R-3 --split train --cls 3
"""
import argparse
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--cls", type=int, default=3)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--out", default="plate_check")
    ap.add_argument("--per-bucket", type=int, default=40)
    a = ap.parse_args()

    root = Path(a.data).expanduser() / a.split
    out = Path(a.out).expanduser()
    (out / "crops").mkdir(parents=True, exist_ok=True)

    sizes = Counter()
    plates = []  # (img_path, W, H, x, y, w, h)
    for img_path in (root / "images").iterdir():
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        lbl = root / "labels" / (img_path.stem + ".txt")
        if not lbl.exists():
            continue
        rows = [l.split() for l in lbl.read_text().splitlines()]
        rows = [r for r in rows if len(r) >= 5 and int(float(r[0])) == a.cls]
        with Image.open(img_path) as im:
            W, H = im.size
        sizes[(W, H)] += 1
        for r in rows:
            x, y, w, h = map(float, r[1:5])
            plates.append((img_path, W, H, x, y, w, h))

    print("== Image resolutions (top 5) ==")
    for (W, H), n in sizes.most_common(5):
        print(f"  {W}x{H}: {n} images")
    print(f"\nTotal plates: {len(plates)}\n")

    wf = np.array([p[5] for p in plates])
    wo = np.array([p[5] * p[1] for p in plates])                      # px, original
    ho = np.array([p[6] * p[2] for p in plates])
    s = np.array([a.imgsz / max(p[1], p[2]) for p in plates])
    wt, ht = wo * s, ho * s                                            # px @ imgsz

    print("== Percentiles ==")
    print(f"{'pct':>5} | {'w_frac':>7} | {'w_orig':>7} {'h_orig':>7} | {'w@'+str(a.imgsz):>7} {'h@'+str(a.imgsz):>7}")
    for q in (5, 10, 15, 20, 25, 30, 50, 75, 90):
        print(f"{q:>4}% | {np.percentile(wf, q):7.4f} | {np.percentile(wo, q):7.1f} {np.percentile(ho, q):7.1f}"
              f" | {np.percentile(wt, q):7.1f} {np.percentile(ht, q):7.1f}")

    print(f"\n== % plates removed (px @ {a.imgsz}) ==")
    for mw in (12, 16, 20, 24, 28, 32):
        only_w = (wt < mw).mean() * 100
        for mh in (6, 8, 10):
            either = ((wt < mw) | (ht < mh)).mean() * 100
            print(f"  w<{mw:>2}: {only_w:5.1f}%   |   w<{mw:>2} OR h<{mh:>2}: {either:5.1f}%")
        print()

    # crop grids by width bucket (px @ imgsz), upscaled so you can judge readability
    edges = [0, 12, 16, 20, 24, 28, 32, 40, 60, 10_000]
    random.seed(0)
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = [i for i in range(len(plates)) if lo <= wt[i] < hi]
        if not idx:
            continue
        pick = random.sample(idx, min(a.per_bucket, len(idx)))
        tiles = []
        for i in pick:
            ip, W, H, x, y, w, h = plates[i]
            with Image.open(ip) as im:
                c = im.convert("RGB").crop((int((x - w / 2) * W), int((y - h / 2) * H),
                                            int((x + w / 2) * W) + 1, int((y + h / 2) * H) + 1))
            tiles.append(c.resize((160, 80), Image.NEAREST))  # nearest = honest pixelation
        cols = 8
        rows = (len(tiles) + cols - 1) // cols
        grid = Image.new("RGB", (cols * 164, rows * 84), (40, 40, 40))
        for k, t in enumerate(tiles):
            grid.paste(t, ((k % cols) * 164 + 2, (k // cols) * 84 + 2))
        name = f"width_{lo:03d}-{hi if hi < 10_000 else 'max'}px.jpg"
        grid.save(out / "crops" / name, quality=92)
        print(f"saved crops/{name}  ({len(idx)} plates in bucket)")


if __name__ == "__main__":
    main()

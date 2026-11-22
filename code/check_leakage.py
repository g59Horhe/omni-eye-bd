#!/usr/bin/env python3
"""
Check whether valid/test images come from the same videos as train (data leakage).

Part 1 - filename groups: for names like "10_MOV-0002_jpg.rf.<hash>.jpg" it tries
         BOTH interpretations and reports overlap between splits:
           A) group = leading number   ("10")        -> 10 = video, 0002 = frame
           B) group = MOV id           ("MOV-0002")  -> MOV-0002 = clip, 10 = frame
Part 2 - pixel content: perceptual hash (dHash, 256 bit, flip-aware) of every image;
         for each valid/test image, finds the most similar train image.
         Saves side-by-side pairs of the closest matches so you can SEE if they
         are the same scene.

Usage:
  python check_leakage.py --data ~/FYDP/TV_R-3 --out ~/FYDP/leak_check
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NAME_RE = re.compile(r"^(\d+)_([A-Za-z]+-\d+)")
HASH = 16  # 16x16 = 256-bit dHash


def list_images(d):
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS) if d.exists() else []


def base_name(p):
    return re.sub(r"_(jpg|jpeg|png)\.rf\..*$", "", p.name)


def dhash(img):
    g = np.asarray(img.convert("L").resize((HASH + 1, HASH), Image.BILINEAR), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


def hash_file(p):
    with Image.open(p) as im:
        return dhash(im), dhash(im.transpose(Image.FLIP_LEFT_RIGHT))


def name_report(splits):
    print("=" * 70 + "\nPART 1: filename-based video grouping\n" + "=" * 70)
    groups = {s: {"A": defaultdict(int), "B": defaultdict(int), "unparsed": 0} for s in splits}
    for s, imgs in splits.items():
        for p in imgs:
            m = NAME_RE.match(base_name(p))
            if not m:
                groups[s]["unparsed"] += 1
                continue
            groups[s]["A"][m.group(1)] += 1
            groups[s]["B"][m.group(2)] += 1
    for s in splits:
        print(f"{s:>6}: {len(splits[s])} images | unique bases {len({base_name(p) for p in splits[s]})}"
              f" | unparsed names {groups[s]['unparsed']}")
    for key, label in (("A", "leading number (e.g. '10')"), ("B", "MOV id (e.g. 'MOV-0002')")):
        print(f"\n-- Interpretation {key}: group = {label}")
        tr = set(groups["train"][key])
        print(f"   train has {len(tr)} groups")
        for s in splits:
            if s == "train":
                continue
            g = set(groups[s][key])
            shared = g & tr
            imgs_shared = sum(groups[s][key][k] for k in shared)
            print(f"   {s}: {len(g)} groups, {len(shared)} also in train "
                  f"-> {imgs_shared}/{len(splits[s])} {s} images ({100*imgs_shared/max(1,len(splits[s])):.0f}%) share a group with train")
            if shared:
                print(f"      examples: {sorted(shared)[:10]}")


def content_report(splits, out, n_pairs):
    print("\n" + "=" * 70 + "\nPART 2: pixel-content near-duplicate search (dHash)\n" + "=" * 70)
    print("hashing train (takes a minute)...")
    tr_imgs = splits["train"]
    tr_h = np.array([hash_file(p)[0] for p in tr_imgs])  # (N, 256) bool
    out.mkdir(parents=True, exist_ok=True)
    for s in splits:
        if s == "train":
            continue
        dists, nn = [], []
        for p in splits[s]:
            h, hf = hash_file(p)
            d = np.minimum((tr_h != h).sum(1), (tr_h != hf).sum(1))
            i = int(d.argmin())
            dists.append(int(d[i]))
            nn.append(i)
        dists = np.array(dists)
        print(f"\n{s}: distance to closest train image (0 = identical, ~128 = unrelated, out of 256 bits)")
        for t in (10, 20, 30, 40, 60):
            print(f"   <= {t:>2}: {(dists <= t).sum():4d} / {len(dists)} ({100*(dists<=t).mean():5.1f}%)")
        print(f"   median: {np.median(dists):.0f}")

        order = np.argsort(dists)
        pair_dir = out / f"{s}_closest_pairs"
        pair_dir.mkdir(exist_ok=True)
        picks = list(order[:n_pairs]) + list(order[len(order)//2: len(order)//2 + 4])  # closest + a few median
        for rank, k in enumerate(picks):
            a = Image.open(splits[s][k]).convert("RGB")
            b = Image.open(tr_imgs[nn[k]]).convert("RGB")
            a.thumbnail((640, 640)); b.thumbnail((640, 640))
            canvas = Image.new("RGB", (a.width + b.width + 10, max(a.height, b.height)), (0, 0, 0))
            canvas.paste(a, (0, 0)); canvas.paste(b, (a.width + 10, 0))
            tag = "median" if rank >= n_pairs else f"{rank:02d}"
            canvas.save(pair_dir / f"{tag}_d{dists[k]:03d}__{base_name(splits[s][k])}__vs__{base_name(tr_imgs[nn[k]])}.jpg",
                        quality=85)
        print(f"   saved {len(picks)} side-by-side pairs to {pair_dir}  (left = {s}, right = train)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="leak_check")
    ap.add_argument("--pairs", type=int, default=20)
    a = ap.parse_args()
    root = Path(a.data).expanduser()
    splits = {s: list_images(root / s / "images") for s in ("train", "valid", "test")}
    splits = {s: v for s, v in splits.items() if v}
    name_report(splits)
    content_report(splits, Path(a.out).expanduser(), a.pairs)


if __name__ == "__main__":
    main()

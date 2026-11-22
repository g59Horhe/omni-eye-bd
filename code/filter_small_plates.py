#!/usr/bin/env python3
"""
Filter small license-plate annotations from a YOLO dataset (Omni-Eye / TV_R v3).

Creates a NEW dataset copy; the source is never modified.
Small plates are either:
  --mode mask  (recommended) : label removed AND the plate region is grey-filled
                                in the image, so the model isn't taught that a
                                visible plate is "background".
  --mode drop                : label removed only, image copied unchanged.

Sizes are measured in pixels at TRAINING scale (long side resized to --imgsz),
which is what the model actually sees.

Examples:
  # 1) Dry run: just report how many plates each threshold would remove
  python filter_small_plates.py --src ~/FYDP/TV_R-3 --dst /tmp/x --min-w 24 --min-h 8 --dry-run

  # 2) Build the filtered dataset
  python filter_small_plates.py --src ~/FYDP/TV_R-3 --dst ~/FYDP/TV_R-3-filtered \
      --min-w 24 --min-h 8 --mode mask
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FILL = 114  # same grey Ultralytics uses for letterbox padding


def find_plate_class(data):
    names = data["names"]
    items = names.items() if isinstance(names, dict) else enumerate(names)
    for i, n in items:
        if "plate" in str(n).lower():
            return int(i)
    raise SystemExit("No class containing 'plate' in data.yaml; pass --plate-cls")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--min-w", type=float, default=24, help="min plate width (px @ imgsz)")
    ap.add_argument("--min-h", type=float, default=8, help="min plate height (px @ imgsz)")
    ap.add_argument("--mode", choices=["mask", "drop"], default="mask")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--plate-cls", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    src = Path(a.src).expanduser()
    dst = Path(a.dst).expanduser()
    data = yaml.safe_load(open(src / "data.yaml"))
    pc = a.plate_cls if a.plate_cls is not None else find_plate_class(data)
    print(f"plate class = {pc} | threshold = w<{a.min_w} or h<{a.min_h} px @ {a.imgsz} "
          f"| mode = {a.mode} | dry_run = {a.dry_run}\n")

    done_splits = []
    for split in a.splits:
        img_dir, lbl_dir = src / split / "images", src / split / "labels"
        if not img_dir.exists():
            print(f"[skip] {split}: no images/ folder")
            continue
        out_img, out_lbl = dst / split / "images", dst / split / "labels"
        if not a.dry_run:
            out_img.mkdir(parents=True, exist_ok=True)
            out_lbl.mkdir(parents=True, exist_ok=True)

        kept = removed = imgs_masked = imgs_no_plates_left = 0
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            lines = lbl_path.read_text().splitlines() if lbl_path.exists() else []
            with Image.open(img_path) as im:
                W, H = im.size
            s = a.imgsz / max(W, H)

            keep, small = [], []
            had_plate = False
            for ln in lines:
                p = ln.split()
                if len(p) < 5:
                    continue
                c = int(float(p[0]))
                x, y, w, h = map(float, p[1:5])
                if c == pc:
                    had_plate = True
                    if w * W * s < a.min_w or h * H * s < a.min_h:
                        small.append((x, y, w, h))
                        continue
                    kept += 1
                keep.append(ln)
            removed += len(small)
            if had_plate and small and not any(int(float(l.split()[0])) == pc for l in keep):
                imgs_no_plates_left += 1

            if a.dry_run:
                continue

            (out_lbl / lbl_path.name).write_text("\n".join(keep) + ("\n" if keep else ""))
            dst_img = out_img / img_path.name
            if small and a.mode == "mask":
                imgs_masked += 1
                arr = np.array(Image.open(img_path).convert("RGB"))
                for x, y, w, h in small:
                    x1 = max(0, int((x - w / 2) * W) - 1)
                    y1 = max(0, int((y - h / 2) * H) - 1)
                    x2 = min(W, int((x + w / 2) * W) + 2)
                    y2 = min(H, int((y + h / 2) * H) + 2)
                    arr[y1:y2, x1:x2] = FILL
                Image.fromarray(arr).save(dst_img, quality=95)
            else:
                shutil.copy2(img_path, dst_img)

        total = kept + removed
        pct = 100 * removed / total if total else 0
        print(f"{split:>6}: plates kept {kept:6d} | removed {removed:6d} ({pct:5.1f}%) "
              f"| images masked {imgs_masked} | images left with no plate {imgs_no_plates_left}")
        done_splits.append(split)

    if a.dry_run:
        print("\nDry run only, nothing written.")
        return

    new = dict(data)
    new["path"] = str(dst)
    for key, split in (("train", "train"), ("val", "valid"), ("test", "test")):
        if split in done_splits:
            new[key] = f"{split}/images"
        else:
            new.pop(key, None)
    with open(dst / "data.yaml", "w") as f:
        yaml.safe_dump(new, f, sort_keys=False)
    print(f"\nWrote {dst / 'data.yaml'}")


if __name__ == "__main__":
    main()

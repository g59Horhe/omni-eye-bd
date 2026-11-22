#!/usr/bin/env python3
"""
Find likely MISSING license-plate labels (Omni-Eye).

Runs a trained model on every source image, and flags plate predictions that
do not match any labelled plate. Output is sorted by confidence so the most
likely misses come first.

Outputs (in --out):
  candidates.csv   one row per flagged plate: idx, file, video, conf, box, width
  sheets/*.jpg     contact sheets, 48 candidates each, numbered by idx
                   red   = model's plate prediction with no label  (check this)
                   green = plates that ARE labelled nearby          (context)
  per-video counts printed at the end

Usage (run on the NO-augmentation Roboflow export):
  python audit_missing_plates.py --data ~/FYDP/TV_R-raw \
      --weights ~/FYDP/runs/tv_r_v3-3/weights/best.pt --out ~/FYDP/plate_audit

Note: the model trained on these images with the missing labels, so it has
partly learned to ignore exactly those plates. It will catch many misses, not
all. A lower --conf finds more, with more false alarms.
"""
import argparse
import csv
import re
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NAME_RE = re.compile(r"^(.+?)_([A-Za-z0-9]+-\d+)$")
TILE_W, TILE_H = 240, 180


def base_name(p):
    return re.sub(r"_(jpg|jpeg|png)\.rf\..*$", "", p.name)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def center_inside(a, b):
    cx, cy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]


def load_gt(img_path, W, H, cls):
    lbl = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
    boxes = []
    if lbl.exists():
        for ln in lbl.read_text().splitlines():
            p = ln.split()
            if len(p) == 5 and int(float(p[0])) == cls:  # boxes only (skip polygons)
                x, y, w, h = map(float, p[1:5])
                boxes.append(((x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H))
    return boxes


def make_tile(c):
    im = Image.open(c["path"]).convert("RGB")
    x1, y1, x2, y2 = c["box"]
    bw, bh = x2 - x1, y2 - y1
    m = max(3 * max(bw, bh), 60)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    L, T = max(0, int(cx - m * 1.33)), max(0, int(cy - m))
    R, B = min(im.width, int(cx + m * 1.33)), min(im.height, int(cy + m))
    d = ImageDraw.Draw(im)
    for g in c["gt"]:
        d.rectangle(g, outline=(0, 220, 0), width=2)
    d.rectangle((x1 - 2, y1 - 2, x2 + 2, y2 + 2), outline=(255, 0, 0), width=2)
    if R <= L or B <= T:
        L, T, R, B = 0, 0, im.width, im.height
    crop = im.crop((L, T, R, B))
    crop.thumbnail((TILE_W, TILE_H - 16))
    tile = Image.new("RGB", (TILE_W, TILE_H), (30, 30, 30))
    tile.paste(crop, ((TILE_W - crop.width) // 2, 16))
    ImageDraw.Draw(tile).text((4, 2), f"#{c['idx']}  {c['conf']:.2f}", fill=(255, 255, 0))
    return tile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="plate_audit")
    ap.add_argument("--cls", type=int, default=3)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--min-w", type=float, default=16, help="ignore predictions narrower than this (px @ imgsz)")
    ap.add_argument("--iou", type=float, default=0.2, help="match threshold to a labelled plate")
    ap.add_argument("--per-sheet", type=int, default=48)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    root, out = Path(a.data).expanduser(), Path(a.out).expanduser()
    (out / "sheets").mkdir(parents=True, exist_ok=True)

    # one image per source (prefer valid/test originals if the export has aug copies)
    chosen = {}
    for s in ("valid", "test", "train"):
        d = root / s / "images"
        if d.exists():
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in IMG_EXTS:
                    chosen.setdefault(base_name(p), p)
    paths = [str(p) for p in chosen.values()]
    if not paths:
        print(f"No images found. Looked for {root}/{{valid,test,train}}/images/*.jpg|png")
        print(f"Contents of {root}: {sorted(x.name for x in root.iterdir()) if root.exists() else 'FOLDER DOES NOT EXIST'}")
        return
    print(f"auditing {len(paths)} source images with {a.weights}")

    model = YOLO(a.weights)
    cands, n_pred, n_match, n_small = [], 0, 0, 0
    def predict_in_chunks(paths, size=16):
        for i in range(0, len(paths), size):
            chunk = paths[i:i + size]
            results = model.predict(chunk, imgsz=a.imgsz, conf=a.conf, classes=[a.cls],
                                    verbose=False, device=a.device)
            yield from zip(chunk, results)  # r.path is unreliable for list input
            if i % 320 == 0:
                print(f"   {i}/{len(paths)}", flush=True)

    for real_path, r in predict_in_chunks(paths):
        p = Path(real_path)
        H, W = r.orig_shape
        s = a.imgsz / max(W, H)
        gt = load_gt(p, W, H, a.cls)
        for box, conf in zip(r.boxes.xyxy.cpu().tolist(), r.boxes.conf.cpu().tolist()):
            n_pred += 1
            if (box[2] - box[0]) * s < a.min_w:
                n_small += 1
                continue
            if any(iou(box, g) >= a.iou or center_inside(box, g) for g in gt):
                n_match += 1
                continue
            m = NAME_RE.match(base_name(p))
            cands.append(dict(path=p, video=m.group(1) if m else "?", conf=conf, box=box,
                              width=(box[2] - box[0]) * s, gt=gt))

    cands.sort(key=lambda c: -c["conf"])
    for i, c in enumerate(cands):
        c["idx"] = i

    with open(out / "candidates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "source_image", "file", "video", "conf", "x1", "y1", "x2", "y2", "width_px@imgsz"])
        for c in cands:
            w.writerow([c["idx"], base_name(c["path"]), c["path"].name, c["video"], f"{c['conf']:.3f}",
                        *[int(v) for v in c["box"]], f"{c['width']:.1f}"])

    cols = 8
    for start in range(0, len(cands), a.per_sheet):
        chunk = cands[start:start + a.per_sheet]
        rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * (TILE_W + 4), rows * (TILE_H + 4)), (0, 0, 0))
        for k, c in enumerate(chunk):
            sheet.paste(make_tile(c), ((k % cols) * (TILE_W + 4), (k // cols) * (TILE_H + 4)))
        sheet.save(out / "sheets" / f"sheet_{start:04d}-{start + len(chunk) - 1:04d}.jpg", quality=90)

    print(f"\nplate predictions: {n_pred} | matched a label: {n_match} | too small (ignored): {n_small}"
          f" | UNLABELLED candidates: {len(cands)}")
    print(f"images with at least one candidate: {len({c['path'] for c in cands})}")
    print("\ncandidates per video (top 15):")
    for v, n in Counter(c["video"] for c in cands).most_common(15):
        print(f"   {v:>45}: {n}")
    print(f"\nreview {out/'sheets'} (start with sheet_0000), look up #idx in {out/'candidates.csv'}")


if __name__ == "__main__":
    main()

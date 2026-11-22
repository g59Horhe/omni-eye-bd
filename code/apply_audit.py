#!/usr/bin/env python3
"""
Add human-verified missing plate labels from the audit to a COPY of the dataset.

You review the audit sheets and note:
  --upto   the last #idx you reviewed (everything after is ignored)
  --reject the #idx numbers that are NOT real plates (or bad boxes)
Every other candidate 0..upto is appended as a license_plate label.

Usage:
  python apply_audit.py --data ~/FYDP/TV_R-raw --audit ~/FYDP/plate_audit_025 \
      --dst ~/FYDP/TV_R-raw-fixed --upto 150 --reject 33 39 41
  (add --dry-run first to see what would change)

Writes <dst>/added_plate_labels.csv (what was added where) for your report.
The source dataset is never modified.
"""
import argparse
import csv
import shutil
from pathlib import Path

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--upto", type=int, required=True, help="last reviewed #idx (inclusive)")
    ap.add_argument("--reject", type=int, nargs="*", default=[])
    ap.add_argument("--cls", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    src = Path(a.data).expanduser()
    dst = Path(a.dst).expanduser()
    rows = list(csv.DictReader(open(Path(a.audit).expanduser() / "candidates.csv")))
    reject = set(a.reject)
    keep = [r for r in rows if int(r["idx"]) <= a.upto and int(r["idx"]) not in reject]
    print(f"{len(rows)} candidates | reviewed 0..{a.upto} | rejected {len(reject)} | adding {len(keep)}")

    if not a.dry_run:
        if dst.exists():
            raise SystemExit(f"{dst} already exists; delete it or choose another --dst")
        shutil.copytree(src, dst)
    root = src if a.dry_run else dst

    # locate each image file in any split
    where = {}
    for s in ("train", "valid", "test"):
        d = root / s / "images"
        if d.exists():
            for p in d.iterdir():
                where[p.name] = p

    added, missing = [], 0
    for r in keep:
        img = where.get(r["file"])
        if img is None:
            missing += 1
            continue
        with Image.open(img) as im:
            W, H = im.size
        x1, y1, x2, y2 = (float(r[k]) for k in ("x1", "y1", "x2", "y2"))
        line = (f"{a.cls} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        lbl = img.parent.parent / "labels" / (img.stem + ".txt")
        if not a.dry_run:
            existing = lbl.read_text() if lbl.exists() else ""
            if existing and not existing.endswith("\n"):
                existing += "\n"
            lbl.write_text(existing + line + "\n")
        added.append([r["idx"], r["source_image"], img.parent.parent.name, r["conf"], line])

    print(f"added {len(added)} plate labels to {len({x[1] for x in added})} images"
          + (f" | {missing} files not found" if missing else ""))
    if a.dry_run:
        print("Dry run only, nothing written.")
        return
    with open(dst / "added_plate_labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "source_image", "split", "conf", "yolo_line"])
        w.writerows(added)
    print(f"Wrote {dst} (+ added_plate_labels.csv)")


if __name__ == "__main__":
    main()

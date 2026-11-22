#!/usr/bin/env python3
"""Split the group C held-out result by footbridge, using the loso_C_* models (trained without any
group C footage). Subsets: Notunbazar (9 daytime WhatsApp clips) and the night clips from
Link Road + Moddho Badda (IMG_1079, IMG_1080, IMG_1083). Ultralytics 8.4.62 (~/fydp_env)."""
import csv, json
from pathlib import Path
from ultralytics import YOLO
ROOT = Path("/home/georgedessa/FYDP"); SRC = ROOT / "TV_R-final-filtered"; OUT = ROOT / "loso"
NAMES = ["3_person_violation", "footpath_violation", "helmet_violation", "license_plate", "wrong_side_violation"]
site = {r["video"]: r["footbridge"] for r in csv.DictReader(open(OUT / "site_map.csv"))}
subsets = {"notunbazar": lambda v: site[v] == "Notunbazar",
           "night_linkroad_moddhobadda": lambda v: site[v] in ("Link Road", "Moddho Badda")}
import re
VR = re.compile(r"^(.*?)_(?:MOV|mp4)-\d+")
rows = []
for name, keep in subsets.items():
    d = OUT / "data" / f"sub_C_{name}"
    for sub in ("images", "labels"):
        (d / "test" / sub).mkdir(parents=True, exist_ok=True)
    n = 0
    for img in SRC.glob("*/images/*.jpg"):
        if not keep(VR.match(img.name).group(1)):
            continue
        lab = img.parent.parent / "labels" / (img.stem + ".txt")
        for s, t in ((img, d / "test/images" / img.name), (lab, d / "test/labels" / lab.name)):
            if s.exists() and not t.exists():
                t.symlink_to(s)
        n += 1
    (d / "data.yaml").write_text(f"path: {d}\ntrain: test/images\nval: test/images\ntest: test/images\n"
                                 f"nc: 5\nnames:\n" + "".join(f"- {x}\n" for x in NAMES))
    for cfg in ("s_aug", "n_aug"):
        m = YOLO(str(ROOT / "runs" / f"loso_C_{cfg}" / "weights" / "best.pt")).val(
            data=str(d / "data.yaml"), split="test", imgsz=1280, batch=8, plots=False, verbose=False,
            project=str(OUT / "eval"), name=f"sub_C_{name}_{cfg}", exist_ok=True)
        r = {"subset": name, "images": n, "config": cfg, "mAP50": round(float(m.box.map50), 4),
             "mAP50_95": round(float(m.box.map), 4)}
        for i, c in enumerate(m.box.ap_class_index):
            r[f"AP50_{NAMES[int(c)]}"] = round(float(m.box.ap50[i]), 4)
        rows.append(r); print(r, flush=True)
keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("subset", "images", "config"), k))
with open(OUT / "c_subsets.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

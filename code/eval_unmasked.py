#!/usr/bin/env python3
"""
Reviewer 8, point 5: score the paper's seed-0 models on the UNMASKED test and valid labels
(TV_R-final, every plate kept, no grey fill) next to the masked labels the paper reports
(TV_R-final-filtered, plates < 16 px removed). Writes seeds/unmasked_eval.csv and .md.
"""
import csv
from pathlib import Path

from ultralytics import YOLO

ROOT = Path("/home/georgedessa/FYDP")
OUT = ROOT / "seeds"
MODELS = ["final_v8n", "final_n", "final_n_aug", "final_s", "final_s_aug"]
DATASETS = {"masked": ROOT / "TV_R-final-filtered" / "data.yaml",
            "unmasked": ROOT / "TV_R-final" / "data.yaml"}

rows = []
for run in MODELS:
    model = YOLO(str(ROOT / "runs" / run / "weights" / "best.pt"))
    for split in ("val", "test"):
        for label, data in DATASETS.items():
            m = model.val(data=str(data), split=split, imgsz=1280, batch=8, plots=False,
                          project=str(OUT / "eval"), name=f"{run}_{split}_{label}", exist_ok=True,
                          verbose=False)
            plate = list(m.box.ap_class_index).index(3)  # 3 = license_plate
            rows.append(dict(model=run, split=split, labels=label,
                             mAP50=round(float(m.box.map50), 4), mAP50_95=round(float(m.box.map), 4),
                             plate_AP50=round(float(m.box.ap50[plate]), 4),
                             plate_AP50_95=round(float(m.box.ap[plate]), 4),
                             plate_P=round(float(m.box.p[plate]), 4), plate_R=round(float(m.box.r[plate]), 4)))
            print(rows[-1], flush=True)

with open(OUT / "unmasked_eval.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
with open(OUT / "unmasked_eval.md", "w") as f:
    f.write("| Model | Split | Labels | mAP50 | mAP50-95 | Plate AP50 | Plate AP50-95 | Plate P | Plate R |\n")
    f.write("|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        f.write("| " + " | ".join(str(v) for v in r.values()) + " |\n")

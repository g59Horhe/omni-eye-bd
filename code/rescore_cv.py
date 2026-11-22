#!/usr/bin/env python3
"""Re-score the 5-fold video-level CV models (trained under 8.4.152) with Ultralytics 8.4.62, the
version behind Table V, so all reported numbers use one metric implementation. Writes seeds/cv_rescored.md."""
import json
from pathlib import Path
import numpy as np
from ultralytics import YOLO
ROOT = Path("/home/georgedessa/FYDP")
NAMES = ["3_person_violation", "footpath_violation", "helmet_violation", "license_plate", "wrong_side_violation"]
rows = []
for k in range(5):
    m = YOLO(str(ROOT / f"runs/cv5_s_aug_fold{k}/weights/best.pt"))
    row = {"fold": k}
    for split in ("val", "test"):
        r = m.val(data=str(ROOT / f"cv/folds/fold{k}/data.yaml"), split=split, imgsz=1280, batch=4, plots=False,
                  verbose=False, project=str(ROOT / "seeds/eval"), name=f"cv{k}_{split}", exist_ok=True)
        row[f"{split}_mAP50"], row[f"{split}_mAP50_95"] = float(r.box.map50), float(r.box.map)
        row[f"{split}_AP50"] = {NAMES[int(c)]: float(r.box.ap50[i]) for i, c in enumerate(r.box.ap_class_index)}
    rows.append(row); print(k, round(row["val_mAP50"], 3), round(row["test_mAP50"], 3), flush=True)
(ROOT / "seeds/cv_rescored.json").write_text(json.dumps(rows, indent=1))
sd = lambda x: np.std(x, ddof=1)
t50 = [r["test_mAP50"] for r in rows]; t95 = [r["test_mAP50_95"] for r in rows]
lines = ["| Fold | val mAP50 | test mAP50 | test mAP50-95 | " + " | ".join(n.split("_")[0] for n in NAMES) + " |",
         "|---|---|---|---|" + "---|" * 5]
for r in rows:
    lines.append(f"| {r['fold']} | {r['val_mAP50']:.3f} | {r['test_mAP50']:.3f} | {r['test_mAP50_95']:.3f} | "
                 + " | ".join(f"{r['test_AP50'][n]:.3f}" if n in r["test_AP50"] else "-" for n in NAMES) + " |")
pc = {n: [r["test_AP50"][n] for r in rows if n in r["test_AP50"]] for n in NAMES}
lines.append(f"| mean ± sd | | {np.mean(t50):.3f} ± {sd(t50):.3f} | {np.mean(t95):.3f} ± {sd(t95):.3f} | "
             + " | ".join(f"{np.mean(v):.3f} ± {sd(v):.3f}" for v in pc.values()) + " |")
(ROOT / "seeds/cv_rescored.md").write_text("Ultralytics 8.4.62 re-score of runs/cv5_s_aug_fold* (YOLO26s + aug, trained under 8.4.152)\n\n" + "\n".join(lines) + "\n")
print("\n".join(lines))

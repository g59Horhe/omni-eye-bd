#!/usr/bin/env python3
"""
Evaluate trained runs on val and test and write paper-ready tables.

Usage:
  python make_results_table.py            # uses the default run list below
  python make_results_table.py --runs final_n final_v8n

Protocol for the paper: choose the final model by VAL, report all models on TEST.
Outputs: results_table.csv and results_table.md in ~/FYDP
"""
import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

HOME = Path.home() / "FYDP"
DEFAULT_RUNS = ["tv_r_v5_clean", "final_n", "final_n_aug", "final_v8n", "final_s_aug"]
NOTES = {
    "tv_r_v5_clean": "YOLO26n, before label fix",
    "final_n": "YOLO26n",
    "final_n_aug": "YOLO26n + aug",
    "final_v8n": "YOLOv8n",
    "final_s_aug": "YOLO26s + aug",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--data", default=str(HOME / "TV_R-final-filtered" / "data.yaml"))
    ap.add_argument("--imgsz", type=int, default=1280)
    a = ap.parse_args()

    rows, names = [], None
    for run in a.runs:
        w = HOME / "runs" / run / "weights" / "best.pt"
        if not w.exists():
            print(f"[skip] {run}: no best.pt")
            continue
        model = YOLO(str(w))
        params = sum(p.numel() for p in model.model.parameters()) / 1e6
        for split in ("val", "test"):
            r = model.val(data=a.data, split=split, imgsz=a.imgsz, batch=16, plots=False, verbose=False)
            names = r.names
            ap50 = dict(zip([int(i) for i in r.box.ap_class_index], r.box.ap50))
            ap5095 = dict(zip([int(i) for i in r.box.ap_class_index], r.box.ap))
            rows.append(dict(run=run, model=NOTES.get(run, run), split=split, params_M=round(params, 2),
                             P=r.box.mp, R=r.box.mr, mAP50=r.box.map50, mAP50_95=r.box.map,
                             ms_per_img=r.speed.get("inference", 0),
                             **{f"AP50_{names[c]}": ap50.get(c, 0) for c in names},
                             **{f"AP50-95_{names[c]}": ap5095.get(c, 0) for c in names}))
            print(f"{run:>15} {split:>4}: mAP50 {r.box.map50:.3f}  mAP50-95 {r.box.map:.3f}")

    if not rows:
        raise SystemExit("no runs evaluated")
    keys = list(rows[0].keys())
    with open(HOME / "results_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    cls = [names[c] for c in sorted(names)]
    short = [c.replace("_violation", "").replace("license_plate", "plate") for c in cls]
    md = []
    for split in ("val", "test"):
        md.append(f"\n### {split.upper()} (AP50 per class)\n")
        md.append("| Model | Params (M) | P | R | mAP50 | mAP50-95 | " + " | ".join(short) + " | ms/img |")
        md.append("|" + "---|" * (7 + len(cls)))
        for r in rows:
            if r["split"] != split:
                continue
            md.append(f"| {r['model']} | {r['params_M']} | {r['P']:.3f} | {r['R']:.3f} | {r['mAP50']:.3f} | "
                      f"{r['mAP50_95']:.3f} | " + " | ".join(f"{r[f'AP50_{c}']:.3f}" for c in cls)
                      + f" | {r['ms_per_img']:.1f} |")
    (HOME / "results_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nsaved {HOME/'results_table.csv'} and {HOME/'results_table.md'}")


if __name__ == "__main__":
    main()

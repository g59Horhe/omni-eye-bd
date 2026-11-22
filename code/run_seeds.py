#!/usr/bin/env python3
"""
Run with ~/fydp_env (Ultralytics 8.4.62), the environment the paper used; 8.4.152 scores differently.
Repeat the five Table V configurations of the ICCIT paper over extra seeds.

Each run is the seed-0 run's exact settings (runs/final_*/args.yaml) with only the seed
changed. Each run trains in its own child process so GPU memory is released between runs,
then best.pt is scored once on the valid and test splits and appended to seeds/results.csv.

  ~/fydp_env/bin/python seeds/run_seeds.py all        # every (seed, config) not yet finished
  python seeds/run_seeds.py status     # what is done
Safe to rerun after a crash: finished runs are skipped, an interrupted one resumes.
"""
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/georgedessa/FYDP")
RUNS = ROOT / "runs"
OUT = ROOT / "seeds"
DATA = ROOT / "TV_R-final-filtered" / "data.yaml"
SEEDS = [1, 2]
# Priority order: the selected model and the closest rival first.
CONFIGS = {
    "s_aug": dict(model="yolo26s.pt", batch=8, hsv_v=0.5, scale=0.7, mixup=0.1),
    "n_aug": dict(model="yolo26n.pt", batch=16, hsv_v=0.5, scale=0.7, mixup=0.1),
    "n":     dict(model="yolo26n.pt", batch=16, hsv_v=0.4, scale=0.5, mixup=0.0),
    "s":     dict(model="yolo26s.pt", batch=8, hsv_v=0.4, scale=0.5, mixup=0.0),
    "v8n":   dict(model="yolov8n.pt", batch=16, hsv_v=0.4, scale=0.5, mixup=0.0),
}
COMMON = dict(epochs=150, patience=30, imgsz=1280, cache=True, workers=16, optimizer="auto",
              deterministic=True, amp=True)


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(OUT / "seeds.log", "a") as f:
        f.write(line + "\n")


def run_name(cfg, seed):
    return f"seed{seed}_{cfg}"


def done(cfg, seed):
    return (RUNS / run_name(cfg, seed) / "SEED_DONE").exists()


def train_one(cfg, seed):
    from ultralytics import YOLO
    rd = RUNS / run_name(cfg, seed)
    last = rd / "weights" / "last.pt"
    if last.exists() and not (rd / "TRAINED").exists():
        log(f"{rd.name}: resuming from last.pt")
        try:
            YOLO(str(last)).train(resume=True)
        except Exception as e:
            if "nothing to resume" not in str(e):
                raise
    elif not (rd / "TRAINED").exists():
        c = dict(CONFIGS[cfg])
        model = str(ROOT / c.pop("model"))
        log(f"{rd.name}: training {Path(model).name} seed={seed} {c}")
        YOLO(model).train(data=str(DATA), project=str(RUNS), name=rd.name, exist_ok=True,
                          seed=seed, **COMMON, **c)
    (rd / "TRAINED").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))

    best = YOLO(str(rd / "weights" / "best.pt"))
    names = best.names
    row = {"config": cfg, "seed": seed}
    for split in ("val", "test"):
        m = best.val(data=str(DATA), split=split, imgsz=1280, batch=16, plots=False,
                     project=str(OUT / "eval"), name=f"{rd.name}_{split}", exist_ok=True, verbose=False)
        row[f"{split}_mAP50"] = round(float(m.box.map50), 4)
        row[f"{split}_mAP50_95"] = round(float(m.box.map), 4)
        row[f"{split}_P"] = round(float(m.box.mp), 4)
        row[f"{split}_R"] = round(float(m.box.mr), 4)
        for i, c in enumerate(m.box.ap_class_index):
            row[f"{split}_AP50_{names[int(c)]}"] = round(float(m.box.ap50[i]), 4)
    (rd / "metrics.json").write_text(json.dumps(row, indent=1))
    log(f"{rd.name}: val mAP50 {row['val_mAP50']}  test mAP50 {row['test_mAP50']}  "
        f"test mAP50-95 {row['test_mAP50_95']}")
    (rd / "SEED_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))


def collect():
    rows = []
    for seed in SEEDS:
        for cfg in CONFIGS:
            p = RUNS / run_name(cfg, seed) / "metrics.json"
            if p.exists():
                rows.append(json.loads(p.read_text()))
    if not rows:
        return
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("config", "seed"), k))
    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "one":
        train_one(sys.argv[2], int(sys.argv[3]))
    elif cmd == "status":
        for seed in SEEDS:
            for cfg in CONFIGS:
                print(f"seed {seed} {cfg:6s} {'done' if done(cfg, seed) else 'pending'}")
    elif cmd == "all":
        log("starting seed runs")
        for seed in SEEDS:
            for cfg in CONFIGS:
                if done(cfg, seed):
                    continue
                r = subprocess.run([sys.executable, __file__, "one", cfg, str(seed)])
                if r.returncode != 0:
                    log(f"{run_name(cfg, seed)}: FAILED (exit {r.returncode}); continuing")
                collect()
        collect()
        log("all seed runs finished")


if __name__ == "__main__":
    main()

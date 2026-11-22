#!/usr/bin/env python3
"""
Reviewer 8, point 1: bootstrap intervals for the Table V models without retraining.

Runs Ultralytics validation (8.4.62, ~/fydp_env, same protocol as the paper) and keeps the
per-image matching statistics (tp at 10 IoU thresholds, confidence, classes). mAP is then
recomputed with Ultralytics' own ap_per_class on resampled sets:

  image   - resample images with replacement (variation from which frames were labelled)
  video   - resample whole videos with replacement (variation from which clips are in the
            split; the honest one when 5 test videos carry the result)

The same resamples are used for every model, so model differences are paired.
Writes seeds/bootstrap_ci.md and seeds/bootstrap_ci.json.

  ~/fydp_env/bin/python seeds/bootstrap_ci.py
"""
import json
import re
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.utils import metrics as um

ROOT = Path("/home/georgedessa/FYDP")
OUT = ROOT / "seeds"
DATA = ROOT / "TV_R-final-filtered" / "data.yaml"
MODELS = {"YOLOv8n": "final_v8n", "YOLO26n": "final_n", "YOLO26n + aug": "final_n_aug",
          "YOLO26s": "final_s", "YOLO26s + aug": "final_s_aug"}
PAIRS = [("YOLO26s + aug", "YOLO26n + aug"), ("YOLO26s + aug", "YOLO26n"), ("YOLO26s", "YOLO26n"),
         ("YOLO26n", "YOLOv8n"), ("YOLO26n + aug", "YOLO26n")]
B = 2000
VIDEO_RE = re.compile(r"^(.*?)_(?:MOV|mp4)-\d+")

_records = []
_orig_update = um.DetMetrics.update_stats


def _capture(self, stat):
    _records.append({k: np.asarray(stat[k]).copy() if k != "im_name" else stat[k]
                     for k in ("tp", "conf", "pred_cls", "target_cls", "im_name")})
    return _orig_update(self, stat)


um.DetMetrics.update_stats = _capture


def per_image_stats(run, split):
    _records.clear()
    m = YOLO(str(ROOT / "runs" / run / "weights" / "best.pt"))
    r = m.val(data=str(DATA), split=split, imgsz=1280, batch=16, plots=False, verbose=False,
              project=str(OUT / "eval"), name=f"boot_{run}_{split}", exist_ok=True)
    recs = sorted(_records, key=lambda d: d["im_name"])
    return recs, float(r.box.map50), float(r.box.map), r.names


def score(recs, idx, nc):
    """mAP50, mAP50-95 and per-class AP50 over the images in idx (repeats allowed)."""
    sel = [recs[i] for i in idx]
    tp = np.concatenate([d["tp"] for d in sel]).reshape(-1, 10)
    conf = np.concatenate([d["conf"] for d in sel])
    pcls = np.concatenate([d["pred_cls"] for d in sel])
    tcls = np.concatenate([d["target_cls"] for d in sel])
    if tcls.size == 0:
        return np.nan, np.nan, np.full(nc, np.nan)
    out = um.ap_per_class(tp, conf, pcls, tcls, plot=False)
    ap, classes = out[5], out[6]
    per_class = np.full(nc, np.nan)
    per_class[classes] = ap[:, 0]
    return ap[:, 0].mean(), ap.mean(), per_class


def ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if x.size else (np.nan, np.nan)


def main():
    results = {}
    for split in ("test", "val"):
        stats, point = {}, {}
        for label, run in MODELS.items():
            recs, m50, m5095, names = per_image_stats(run, split)
            stats[label] = recs
            point[label] = (m50, m5095)
        nc = len(names)
        recs0 = stats[next(iter(MODELS))]
        im_names = [d["im_name"] for d in recs0]
        for label in MODELS:  # every model must have seen the same images in the same order
            assert [d["im_name"] for d in stats[label]] == im_names
        videos = np.array([VIDEO_RE.match(n).group(1) for n in im_names])
        vid_ids = sorted(set(videos))
        by_video = {v: np.flatnonzero(videos == v) for v in vid_ids}

        rng = np.random.default_rng(0)
        n = len(im_names)
        designs = {
            "image": [rng.integers(0, n, n) for _ in range(B)],
            "video": [np.concatenate([by_video[v] for v in rng.choice(vid_ids, len(vid_ids))])
                      for _ in range(B)],
        }
        full = np.arange(n)
        res = {"n_images": n, "videos": vid_ids, "B": B, "models": {}, "pairs": {}}
        draws = {}
        for label in MODELS:
            m50, m5095, pc = score(stats[label], full, nc)
            assert abs(m50 - point[label][0]) < 1e-6, (label, m50, point[label][0])
            entry = {"mAP50": m50, "mAP50_95": m5095,
                     "AP50": {names[c]: pc[c] for c in range(nc)}}
            for d, idxs in designs.items():
                s = [score(stats[label], ix, nc) for ix in idxs]
                draws[(label, d)] = np.array([x[0] for x in s])
                entry[f"mAP50_ci_{d}"] = ci([x[0] for x in s])
                entry[f"mAP50_95_ci_{d}"] = ci([x[1] for x in s])
                entry[f"AP50_ci_{d}"] = {names[c]: ci([x[2][c] for x in s]) for c in range(nc)}
            res["models"][label] = entry
            print(split, label, round(m50, 3), entry["mAP50_ci_image"], entry["mAP50_ci_video"], flush=True)
        for a, b in PAIRS:
            p = {"diff": res["models"][a]["mAP50"] - res["models"][b]["mAP50"]}
            for d in designs:
                dd = draws[(a, d)] - draws[(b, d)]
                p[f"ci_{d}"] = ci(dd)
                p[f"P(a>b)_{d}"] = float(np.nanmean(dd > 0))
            res["pairs"][f"{a} - {b}"] = p
        results[split] = res

    (OUT / "bootstrap_ci.json").write_text(json.dumps(results, indent=1, default=float))
    f3 = lambda x: f"{x:.3f}"
    rng_s = lambda t: f"[{t[0]:.3f}, {t[1]:.3f}]"
    lines = [f"# Bootstrap 95% intervals (B={B}, percentile; Ultralytics 8.4.62)", ""]
    for split, res in results.items():
        lines += [f"## {split} split: {res['n_images']} images, {len(res['videos'])} videos "
                  f"({', '.join(res['videos'])})", "",
                  "| Model | mAP50 | 95% CI, images | 95% CI, videos | mAP50-95 | 95% CI, images | 95% CI, videos |",
                  "|---|---|---|---|---|---|---|"]
        for label, e in res["models"].items():
            lines.append(f"| {label} | {f3(e['mAP50'])} | {rng_s(e['mAP50_ci_image'])} | {rng_s(e['mAP50_ci_video'])} | "
                         f"{f3(e['mAP50_95'])} | {rng_s(e['mAP50_95_ci_image'])} | {rng_s(e['mAP50_95_ci_video'])} |")
        lines += ["", "Paired mAP50 differences (same resamples for both models):", "",
                  "| Comparison | Diff | 95% CI, images | P(A>B), images | 95% CI, videos | P(A>B), videos |",
                  "|---|---|---|---|---|---|"]
        for k, p in res["pairs"].items():
            lines.append(f"| {k} | {p['diff']:+.3f} | {rng_s(p['ci_image'])} | {p['P(a>b)_image']:.2f} | "
                         f"{rng_s(p['ci_video'])} | {p['P(a>b)_video']:.2f} |")
        e = res["models"]["YOLO26s + aug"]
        lines += ["", "Per-class AP50 of YOLO26s + aug:", "",
                  "| Class | AP50 | 95% CI, images | 95% CI, videos |", "|---|---|---|---|"]
        for c, v in e["AP50"].items():
            lines.append(f"| {c} | {f3(v)} | {rng_s(e['AP50_ci_image'][c])} | {rng_s(e['AP50_ci_video'][c])} |")
        lines.append("")
    (OUT / "bootstrap_ci.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

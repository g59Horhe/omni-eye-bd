#!/usr/bin/env python3
"""
5-fold video-level cross-validation for Omni-Eye.

Every video is in the test set exactly once. Folds are built from whole video
groups, so no video (and no near-duplicate frame) is ever in two splits:
  * frames are grouped by source video,
  * the nine re-shared WhatsApp clips stay together (locations.txt),
  * any two frames within 20 bits dHash (flip-aware) are merged,
exactly as in resplit_by_group.py. Groups are then dealt into 5 test folds
balanced on per-class instance counts, image count and night-time frames. For
each fold, a validation set (~15 % of all frames, used only for early stopping
and picking best.pt) is taken from the other four folds; the rest is train.

Each fold trains the paper's selected setup (YOLO26s + aug: mixup 0.1,
scale 0.7, hsv_v 0.5, imgsz 1280, 150 epochs, patience 30, batch 8, seed 0),
then best.pt is scored once on that fold's test set, overall and per video.

Commands (run from ~/FYDP with the venv active):
  python cv_experiment.py folds            # build the folds, print the summary (1–2 min)
  python cv_experiment.py train            # train every fold that is not done (resumes)
  python cv_experiment.py eval             # (re)score finished folds on their test sets
  python cv_experiment.py report           # tables: cv/cv_results.md, .csv, .tex, per-video
  python cv_experiment.py status           # progress of every fold
  python cv_experiment.py all              # folds (if missing) + train + eval + report
  bash run_cv.sh                           # `all` in the background, log in cv/cv.log
  python cv_experiment.py final            # final model on all 40 videos (fixed epochs from the folds)
  python cv_experiment.py test --data ~/FYDP/pixel_test   # score it on newly labelled footage

Useful options: --folds 0 2  (only some folds), --epochs/--batch/--workers/--model
to override the training setup, `folds --force` to rebuild (refused once training started).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent            # ~/FYDP
CV = ROOT / "cv"
FOLDS_DIR = CV / "folds"
RES_DIR = CV / "results"
RUNS = ROOT / "runs"
DEFAULT_SRC = ROOT / "TV_R-final-filtered"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NAME_RE = re.compile(r"^(.+?)_([A-Za-z0-9]+-\d+)$")
SPLITS = ("train", "valid", "test")
NIGHT_VIDEOS = ["IMG_1079", "IMG_1080", "IMG_1082", "IMG_1083"]   # the 406 night-time frames
RUN_PREFIX = "cv5_s_aug_fold"
HASH = 16

# The paper's selected configuration (runs/final_s_aug/args.yaml)
TRAIN_CFG = dict(epochs=150, patience=30, batch=8, imgsz=1280, cache=True, workers=16, seed=0,
                 deterministic=True, hsv_v=0.5, scale=0.7, mixup=0.1, plots=True)
DEFAULT_MODEL = "yolo26s.pt"


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# --------------------------------------------------------------------------- #
# Dataset scanning / grouping (same rules as resplit_by_group.py)
# --------------------------------------------------------------------------- #
def base_name(p: Path) -> str:
    return re.sub(r"_(jpg|jpeg|png)\.rf\..*$", "", p.stem if p.suffix else p.name)


def video_of(base: str) -> str:
    m = NAME_RE.match(base)
    return m.group(1) if m else base


def lbl_path(img: Path) -> Path:
    return img.parent.parent / "labels" / (img.stem + ".txt")


def dhash(path: Path):
    from PIL import Image
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").resize((HASH + 1, HASH), Image.BILINEAR), dtype=np.int16)
        gf = g[:, ::-1]
    return (g[:, 1:] > g[:, :-1]).flatten(), (gf[:, 1:] > gf[:, :-1]).flatten()


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def load_location_map(path: Path | None):
    m = {}
    if path and path.exists():
        for ln in path.read_text().splitlines():
            ln = ln.split("#")[0].replace(",", " ").split()
            if len(ln) >= 2:
                m[ln[0]] = ln[1]
    return m


def read_names(src: Path):
    names = yaml.safe_load(open(src / "data.yaml"))["names"]
    return [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)


def count_labels(lp: Path, nc: int):
    c = np.zeros(nc, dtype=np.int64)
    if lp.exists():
        for ln in lp.read_text().splitlines():
            parts = ln.split()
            if parts:
                k = int(float(parts[0]))
                if 0 <= k < nc:
                    c[k] += 1
    return c


def scan(src: Path, loc_map: dict, hash_thresh: int):
    """One entry per image in the (already de-duplicated) final dataset."""
    names = read_names(src)
    nc = len(names)
    items = []
    for s in SPLITS:
        d = src / s / "images"
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                b = base_name(p)
                v = video_of(b)
                items.append({"img": p, "lbl": lbl_path(p), "base": b, "video": v,
                              "loc": loc_map.get(v, v), "orig_split": s})
    if not items:
        raise SystemExit(f"no images found under {src}/{{train,valid,test}}/images")
    dup = defaultdict(list)
    for it in items:
        dup[it["base"]].append(it)
    if any(len(v) > 1 for v in dup.values()):
        n = sum(len(v) - 1 for v in dup.values())
        log(f"note: {n} images share a source name with another image (augmented copies?); kept together")
    n = len(items)
    C = np.stack([count_labels(it["lbl"], nc) for it in items])

    log(f"hashing {n} images for near-duplicate check…")
    H = np.zeros((n, HASH * HASH), dtype=bool)
    HF = np.zeros_like(H)
    for i, it in enumerate(items):
        H[i], HF[i] = dhash(it["img"])

    uf = UF(n)
    first = {}
    for i, it in enumerate(items):          # same video / same location group / same source name
        for key in (("loc", it["loc"]), ("base", it["base"])):
            if key in first:
                uf.union(i, first[key])
            else:
                first[key] = i
    cross = defaultdict(int)
    for i in range(n - 1):
        d = np.minimum((H[i + 1:] != H[i]).sum(1), (HF[i + 1:] != H[i]).sum(1))
        for j in np.nonzero(d <= hash_thresh)[0]:
            j = i + 1 + int(j)
            if uf.find(i) != uf.find(j):
                cross[tuple(sorted((items[i]["loc"], items[j]["loc"])))] += 1
            uf.union(i, j)
    if cross:
        log("merged because of near-identical frames: "
            + ", ".join(f"{a}~{b} ({c})" for (a, b), c in sorted(cross.items(), key=lambda x: -x[1])))

    roots = sorted({uf.find(i) for i in range(n)})
    gid = {r: g for g, r in enumerate(roots)}
    for i, it in enumerate(items):
        it["group"] = gid[uf.find(i)]
    return items, C, names


# --------------------------------------------------------------------------- #
# Fold assignment
# --------------------------------------------------------------------------- #
def group_arrays(items, C, night_videos):
    G = 1 + max(it["group"] for it in items)
    nc = C.shape[1]
    GC = np.zeros((G, nc))
    GS = np.zeros(G)
    GN = np.zeros(G)
    for i, it in enumerate(items):
        g = it["group"]
        GC[g] += C[i]
        GS[g] += 1
        GN[g] += it["video"] in night_videos
    return GC, GS, GN


def assign_test_folds(GC, GS, GN, K, tries, rng):
    G, nc = GC.shape
    exp_c = GC.sum(0) / K
    exp_n = GS.sum() / K
    exp_night = GN.sum() / K

    def score(asg):
        S = np.zeros((K, nc)); Z = np.zeros(K); N = np.zeros(K)
        np.add.at(S, asg, GC); np.add.at(Z, asg, GS); np.add.at(N, asg, GN)
        cls = (((S - exp_c) / np.maximum(exp_c, 1)) ** 2).sum()
        size = (((Z - exp_n) / exp_n) ** 2).sum()
        night = (((N - exp_night) / max(exp_night, 1)) ** 2).sum() if exp_night else 0
        empty = ((S == 0) & (exp_c > 0)).sum()          # a class missing from a whole test fold
        return cls + 3 * size + 0.5 * night + 2 * empty

    best = None
    order = np.argsort(-GS)
    for t in range(tries):
        load = np.zeros(K)
        asg = np.empty(G, dtype=int)
        # big groups first (with some shuffling), each to the fold furthest below its share
        noisy = order[np.argsort(np.arange(G) + rng.normal(0, 3, G))]
        for g in noisy:
            k = int(np.argmax((exp_n - load) / exp_n + rng.uniform(0, 0.35, K)))
            asg[g] = k
            load[k] += GS[g]
        sc = score(asg)
        if best is None or sc < best[0]:
            best = (sc, asg.copy())
    return best[1], best[0]


def pick_val(GC, GS, cand, val_frac, tries, rng):
    """Choose validation groups among `cand` (non-test groups): ~val_frac of ALL frames, class-balanced."""
    cand = np.array(sorted(cand))
    total = GS.sum()
    target = val_frac * total
    pool_c = GC[cand].sum(0)
    share = target / GS[cand].sum()
    exp_c = share * pool_c

    def score(sel):
        S = GC[sel].sum(0); Z = GS[sel].sum()
        return (((S - exp_c) / np.maximum(exp_c, 1)) ** 2).sum() + 3 * ((Z - target) / target) ** 2 \
            + 2 * ((S == 0) & (exp_c > 0)).sum()

    best = None
    for _ in range(tries):
        sel, size = [], 0.0
        for g in rng.permutation(cand):
            if abs(size + GS[g] - target) < abs(size - target):
                sel.append(g)
                size += GS[g]
        if not sel:
            continue
        sc = score(sel)
        if best is None or sc < best[0]:
            best = (sc, sorted(sel))
    return set(int(g) for g in best[1])


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def training_started():
    return any(RUNS.glob(f"{RUN_PREFIX}*/weights/*.pt")) or any(RUNS.glob(f"{RUN_PREFIX}*/results.csv"))


def cmd_folds(a):
    if (FOLDS_DIR / "folds.json").exists() and not a.force:
        log(f"folds already exist in {FOLDS_DIR} — leaving them as they are (use --force to rebuild)")
        print((FOLDS_DIR / "summary.md").read_text())
        return
    if a.force and training_started():
        raise SystemExit("refusing to rebuild folds: training runs already exist for the current folds "
                         f"({RUNS}/{RUN_PREFIX}*). Move or delete them first.")
    src = Path(a.src).expanduser().resolve()
    loc_path = Path(a.location_map).expanduser() if a.location_map else None
    if loc_path is None:
        for cand in (ROOT / "release" / "splits" / "locations.txt", ROOT / "locations.txt"):
            if cand.exists():
                loc_path = cand
                break
    loc_map = load_location_map(loc_path)
    log(f"source: {src}   location map: {loc_path} ({len(loc_map)} videos)")

    items, C, names = scan(src, loc_map, a.hash_thresh)
    GC, GS, GN = group_arrays(items, C, set(a.night_videos))
    G, nc, K = len(GS), len(names), a.k
    vids = sorted({it["video"] for it in items})
    log(f"{len(items)} images · {len(vids)} videos · {G} groups · {K} folds")
    if G < K:
        raise SystemExit(f"only {G} groups — cannot make {K} folds")

    rng = np.random.default_rng(a.seed)
    test_of, sc = assign_test_folds(GC, GS, GN, K, a.search, rng)
    log(f"fold balance score {sc:.3f} (lower is better) after {a.search} tries")

    splits = []   # per fold: dict group -> split
    for k in range(K):
        test_g = set(np.nonzero(test_of == k)[0].tolist())
        rest = [g for g in range(G) if g not in test_g]
        val_g = pick_val(GC, GS, rest, a.val_frac, 4000, rng)
        splits.append({g: ("test" if g in test_g else "valid" if g in val_g else "train") for g in range(G)})

    # sanity: each group tested once, no group in two splits within a fold
    for g in range(G):
        assert sum(splits[k][g] == "test" for k in range(K)) == 1

    # ---- write fold datasets (symlinks, no copies)
    if FOLDS_DIR.exists():
        shutil.rmtree(FOLDS_DIR)
    FOLDS_DIR.mkdir(parents=True)
    for k in range(K):
        fd = FOLDS_DIR / f"fold{k}"
        for s in SPLITS:
            (fd / s / "images").mkdir(parents=True)
            (fd / s / "labels").mkdir(parents=True)
        for it in items:
            s = splits[k][it["group"]]
            (fd / s / "images" / it["img"].name).symlink_to(it["img"].resolve())
            if it["lbl"].exists():
                (fd / s / "labels" / it["lbl"].name).symlink_to(it["lbl"].resolve())
        yaml.safe_dump({"path": str(fd), "train": "train/images", "val": "valid/images",
                        "test": "test/images", "nc": nc, "names": names},
                       open(fd / "data.yaml", "w"), sort_keys=False)

    # ---- records + summary
    with open(FOLDS_DIR / "folds.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "video", "group", "test_fold"] + [f"fold{k}" for k in range(K)])
        for it in items:
            g = it["group"]
            w.writerow([it["img"].name, it["video"], g, int(test_of[g])] + [splits[k][g] for k in range(K)])
    meta = {"k": K, "seed": a.seed, "src": str(src), "val_frac": a.val_frac, "hash_thresh": a.hash_thresh,
            "names": names, "created": time.strftime("%Y-%m-%d %H:%M"),
            "folds": []}
    lines = [f"# Cross-validation folds\n",
             f"{len(items)} images, {len(vids)} videos, {G} groups (source: `{src.name}`, seed {a.seed}).\n",
             "| Fold | Split | Videos | Images | " + " | ".join(n.replace('_violation', '').replace('license_plate', 'plate') for n in names) + " | Night imgs |",
             "|---|---|---|---|" + "---|" * nc + "---|"]
    tot_c = C.sum(0)
    for k in range(K):
        fm = {"fold": k}
        for s in ("test", "valid", "train"):
            idx = [i for i, it in enumerate(items) if splits[k][it["group"]] == s]
            fv = sorted({items[i]["video"] for i in idx})
            cc = C[idx].sum(0) if idx else np.zeros(nc, int)
            night = sum(items[i]["video"] in a.night_videos for i in idx)
            fm[s] = {"videos": fv, "display": display_videos([items[i] for i in idx]), "images": len(idx),
                     "instances": cc.tolist(), "night_images": night}
            vtxt = ", ".join(fm[s]["display"]) if s != "train" else f"{len(fv)} videos"
            lines.append(f"| {k} | {s} | {vtxt} | {len(idx)} | "
                         + " | ".join(f"{int(v)} ({100 * v / max(t, 1):.0f}%)" for v, t in zip(cc, tot_c))
                         + f" | {night} |")
        meta["folds"].append(fm)
    lines.append("\nPercentages are the share of all instances of that class. Every video is in exactly one test fold.")
    (FOLDS_DIR / "folds.json").write_text(json.dumps(meta, indent=1))
    (FOLDS_DIR / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    log(f"wrote {K} fold datasets to {FOLDS_DIR} (symlinks to {src.name})")


def display_videos(its):
    """Video names for tables; re-shared clips that form one location group are shown once."""
    by_loc = defaultdict(set)
    for it in its:
        by_loc[it["loc"]].add(it["video"])
    out = []
    for loc, vs in by_loc.items():
        out += [f"{loc} ({len(vs)} clips)"] if len(vs) > 1 and loc not in vs else sorted(vs)
    return sorted(out, key=lambda x: (0, int(x), "") if x.isdigit() else (1, 0, x.lower()))


def load_meta():
    p = FOLDS_DIR / "folds.json"
    if not p.exists():
        raise SystemExit("no folds yet — run:  python cv_experiment.py folds")
    return json.loads(p.read_text())


def fold_list(a, meta):
    return a.folds if a.folds else list(range(meta["k"]))


def run_dir(k):
    return RUNS / f"{RUN_PREFIX}{k}"


def train_overrides(a):
    cfg = dict(TRAIN_CFG)
    for key in ("epochs", "batch", "workers", "imgsz", "patience"):
        v = getattr(a, key, None)
        if v is not None:
            cfg[key] = v
    if getattr(a, "no_cache", False):
        cfg["cache"] = False
    return cfg


def cmd_train_one(a):
    """Runs in a child process (one per fold) so GPU memory is fully released between folds."""
    k = a.fold
    rd = run_dir(k)
    if (rd / "CV_DONE").exists():
        log(f"fold {k}: already trained")
        return
    from ultralytics import YOLO
    last = rd / "weights" / "last.pt"
    if last.exists():
        log(f"fold {k}: resuming from {last}")
        try:
            YOLO(str(last)).train(resume=True)
        except Exception as e:           # finished runs cannot be resumed
            if "nothing to resume" in str(e) and (rd / "weights" / "best.pt").exists():
                log(f"fold {k}: training had already finished")
            else:
                raise
    else:
        model = a.model or (str(ROOT / DEFAULT_MODEL) if (ROOT / DEFAULT_MODEL).exists() else DEFAULT_MODEL)
        cfg = train_overrides(a)
        log(f"fold {k}: training {Path(model).name} with {cfg}")
        YOLO(model).train(data=str(FOLDS_DIR / f"fold{k}" / "data.yaml"), project=str(RUNS),
                          name=f"{RUN_PREFIX}{k}", exist_ok=True, **cfg)
    if not (rd / "weights" / "best.pt").exists():
        raise SystemExit(f"fold {k}: training ended without best.pt")
    (rd / "CV_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    log(f"fold {k}: training done")


def child_args(a, extra):
    out = []
    for key in ("epochs", "batch", "workers", "imgsz", "patience"):
        v = getattr(a, key, None)
        if v is not None:
            out += [f"--{key}", str(v)]
    if getattr(a, "model", None):
        out += ["--model", a.model]
    if getattr(a, "no_cache", False):
        out += ["--no-cache"]
    return out + extra


def cmd_train(a):
    meta = load_meta()
    for k in fold_list(a, meta):
        if (run_dir(k) / "CV_DONE").exists():
            log(f"fold {k}: already trained, skipping")
        else:
            t = time.time()
            log(f"===== fold {k}: training =====")
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "_train_one", "--fold", str(k)]
                               + child_args(a, []))
            if r.returncode != 0:
                raise SystemExit(f"fold {k}: training failed (exit {r.returncode}). Fix the error above and "
                                 "run the same command again — finished folds are skipped, this one resumes.")
            log(f"fold {k}: trained in {(time.time() - t) / 60:.0f} min")
        if not getattr(a, "no_eval", False) and not (RES_DIR / f"fold{k}.json").exists():
            eval_in_child(k, a)


def eval_in_child(k, a):
    """Score in a separate process too, so nothing stays on the GPU while the next fold trains."""
    extra = ["--imgsz", str(a.imgsz)] if getattr(a, "imgsz", None) else []
    r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "eval", "--folds", str(k)] + extra)
    if r.returncode != 0:
        log(f"fold {k}: evaluation failed (exit {r.returncode}); run `python cv_experiment.py eval --folds {k}` later")


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def labels_count(split_dir: Path, nc: int):
    c = np.zeros(nc, dtype=np.int64)
    for lp in (split_dir / "labels").glob("*.txt"):
        c += count_labels(lp, nc)
    return c


def metrics_dict(r, names, counts):
    idx = [int(i) for i in r.box.ap_class_index]
    ap50 = dict(zip(idx, [float(x) for x in r.box.ap50]))
    ap = dict(zip(idx, [float(x) for x in r.box.ap]))
    return {"P": float(r.box.mp), "R": float(r.box.mr), "mAP50": float(r.box.map50), "mAP50_95": float(r.box.map),
            "ms_per_img": float(r.speed.get("inference", 0.0)),
            "per_class": {names[c]: {"AP50": ap50.get(c) if counts[c] else None,
                                     "AP50_95": ap.get(c) if counts[c] else None,
                                     "n": int(counts[c])} for c in range(len(names))}}


def evaluate_fold(k, meta, a):
    from ultralytics import YOLO
    w = run_dir(k) / "weights" / "best.pt"
    if not w.exists():
        log(f"fold {k}: no best.pt yet, skipping evaluation")
        return
    fd = FOLDS_DIR / f"fold{k}"
    names = meta["names"]
    nc = len(names)
    imgsz = getattr(a, "imgsz", None) or TRAIN_CFG["imgsz"]
    log(f"fold {k}: scoring best.pt on its test set")
    model = YOLO(str(w))
    out = {"fold": k, "weights": str(w), "evaluated": time.strftime("%Y-%m-%d %H:%M"),
           "test_videos": meta["folds"][k]["test"].get("display", meta["folds"][k]["test"]["videos"]),
           "val_videos": meta["folds"][k]["valid"].get("display", meta["folds"][k]["valid"]["videos"])}
    for split, key in (("test", "test"), ("val", "valid")):
        counts = labels_count(fd / key, nc)
        r = model.val(data=str(fd / "data.yaml"), split=split, imgsz=imgsz, batch=16, plots=(split == "test"),
                      project=str(RES_DIR), name=f"fold{k}_{split}", exist_ok=True, verbose=False)
        out[split] = metrics_dict(r, names, counts)
        out[split]["images"] = len(list((fd / key / "images").iterdir()))
        log(f"fold {k} {split}: mAP50 {out[split]['mAP50']:.3f}  mAP50-95 {out[split]['mAP50_95']:.3f}")

    # per video on the test set
    tmp = CV / "tmp_pervideo"
    shutil.rmtree(tmp, ignore_errors=True)
    by_vid = defaultdict(list)
    for p in sorted((fd / "test" / "images").iterdir()):
        by_vid[video_of(base_name(p))].append(p)
    out["per_video"] = []
    for v, imgs in sorted(by_vid.items()):
        d = tmp / re.sub(r"[^A-Za-z0-9_-]", "_", v)
        (d / "images").mkdir(parents=True)
        (d / "labels").mkdir(parents=True)
        counts = np.zeros(nc, dtype=np.int64)
        for p in imgs:
            (d / "images" / p.name).symlink_to(p.resolve())
            lp = fd / "test" / "labels" / (p.stem + ".txt")
            if lp.exists():
                (d / "labels" / lp.name).symlink_to(lp.resolve())
                counts += count_labels(lp, nc)
        yml = d / "data.yaml"
        yaml.safe_dump({"path": str(d), "train": "images", "val": "images", "names": names}, open(yml, "w"))
        if counts.sum() == 0:
            out["per_video"].append({"video": v, "images": len(imgs), "mAP50": None, "per_class": {}})
            continue
        r = model.val(data=str(yml), imgsz=imgsz, batch=16, plots=False, verbose=False,
                      project=str(tmp / "_val"), name=d.name, exist_ok=True)
        m = metrics_dict(r, names, counts)
        out["per_video"].append({"video": v, "images": len(imgs), "mAP50": m["mAP50"], "per_class": m["per_class"]})
    shutil.rmtree(tmp, ignore_errors=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    (RES_DIR / f"fold{k}.json").write_text(json.dumps(out, indent=1))
    log(f"fold {k}: results saved to {RES_DIR / f'fold{k}.json'}")


def cmd_eval(a):
    meta = load_meta()
    for k in fold_list(a, meta):
        evaluate_fold(k, meta, a)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def short(n):
    return n.replace("_violation", "").replace("license_plate", "plate").replace("3_person", "3-person") \
        .replace("wrong_side", "wrong-side")


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    return float(np.mean(xs)), (float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0), len(xs)


def paper_reference():
    p = ROOT / "results_table.csv"
    if not p.exists():
        return None
    for r in csv.DictReader(open(p)):
        if r.get("run") == "final_s_aug" and r.get("split") == "test":
            return r
    return None


def cmd_report(a):
    meta = load_meta()
    names = meta["names"]
    K = meta["k"]
    res = {}
    for k in range(K):
        p = RES_DIR / f"fold{k}.json"
        if p.exists():
            res[k] = json.loads(p.read_text())
    if not res:
        raise SystemExit("no evaluated folds yet — run train/eval first")
    folds = sorted(res)
    missing = [k for k in range(K) if k not in res]

    # ---- CSV (one row per fold + mean + std)
    CV.mkdir(exist_ok=True)
    cols = ["fold", "test_videos", "test_images", "val_mAP50", "val_mAP50_95", "P", "R", "mAP50", "mAP50_95"] \
        + [f"AP50_{n}" for n in names] + [f"n_{n}" for n in names] + [f"AP50_95_{n}" for n in names]
    rows = []
    for k in folds:
        t = res[k]["test"]
        rows.append({"fold": k, "test_videos": ", ".join(res[k]["test_videos"]), "test_images": t["images"],
                     "val_mAP50": res[k]["val"]["mAP50"], "val_mAP50_95": res[k]["val"]["mAP50_95"],
                     "P": t["P"], "R": t["R"], "mAP50": t["mAP50"], "mAP50_95": t["mAP50_95"],
                     **{f"AP50_{n}": t["per_class"][n]["AP50"] for n in names},
                     **{f"n_{n}": t["per_class"][n]["n"] for n in names},
                     **{f"AP50_95_{n}": t["per_class"][n]["AP50_95"] for n in names}})
    agg_keys = ["val_mAP50", "val_mAP50_95", "P", "R", "mAP50", "mAP50_95"] + [f"AP50_{n}" for n in names] \
        + [f"AP50_95_{n}" for n in names]
    stats = {key: mean_std([r[key] for r in rows]) for key in agg_keys}
    with open(CV / "cv_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (f"{r[c]:.4f}" if isinstance(r[c], float) else ("" if r[c] is None else r[c])) for c in cols})
        for label, i in (("mean", 0), ("std", 1)):
            w.writerow({"fold": label, **{key: (f"{stats[key][i]:.4f}" if stats[key][i] is not None else "")
                                          for key in agg_keys}})

    # ---- per-video CSV (every video once)
    pv = []
    for k in folds:
        for v in res[k]["per_video"]:
            pv.append((v["video"], k, v))
    pv.sort(key=lambda x: x[0])
    with open(CV / "cv_per_video.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "fold", "images", "mAP50"] + [f"AP50_{n}" for n in names] + [f"n_{n}" for n in names])
        for v, k, d in pv:
            pc = d["per_class"]
            w.writerow([v, k, d["images"], "" if d["mAP50"] is None else f"{d['mAP50']:.4f}"]
                       + [("" if not pc or pc[n]["AP50"] is None else f"{pc[n]['AP50']:.4f}") for n in names]
                       + [(pc[n]["n"] if pc else 0) for n in names])

    # ---- markdown
    f3 = lambda x: "–" if x is None else f"{x:.3f}"
    ms = lambda key: "–" if stats[key][0] is None else f"{stats[key][0]:.3f} ± {stats[key][1]:.3f}"
    best = max(folds, key=lambda k: res[k]["test"]["mAP50"])
    worst = min(folds, key=lambda k: res[k]["test"]["mAP50"])
    md = [f"# 5-fold video-level cross-validation: YOLO26s + aug\n",
          f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Folds: `cv/folds/summary.md`. "
          f"Each fold's best.pt was chosen on its own validation videos and scored once on its test videos.\n"]
    if missing:
        md.append(f"> **Incomplete:** folds {missing} have no results yet. Mean ± std below uses {len(folds)} folds.\n")
    md += ["## Test results per fold (AP50 per class, instances in brackets)\n",
           "| Fold | Test videos | Imgs | val mAP50 | **mAP50** | **mAP50-95** | P | R | "
           + " | ".join(short(n) for n in names) + " |",
           "|---|---|---|---|---|---|---|---|" + "---|" * len(names)]
    for r in rows:
        k = r["fold"]
        tag = " (highest)" if k == best and len(folds) > 1 else " (lowest)" if k == worst and len(folds) > 1 else ""
        md.append(f"| {k}{tag} | {r['test_videos']} | {r['test_images']} | {f3(r['val_mAP50'])} | "
                  f"**{f3(r['mAP50'])}** | **{f3(r['mAP50_95'])}** | {f3(r['P'])} | {f3(r['R'])} | "
                  + " | ".join("–" if r[f"n_{n}"] == 0 else f"{f3(r[f'AP50_{n}'])} ({r[f'n_{n}']})" for n in names)
                  + " |")
    md.append(f"| **Mean ± std** | | | {ms('val_mAP50')} | **{ms('mAP50')}** | **{ms('mAP50_95')}** | {ms('P')} | "
              f"{ms('R')} | " + " | ".join(ms(f"AP50_{n}") for n in names) + " |")
    ref = paper_reference()
    if ref:
        md.append(f"| Paper split (single run) | 13, 15, 7, IMG_0054, IMG_1080 | | | {float(ref['mAP50']):.3f} | "
                  f"{float(ref['mAP50_95']):.3f} | {float(ref['P']):.3f} | {float(ref['R']):.3f} | "
                  + " | ".join(f"{float(ref.get(f'AP50_{n}', 'nan')):.3f}" for n in names) + " |")
    md.append("\nStd is the sample standard deviation across folds. A class's mean only uses folds whose "
              "test set contains that class. mAP50 per fold averages only over the classes present in that fold.\n")
    if len(folds) > 1:
        spread = res[best]["test"]["mAP50"] - res[worst]["test"]["mAP50"]
        md += ["## Reading this\n",
               f"- Test mAP50 ranges from {res[worst]['test']['mAP50']:.3f} (fold {worst}) to "
               f"{res[best]['test']['mAP50']:.3f} (fold {best}), a spread of {spread:.3f}. "
               "The folds use the same model and training recipe. The only difference is which videos "
               "are held out, so the spread shows how much one test split's score depends on its videos.",
               f"- **{ms('mAP50')} mAP50 / {ms('mAP50_95')} mAP50-95** is the estimate to report for unseen videos. "
               "It uses all 40 videos as test data once.",
               "- The highest-scoring fold is useful to show which videos are easy, but it is not a better model. "
               "Picking it because of its test score would make that score optimistic. To pick a single model "
               "to deploy, compare the folds on **val mAP50** instead, or retrain on all videos.\n"]
    md += ["## Per video (each video is tested exactly once)\n", "See `cv/cv_per_video.csv`.\n",
           "| Video | Fold | Imgs | mAP50 | " + " | ".join(short(n) for n in names) + " |",
           "|---|---|---|---|" + "---|" * len(names)]
    for v, k, d in pv:
        pc = d["per_class"]
        md.append(f"| {v} | {k} | {d['images']} | {f3(d['mAP50'])} | "
                  + " | ".join("–" if not pc or pc[n]["n"] == 0 else f"{f3(pc[n]['AP50'])} ({pc[n]['n']})" for n in names)
                  + " |")
    (CV / "cv_results.md").write_text("\n".join(md) + "\n")

    # ---- LaTeX (IEEE, booktabs-free so it drops into the existing paper)
    tex = ["% 5-fold video-level CV, YOLO26s + aug. Generated by cv_experiment.py",
           "\\begin{table}[t]", "\\centering", "\\footnotesize",
           "\\caption{Five-fold video-level cross-validation (YOLO26s + aug), test AP50 per fold.}",
           "\\label{tab:cv}", "\\setlength{\\tabcolsep}{3pt}",
           "\\begin{tabular}{c" + "c" * (2 + len(names)) + "}", "\\hline",
           "Fold & mAP50 & mAP50-95 & " + " & ".join(
               {"3_person_violation": "3P", "footpath_violation": "FP", "helmet_violation": "HV",
                "license_plate": "LP", "wrong_side_violation": "WS"}.get(n, short(n)) for n in names) + " \\\\", "\\hline"]
    for r in rows:
        tex.append(f"{r['fold'] + 1} & {f3(r['mAP50'])} & {f3(r['mAP50_95'])} & "
                   + " & ".join("--" if r[f"n_{n}"] == 0 else f3(r[f"AP50_{n}"]) for n in names) + " \\\\")
    tex += ["\\hline",
            "Mean & " + " & ".join(
                (f"{stats[key][0]:.3f}$\\pm${stats[key][1]:.3f}" if stats[key][0] is not None else "--")
                for key in ["mAP50", "mAP50_95"] + [f"AP50_{n}" for n in names]) + " \\\\",
            "\\hline", "\\end{tabular}", "\\end{table}"]
    (CV / "cv_table.tex").write_text("\n".join(tex) + "\n")

    print("\n".join(md[:6 + len(rows) + 2]))
    log(f"wrote {CV/'cv_results.md'}, cv_results.csv, cv_per_video.csv, cv_table.tex")


# --------------------------------------------------------------------------- #
def cmd_status(a):
    meta = load_meta()
    print(f"folds built {meta['created']} · {meta['k']} folds")
    for k in range(meta["k"]):
        rd = run_dir(k)
        state = "not started"
        extra = ""
        rc = rd / "results.csv"
        if rc.exists():
            rowsr = list(csv.DictReader(open(rc)))
            if rowsr:
                last = {kk.strip(): v for kk, v in rowsr[-1].items()}
                best_m = max(float({kk.strip(): v for kk, v in r.items()}.get("metrics/mAP50-95(B)", 0)) for r in rowsr)
                extra = (f"epoch {int(float(last['epoch']))}, {float(last.get('time', 0)) / 60:.0f} min, "
                         f"val mAP50 {float(last['metrics/mAP50(B)']):.3f}, best val mAP50-95 {best_m:.3f}")
            state = "training"
        if (rd / "CV_DONE").exists():
            state = "trained"
        if (RES_DIR / f"fold{k}.json").exists():
            t = json.loads((RES_DIR / f"fold{k}.json").read_text())["test"]
            state = "done"
            extra = f"test mAP50 {t['mAP50']:.3f}, mAP50-95 {t['mAP50_95']:.3f}"
        tv = ", ".join(meta["folds"][k]["test"].get("display", meta["folds"][k]["test"]["videos"]))
        print(f"  fold {k}: {state:<11} {extra}\n          test videos: {tv}")
    lock = CV / ".lock"
    if lock.exists():
        pid = lock.read_text().strip()
        alive = pid.isdigit() and Path(f"/proc/{pid}").exists()
        print(f"\nbackground job {'running' if alive else 'NOT running'} (pid {pid}); log: {CV / 'cv.log'}")


def with_lock(fn):
    def wrapped(a):
        CV.mkdir(exist_ok=True)
        lock = CV / ".lock"
        if lock.exists():
            pid = lock.read_text().strip()
            if pid.isdigit() and Path(f"/proc/{pid}").exists() and int(pid) != os.getpid():
                raise SystemExit(f"another cross-validation job is already running (pid {pid}). "
                                 f"Check it with: python cv_experiment.py status")
        lock.write_text(str(os.getpid()))
        try:
            return fn(a)
        finally:
            if lock.exists() and lock.read_text().strip() == str(os.getpid()):
                lock.unlink()
    return wrapped


def cmd_all(a):
    if not (FOLDS_DIR / "folds.json").exists():
        cmd_folds(a)
    cmd_train(a)
    cmd_eval_missing(a)
    cmd_report(a)


def cmd_eval_missing(a):
    meta = load_meta()
    for k in fold_list(a, meta):
        if (run_dir(k) / "CV_DONE").exists() and not (RES_DIR / f"fold{k}.json").exists():
            eval_in_child(k, a)


# --------------------------------------------------------------------------- #
# Final model on all 40 videos + scoring on new footage
# --------------------------------------------------------------------------- #
FINAL_RUN = "final_all_s_aug"
FINAL_DATA = CV / "final_all"


def cv_best_epochs():
    """Best epoch of every finished fold (Ultralytics fitness = 0.1·mAP50 + 0.9·mAP50-95)."""
    out = []
    for rc in sorted(RUNS.glob(f"{RUN_PREFIX}*/results.csv")):
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(rc))]
        if rows:
            fit = lambda r: 0.1 * float(r["metrics/mAP50(B)"]) + 0.9 * float(r["metrics/mAP50-95(B)"])
            out.append(int(float(max(rows, key=fit)["epoch"])))
    return out


def build_all_dataset(src: Path):
    names = read_names(src)
    if FINAL_DATA.exists():
        shutil.rmtree(FINAL_DATA)
    (FINAL_DATA / "train" / "images").mkdir(parents=True)
    (FINAL_DATA / "train" / "labels").mkdir(parents=True)
    n = 0
    for s in SPLITS:
        d = src / s / "images"
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                (FINAL_DATA / "train" / "images" / p.name).symlink_to(p.resolve())
                lp = lbl_path(p)
                if lp.exists():
                    (FINAL_DATA / "train" / "labels" / lp.name).symlink_to(lp.resolve())
                n += 1
    # Ultralytics needs a val entry; there are no held-out videos, so it points at train
    # and the per-epoch validation is switched off (val=False).
    yaml.safe_dump({"path": str(FINAL_DATA), "train": "train/images", "val": "train/images",
                    "nc": len(names), "names": names}, open(FINAL_DATA / "data.yaml", "w"), sort_keys=False)
    return n


def cmd_final(a):
    rd = RUNS / FINAL_RUN
    if (rd / "CV_DONE").exists():
        log(f"final model already trained: {rd / 'weights' / 'last.pt'}")
        return
    from ultralytics import YOLO
    last = rd / "weights" / "last.pt"
    if last.exists():
        log(f"resuming final model from {last}")
        try:
            YOLO(str(last)).train(resume=True)
        except Exception as e:
            if "nothing to resume" not in str(e):
                raise
    else:
        src = Path(a.src).expanduser().resolve()
        n = build_all_dataset(src)
        best = cv_best_epochs()
        epochs = a.epochs or (int(np.median(best)) if best else 40)
        log(f"training on all {n} images from {src.name}; fold best epochs {best} → {epochs} fixed epochs, "
            "no early stopping, no validation")
        model = a.model or (str(ROOT / DEFAULT_MODEL) if (ROOT / DEFAULT_MODEL).exists() else DEFAULT_MODEL)
        cfg = train_overrides(a)
        cfg.update(epochs=epochs, patience=0, val=False)
        YOLO(model).train(data=str(FINAL_DATA / "data.yaml"), project=str(RUNS), name=FINAL_RUN,
                          exist_ok=True, **cfg)
    # With no validation "best" means nothing: make best.pt the last epoch so tools that
    # look for best.pt (the dashboard) use the right weights.
    if last.exists():
        shutil.copy2(last, rd / "weights" / "best.pt")
    (rd / "CV_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    (rd / "README.txt").write_text(
        "Final model: YOLO26s + aug trained on all 40 videos with a fixed number of epochs\n"
        "(median best epoch of the 5 CV folds), no validation. best.pt is a copy of last.pt.\n"
        "Its expected performance is the cross-validation result in cv/cv_results.md.\n")
    log(f"final model done: {rd / 'weights' / 'last.pt'} (also copied to best.pt)")


def cmd_test(a):
    """Score a model on a separately labelled set of NEW footage (e.g. the Pixel clips)."""
    from PIL import Image
    from ultralytics import YOLO
    root = Path(a.data).expanduser().resolve()
    weights = Path(a.weights).expanduser() if a.weights else RUNS / FINAL_RUN / "weights" / "last.pt"
    if not weights.exists():
        raise SystemExit(f"no weights at {weights}")
    model = YOLO(str(weights))
    mnames = [model.names[k] for k in sorted(model.names)]
    dnames = mnames
    if (root / "data.yaml").exists():
        dn = yaml.safe_load(open(root / "data.yaml")).get("names")
        if dn:
            dnames = [dn[k] for k in sorted(dn)] if isinstance(dn, dict) else list(dn)
    remap = {i: mnames.index(n) for i, n in enumerate(dnames) if n in mnames}
    unknown = [n for n in dnames if n not in mnames]
    if unknown:
        log(f"warning: classes not known to the model are ignored: {unknown}")
    plate = mnames.index("license_plate") if "license_plate" in mnames else None

    # every image under root (Roboflow may have split the export into train/valid/test —
    # for a test set they all count)
    imgs = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS and p.parent.name == "images")
    if not imgs:
        raise SystemExit(f"no */images/*.jpg under {root}")
    tag = a.name or root.name
    tmp = CV / "tmp_test" / tag
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "images").mkdir(parents=True)
    (tmp / "labels").mkdir(parents=True)
    nc = len(mnames)
    masked = dropped = 0
    per_img = {}
    for p in imgs:
        lp = p.parent.parent / "labels" / (p.stem + ".txt")
        lines = lp.read_text().splitlines() if lp.exists() else []
        with Image.open(p) as im:
            W, H = im.size
        s = a.imgsz / max(W, H)
        keep, small = [], []
        for ln in lines:
            q = ln.split()
            if len(q) < 5 or int(float(q[0])) not in remap:
                continue
            c = remap[int(float(q[0]))]
            x, y, w, h = map(float, q[1:5])
            if c == plate and a.min_plate_w and w * W * s < a.min_plate_w:
                small.append((x, y, w, h))           # same rule as the training data
                continue
            keep.append(f"{c} {x} {y} {w} {h}")
        name = p.name
        if small:
            arr = np.array(Image.open(p).convert("RGB"))
            for x, y, w, h in small:
                arr[max(0, int((y - h / 2) * H) - 1):min(H, int((y + h / 2) * H) + 2),
                    max(0, int((x - w / 2) * W) - 1):min(W, int((x + w / 2) * W) + 2)] = 114
            Image.fromarray(arr).save(tmp / "images" / name, quality=95)
            masked += 1
            dropped += len(small)
        else:
            (tmp / "images" / name).symlink_to(p)
        (tmp / "labels" / (p.stem + ".txt")).write_text("\n".join(keep) + ("\n" if keep else ""))
        per_img[name] = keep
    log(f"{len(imgs)} images from {root.name}; {dropped} plates narrower than {a.min_plate_w}px "
        f"(at {a.imgsz}) masked in {masked} images, as in training")

    yml = tmp / "data.yaml"
    yaml.safe_dump({"path": str(tmp), "train": "images", "val": "images", "names": mnames}, open(yml, "w"))
    counts = np.zeros(nc, dtype=np.int64)
    for keep in per_img.values():
        for ln in keep:
            counts[int(ln.split()[0])] += 1
    r = model.val(data=str(yml), imgsz=a.imgsz, batch=16, plots=True, verbose=False,
                  project=str(CV / "test_results"), name=tag, exist_ok=True)
    out = {"data": str(root), "weights": str(weights), "images": len(imgs), "evaluated": time.strftime("%Y-%m-%d %H:%M"),
           "overall": metrics_dict(r, mnames, counts), "per_video": []}

    # per video
    by_vid = defaultdict(list)
    for p in sorted((tmp / "images").iterdir()):
        by_vid[video_of(base_name(p))].append(p)
    for v, vimgs in sorted(by_vid.items()):
        d = tmp / "_videos" / re.sub(r"[^A-Za-z0-9_-]", "_", v)
        (d / "images").mkdir(parents=True)
        (d / "labels").mkdir(parents=True)
        vc = np.zeros(nc, dtype=np.int64)
        for p in vimgs:
            (d / "images" / p.name).symlink_to(p.resolve())
            txt = tmp / "labels" / (p.stem + ".txt")
            shutil.copy2(txt, d / "labels" / txt.name)
            vc += count_labels(txt, nc)
        if vc.sum() == 0:
            continue
        yaml.safe_dump({"path": str(d), "train": "images", "val": "images", "names": mnames}, open(d / "data.yaml", "w"))
        rv = model.val(data=str(d / "data.yaml"), imgsz=a.imgsz, batch=16, plots=False, verbose=False,
                       project=str(tmp / "_val"), name=d.name, exist_ok=True)
        m = metrics_dict(rv, mnames, vc)
        out["per_video"].append({"video": v, "images": len(vimgs), "mAP50": m["mAP50"], "per_class": m["per_class"]})

    # report
    f3 = lambda x: "–" if x is None else f"{x:.3f}"
    o = out["overall"]
    md = [f"# Test on new footage: {tag}\n",
          f"Model `{weights.relative_to(ROOT) if weights.is_relative_to(ROOT) else weights}` · {len(imgs)} images · "
          f"imgsz {a.imgsz} · evaluated {out['evaluated']}\n",
          "| | mAP50 | mAP50-95 | P | R | " + " | ".join(short(n) for n in mnames) + " |",
          "|---|---|---|---|---|" + "---|" * nc,
          f"| **This test set** | **{o['mAP50']:.3f}** | **{o['mAP50_95']:.3f}** | {o['P']:.3f} | {o['R']:.3f} | "
          + " | ".join("–" if o["per_class"][n]["n"] == 0 else f"{f3(o['per_class'][n]['AP50'])} ({o['per_class'][n]['n']})"
                       for n in mnames) + " |"]
    cvp = CV / "cv_results.csv"
    if cvp.exists():
        mean = next((row for row in csv.DictReader(open(cvp)) if row["fold"] == "mean"), None)
        if mean:
            md.append(f"| 5-fold CV mean (for comparison) | {float(mean['mAP50']):.3f} | {float(mean['mAP50_95']):.3f} | "
                      f"{float(mean['P']):.3f} | {float(mean['R']):.3f} | "
                      + " | ".join(f3(float(mean[f'AP50_{n}'])) if mean.get(f'AP50_{n}') else "–" for n in mnames) + " |")
    md.append("\nInstances in brackets. A class with only a few instances gives a noisy AP.\n")
    md += ["## Per video\n", "| Video | Imgs | mAP50 | " + " | ".join(short(n) for n in mnames) + " |",
           "|---|---|---|" + "---|" * nc]
    for v in out["per_video"]:
        pc = v["per_class"]
        md.append(f"| {v['video']} | {v['images']} | {f3(v['mAP50'])} | "
                  + " | ".join("–" if pc[n]["n"] == 0 else f"{f3(pc[n]['AP50'])} ({pc[n]['n']})" for n in mnames) + " |")
    (CV / f"test_{tag}.md").write_text("\n".join(md) + "\n")
    (CV / f"test_{tag}.json").write_text(json.dumps(out, indent=1))
    shutil.rmtree(tmp, ignore_errors=True)
    print("\n".join(md))
    log(f"wrote {CV / f'test_{tag}.md'} (plots in {CV / 'test_results' / tag})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_fold_opts(p):
        p.add_argument("--src", default=str(DEFAULT_SRC), help="dataset with train/valid/test (default TV_R-final-filtered)")
        p.add_argument("--location-map", default=None, help="video→group file (default release/splits/locations.txt)")
        p.add_argument("--k", type=int, default=5)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--val-frac", type=float, default=0.15, help="validation share of ALL images")
        p.add_argument("--hash-thresh", type=int, default=20)
        p.add_argument("--search", type=int, default=20000, help="random fold assignments to try")
        p.add_argument("--night-videos", nargs="*", default=NIGHT_VIDEOS)
        p.add_argument("--force", action="store_true", help="rebuild folds (only before training started)")

    def add_train_opts(p):
        p.add_argument("--folds", type=int, nargs="*", help="only these folds (default: all)")
        p.add_argument("--model", default=None, help=f"starting weights (default {DEFAULT_MODEL})")
        p.add_argument("--epochs", type=int)
        p.add_argument("--patience", type=int)
        p.add_argument("--batch", type=int)
        p.add_argument("--workers", type=int)
        p.add_argument("--imgsz", type=int)
        p.add_argument("--no-cache", action="store_true", help="do not cache images in RAM")
        p.add_argument("--no-eval", action="store_true", help="train only; score later with `eval`")

    p = sub.add_parser("folds", help="build the folds")
    add_fold_opts(p)
    p.set_defaults(fn=cmd_folds)
    p = sub.add_parser("train", help="train all unfinished folds")
    add_train_opts(p)
    p.set_defaults(fn=with_lock(cmd_train))
    p = sub.add_parser("_train_one")
    p.add_argument("--fold", type=int, required=True)
    add_train_opts(p)
    p.set_defaults(fn=cmd_train_one)
    p = sub.add_parser("eval", help="score trained folds on their test sets")
    p.add_argument("--folds", type=int, nargs="*")
    p.add_argument("--imgsz", type=int)
    p.set_defaults(fn=cmd_eval)
    p = sub.add_parser("report", help="write the result tables")
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("status", help="show progress")
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser("final", help="train the final model on all 40 videos (fixed epochs, no validation)")
    p.add_argument("--src", default=str(DEFAULT_SRC))
    add_train_opts(p)
    p.set_defaults(fn=with_lock(cmd_final))
    p = sub.add_parser("test", help="score a model on a labelled set of new footage")
    p.add_argument("--data", required=True, help="folder with images/ + labels/ (e.g. a Roboflow YOLO export)")
    p.add_argument("--weights", default=None, help=f"default runs/{FINAL_RUN}/weights/last.pt")
    p.add_argument("--name", default=None, help="name for the report (default: folder name)")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--min-plate-w", type=float, default=16,
                   help="mask plates narrower than this at imgsz, like the training data (0 = keep all)")
    p.set_defaults(fn=cmd_test)
    p = sub.add_parser("all", help="folds + train + eval + report")
    add_fold_opts(p)
    add_train_opts(p)
    p.set_defaults(fn=with_lock(cmd_all))
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()

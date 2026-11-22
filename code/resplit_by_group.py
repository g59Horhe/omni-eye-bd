#!/usr/bin/env python3
"""
Leak-free, class-balanced re-split of a YOLO dataset (Omni-Eye / TV_R).

Images that belong together never end up in different splits:
  1. same video       : filename "<video>_MOV-<frame>", e.g. IMG_0054_MOV-0003, 10_MOV-0002
  2. same location    : optional --location-map file ("video location" per line)
  3. similar pixels   : dHash distance <= --hash-thresh (near-duplicate frames; also
                        reveals different videos that look like the same place)
Then it tries --search random assignments of these groups to train/valid/test and
keeps the one whose per-class instance counts best match --ratios.

Roboflow-augmented copies ("<base>_jpg.rf.<hash>.jpg"):
  train gets ALL copies; valid/test get ONE copy per source (an original valid/test
  copy when available, otherwise an augmented train copy -> reported).
  Best: run on a Roboflow export with NO augmentation, so this count is 0.

Usage:
  python resplit_by_group.py --src ~/FYDP/TV_R-3 --dst ~/FYDP/TV_R-4 --dry-run
  python resplit_by_group.py --src ~/FYDP/TV_R-3 --dst ~/FYDP/TV_R-4 --location-map locations.txt --dry-run

locations.txt example (videos sharing a footbridge get the same label):
  IMG_0054  bridgeA
  IMG_0055  bridgeA
  10        bridgeB
"""
import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NAME_RE = re.compile(r"^(.+?)_([A-Za-z0-9]+-\d+)$")
HASH = 16
SPLITS = ("train", "valid", "test")


def base_name(p):
    return re.sub(r"_(jpg|jpeg|png)\.rf\..*$", "", p.name)


def lbl_path(img):
    return img.parent.parent / "labels" / (img.stem + ".txt")


def dhash(img):
    g = np.asarray(img.convert("L").resize((HASH + 1, HASH), Image.BILINEAR), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


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


def load_location_map(path):
    m = {}
    for ln in Path(path).expanduser().read_text().splitlines():
        ln = ln.split("#")[0].replace(",", " ").split()
        if len(ln) >= 2:
            m[ln[0]] = ln[1]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--group-by", choices=["video", "prefix", "none"], default="video",
                    help="'prefix' is an alias of 'video'")
    ap.add_argument("--location-map", default=None)
    ap.add_argument("--hash-thresh", type=int, default=20)
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.15, 0.15])
    ap.add_argument("--search", type=int, default=3000, help="random assignments to try")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    src, dst = Path(a.src).expanduser(), Path(a.dst).expanduser()
    data = yaml.safe_load(open(src / "data.yaml"))
    names = data["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    nc = len(names)

    # ---------- collect sources
    bases = defaultdict(lambda: defaultdict(list))
    for s in SPLITS:
        d = src / s / "images"
        if d.exists():
            for p in d.iterdir():
                if p.suffix.lower() in IMG_EXTS:
                    bases[base_name(p)][s].append(p)
    keys = sorted(bases)
    n = len(keys)

    def rep(k):
        for s in ("valid", "test", "train"):
            if bases[k][s]:
                return sorted(bases[k][s])[0]

    # ---------- video / location of each source
    video_of = {}
    for k in keys:
        m = NAME_RE.match(k)
        video_of[k] = m.group(1) if m else None
    loc_map = load_location_map(a.location_map) if a.location_map else {}
    loc_of = {k: (loc_map.get(v, v) if v else None) for k, v in video_of.items()}
    vids = sorted({v for v in video_of.values() if v})
    unparsed = [k for k in keys if video_of[k] is None]
    print(f"{n} source images | {len(vids)} videos | {len(set(l for l in loc_of.values() if l))} locations "
          f"| {len(unparsed)} unrecognised names")
    if unparsed:
        print("   unrecognised examples:", unparsed[:5])
    if loc_map:
        missing = [v for v in vids if v not in loc_map]
        if missing:
            print(f"   videos not in location map (treated as their own location): {missing}")

    # ---------- class counts per source (one copy)
    C = np.zeros((n, nc), dtype=np.int64)
    for i, k in enumerate(keys):
        lp = lbl_path(rep(k))
        if lp.exists():
            for ln in lp.read_text().splitlines():
                parts = ln.split()
                if parts:
                    c = int(float(parts[0]))
                    if 0 <= c < nc:
                        C[i, c] += 1

    # ---------- hashing
    print("hashing...")
    H, HF = [], []
    for k in keys:
        with Image.open(rep(k)) as im:
            H.append(dhash(im))
            HF.append(dhash(im.transpose(Image.FLIP_LEFT_RIGHT)))
    H, HF = np.array(H), np.array(HF)

    # ---------- grouping
    uf = UF(n)
    if a.group_by != "none":
        first = {}
        for i, k in enumerate(keys):
            g = loc_of[k]
            if g is not None:
                if g in first:
                    uf.union(i, first[g])
                else:
                    first[g] = i
    cross_video = defaultdict(int)
    for i in range(n):
        d = np.minimum((H[i + 1:] != H[i]).sum(1), (HF[i + 1:] != H[i]).sum(1))
        for j in np.nonzero(d <= a.hash_thresh)[0]:
            j = i + 1 + int(j)
            uf.union(i, j)
            vi, vj = video_of[keys[i]], video_of[keys[j]]
            if vi and vj and vi != vj:
                cross_video[tuple(sorted((vi, vj)))] += 1

    if cross_video:
        print("\nDifferent videos with near-identical frames (likely SAME location):")
        for (v1, v2), c in sorted(cross_video.items(), key=lambda x: -x[1])[:30]:
            print(f"   {v1:>10} ~ {v2:<10} {c} frame pairs")

    root_of = [uf.find(i) for i in range(n)]
    gid = {r: j for j, r in enumerate(sorted(set(root_of)))}
    g_of = np.array([gid[r] for r in root_of])
    G = len(gid)
    GS = np.bincount(g_of, minlength=G).astype(float)
    GC = np.zeros((G, nc))
    np.add.at(GC, g_of, C)
    print(f"\n{G} groups | largest (sources): {sorted(GS.astype(int).tolist(), reverse=True)[:8]}")
    if GS.max() > a.ratios[1] * n:
        print("   note: largest group is bigger than a whole valid split; it will land in train")

    # merged-video summary
    vids_in_group = defaultdict(set)
    for i, k in enumerate(keys):
        if video_of[k]:
            vids_in_group[g_of[i]].add(video_of[k])
    multi = [sorted(v) for v in vids_in_group.values() if len(v) > 1]
    if multi:
        print("   groups spanning several videos:", multi)

    # ---------- search for the most class-balanced assignment
    rng = np.random.default_rng(a.seed)
    R = np.array(a.ratios)
    tot = C.sum(0).astype(float)
    exp_c = R[:, None] * tot[None, :]
    exp_n = R * n

    def score(asg):
        S = np.zeros((3, nc)); Z = np.zeros(3)
        np.add.at(S, asg, GC); np.add.at(Z, asg, GS)
        cls_err = (((S - exp_c) / np.maximum(exp_c, 1)) ** 2).sum()
        size_err = (((Z - exp_n) / exp_n) ** 2).sum()
        return cls_err + 2 * size_err, S, Z

    best = None
    for _ in range(max(1, a.search)):
        count = np.zeros(3)
        asg = np.empty(G, dtype=int)
        for g in rng.permutation(G):
            s = int(np.argmax((exp_n - count) / exp_n + rng.uniform(0, 0.25, 3)))
            asg[g] = s
            count[s] += GS[g]
        sc, S, Z = score(asg)
        if best is None or sc < best[0]:
            best = (sc, asg, S, Z)
    sc, asg, S, Z = best
    assign = {k: SPLITS[asg[g_of[i]]] for i, k in enumerate(keys)}

    # ---------- files per split
    out_files = {s: [] for s in SPLITS}
    aug_in_eval = 0
    for k, s in assign.items():
        if s == "train":
            files = [p for ss in SPLITS for p in bases[k][ss]]
        else:
            orig = bases[k]["valid"] + bases[k]["test"]
            files = [sorted(orig)[0]] if orig else [sorted(bases[k]["train"])[0]]
            aug_in_eval += 0 if (orig or len(bases[k]["train"]) == 1) else 1
        out_files[s].extend(files)

    # ---------- report
    print(f"\nbest balance score: {sc:.3f} (lower is better) after {a.search} tries")
    print(f"{'split':>6} | {'sources':>7} | {'files':>6} | " + " | ".join(f"{c[:12]:>12}" for c in names))
    for i, s in enumerate(SPLITS):
        print(f"{s:>6} | {int(Z[i]):7d} | {len(out_files[s]):6d} | "
              + " | ".join(f"{int(v):5d} ({100*v/max(t,1):3.0f}%)" for v, t in zip(S[i], tot)))
    print("(class counts use one copy per source; % = share of that class)")
    print(f"\nvalid/test images that had to come from augmented train copies: {aug_in_eval}")
    print("\nVideos per split (source images in brackets):")
    for s in SPLITS:
        vc = defaultdict(int)
        for k, ss in assign.items():
            if ss == s and video_of[k]:
                vc[video_of[k]] += 1
        print(f"  {s:>5} ({len(vc)}): " + ", ".join(f"{v}[{c}]" for v, c in sorted(vc.items())))
    shared = [v for v in vids if len({assign[k] for k in keys if video_of[k] == v}) > 1]
    print(f"\nvideos appearing in more than one split: {shared if shared else 'none'}")

    if a.dry_run:
        print("\nDry run only, nothing written.")
        return

    for s in SPLITS:
        (dst / s / "images").mkdir(parents=True, exist_ok=True)
        (dst / s / "labels").mkdir(parents=True, exist_ok=True)
        for p in out_files[s]:
            shutil.copy2(p, dst / s / "images" / p.name)
            lp = lbl_path(p)
            if lp.exists():
                shutil.copy2(lp, dst / s / "labels" / lp.name)
    new = dict(data)
    new.update(path=str(dst), train="train/images", val="valid/images", test="test/images")
    yaml.safe_dump(new, open(dst / "data.yaml", "w"), sort_keys=False)
    with open(dst / "split_groups.txt", "w") as f:
        for k in keys:
            f.write(f"{k}\t{video_of[k]}\t{loc_of[k]}\t{assign[k]}\n")
    print(f"\nWrote {dst} (+ split_groups.txt: source, video, location, split)")


if __name__ == "__main__":
    main()

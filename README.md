# Omni-Eye: annotations, split definitions, label-audit logs and code

Supplementary material for *"Omni-Eye: Traffic Violation Detection and the Hidden
Cost of Frame-Level Data Splits"* (ICCIT 2026).

This release contains everything needed to **reproduce our splits, verify every
label change we made, and re-score our models**: all annotations, the video-level
split, the cross-validation and leave-one-site-out folds, the label-audit logs, the
code and the result tables. It does not contain the video frames themselves (see
*Images* below).

---

## `annotations/` — all labels (YOLO format)

One `.txt` per frame, named after the frame (`<video>_<MOV|mp4>-<frame>_jpg.rf.<hash>.txt`),
in `train/`, `valid/` and `test/` folders following our video-level split. Each line is
`class x_center y_center width height` (or a polygon `class x1 y1 x2 y2 ...`), with
coordinates as fractions of image width and height. The files contain no image data.

Classes: `0` 3_person_violation, `1` footpath_violation, `2` helmet_violation,
`3` license_plate, `4` wrong_side_violation.

| Folder | What it is |
|---|---|
| `labels/` | Final labels after the two-stage plate audit (2,843 files). Use these to compare with other work. |
| `labels_masked/` | The same with plates narrower than 16 px removed (`code/filter_small_plates.py` also grey-fills those regions in the images). This is the version the paper's main results are scored on. |

---

## `splits/` — how the data was divided

| File | What it is |
|---|---|
| `split_groups.txt` | One row per source frame: `source_name`, `video`, `location_group`, `split`. 2,843 rows. This is the authoritative record of our train/valid/test assignment. |
| `locations.txt` | Videos known to share a recording session, grouped so they cannot be split apart. Covers the nine re-shared WhatsApp clips. |
| `data.yaml` | Ultralytics dataset config for the final split. |
| `data_filtered.yaml` | Same, for the copy with plates narrower than 16 px masked. |
| `cv_folds.csv` | Five-fold video-level cross-validation: for each image, its video, group and split in each fold. `cv_folds_summary.md` gives per-fold counts. |
| `site_map.csv` | Footbridge and leave-one-site-out group (A, B or C) of every video. `site_summary.md` gives counts per group. |
| `loso_folds.md` | Leave-one-site-out folds: the validation and held-out videos for each site group. |

Split summary (instances per class, after the label audit):

| Split | Videos | Images | helmet | 3-person | footpath | wrong-side | plate |
|---|---|---|---|---|---|---|---|
| train | 29 | 1,899 | 903 | 233 | 362 | 1,002 | 1,377 |
| valid | 6 | 504 | 328 | 43 | 84 | 195 | 319 |
| test | 5 | 440 | 190 | 47 | 83 | 155 | 356 |

No video appears in more than one split. Valid videos: 17, 19, 5, 9, IMG_0055,
IMG_1079. Test videos: 13, 15, 7, IMG_0054, IMG_1080.

The fixed split separates *videos*, not *sites*: different clips from the same
footbridge can appear in train and test. The leave-one-site-out folds (`site_map.csv`,
`loso_folds.md`) test unseen locations: site A is the Merul Badda footbridge, site B
Rampura, and group C combines the three smaller sites, Notunbazar (9 daytime clips),
Moddho Badda (2 night clips) and Link Road (1 night clip). `results/c_subsets.csv`
splits the group C result by footbridge.

---

## `label_audit/` — the 704 recovered plate labels

License plates had been annotated inconsistently (0 to 4.9 plates per image
across videos from the same fixed camera). We recovered missing plates in two
human-verified rounds. Every proposed box was reviewed by a person; only
confirmed boxes were added.

| File | What it is |
|---|---|
| `candidates_round1_conf025.csv` | 283 plate boxes proposed by the model trained on the original split, confidence ≥ 0.25. Columns: `idx, source_image, file, video, conf, x1, y1, x2, y2, width_px@imgsz`. |
| `added_round1_self_audit.csv` | The 238 of those a reviewer accepted. 45 were rejected (plate cut off at the image border, night-time box on a light blob, box on painted text or a non-plate object). |
| `candidates_round2_cross.csv` | 521 boxes proposed by a **separate auditor model** trained only on the 23 well-labelled videos (≥ 0.3 plates/image) and applied to the 17 sparsely labelled videos it had never seen. |
| `added_round2_cross_audit.csv` | The 466 accepted. 55 rejected, mostly night motorcycles where only a tail-light blob was visible. |

Why the second round used a separate model: a detector trained on frames whose
plates were left unlabelled learns to suppress exactly those plates, so it
cannot find them. The cross-audit model had never seen the videos it inspected,
and recovered 62% of the plates already labelled there — evidence that it
transfers rather than memorises.

Total added: **704 labels**, roughly a 40% increase over the original ~1,720.

---

## `results/`

| File | What it is |
|---|---|
| `results_table.csv` | Six runs × two splits, with P, R, mAP50, mAP50-95, per-class AP50 and AP50-95, parameter counts and inference time. Source for Tables IV and V. |
| `results_table.md` | The same, rendered. |
| `per_video_results.csv` | Per-video, per-class AP50 for the final model on the test split. Source for Table VII. Re-run `code/per_video_eval.py` with `--splits valid train` for the other splits. |
| `seed_results.csv` | Every detector retrained with seeds 1 and 2 (seed 0 is in `results_table.csv`): validation and test mAP and per-class AP50. |
| `bootstrap_ci.md` | 95% bootstrap intervals (2,000 resamples of images or of whole videos) for each detector, and paired differences between detectors. |
| `cv_results.csv`, `cv_rescored.md` | Five-fold cross-validation results per fold and per class. |
| `loso_results.csv` | Leave-one-site-out results for each held-out site group. |
| `c_subsets.csv` | The group C result split by footbridge: Notunbazar (day) and Link Road + Moddho Badda (night). |
| `unmasked_eval.md` | Scores against `labels/` instead of `labels_masked/` (plate AP without masking). |

All scores use Ultralytics 8.4.62. Newer versions score the same weights about 0.02
mAP50 higher, so compare like with like.

---

## `code/`

The curation pipeline, in the order it was applied.

| Script | Purpose |
|---|---|
| `check_leakage.py` | Detects cross-split leakage: filename grouping plus a 256-bit dHash near-duplicate search, flip-aware. Saves side-by-side pairs. |
| `resplit_by_group.py` | Leak-free, class-balanced re-split. Groups frames by source video, merges near-duplicate and re-shared clips, then searches assignments for the one closest to 70/15/15 per class. |
| `plate_threshold_check.py` | Measures plate sizes in pixels at training resolution and saves crop grids per size bucket, to choose the readability threshold. |
| `filter_small_plates.py` | Masks plates below the threshold: label removed **and** the region grey-filled, so the model is not taught that a visible plate is background. |
| `audit_missing_plates.py` | Proposes plate boxes that overlap no existing label; writes candidates and confidence-sorted contact sheets for human review. |
| `make_plate_auditor_data.py` | Splits videos by plate density to build the cross-audit train/audit sets (symlinks only). |
| `apply_audit.py` | Applies the reviewer's accept/reject decisions to a copy of the dataset; logs every added box. |
| `per_video_eval.py` | Scores a model separately on each video in a split. |
| `make_results_table.py` | Evaluates runs on valid and test, writes the paper tables. |
| `cv_experiment.py` | Builds the five cross-validation folds and trains and scores each one. |
| `run_seeds.py` | Retrains every detector with seeds 1 and 2. |
| `bootstrap_ci.py` | Image- and video-level bootstrap intervals and paired model differences. |
| `rescore_cv.py` | Re-scores the cross-validation models with Ultralytics 8.4.62. |
| `eval_unmasked.py` | Scores the models against the unmasked labels. |
| `run_loso.py` | Builds the leave-one-site-out folds from `site_map.csv`, trains and scores. |
| `eval_c_subsets.py` | Splits the group C held-out result by footbridge (Notunbazar vs. the night clips). |

Paths at the top of each script point at our working tree; change them to your own.

Requires `ultralytics`, `numpy`, `Pillow`, `pyyaml`.

> `download_data.py` from our working tree is **not** included: it contained a
> Roboflow API key. Point the scripts at your own export instead.

---

## Images

The frames are not redistributed. They are public-space footage containing
identifiable faces and readable license plates. Researchers can request them from
the authors under an agreement not to redistribute them. The annotations reference
frames by name, so anyone with the frames can rebuild our exact splits and labels.

## Citation

<!-- Replace once the paper has a DOI / proceedings entry. -->
```bibtex
@inproceedings{omnieye2026,
  author    = {Ahsan, Alif and Dessa, George Jefferson and Urmi, Mst. Rumana Radia
               and Payal, Jinia Alam and Tasin, Md. Parvez Ahmed and Ferdous, Jannatul
               and Shourav, Shoib Ahmed},
  title     = {Omni-Eye: Traffic Violation Detection and the Hidden Cost of
               Frame-Level Data Splits},
  booktitle = {Proc. 29th International Conference on Computer and Information
               Technology (ICCIT)},
  year      = {2026}
}
```

## License

Everything in this repository (annotations, split files, audit logs, results and
code) is released under the Creative Commons Attribution 4.0 International license
(CC BY 4.0); see `LICENSE`.

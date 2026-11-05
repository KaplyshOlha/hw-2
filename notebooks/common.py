"""Shared utilities for HW2 (UCU AI 2026, VisDrone2019-VID detection).

Used by both teammates: data loading, frame subsampling, ignored-region helpers,
fixed GroupKFold split, COCO-style evaluation (pycocotools), result logging and
Kaggle submission writing.

Conventions
- `images`: one row per labelled frame (train + val), index == `img_id` (global int id).
- `anns`:   one row per box; `cls` is 0..7 (submission ids), ignored regions have
            `iscrowd == 1` and `cls == -1`.
- predictions: DataFrame with columns img_id (or image_id for test), cls (0..7),
            score, x, y, w, h in ORIGINAL pixel coordinates, (x, y) = top-left.
"""
import contextlib
import io
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SLUG = "multiple-object-video-detection-hw-2-ucu-ai-2026"
CLASSES = ["person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"]
NUM_CLASSES = len(CLASSES)

# Fixed protocol (see project instructions)
SEED = 42
N_FOLDS = 3
TRAIN_STRIDE = 20   # frame stride for comparison runs
EVAL_STRIDE = 10    # frame stride for validation frames (keeps COCOeval fast)
IMGSZ = 640         # longest image side fed to every model
MAX_DETS = 500      # Kaggle scores at most 500 detections per image
FILL_VALUE = 114    # gray used to paint ignored regions

METRICS = ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
           "AR1", "AR100", "AR500", "AR_small", "AR_medium", "AR_large"]
RESULT_COLUMNS = ["run_id", "timestamp", "owner", "model", "experiment", "fold",
                  "train_stride", "eval_stride", "imgsz", "epochs",
                  "mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR500",
                  "fps", "train_min", "public_lb", "notes"]


# ---------------------------------------------------------------- misc
def seed_everything(seed=SEED):
    """Seed python, numpy and (if installed) torch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def find_data_root():
    """Return the competition data root (env HW2_DATA overrides the Kaggle paths)."""
    cands = [os.environ.get("HW2_DATA"), f"/kaggle/input/{SLUG}",
             f"/kaggle/input/competitions/{SLUG}"]
    for pat in ["/kaggle/input/*", "/kaggle/input/*/*", "/kaggle/input/*/*/*"]:
        cands += sorted(str(p) for p in Path("/").glob(pat.lstrip("/")))
    for c in cands:
        if c and (Path(c) / "train" / "annotations_coco.json").is_file():
            return Path(c)
    raise FileNotFoundError("Competition data not found; set HW2_DATA to the data root.")


# ---------------------------------------------------------------- ground truth
def _col(df, name, default):
    return df[name].fillna(default) if name in df else pd.Series(default, index=df.index)


def _scan_images(split_dir):
    """Map (clip, frame stem) -> absolute image path for one split."""
    idx = {}
    img_dir = Path(split_dir) / "images"
    for clip in sorted(os.listdir(img_dir)):
        if not (img_dir / clip).is_dir():
            continue
        for e in os.scandir(img_dir / clip):
            if e.name.lower().endswith((".jpg", ".jpeg", ".png")):
                idx[(clip, os.path.splitext(e.name)[0])] = e.path
    return idx


def _load_split(root, split):
    split_dir = Path(root) / split
    with open(split_dir / "annotations_coco.json") as f:
        d = json.load(f)
    names = {c["id"]: c["name"] for c in d.get("categories", [])}
    expected = {i + 1: n for i, n in enumerate(CLASSES)}
    bad = {k: v for k, v in expected.items() if k in names and names[k] != v}
    if bad:
        print(f"WARNING [{split}]: category names differ from CLASSES: {bad} vs {names}")

    fs = _scan_images(split_dir)
    fs_clips = {k[0] for k in fs}
    im = pd.DataFrame(d["images"])
    fn = im["file_name"].astype(str)
    clip = fn.map(lambda s: Path(s).parent.name)
    if "video" in im:
        clip = clip.where(clip.isin(fs_clips), im["video"].astype(str))
    stem = fn.map(lambda s: Path(s).stem)
    paths = [fs.get(k) for k in zip(clip, stem)]
    missing = sum(p is None for p in paths)
    if missing:
        raise FileNotFoundError(f"[{split}] {missing} COCO images not found on disk")
    frame = (pd.to_numeric(im["frame_id"], errors="coerce") if "frame_id" in im
             else pd.to_numeric(stem, errors="coerce"))
    images = pd.DataFrame({
        "split": split, "orig_id": im["id"].astype(np.int64).to_numpy(),
        "clip": clip.to_numpy(), "frame_id": frame.fillna(-1).astype(np.int64).to_numpy(),
        "width": im["width"].astype(np.int32).to_numpy(),
        "height": im["height"].astype(np.int32).to_numpy(), "path": paths})

    a = pd.DataFrame(d["annotations"])
    del d
    bb = np.asarray(a["bbox"].tolist(), dtype=np.float32).reshape(-1, 4)
    cat = a["category_id"].astype(np.int64).to_numpy()
    crowd = _col(a, "iscrowd", 0).astype(np.int64).to_numpy()
    target = (crowd == 0) & (cat >= 1) & (cat <= NUM_CLASSES)
    anns = pd.DataFrame({
        "split": split, "orig_img": a["image_id"].astype(np.int64).to_numpy(),
        "cls": np.where(target, cat - 1, -1).astype(np.int8),
        "x": bb[:, 0], "y": bb[:, 1], "w": bb[:, 2], "h": bb[:, 3],
        "iscrowd": (~target).astype(np.int8),
        "occlusion": _col(a, "occlusion", -1).astype(np.int8).to_numpy(),
        "truncation": _col(a, "truncation", -1).astype(np.int8).to_numpy(),
        "track_id": _col(a, "track_id", -1).astype(np.int64).to_numpy(),
        "orig_cat": _col(a, "orig_category_id", -1).astype(np.int16).to_numpy(),
        "category_id": cat.astype(np.int16)})
    return images, anns


def load_gt(root=None, cache=None):
    """Load train+val COCO annotations into (images, anns) with global img_id.

    Non-target categories and iscrowd=1 boxes become ignored regions (cls=-1, iscrowd=1).
    """
    if cache and Path(cache).exists():
        d = pd.read_pickle(cache)
        return d["images"], d["anns"]
    root = Path(root or find_data_root())
    parts = [_load_split(root, s) for s in ["train", "val"]]
    images = pd.concat([p[0] for p in parts], ignore_index=True)
    anns = pd.concat([p[1] for p in parts], ignore_index=True)
    if images.groupby("clip")["split"].nunique().max() > 1:
        raise ValueError("Clip names collide between train and val")
    images = images.sort_values(["clip", "frame_id"]).reset_index(drop=True)
    images.insert(0, "img_id", np.arange(len(images), dtype=np.int64))
    key = images.set_index(["split", "orig_id"])["img_id"]
    anns.insert(0, "img_id", key.reindex(pd.MultiIndex.from_arrays(
        [anns["split"], anns["orig_img"]])).to_numpy())
    anns = anns.dropna(subset=["img_id"]).astype({"img_id": np.int64})
    anns = (anns.drop(columns=["split", "orig_img"])
            .sort_values(["img_id", "iscrowd"], kind="stable").reset_index(drop=True))
    images.index = images["img_id"].to_numpy()
    if cache:
        pd.to_pickle({"images": images, "anns": anns}, cache)
    return images, anns


def subsample(images, stride=1, offset=0):
    """Keep every `stride`-th frame of each clip (by frame order), starting at `offset`."""
    im = images.sort_values(["clip", "frame_id"])
    if stride <= 1:
        return im.copy()
    rank = im.groupby("clip").cumcount()
    return im[(rank % stride) == (offset % stride)].copy()


# ---------------------------------------------------------------- ignored regions
def ignore_mask(h, w, regions):
    """uint8 mask (h, w) with 1 inside any ignored region (regions: N x [x, y, w, h])."""
    m = np.zeros((int(h), int(w)), np.uint8)
    for x, y, rw, rh in np.asarray(regions, dtype=np.float64).reshape(-1, 4):
        x0, y0 = max(int(math.floor(x)), 0), max(int(math.floor(y)), 0)
        x1, y1 = min(int(math.ceil(x + rw)), int(w)), min(int(math.ceil(y + rh)), int(h))
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = 1
    return m


def box_coverage(boxes_xywh, mask):
    """Fraction of each box (x, y, w, h) covered by the ignore mask."""
    b = np.asarray(boxes_xywh, dtype=np.float64).reshape(-1, 4)
    h, w = mask.shape
    x0 = np.clip(np.floor(b[:, 0]), 0, w).astype(int)
    y0 = np.clip(np.floor(b[:, 1]), 0, h).astype(int)
    x1 = np.clip(np.ceil(b[:, 0] + b[:, 2]), 0, w).astype(int)
    y1 = np.clip(np.ceil(b[:, 1] + b[:, 3]), 0, h).astype(int)
    out = np.zeros(len(b))
    for i in range(len(b)):
        area = (x1[i] - x0[i]) * (y1[i] - y0[i])
        if area > 0:
            out[i] = mask[y0[i]:y1[i], x0[i]:x1[i]].sum() / area
    return out


def fill_ignore(img, mask, value=FILL_VALUE):
    """Paint ignored pixels with a constant gray (returns a copy)."""
    img = img.copy()
    img[mask.astype(bool)] = value
    return img


# ---------------------------------------------------------------- splits
def make_splits(images, n_folds=N_FOLDS):
    """Fixed GroupKFold by clip (deterministic, balanced by number of frames)."""
    from sklearn.model_selection import GroupKFold
    df = images.sort_values(["clip", "frame_id"]).reset_index(drop=True)
    clips = sorted(df["clip"].unique())
    folds = []
    for k, (_, va) in enumerate(GroupKFold(n_splits=n_folds).split(df, groups=df["clip"])):
        val = sorted(df["clip"].iloc[va].unique())
        folds.append({"fold": k, "train_clips": [c for c in clips if c not in set(val)],
                      "val_clips": val})
    return {"method": f"GroupKFold(n_splits={n_folds}) by clip over train+val",
            "seed": SEED, "n_folds": n_folds,
            "clip_source": df.groupby("clip")["split"].first().to_dict(), "folds": folds}


def save_splits(splits, path="splits.json"):
    with open(path, "w") as f:
        json.dump(splits, f, indent=1)


def load_splits(path="splits.json"):
    with open(path) as f:
        return json.load(f)


def clip_to_fold(splits):
    return {c: f["fold"] for f in splits["folds"] for c in f["val_clips"]}


def fold_images(images, splits, fold, part="val", stride=1, offset=0):
    """Frames of the train/val part of a fold, optionally subsampled."""
    clips = set(splits["folds"][fold][f"{part}_clips"])
    return subsample(images[images["clip"].isin(clips)], stride, offset)


# ---------------------------------------------------------------- evaluation
def make_coco_gt(images, anns, img_ids):
    """pycocotools COCO object for the given frames.

    Ignored regions have no class, so each one is copied to all 8 categories with
    iscrowd=1: any detection inside it is then ignored regardless of its class.
    """
    from pycocotools.coco import COCO
    ids = np.unique(np.asarray(img_ids, dtype=np.int64))
    im = images[images["img_id"].isin(ids)]
    a = anns[anns["img_id"].isin(ids)]
    obj, ign = a[a["iscrowd"] == 0], a[a["iscrowd"] == 1]
    rep = NUM_CLASSES
    img = np.concatenate([obj["img_id"].to_numpy(), np.repeat(ign["img_id"].to_numpy(), rep)])
    box = np.concatenate([obj[["x", "y", "w", "h"]].to_numpy(np.float64),
                          np.repeat(ign[["x", "y", "w", "h"]].to_numpy(np.float64), rep, 0)])
    cat = np.concatenate([obj["cls"].to_numpy(np.int64) + 1,
                          np.tile(np.arange(1, NUM_CLASSES + 1), len(ign))])
    crowd = np.concatenate([np.zeros(len(obj), int), np.ones(len(ign) * rep, int)])
    annotations = [{"id": i + 1, "image_id": int(m), "category_id": int(c),
                    "bbox": [float(v) for v in b], "area": float(b[2] * b[3]), "iscrowd": int(z)}
                   for i, (m, c, b, z) in enumerate(zip(img.tolist(), cat.tolist(), box, crowd.tolist()))]
    coco = COCO()
    coco.dataset = {
        "images": [{"id": int(i), "width": int(w), "height": int(h)}
                   for i, w, h in zip(im["img_id"], im["width"], im["height"])],
        "annotations": annotations,
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(CLASSES)]}
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def _clean_preds(preds, ids, max_dets):
    p = preds[preds["img_id"].isin(ids) & (preds["w"] > 0) & (preds["h"] > 0)]
    return (p.sort_values("score", ascending=False, kind="stable")
            .groupby("img_id", sort=False).head(max_dets))


def evaluate(preds, images, anns, img_ids, max_dets=MAX_DETS, per_class=False,
             coco_gt=None, verbose=False):
    """COCO metrics (mAP@[.5:.95], AP50, AP_small, ...) on the given frames.

    maxDets = [1, 100, max_dets]; AP is reported at max_dets (500 like Kaggle).
    """
    from pycocotools.cocoeval import COCOeval
    ids = np.unique(np.asarray(img_ids, dtype=np.int64))
    gt = coco_gt if coco_gt is not None else make_coco_gt(images, anns, ids)
    p = _clean_preds(preds, ids, max_dets)
    if len(p) == 0:
        res = {k: 0.0 for k in METRICS}
        if per_class:
            res.update({f"AP_{c}": 0.0 for c in CLASSES})
        return res
    arr = np.column_stack([p["img_id"], p["x"], p["y"], p["w"], p["h"], p["score"],
                           p["cls"].astype(np.int64) + 1]).astype(np.float64)
    out = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
    with out:
        dt = gt.loadRes(arr)
        E = COCOeval(gt, dt, "bbox")
        E.params.imgIds = ids.tolist()
        E.params.maxDets = [1, 100, max_dets]
        E.evaluate()
        E.accumulate()
        E.summarize()
    res = dict(zip(METRICS, map(float, E.stats)))
    if per_class:
        prec = E.eval["precision"][:, :, :, 0, -1]
        for k, name in enumerate(CLASSES):
            v = prec[:, :, k]
            v = v[v > -1]
            res[f"AP_{name}"] = float(v.mean()) if v.size else float("nan")
    return res


def evaluate_oof(fold_preds, images, anns, splits, eval_stride=EVAL_STRIDE, per_class=False):
    """Per-fold metrics, mean/std across folds and pooled out-of-fold metrics.

    fold_preds: {fold: predictions on that fold's val frames}.
    """
    rows, all_ids = {}, []
    for k, p in sorted(fold_preds.items()):
        ids = fold_images(images, splits, k, "val", eval_stride)["img_id"].to_numpy()
        rows[k] = evaluate(p, images, anns, ids, per_class=per_class)
        all_ids.append(ids)
    df = pd.DataFrame(rows).T
    mean, std = df.mean(), df.std(ddof=0)
    df.loc["mean"], df.loc["std"] = mean, std
    pooled = pd.concat(list(fold_preds.values()), ignore_index=True)
    df.loc["oof"] = pd.Series(evaluate(pooled, images, anns, np.concatenate(all_ids),
                                       per_class=per_class))
    return df


# ---------------------------------------------------------------- logging
def log_results(metrics, path="results.csv", **info):
    """Append one row per index of `metrics` (folds, mean, std, oof) to results.csv.

    info: owner, model, experiment, train_stride, eval_stride, imgsz, epochs, fps,
    train_min, notes, ... (unknown keys become extra columns).
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_id = f"{info.get('experiment', 'exp')}_{datetime.now():%Y%m%d%H%M%S}"
    rows = []
    for fold, m in metrics.iterrows():
        r = {"run_id": run_id, "timestamp": ts, "fold": fold, **info}
        r.update({k: round(float(v), 4) for k, v in m.items() if pd.notna(v)})
        rows.append(r)
    new = pd.DataFrame(rows)
    out = pd.concat([pd.read_csv(path), new], ignore_index=True) if Path(path).exists() else new
    cols = RESULT_COLUMNS + [c for c in out.columns if c not in RESULT_COLUMNS]
    out.reindex(columns=cols).to_csv(path, index=False)
    return new


def merge_results(src, dst):
    """Merge a results.csv downloaded from Kaggle into the repo copy (dedup by run_id, fold)."""
    frames = [pd.read_csv(p) for p in [dst, src] if Path(p).exists()]
    out = pd.concat(frames, ignore_index=True).astype({"fold": str})
    out = out.drop_duplicates(subset=["run_id", "fold"], keep="last").sort_values(
        ["timestamp", "run_id"], kind="stable")
    cols = RESULT_COLUMNS + [c for c in out.columns if c not in RESULT_COLUMNS]
    out.reindex(columns=cols).to_csv(dst, index=False)
    return out


# ---------------------------------------------------------------- submission
def load_test_index(root=None):
    root = Path(root or find_data_root())
    t = pd.read_csv(root / "test_images.csv")
    t["path"] = [str(root / f) for f in t["file_name"]]
    return t


def write_submission(preds, test_index, path="submission.csv", max_dets=MAX_DETS):
    """preds: image_id (str), cls (0..7), score, x, y, w, h in original pixels."""
    size = test_index.set_index("image_id")[["width", "height"]]
    p = preds.merge(size, left_on="image_id", right_index=True, how="inner")
    W, H = p["width"].to_numpy(float), p["height"].to_numpy(float)
    x1 = np.clip(p["x"].to_numpy(float), 0, W)
    y1 = np.clip(p["y"].to_numpy(float), 0, H)
    x2 = np.clip(p["x"].to_numpy(float) + p["w"].to_numpy(float), 0, W)
    y2 = np.clip(p["y"].to_numpy(float) + p["h"].to_numpy(float), 0, H)
    p = p.assign(x=x1, y=y1, w=x2 - x1, h=y2 - y1)
    p = p[(p["w"] > 0) & (p["h"] > 0)]
    p = (p.sort_values("score", ascending=False, kind="stable")
         .groupby("image_id", sort=False).head(max_dets))
    txt = [f"{int(c)} {s:.4f} {x:.1f} {y:.1f} {w:.1f} {h:.1f}"
           for c, s, x, y, w, h in zip(p["cls"], p["score"], p["x"], p["y"], p["w"], p["h"])]
    strings = p.assign(txt=txt).groupby("image_id", sort=False)["txt"].agg(" ".join)
    sub = pd.DataFrame({"image_id": test_index["image_id"].to_numpy()})
    sub["PredictionString"] = sub["image_id"].map(strings).fillna("")
    sub.to_csv(path, index=False)
    return path


def check_submission(path, sample_path):
    """Validate ids/order against sample_submission and the 6-value group format."""
    sub = pd.read_csv(path, keep_default_na=False)
    ref = pd.read_csv(sample_path, keep_default_na=False)
    n = [len(s.split()) for s in sub["PredictionString"]]
    vals = [s.split() for s in sub["PredictionString"] if s]
    cls_ok = all(0 <= int(v[i]) < NUM_CLASSES for v in vals for i in range(0, len(v), 6))
    return {"rows": len(sub), "same_ids_and_order": sub["image_id"].tolist() == ref["image_id"].tolist(),
            "groups_of_6": all(k % 6 == 0 for k in n), "max_dets": max(n) // 6 if n else 0,
            "classes_ok": cls_ok, "empty_rows": int(sum(k == 0 for k in n))}

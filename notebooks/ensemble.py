"""Model ensembling with Weighted Boxes Fusion (Solovyev et al., 2021, arXiv:1910.13302).

WBF clusters boxes of the same class from several models by IoU and replaces each cluster with a
score-weighted average box, instead of keeping one box and deleting the rest (NMS). Averaging the
coordinates of several models improves localisation, which matters for AP@[.5:.95].
Implementation: `ensemble_boxes` package (ZFTurbo), MIT licence.

Predictions use the project format: key, cls, score, x, y, w, h (top-left, original pixels).
"""
import warnings

import numpy as np
import pandas as pd


def topk(preds, k, key="key"):
    """Keep the k highest-scoring boxes per image (e.g. compare max_det 300 vs 500)."""
    return (preds.sort_values("score", ascending=False, kind="stable")
            .groupby(key, sort=False).head(k).reset_index(drop=True))


def fuse(pred_list, sizes, weights=None, iou_thr=0.6, skip_box_thr=0.001, conf_type="avg",
         max_det=500, key="key"):
    """Fuse predictions of several models image by image.

    pred_list: list of DataFrames (one per model) with key, cls, score, x, y, w, h.
    sizes: DataFrame indexed by key with columns width, height.
    Returns one DataFrame in the same format, at most max_det boxes per image.
    """
    from ensemble_boxes import weighted_boxes_fusion
    weights = weights or [1] * len(pred_list)
    groups = [dict(tuple(p.groupby(key, sort=False))) for p in pred_list]
    keys = pd.unique(np.concatenate([p[key].to_numpy() for p in pred_list]))
    out = []
    for k in keys:
        W, H = float(sizes.at[k, "width"]), float(sizes.at[k, "height"])
        boxes, scores, labels, wts = [], [], [], []
        for g, w in zip(groups, weights):
            d = g.get(k)
            if d is None or len(d) == 0:
                continue
            x1 = np.clip(d["x"].to_numpy() / W, 0, 1)
            y1 = np.clip(d["y"].to_numpy() / H, 0, 1)
            x2 = np.clip((d["x"].to_numpy() + d["w"].to_numpy()) / W, 0, 1)
            y2 = np.clip((d["y"].to_numpy() + d["h"].to_numpy()) / H, 0, 1)
            boxes.append(np.c_[x1, y1, x2, y2]); scores.append(d["score"].to_numpy())
            labels.append(d["cls"].to_numpy()); wts.append(w)
        if not boxes:
            continue
        with warnings.catch_warnings():   # "zero area box skipped" warnings would flood the notebook
            warnings.simplefilter("ignore")
            b, s, l = weighted_boxes_fusion(boxes, scores, labels, weights=wts, iou_thr=iou_thr,
                                            skip_box_thr=skip_box_thr, conf_type=conf_type)
        if len(b) == 0:
            continue
        o = np.argsort(-s)[:max_det]
        b, s, l = b[o], s[o], l[o]
        out.append(pd.DataFrame({key: k, "cls": l.astype(int), "score": s,
                                 "x": b[:, 0] * W, "y": b[:, 1] * H,
                                 "w": (b[:, 2] - b[:, 0]) * W, "h": (b[:, 3] - b[:, 1]) * H}))
    cols = [key, "cls", "score", "x", "y", "w", "h"]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)

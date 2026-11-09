"""Temporal post-processing of per-frame detections (bonus: data temporality).

Frames of a clip are consecutive, so a true object detected with a low score in one frame is
usually detected with a higher score in its neighbours, while background false positives flicker.
We exploit this without retraining:

1. Camera-motion compensation: the drone moves, so a static object shifts between frames. A global
   translation between consecutive frames is estimated with phase correlation on downscaled
   grayscale frames (cv2.phaseCorrelate) and applied to the previous boxes before matching.
2. Linking: detections of the same class in consecutive frames are linked greedily by IoU
   (after motion compensation) into tracks; a track survives `max_gap` frames without a match.
   Similar in spirit to SORT (Bewley et al., 2016) without a Kalman filter, and to Seq-NMS
   (Han et al., 2016) which links boxes across frames and rescores them along the sequence.
3. Rescoring: each detection's score is blended with the mean score of its track within a
   temporal window:  s' = (1 - alpha) * s + alpha * mean(track scores in [t - w, t + w]).
   Long, consistently confident tracks lift weak detections; isolated flickers keep or lose score.
4. Gap filling (optional): when a track is missing in frame t but present at t-k and t+j
   (k, j <= max_gap), a box is interpolated at t with a decayed score (recovers missed detections,
   the idea behind T-CNN's propagation; Kang et al., 2016).

All functions work on a predictions DataFrame with columns
    key, frame, cls, score, x, y, w, h     (x, y = top-left, original pixels)
grouped by clip; `key` is img_id (validation) or image_id (test).
"""
import cv2
import numpy as np
import pandas as pd


# ---------------------------------------------------------------- camera motion
def frame_shifts(paths, scale=0.25):
    """Global (dx, dy) translation from frame i-1 to frame i (pixels of the original frame),
    estimated by phase correlation on downscaled grayscale images. shifts[0] = (0, 0)."""
    out, prev = [(0.0, 0.0)], None
    for p in paths:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).astype(np.float32)
        if prev is not None:
            if prev.shape != g.shape:
                out.append((0.0, 0.0))
            else:
                win = cv2.createHanningWindow(g.shape[::-1], cv2.CV_32F)
                (dx, dy), _ = cv2.phaseCorrelate(prev, g, win)
                out.append((dx / scale, dy / scale))
        prev = g
    return np.asarray(out[: len(paths)], np.float64)


# ---------------------------------------------------------------- linking
def _iou(a, b):
    """IoU matrix between xywh boxes a (n, 4) and b (m, 4)."""
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return inter / np.maximum(union, 1e-9)


def link_clip(p, shifts=None, iou_thr=0.3, max_gap=2, min_score=0.05):
    """Assign a track id to every detection of one clip (p sorted by frame).

    Only detections with score >= min_score start or extend tracks; lower ones get track -1
    (they are numerous and mostly background). shifts: (n_frames, 2) per-frame translation,
    indexed by the position of the frame in the sorted list of frames of this clip.
    """
    p = p.sort_values(["frame", "score"], ascending=[True, False], kind="stable").copy()
    frames = np.sort(p["frame"].unique())
    pos = {f: i for i, f in enumerate(frames)}
    cum = np.zeros((len(frames), 2)) if shifts is None else np.cumsum(shifts, axis=0)
    track = np.full(len(p), -1, np.int64)
    active = {}   # track id -> (cls, last box xywh in "world" coords, last frame pos)
    next_id = 0
    idx_by_frame = p.groupby("frame", sort=True).indices
    boxes = p[["x", "y", "w", "h"]].to_numpy(np.float64)
    cls = p["cls"].to_numpy()
    score = p["score"].to_numpy()
    for f in frames:
        i = pos[f]
        rows = idx_by_frame[f]
        rows = rows[score[rows] >= min_score]
        # world coordinates: undo cumulative camera translation
        world = boxes[rows].copy()
        world[:, 0] -= cum[i, 0]
        world[:, 1] -= cum[i, 1]
        for t in [t for t, (_, _, last) in active.items() if i - last > max_gap + 1]:
            del active[t]
        used = set()
        if active and len(rows):
            tids = list(active)
            tb = np.array([active[t][1] for t in tids])
            tc = np.array([active[t][0] for t in tids])
            iou = _iou(world, tb)
            iou[cls[rows][:, None] != tc[None]] = 0
            # greedy matching, highest IoU first
            order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
            taken_r, taken_t = set(), set()
            for r, c in order:
                if iou[r, c] < iou_thr:
                    break
                if r in taken_r or c in taken_t:
                    continue
                taken_r.add(r); taken_t.add(c)
                track[rows[r]] = tids[c]
                active[tids[c]] = (cls[rows[r]], world[r], i)
                used.add(r)
        for r in range(len(rows)):
            if r not in used:
                track[rows[r]] = next_id
                active[next_id] = (cls[rows[r]], world[r], i)
                next_id += 1
    p["track"] = track
    return p


# ---------------------------------------------------------------- rescoring
def rescore(p, alpha=0.5, window=5, min_len=1, causal=False):
    """s' = (1 - alpha) * s + alpha * mean score of the same track within +-window frames
    (causal=True: only frames t-window..t, i.e. only past predictions are used).
    Detections without a track (track == -1) or in tracks shorter than min_len keep s."""
    p = p.copy()
    p["score_raw"] = p["score"]
    tracked = p[p["track"] >= 0]
    new = p["score"].to_numpy().copy()
    for _, g in tracked.groupby("track", sort=False):
        if len(g) < min_len:
            continue
        f = g["frame"].to_numpy()
        s = g["score"].to_numpy()
        d = np.abs(f[:, None] - f[None])
        m = d <= window
        if causal:
            m &= f[None] <= f[:, None]
        m = m.astype(np.float64)
        mean = (m @ s) / m.sum(1)
        new[p.index.get_indexer(g.index)] = (1 - alpha) * s + alpha * mean
    p["score"] = new
    return p


def fill_gaps(p, max_gap=2, decay=0.5, min_len=3):
    """Interpolate missing boxes inside tracks (gaps of <= max_gap frames) with score
    decay * min(neighbour scores). Only tracks with >= min_len detections are filled."""
    add = []
    for t, g in p[p["track"] >= 0].groupby("track", sort=False):
        if len(g) < min_len:
            continue
        g = g.sort_values("frame")
        f = g["frame"].to_numpy()
        for a, b in zip(range(len(g) - 1), range(1, len(g))):
            gap = f[b] - f[a]
            if 1 < gap <= max_gap + 1:
                ra, rb = g.iloc[a], g.iloc[b]
                for k in range(1, gap):
                    w = k / gap
                    add.append({"key": None, "frame": f[a] + k, "cls": ra["cls"],
                                "score": decay * min(ra["score"], rb["score"]),
                                "x": (1 - w) * ra["x"] + w * rb["x"], "y": (1 - w) * ra["y"] + w * rb["y"],
                                "w": (1 - w) * ra["w"] + w * rb["w"], "h": (1 - w) * ra["h"] + w * rb["h"],
                                "track": t, "filled": True})
    if not add:
        return p.assign(filled=False)
    out = pd.concat([p.assign(filled=False), pd.DataFrame(add)], ignore_index=True)
    return out


def apply(preds, frame_index, alpha=0.5, window=5, iou_thr=0.3, max_gap=2, min_score=0.05,
          fill=False, decay=0.5, shifts=None, causal=False):
    """Temporal post-processing for all clips.

    preds: key, cls, score, x, y, w, h.  frame_index: key, clip, frame (all frames that exist,
    used to map filled boxes back to keys).  shifts: {clip: (n_frames, 2)} or None.
    Returns predictions with the same columns (+ score_raw, track, filled).
    """
    p = preds.merge(frame_index[["key", "clip", "frame"]], on="key", how="inner")
    out = []
    for clip, g in p.groupby("clip", sort=False):
        sh = None if shifts is None else shifts.get(clip)
        g = link_clip(g, sh, iou_thr=iou_thr, max_gap=max_gap, min_score=min_score)
        g = rescore(g, alpha=alpha, window=window, causal=causal)
        if fill and not causal:   # gap filling needs the future frame
            g = fill_gaps(g, max_gap=max_gap, decay=decay)
            fk = frame_index[frame_index["clip"] == clip].set_index("frame")["key"]
            miss = g["key"].isna()
            g.loc[miss, "key"] = g.loc[miss, "frame"].map(fk)
            g = g[g["key"].notna()]
        g["clip"] = clip
        out.append(g)
    res = pd.concat(out, ignore_index=True)
    if "filled" not in res:
        res["filled"] = False
    return res

"""Torchvision detection dataset for VisDrone-VID frames (Faster R-CNN / RetinaNet).

- ignored regions are painted gray (114); boxes covered by them >= `drop_cover` are dropped
- frames are subsampled per clip with `stride`
- the image is resized in the worker so that its longest side == `max_side`
  (use the model with min_size=max_size=max_side so its own transform does not resize again)
- images are returned as uint8 tensors; `to_device` converts to float on the GPU
"""
import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import common


class VisDroneDet(Dataset):
    def __init__(self, images, anns, clips=None, stride=1, offset=0, max_side=common.IMGSZ,
                 train=False, hflip_p=0.5, fill_value=common.FILL_VALUE, drop_cover=0.5,
                 min_size=1.0, label_offset=1):
        im = images if clips is None else images[images["clip"].isin(set(clips))]
        im = common.subsample(im, stride, offset)
        self.ids = im["img_id"].to_numpy()
        self.paths = im["path"].tolist()
        self.max_side, self.train, self.hflip_p = max_side, train, hflip_p
        self.fill_value, self.drop_cover, self.min_size = fill_value, drop_cover, min_size
        self.label_offset = label_offset
        a = anns[anns["img_id"].isin(set(self.ids.tolist()))]
        obj, ign = a[a["iscrowd"] == 0], a[a["iscrowd"] == 1]
        self.obj = {k: (g[["x", "y", "w", "h"]].to_numpy(np.float32), g["cls"].to_numpy(np.int64))
                    for k, g in obj.groupby("img_id")}
        self.ign = {k: g[["x", "y", "w", "h"]].to_numpy(np.float32) for k, g in ign.groupby("img_id")}

    def __len__(self):
        return len(self.ids)

    def load(self, i):
        """RGB uint8 image with ignored regions filled + boxes (xyxy, original px) + labels 0..7."""
        iid = int(self.ids[i])
        img = cv2.cvtColor(cv2.imread(self.paths[i], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        b, c = self.obj.get(iid, (np.zeros((0, 4), np.float32), np.zeros(0, np.int64)))
        reg = self.ign.get(iid)
        if reg is not None and len(reg):
            m = common.ignore_mask(h, w, reg)
            img[m.astype(bool)] = self.fill_value
            if len(b):
                keep = common.box_coverage(b, m) < self.drop_cover
                b, c = b[keep], c[keep]
        xyxy = np.stack([np.clip(b[:, 0], 0, w), np.clip(b[:, 1], 0, h),
                         np.clip(b[:, 0] + b[:, 2], 0, w), np.clip(b[:, 1] + b[:, 3], 0, h)], 1)
        keep = ((xyxy[:, 2] - xyxy[:, 0]) >= self.min_size) & ((xyxy[:, 3] - xyxy[:, 1]) >= self.min_size)
        return img, xyxy[keep].astype(np.float32), c[keep]

    def __getitem__(self, i):
        img, boxes, labels = self.load(i)
        h, w = img.shape[:2]
        s = 1.0
        if self.max_side and max(h, w) != self.max_side:
            s = self.max_side / max(h, w)
            interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=interp)
            boxes = boxes * s
        if self.train and random.random() < self.hflip_p:
            img = img[:, ::-1]
            W = img.shape[1]
            boxes = boxes[:, [2, 1, 0, 3]] * np.array([-1, 1, -1, 1], np.float32) \
                + np.array([W, 0, W, 0], np.float32)
        target = {"boxes": torch.from_numpy(np.ascontiguousarray(boxes, np.float32)).reshape(-1, 4),
                  "labels": torch.from_numpy(labels + self.label_offset),
                  "image_id": int(self.ids[i]), "scale": s, "orig_size": (h, w)}
        img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


def seed_worker(worker_id):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def make_loader(ds, batch_size=4, shuffle=None, num_workers=2, seed=common.SEED):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=ds.train if shuffle is None else shuffle,
                      num_workers=num_workers, collate_fn=collate_fn, worker_init_fn=seed_worker,
                      generator=g, pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0)


def to_device(images, targets, device):
    """uint8 -> float [0, 1] on the target device (cheaper host->GPU copies)."""
    imgs = [im.to(device, non_blocking=True).float().div_(255) for im in images]
    tgts = [{k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in t.items()}
            for t in targets]
    return imgs, tgts


def preds_to_df(outputs, targets, label_offset=1):
    """torchvision outputs (xyxy in network scale) -> common prediction DataFrame (original px)."""
    rows = []
    for o, t in zip(outputs, targets):
        b = o["boxes"].detach().float().cpu().numpy() / t["scale"]
        n = len(b)
        if n == 0:
            continue
        rows.append(pd_frame(np.full(n, t["image_id"]), o["labels"].cpu().numpy() - label_offset,
                             o["scores"].detach().float().cpu().numpy(), b))
    if not rows:
        return pd_frame(np.zeros(0, int), np.zeros(0, int), np.zeros(0), np.zeros((0, 4)))
    return pd.concat(rows, ignore_index=True)


def pd_frame(img_id, cls, score, xyxy):
    return pd.DataFrame({"img_id": img_id.astype(np.int64), "cls": cls.astype(np.int64), "score": score,
                         "x": xyxy[:, 0], "y": xyxy[:, 1],
                         "w": xyxy[:, 2] - xyxy[:, 0], "h": xyxy[:, 3] - xyxy[:, 1]})

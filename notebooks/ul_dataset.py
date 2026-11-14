"""Ultralytics side of HW2 (Person B): YOLO (anchor-free) and RT-DETR (query-based).

Same protocol as the torchvision models (common.py): fixed 3-fold GroupKFold by clip
(splits.json), train frames every TRAIN_STRIDE-th frame, validation frames every
EVAL_STRIDE-th frame, IMGSZ input, equal epochs, OOF COCO metrics via common.evaluate.

Data layout written to <root> (once per Kaggle session, reused by every fold/experiment):
    <root>/filled/images/<clip>/<stem>.jpg   train frames: ignored regions painted gray (114),
    <root>/filled/labels/<clip>/<stem>.txt   boxes >= 50 % inside ignore dropped (as tv_dataset)
    <root>/raw/images/<clip>/<stem>.jpg      val frames: symlinks to the original (unfilled) frames
    <root>/raw/labels/<clip>/<stem>.txt      YOLO labels of all target boxes
    <root>/folds/fold<k>_{train,val}.txt     image lists; data_fold<k>.yaml points to them
Ultralytics finds the label of an image by replacing /images/ with /labels/ in its path.

Filled train frames are stored with the longest side <= STORE_SIDE (1280), so the same files
serve 640 px comparison runs and higher-resolution final runs.

Evaluation never uses Ultralytics' own val metric (it does not know ignored regions): we
predict the ORIGINAL validation frames with conf=0.001 and score them with common.evaluate.
The model after the last epoch (last.pt) is evaluated, not best.pt: best.pt is selected on
the same OOF fold and would leak validation information into the reported metric.

Sources: Ultralytics docs (train / predict arguments, dataset YAML with .txt image lists);
torchvision-side conventions from tv_dataset.py / tv_train.py (teammate's code).
"""
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import common

STORE_SIDE = 1280
RUNTIME_FIELDS = {"workers", "device", "notes", "fps_iters", "fps_warmup", "pred_batch"}


# ---------------------------------------------------------------- config
@dataclass
class Config:
    experiment: str = "yolo_base"
    model: str = "yolo"                  # "yolo" | "rtdetr"
    owner: str = "B"
    weights: str = "yolov8s.pt"          # COCO-pretrained weights
    arch: str = ""                       # optional model YAML (e.g. yolov8s-p2.yaml); weights are transferred
    imgsz: int = common.IMGSZ
    train_stride: int = common.TRAIN_STRIDE
    eval_stride: int = common.EVAL_STRIDE
    epochs: int = 12
    batch: int = 16
    optimizer: str = "SGD"
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 5e-4
    warmup_epochs: float = 1.0
    warmup_bias_lr: float = 0.1          # Ultralytics default (tuned for YOLO+SGD); 0 = no bias-lr spike
    close_mosaic: int = 3                # default 10 would switch mosaic off after epoch 2 of 12
    amp: bool = True
    cache: bool = False                  # "ram" is faster but Ultralytics warns it is non-deterministic
    # loss gains / modification
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5
    nwd: float = 0.0                     # >0: box loss = (1-nwd)*CIoU + nwd*NWD  (see patch_nwd)
    nwd_c: float = 12.8                  # NWD normalising constant (Wang et al. 2021, AI-TOD)
    # RT-DETR
    num_queries: int = 300
    backbone_lr_mult: float = 1.0        # <1: backbone (layers 0-9 of rtdetr-l) trained with lr0 * mult
    num_denoising: int = 100             # RT-DETR denoising queries (0 = no denoising loss)
    rt_bbox_gain: float = 5.0            # RT-DETR loss weights of the L1 and GIoU box terms (Ultralytics defaults)
    rt_giou_gain: float = 2.0
    # augmentation (Ultralytics defaults unless changed)
    mosaic: float = 1.0
    scale: float = 0.5
    fliplr: float = 0.5
    # inference
    conf: float = 0.001
    iou: float = 0.7
    max_det: int = 300
    # runtime only (do not change results)
    workers: int = 4
    device: str = "0"
    pred_batch: int = 16
    fps_warmup: int = 10
    fps_iters: int = 50
    notes: str = ""

    def key(self):
        return {k: v for k, v in asdict(self).items() if k not in RUNTIME_FIELDS}


def make_config(base=None, **changes):
    return replace(base or Config(), **changes)


# ---------------------------------------------------------------- data conversion
def _stem(path):
    return Path(path).stem


def _yolo_lines(boxes_xywh, cls, w, h):
    b = np.asarray(boxes_xywh, np.float64).reshape(-1, 4)
    x0 = np.clip(b[:, 0], 0, w); y0 = np.clip(b[:, 1], 0, h)
    x1 = np.clip(b[:, 0] + b[:, 2], 0, w); y1 = np.clip(b[:, 1] + b[:, 3], 0, h)
    keep = ((x1 - x0) >= 1) & ((y1 - y0) >= 1)
    cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
    bw, bh = (x1 - x0) / w, (y1 - y0) / h
    return [f"{int(c)} {a:.6f} {b_:.6f} {c_:.6f} {d:.6f}"
            for c, a, b_, c_, d in zip(np.asarray(cls)[keep], cx[keep], cy[keep], bw[keep], bh[keep])]


class Converter:
    """Writes filled train frames / raw val symlinks + YOLO labels; skips files that exist."""

    def __init__(self, images, anns, root, store_side=STORE_SIDE, drop_cover=0.5,
                 fill_value=common.FILL_VALUE):
        self.images, self.root = images, Path(root)
        self.store_side, self.drop_cover, self.fill_value = store_side, drop_cover, fill_value
        obj, ign = anns[anns["iscrowd"] == 0], anns[anns["iscrowd"] == 1]
        self.obj = {k: (g[["x", "y", "w", "h"]].to_numpy(np.float32), g["cls"].to_numpy(np.int64))
                    for k, g in obj.groupby("img_id")}
        self.ign = {k: g[["x", "y", "w", "h"]].to_numpy(np.float32) for k, g in ign.groupby("img_id")}

    def path(self, img_id, kind):
        r = self.images.loc[img_id]
        return self.root / kind / "images" / r["clip"] / f"{_stem(r['path'])}.jpg"

    def _write_filled(self, iid):
        out = self.path(iid, "filled")
        lab = Path(str(out).replace("/images/", "/labels/")).with_suffix(".txt")
        if out.exists() and lab.exists():
            return
        r = self.images.loc[iid]
        img = cv2.imread(r["path"], cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        b, c = self.obj.get(iid, (np.zeros((0, 4), np.float32), np.zeros(0, np.int64)))
        reg = self.ign.get(iid)
        if reg is not None and len(reg):
            m = common.ignore_mask(h, w, reg)
            img[m.astype(bool)] = self.fill_value
            if len(b):
                keep = common.box_coverage(b, m) < self.drop_cover
                b, c = b[keep], c[keep]
        s = min(1.0, self.store_side / max(h, w))
        if s < 1:
            img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        out.parent.mkdir(parents=True, exist_ok=True)
        lab.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lab.write_text("\n".join(_yolo_lines(b, c, w, h)))  # normalised: independent of resize

    def _write_raw(self, iid):
        out = self.path(iid, "raw")
        lab = Path(str(out).replace("/images/", "/labels/")).with_suffix(".txt")
        if out.exists() and lab.exists():
            return
        r = self.images.loc[iid]
        b, c = self.obj.get(iid, (np.zeros((0, 4), np.float32), np.zeros(0, np.int64)))
        out.parent.mkdir(parents=True, exist_ok=True)
        lab.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            os.symlink(r["path"], out)
        lab.write_text("\n".join(_yolo_lines(b, c, int(r["width"]), int(r["height"]))))

    def write(self, img_ids, kind, threads=8):
        fn = self._write_filled if kind == "filled" else self._write_raw
        t0 = time.time()
        with ThreadPoolExecutor(threads) as ex:
            list(ex.map(fn, [int(i) for i in img_ids]))
        return time.time() - t0


def prepare_fold(conv, splits, fold, cfg, threads=8):
    """Write the frames of one fold and its dataset YAML; returns the YAML path.

    fold == "all": every clip is used for training (final / leaderboard model);
    a small raw val list (fold 0 val frames) is kept only so Ultralytics can run.
    """
    images = conv.images
    if fold == "all":
        tr = common.subsample(images, cfg.train_stride)
        va = common.fold_images(images, splits, 0, "val", max(cfg.eval_stride, 40))
    else:
        tr = common.fold_images(images, splits, fold, "train", cfg.train_stride)
        va = common.fold_images(images, splits, fold, "val", cfg.eval_stride)
    conv.write(tr["img_id"], "filled", threads)
    conv.write(va["img_id"], "raw", threads)
    d = conv.root / "folds"
    d.mkdir(parents=True, exist_ok=True)
    tag = f"fold{fold}_s{cfg.train_stride}_e{cfg.eval_stride}"
    (d / f"{tag}_train.txt").write_text("\n".join(str(conv.path(i, "filled")) for i in tr["img_id"]))
    (d / f"{tag}_val.txt").write_text("\n".join(str(conv.path(i, "raw")) for i in va["img_id"]))
    yaml = d / f"data_{tag}.yaml"
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(common.CLASSES))
    yaml.write_text(f"path: {d}\ntrain: {d / (tag + '_train.txt')}\nval: {d / (tag + '_val.txt')}\n"
                    f"names:\n{names}\n")
    return yaml, tr, va


# ---------------------------------------------------------------- NWD loss (optional)
def nwd(box1, box2, c=12.8, eps=1e-7):
    """Normalized Wasserstein Distance between xyxy boxes (Wang et al., 2021, arXiv:2110.13389)."""
    import torch
    cx1, cy1 = (box1[..., 0] + box1[..., 2]) / 2, (box1[..., 1] + box1[..., 3]) / 2
    cx2, cy2 = (box2[..., 0] + box2[..., 2]) / 2, (box2[..., 1] + box2[..., 3]) / 2
    w1, h1 = box1[..., 2] - box1[..., 0], box1[..., 3] - box1[..., 1]
    w2, h2 = box2[..., 2] - box2[..., 0], box2[..., 3] - box2[..., 1]
    d2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2 + ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4
    return torch.exp(-torch.sqrt(d2 + eps) / c)


def patch_nwd(ratio, c=12.8):
    """Mix NWD into the YOLO box loss: loss_iou = (1-r)*(1-CIoU) + r*(1-NWD).

    Monkeypatches ultralytics.utils.loss.BboxLoss.forward (Ultralytics 8.4). Boxes there are
    in grid units (pixels / stride); they are multiplied by the per-anchor stride so NWD and
    its constant c are in input pixels, as in the paper. DFL is unchanged.
    Returns a function that restores the original forward.
    """
    from ultralytics.utils import loss as L
    orig = L.BboxLoss.forward
    if ratio <= 0:
        return lambda: None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores,
                target_scores_sum, fg_mask, imgsz, stride):
        loss_iou, loss_dfl = orig(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                                  target_scores, target_scores_sum, fg_mask, imgsz, stride)
        idx = fg_mask.nonzero(as_tuple=True)
        weight = target_scores[idx].sum(-1, keepdim=True)
        pb, tb = (pred_bboxes * stride)[idx], (target_bboxes * stride)[idx]
        loss_nwd = ((1.0 - nwd(pb, tb, c=c).unsqueeze(-1)) * weight).sum() / target_scores_sum
        return (1 - ratio) * loss_iou + ratio * loss_nwd, loss_dfl

    L.BboxLoss.forward = forward
    return lambda: setattr(L.BboxLoss, "forward", orig)


def patch_rtdetr_loss(bbox=5.0, giou=2.0):
    """RT-DETR loss with other weights of the box terms: loss = class + bbox * L1 + giou * GIoU.

    Monkeypatches RTDETRDetectionModel.init_criterion (Ultralytics 8.4; defaults class 1, bbox 5,
    giou 2). The Hungarian matcher costs are unchanged, so only the training signal changes, not
    which query is matched to which object. Returns a function that restores the original.
    """
    from ultralytics.nn.tasks import RTDETRDetectionModel
    orig = RTDETRDetectionModel.init_criterion
    if bbox == 5.0 and giou == 2.0:
        return lambda: None

    def init_criterion(self):
        from ultralytics.models.utils.loss import RTDETRDetectionLoss
        return RTDETRDetectionLoss(nc=self.nc, use_vfl=True, loss_gain={"class": 1, "bbox": bbox, "giou": giou})

    RTDETRDetectionModel.init_criterion = init_criterion
    return lambda: setattr(RTDETRDetectionModel, "init_criterion", orig)


# ---------------------------------------------------------------- model / training
def ensure_weights(name, tries=8, wait=60):
    """Pretrained weights: local file, else a copy from attached inputs (e.g. an earlier notebook
    output), else download with retries (GitHub release downloads sometimes return 5xx)."""
    from ultralytics.utils.downloads import attempt_download_asset
    if Path(name).exists() or not str(name).endswith(".pt"):
        return name
    found = _input_files("/kaggle/input", Path(name).name)
    if found:
        shutil.copy(found[0], name)
        print(f"{name}: copied from {found[0]}")
        return name
    for i in range(tries):
        try:
            return attempt_download_asset(name)
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            print(f"download of {name} failed ({type(e).__name__}); retry {i + 1}/{tries - 1} in {wait}s", flush=True)
            time.sleep(wait)


def build_model(cfg):
    """COCO-pretrained Ultralytics model (optionally another architecture YAML with weights transferred)."""
    ensure_weights(cfg.weights)
    if cfg.model == "rtdetr":
        from ultralytics import RTDETR
        if cfg.num_queries != 300 or cfg.num_denoising != 100:
            return _rtdetr_with_queries(cfg)
        return RTDETR(cfg.weights)
    from ultralytics import YOLO
    if cfg.arch:
        if cfg.arch in CUSTOM_YOLO:
            return _load_custom_yolo(custom_yolo_yaml(cfg.arch), cfg.weights)
        return YOLO(cfg.arch).load(cfg.weights)   # transfer matching layers (e.g. the P2 head is new)
    return YOLO(cfg.weights)


# Custom YOLOv8 heads. The layers of yolov8.yaml keep their indices (0-21), so every pretrained
# layer that is kept is transferred; new layers are appended after them, and the new detection level
# is the last input of Detect (its regression / classification branches cv2.k, cv3.k for the old
# levels keep their indices and shapes, so they are transferred as well).
CUSTOM_YOLO = {
    # P2 level (stride 4): upsample the P3 output (layer 15) and fuse it with backbone layer 2 (stride 4)
    "yolov8s-p2last.yaml": dict(drop_p5=False, p2=True),
    # no P5 level (stride 32): the bottom-up path stops at P4 (layers 19-21 removed)
    "yolov8s-nop5.yaml": dict(drop_p5=True, p2=False),
    # both: levels P3, P4, P2
    "yolov8s-p2last-nop5.yaml": dict(drop_p5=True, p2=True),
}


def _load_custom_yolo(yaml_path, weights):
    """Build a custom YOLOv8 and transfer the pretrained weights by layer; the pretrained Detect
    (layer 22 of yolov8.yaml) is mapped to the index of the new Detect, whose first levels are the
    same P3 / P4 (/ P5) branches. Prints how many tensors were transferred."""
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import intersect_dicts
    model = YOLO(str(yaml_path))
    det = len(model.model.model) - 1
    pre = YOLO(weights).model.float().state_dict()   # Ultralytics resolves / downloads the file itself
    pre = {(f"model.{det}." + k[len("model.22."):] if k.startswith("model.22.") else k): v for k, v in pre.items()}
    sd = model.model.state_dict()
    inter = intersect_dicts(pre, sd)
    model.model.load_state_dict(inter, strict=False)
    model.overrides.pop("pretrained", None)
    model.ckpt, model.ckpt_path = {"model": model.model}, None   # train() then starts from these weights
    new = sorted({int(k.split(".")[1]) for k in sd if k not in inter})
    print(f"{Path(yaml_path).name}: transferred {len(inter)}/{len(sd)} tensors; new or partly new layers {new}")
    return model


def custom_yolo_yaml(name):
    import yaml as pyyaml
    from ultralytics.utils import ROOT
    spec = CUSTOM_YOLO[name]
    d = pyyaml.safe_load((ROOT / "cfg" / "models" / "v8" / "yolov8.yaml").read_text())
    d["scale"] = name.split("-")[0][-1]              # "s" from "yolov8s-..."
    head = d["head"][:-1]                            # layers 10-21 (Detect removed)
    levels = [15, 18, 21]
    if spec["drop_p5"]:
        head, levels = head[:-3], [15, 18]           # remove layers 19-21 (P5 path)
    if spec["p2"]:
        n = 10 + len(head)                           # index of the first new layer
        head += [[15, 1, "nn.Upsample", [None, 2, "nearest"]],
                 [[-1, 2], 1, "Concat", [1]],
                 [-1, 3, "C2f", [128]]]
        levels.append(n + 2)
    d["head"] = head + [[levels, 1, "Detect", ["nc"]]]
    out = Path("/tmp") / name
    out.write_text(pyyaml.safe_dump(d, sort_keys=False))
    return out


def _rtdetr_with_queries(cfg):
    """RT-DETR with a different number of decoder queries and/or denoising queries; pretrained weights
    are transferred (queries are selected from encoder features, so no weight depends on their number)."""
    import yaml as pyyaml
    from ultralytics import RTDETR
    from ultralytics.utils import ROOT
    src = ROOT / "cfg" / "models" / "rt-detr" / (cfg.arch or cfg.weights.replace(".pt", ".yaml"))
    d = pyyaml.safe_load(src.read_text())
    head = d["head"][-1]          # [[P3, P4, P5], 1, RTDETRDecoder, [nc]]
    head[-1] = ["nc", 256, cfg.num_queries]  # parsed as RTDETRDecoder(nc, ch, hd=256, nq)
    if cfg.num_denoising != 100:
        # RTDETRDecoder(nc, ch, hd, nq, ndp, nh, ndl, d_ffn, dropout, act, eval_idx, nd): defaults except nd
        head[-1] += [4, 8, 6, 1024, 0.0, None, -1, cfg.num_denoising]
    out = Path("/tmp") / f"rtdetr_nq{cfg.num_queries}_nd{cfg.num_denoising}.yaml"
    out.write_text(pyyaml.safe_dump(d, sort_keys=False))
    return RTDETR(str(out)).load(cfg.weights)


def backbone_lr_trainer(mult, n_backbone=10):
    """RT-DETR trainer whose optimizer gives the backbone (first `n_backbone` layers; 0-9 = HGNetv2
    in rtdetr-l.yaml) a learning rate lr0 * mult; the encoder/decoder keep lr0. Every Ultralytics
    parameter group (weights / bn / bias) is split in two, so weight decay and bias warmup are kept;
    the LambdaLR schedule then scales each group from its own initial lr."""
    from ultralytics.models.rtdetr.train import RTDETRTrainer
    from ultralytics.utils.torch_utils import unwrap_model

    class BackboneLRTrainer(RTDETRTrainer):
        def build_optimizer(self, model, *args, **kwargs):
            opt = super().build_optimizer(model, *args, **kwargs)
            bb = {id(p) for m in unwrap_model(model).model[:n_backbone] for p in m.parameters()}
            extra = []
            for g in opt.param_groups:
                back = [p for p in g["params"] if id(p) in bb]
                g["params"] = [p for p in g["params"] if id(p) not in bb]
                if back:
                    extra.append({**{k: v for k, v in g.items() if k != "params"},
                                  "params": back, "lr": g["lr"] * mult})
            for g in extra:
                opt.add_param_group(g)
            return opt

    return BackboneLRTrainer


def train_args(cfg, data_yaml, project, name):
    a = dict(data=str(data_yaml), epochs=cfg.epochs, imgsz=cfg.imgsz, batch=cfg.batch,
             optimizer=cfg.optimizer, lr0=cfg.lr0, lrf=cfg.lrf, momentum=cfg.momentum,
             weight_decay=cfg.weight_decay, warmup_epochs=cfg.warmup_epochs,
             warmup_bias_lr=cfg.warmup_bias_lr,
             close_mosaic=cfg.close_mosaic, amp=cfg.amp, cache=cfg.cache,
             mosaic=cfg.mosaic, scale=cfg.scale, fliplr=cfg.fliplr,
             seed=common.SEED, deterministic=True, workers=cfg.workers, device=cfg.device,
             project=str(Path(project).resolve()), name=name, exist_ok=True, plots=False, verbose=False,
             max_det=cfg.max_det, val=True)
    if cfg.model == "yolo":
        a.update(box=cfg.box, cls=cfg.cls, dfl=cfg.dfl)
    return a


def train_fold(cfg, data_yaml, run_dir):
    """Train (or resume) one fold; returns path to last.pt and training minutes."""
    run_dir = Path(run_dir)
    last = run_dir / "train" / "weights" / "last.pt"
    restore = (patch_nwd(cfg.nwd, cfg.nwd_c) if cfg.model == "yolo"
               else patch_rtdetr_loss(cfg.rt_bbox_gain, cfg.rt_giou_gain))
    t0 = time.time()
    try:
        extra = {}
        if cfg.model == "rtdetr" and cfg.backbone_lr_mult != 1.0:
            extra["trainer"] = backbone_lr_trainer(cfg.backbone_lr_mult)
        if last.exists() and not (run_dir / "train_done").exists():
            from ultralytics import RTDETR, YOLO
            (RTDETR if cfg.model == "rtdetr" else YOLO)(str(last)).train(resume=True, **extra)
        elif not last.exists():
            build_model(cfg).train(**train_args(cfg, data_yaml, run_dir, "train"), **extra)
    finally:
        restore()
    (run_dir / "train_done").touch()
    return last, (time.time() - t0) / 60


# ---------------------------------------------------------------- prediction
def load_trained(cfg, weights):
    from ultralytics import RTDETR, YOLO
    return (RTDETR if cfg.model == "rtdetr" else YOLO)(str(weights))


def _prec(cfg):
    """fp16 inference on GPU (Ultralytics 8.4 replaced `half` with `quantize`)."""
    return 16 if cfg.amp and cfg.device != "cpu" else None


def predict(model, paths, keys, cfg, imgsz=None, augment=False):
    """Predict image files; boxes are returned in ORIGINAL pixels (Ultralytics rescales them).

    keys: img_id (int, validation) or image_id (str, test), one per path.
    """
    rows = []
    imgsz = imgsz or cfg.imgsz
    for i in range(0, len(paths), cfg.pred_batch):
        res = model.predict(list(paths[i:i + cfg.pred_batch]), imgsz=imgsz, conf=cfg.conf,
                            iou=cfg.iou, max_det=cfg.max_det, quantize=_prec(cfg), augment=augment,
                            device=cfg.device, verbose=False)
        for k, r in zip(keys[i:i + cfg.pred_batch], res):
            b = r.boxes
            if b is None or len(b) == 0:
                continue
            xyxy = b.xyxy.float().cpu().numpy()
            rows.append(pd.DataFrame({"key": k, "cls": b.cls.int().cpu().numpy(),
                                      "score": b.conf.float().cpu().numpy(),
                                      "x": xyxy[:, 0], "y": xyxy[:, 1],
                                      "w": xyxy[:, 2] - xyxy[:, 0], "h": xyxy[:, 3] - xyxy[:, 1]}))
    cols = ["key", "cls", "score", "x", "y", "w", "h"]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)


def measure_fps(model, paths, cfg, imgsz=None):
    """Images/s at batch 1 (includes pre/post-processing), same convention as tv_train."""
    import torch
    imgsz = imgsz or cfg.imgsz
    paths = list(paths)[: cfg.fps_warmup + cfg.fps_iters]
    kw = dict(imgsz=imgsz, conf=cfg.conf, iou=cfg.iou, max_det=cfg.max_det, quantize=_prec(cfg),
              device=cfg.device, verbose=False)
    for p in paths[: cfg.fps_warmup]:
        model.predict(p, **kw)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for p in paths[cfg.fps_warmup:]:
        model.predict(p, **kw)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    n = len(paths) - cfg.fps_warmup
    return n / (time.time() - t0) if n > 0 else float("nan")


# ---------------------------------------------------------------- experiment loop
def run_experiment(cfg, images, anns, splits, folds, run_root, results_csv, conv,
                   log=True, threads=8, redo=()):
    """Train + predict every fold in `folds` (finished folds are skipped) and log OOF metrics
    once all 3 folds are done. Returns the OOF metrics table (or None if folds are missing).
    redo: folds to retrain from scratch even if finished (e.g. a fold whose training diverged)."""
    exp_dir = Path(run_root) / cfg.experiment
    exp_dir.mkdir(parents=True, exist_ok=True)
    for k in redo:
        if (exp_dir / f"fold{k}").exists():
            shutil.rmtree(exp_dir / f"fold{k}")
            (exp_dir / "logged").unlink(missing_ok=True)   # OOF metrics are logged again as a new run
            print(f"[{cfg.experiment}] fold {k}: removed for retraining")
    cfg_file = exp_dir / "config.json"
    if cfg_file.exists():
        old, new, default = json.loads(cfg_file.read_text()), cfg.key(), Config().key()
        # fields added to Config later are fine as long as they keep their default value
        diff = {k for k in old if old[k] != new.get(k)} | {k for k in new if k not in old and new[k] != default[k]}
        if diff:
            raise ValueError(f"{cfg_file} has another config ({sorted(diff)}); use a new experiment name")
    cfg_file.write_text(json.dumps(cfg.key(), indent=1))
    for k in folds:
        fdir = exp_dir / f"fold{k}"
        if (fdir / "done.json").exists():
            print(f"[{cfg.experiment}] fold {k}: done, skipped")
            continue
        common.seed_everything()
        yaml, tr, va = prepare_fold(conv, splits, k, cfg, threads)
        print(f"[{cfg.experiment}] fold {k}: {len(tr)} train / {len(va)} val frames")
        last, minutes = train_fold(cfg, yaml, fdir)
        model = load_trained(cfg, last)
        p = predict(model, va["path"].tolist(), va["img_id"].tolist(), cfg)
        p.rename(columns={"key": "img_id"}).to_csv(fdir / "preds.csv.gz", index=False)
        fps = measure_fps(model, va["path"].tolist(), cfg)
        hist = _history(fdir)
        (fdir / "done.json").write_text(json.dumps({"fps": fps, "train_min": minutes,
                                                    "history": hist}))
        shutil.copy(last, fdir / "final.pt")
        print(f"[{cfg.experiment}] fold {k}: {minutes:.1f} min, {fps:.1f} FPS")
    return finish_experiment(cfg, images, anns, splits, run_root, results_csv, log)


def _history(fdir):
    f = Path(fdir) / "train" / "results.csv"
    if not f.exists():
        return []
    h = pd.read_csv(f)
    h.columns = [c.strip() for c in h.columns]
    return h.to_dict("records")


def load_fold_preds(run_root, experiment):
    out = {}
    for k in range(common.N_FOLDS):
        f = Path(run_root) / experiment / f"fold{k}" / "preds.csv.gz"
        if f.exists():
            out[k] = pd.read_csv(f)
    return out


def load_done(run_root, experiment):
    out = {}
    for k in range(common.N_FOLDS):
        f = Path(run_root) / experiment / f"fold{k}" / "done.json"
        if f.exists():
            out[k] = json.loads(f.read_text())
    return out


def finish_experiment(cfg, images, anns, splits, run_root, results_csv, log=True):
    preds, done = load_fold_preds(run_root, cfg.experiment), load_done(run_root, cfg.experiment)
    if len(preds) < common.N_FOLDS:
        print(f"[{cfg.experiment}] {len(preds)}/{common.N_FOLDS} folds finished; OOF metrics later")
        return None
    m = common.evaluate_oof(preds, images, anns, splits, cfg.eval_stride, per_class=True)
    marker = Path(run_root) / cfg.experiment / "logged"
    if log and not marker.exists():
        common.log_results(m, results_csv, owner=cfg.owner, model=cfg.model,
                           experiment=cfg.experiment, train_stride=cfg.train_stride,
                           eval_stride=cfg.eval_stride, imgsz=cfg.imgsz, epochs=cfg.epochs,
                           fps=round(float(np.mean([d["fps"] for d in done.values()])), 2),
                           train_min=round(float(np.sum([d["train_min"] for d in done.values()])), 1),
                           notes=cfg.notes)
        marker.touch()
    return m


# ---------------------------------------------------------------- test submission
def predict_test(model, cfg, root=None, imgsz=None, augment=False):
    t = common.load_test_index(root)
    p = predict(model, t["path"].tolist(), t["image_id"].tolist(), cfg, imgsz=imgsz, augment=augment)
    return p.rename(columns={"key": "image_id"}), t


def _input_files(input_root, pattern, max_depth=3):
    """Files matching `pattern` under attached inputs, WITHOUT scanning the competition data
    (a recursive glob over its ~77k files on Kaggle's network disk takes minutes)."""
    root = Path(input_root)
    if not root.exists():
        return []
    skip = {"competitions", common.SLUG}
    out = []
    for top in sorted(p for p in root.iterdir() if p.is_dir() and p.name not in skip):
        for d in range(max_depth + 1):
            out += sorted(top.glob("/".join(["*"] * d + [pattern])))
    return out


def restore_from_inputs(run_root, input_root="/kaggle/input"):
    """Copy finished folds (done.json + preds + final.pt) from attached outputs of earlier
    notebook versions, so a new Kaggle session continues instead of retraining."""
    run_root = Path(run_root)
    n = 0
    for done in _input_files(input_root, "ul_runs/*/fold*/done.json"):
        fdir = done.parent
        dst = run_root / fdir.parent.name / fdir.name
        if (dst / "done.json").exists():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for f in ["done.json", "preds.csv.gz", "final.pt"]:
            if (fdir / f).exists():
                shutil.copy(fdir / f, dst / f)
        (dst / "train_done").touch()
        cfgf = fdir.parent / "config.json"
        if cfgf.exists() and not (dst.parent / "config.json").exists():
            shutil.copy(cfgf, dst.parent / "config.json")
        n += 1
    print(f"restored {n} finished folds from {input_root}")
    return n

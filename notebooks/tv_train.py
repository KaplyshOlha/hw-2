"""Shared train / predict / evaluate loop for the torchvision detectors (HW2, Person A).

Used by 02_faster_rcnn.ipynb and 03_retinanet.ipynb.

Pipeline per fold: build model -> precache frames (tv_dataset) -> SGD + warmup + multistep LR
with AMP -> checkpoint every epoch -> predict val frames -> FPS (batch 1) -> save preds.
`run_experiment` loops over folds, skips finished ones, resumes interrupted ones and logs
3-fold OOF metrics with common.log_results.

Run directory layout (all small except final.pt):
    <run_root>/<experiment>/config.json
    <run_root>/<experiment>/fold<k>/last.pt        full training state, deleted when the fold is done
    <run_root>/<experiment>/fold<k>/final.pt       fp16 weights for analysis / test inference
    <run_root>/<experiment>/fold<k>/preds.csv.gz   val predictions (common format, original px)
    <run_root>/<experiment>/fold<k>/done.json      fps, train minutes, per-epoch history

Sources: torchvision detection reference scripts (SGD, warmup, AMP);
mmdetection faster_rcnn_r50_fpn_giou_1x config (GIoU weight 10 on decoded boxes);
Rezatofighi et al. 2019 (GIoU); Goyal et al. 2017 (linear LR scaling, warmup);
Lin et al. 2017 (RetinaNet / focal loss: prior bias init, fg/bg IoU 0.5/0.4).
"""
import json
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from torchvision.models.detection import (FasterRCNN, FasterRCNN_ResNet50_FPN_Weights, RetinaNet,
                                          RetinaNet_ResNet50_FPN_Weights)
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection._utils import Matcher
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import (RetinaNetClassificationHead,
                                                    RetinaNetRegressionHead, _default_anchorgen)
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import RPNHead
from torchvision.ops import box_iou, generalized_box_iou_loss, sigmoid_focal_loss
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.ops.misc import FrozenBatchNorm2d

import common
import tv_dataset as tvd

RUNTIME_FIELDS = {"num_workers", "stop_after_epoch", "notes", "fps_warmup", "fps_iters",
                  "curve_every", "curve_stride", "anchors_json"}


# ---------------------------------------------------------------- config
@dataclass
class Config:
    experiment: str = "frcnn_base"
    model: str = "faster_rcnn"          # "faster_rcnn" | "retinanet"
    owner: str = "A"
    pretrained: bool = True             # COCO weights
    trainable_backbone_layers: int = 3  # torchvision default for fine-tuning
    imgsz: int = common.IMGSZ
    train_stride: int = common.TRAIN_STRIDE
    eval_stride: int = common.EVAL_STRIDE
    val_fill: bool = False              # False: val frames stay raw like test frames
    hflip_p: float = 0.5
    # optimisation
    epochs: int = 12
    batch_size: int = 8
    lr: float = 0.01                    # 0.02 * batch / 16 (linear scaling rule)
    momentum: float = 0.9
    weight_decay: float = 1e-4
    warmup_iters: int = 200
    warmup_factor: float = 1e-3
    milestones: tuple = (8, 11)         # epochs where LR is multiplied by gamma (1x-like)
    gamma: float = 0.1
    amp: bool = True
    grad_clip: float = 10.0             # 0 disables
    max_iters: int = 0                  # cap on iterations per epoch (quick mode)
    # anchors (None = torchvision default)
    anchor_sizes: Optional[list] = None
    anchor_ratios: list = field(default_factory=lambda: [0.5, 1.0, 2.0])
    anchor_key: Optional[str] = None    # key in anchors.json (fitted per fold), e.g. "x0.5"
    anchors_json: str = "anchors.json"
    # inference
    score_thresh: float = 0.001
    nms_thresh: float = 0.5
    detections_per_img: int = 300
    # Faster R-CNN
    rpn_pre_nms_top_n_train: int = 2000
    rpn_post_nms_top_n_train: int = 2000
    rpn_pre_nms_top_n_test: int = 1000
    rpn_post_nms_top_n_test: int = 1000
    rpn_batch_size_per_image: int = 256
    rpn_positive_fraction: float = 0.5
    box_batch_size_per_image: int = 512
    box_positive_fraction: float = 0.25
    box_loss: str = "smooth_l1"         # "smooth_l1" | "giou" (second stage)
    box_loss_weight: float = 10.0       # used for giou
    # RetinaNet
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    fg_iou_thresh: float = 0.5
    bg_iou_thresh: float = 0.4
    retina_reg_loss: str = "l1"         # "l1" (torchvision default) | "smooth_l1" | "giou" | "diou" | "ciou"
    # runtime only (do not change results)
    num_workers: int = 2
    fps_warmup: int = 10
    fps_iters: int = 50
    curve_every: int = 0                # >0: quick val mAP every N epochs (stride curve_stride)
    curve_stride: int = 40
    stop_after_epoch: int = 0           # >0: simulate an interruption (quick-mode resume test)
    notes: str = ""

    @property
    def label_offset(self):
        return 1 if self.model == "faster_rcnn" else 0

    @property
    def num_classes(self):
        return common.NUM_CLASSES + self.label_offset

    def key(self):
        """Fields that define the experiment (used to refuse resuming with another config)."""
        d = asdict(self)
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()
                if k not in RUNTIME_FIELDS}


def make_config(base=None, **changes):
    """New Config from a base Config (or defaults) with some fields changed."""
    return replace(base or Config(), **changes)


# ---------------------------------------------------------------- model
class GIoURoIHeads(RoIHeads):
    """RoIHeads whose box loss is GIoU on decoded boxes instead of smooth L1 on deltas."""
    box_loss_weight = 10.0

    def forward(self, features, proposals, image_shapes, targets=None):
        if not self.training:
            return super().forward(features, proposals, image_shapes, targets)
        proposals, _, labels, reg_targets = self.select_training_samples(proposals, targets)
        feats = self.box_head(self.box_roi_pool(features, proposals, image_shapes))
        class_logits, box_regression = self.box_predictor(feats)
        labels = torch.cat(labels, 0)
        reg_targets = torch.cat(reg_targets, 0).float()
        props = torch.cat(proposals, 0).float()
        loss_cls = torch.nn.functional.cross_entropy(class_logits.float(), labels)
        pos = torch.where(labels > 0)[0]
        if len(pos) == 0:
            loss_box = box_regression.sum() * 0.0
        else:
            deltas = box_regression.float().reshape(len(labels), -1, 4)[pos, labels[pos]]
            pred = self.box_coder.decode_single(deltas, props[pos])
            gt = self.box_coder.decode_single(reg_targets[pos], props[pos])
            loss_box = generalized_box_iou_loss(pred, gt, reduction="sum") / labels.numel()
            loss_box = loss_box * self.box_loss_weight
        return [], {"loss_classifier": loss_cls, "loss_box_reg": loss_box}


class FocalClassificationHead(RetinaNetClassificationHead):
    """RetinaNet classification head with tunable focal-loss alpha / gamma."""
    alpha, gamma = 0.25, 2.0

    def compute_loss(self, targets, head_outputs, matched_idxs):
        losses = []
        for t, logits, m in zip(targets, head_outputs["cls_logits"], matched_idxs):
            fg = m >= 0
            tgt = torch.zeros_like(logits)
            tgt[fg, t["labels"][m[fg]]] = 1.0
            valid = m != self.BETWEEN_THRESHOLDS
            losses.append(sigmoid_focal_loss(logits[valid], tgt[valid], alpha=self.alpha,
                                             gamma=self.gamma, reduction="sum") / max(1, int(fg.sum())))
        return sum(losses) / len(targets)


def _amp_backbone_forward(self, x):
    if getattr(self, "amp_enabled", False) and x.is_cuda:
        with torch.autocast("cuda", dtype=torch.float16):
            out = super(type(self), self).forward(x)
        return type(out)((k, v.float()) for k, v in out.items())
    return super(type(self), self).forward(x)


def set_backbone_amp(model, enabled):
    """Run only the backbone+FPN in fp16; heads and box decoding stay fp32 at inference.

    Full autocast decodes boxes in fp16 (~0.25-0.5 px steps at 640 px), which costs
    localisation accuracy on 10-20 px objects. State-dict keys are unchanged.
    """
    bb = model.backbone
    if not getattr(type(bb), "_amp_wrapped", False):
        bb.__class__ = type("Amp" + type(bb).__name__, (type(bb),),
                            {"forward": _amp_backbone_forward, "_amp_wrapped": True})
    bb.amp_enabled = bool(enabled)
    return model


def resolve_anchors(cfg, fold):
    """Anchor sizes per FPN level: explicit list, anchors.json[fold] entry, or None (default)."""
    if cfg.anchor_sizes is not None:
        sizes = cfg.anchor_sizes
        ratios = [list(cfg.anchor_ratios)] * len(sizes)
    elif cfg.anchor_key:
        with open(cfg.anchors_json) as f:
            entry = json.load(f)[f"fold{fold}@{cfg.imgsz}"]["retinanet"][cfg.anchor_key]
        sizes, ratios = entry["sizes"], entry["aspect_ratios"]
    else:
        return None
    sizes = [[s] if np.isscalar(s) else list(s) for s in sizes]
    if cfg.model == "faster_rcnn" and cfg.anchor_sizes is None:
        sizes = [s[:1] for s in sizes]  # one size per level, like the torchvision default
    return tuple(tuple(int(round(v)) for v in s) for s in sizes), \
        tuple(tuple(float(r) for r in rs) for rs in ratios)


def _coco_arch(cfg, cls, weights, **kwargs):
    """Same architecture as torchvision's *_resnet50_fpn builders (FrozenBN, eps=0 as in COCO_V1),
    with COCO weights loaded only if cfg.pretrained (so checkpoints load identically offline)."""
    backbone = resnet50(weights=None, norm_layer=FrozenBatchNorm2d)
    if cls is RetinaNet:
        backbone = _resnet_fpn_extractor(backbone, cfg.trainable_backbone_layers,
                                         returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(256, 256))
    else:
        backbone = _resnet_fpn_extractor(backbone, cfg.trainable_backbone_layers)
    model = cls(backbone, num_classes=91, min_size=cfg.imgsz, max_size=cfg.imgsz, **kwargs)
    if cfg.pretrained:
        model.load_state_dict(weights.get_state_dict(progress=False, check_hash=True))
    overwrite_eps(model, 0.0)
    return model


def build_model(cfg, fold=0):
    """COCO-pretrained torchvision detector with new class heads and the cfg settings."""
    anchors = resolve_anchors(cfg, fold)
    if cfg.model == "faster_rcnn":
        model = _coco_arch(cfg, FasterRCNN, FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
            box_score_thresh=cfg.score_thresh, box_nms_thresh=cfg.nms_thresh,
            box_detections_per_img=cfg.detections_per_img,
            rpn_pre_nms_top_n_train=cfg.rpn_pre_nms_top_n_train,
            rpn_post_nms_top_n_train=cfg.rpn_post_nms_top_n_train,
            rpn_pre_nms_top_n_test=cfg.rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_test=cfg.rpn_post_nms_top_n_test,
            rpn_batch_size_per_image=cfg.rpn_batch_size_per_image,
            rpn_positive_fraction=cfg.rpn_positive_fraction,
            box_batch_size_per_image=cfg.box_batch_size_per_image,
            box_positive_fraction=cfg.box_positive_fraction)
        if anchors is not None:
            gen = AnchorGenerator(*anchors)
            model.rpn.anchor_generator = gen
            n = gen.num_anchors_per_location()[0]
            if n != model.rpn.head.cls_logits.out_channels:
                model.rpn.head = RPNHead(model.backbone.out_channels, n)
        in_feat = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, cfg.num_classes)
        if cfg.box_loss == "giou":
            model.roi_heads.__class__ = GIoURoIHeads
            model.roi_heads.box_loss_weight = cfg.box_loss_weight
        elif cfg.box_loss != "smooth_l1":
            raise ValueError(f"unknown box_loss {cfg.box_loss}")
    elif cfg.model == "retinanet":
        model = _coco_arch(cfg, RetinaNet, RetinaNet_ResNet50_FPN_Weights.COCO_V1,
            score_thresh=cfg.score_thresh, nms_thresh=cfg.nms_thresh,
            detections_per_img=cfg.detections_per_img,
            fg_iou_thresh=cfg.fg_iou_thresh, bg_iou_thresh=cfg.bg_iou_thresh)
        if anchors is not None:
            model.anchor_generator = AnchorGenerator(*anchors)
        n = model.anchor_generator.num_anchors_per_location()[0]
        ch = model.backbone.out_channels
        if n != model.head.regression_head.bbox_reg.out_channels // 4:
            model.head.regression_head = RetinaNetRegressionHead(ch, n)
        old = model.head.classification_head
        head = FocalClassificationHead(ch, n, cfg.num_classes)
        if old.num_anchors == n:
            head.conv.load_state_dict(old.conv.state_dict())  # keep the COCO conv tower
        head.alpha, head.gamma = cfg.focal_alpha, cfg.focal_gamma
        model.head.classification_head = head
        if cfg.retina_reg_loss not in ("l1", "smooth_l1", "giou", "diou", "ciou"):
            raise ValueError(f"unknown retina_reg_loss {cfg.retina_reg_loss}")
        model.head.regression_head._loss_type = cfg.retina_reg_loss
    else:
        raise ValueError(f"unknown model {cfg.model}")
    return set_backbone_amp(model, cfg.amp)


def set_rpn_topn(model, pre=None, post=None):
    """Change the number of RPN proposals used at test time (no retraining needed)."""
    if pre is not None:
        model.rpn._pre_nms_top_n["testing"] = int(pre)
    if post is not None:
        model.rpn._post_nms_top_n["testing"] = int(post)


# ---------------------------------------------------------------- data
def make_datasets(cfg, images, anns, splits, fold, precache=True):
    """Train / val VisDroneDet for one fold (val frames raw unless cfg.val_fill)."""
    f = splits["folds"][fold]
    tr = tvd.VisDroneDet(images, anns, f["train_clips"], stride=cfg.train_stride,
                         max_side=cfg.imgsz, train=True, hflip_p=cfg.hflip_p,
                         label_offset=cfg.label_offset)
    va = tvd.VisDroneDet(images, anns, f["val_clips"], stride=cfg.eval_stride,
                         max_side=cfg.imgsz, train=False, label_offset=cfg.label_offset)
    if not cfg.val_fill:
        va.ign = {}
    if precache:
        tr.precache()
        va.precache()
    return tr, va


def make_loader(ds, batch_size, shuffle, num_workers, seed=common.SEED):
    g = torch.Generator()
    g.manual_seed(seed)
    num_workers = min(num_workers, os.cpu_count() or 1)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=tvd.collate_fn, worker_init_fn=tvd.seed_worker, generator=g,
                      pin_memory=torch.cuda.is_available(), persistent_workers=False)


# ---------------------------------------------------------------- schedule / checkpoints
def lr_at(cfg, it, iters_per_epoch):
    """Linear warmup then step decay at cfg.milestones (epochs)."""
    epoch = it / max(iters_per_epoch, 1)
    f = cfg.gamma ** sum(epoch >= m for m in cfg.milestones)
    warm = min(cfg.warmup_iters, iters_per_epoch * cfg.epochs)
    if it < warm:
        a = it / warm
        f *= cfg.warmup_factor * (1 - a) + a
    return cfg.lr * f


def _rng_state():
    s = {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def _set_rng_state(s):
    random.setstate(s["py"])
    np.random.set_state(s["np"])
    torch.set_rng_state(s["torch"].cpu())  # torch.load(map_location=cuda) moves RNG states to the GPU
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in s["cuda"]])


def _atomic_save(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_final_weights(model, path):
    sd = {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()}
    _atomic_save(sd, path)


def load_weights(model, path, device="cpu"):
    sd = torch.load(path, map_location=device, weights_only=True)
    ref = model.state_dict()
    model.load_state_dict({k: v.to(ref[k].dtype) for k, v in sd.items()})
    return model


def check_config(cfg, exp_dir):
    """Write config.json, or refuse to continue an experiment saved with different settings."""
    p = Path(exp_dir) / "config.json"
    if p.exists():
        old = json.loads(p.read_text())
        default = Config().key()
        diff = {k: (old.get(k), v) for k, v in cfg.key().items()
                if old.get(k, default.get(k)) != v}
        if diff:
            raise ValueError(f"{cfg.experiment}: config differs from the saved run {diff}. "
                             "Use a new EXPERIMENT name or delete the old run directory.")
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cfg.key(), indent=1))


# ---------------------------------------------------------------- predict / speed
@torch.no_grad()
def predict(model, ds, device, batch_size=8, num_workers=2):
    """Predictions on every frame of ds as a common-format DataFrame (original px)."""
    model.eval()
    loader = make_loader(ds, batch_size, False, num_workers)
    out = []
    for imgs, tgts in loader:
        imgs, _ = tvd.to_device(imgs, [], device)
        out.append(tvd.preds_to_df(model(imgs), tgts, label_offset=ds.label_offset))
    return pd.concat(out, ignore_index=True) if out else tvd.preds_to_df([], [])


@torch.no_grad()
def measure_fps(model, ds, device, warmup=10, iters=50):
    """Frames/s at batch 1 (host->GPU copy + model incl. pre/post-processing; no JPEG decode)."""
    model.eval()
    if len(ds) == 0 or iters <= 0:
        return float("nan")
    frames = [ds[i][0] for i in range(min(len(ds), 20))]
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    for i in range(warmup):
        model([frames[i % len(frames)].to(device).float().div_(255)])
    sync()
    t0 = time.perf_counter()
    for i in range(iters):
        model([frames[i % len(frames)].to(device).float().div_(255)])
    sync()
    return iters / (time.perf_counter() - t0)


# ---------------------------------------------------------------- training
class Interrupted(RuntimeError):
    """Raised when cfg.stop_after_epoch simulates an interrupted session."""


def train_fold(cfg, fold, images, anns, splits, run_root, device, log=print):
    """Train one fold (resuming from last.pt if present); returns the done.json dict."""
    fold_dir = Path(run_root) / cfg.experiment / f"fold{fold}"
    done_p = fold_dir / "done.json"
    if done_p.exists():
        log(f"[{cfg.experiment}] fold {fold}: already done, skipped")
        return json.loads(done_p.read_text())
    fold_dir.mkdir(parents=True, exist_ok=True)
    common.seed_everything(common.SEED)
    t_data = time.time()
    tr, va = make_datasets(cfg, images, anns, splits, fold)
    log(f"[{cfg.experiment}] fold {fold}: {len(tr)} train / {len(va)} val frames "
        f"cached in {time.time() - t_data:.0f}s")

    model = build_model(cfg, fold).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loader = make_loader(tr, cfg.batch_size, True, cfg.num_workers)
    ipe = len(loader) if not cfg.max_iters else min(len(loader), cfg.max_iters)

    start, history, train_sec = 0, [], 0.0
    last_p = fold_dir / "last.pt"
    if last_p.exists():
        ck = torch.load(last_p, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        _set_rng_state(ck["rng"])
        start, history, train_sec = ck["epoch"], ck["history"], ck["train_sec"]
        log(f"[{cfg.experiment}] fold {fold}: resumed after epoch {start}")

    for epoch in range(start, cfg.epochs):
        model.train()
        loader.generator.manual_seed(common.SEED * 1000 + epoch)
        t0, sums, n, skipped = time.time(), {}, 0, 0
        for i, (imgs, tgts) in enumerate(loader):
            if i >= ipe:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(cfg, epoch * ipe + i, ipe)
            imgs, tgts = tvd.to_device(imgs, tgts, device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                losses = model(imgs, tgts)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                skipped += 1
                opt.zero_grad(set_to_none=True)
                if skipped > 20:
                    raise FloatingPointError("more than 20 non-finite losses in one epoch")
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + v.detach()
            n += 1
        sec = time.time() - t0
        train_sec += sec
        rec = {"epoch": epoch + 1, "lr": opt.param_groups[0]["lr"], "sec": round(sec, 1),
               "skipped": skipped, **{k: float(v) / max(n, 1) for k, v in sums.items()}}
        rec["loss"] = sum(v for k, v in rec.items() if k.startswith("loss_") or
                          k in ("classification", "bbox_regression"))
        if cfg.curve_every and (epoch + 1) % cfg.curve_every == 0:
            rec["val_mAP"] = _curve_map(model, va, cfg, images, anns, device)
        history.append(rec)
        _atomic_save({"model": model.state_dict(), "opt": opt.state_dict(),
                      "scaler": scaler.state_dict(), "rng": _rng_state(), "epoch": epoch + 1,
                      "history": history, "train_sec": train_sec}, last_p)
        extra = f" val_mAP {rec['val_mAP']:.4f}" if "val_mAP" in rec else ""
        log(f"[{cfg.experiment}] fold {fold} epoch {epoch + 1}/{cfg.epochs} "
            f"loss {rec['loss']:.4f} lr {rec['lr']:.5f} {sec:.0f}s{extra}"
            + (f" skipped {skipped}" if skipped else ""))
        if cfg.stop_after_epoch and epoch + 1 >= cfg.stop_after_epoch and epoch + 1 < cfg.epochs:
            raise Interrupted(f"stopped after epoch {epoch + 1} (simulated interruption)")

    t0 = time.time()
    preds = predict(model, va, device, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    pred_sec = time.time() - t0
    fps = measure_fps(model, va, device, cfg.fps_warmup, cfg.fps_iters)
    preds.to_csv(fold_dir / "preds.csv.gz", index=False, float_format="%.2f")
    save_final_weights(model, fold_dir / "final.pt")
    info = {"experiment": cfg.experiment, "fold": fold, "fps": fps, "train_min": train_sec / 60,
            "predict_sec": pred_sec, "n_train": len(tr), "n_val": len(va),
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "history": history}
    done_p.write_text(json.dumps(info, indent=1))
    last_p.unlink(missing_ok=True)
    log(f"[{cfg.experiment}] fold {fold}: done, {len(preds)} boxes, {fps:.1f} FPS, "
        f"{train_sec / 60:.1f} train min")
    del model, opt, tr, va
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return info


def _curve_map(model, va, cfg, images, anns, device):
    """Fast val mAP on every curve_stride-th cached val frame (for training curves)."""
    sub = replace_subset(va, max(1, cfg.curve_stride // cfg.eval_stride))
    p = predict(model, sub, device, cfg.batch_size, 0)
    model.train()
    return common.evaluate(p, images, anns, sub.ids)["mAP"]


def replace_subset(ds, step):
    """Shallow copy of a dataset keeping every `step`-th frame (cache included)."""
    import copy
    s = copy.copy(ds)
    idx = list(range(0, len(ds), step))
    s.ids, s.paths = ds.ids[idx], [ds.paths[i] for i in idx]
    if ds.cache is not None:
        s.cache = [ds.cache[i] for i in idx]
    return s


def predict_from_checkpoints(cfg, images, anns, splits, run_root, device, folds=None,
                             rpn_topn=None, fps=True, log=print):
    """Re-run inference with the saved final.pt of each finished fold (no training).

    rpn_topn: optional (pre_nms, post_nms) test-time RPN proposal counts.
    Returns ({fold: preds}, {fold: fps}).
    """
    preds, speeds = {}, {}
    for k in sorted(load_done(run_root, cfg.experiment)):
        if folds is not None and k not in folds:
            continue
        _, va = make_datasets(cfg, images, anns, splits, k, precache=False)
        model = build_model(replace(cfg, pretrained=False), k)
        model = load_weights(model, Path(run_root) / cfg.experiment / f"fold{k}" / "final.pt").to(device)
        if rpn_topn is not None:
            set_rpn_topn(model, *rpn_topn)
        t0 = time.time()
        preds[k] = predict(model, va, device, cfg.batch_size, cfg.num_workers)
        speeds[k] = measure_fps(model, va, device, cfg.fps_warmup, cfg.fps_iters) if fps else np.nan
        log(f"[{cfg.experiment}] fold {k}: re-predicted {len(va)} frames in {time.time() - t0:.0f}s"
            + (f", rpn_topn={rpn_topn}" if rpn_topn else ""))
        del model
    return preds, speeds


def load_model(cfg, fold, run_root, device):
    """Model of a finished fold with its final weights, in eval mode."""
    model = build_model(replace(cfg, pretrained=False), fold)
    return load_weights(model, Path(run_root) / cfg.experiment / f"fold{fold}" / "final.pt").to(device).eval()


def load_fold_preds(run_root, experiment, folds=None):
    """{fold: preds DataFrame} for finished folds of an experiment."""
    out = {}
    for d in sorted((Path(run_root) / experiment).glob("fold*")):
        k = int(d.name[4:])
        if (d / "done.json").exists() and (folds is None or k in folds):
            out[k] = pd.read_csv(d / "preds.csv.gz")
    return out


def load_done(run_root, experiment):
    return {int(d.parent.name[4:]): json.loads(d.read_text())
            for d in sorted((Path(run_root) / experiment).glob("fold*/done.json"))}


def run_experiment(cfg, images, anns, splits, folds, run_root, results_path, device,
                   require_all_folds=True, log=print):
    """Train/predict all requested folds, then evaluate OOF and append to results.csv.

    Metrics are logged only when every fold of the split is finished (unless
    require_all_folds=False, used in quick mode). Returns the metrics DataFrame or None.
    """
    check_config(cfg, Path(run_root) / cfg.experiment)
    for k in folds:
        train_fold(cfg, k, images, anns, splits, run_root, device, log)
    all_folds = list(range(splits["n_folds"]))
    preds = load_fold_preds(run_root, cfg.experiment)
    have = sorted(preds)
    if require_all_folds and have != all_folds:
        log(f"[{cfg.experiment}] folds finished: {have}; metrics are logged once all "
            f"{len(all_folds)} folds are done")
        return None
    metrics = common.evaluate_oof(preds, images, anns, splits, eval_stride=cfg.eval_stride,
                                  per_class=True)
    done = load_done(run_root, cfg.experiment)
    fps = float(np.nanmean([done[k]["fps"] for k in have]))
    tmin = float(np.mean([done[k]["train_min"] for k in have]))
    logged = Path(run_root) / cfg.experiment / "logged.json"
    if not logged.exists():
        common.log_results(metrics, path=results_path, owner=cfg.owner, model=cfg.model,
                           experiment=cfg.experiment, train_stride=cfg.train_stride,
                           eval_stride=cfg.eval_stride, imgsz=cfg.imgsz, epochs=cfg.epochs,
                           fps=round(fps, 1), train_min=round(tmin, 1), notes=cfg.notes)
        logged.write_text(json.dumps({"folds": have}))
        log(f"[{cfg.experiment}] metrics appended to {results_path}")
    return metrics


# ---------------------------------------------------------------- Kaggle resume helpers
def find_runs_dirs(root, name="runs", max_depth=5):
    """Directories called `name` under root (previous notebook versions attached as input)."""
    out, root = [], Path(root)
    if not root.exists():
        return out
    skip = {"train", "val", "test", "images", "labels", "ignore"}
    for cur, dirs, _ in os.walk(root):
        depth = len(Path(cur).relative_to(root).parts)
        if Path(cur).name == name and Path(cur) != root:
            out.append(Path(cur))
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if d not in skip] if depth < max_depth else []
    return out


def restore_from_inputs(run_root, results_path, input_root="/kaggle/input", name="runs",
                        log=print):
    """Copy finished/partial folds and results.csv from attached previous-version outputs.

    A fold is copied when it is finished there but not here, or when its checkpoint there
    is from a later epoch. results.csv files next to those run dirs are merged into ours.
    """
    run_root = Path(run_root)
    n_copied = 0
    for src_runs in find_runs_dirs(input_root, name):
        for src in sorted(src_runs.glob("*/fold*")):
            dst = run_root / src.parent.name / src.name
            if (dst / "done.json").exists():
                continue
            take = (src / "done.json").exists()
            if not take and (src / "last.pt").exists():
                ep = lambda p: torch.load(p, map_location="cpu", weights_only=False)["epoch"]
                take = not (dst / "last.pt").exists() or ep(src / "last.pt") > ep(dst / "last.pt")
            if take:
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                n_copied += 1
            cfg_src, cfg_dst = src.parent / "config.json", dst.parent / "config.json"
            if cfg_src.exists() and not cfg_dst.exists():
                shutil.copy(cfg_src, cfg_dst)
            lg_src, lg_dst = src.parent / "logged.json", dst.parent / "logged.json"
            if lg_src.exists() and not lg_dst.exists():
                shutil.copy(lg_src, lg_dst)
        res = src_runs.parent / Path(results_path).name
        if res.exists():
            common.merge_results(str(res), str(results_path))
    log(f"restored {n_copied} fold dirs from {input_root}")
    return n_copied


# ---------------------------------------------------------------- results tables
def summary(results_path, model=None, owner=None, experiments=None):
    """Latest logged run per experiment: OOF + mean±std metrics, per-class AP, FPS."""
    if not Path(results_path).exists():
        return pd.DataFrame()
    r = pd.read_csv(results_path, dtype={"fold": str})
    if model:
        r = r[r["model"] == model]
    if owner:
        r = r[r["owner"].astype(str) == str(owner)]
    if experiments:
        r = r[r["experiment"].isin(experiments)]
    if r.empty:
        return pd.DataFrame()
    last = r.sort_values("timestamp").groupby("experiment")["run_id"].last()
    r = r[r["run_id"].isin(last)]
    rows = []
    for exp, g in r.groupby("experiment", sort=False):
        g = g.set_index("fold")
        row = {"experiment": exp}
        for m in ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR500"]:
            ok = m in g.columns
            row[f"{m}_oof"] = g.loc["oof", m] if ok and "oof" in g.index else np.nan
            row[f"{m}_mean"] = g.loc["mean", m] if ok and "mean" in g.index else np.nan
            row[f"{m}_std"] = g.loc["std", m] if ok and "std" in g.index else np.nan
        for c in common.CLASSES:
            if f"AP_{c}" in g.columns and "oof" in g.index:
                row[f"AP_{c}"] = g.loc["oof", f"AP_{c}"]
        row["fps"] = g["fps"].iloc[0]
        row["train_min"] = g["train_min"].iloc[0]
        row["timestamp"] = g["timestamp"].iloc[0]
        rows.append(row)
    order = experiments or None
    out = pd.DataFrame(rows).set_index("experiment")
    return out.reindex([e for e in order if e in out.index]) if order else out


# ---------------------------------------------------------------- analysis helpers
SIZE_BINS = [0, 8, 16, 32, 64, 128, 1e9]
SIZE_LABELS = ["<8", "8-16", "16-32", "32-64", "64-128", ">128"]


def _size_bin(side):
    return pd.cut(side, SIZE_BINS, labels=SIZE_LABELS, right=False)


def error_analysis(preds, images, anns, img_ids, score_thr=0.3, iou_thr=0.5, ignore_cover=0.5):
    """Greedy matching of thresholded predictions to GT on the given frames.

    Returns (gt, dt): gt rows with matched / confused_with / best IoU, size, occlusion and
    frame density; dt rows with an error type in {tp, cls, loc, dup, bkg}. Predictions
    mostly inside ignored regions are dropped (like COCO crowd matching).
    """
    ids = np.unique(np.asarray(img_ids, dtype=np.int64))
    a = anns[anns["img_id"].isin(ids)]
    p = preds[preds["img_id"].isin(ids) & (preds["score"] >= score_thr)]
    obj = {k: g for k, g in a[a["iscrowd"] == 0].groupby("img_id")}
    ign = {k: g[["x", "y", "w", "h"]].to_numpy() for k, g in a[a["iscrowd"] == 1].groupby("img_id")}
    pg = {k: g.sort_values("score", ascending=False) for k, g in p.groupby("img_id")}
    gts, dts = [], []
    for iid in ids:
        g = obj.get(iid, a.iloc[:0])
        d = pg.get(iid, p.iloc[:0])
        H, W = images.loc[iid, ["height", "width"]]
        if iid in ign and len(d):
            cov = common.box_coverage(d[["x", "y", "w", "h"]].to_numpy(), common.ignore_mask(H, W, ign[iid]))
            d = d[cov < ignore_cover]
        gb = torch.tensor(g[["x", "y", "w", "h"]].to_numpy(np.float64))
        db = torch.tensor(d[["x", "y", "w", "h"]].to_numpy(np.float64))
        gb[:, 2:] += gb[:, :2]
        db[:, 2:] += db[:, :2]
        iou = box_iou(db, gb).numpy() if len(db) and len(gb) else np.zeros((len(db), len(gb)))
        gcls, dcls = g["cls"].to_numpy(), d["cls"].to_numpy()
        matched = np.full(len(g), -1)
        confused = np.full(len(g), -1)
        types = []
        for j in range(len(d)):
            same = gcls == dcls[j]
            cand = np.where(same & (matched < 0) & (iou[j] >= iou_thr))[0]
            if len(cand):
                m = cand[np.argmax(iou[j, cand])]
                matched[m] = j
                types.append("tp")
            elif len(gb) and (same & (iou[j] >= iou_thr)).any():
                types.append("dup")
            elif len(gb) and ((~same) & (iou[j] >= iou_thr)).any():
                k = np.where(~same)[0][np.argmax(iou[j, ~same])]
                if matched[k] < 0:
                    confused[k] = dcls[j]
                types.append("cls")
            elif len(gb) and (same & (iou[j] >= 0.1)).any():
                types.append("loc")
            else:
                types.append("bkg")
        side = np.sqrt(g["w"].to_numpy() * g["h"].to_numpy())
        gts.append(pd.DataFrame({
            "img_id": iid, "cls": gcls, "x": g["x"].to_numpy(), "y": g["y"].to_numpy(),
            "w": g["w"].to_numpy(), "h": g["h"].to_numpy(), "side": side,
            "occlusion": g["occlusion"].to_numpy(), "n_in_frame": len(g),
            "matched": matched >= 0, "confused_with": np.where(matched >= 0, -1, confused),
            "best_iou": iou.max(0) if len(db) else np.zeros(len(g))}))
        dts.append(d.assign(type=types))
    gt = pd.concat(gts, ignore_index=True)
    dt = pd.concat(dts, ignore_index=True)
    gt["size_bin"] = _size_bin(gt["side"])
    return gt, dt


@torch.no_grad()
def proposal_recall(model, ds, device, topn=(100, 300, 1000, 2000), iou_thr=0.5,
                    max_images=300, det_score=0.05):
    """Recall of RPN proposals (top-N) vs final detections, by GT size at network scale.

    Shows what each stage contributes: the RPN must cover the object with some proposal,
    the second stage must classify it and refine the box. Also returns mean best IoU of
    proposals vs final boxes for covered objects (box refinement).
    """
    model.eval()
    sub = replace_subset(ds, max(1, len(ds) // max_images))
    saved = (model.rpn._pre_nms_top_n["testing"], model.rpn._post_nms_top_n["testing"])
    cap = {}
    h = model.rpn.register_forward_hook(lambda m, i, o: cap.__setitem__("p", o[0]))
    rows = []
    try:
        set_rpn_topn(model, max(saved[0], max(topn)), max(topn))
        props = []
        for i in range(len(sub)):
            img, t = sub[i]
            model([img.to(device).float().div_(255)])
            props.append(cap["p"][0].float().cpu())
        set_rpn_topn(model, *saved)
        for i in range(len(sub)):
            img, t = sub[i]
            out = model([img.to(device).float().div_(255)])[0]
            gtb = t["boxes"]
            if len(gtb) == 0:
                continue
            side = ((gtb[:, 2] - gtb[:, 0]) * (gtb[:, 3] - gtb[:, 1])).clamp(min=0).sqrt().numpy()
            r = {"side": side}
            P = props[i]
            for n in topn:
                iou = box_iou(gtb, P[:n]) if len(P) else torch.zeros(len(gtb), 1)
                r[f"prop@{n}"] = (iou.max(1).values >= iou_thr).numpy() if iou.numel() else np.zeros(len(gtb), bool)
            r["prop_best_iou"] = box_iou(gtb, P[:saved[1]]).max(1).values.numpy() if len(P) else np.zeros(len(gtb))
            keep = out["scores"].cpu() >= det_score
            db, dl = out["boxes"].float().cpu()[keep], out["labels"].cpu()[keep]
            if len(db):
                iou = box_iou(gtb, db)
                iou_same = iou * (t["labels"][:, None] == dl[None, :])
                r["det_any_class"] = (iou.max(1).values >= iou_thr).numpy()
                r["det_correct_class"] = (iou_same.max(1).values >= iou_thr).numpy()
                r["det_best_iou"] = iou_same.max(1).values.numpy()
            else:
                z = np.zeros(len(gtb))
                r.update(det_any_class=z.astype(bool), det_correct_class=z.astype(bool), det_best_iou=z)
            rows.append(pd.DataFrame(r))
    finally:
        h.remove()
        set_rpn_topn(model, *saved)
    df = pd.concat(rows, ignore_index=True)
    df["size_bin"] = _size_bin(df["side"])
    return df


CLASS_COLORS = {"tp": (40, 200, 40), "missed": (230, 30, 30), "cls": (220, 0, 220),
                "loc": (255, 160, 0), "dup": (0, 170, 255), "bkg": (255, 230, 0)}


def draw_errors(img_id, images, anns, gt, dt, crop=None):
    """RGB frame with ignored regions shaded, missed GT red, TPs green, FP types coloured.

    crop: (x0, y0, x1, y1) in original px, or "auto" = window around the errors.
    """
    path = images.loc[img_id, "path"]
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    a = anns[(anns["img_id"] == img_id) & (anns["iscrowd"] == 1)]
    over = img.copy()
    for x, y, w, h in a[["x", "y", "w", "h"]].to_numpy():
        cv2.rectangle(over, (int(x), int(y)), (int(x + w), int(y + h)), (128, 128, 128), -1)
    img = cv2.addWeighted(over, 0.45, img, 0.55, 0)
    g, d = gt[gt["img_id"] == img_id], dt[dt["img_id"] == img_id]
    th = max(1, round(max(img.shape[:2]) / 900))
    for _, r in d.iterrows():
        c = CLASS_COLORS[r["type"]]
        cv2.rectangle(img, (int(r.x), int(r.y)), (int(r.x + r.w), int(r.y + r.h)), c, th)
    for _, r in g[~g["matched"]].iterrows():
        cv2.rectangle(img, (int(r.x), int(r.y)), (int(r.x + r.w), int(r.y + r.h)),
                      CLASS_COLORS["missed"], th)
    if crop == "auto":
        err = pd.concat([g[~g["matched"]][["x", "y", "w", "h"]],
                         d[d["type"] != "tp"][["x", "y", "w", "h"]]])
        if len(err):
            H, W = img.shape[:2]
            cw, ch = W // 2, H // 2
            cx = float(np.median(err["x"] + err["w"] / 2))
            cy = float(np.median(err["y"] + err["h"] / 2))
            x0 = int(np.clip(cx - cw / 2, 0, W - cw))
            y0 = int(np.clip(cy - ch / 2, 0, H - ch))
            crop = (x0, y0, x0 + cw, y0 + ch)
        else:
            crop = None
    if crop:
        x0, y0, x1, y1 = crop
        img = img[y0:y1, x0:x1]
    return img


# ---------------------------------------------------------------- RetinaNet anchor analysis
def anchor_generator(cfg, fold=0):
    """AnchorGenerator that build_model would use for cfg (RetinaNet)."""
    anchors = resolve_anchors(cfg, fold)
    return AnchorGenerator(*anchors) if anchors is not None else _default_anchorgen()


def _grid_anchors(gen, h, w, strides=(8, 16, 32, 64, 128), divisible=32):
    """All anchors (xyxy) for an h x w network input padded like GeneralizedRCNNTransform."""
    H, W = int(np.ceil(h / divisible) * divisible), int(np.ceil(w / divisible) * divisible)
    feats = []
    for s in strides:
        fh, fw = (H // s, W // s) if s <= 32 else (int(np.ceil(feats[-1].shape[-2] / 2)),
                                                    int(np.ceil(feats[-1].shape[-1] / 2)))
        feats.append(torch.zeros(1, 1, fh, fw))
    return gen(ImageList(torch.zeros(1, 3, H, W), [(h, w)]), feats)[0], [f.shape[-2] * f.shape[-1] for f in feats]


@torch.no_grad()
def anchor_match_stats(cfgs, ds, fold=0, thresholds=((0.5, 0.4),), max_images=200):
    """How RetinaNet's IoU matcher assigns anchors to GT boxes, without training a model.

    cfgs: {name: Config}; ds: training VisDroneDet (boxes at network scale, ignore-filtered).
    Uses torchvision's Matcher(fg, bg, allow_low_quality_matches=True) exactly as RetinaNet does.
    Returns (gt, img): per-GT rows (side, best IoU, positives, level of the best anchor) and
    per-image rows (anchors, positives, ignored 'between' anchors, negatives).
    """
    sub = replace_subset(ds, max(1, len(ds) // max_images))
    samples = []
    for i in range(len(sub)):
        img, boxes, _, _, _ = sub.cache[i] if sub.cache is not None else sub.prepare(i)
        samples.append((img.shape[0], img.shape[1], torch.from_numpy(np.asarray(boxes, np.float32))))
    gts, ims = [], []
    for name, cfg in cfgs.items():
        gen, cache = anchor_generator(cfg, fold), {}
        for h, w, b in samples:
            if (h, w) not in cache:
                a, counts = _grid_anchors(gen, h, w)
                per_loc = gen.num_anchors_per_location()
                lvl = torch.cat([torch.full((c * k,), j) for j, (c, k) in enumerate(zip(counts, per_loc))])
                cache[(h, w)] = (a, lvl)
            a, lvl = cache[(h, w)]
            iou = box_iou(b, a) if len(b) else None
            side = (b[:, 2:] - b[:, :2]).clamp(min=0).prod(1).sqrt().numpy() if len(b) else np.zeros(0)
            for fg, bg in thresholds:
                tag = dict(variant=name, fg=fg, bg=bg)
                if iou is None:
                    ims.append({**tag, "anchors": len(a), "pos": 0, "between": 0, "neg": len(a), "n_gt": 0})
                    continue
                m = Matcher(fg, bg, allow_low_quality_matches=True)(iou)
                npos = torch.bincount(m[m >= 0], minlength=len(b)).numpy()
                best = iou.max(1)
                gts.append(pd.DataFrame({**tag, "side": side, "best_iou": best.values.numpy(),
                                         "n_pos": npos, "natural_pos": (best.values >= fg).numpy(),
                                         "best_level": lvl[best.indices].numpy() + 3}))
                ims.append({**tag, "anchors": len(a), "pos": int((m >= 0).sum()),
                            "between": int((m == Matcher.BETWEEN_THRESHOLDS).sum()),
                            "neg": int((m == Matcher.BELOW_LOW_THRESHOLD).sum()), "n_gt": len(b)})
    gt = pd.concat(gts, ignore_index=True) if gts else pd.DataFrame(
        columns=["variant", "fg", "bg", "side", "best_iou", "n_pos", "natural_pos", "best_level"])
    gt["size_bin"] = _size_bin(gt["side"].astype(float))
    return gt, pd.DataFrame(ims)


def f1_threshold(preds, images, anns, img_ids, grid=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6), step=4):
    """Score threshold with the best F1 (IoU 0.5, same class) on every `step`-th frame.

    Focal-loss sigmoid scores and softmax scores are calibrated differently, so error
    analyses of two detectors are compared each at its own F1-optimal threshold.
    """
    ids = np.unique(np.asarray(img_ids, dtype=np.int64))[::step]
    rows = []
    for t in grid:
        gt, dt = error_analysis(preds, images, anns, ids, score_thr=t)
        tp = int((dt["type"] == "tp").sum()) if len(dt) else 0
        p, r = tp / max(len(dt), 1), tp / max(len(gt), 1)
        rows.append({"thr": t, "precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9),
                     "dets_per_frame": len(dt) / max(len(ids), 1)})
    df = pd.DataFrame(rows).set_index("thr")
    return float(df["f1"].idxmax()), df

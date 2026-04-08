#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate Gaussian Grouping (instance id map) segmentation with open-vocabulary prompts using GroundingDINO
for instance semantic labeling, then evaluate on Split Ground Truth.

Inputs:
  --pred_inst_dir: dir of instance id png (16-bit grayscale, each pixel is instance id; 0 = background)
  --images_dir: dir of RGB images (same basename)
  --gt_split_root: contains fused/, layer_zaha_kept/, layer_ai_filled/ (npy semantic maps)

Outputs:
  - 3-level evaluation (aligned to your previous 3-row scripts):
      1) building (binary, id=200) with method#3 but restricted to GT coverage:
            eval_mask = (GT_valid) & (GT_is_building | Pred_is_building)
      2) parts (multiclass) evaluated only on GT pixels in part_ids (default 1/2/3/12)
      3) nonbuilding (multiclass) evaluated only on GT pixels in nonbuild_ids (default 101/103/104)
  - visualization: 3 rows x 3 cols
"""

import argparse
import glob
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch

# GroundingDINO (needs to be installed in your env)
try:
    from GroundingDINO.groundingdino.util.inference import load_model as dino_load_model
    from GroundingDINO.groundingdino.util.inference import predict as dino_predict
except Exception:
    dino_load_model = None
    dino_predict = None


# =========================
# Defaults (改成你自己的)
# =========================
DEFAULT_IMAGES_DIR = "/workspace/test_subset5"
DEFAULT_PRED_INST_DIR = "/workspace/CityGMLGaussian/output/subset_building5_gaga/test/ours_10000/objects_test"
DEFAULT_GT_SPLIT_ROOT = "/workspace/zaha_eval/gt/subset5_499_dist60_test"
DEFAULT_OUTPUT_DIR = "/workspace/zaha_eval/eval_results/subset5_gaga_sem_test"

DEFAULT_CLASS_MAPPING_PATH = "class_mapping.json"
DEFAULT_CLASS_COLORS_PATH = "class_colors.json"
DEFAULT_GT_MERGE_MAP_PATH = "gt_merge_map.json"

DEFAULT_DINO_CONFIG = "/workspace/zaha_eval/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DEFAULT_DINO_CHECKPOINT = "/workspace/zaha_eval/GroundingDINO/weights/groundingdino_swint_ogc.pth"

DEFAULT_BOX_THRESH = 0.35
DEFAULT_TEXT_THRESH = 0.25

# instance->class assignment thresholds
DEFAULT_INSTANCE_MIN_OVERLAP = 0.30   # area(I∩B)/area(I) >= this
DEFAULT_INSTANCE_MIN_SCORE = 0.20     # dino_score * overlap >= this else ignore

DEFAULT_SAVE_VISUALIZATIONS = True
DEFAULT_NUM_IMAGES = None
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 3-row evaluation defaults
DEFAULT_BUILDING_ID = 200
DEFAULT_PART_IDS = [1, 2, 3, 12]
DEFAULT_NONBUILD_IDS = [101, 103, 104]
DEFAULT_PARTS_MERGE_TARGET = 1

DEFAULT_GT_LAYER_BUILDING = "fused"   # fused|zaha|ai
DEFAULT_GT_LAYER_PARTS = "zaha"       # fused|zaha|ai
DEFAULT_GT_LAYER_NONBUILD = "ai"      # fused|zaha|ai


# =========================
# Logger
# =========================
def get_logger(name, log_file=None, log_level=logging.INFO):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    sh = logging.StreamHandler()
    handlers = [sh]
    if log_file is not None:
        fh = logging.FileHandler(log_file, "w")
        handlers.append(fh)
    fmt = logging.Formatter("%(asctime)s - %(message)s")
    for h in handlers:
        h.setFormatter(fmt)
        h.setLevel(log_level)
        logger.addHandler(h)
    logger.setLevel(log_level)
    return logger


# =========================
# GT helpers
# =========================
def apply_label_merges(label_map: np.ndarray, merge_map: Dict[int, List[int]]) -> np.ndarray:
    if not merge_map:
        return label_map
    out = label_map.copy()
    for target_id, source_ids in merge_map.items():
        target_id = int(target_id)
        for src_id in source_ids:
            src_id = int(src_id)
            if src_id == target_id:
                continue
            out[out == src_id] = target_id
    return out


def load_ground_truth_layer(
    layer_dir: str,
    image_names: List[str],
    class_mapping: Dict[int, Union[str, List[str]]],
    logger: logging.Logger,
    label_merge_map: Optional[Dict[int, List[int]]] = None,
    layer_name: str = "GT",
) -> Dict[str, Dict]:
    logger.info(f"   Loading {layer_name} from: {layer_dir}")
    gt_data = {}
    if not os.path.exists(layer_dir):
        logger.error(f"   [ERROR] Directory not found: {layer_dir}")
        return {}

    valid_class_ids = set(class_mapping.keys())

    for img_name in tqdm(image_names, desc=f"Loading {layer_name}", leave=False):
        p = os.path.join(layer_dir, f"{img_name}.npy")
        if not os.path.exists(p):
            continue
        sem = np.load(p)

        if label_merge_map:
            sem = apply_label_merges(sem, label_merge_map)

        invalid_mask = (sem >= 0) & (~np.isin(sem, list(valid_class_ids)))
        sem[invalid_mask] = -1

        cov = sem >= 0
        cls = np.unique(sem[cov]).tolist()
        gt_data[img_name] = {"semantic_map": sem, "coverage_mask": cov, "classes": cls}

    logger.info(f"   Loaded {len(gt_data)} images for {layer_name}.")
    return gt_data


# =========================
# Metrics
# =========================
def compute_multiclass_metrics(pred_semantic, gt_semantic, eval_mask, class_mapping):
    pred_valid = pred_semantic[eval_mask]
    gt_valid = gt_semantic[eval_mask]
    if gt_valid.size == 0:
        return None

    gt_classes = np.unique(gt_valid).tolist()
    results = {
        "class_iou": {},
        "pixel_count": int(gt_valid.size),
        "correct_count": int((pred_valid == gt_valid).sum()),
    }

    for cls_id in gt_classes:
        cls_name = class_mapping.get(int(cls_id), f"ID_{cls_id}")
        if isinstance(cls_name, list):
            cls_name = cls_name[0]

        gt_cls_mask = (gt_valid == cls_id)
        pred_cls_mask = (pred_valid == cls_id)

        inter = np.logical_and(pred_cls_mask, gt_cls_mask).sum()
        union = np.logical_or(pred_cls_mask, gt_cls_mask).sum()

        results["class_iou"][cls_name] = float(inter / union) if union > 0 else 0.0

    results["mean_iou"] = float(np.mean(list(results["class_iou"].values()))) if results["class_iou"] else 0.0
    results["pixel_acc"] = float(results["correct_count"] / results["pixel_count"])
    return results


def compute_binary_metrics_union_mask(pred_is_pos: np.ndarray, gt_is_pos: np.ndarray, eval_mask: np.ndarray):
    """
    Building-level:
      eval_mask = (GT_valid) & (GT_pos | Pred_pos)  -> excludes TN and doesn't penalize outside GT coverage.
    """
    pb = pred_is_pos[eval_mask].astype(bool)
    gb = gt_is_pos[eval_mask].astype(bool)
    if gb.size == 0:
        return None

    tp = int(np.logical_and(pb, gb).sum())
    fp = int(np.logical_and(pb, ~gb).sum())
    fn = int(np.logical_and(~pb, gb).sum())

    iou = tp / (tp + fp + fn + 1e-9)
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = (2 * prec * rec) / (prec + rec + 1e-9)

    return {
        "iou": float(iou),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tp": tp, "fp": fp, "fn": fn,
        "pixel_count": int(gb.size),
    }


# =========================
# Visualization: 3 rows
# =========================
def visualize_three_row_error(
    # row1 building
    pred_building: np.ndarray, gt_building_merged: np.ndarray,
    # row2 parts
    pred_parts: np.ndarray, gt_parts: np.ndarray,
    # row3 nonbuilding
    pred_non: np.ndarray, gt_non: np.ndarray,
    class_colors: Dict[int, List[int]],
    save_path: str,
    building_id: int,
    part_ids: List[int],
    nonbuild_ids: List[int],
):
    """
    Layout (3x3):
      Row 1: [GT Building] [Pred Building] [Error Building]
        - Error only on eval_mask = (GT_valid) & (GT_is_b | Pred_is_b)
      Row 2: [GT Parts]    [Pred Parts]    [Error Parts]  (only show/eval GT in part_ids)
      Row 3: [GT NonB]     [Pred NonB]     [Error NonB]   (only show/eval GT in nonbuild_ids)
    """

    def build_palette(unique_ids):
        palette = {}
        cmap = plt.cm.get_cmap("tab20", max(len(unique_ids), 1) + 5)
        for i, uid in enumerate(sorted(list(unique_ids))):
            if uid < 0:
                continue
            if class_colors and uid in class_colors:
                palette[uid] = np.array(class_colors[uid], dtype=np.uint8)
            else:
                palette[uid] = (np.array(cmap(i)[:3]) * 255).astype(np.uint8)
        return palette

    def colorize(mask, palette):
        h, w = mask.shape
        col = np.zeros((h, w, 3), dtype=np.uint8)
        col[mask < 0] = [30, 30, 30]
        for cid in np.unique(mask):
            if cid < 0:
                continue
            col[mask == cid] = palette.get(int(cid), np.array([128, 128, 128], dtype=np.uint8))
        return col

    def error_map_multiclass_on_mask(pred, gt, eval_mask):
        err = np.full((*gt.shape, 3), 30, dtype=np.uint8)
        correct = (pred == gt) & eval_mask
        wrong = (pred != gt) & eval_mask
        err[correct] = [0, 255, 0]
        err[wrong] = [255, 0, 0]
        return err

    def error_map_building(pred_is_b, gt_is_b, eval_mask):
        err = np.full((*gt_is_b.shape, 3), 30, dtype=np.uint8)
        correct = (pred_is_b == gt_is_b) & eval_mask
        wrong = (pred_is_b != gt_is_b) & eval_mask
        err[correct] = [0, 255, 0]
        err[wrong] = [255, 0, 0]
        return err

    unique_ids = set(np.unique(gt_parts)) | set(np.unique(gt_non)) | set(np.unique(pred_parts)) | set(np.unique(pred_non)) | {building_id}
    palette = build_palette(unique_ids)

    # -------- Row1: building (GT-coverage restricted union) --------
    gt_valid_b = (gt_building_merged >= 0)
    gt_is_b = (gt_building_merged == building_id)
    pred_is_b = (pred_building == building_id)
    eval_b = gt_valid_b & (gt_is_b | pred_is_b)

    gt_b_vis = np.full_like(gt_building_merged, -1)
    pr_b_vis = np.full_like(pred_building, -1)
    gt_b_vis[gt_is_b] = building_id
    pr_b_vis[pred_is_b] = building_id
    vis_gt_b = colorize(gt_b_vis, palette)
    vis_pr_b = colorize(pr_b_vis, palette)
    err_b = error_map_building(pred_is_b, gt_is_b, eval_b)

    # -------- Row2: parts (only GT pixels in part_ids) --------
    gt_valid_p = (gt_parts >= 0) & np.isin(gt_parts, part_ids)
    gt_p_vis = gt_parts.copy()
    pr_p_vis = pred_parts.copy()
    gt_p_vis[~gt_valid_p] = -1
    pr_p_vis[~gt_valid_p] = -1  # show only evaluated region
    vis_gt_p = colorize(gt_p_vis, palette)
    vis_pr_p = colorize(pr_p_vis, palette)
    err_p = error_map_multiclass_on_mask(pr_p_vis, gt_p_vis, gt_valid_p)

    # -------- Row3: nonbuilding (only GT pixels in nonbuild_ids) --------
    gt_valid_n = (gt_non >= 0) & np.isin(gt_non, nonbuild_ids)
    gt_n_vis = gt_non.copy()
    pr_n_vis = pred_non.copy()
    gt_n_vis[~gt_valid_n] = -1
    pr_n_vis[~gt_valid_n] = -1
    vis_gt_n = colorize(gt_n_vis, palette)
    vis_pr_n = colorize(pr_n_vis, palette)
    err_n = error_map_multiclass_on_mask(pr_n_vis, gt_n_vis, gt_valid_n)

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    axes[0, 0].imshow(vis_gt_b); axes[0, 0].set_title(f"GT Building (merged to {building_id})")
    axes[0, 1].imshow(vis_pr_b); axes[0, 1].set_title("Pred Building (DINO labeling)")
    axes[0, 2].imshow(err_b);    axes[0, 2].set_title("Error (Building) [GTcov & (GT|Pred)]\nGreen=TP, Red=FP/FN")

    axes[1, 0].imshow(vis_gt_p); axes[1, 0].set_title("GT Parts (Zaha)")
    axes[1, 1].imshow(vis_pr_p); axes[1, 1].set_title("Pred Parts")
    axes[1, 2].imshow(err_p);    axes[1, 2].set_title("Error (Parts)\nGreen=Correct, Red=Wrong")

    axes[2, 0].imshow(vis_gt_n); axes[2, 0].set_title("GT Non-building (AI)")
    axes[2, 1].imshow(vis_pr_n); axes[2, 1].set_title("Pred Non-building")
    axes[2, 2].imshow(err_n);    axes[2, 2].set_title("Error (Non-building)\nGreen=Correct, Red=Wrong")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# =========================
# DINO utilities
# =========================
def load_rgb(image_path: str) -> np.ndarray:
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(image_path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _preprocess_for_dino(image_rgb: np.ndarray, target_size: int = 800, max_size: int = 1333) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Resize shortest side to target_size but keep longest <= max_size.
    Return:
      - tensor (3,H,W) normalized
      - resized_size (H_resized, W_resized)
    """
    pil = Image.fromarray(image_rgb)  # RGB
    w, h = pil.size

    scale = target_size / min(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    if max(new_h, new_w) > max_size:
        scale = max_size / max(new_h, new_w)
        new_h, new_w = int(round(new_h * scale)), int(round(new_w * scale))

    pil = pil.resize((new_w, new_h), resample=Image.BILINEAR)

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return tfm(pil), (new_h, new_w)


def dino_detect(model, image_rgb: np.ndarray, prompt: str, box_th: float, text_th: float, device: str):
    """
    Preprocess RGB -> torch tensor -> dino_predict -> map boxes back to original pixels.
    Handles CUDA OOM with fallback.
    """
    img_t, (new_h, new_w) = _preprocess_for_dino(image_rgb, target_size=800, max_size=1333)

    def _run(dev: str, img_tensor: torch.Tensor):
        with torch.no_grad():
            if dev.startswith("cuda"):
                with torch.cuda.amp.autocast(True):
                    return dino_predict(
                        model=model,
                        image=img_tensor.to(dev),
                        caption=prompt,
                        box_threshold=box_th,
                        text_threshold=text_th,
                        device=dev,
                    )
            else:
                return dino_predict(
                    model=model,
                    image=img_tensor.to(dev),
                    caption=prompt,
                    box_threshold=box_th,
                    text_threshold=text_th,
                    device=dev,
                )

    try:
        boxes, logits, _phrases = _run(device, img_t)
    except torch.cuda.OutOfMemoryError:
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        # retry smaller
        img_small, (new_h, new_w) = _preprocess_for_dino(image_rgb, target_size=640, max_size=1024)
        try:
            boxes, logits, _phrases = _run(device, img_small)
        except torch.cuda.OutOfMemoryError:
            boxes, logits, _phrases = _run("cpu", img_t)

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(logits, dtype=np.float32).reshape(-1)

    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    H0, W0 = image_rgb.shape[:2]

    # ---- 1) normalize -> resized pixel space if needed ----
    if np.max(boxes) <= 1.5:
        # could be normalized cxcywh or normalized xyxy; scale first
        boxes_scaled = boxes.copy()
        boxes_scaled[:, [0, 2]] *= float(new_w)
        boxes_scaled[:, [1, 3]] *= float(new_h)
    else:
        boxes_scaled = boxes.copy()

    # ---- 2) detect format: xyxy vs cxcywh ----
    # Heuristic: if many boxes have x2<x1 or y2<y1, it's probably cxcywh
    bad_xyxy = np.mean((boxes_scaled[:, 2] < boxes_scaled[:, 0]) | (boxes_scaled[:, 3] < boxes_scaled[:, 1]))
    if bad_xyxy > 0.3:
        # treat as cxcywh in resized pixel space
        cx = boxes_scaled[:, 0]
        cy = boxes_scaled[:, 1]
        w = boxes_scaled[:, 2]
        h = boxes_scaled[:, 3]
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    else:
        boxes_xyxy = boxes_scaled

    # ---- 3) map resized pixel coords back to original pixel coords ----
    sx = W0 / float(new_w)
    sy = H0 / float(new_h)
    boxes_xyxy[:, [0, 2]] *= sx
    boxes_xyxy[:, [1, 3]] *= sy

    # ---- 4) sanitize: ensure x1<x2, y1<y2 and clip ----
    x1 = np.minimum(boxes_xyxy[:, 0], boxes_xyxy[:, 2])
    y1 = np.minimum(boxes_xyxy[:, 1], boxes_xyxy[:, 3])
    x2 = np.maximum(boxes_xyxy[:, 0], boxes_xyxy[:, 2])
    y2 = np.maximum(boxes_xyxy[:, 1], boxes_xyxy[:, 3])

    x1 = np.clip(x1, 0, W0 - 1)
    y1 = np.clip(y1, 0, H0 - 1)
    x2 = np.clip(x2, 0, W0 - 1)
    y2 = np.clip(y2, 0, H0 - 1)

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    return boxes_xyxy.astype(np.float32), scores.astype(np.float32)


def build_queries_for_ids(class_mapping: Dict[int, Union[str, List[str]]], ids: List[int]) -> Tuple[List[str], np.ndarray]:
    """
    Flatten prompts only for given class ids.
    """
    queries = []
    query_ids = []
    for cid in ids:
        cid = int(cid)
        if cid not in class_mapping:
            continue
        lbl = class_mapping[cid]
        if isinstance(lbl, list):
            for s in lbl:
                queries.append(str(s))
                query_ids.append(cid)
        else:
            queries.append(str(lbl))
            query_ids.append(cid)
    return queries, np.array(query_ids, dtype=np.int32)


# =========================
# Instance labeling: instance -> class
# =========================
def overlap_ratio_instance_in_box(inst_mask: np.ndarray, box_xyxy: np.ndarray) -> float:
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy.tolist()]
    h, w = inst_mask.shape
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inst_area = float(inst_mask.sum())
    if inst_area <= 0:
        return 0.0
    inter = float(inst_mask[y1:y2, x1:x2].sum())
    return inter / inst_area


def label_instances_with_dino(
    pred_inst: np.ndarray,
    image_rgb: np.ndarray,
    dino_model,
    queries: List[str],
    query_ids: np.ndarray,
    box_th: float,
    text_th: float,
    min_overlap: float,
    min_score: float,
    device: str,
) -> Dict[int, int]:
    """
    Return: inst_id -> class_id (each instance chooses best score among all prompts/boxes)
    """
    inst_ids = np.unique(pred_inst)
    inst_ids = inst_ids[(inst_ids != 0) & (inst_ids != -1)]  # 0 background; -1 ignore
    if len(inst_ids) == 0 or len(queries) == 0:
        return {}

    inst_masks = {int(iid): (pred_inst == iid) for iid in inst_ids}
    best_class = {int(iid): -1 for iid in inst_ids}
    best_score = {int(iid): 0.0 for iid in inst_ids}

    for qi, prompt in enumerate(queries):
        cid = int(query_ids[qi])
        boxes, scores = dino_detect(dino_model, image_rgb, prompt, box_th, text_th, device)
        if boxes.shape[0] == 0:
            continue

        for bi in range(boxes.shape[0]):
            b = boxes[bi]
            s = float(scores[bi])

            for iid, m in inst_masks.items():
                overlap = overlap_ratio_instance_in_box(m, b)
                if overlap < min_overlap:
                    continue
                sc = s * overlap
                if sc > best_score[iid]:
                    best_score[iid] = sc
                    best_class[iid] = cid

    out = {}
    for iid in inst_ids:
        iid = int(iid)
        if best_class[iid] >= 0 and best_score[iid] >= min_score:
            out[iid] = int(best_class[iid])
    return out


def rasterize_instance_labels(pred_inst: np.ndarray, inst2cls: Dict[int, int]) -> np.ndarray:
    pred_sem = np.full_like(pred_inst, -1, dtype=np.int32)
    for iid, cid in inst2cls.items():
        pred_sem[pred_inst == iid] = int(cid)
    return pred_sem


# =========================
# Main evaluation
# =========================
def evaluate(args, logger):
    device = args.device

    dir_fused = os.path.join(args.gt_split_root, "fused")
    dir_zaha = os.path.join(args.gt_split_root, "layer_zaha_kept")
    dir_ai = os.path.join(args.gt_split_root, "layer_ai_filled")

    logger.info("Target Evaluation Directories:")
    logger.info(f"  > Fused: {dir_fused}")
    logger.info(f"  > Zaha : {dir_zaha}")
    logger.info(f"  > AI   : {dir_ai}")

    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "visualizations_3rows")
    if args.save_visualizations:
        os.makedirs(vis_dir, exist_ok=True)

    with open(args.class_mapping, "r") as f:
        class_mapping = {int(k): v for k, v in json.load(f).items()}

    # merge maps: split into parts vs building (like your previous scripts)
    gt_merge_map_all = None
    if args.gt_merge_map and os.path.exists(args.gt_merge_map):
        with open(args.gt_merge_map, "r") as f:
            raw = json.load(f)
        gt_merge_map_all = {int(k): [int(v) for v in vals] for k, vals in raw.items()}

    parts_merge_map = None
    building_merge_map = None
    if gt_merge_map_all:
        if args.parts_merge_target in gt_merge_map_all:
            parts_merge_map = {int(args.parts_merge_target): gt_merge_map_all[int(args.parts_merge_target)]}
        if args.building_id in gt_merge_map_all:
            building_merge_map = {int(args.building_id): gt_merge_map_all[int(args.building_id)]}

    class_colors = {}
    if args.class_colors and os.path.exists(args.class_colors):
        with open(args.class_colors, "r") as f:
            class_colors = {int(k): v for k, v in json.load(f).items()}

    # collect names from pred_inst png
    pred_files = sorted(glob.glob(os.path.join(args.pred_inst_dir, "*.png")))
    names_all = [Path(p).stem for p in pred_files]
    if args.num_images:
        names_all = names_all[: args.num_images]

    # load GT layers (use parts_merge_map only, to keep parts alignment consistent)
    gt_fused = load_ground_truth_layer(dir_fused, names_all, class_mapping, logger, parts_merge_map, "Fused")
    gt_zaha = load_ground_truth_layer(dir_zaha, names_all, class_mapping, logger, parts_merge_map, "Zaha")
    gt_ai = load_ground_truth_layer(dir_ai, names_all, class_mapping, logger, parts_merge_map, "AI")

    common = sorted(list(set(gt_fused.keys()) & set(names_all)))
    logger.info(f"Evaluating on {len(common)} images (must have fused GT + pred).")

    # build prompt sets for 3 rows
    building_id = int(args.building_id)
    part_ids = [int(x) for x in args.part_ids]
    nonbuild_ids = [int(x) for x in args.nonbuild_ids]

    building_queries, building_qids = build_queries_for_ids(class_mapping, [building_id])
    parts_queries, parts_qids = build_queries_for_ids(class_mapping, part_ids)
    non_queries, non_qids = build_queries_for_ids(class_mapping, nonbuild_ids)

    logger.info(f"[Prompts] building: {len(building_queries)}, parts: {len(parts_queries)}, nonbuild: {len(non_queries)}")

    # load DINO
    if dino_load_model is None:
        raise ImportError("GroundingDINO not available. Please ensure groundingdino is installed and importable.")
    if not os.path.exists(args.dino_config):
        raise FileNotFoundError(f"DINO config not found: {args.dino_config}")
    if not os.path.exists(args.dino_checkpoint):
        raise FileNotFoundError(f"DINO checkpoint not found: {args.dino_checkpoint}")

    logger.info("Loading GroundingDINO model...")
    dino_model = dino_load_model(args.dino_config, args.dino_checkpoint)
    dino_model = dino_model.to(device)
    dino_model.eval()

    # choose GT per row
    def pick_gt(img_name: str, which: str) -> Optional[Dict]:
        if which == "fused":
            return gt_fused.get(img_name)
        if which == "zaha":
            return gt_zaha.get(img_name)
        if which == "ai":
            return gt_ai.get(img_name)
        # allow legacy strings
        if which == "layer_zaha_kept":
            return gt_zaha.get(img_name)
        if which == "layer_ai_filled":
            return gt_ai.get(img_name)
        return gt_fused.get(img_name)

    # logs aligned with previous 3-row script
    metrics_log = {
        "building": [],     # binary union restricted to GT coverage
        "parts": [],        # multiclass on GT pixels in part_ids
        "nonbuilding": [],  # multiclass on GT pixels in nonbuild_ids
    }

    for name in tqdm(common, desc="Evaluating"):
        pred_path = os.path.join(args.pred_inst_dir, f"{name}.png")

        # 16-bit grayscale instance id map
        pred_inst = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)
        if pred_inst is None:
            continue
        if pred_inst.ndim == 3:
            pred_inst = pred_inst[:, :, 0]
        # IMPORTANT: keep ids as int32
        pred_inst = pred_inst.astype(np.int32)

        # find RGB
        img_path = None
        for ext in [".png", ".jpg", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
            p = os.path.join(args.images_dir, name + ext)
            if os.path.exists(p):
                img_path = p
                break
        if img_path is None:
            logger.warning(f"[Skip] RGB image not found for {name}")
            continue
        image_rgb = load_rgb(img_path)

        # resize pred_inst to fused GT size (canonical)
        gt_base = gt_fused[name]["semantic_map"]
        H, W = gt_base.shape
        if pred_inst.shape != (H, W):
            pred_inst = cv2.resize(pred_inst, (W, H), interpolation=cv2.INTER_NEAREST)

        # ===== DEBUG 1: instance id map stats =====
        u = np.unique(pred_inst)
        logger.info(f"[{name}] pred_inst unique count={len(u)}, min={u.min()}, max={u.max()}, nonzero={(pred_inst>0).sum()}")
        
        # ===== DEBUG 2: DINO sanity check on a simple prompt =====
        boxes_test, scores_test = dino_detect(dino_model, image_rgb, "window", args.box_thresh, args.text_thresh, device)
        logger.info(f"[{name}] DINO test prompt='window': boxes={len(boxes_test)}")
        if len(boxes_test) > 0:
            logger.info(f"[{name}] DINO test top score={float(scores_test.max()):.3f}, box0={boxes_test[0].tolist()}")
        # -----------------------
        # 1) Building prediction (only building prompts)
        # -----------------------
        inst2b = label_instances_with_dino(
            pred_inst=pred_inst,
            image_rgb=image_rgb,
            dino_model=dino_model,
            queries=building_queries,
            query_ids=building_qids,
            box_th=args.box_thresh,
            text_th=args.text_thresh,
            min_overlap=args.inst_min_overlap,
            min_score=args.inst_min_score,
            device=device,
        )
        pred_building = rasterize_instance_labels(pred_inst, inst2b)  # only 200 or -1

        # -----------------------
        # 2) Parts prediction (only part prompts)
        # -----------------------
        inst2p = label_instances_with_dino(
            pred_inst=pred_inst,
            image_rgb=image_rgb,
            dino_model=dino_model,
            queries=parts_queries,
            query_ids=parts_qids,
            box_th=args.box_thresh,
            text_th=args.text_thresh,
            min_overlap=args.inst_min_overlap,
            min_score=args.inst_min_score,
            device=device,
        )
        pred_parts = rasterize_instance_labels(pred_inst, inst2p)
        if parts_merge_map:
            pred_parts = apply_label_merges(pred_parts, parts_merge_map)

        # -----------------------
        # 3) Non-building prediction (only nonbuild prompts)
        # -----------------------
        inst2n = label_instances_with_dino(
            pred_inst=pred_inst,
            image_rgb=image_rgb,
            dino_model=dino_model,
            queries=non_queries,
            query_ids=non_qids,
            box_th=args.box_thresh,
            text_th=args.text_thresh,
            min_overlap=args.inst_min_overlap,
            min_score=args.inst_min_score,
            device=device,
        )
        pred_non = rasterize_instance_labels(pred_inst, inst2n)
        if parts_merge_map:
            pred_non = apply_label_merges(pred_non, parts_merge_map)

        # -----------------------
        # Building GT + metrics (GT coverage restricted union)
        # -----------------------
        gt_pack_b = pick_gt(name, args.gt_layer_building)
        if gt_pack_b is not None:
            gt_b = gt_pack_b["semantic_map"]
            gt_bm = apply_label_merges(gt_b, building_merge_map) if building_merge_map else gt_b

            gt_valid = (gt_bm >= 0)
            gt_is_b = (gt_bm == building_id)
            pred_is_b = (pred_building == building_id)

            eval_mask_b = gt_valid & (gt_is_b | pred_is_b)  # ✅ critical requirement
            bm = compute_binary_metrics_union_mask(pred_is_b, gt_is_b, eval_mask_b)
            if bm:
                metrics_log["building"].append(bm)

        # -----------------------
        # Parts GT + metrics (only GT pixels in part_ids)
        # -----------------------
        gt_pack_p = pick_gt(name, args.gt_layer_parts)
        if gt_pack_p is not None:
            gt_p = gt_pack_p["semantic_map"]
            eval_mask_p = (gt_p >= 0) & np.isin(gt_p, part_ids)
            pm = compute_multiclass_metrics(pred_parts, gt_p, eval_mask_p, class_mapping)
            if pm:
                metrics_log["parts"].append(pm)

        # -----------------------
        # Non-building GT + metrics (only GT pixels in nonbuild_ids)
        # -----------------------
        gt_pack_n = pick_gt(name, args.gt_layer_nonbuild)
        if gt_pack_n is not None:
            gt_n = gt_pack_n["semantic_map"]
            eval_mask_n = (gt_n >= 0) & np.isin(gt_n, nonbuild_ids)
            nm = compute_multiclass_metrics(pred_non, gt_n, eval_mask_n, class_mapping)
            if nm:
                metrics_log["nonbuilding"].append(nm)

        # -----------------------
        # Visualization (3 rows)
        # -----------------------
        if args.save_visualizations and (gt_pack_b is not None) and (gt_pack_p is not None) and (gt_pack_n is not None):
            gt_b = gt_pack_b["semantic_map"]
            gt_bm = apply_label_merges(gt_b, building_merge_map) if building_merge_map else gt_b

            visualize_three_row_error(
                pred_building, gt_bm,
                pred_parts, gt_pack_p["semantic_map"],
                pred_non, gt_pack_n["semantic_map"],
                class_colors,
                os.path.join(vis_dir, f"{name}_3rows.png"),
                building_id=building_id,
                part_ids=part_ids,
                nonbuild_ids=nonbuild_ids,
            )

    # =========================
    # Aggregate report (aligned with your previous 3-row scripts)
    # =========================
    final_report = {}

    logger.info("\n" + "=" * 70)
    logger.info("FINAL 3-ROW EVALUATION RESULTS (GT coverage restricted)")
    logger.info("=" * 70)

    # building
    bdata = metrics_log["building"]
    if bdata:
        final_report["building"] = {
            "N": int(len(bdata)),
            "IoU": float(np.mean([d["iou"] for d in bdata])),
            "Precision": float(np.mean([d["precision"] for d in bdata])),
            "Recall": float(np.mean([d["recall"] for d in bdata])),
            "F1": float(np.mean([d["f1"] for d in bdata])),
        }
        logger.info(f"\n[BUILDING] (ID={building_id}, GT={args.gt_layer_building}, N={len(bdata)})")
        logger.info(f"  IoU:       {final_report['building']['IoU']:.4f}")
        logger.info(f"  Precision: {final_report['building']['Precision']:.4f}")
        logger.info(f"  Recall:    {final_report['building']['Recall']:.4f}")
        logger.info(f"  F1:        {final_report['building']['F1']:.4f}")
    else:
        logger.warning("\n[BUILDING] No valid metrics.")

    # parts
    pdata = metrics_log["parts"]
    if pdata:
        mean_iou = float(np.mean([d["mean_iou"] for d in pdata]))
        mean_acc = float(np.mean([d["pixel_acc"] for d in pdata]))
        class_ious = defaultdict(list)
        for d in pdata:
            for c, iou in d["class_iou"].items():
                class_ious[c].append(iou)
        per_class_iou = {c: float(np.mean(v)) for c, v in class_ious.items()}

        final_report["parts"] = {
            "N": int(len(pdata)),
            "mIoU": mean_iou,
            "pixel_acc": mean_acc,
            "per_class_iou": per_class_iou,
        }

        logger.info(f"\n[PARTS] (IDs={part_ids}, GT={args.gt_layer_parts}, N={len(pdata)})")
        logger.info(f"  mIoU:      {mean_iou:.4f}")
        logger.info(f"  Pixel Acc: {mean_acc:.4f}")
        logger.info("  Per Class IoU:")
        for c, iou in sorted(per_class_iou.items()):
            logger.info(f"    {c:<25}: {iou:.4f}")
    else:
        logger.warning("\n[PARTS] No valid metrics.")

    # nonbuilding
    ndata = metrics_log["nonbuilding"]
    if ndata:
        mean_iou = float(np.mean([d["mean_iou"] for d in ndata]))
        mean_acc = float(np.mean([d["pixel_acc"] for d in ndata]))
        class_ious = defaultdict(list)
        for d in ndata:
            for c, iou in d["class_iou"].items():
                class_ious[c].append(iou)
        per_class_iou = {c: float(np.mean(v)) for c, v in class_ious.items()}

        final_report["nonbuilding"] = {
            "N": int(len(ndata)),
            "mIoU": mean_iou,
            "pixel_acc": mean_acc,
            "per_class_iou": per_class_iou,
        }

        logger.info(f"\n[NON-BUILDING] (IDs={nonbuild_ids}, GT={args.gt_layer_nonbuild}, N={len(ndata)})")
        logger.info(f"  mIoU:      {mean_iou:.4f}")
        logger.info(f"  Pixel Acc: {mean_acc:.4f}")
        logger.info("  Per Class IoU:")
        for c, iou in sorted(per_class_iou.items()):
            logger.info(f"    {c:<25}: {iou:.4f}")
    else:
        logger.warning("\n[NON-BUILDING] No valid metrics.")

    out_json = os.path.join(args.output_dir, "three_rows_results.json")
    with open(out_json, "w") as f:
        json.dump(final_report, f, indent=2)

    logger.info(f"\n3-row results saved to: {out_json}")
    logger.info(f"Visualizations saved to: {vis_dir}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--images_dir", type=str, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--pred_inst_dir", type=str, default=DEFAULT_PRED_INST_DIR)
    parser.add_argument("--gt_split_root", type=str, default=DEFAULT_GT_SPLIT_ROOT)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--class_mapping", type=str, default=DEFAULT_CLASS_MAPPING_PATH)
    parser.add_argument("--class_colors", type=str, default=DEFAULT_CLASS_COLORS_PATH)
    parser.add_argument("--gt_merge_map", type=str, default=DEFAULT_GT_MERGE_MAP_PATH)

    parser.add_argument("--dino_config", type=str, default=DEFAULT_DINO_CONFIG)
    parser.add_argument("--dino_checkpoint", type=str, default=DEFAULT_DINO_CHECKPOINT)
    parser.add_argument("--box_thresh", type=float, default=DEFAULT_BOX_THRESH)
    parser.add_argument("--text_thresh", type=float, default=DEFAULT_TEXT_THRESH)

    parser.add_argument("--inst_min_overlap", type=float, default=DEFAULT_INSTANCE_MIN_OVERLAP)
    parser.add_argument("--inst_min_score", type=float, default=DEFAULT_INSTANCE_MIN_SCORE)

    parser.add_argument("--save_visualizations", action="store_true", default=DEFAULT_SAVE_VISUALIZATIONS)
    parser.add_argument("--num_images", type=int, default=DEFAULT_NUM_IMAGES)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)

    # 3-row params
    parser.add_argument("--building_id", type=int, default=DEFAULT_BUILDING_ID)
    parser.add_argument("--parts_merge_target", type=int, default=DEFAULT_PARTS_MERGE_TARGET)
    parser.add_argument("--part_ids", nargs="+", type=int, default=DEFAULT_PART_IDS)
    parser.add_argument("--nonbuild_ids", nargs="+", type=int, default=DEFAULT_NONBUILD_IDS)

    parser.add_argument("--gt_layer_building", type=str, default=DEFAULT_GT_LAYER_BUILDING, help="fused|zaha|ai")
    parser.add_argument("--gt_layer_parts", type=str, default=DEFAULT_GT_LAYER_PARTS, help="fused|zaha|ai")
    parser.add_argument("--gt_layer_nonbuild", type=str, default=DEFAULT_GT_LAYER_NONBUILD, help="fused|zaha|ai")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger("gg_dino_3rows_eval", os.path.join(args.output_dir, "eval.log"))

    evaluate(args, logger)


if __name__ == "__main__":
    main()
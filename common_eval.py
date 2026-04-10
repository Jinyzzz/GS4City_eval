#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Common utilities for unified 2-level evaluation.

Levels:
  - whole: building-only binary evaluation
  - part : global fine semantic classes from class_mapping / ignore

GT logic:
  - DO NOT use fused GT
  - load zaha GT + ai GT
  - fuse them into one fine-level GT with priority:
        zaha valid > ai valid > -1
  - whole GT is derived from this fused fine-level GT as:
        building vs not-building
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.gridspec import GridSpec


# =========================================================
# Global plotting style
# =========================================================
PLOT_FONT_FAMILY = "DejaVu Sans"
TITLE_FONT_SIZE = 11
TEXT_FONT_SIZE = 11
HEADER_FONT_SIZE = 12
ROW_LABEL_FONT_SIZE = 12

plt.rcParams["font.family"] = PLOT_FONT_FAMILY
plt.rcParams["font.size"] = TEXT_FONT_SIZE
plt.rcParams["axes.titlesize"] = TITLE_FONT_SIZE
plt.rcParams["axes.titleweight"] = "normal"


# =========================================================
# Fixed colors
# =========================================================
WHOLE_BUILDING_COLOR = np.array([168, 123, 204], dtype=np.uint8)
WHOLE_NONBUILDING_COLOR = np.array([182, 208, 167], dtype=np.uint8)
WHITE = np.array([255, 255, 255], dtype=np.uint8)


# =========================================================
# Logger
# =========================================================
def get_logger(name, log_file=None, log_level=logging.INFO):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False

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


# =========================================================
# JSON helpers
# =========================================================
def load_json_int_keys(path: str):
    with open(path, "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    return obj


# =========================================================
# GT / Validation helpers
# =========================================================
def list_npy_stems(folder: str) -> List[str]:
    if not os.path.exists(folder):
        return []
    return sorted([
        Path(f).stem
        for f in os.listdir(folder)
        if f.endswith(".npy")
    ])


def require_complete_gt_pair(
    dir_zaha: str,
    dir_ai: str,
    logger: logging.Logger,
) -> List[str]:
    zaha_names = set(list_npy_stems(dir_zaha))
    ai_names = set(list_npy_stems(dir_ai))

    all_names = sorted(zaha_names | ai_names)
    missing_msgs = []

    for name in all_names:
        missing = []
        if name not in zaha_names:
            missing.append(dir_zaha)
        if name not in ai_names:
            missing.append(dir_ai)

        if missing:
            missing_msgs.append(
                f"[GT missing] image='{name}' not found in: {', '.join(missing)}"
            )

    if missing_msgs:
        for msg in missing_msgs[:200]:
            logger.error(msg)
        if len(missing_msgs) > 200:
            logger.error(f"... and {len(missing_msgs) - 200} more missing GT records.")
        raise FileNotFoundError(
            "GT folders are incomplete. See the log for missing image details."
        )

    common = sorted(zaha_names & ai_names)
    logger.info(f"GT complete pair count (zaha + ai): {len(common)}")
    return common


def load_ground_truth_layer(
    layer_dir: str,
    image_names: List[str],
    class_mapping: Dict[int, Union[str, List[str]]],
    logger: logging.Logger,
    layer_name: str = "GT",
) -> Dict[str, Dict]:
    logger.info(f"   Loading {layer_name} from: {layer_dir}")
    gt_data = {}

    if not os.path.exists(layer_dir):
        logger.error(f"   [ERROR] Directory not found: {layer_dir}")
        return {}

    valid_class_ids = set(class_mapping.keys())

    for img_name in tqdm(image_names, desc=f"Loading {layer_name}", leave=False):
        gt_path = os.path.join(layer_dir, f"{img_name}.npy")
        if not os.path.exists(gt_path):
            continue

        semantic_map = np.load(gt_path).astype(np.int32)

        invalid_mask = (semantic_map >= 0) & (~np.isin(semantic_map, list(valid_class_ids)))
        semantic_map[invalid_mask] = -1

        coverage_mask = semantic_map >= 0
        unique_classes = np.unique(semantic_map[coverage_mask]).tolist()

        gt_data[img_name] = {
            "semantic_map": semantic_map,
            "coverage_mask": coverage_mask,
            "classes": unique_classes,
        }

    logger.info(f"   Loaded {len(gt_data)} images for {layer_name}.")
    return gt_data


def fuse_fine_gt_zaha_ai(zaha_map: np.ndarray, ai_map: np.ndarray) -> np.ndarray:
    if zaha_map.shape != ai_map.shape:
        raise ValueError(f"Shape mismatch in GT fusion: zaha={zaha_map.shape}, ai={ai_map.shape}")

    out = np.full_like(zaha_map, -1, dtype=np.int32)

    ai_valid = ai_map >= 0
    zaha_valid = zaha_map >= 0

    out[ai_valid] = ai_map[ai_valid]
    out[zaha_valid] = zaha_map[zaha_valid]

    return out


def build_binary_building_gt(
    fine_gt: np.ndarray,
    building_fine_ids: List[int],
) -> np.ndarray:
    gt = np.full_like(fine_gt, -1, dtype=np.int32)
    valid = fine_gt >= 0
    gt[valid] = 0
    gt[np.isin(fine_gt, building_fine_ids)] = 1
    return gt


def merge_nonbuilding_for_whole_vis(
    fine_map: np.ndarray,
    building_fine_ids: List[int],
    nonbuilding_fine_ids: List[int],
    whole_building_id: int,
    whole_nonbuilding_id: int,
) -> np.ndarray:
    out = np.full_like(fine_map, -1, dtype=np.int32)
    out[np.isin(fine_map, nonbuilding_fine_ids)] = int(whole_nonbuilding_id)
    out[np.isin(fine_map, building_fine_ids)] = int(whole_building_id)
    return out


# =========================================================
# Metrics
# =========================================================
def compute_binary_metrics(pred_is_pos: np.ndarray, gt_is_pos: np.ndarray, eval_mask: np.ndarray):
    pb = pred_is_pos[eval_mask].astype(bool)
    gb = gt_is_pos[eval_mask].astype(bool)

    if gb.size == 0:
        return None

    tp = int(np.logical_and(pb, gb).sum())
    tn = int(np.logical_and(~pb, ~gb).sum())
    fp = int(np.logical_and(pb, ~gb).sum())
    fn = int(np.logical_and(~pb, gb).sum())

    iou = tp / (tp + fp + fn + 1e-9)
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    f1 = (2 * tp) / (2 * tp + fp + fn + 1e-9)

    return {
        "iou": float(iou),
        "precision": float(prec),
        "recall": float(rec),
        "accuracy": float(acc),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "pixel_count": int(gb.size),
    }


def mask_to_boundary(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:
    mask = mask.astype(np.uint8)
    h, w = mask.shape
    diag_len = np.sqrt(h * h + w * w)
    dilation = max(1, int(round(dilation_ratio * diag_len)))

    kernel = np.ones((3, 3), dtype=np.uint8)
    padded_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    eroded = cv2.erode(padded_mask, kernel, iterations=dilation)
    eroded = eroded[1:-1, 1:-1]
    boundary = mask - eroded
    return boundary.astype(bool)


def compute_boundary_iou_binary(pred_mask: np.ndarray, gt_mask: np.ndarray, dilation_ratio: float = 0.02) -> float:
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    pred_boundary = mask_to_boundary(pred_mask.astype(np.uint8), dilation_ratio=dilation_ratio)
    gt_boundary = mask_to_boundary(gt_mask.astype(np.uint8), dilation_ratio=dilation_ratio)

    union = np.logical_or(pred_boundary, gt_boundary).sum()
    if union == 0:
        return 1.0

    inter = np.logical_and(pred_boundary, gt_boundary).sum()
    return float(inter / union)


def compute_multiclass_metrics(pred_semantic, gt_semantic, eval_mask, class_mapping, boundary_dilation_ratio=0.02):
    pred_valid = pred_semantic[eval_mask]
    gt_valid = gt_semantic[eval_mask]

    if gt_valid.size == 0:
        return None

    gt_classes = np.unique(gt_valid).tolist()
    results = {
        "class_iou": {},
        "class_precision": {},
        "class_recall": {},
        "class_counts": {},
        "class_boundary_iou": {},
        "pixel_count": int(gt_valid.size),
        "correct_count": int((pred_valid == gt_valid).sum()),
    }

    for cls_id in gt_classes:
        cls_name = class_mapping.get(int(cls_id), f"class_{cls_id}")
        if isinstance(cls_name, list):
            cls_name = cls_name[0]

        gt_cls_mask_flat = (gt_valid == cls_id)
        pred_cls_mask_flat = (pred_valid == cls_id)

        tp = int(np.logical_and(pred_cls_mask_flat, gt_cls_mask_flat).sum())
        fp = int(np.logical_and(pred_cls_mask_flat, ~gt_cls_mask_flat).sum())
        fn = int(np.logical_and(~pred_cls_mask_flat, gt_cls_mask_flat).sum())
        tn = int(np.logical_and(~pred_cls_mask_flat, ~gt_cls_mask_flat).sum())

        iou = tp / (tp + fp + fn + 1e-9)
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)

        results["class_iou"][cls_name] = iou
        results["class_precision"][cls_name] = precision
        results["class_recall"][cls_name] = recall
        results["class_counts"][cls_name] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}

        pred_cls_mask_full = (pred_semantic == cls_id) & eval_mask
        gt_cls_mask_full = (gt_semantic == cls_id) & eval_mask
        biou = compute_boundary_iou_binary(pred_cls_mask_full, gt_cls_mask_full, dilation_ratio=boundary_dilation_ratio)
        results["class_boundary_iou"][cls_name] = biou

    results["mean_iou"] = float(np.mean(list(results["class_iou"].values()))) if results["class_iou"] else 0.0
    results["mean_boundary_iou"] = float(np.mean(list(results["class_boundary_iou"].values()))) if results["class_boundary_iou"] else 0.0
    results["pixel_acc"] = float(results["correct_count"] / results["pixel_count"])
    return results


# =========================================================
# Visualization helpers
# =========================================================
def _fallback_distinct_color_by_id(class_id: int) -> np.ndarray:
    fallback_colors = [
        [31, 119, 180],
        [255, 127, 14],
        [148, 103, 189],
        [140, 86, 75],
        [227, 119, 194],
        [127, 127, 127],
        [188, 189, 34],
        [23, 190, 207],
        [174, 199, 232],
        [255, 187, 120],
        [197, 176, 213],
        [196, 156, 148],
        [247, 182, 210],
        [199, 199, 199],
        [219, 219, 141],
        [158, 218, 229],
    ]
    return np.array(fallback_colors[int(class_id) % len(fallback_colors)], dtype=np.uint8)


def build_fixed_palette(unique_ids, class_colors):
    palette = {}
    for uid in sorted([int(uid) for uid in unique_ids if int(uid) >= 0]):
        if uid == 200:
            palette[uid] = WHOLE_BUILDING_COLOR.copy()
        elif uid == 201:
            palette[uid] = WHOLE_NONBUILDING_COLOR.copy()
        elif class_colors and uid in class_colors:
            palette[uid] = np.array(class_colors[uid], dtype=np.uint8)
        else:
            palette[uid] = _fallback_distinct_color_by_id(uid)
    return palette


def colorize(mask, palette, neg_color=(255, 255, 255)):
    h, w = mask.shape
    col = np.zeros((h, w, 3), dtype=np.uint8)
    col[mask < 0] = np.array(neg_color, dtype=np.uint8)
    for cid in np.unique(mask):
        if cid < 0:
            continue
        col[mask == cid] = palette.get(int(cid), np.array([160, 160, 160], dtype=np.uint8))
    return col


def error_map_multiclass_on_mask(pred, gt, eval_mask):
    err = np.full((*gt.shape, 3), 255, dtype=np.uint8)
    correct = (pred == gt) & eval_mask
    wrong = (pred != gt) & eval_mask
    err[correct] = [0, 255, 0]
    err[wrong] = [255, 0, 0]
    return err


def error_map_binary(pred_is_pos, gt_is_pos, eval_mask):
    err = np.full((*gt_is_pos.shape, 3), 255, dtype=np.uint8)
    correct = (pred_is_pos == gt_is_pos) & eval_mask
    wrong = (pred_is_pos != gt_is_pos) & eval_mask
    err[correct] = [0, 255, 0]
    err[wrong] = [255, 0, 0]
    return err


def format_whole_metric_text(metrics: Optional[Dict]) -> str:
    if not metrics:
        return "N/A"

    return "\n".join([
        f"IoU   {metrics.get('iou', 0.0):.3f}",
        f"P     {metrics.get('precision', 0.0):.3f}",
        f"R     {metrics.get('recall', 0.0):.3f}",
        f"Acc   {metrics.get('accuracy', 0.0):.3f}",
        f"F1    {metrics.get('f1', 0.0):.3f}",
    ])


def format_part_metric_text(metrics: Optional[Dict]) -> str:
    if not metrics:
        return "N/A"

    return "\n".join([
        f"mIoU  {metrics.get('mean_iou', 0.0):.3f}",
        f"BIoU  {metrics.get('mean_boundary_iou', 0.0):.3f}",
        f"Acc   {metrics.get('pixel_acc', 0.0):.3f}",
    ])


def draw_right_side_annotation(ax, header_text: str, metric_text: str):
    ax.text(
        1.04, 0.72, header_text,
        transform=ax.transAxes,
        va="center", ha="left",
        fontsize=HEADER_FONT_SIZE,
        fontweight="bold",
        fontfamily=PLOT_FONT_FAMILY,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, pad=2.0),
    )
    ax.text(
        1.04, 0.40, metric_text,
        transform=ax.transAxes,
        va="center", ha="left",
        fontsize=TEXT_FONT_SIZE,
        fontweight="normal",
        fontfamily=PLOT_FONT_FAMILY,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, pad=2.0),
    )


def visualize_method_two_level_error(
    pred_whole_vis: np.ndarray,
    gt_whole_vis: np.ndarray,
    pred_whole_building: np.ndarray,
    gt_whole_building: np.ndarray,
    pred_part: np.ndarray,
    gt_part: np.ndarray,
    class_colors: Dict[int, List[int]],
    save_path: str,
    whole_metrics: Optional[Dict] = None,
    part_metrics: Optional[Dict] = None,
):
    unique_ids = (
        set(np.unique(gt_whole_vis))
        | set(np.unique(pred_whole_vis))
        | set(np.unique(gt_part))
        | set(np.unique(pred_part))
    )
    palette = build_fixed_palette(unique_ids, class_colors)

    gt_valid_whole = gt_whole_building >= 0

    gt_whole_vis_masked = gt_whole_vis.copy()
    gt_whole_vis_masked[~gt_valid_whole] = -1
    vis_gt_whole = colorize(gt_whole_vis_masked, palette, neg_color=(255, 255, 255))

    vis_pr_whole_raw = colorize(pred_whole_vis, palette, neg_color=(255, 255, 255))

    pred_whole_vis_masked = pred_whole_vis.copy()
    pred_whole_vis_masked[~gt_valid_whole] = -1
    vis_pr_whole_masked = colorize(pred_whole_vis_masked, palette, neg_color=(255, 255, 255))

    err_whole = error_map_binary(
        pred_whole_building == 1,
        gt_whole_building == 1,
        gt_valid_whole,
    )

    gt_valid_part = gt_part >= 0

    gt_part_vis = gt_part.copy()
    gt_part_vis[~gt_valid_part] = -1
    vis_gt_part = colorize(gt_part_vis, palette, neg_color=(255, 255, 255))

    vis_pr_part_raw = colorize(pred_part, palette, neg_color=(255, 255, 255))

    pred_part_vis = pred_part.copy()
    pred_part_vis[~gt_valid_part] = -1
    vis_pr_part_masked = colorize(pred_part_vis, palette, neg_color=(255, 255, 255))

    err_part = error_map_multiclass_on_mask(pred_part_vis, gt_part_vis, gt_valid_part)

    row_labels = ["Whole", "Part"]
    metric_texts = [
        format_whole_metric_text(whole_metrics),
        format_part_metric_text(part_metrics),
    ]
    rows = [
        (vis_gt_whole, vis_pr_whole_raw, vis_pr_whole_masked, err_whole),
        (vis_gt_part, vis_pr_part_raw, vis_pr_part_masked, err_part),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(19, 6.3), facecolor="white")

    axes[0, 0].set_title("GT", fontsize=TEXT_FONT_SIZE, fontweight="normal", fontfamily=PLOT_FONT_FAMILY, pad=4)
    axes[0, 1].set_title("Prediction\n(Unmasked)", fontsize=TEXT_FONT_SIZE, fontweight="normal", fontfamily=PLOT_FONT_FAMILY, pad=4)
    axes[0, 2].set_title("Prediction\n(Masked by GT)", fontsize=TEXT_FONT_SIZE, fontweight="normal", fontfamily=PLOT_FONT_FAMILY, pad=4)
    axes[0, 3].set_title("Error\nGreen = Correct, Red = Wrong", fontsize=TEXT_FONT_SIZE, fontweight="normal", fontfamily=PLOT_FONT_FAMILY, pad=4)

    for r in range(2):
        for c in range(4):
            axes[r, c].imshow(rows[r][c], aspect="auto")
            axes[r, c].set_aspect("auto")
            axes[r, c].axis("off")

        axes[r, 0].set_ylabel(
            row_labels[r],
            fontsize=ROW_LABEL_FONT_SIZE,
            rotation=90,
            labelpad=14,
            fontweight="bold",
            fontfamily=PLOT_FONT_FAMILY,
        )

        draw_right_side_annotation(
            axes[r, 3],
            header_text=row_labels[r],
            metric_text=metric_texts[r],
        )

    plt.subplots_adjust(
        left=0.05,
        right=0.88,
        top=0.93,
        bottom=0.03,
        wspace=0.015,
        hspace=0.015,
    )
    plt.savefig(save_path, dpi=120, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)


# =========================================================
# Report
# =========================================================
def aggregate_metrics(metrics_log):
    final_report = {}

    whole_data = metrics_log["whole"]
    if whole_data:
        final_report["whole"] = {
            "N": int(len(whole_data)),
            "IoU": float(np.mean([d["iou"] for d in whole_data])),
            "Precision": float(np.mean([d["precision"] for d in whole_data])),
            "Recall": float(np.mean([d["recall"] for d in whole_data])),
            "Accuracy": float(np.mean([d["accuracy"] for d in whole_data])),
            "F1": float(np.mean([d["f1"] for d in whole_data])),
        }

    part_data = metrics_log["part"]
    if part_data:
        mean_iou = float(np.mean([d["mean_iou"] for d in part_data]))
        mean_biou = float(np.mean([d["mean_boundary_iou"] for d in part_data]))
        mean_acc = float(np.mean([d["pixel_acc"] for d in part_data]))

        class_precisions = defaultdict(list)
        class_recalls = defaultdict(list)
        class_ious = defaultdict(list)
        class_bious = defaultdict(list)
        class_tps, class_fps, class_fns, class_tns = [defaultdict(int) for _ in range(4)]

        for d in part_data:
            for c, p in d["class_precision"].items():
                class_precisions[c].append(p)
            for c, r in d["class_recall"].items():
                class_recalls[c].append(r)
            for c, i in d["class_iou"].items():
                class_ious[c].append(i)
            for c, biou in d["class_boundary_iou"].items():
                class_bious[c].append(biou)

            for c, counts in d["class_counts"].items():
                class_tps[c] += counts["tp"]
                class_fps[c] += counts["fp"]
                class_fns[c] += counts["fn"]
                class_tns[c] += counts["tn"]

        final_report["part"] = {
            "N": int(len(part_data)),
            "mIoU": mean_iou,
            "Boundary_IoU": mean_biou,
            "pixel_acc": mean_acc,
            "per_class_iou": {
                c: float(np.mean(v)) for c, v in class_ious.items()
            },
            "per_class_boundary_iou": {
                c: float(np.mean(v)) for c, v in class_bious.items()
            },
            "per_class_precision": {
                c: float(np.mean(v)) for c, v in class_precisions.items()
            },
            "per_class_recall": {
                c: float(np.mean(v)) for c, v in class_recalls.items()
            },
            "per_class_raw_counts": {
                c: {
                    "tp": class_tps[c],
                    "fp": class_fps[c],
                    "fn": class_fns[c],
                    "tn": class_tns[c],
                }
                for c in class_tps.keys()
            },
        }

    return final_report


# =========================================================
# Unified evaluator
# =========================================================
class UnifiedTwoLevelEvaluator:
    """
    predictor must implement:
        - setup()
        - required_paths(image_name: str) -> List[Dict]
        - predict(image_name: str) -> dict with:
            {
              "pred_whole": np.ndarray,
              "pred_part": np.ndarray,
            }
    """

    def __init__(
        self,
        predictor,
        gt_split_root: str,
        class_mapping: Dict[int, Union[str, List[str]]],
        output_dir: str,
        logger,
        class_colors: Optional[Dict[int, List[int]]] = None,
        whole_building_id: int = 200,
        whole_nonbuilding_id: int = 201,
        whole_building_fine_ids: Optional[List[int]] = None,
        whole_nonbuilding_fine_ids: Optional[List[int]] = None,
        save_visualizations: bool = True,
        save_prediction_cache: bool = True,
        boundary_dilation_ratio: float = 0.02,
        validated_image_names: Optional[List[str]] = None,
    ):
        self.predictor = predictor
        self.gt_split_root = gt_split_root
        self.class_mapping = class_mapping
        self.output_dir = output_dir
        self.logger = logger
        self.class_colors = class_colors or {}
        self.whole_building_id = int(whole_building_id)
        self.whole_nonbuilding_id = int(whole_nonbuilding_id)
        self.whole_building_fine_ids = [int(x) for x in (whole_building_fine_ids or [])]
        self.whole_nonbuilding_fine_ids = [int(x) for x in (whole_nonbuilding_fine_ids or [])]
        self.save_visualizations = save_visualizations
        self.save_prediction_cache = save_prediction_cache
        self.boundary_dilation_ratio = float(boundary_dilation_ratio)
        self.validated_image_names = list(validated_image_names) if validated_image_names is not None else None

    def _validate_predictor_inputs(self, image_names: List[str]) -> None:
        missing_msgs = []

        for name in image_names:
            reqs = self.predictor.required_paths(name)
            for req in reqs:
                label = req.get("label", "input")
                if req.get("any_of", False):
                    candidates = req.get("paths", [])
                    if not any(os.path.exists(p) for p in candidates):
                        missing_msgs.append(
                            f"[{getattr(self.predictor, 'method_name', 'method')} missing] "
                            f"image='{name}' missing {label}; checked: {candidates}"
                        )
                else:
                    path = req.get("path")
                    if path is None or not os.path.exists(path):
                        missing_msgs.append(
                            f"[{getattr(self.predictor, 'method_name', 'method')} missing] "
                            f"image='{name}' missing {label}: {path}"
                        )

        if missing_msgs:
            for msg in missing_msgs[:200]:
                self.logger.error(msg)
            if len(missing_msgs) > 200:
                self.logger.error(f"... and {len(missing_msgs) - 200} more missing predictor records.")
            raise FileNotFoundError(
                f"Predictor inputs are incomplete for method "
                f"{getattr(self.predictor, 'method_name', 'method')}. See the log for details."
            )

    def _log_final_report(self, report):
        self.logger.info("\n" + "=" * 70)
        self.logger.info("FINAL 2-LEVEL EVALUATION RESULTS")
        self.logger.info("=" * 70)

        if "whole" in report:
            r = report["whole"]
            self.logger.info(f"\n[WHOLE / BUILDING] (N={r['N']})")
            self.logger.info(f"  IoU:       {r['IoU']:.4f}")
            self.logger.info(f"  Precision: {r['Precision']:.4f}")
            self.logger.info(f"  Recall:    {r['Recall']:.4f}")
            self.logger.info(f"  Accuracy:  {r['Accuracy']:.4f}")
            self.logger.info(f"  F1:        {r['F1']:.4f}")
        else:
            self.logger.warning("\n[WHOLE / BUILDING] No valid metrics.")

        if "part" in report:
            r = report["part"]
            self.logger.info(f"\n[PART] (N={r['N']})")
            self.logger.info(f"  mIoU:      {r['mIoU']:.4f}")
            self.logger.info(f"  BIoU:      {r['Boundary_IoU']:.4f}")
            self.logger.info(f"  Pixel Acc: {r['pixel_acc']:.4f}")
            self.logger.info("  Per Class IoU:")
            for c, iou in sorted(r["per_class_iou"].items()):
                self.logger.info(f"    {c:<25}: {iou:.4f}")
            self.logger.info("  Per Class Boundary IoU:")
            for c, biou in sorted(r["per_class_boundary_iou"].items()):
                self.logger.info(f"    {c:<25}: {biou:.4f}")
        else:
            self.logger.warning("\n[PART] No valid metrics.")

    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)
        vis_dir = os.path.join(self.output_dir, "visualizations_2levels")
        pred_dir = os.path.join(self.output_dir, "predictions")
        if self.save_visualizations:
            os.makedirs(vis_dir, exist_ok=True)
        if self.save_prediction_cache:
            os.makedirs(pred_dir, exist_ok=True)

        dir_zaha = os.path.join(self.gt_split_root, "layer_zaha_kept")
        dir_ai = os.path.join(self.gt_split_root, "layer_ai_filled")

        self.logger.info("Target evaluation GT directories:")
        self.logger.info(f"  > Zaha : {dir_zaha}")
        self.logger.info(f"  > AI   : {dir_ai}")
        self.logger.info("GT fusion policy: zaha valid > ai valid > -1")
        self.logger.info(f"[CONFIG] whole_building_fine_ids={self.whole_building_fine_ids}")
        self.logger.info(f"[CONFIG] whole_nonbuilding_fine_ids={self.whole_nonbuilding_fine_ids}")

        if self.validated_image_names is not None:
            gt_names = list(self.validated_image_names)
        else:
            gt_names = require_complete_gt_pair(
                dir_zaha=dir_zaha,
                dir_ai=dir_ai,
                logger=self.logger,
            )
            self._validate_predictor_inputs(gt_names)

        self.predictor.setup()

        gt_zaha = load_ground_truth_layer(dir_zaha, gt_names, self.class_mapping, self.logger, "Zaha")
        gt_ai = load_ground_truth_layer(dir_ai, gt_names, self.class_mapping, self.logger, "AI")

        common = sorted(set(gt_names) & set(gt_zaha.keys()) & set(gt_ai.keys()))
        self.logger.info(f"Evaluating on {len(common)} images.")

        metrics_log = {"whole": [], "part": []}
        per_image_metrics = {}

        for name in tqdm(common, desc=f"Evaluating [{getattr(self.predictor, 'method_name', 'method')}]"):
            pred = self.predictor.predict(name)
            pred_whole = pred["pred_whole"]
            pred_part = pred["pred_part"]

            gt_part = fuse_fine_gt_zaha_ai(
                gt_zaha[name]["semantic_map"],
                gt_ai[name]["semantic_map"],
            )
            gt_whole = build_binary_building_gt(
                gt_part,
                building_fine_ids=self.whole_building_fine_ids,
            )

            H, W = gt_part.shape
            if pred_whole.shape != (H, W):
                pred_whole = cv2.resize(pred_whole.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST)
            if pred_part.shape != (H, W):
                pred_part = cv2.resize(pred_part.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST)

            eval_mask_whole = gt_whole >= 0
            gt_whole_pos = gt_whole == 1
            pred_whole_pos = pred_whole == 1

            wm = compute_binary_metrics(
                pred_is_pos=pred_whole_pos,
                gt_is_pos=gt_whole_pos,
                eval_mask=eval_mask_whole,
            )

            eval_mask_part = gt_part >= 0
            pm = compute_multiclass_metrics(
                pred_part,
                gt_part,
                eval_mask_part,
                self.class_mapping,
                boundary_dilation_ratio=self.boundary_dilation_ratio,
            )

            if wm:
                metrics_log["whole"].append(wm)
            if pm:
                metrics_log["part"].append(pm)

            per_image_metrics[name] = {
                "whole": wm,
                "part": pm,
            }

            pred_whole_vis = merge_nonbuilding_for_whole_vis(
                fine_map=pred_part,
                building_fine_ids=self.whole_building_fine_ids,
                nonbuilding_fine_ids=self.whole_nonbuilding_fine_ids,
                whole_building_id=self.whole_building_id,
                whole_nonbuilding_id=self.whole_nonbuilding_id,
            )
            gt_whole_vis = merge_nonbuilding_for_whole_vis(
                fine_map=gt_part,
                building_fine_ids=self.whole_building_fine_ids,
                nonbuilding_fine_ids=self.whole_nonbuilding_fine_ids,
                whole_building_id=self.whole_building_id,
                whole_nonbuilding_id=self.whole_nonbuilding_id,
            )

            pred_whole_vis[pred_whole_pos] = self.whole_building_id

            if self.save_prediction_cache:
                np.savez_compressed(
                    os.path.join(pred_dir, f"{name}.npz"),
                    pred_whole=pred_whole.astype(np.int32),
                    pred_part=pred_part.astype(np.int32),
                    pred_whole_vis=pred_whole_vis.astype(np.int32),
                )

            if self.save_visualizations:
                visualize_method_two_level_error(
                    pred_whole_vis=pred_whole_vis,
                    gt_whole_vis=gt_whole_vis,
                    pred_whole_building=pred_whole.astype(np.int32),
                    gt_whole_building=gt_whole.astype(np.int32),
                    pred_part=pred_part,
                    gt_part=gt_part,
                    class_colors=self.class_colors,
                    save_path=os.path.join(vis_dir, f"{name}_2levels.png"),
                    whole_metrics=wm,
                    part_metrics=pm,
                )

        report = aggregate_metrics(metrics_log)
        self._log_final_report(report)

        out_json = os.path.join(self.output_dir, "two_level_results.json")
        with open(out_json, "w") as f:
            json.dump(convert_to_serializable(report), f, indent=2)

        per_image_json = os.path.join(self.output_dir, "per_image_metrics.json")
        with open(per_image_json, "w") as f:
            json.dump(convert_to_serializable(per_image_metrics), f, indent=2)

        self.logger.info(f"\n2-level results saved to: {out_json}")
        self.logger.info(f"Per-image metrics saved to: {per_image_json}")
        if self.save_prediction_cache:
            self.logger.info(f"Prediction cache saved to: {pred_dir}")
        if self.save_visualizations:
            self.logger.info(f"Visualizations saved to: {vis_dir}")

        return report


# =========================================================
# Shared helpers for cross-method panels
# =========================================================
def _rgb_tuple(color) -> tuple:
    arr = np.asarray(color, dtype=np.uint8).reshape(-1)
    return (int(arr[0]), int(arr[1]), int(arr[2]))


def build_combined_part_palette(
    class_mapping: Dict[int, Union[str, List[str]]],
    class_colors: Dict[int, List[int]],
) -> Dict[int, np.ndarray]:
    palette = {}
    for cid in sorted(class_mapping.keys()):
        if class_colors and cid in class_colors:
            palette[int(cid)] = np.array(class_colors[cid], dtype=np.uint8)
        else:
            palette[int(cid)] = _fallback_distinct_color_by_id(int(cid))
    return palette


def colorize_combined_whole(mask: np.ndarray, whole_building_id: int, whole_nonbuilding_id: int) -> np.ndarray:
    h, w = mask.shape
    vis = np.full((h, w, 3), 255, dtype=np.uint8)
    vis[mask == whole_nonbuilding_id] = WHOLE_NONBUILDING_COLOR
    vis[mask == whole_building_id] = WHOLE_BUILDING_COLOR
    return vis


def colorize_combined_part(label_map: np.ndarray, palette: Dict[int, np.ndarray]) -> np.ndarray:
    h, w = label_map.shape
    vis = np.full((h, w, 3), 255, dtype=np.uint8)
    for cid in np.unique(label_map):
        if cid < 0:
            continue
        vis[label_map == cid] = np.array(palette.get(int(cid), [160, 160, 160]), dtype=np.uint8)
    return vis


def _draw_side_title(ax, text: str):
    ax.axis("off")
    ax.text(0.88, 0.5, text, ha="right", va="center", fontsize=13, fontweight="bold", color="black", linespacing=1.0)


def _draw_strip_item(ax, cx, y, color_rgb, label, strip_w=0.125, strip_h=0.040, text_gap=0.008, fontsize=12):
    x0 = cx - strip_w / 2
    y0 = y - strip_h / 2

    rect = Rectangle(
        (x0, y0), strip_w, strip_h,
        facecolor=np.array(color_rgb) / 255.0,
        edgecolor="black", linewidth=0.8,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(rect)

    ax.text(cx, y0 - text_gap, label, transform=ax.transAxes, va="top", ha="center", fontsize=fontsize, color="black")


def _draw_whole_legend(ax):
    ax.axis("off")
    ax.text(0.01, 0.96, "Whole legend", transform=ax.transAxes, ha="left", va="top", fontsize=12.5, fontweight="bold", color="black")
    _draw_strip_item(ax, 0.23, 0.64, WHOLE_BUILDING_COLOR, "building")
    _draw_strip_item(ax, 0.23, 0.40, WHOLE_NONBUILDING_COLOR, "non-building")


def _draw_part_legend(ax, class_mapping: Dict[int, Union[str, List[str]]], palette: Dict[int, np.ndarray]):
    ax.axis("off")
    ax.text(0.01, 0.98, "Part legend", transform=ax.transAxes, ha="left", va="top", fontsize=12.5, fontweight="bold", color="black")

    valid_ids = [cid for cid in sorted(class_mapping.keys()) if cid in palette]
    ys = np.linspace(0.80, 0.12, num=max(1, len(valid_ids)))
    for cid, y in zip(valid_ids, ys):
        lbl = class_mapping[cid]
        if isinstance(lbl, list):
            lbl = lbl[0]
        _draw_strip_item(ax, 0.35, float(y), palette[cid], str(lbl), fontsize=11)


def _draw_gap_arrow_on_overlay(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    arrow = FancyArrowPatch(
        (0.50, 0.53), (0.50, 0.47),
        arrowstyle="-|>", mutation_scale=16,
        linewidth=2.0, color="black",
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(arrow)


def _load_rgb_image(image_path: str) -> np.ndarray:
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(image_path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# =========================================================
# Cross-method prediction panels
# =========================================================
def create_cross_method_prediction_panels(
    root_output_dir: str,
    rgb_dir: str,
    gt_split_root: str,
    class_mapping: Dict[int, Union[str, List[str]]],
    class_colors: Dict[int, List[int]],
    whole_building_id: int,
    whole_nonbuilding_id: int,
    whole_building_fine_ids: List[int],
    whole_nonbuilding_fine_ids: List[int],
    method_output_dirs: Dict[str, str],
    masked_by_gt: bool,
    output_subdir: str,
):
    logger = get_logger(f"cross_method_panels_{output_subdir}")

    method_order = ["langsplat", "gaga_dino", "citygml_clip"]
    method_labels = {
        "langsplat": "LangSplat",
        "gaga_dino": "GaGa",
        "citygml_clip": "Ours",
    }

    pred_maps = {}
    common_images = None
    for method_name in method_order:
        pred_dir = os.path.join(method_output_dirs[method_name], "predictions")
        if not os.path.exists(pred_dir):
            continue
        names = {Path(p).stem for p in Path(pred_dir).glob("*.npz")}
        pred_maps[method_name] = pred_dir
        common_images = names if common_images is None else (common_images & names)

    if not pred_maps or not common_images:
        logger.info("Skip cross-method panels: no common predictions found.")
        return

    rgb_map = {}
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]:
        for p in Path(rgb_dir).rglob(ext):
            stem = p.stem
            if stem not in rgb_map:
                rgb_map[stem] = str(p)

    dir_zaha = os.path.join(gt_split_root, "layer_zaha_kept")
    dir_ai = os.path.join(gt_split_root, "layer_ai_filled")
    zaha_map = {p.stem: str(p) for p in Path(dir_zaha).rglob("*.npy")}
    ai_map = {p.stem: str(p) for p in Path(dir_ai).rglob("*.npy")}

    valid_names = sorted([n for n in common_images if n in rgb_map and n in zaha_map and n in ai_map])
    if len(valid_names) == 0:
        logger.info("Skip cross-method panels: no common RGB/GT/pred files found.")
        return

    output_dir = os.path.join(root_output_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    fig_w = 18.8
    row_h = 3.95
    part_palette = build_combined_part_palette(class_mapping, class_colors)

    for image_name in tqdm(valid_names, desc=f"Saving {output_subdir}"):
        rgb_img = _load_rgb_image(rgb_map[image_name])

        # Load GT and keep only IDs defined in class_mapping
        gt_zaha_raw = np.load(zaha_map[image_name]).astype(np.int32)
        gt_ai_raw = np.load(ai_map[image_name]).astype(np.int32)

        valid_ids = list(class_mapping.keys())
        gt_zaha_raw[~np.isin(gt_zaha_raw, valid_ids)] = -1
        gt_ai_raw[~np.isin(gt_ai_raw, valid_ids)] = -1

        # Fuse the cleaned GT maps
        gt_part = fuse_fine_gt_zaha_ai(gt_zaha_raw, gt_ai_raw)

        # Build a palette for all IDs appearing in the fused map
        current_palette = build_fixed_palette(np.unique(gt_part), class_colors)

        # Restrict the valid mask to IDs defined in class_mapping
        gt_valid_mask = np.isin(gt_part, valid_ids)

        # Create the fused GT visualization for the part level
        gt_part_vis_map = np.full_like(gt_part, -1)
        gt_part_vis_map[gt_valid_mask] = gt_part[gt_valid_mask]
        gt_part_vis_rgb = colorize_combined_part(gt_part_vis_map, current_palette)

        method_data = []
        skip_flag = False

        for method in method_order:
            pred_path = os.path.join(pred_maps[method], f"{image_name}.npz")
            if not os.path.exists(pred_path):
                skip_flag = True
                break

            data = np.load(pred_path)
            if "pred_part" not in data or "pred_whole_vis" not in data:
                skip_flag = True
                break

            pred_part = data["pred_part"].astype(np.int32)
            pred_whole_vis = data["pred_whole_vis"].astype(np.int32)

            if masked_by_gt:
                pred_whole_vis_show = pred_whole_vis.copy()
                pred_whole_vis_show[~gt_valid_mask] = -1

                pred_part_show = pred_part.copy()
                pred_part_show[~gt_valid_mask] = -1
            else:
                pred_whole_vis_show = pred_whole_vis
                pred_part_show = pred_part

            method_data.append({
                "display_name": method_labels.get(method, method),
                "whole_vis": colorize_combined_whole(
                    pred_whole_vis_show,
                    whole_building_id,
                    whole_nonbuilding_id,
                ),
                "part_vis": colorize_combined_part(
                    pred_part_show,
                    current_palette,
                ),
            })

        if skip_flag:
            continue

        fig = plt.figure(figsize=(fig_w, row_h * 2), facecolor="white")
        gs = GridSpec(
            nrows=2,
            ncols=6,
            width_ratios=[1.0, 0.18, 1.0, 1.0, 1.0, 0.55],
            height_ratios=[1.0, 1.0],
            figure=fig,
        )

        fig.suptitle(image_name, fontsize=16, fontweight="bold", y=0.975)

        ax_rgb = fig.add_subplot(gs[0, 0])
        ax_rgb.imshow(rgb_img, interpolation="nearest")
        ax_rgb.axis("off")
        ax_rgb.set_title("RGB", fontsize=14, pad=4, fontweight="bold")

        ax_gt_part = fig.add_subplot(gs[1, 0])
        ax_gt_part.imshow(gt_part_vis_rgb, interpolation="nearest")
        ax_gt_part.axis("off")
        ax_gt_part.set_title("Fused GT", fontsize=14, pad=4, fontweight="bold")

        ax_title_whole = fig.add_subplot(gs[0, 1])
        _draw_side_title(ax_title_whole, "Whole\nlevel")

        ax_title_part = fig.add_subplot(gs[1, 1])
        _draw_side_title(ax_title_part, "Part\nlevel")

        overlay_axes = []
        for col_idx, row in enumerate(method_data, start=2):
            ax_whole = fig.add_subplot(gs[0, col_idx])
            ax_part = fig.add_subplot(gs[1, col_idx])

            ax_whole.imshow(row["whole_vis"], interpolation="nearest")
            ax_whole.axis("off")
            ax_whole.set_title(row["display_name"], fontsize=14, pad=4, fontweight="bold")

            ax_part.imshow(row["part_vis"], interpolation="nearest")
            ax_part.axis("off")

            ax_overlay = fig.add_subplot(gs[:, col_idx], frameon=False)
            ax_overlay.set_zorder(10)
            ax_overlay.patch.set_alpha(0.0)
            overlay_axes.append(ax_overlay)

        ax_leg_whole = fig.add_subplot(gs[0, 5])
        _draw_whole_legend(ax_leg_whole)

        ax_leg_part = fig.add_subplot(gs[1, 5])
        _draw_part_legend(ax_leg_part, class_mapping, part_palette)

        plt.subplots_adjust(left=0.02, right=0.99, top=0.89, bottom=0.06, wspace=0.045, hspace=0.032)

        for ax_overlay in overlay_axes:
            _draw_gap_arrow_on_overlay(ax_overlay)

        save_path = os.path.join(output_dir, f"{image_name}.png")
        plt.savefig(save_path, dpi=180, bbox_inches="tight", pad_inches=0.018, facecolor="white")
        plt.close(fig)
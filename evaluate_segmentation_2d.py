#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation Script with Optional N-to-1 Matching.

Features:
1. Default: Strict (1-to-1) evaluation.
2. Optional: Relaxed (N-to-1) mode via --enable_n_to_1 flag.
3. Visualization automatically adapts to the selected mode.
"""

import argparse
import os
import glob
import json
import logging
import numpy as np
import cv2
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import label as scipy_label
from skimage.morphology import binary_dilation, disk
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================
DEFAULT_GT_DIR = "/workspace/zaha_eval/outputs/gt_semantic_dist60_499_test"
# DEFAULT_PRED_DIR = "/workspace/CityGMLGaussian/output/gml_10000/train/ours_10000/objects_test"
# DEFAULT_OUTPUT_DIR = "/workspace/zaha_eval/eval_results/gml_seg"

# DEFAULT_PRED_DIR = "/workspace/CityGMLGaussian/output/gaga_10000_train/train/ours_10000/objects_test"
# DEFAULT_OUTPUT_DIR = "/workspace/zaha_eval/eval_results/gaga_seg_test"

DEFAULT_PRED_DIR = "/workspace/Gaga2/output/gaga_original_10000_train/train/ours_10000/objects_test"
DEFAULT_OUTPUT_DIR = "/workspace/zaha_eval/eval_results/gaga_original_seg"

DEFAULT_GT_MERGE_MAP = "gt_merge_map.json"
BOUNDARY_TOLERANCE_PIXELS = 5
MIN_REGION_SIZE = 100 

# [关键参数] N-to-1 包含阈值
PRED_INCLUSION_THRESH = 0.30  # 建议改回 0.60，0.90 太严格了

def get_logger(output_dir):
    logger = logging.getLogger('inst_eval')
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    fh = logging.FileHandler(os.path.join(output_dir, 'eval_metrics.log'), 'w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

# ============================================================================
# ALGORITHMS
# ============================================================================

def calculate_intersection_stats(gt_inst_map, pred_inst_map, gt_ids, pred_ids):
    max_pred_id = pred_ids.max()
    offset = max_pred_id + 1
    
    gt_flat = gt_inst_map.ravel()
    pred_flat = pred_inst_map.ravel()
    
    valid_mask = (gt_flat != 0) & (gt_flat != -1)
    if valid_mask.sum() == 0:
        return None, None, None
        
    gt_flat = gt_flat[valid_mask]
    pred_flat = pred_flat[valid_mask]
    
    hashed = gt_flat.astype(np.int64) * offset + pred_flat.astype(np.int64)
    unique_hashes, counts = np.unique(hashed, return_counts=True)
    
    inter_gt = unique_hashes // offset
    inter_pred = unique_hashes % offset
    
    gt_id_to_idx = {gid: i for i, gid in enumerate(gt_ids)}
    pred_id_to_idx = {pid: i for i, pid in enumerate(pred_ids)}
    
    intersection_mat = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)
    for g, p, c in zip(inter_gt, inter_pred, counts):
        if g in gt_id_to_idx and p in pred_id_to_idx:
            intersection_mat[gt_id_to_idx[g], pred_id_to_idx[p]] = c
            
    gt_areas = np.zeros(len(gt_ids))
    for i, gid in enumerate(gt_ids):
        gt_areas[i] = (gt_inst_map == gid).sum()
        
    pred_areas = np.zeros(len(pred_ids))
    for i, pid in enumerate(pred_ids):
        pred_areas[i] = (pred_inst_map == pid).sum()
        
    return intersection_mat, gt_areas, pred_areas

def mask_to_boundary(mask):
    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boundary = np.zeros_like(mask_u8)
    cv2.drawContours(boundary, contours, -1, 1, 1)
    return boundary

def compute_boundary_score(gt_mask, pred_mask, tolerance=2):
    gt_boundary = mask_to_boundary(gt_mask)
    pred_boundary = mask_to_boundary(pred_mask)
    
    if gt_boundary.sum() == 0 and pred_boundary.sum() == 0: return 1.0
    if gt_boundary.sum() == 0 or pred_boundary.sum() == 0: return 0.0

    gt_dilated = binary_dilation(gt_boundary, disk(tolerance))
    pred_in_gt = np.logical_and(pred_boundary, gt_dilated).sum()
    precision = pred_in_gt / (pred_boundary.sum() + 1e-8)
    
    pred_dilated = binary_dilation(pred_boundary, disk(tolerance))
    gt_in_pred = np.logical_and(gt_boundary, pred_dilated).sum()
    recall = gt_in_pred / (gt_boundary.sum() + 1e-8)
    
    if precision + recall == 0: return 0.0
    return 2 * precision * recall / (precision + recall)

def apply_label_merges(label_map, merge_map):
    if not merge_map: return label_map
    out = label_map.copy()
    for target, sources in merge_map.items():
        target = int(target)
        for src in sources:
            src = int(src)
            if src == target: continue
            out[out == src] = target
    return out

def semantic_to_instances(semantic_map, ignore_val=-1, min_size=MIN_REGION_SIZE):
    instance_map = np.zeros_like(semantic_map, dtype=np.int32)
    current_inst_id = 1
    unique_classes = np.unique(semantic_map)
    for cls in unique_classes:
        if cls == ignore_val: continue
        cls_mask = (semantic_map == cls)
        labeled_array, num_features = scipy_label(cls_mask, structure=np.ones((3,3)))
        if num_features > 0:
            sizes = np.bincount(labeled_array.ravel())
            valid_indices = np.where(sizes > min_size)[0]
            valid_indices = valid_indices[valid_indices != 0]
            for comp_idx in valid_indices:
                instance_map[labeled_array == comp_idx] = current_inst_id
                current_inst_id += 1
    return instance_map

# ============================================================================
# METRIC FUNCTIONS
# ============================================================================

def compute_metrics_1_to_1(intersection_mat, gt_areas, pred_areas, gt_ids, pred_ids, gt_inst_map, pred_inst_map):
    """Standard Hungarian Matching."""
    area_sum = gt_areas[:, None] + pred_areas[None, :]
    union_mat = area_sum - intersection_mat
    iou_matrix = np.divide(intersection_mat, union_mat, out=np.zeros_like(intersection_mat), where=union_mat!=0)
    
    cost_matrix = 1.0 - iou_matrix
    cost_matrix = np.nan_to_num(cost_matrix, nan=1.0)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    ious = []
    bfs = []
    
    # Store mapping for visualization (Strict 1-to-1 map)
    pred_to_gt_map = {}
    
    for r, c in zip(row_ind, col_ind):
        iou = iou_matrix[r, c]
        gid, pid = gt_ids[r], pred_ids[c]
        
        if iou > 1e-6:
            pred_to_gt_map[pid] = gid 
            mask_gt = (gt_inst_map == gid).astype(bool)
            mask_pred = (pred_inst_map == pid).astype(bool)
            bf = compute_boundary_score(mask_gt, mask_pred, tolerance=BOUNDARY_TOLERANCE_PIXELS)
        else:
            bf = 0.0
        ious.append(iou)
        bfs.append(bf)
        
    unmatched = len(gt_ids) - len(row_ind)
    ious.extend([0.0] * unmatched)
    bfs.extend([0.0] * unmatched)
    
    return ious, bfs, pred_to_gt_map

def compute_metrics_n_to_1(intersection_mat, gt_areas, pred_areas, gt_ids, pred_ids, gt_inst_map, pred_inst_map):
    """N-to-1 Matching (Merge-and-Evaluate)."""
    ious = []
    bfs = []
    
    pred_to_gt_map = {}
    pred_used = np.zeros(len(pred_ids), dtype=bool)
    
    for i, gid in enumerate(gt_ids):
        gt_mask = (gt_inst_map == gid).astype(bool)
        
        # Criteria: Intersection / Pred_Area > Threshold
        candidates_idx = []
        for j, pid in enumerate(pred_ids):
            if pred_used[j]: continue
            
            p_area = pred_areas[j]
            if p_area == 0: continue
            
            inclusion = intersection_mat[i, j] / p_area
            if inclusion > PRED_INCLUSION_THRESH:
                candidates_idx.append(j)
        
        if not candidates_idx:
            ious.append(0.0)
            bfs.append(0.0)
            continue
            
        fused_pred_mask = np.zeros_like(gt_inst_map, dtype=bool)
        
        for idx in candidates_idx:
            pid = pred_ids[idx]
            pred_to_gt_map[pid] = gid 
            
            p_mask = (pred_inst_map == pid).astype(bool)
            fused_pred_mask = np.logical_or(fused_pred_mask, p_mask)
            pred_used[idx] = True
            
        intersection = np.logical_and(gt_mask, fused_pred_mask).sum()
        union = np.logical_or(gt_mask, fused_pred_mask).sum()
        
        iou = intersection / (union + 1e-6)
        bf = compute_boundary_score(gt_mask, fused_pred_mask, tolerance=BOUNDARY_TOLERANCE_PIXELS)
        
        ious.append(iou)
        bfs.append(bf)
        
    return ious, bfs, pred_to_gt_map

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_results(gt_inst_map, pred_inst_map, gt_ids, pred_ids, pred_to_gt_map, save_path, mode_name="Strict"):
    """
    Visualize using the provided mapping.
    Matched (or Merged) Preds get the SAME color as their GT counterpart.
    """
    # 1. Generate Colors
    num_colors = len(gt_ids) + len(pred_ids) + 100
    hues = (np.arange(num_colors) * 137.508) % 360
    hues = hues.astype(np.uint8)
    sats = np.full(num_colors, 200, dtype=np.uint8)
    vals = np.full(num_colors, 200, dtype=np.uint8)
    hsv_colors = np.dstack([hues, sats, vals]).reshape(-1, 1, 3)
    rgb_colors = cv2.cvtColor(hsv_colors, cv2.COLOR_HSV2RGB).reshape(-1, 3)
    rgb_colors[0] = [0, 0, 0]

    max_gt = gt_inst_map.max()
    max_pred = pred_inst_map.max()
    lut_gt = np.zeros((max_gt + 1, 3), dtype=np.uint8)
    lut_pred = np.zeros((max_pred + 1, 3), dtype=np.uint8)
    
    gt_color_map = {}
    current_idx = 1
    
    for gid in gt_ids:
        gt_color_map[gid] = current_idx
        lut_gt[gid] = rgb_colors[current_idx]
        current_idx += 1
        
    # Assign colors based on mapping (Strict or N-to-1)
    for pid in pred_ids:
        if pid in pred_to_gt_map:
            matched_gid = pred_to_gt_map[pid]
            if matched_gid in gt_color_map:
                color_idx = gt_color_map[matched_gid]
                lut_pred[pid] = rgb_colors[color_idx]
        else:
            lut_pred[pid] = rgb_colors[current_idx]
            current_idx += 1
            if current_idx >= len(rgb_colors): current_idx = 1 

    vis_gt = lut_gt[gt_inst_map]
    vis_pred = lut_pred[pred_inst_map]

    # Error Map
    remap_lut = np.zeros(max_pred + 1, dtype=gt_inst_map.dtype)
    for pid, gid in pred_to_gt_map.items():
        remap_lut[pid] = gid
    pred_remapped = remap_lut[pred_inst_map]

    error_map = np.zeros((*gt_inst_map.shape, 3), dtype=np.uint8)
    error_map[:] = [255, 0, 0] # Red = Error
    correct_mask = (pred_remapped == gt_inst_map) & (gt_inst_map != 0)
    error_map[correct_mask] = [0, 255, 0] # Green = Correct
    ignore_mask = (gt_inst_map == 0)
    error_map[ignore_mask] = [0, 0, 0] # Black = Background

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(vis_gt)
    axes[0].set_title('Ground Truth', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(vis_pred)
    axes[1].set_title(f'Prediction ({mode_name} Colors)', fontsize=14)
    axes[1].axis('off')

    axes[2].imshow(error_map)
    axes[2].set_title(f'Error Map ({mode_name})', fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

# ============================================================================
# MAIN
# ============================================================================

def evaluate(args):
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "visualizations")
    if args.visualize:
        os.makedirs(vis_dir, exist_ok=True)
        
    logger = get_logger(args.output_dir)
    
    gt_merge_map = None
    if args.gt_merge_map and os.path.exists(args.gt_merge_map):
        with open(args.gt_merge_map, 'r') as f:
            merge_raw = json.load(f)
        gt_merge_map = {int(k): [int(v) for v in values] for k, values in merge_raw.items()}

    gt_files = sorted(glob.glob(os.path.join(args.gt_dir, "*.npy")))
    pred_files = sorted(glob.glob(os.path.join(args.pred_dir, "*.png")))
    
    gt_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in gt_files}
    pred_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in pred_files}
    common_names = sorted(list(set(gt_dict.keys()) & set(pred_dict.keys())))
    
    logger.info(f"Evaluating {len(common_names)} images...")
    logger.info(f"Mode: {'Strict 1-to-1 + Relaxed N-to-1' if args.enable_n_to_1 else 'Strict 1-to-1 ONLY'}")
    if args.enable_n_to_1:
        logger.info(f"N-to-1 Merge Threshold: {PRED_INCLUSION_THRESH}")

    metrics_strict = {"iou": [], "bf": []}
    metrics_relaxed = {"iou": [], "bf": []}

    for name in tqdm(common_names, desc="Processing"):
        gt_sem = np.load(gt_dict[name])
        pred_inst_map = cv2.imread(pred_dict[name], cv2.IMREAD_UNCHANGED)
        if pred_inst_map.ndim == 3: pred_inst_map = pred_inst_map[:,:,0]
        pred_inst_map = pred_inst_map.astype(np.int32)
        
        ph, pw = pred_inst_map.shape
        gh, gw = gt_sem.shape
        if (ph, pw) != (gh, gw):
            gt_sem = cv2.resize(gt_sem, (pw, ph), interpolation=cv2.INTER_NEAREST)

        if gt_merge_map: gt_sem = apply_label_merges(gt_sem, gt_merge_map)
        gt_inst_map = semantic_to_instances(gt_sem, ignore_val=-1, min_size=MIN_REGION_SIZE)
        
        gt_ids = np.unique(gt_inst_map)
        gt_ids = gt_ids[gt_ids != 0]
        pred_ids = np.unique(pred_inst_map)
        pred_ids = pred_ids[(pred_ids != 0) & (pred_ids != -1)]
        
        if len(gt_ids) == 0: continue
        
        intersection_mat, gt_areas, pred_areas = None, None, None
        if len(pred_ids) > 0:
            intersection_mat, gt_areas, pred_areas = calculate_intersection_stats(gt_inst_map, pred_inst_map, gt_ids, pred_ids)

        # ---------------------------------------------------------------------
        # 1. Always Run Strict (1-to-1)
        # ---------------------------------------------------------------------
        pred_to_gt_map_vis = {} # This will determine visualization colors
        vis_mode_name = "Strict 1-to-1"

        if intersection_mat is not None:
            siou, sbf, map_strict = compute_metrics_1_to_1(intersection_mat, gt_areas, pred_areas, gt_ids, pred_ids, gt_inst_map, pred_inst_map)
            metrics_strict["iou"].extend(siou)
            metrics_strict["bf"].extend(sbf)
            pred_to_gt_map_vis = map_strict 
        else:
            unmatched = len(gt_ids)
            metrics_strict["iou"].extend([0.0] * unmatched)
            metrics_strict["bf"].extend([0.0] * unmatched)

        # ---------------------------------------------------------------------
        # 2. Optionally Run Relaxed (N-to-1)
        # ---------------------------------------------------------------------
        if args.enable_n_to_1:
            vis_mode_name = "Relaxed N-to-1"
            if intersection_mat is not None:
                riou, rbf, map_relaxed = compute_metrics_n_to_1(intersection_mat, gt_areas, pred_areas, gt_ids, pred_ids, gt_inst_map, pred_inst_map)
                metrics_relaxed["iou"].extend(riou)
                metrics_relaxed["bf"].extend(rbf)
                # OVERRIDE visualization map with the merged one
                pred_to_gt_map_vis = map_relaxed
            else:
                unmatched = len(gt_ids)
                metrics_relaxed["iou"].extend([0.0] * unmatched)
                metrics_relaxed["bf"].extend([0.0] * unmatched)

        # ---------------------------------------------------------------------
        # 3. Visualization
        # ---------------------------------------------------------------------
        if args.visualize:
            vis_path = os.path.join(vis_dir, f"{name}_vis.png")
            # Passes whichever map is active (Strict or N-to-1)
            visualize_results(gt_inst_map, pred_inst_map, gt_ids, pred_ids, pred_to_gt_map_vis, vis_path, vis_mode_name)

    # Summary
    res = {
        "strict_1to1": {
            "mIoU": float(np.mean(metrics_strict["iou"])) if metrics_strict["iou"] else 0.0,
            "mBF": float(np.mean(metrics_strict["bf"])) if metrics_strict["bf"] else 0.0
        },
        "params": {
            "min_region_size": MIN_REGION_SIZE
        }
    }

    # Only add relaxed results if enabled
    if args.enable_n_to_1:
        res["relaxed_nto1"] = {
            "mIoU": float(np.mean(metrics_relaxed["iou"])) if metrics_relaxed["iou"] else 0.0,
            "mBF": float(np.mean(metrics_relaxed["bf"])) if metrics_relaxed["bf"] else 0.0
        }
        res["params"]["inclusion_thresh"] = PRED_INCLUSION_THRESH
    
    logger.info(f"{'='*40}")
    logger.info(f"RESULTS")
    logger.info(f"{'='*40}")
    logger.info(f"[Strict  1-to-1] mIoU: {res['strict_1to1']['mIoU']:.4f} | mBF: {res['strict_1to1']['mBF']:.4f}")
    
    if args.enable_n_to_1:
        logger.info(f"[Relaxed N-to-1] mIoU: {res['relaxed_nto1']['mIoU']:.4f} | mBF: {res['relaxed_nto1']['mBF']:.4f}")
        logger.info(f"(Merged over-segmented parts if >{PRED_INCLUSION_THRESH*100}% inside GT)")
    else:
        logger.info("(N-to-1 evaluation skipped. Use --enable_n_to_1 to enable.)")
        
    logger.info(f"{'='*40}")

    with open(os.path.join(args.output_dir, "instance_metrics.json"), "w") as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", type=str, default=DEFAULT_GT_DIR)
    parser.add_argument("--pred_dir", type=str, default=DEFAULT_PRED_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gt_merge_map", type=str, default=DEFAULT_GT_MERGE_MAP)
    parser.add_argument("--visualize", action='store_true', default=True)
    # 新增的控制开关
    parser.add_argument("--enable_n_to_1", action='store_true', help="Enable N-to-1 relaxed evaluation mode")
    args = parser.parse_args()
    evaluate(args)
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation Script: Separate Zaha vs. AI Layer Evaluation.

Features:
1. Inputs: 
   - Zaha Layer GT (layer_zaha_kept)
   - AI Layer GT (layer_ai_filled)
   - Fused GT (fused)
2. Output: Metrics calculated independently for each layer.
3. Visualization: Dual-row comparison (Top: Zaha, Bottom: AI).
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

# [重要] 这里填写上一布生成的 GT "根目录"
# 脚本会自动寻找该目录下的 /layer_zaha_kept, /layer_ai_filled, /fused
DEFAULT_GT_ROOT = "/workspace/zaha_eval/outputs/gt_semantic_dist60_499_test_fused_split"

# 预测结果路径 (保留了你的所有注释)
# DEFAULT_PRED_DIR = "/workspace/CityGMLGaussian/output/gaga_10000_train/train/ours_10000/objects_test"
# DEFAULT_OUTPUT_DIR = "/workspace/zaha_eval/eval_results/gaga_level1_seg_test"

# DEFAULT_PRED_DIR = "/workspace/CityGMLGaussian/output/gaga_original_10000_train/train/ours_10000/objects_test"
# DEFAULT_OUTPUT_DIR = "/workspace/zaha_eval/eval_results/gaga_original_seg_test"

DEFAULT_PRED_DIR = "/workspace/CityGMLGaussian/output/gml_10000_train/train/ours_10000/objects_test"
DEFAULT_OUTPUT_DIR = "/workspace/zaha_eval/eval_results/gml_seg_split_test"

# 参数
BOUNDARY_TOLERANCE_PIXELS = 5
MIN_REGION_SIZE = 100 
PRED_INCLUSION_THRESH = 0.80 # N-to-1 阈值

def get_logger(output_dir):
    logger = logging.getLogger('split_eval')
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    fh = logging.FileHandler(os.path.join(output_dir, 'eval_log.txt'), 'w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

# ============================================================================
# ALGORITHMS (通用工具函数)
# ============================================================================

def mask_to_boundary(mask):
    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boundary = np.zeros_like(mask_u8)
    cv2.drawContours(boundary, contours, -1, 1, 1)
    return boundary

def compute_boundary_iou(gt_mask, pred_mask, tolerance=2):
    gt_boundary = mask_to_boundary(gt_mask)
    pred_boundary = mask_to_boundary(pred_mask)
    
    if gt_boundary.sum() == 0 and pred_boundary.sum() == 0: return 1.0
    if gt_boundary.sum() == 0 or pred_boundary.sum() == 0: return 0.0

    gt_dilated = binary_dilation(gt_boundary, disk(tolerance))
    pred_dilated = binary_dilation(pred_boundary, disk(tolerance))
    
    intersection = np.logical_and(gt_dilated, pred_dilated).sum()
    union = np.logical_or(gt_dilated, pred_dilated).sum()

    if union == 0: return 0.0
    return intersection / union

def semantic_to_instances(semantic_map, ignore_val=0, min_size=MIN_REGION_SIZE):
    """
    将语义分割图转换为实例图 (通过连通域分析)
    ignore_val: 通常为0 (背景)
    """
    instance_map = np.zeros_like(semantic_map, dtype=np.int32)
    current_inst_id = 1
    unique_classes = np.unique(semantic_map)
    for cls in unique_classes:
        if cls == ignore_val: continue # 忽略背景/空区域
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

def calculate_intersection_stats(gt_inst_map, pred_inst_map, gt_ids, pred_ids):
    max_pred_id = pred_ids.max() if len(pred_ids) > 0 else 0
    offset = max_pred_id + 1
    
    gt_flat = gt_inst_map.ravel()
    pred_flat = pred_inst_map.ravel()
    
    # 只考虑 GT 非0且非-1的区域
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

# ============================================================================
# CORE MATCHING LOGIC
# ============================================================================

def match_and_score(gt_inst_map, pred_inst_map, mode="strict"):
    """
    核心匹配函数。
    输入: GT 实例图, Pred 实例图
    输出: (mIoU, mBIoU, pred_to_gt_map)
    """
    gt_ids = np.unique(gt_inst_map)
    gt_ids = gt_ids[gt_ids != 0] # 0 is background
    
    pred_ids = np.unique(pred_inst_map)
    pred_ids = pred_ids[(pred_ids != 0) & (pred_ids != -1)]
    
    # 如果该层没有 GT (例如纯背景图)，返回 None
    if len(gt_ids) == 0:
        return [], {}

    if len(pred_ids) == 0:
        # 有 GT 但没 Pred -> 全 0
        return [(gid, 0.0, 0.0) for gid in gt_ids], {}

    intersection_mat, gt_areas, pred_areas = calculate_intersection_stats(gt_inst_map, pred_inst_map, gt_ids, pred_ids)
    
    if intersection_mat is None:
        return [(gid, 0.0, 0.0) for gid in gt_ids], {}

    results = []
    pred_to_gt_map = {}

    # --- 策略 A: Strict 1-to-1 (Hungarian) ---
    if mode == "strict":
        area_sum = gt_areas[:, None] + pred_areas[None, :]
        union_mat = area_sum - intersection_mat
        iou_matrix = np.divide(intersection_mat, union_mat, out=np.zeros_like(intersection_mat), where=union_mat!=0)
        
        cost_matrix = 1.0 - iou_matrix
        cost_matrix = np.nan_to_num(cost_matrix, nan=1.0)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_rows = set(row_ind)
        for r, c in zip(row_ind, col_ind):
            gid = gt_ids[r]
            pid = pred_ids[c]
            iou = iou_matrix[r, c]
            
            if iou > 1e-6:
                pred_to_gt_map[pid] = gid 
                mask_gt = (gt_inst_map == gid).astype(bool)
                mask_pred = (pred_inst_map == pid).astype(bool)
                biou = compute_boundary_iou(mask_gt, mask_pred, tolerance=BOUNDARY_TOLERANCE_PIXELS)
            else:
                biou = 0.0
            results.append((gid, iou, biou))
            
        for i in range(len(gt_ids)):
            if i not in matched_rows:
                results.append((gt_ids[i], 0.0, 0.0))

    # --- 策略 B: Relaxed N-to-1 ---
    elif mode == "relaxed":
        pred_used = np.zeros(len(pred_ids), dtype=bool)
        for i, gid in enumerate(gt_ids):
            gt_mask = (gt_inst_map == gid).astype(bool)
            candidates_idx = []
            
            # 寻找所有包含度高的预测框
            for j, pid in enumerate(pred_ids):
                if pred_used[j]: continue
                p_area = pred_areas[j]
                if p_area == 0: continue
                inclusion = intersection_mat[i, j] / p_area
                if inclusion > PRED_INCLUSION_THRESH:
                    candidates_idx.append(j)
            
            if not candidates_idx:
                results.append((gid, 0.0, 0.0))
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
            biou = compute_boundary_iou(gt_mask, fused_pred_mask, tolerance=BOUNDARY_TOLERANCE_PIXELS)
            results.append((gid, iou, biou))
            
    return results, pred_to_gt_map

# ============================================================================
# VISUALIZATION (Dual Row)
# ============================================================================

def visualize_dual_row(gt_zaha, gt_ai, pred_inst, map_zaha, map_ai, save_path):
    """
    两行可视化：
    Row 1: Zaha GT | Pred matched to Zaha | Error Zaha
    Row 2: AI GT   | Pred matched to AI   | Error AI
    """
    
    # Generate common palette
    max_id = max(
        gt_zaha.max(), gt_ai.max(), pred_inst.max()
    ) + 100
    
    np.random.seed(42)
    palette = np.random.randint(0, 255, (max_id + 1, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0] # Background black
    
    # Helper to colorize
    def get_vis(mask, id_map=None):
        vis = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        # 如果提供了 id_map (pred -> gt)，则 pred 颜色跟随 gt
        if id_map:
            # 只有在 map 里的才上色
            for pid, gid in id_map.items():
                vis[mask == pid] = palette[gid % len(palette)]
            # 没匹配上的保持黑色或灰色？这里保持黑色以减少噪音
        else:
            # GT 直接上色
            unique_ids = np.unique(mask)
            for uid in unique_ids:
                if uid == 0: continue
                vis[mask == uid] = palette[uid % len(palette)]
        return vis

    # Helper for Error Map
    def get_error(gt_mask, pred_mask, mapping):
        # Remap pred to gt IDs
        remapped = np.zeros_like(gt_mask)
        for pid, gid in mapping.items():
            remapped[pred_mask == pid] = gid
        
        err_vis = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
        err_vis[:] = [30, 30, 30] # Dark grey background
        
        # True Positives (Green)
        tp = (remapped == gt_mask) & (gt_mask != 0)
        err_vis[tp] = [0, 255, 0]
        
        # False Negatives / Mismatch (Red) -> GT exists but pred is wrong/missing
        fn = (gt_mask != 0) & (~tp)
        err_vis[fn] = [255, 0, 0]
        
        # False Positives (Blue) -> Pred exists (mapped) but GT is 0
        # 注意：这里我们只关心针对该层的 GT。如果 Pred 匹配到了该层 ID 但 GT 是 0，则是 FP
        fp = (remapped != 0) & (gt_mask == 0)
        err_vis[fp] = [0, 0, 255]
        
        return err_vis

    # Row 1 Images
    vis_zaha_gt = get_vis(gt_zaha)
    vis_zaha_pred = get_vis(pred_inst, map_zaha)
    vis_zaha_err = get_error(gt_zaha, pred_inst, map_zaha)
    
    # Row 2 Images
    vis_ai_gt = get_vis(gt_ai)
    vis_ai_pred = get_vis(pred_inst, map_ai)
    vis_ai_err = get_error(gt_ai, pred_inst, map_ai)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Titles
    fs = 14
    axes[0,0].set_title("Zaha GT (Core)", fontsize=fs)
    axes[0,1].set_title("Pred (Matched to Zaha)", fontsize=fs)
    axes[0,2].set_title("Zaha Errors (Green=TP, Red=FN)", fontsize=fs)
    
    axes[1,0].set_title("AI GT (Filled)", fontsize=fs)
    axes[1,1].set_title("Pred (Matched to AI)", fontsize=fs)
    axes[1,2].set_title("AI Errors (Green=TP, Red=FN)", fontsize=fs)

    # Plot
    axes[0,0].imshow(vis_zaha_gt); axes[0,0].axis('off')
    axes[0,1].imshow(vis_zaha_pred); axes[0,1].axis('off')
    axes[0,2].imshow(vis_zaha_err); axes[0,2].axis('off')
    
    axes[1,0].imshow(vis_ai_gt); axes[1,0].axis('off')
    axes[1,1].imshow(vis_ai_pred); axes[1,1].axis('off')
    axes[1,2].imshow(vis_ai_err); axes[1,2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)

# ============================================================================
# MAIN EVALUATION LOOP
# ============================================================================

def evaluate(args):
    # 1. 路径设置
    dir_fused_gt = os.path.join(args.gt_root, "fused")
    dir_zaha_gt = os.path.join(args.gt_root, "layer_zaha_kept")
    dir_ai_gt = os.path.join(args.gt_root, "layer_ai_filled")
    
    if not os.path.exists(dir_zaha_gt) or not os.path.exists(dir_ai_gt):
        print(f"Error: subfolders 'layer_zaha_kept' or 'layer_ai_filled' not found in {args.gt_root}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "visualizations")
    if args.visualize: os.makedirs(vis_dir, exist_ok=True)
    
    logger = get_logger(args.output_dir)
    
    # 2. 文件匹配
    gt_files = sorted(glob.glob(os.path.join(dir_fused_gt, "*.npy")))
    pred_files = sorted(glob.glob(os.path.join(args.pred_dir, "*.png")))
    
    gt_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in gt_files}
    pred_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in pred_files}
    common_names = sorted(list(set(gt_dict.keys()) & set(pred_dict.keys())))
    
    logger.info(f"Evaluating {len(common_names)} images...")
    mode = "relaxed" if args.enable_n_to_1 else "strict"
    logger.info(f"Mode: {mode}")

    # 3. 结果容器
    metrics = {
        "zaha": {"iou": [], "biou": []},
        "ai":   {"iou": [], "biou": []},
        "all":  {"iou": [], "biou": []}
    }

    for name in tqdm(common_names, desc="Evaluating"):
        # A. Load Prediction (Instance Map)
        pred_inst_map = cv2.imread(pred_dict[name], cv2.IMREAD_UNCHANGED)
        if pred_inst_map.ndim == 3: pred_inst_map = pred_inst_map[:,:,0]
        pred_inst_map = pred_inst_map.astype(np.int32)
        
        # B. Load 3 Semantic GTs
        path_zaha = os.path.join(dir_zaha_gt, f"{name}.npy")
        path_ai = os.path.join(dir_ai_gt, f"{name}.npy")
        path_all = os.path.join(dir_fused_gt, f"{name}.npy")
        
        sem_zaha = np.load(path_zaha)
        sem_ai = np.load(path_ai)
        sem_all = np.load(path_all)

        # Resize GT if needed
        ph, pw = pred_inst_map.shape
        if sem_all.shape != (ph, pw):
            sem_zaha = cv2.resize(sem_zaha, (pw, ph), interpolation=cv2.INTER_NEAREST)
            sem_ai = cv2.resize(sem_ai, (pw, ph), interpolation=cv2.INTER_NEAREST)
            sem_all = cv2.resize(sem_all, (pw, ph), interpolation=cv2.INTER_NEAREST)

        # C. Convert Semantic to Instance (On the fly)
        # 注意: 这里的 sem_zaha 中非核心部分已经是0了，可以直接转
        inst_zaha = semantic_to_instances(sem_zaha)
        inst_ai = semantic_to_instances(sem_ai)
        inst_all = semantic_to_instances(sem_all)

        # D. Run Evaluation for each layer
        # 1. Zaha Layer
        res_zaha, map_zaha = match_and_score(inst_zaha, pred_inst_map, mode=mode)
        for _, iou, biou in res_zaha:
            metrics["zaha"]["iou"].append(iou)
            metrics["zaha"]["biou"].append(biou)
            
        # 2. AI Layer
        res_ai, map_ai = match_and_score(inst_ai, pred_inst_map, mode=mode)
        for _, iou, biou in res_ai:
            metrics["ai"]["iou"].append(iou)
            metrics["ai"]["biou"].append(biou)
            
        # 3. Overall (Fused)
        res_all, map_all = match_and_score(inst_all, pred_inst_map, mode=mode)
        for _, iou, biou in res_all:
            metrics["all"]["iou"].append(iou)
            metrics["all"]["biou"].append(biou)

        # E. Visualization
        if args.visualize:
            save_path = os.path.join(vis_dir, f"{name}_eval.png")
            # 传入 inst_zaha 和 inst_ai 供显示
            # 传入 map_zaha 和 map_ai 供上色匹配
            visualize_dual_row(inst_zaha, inst_ai, pred_inst_map, map_zaha, map_ai, save_path)

    # 4. Final Stats
    def get_stats(m_dict):
        count = len(m_dict["iou"])
        return {
            "mIoU": float(np.mean(m_dict["iou"])) if count > 0 else 0.0,
            "mBIoU": float(np.mean(m_dict["biou"])) if count > 0 else 0.0,
            "count": count
        }

    final_results = {
        "mode": mode,
        "zaha_layer": get_stats(metrics["zaha"]),
        "ai_layer": get_stats(metrics["ai"]),
        "overall_fused": get_stats(metrics["all"])
    }

    logger.info("="*40)
    logger.info(f"FINAL RESULTS ({mode})")
    logger.info("="*40)
    logger.info(f"ZAHA LAYER (Core)   : mIoU={final_results['zaha_layer']['mIoU']:.4f} | mBIoU={final_results['zaha_layer']['mBIoU']:.4f}")
    logger.info(f"AI LAYER   (Filled) : mIoU={final_results['ai_layer']['mIoU']:.4f} | mBIoU={final_results['ai_layer']['mBIoU']:.4f}")
    logger.info(f"OVERALL    (Fused)  : mIoU={final_results['overall_fused']['mIoU']:.4f} | mBIoU={final_results['overall_fused']['mBIoU']:.4f}")

    with open(os.path.join(args.output_dir, "metrics_summary.json"), "w") as f:
        json.dump(final_results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_root", type=str, default=DEFAULT_GT_ROOT, help="Parent dir containing 'fused', 'layer_zaha_kept', 'layer_ai_filled'")
    parser.add_argument("--pred_dir", type=str, default=DEFAULT_PRED_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--visualize", action='store_true', default=True)
    parser.add_argument("--enable_n_to_1", action='store_true')
    args = parser.parse_args()
    evaluate(args)
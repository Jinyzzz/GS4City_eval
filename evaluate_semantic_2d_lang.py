#!/usr/bin/env python
"""
Evaluate LangSplat semantic segmentation using Split Ground Truth (Zaha vs AI vs Fused),
with 3-row comparison:
  1) Building-scale (merge 1/2/3/12/... -> 200, prompt "building")
     - Evaluation/Visualization uses method #3 BUT restricted to GT coverage:
       eval ONLY pixels where (GT is valid) AND ((GT==building) OR (Pred==building))
       -> excludes TN and does NOT penalize predictions outside GT coverage (-1).
  2) Building parts (ids: 1/2/3/12) on Zaha layer (default)
  3) Non-building (ids: 101/103/104) on AI layer (default)

Structure Requirement for gt_split_root:
    /fused             -> Combined GTs
    /layer_zaha_kept   -> Only Zaha architecture parts
    /layer_ai_filled   -> Only AI filled background parts
"""

# ============================================================================
# DEFAULT PARAMETERS
# ============================================================================

DEFAULT_RENDERED_FEATURES_DIR = "/workspace/LangSplat/output/subset_building5_1/test/ours_None/renders_npy"
DEFAULT_AE_CHECKPOINT = "/workspace/LangSplat/autoencoder/ckpt/subset_building5/best_ckpt.pth"
DEFAULT_CLASS_MAPPING_PATH = "class_mapping.json"

DEFAULT_GT_SPLIT_ROOT = "/workspace/zaha_eval/gt/subset5_499_dist60_test"
DEFAULT_OUTPUT_DIR = "/workspace/zaha_eval/eval_results/subset5_lang_sem_test0.4"

DEFAULT_MASK_THRESH = 0.4
DEFAULT_USE_SOFTMAX = False
DEFAULT_SAVE_VISUALIZATIONS = True
DEFAULT_NUM_IMAGES = None

DEFAULT_ENCODER_DIMS = [256, 128, 64, 32, 3]
DEFAULT_DECODER_DIMS = [16, 32, 64, 128, 256, 256, 512]

DEFAULT_CLASS_COLORS_PATH = "class_colors.json"
DEFAULT_GT_MERGE_MAP_PATH = "gt_merge_map.json"

# Three-row groups
DEFAULT_BUILDING_ID = 200
DEFAULT_PART_IDS = [1, 2, 3, 12]
DEFAULT_NONBUILD_IDS = [101, 103, 104]

# Which GT layer to use for each row (fused / zaha / ai)
DEFAULT_GT_LAYER_FOR_BUILDING = "fused"
DEFAULT_GT_LAYER_FOR_PARTS = "layer_zaha_kept"
DEFAULT_GT_LAYER_FOR_NONBUILD = "layer_ai_filled"

import argparse
import json
import os
import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from collections import defaultdict

import numpy as np
import torch
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

import colormaps  # noqa: F401
from autoencoder.model import Autoencoder
from openclip_encoder import OpenCLIPNetwork


# ============================================================================
# Logger
# ============================================================================
def get_logger(name, log_file=None, log_level=logging.INFO):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        handlers.append(file_handler)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


# ============================================================================
# Ground Truth Loading & Processing
# ============================================================================
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
    layer_name: str = "GT"
) -> Dict[str, Dict]:
    """
    Load a specific layer of ground truth.
    """
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

        semantic_map = np.load(gt_path)

        # 1) Label merge (optional)
        if label_merge_map:
            semantic_map = apply_label_merges(semantic_map, label_merge_map)

        # 2) Filter invalid class ids -> -1(ignore)
        invalid_mask = (semantic_map >= 0) & (~np.isin(semantic_map, list(valid_class_ids)))
        semantic_map[invalid_mask] = -1

        coverage_mask = semantic_map >= 0
        unique_classes = np.unique(semantic_map[coverage_mask]).tolist()

        gt_data[img_name] = {
            'semantic_map': semantic_map,
            'coverage_mask': coverage_mask,
            'classes': unique_classes
        }

    logger.info(f"   Loaded {len(gt_data)} images for {layer_name}.")
    return gt_data


# ============================================================================
# Feature Decoding & Query
# ============================================================================
def decode_features(compressed_features, autoencoder, device):
    H, W, _ = compressed_features.shape
    with torch.no_grad():
        flat = compressed_features.reshape(-1, 3).to(device)
        decoded = autoencoder.decode(flat).reshape(H, W, 512)
    return decoded


def query_semantic_map(features_512, clip_model, queries, use_softmax=False):
    """
    Returns:
      pred_class_idx : (H,W) each pixel chooses query index [0..n_queries-1]
      heatmaps       : dict query->(H,W) relevance map
      confidence     : (H,W) pseudo confidence (softmax prob if use_softmax else per-query minmax score)
    """
    H, W, _ = features_512.shape
    clip_model.set_positives(queries)
    features_input = features_512.unsqueeze(0)

    with torch.no_grad():
        relevance_maps = clip_model.get_max_across(features_input).squeeze(0)  # (nq, H, W)

    heatmaps = {q: relevance_maps[i].cpu().numpy() for i, q in enumerate(queries)}

    if use_softmax:
        probs = torch.softmax(relevance_maps, dim=0)
        pred_class = torch.argmax(probs, dim=0).cpu().numpy()
        confidence = torch.max(probs, dim=0)[0].cpu().numpy()
    else:
        argmax_class = relevance_maps.argmax(dim=0).cpu().numpy()
        confidence = np.zeros((H, W), dtype=np.float32)
        for i in range(len(queries)):
            rel = relevance_maps[i].cpu().numpy()
            rel_norm = (rel - rel.min()) / (rel.max() - rel.min() + 1e-9)
            mask = (argmax_class == i)
            confidence[mask] = rel_norm[mask]
        pred_class = argmax_class

    return pred_class, heatmaps, confidence


# ============================================================================
# Metrics
# ============================================================================
def compute_multiclass_metrics(pred_semantic, gt_semantic, eval_mask, class_mapping):
    """
    Multi-class metrics restricted to eval_mask.
    """
    pred_valid = pred_semantic[eval_mask]
    gt_valid = gt_semantic[eval_mask]

    if gt_valid.size == 0:
        return None

    gt_classes = np.unique(gt_valid).tolist()
    results = {
        'class_iou': {}, 'class_acc': {},
        'pixel_count': int(gt_valid.size),
        'correct_count': int((pred_valid == gt_valid).sum())
    }

    for cls_id in gt_classes:
        cls_name = class_mapping.get(int(cls_id), f"ID_{cls_id}")
        if isinstance(cls_name, list):
            cls_name = cls_name[0]

        gt_cls_mask = (gt_valid == cls_id)
        pred_cls_mask = (pred_valid == cls_id)

        intersection = np.logical_and(pred_cls_mask, gt_cls_mask).sum()
        union = np.logical_or(pred_cls_mask, gt_cls_mask).sum()

        results['class_iou'][cls_name] = float(intersection / union) if union > 0 else 0.0
        results['class_acc'][cls_name] = float(intersection / gt_cls_mask.sum()) if gt_cls_mask.sum() > 0 else 0.0

    results['mean_iou'] = float(np.mean(list(results['class_iou'].values()))) if results['class_iou'] else 0.0
    results['pixel_acc'] = float(results['correct_count'] / results['pixel_count'])
    return results


def compute_binary_metrics_union_mask(pred_is_pos: np.ndarray, gt_is_pos: np.ndarray, eval_mask: np.ndarray):
    """
    Binary metrics on eval_mask.
    We use eval_mask = (GT_valid) & (GT_pos | Pred_pos):
      - excludes TN
      - does not penalize predictions outside GT coverage.
    Reports IoU / Precision / Recall / F1.
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


# ============================================================================
# Visualization: 3 Rows with Error Maps
# ============================================================================
def visualize_three_row_error(
    # row1 building
    pred_building_mask: np.ndarray, gt_building_mask: np.ndarray,
    # row2 parts
    pred_parts: np.ndarray, gt_parts: np.ndarray,
    # row3 non-building
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
        - Error only on (GT_valid) & (GT_building | Pred_building)
        - Outside that mask: gray
      Row 2: [GT Parts]    [Pred Parts]    [Error Parts]
      Row 3: [GT NonB]     [Pred NonB]     [Error NonB]
    """

    def colorize(mask, palette):
        h, w = mask.shape
        col = np.zeros((h, w, 3), dtype=np.uint8)
        col[mask < 0] = [30, 30, 30]
        for cid in np.unique(mask):
            if cid < 0:
                continue
            if cid in palette:
                col[mask == cid] = palette[cid]
            else:
                col[mask == cid] = [128, 128, 128]
        return col

    def get_error_map_multiclass(prediction, ground_truth):
        """
        Green=Correct, Red=Wrong, Gray=Ignore (where GT<0)
        """
        h, w = ground_truth.shape
        error_img = np.full((h, w, 3), 30, dtype=np.uint8)
        valid_mask = (ground_truth >= 0)
        correct_mask = (prediction == ground_truth) & valid_mask
        wrong_mask = (prediction != ground_truth) & valid_mask
        error_img[correct_mask] = [0, 255, 0]
        error_img[wrong_mask] = [255, 0, 0]
        return error_img

    def get_error_map_binary_union(pred_is_pos: np.ndarray, gt_is_pos: np.ndarray, eval_mask: np.ndarray):
        """
        Only eval_mask pixels are colored:
          - Green: TP (True==True)
          - Red:   FP or FN (True!=False)
          - Gray:  outside eval_mask
        """
        h, w = gt_is_pos.shape
        error_img = np.full((h, w, 3), 30, dtype=np.uint8)
        correct = (pred_is_pos == gt_is_pos) & eval_mask
        wrong = (pred_is_pos != gt_is_pos) & eval_mask
        error_img[correct] = [0, 255, 0]
        error_img[wrong] = [255, 0, 0]
        return error_img

    # Build palette
    unique_ids = set(np.unique(pred_parts)) | set(np.unique(gt_parts)) | set(np.unique(pred_non)) | set(np.unique(gt_non))
    unique_ids |= {building_id}
    cmap_tab20 = plt.cm.get_cmap('tab20', max(len(unique_ids), 1) + 5)

    palette = {}
    for i, uid in enumerate(sorted(list(unique_ids))):
        if uid < 0:
            continue
        if class_colors and uid in class_colors:
            palette[uid] = np.array(class_colors[uid], dtype=np.uint8)
        else:
            palette[uid] = (np.array(cmap_tab20(i)[:3]) * 255).astype(np.uint8)

    # -------------------------
    # Row1: Building (restricted to GT coverage)
    # -------------------------
    gt_valid = (gt_building_mask >= 0)
    gt_is_b = (gt_building_mask == building_id)
    pred_is_b = (pred_building_mask == building_id)

    # IMPORTANT: do not penalize outside GT coverage
    eval_b = gt_valid & (gt_is_b | pred_is_b)

    # For display: GT only shows building, Pred only shows building (can show outside coverage too)
    gt_b_vis = gt_building_mask.copy()
    pred_b_vis = pred_building_mask.copy()
    gt_b_vis[~gt_is_b] = -1
    pred_b_vis[~pred_is_b] = -1

    vis_gt_b = colorize(gt_b_vis, palette)
    vis_pr_b = colorize(pred_b_vis, palette)
    err_b = get_error_map_binary_union(pred_is_b, gt_is_b, eval_b)

    # -------------------------
    # Row2: Parts
    # -------------------------
    gt_parts_vis = gt_parts.copy()
    pred_parts_vis = pred_parts.copy()
    gt_parts_vis[~np.isin(gt_parts_vis, part_ids)] = -1
    pred_parts_vis[~np.isin(pred_parts_vis, part_ids)] = -1
    vis_gt_p = colorize(gt_parts_vis, palette)
    vis_pr_p = colorize(pred_parts_vis, palette)
    err_p = get_error_map_multiclass(pred_parts_vis, gt_parts_vis)

    # -------------------------
    # Row3: Non-building
    # -------------------------
    gt_non_vis = gt_non.copy()
    pred_non_vis = pred_non.copy()
    gt_non_vis[~np.isin(gt_non_vis, nonbuild_ids)] = -1
    pred_non_vis[~np.isin(pred_non_vis, nonbuild_ids)] = -1
    vis_gt_n = colorize(gt_non_vis, palette)
    vis_pr_n = colorize(pred_non_vis, palette)
    err_n = get_error_map_multiclass(pred_non_vis, gt_non_vis)

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    axes[0, 0].imshow(vis_gt_b)
    axes[0, 0].set_title("GT Building (ID=200)")
    axes[0, 1].imshow(vis_pr_b)
    axes[0, 1].set_title("Pred Building")
    axes[0, 2].imshow(err_b)
    axes[0, 2].set_title("Error (Building) [GTcov & (GT|Pred)]\nGreen=TP, Red=FP/FN")

    axes[1, 0].imshow(vis_gt_p)
    axes[1, 0].set_title("GT Parts (Zaha)")
    axes[1, 1].imshow(vis_pr_p)
    axes[1, 1].set_title("Pred Parts")
    axes[1, 2].imshow(err_p)
    axes[1, 2].set_title("Error (Parts)\nGreen=Correct, Red=Wrong")

    axes[2, 0].imshow(vis_gt_n)
    axes[2, 0].set_title("GT Non-building (AI)")
    axes[2, 1].imshow(vis_pr_n)
    axes[2, 1].set_title("Pred Non-building")
    axes[2, 2].imshow(err_n)
    axes[2, 2].set_title("Error (Non-building)\nGreen=Correct, Red=Wrong")

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main Evaluation Loop
# ============================================================================
def evaluate(args, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dir_fused = os.path.join(args.gt_split_root, "fused")
    dir_zaha = os.path.join(args.gt_split_root, "layer_zaha_kept")
    dir_ai = os.path.join(args.gt_split_root, "layer_ai_filled")

    logger.info(f"Target Evaluation Directories:")
    logger.info(f"  > Fused: {dir_fused}")
    logger.info(f"  > Zaha:  {dir_zaha}")
    logger.info(f"  > AI:    {dir_ai}")

    clip_model = OpenCLIPNetwork(device)
    autoencoder = Autoencoder(args.encoder_dims, args.decoder_dims).to(device)
    autoencoder.load_state_dict(torch.load(args.ae_checkpoint, map_location=device))
    autoencoder.eval()

    with open(args.class_mapping, 'r') as f:
        class_mapping = {int(k): v for k, v in json.load(f).items()}

    # merge maps (split into parts vs building)
    parts_merge_map = None
    building_merge_map = None
    if args.gt_merge_map and os.path.exists(args.gt_merge_map):
        with open(args.gt_merge_map, 'r') as f:
            raw = json.load(f)
        raw_int = {int(k): [int(v) for v in vals] for k, vals in raw.items()}

        if args.parts_merge_target in raw_int:
            parts_merge_map = {args.parts_merge_target: raw_int[args.parts_merge_target]}

        if args.building_id in raw_int:
            building_merge_map = {args.building_id: raw_int[args.building_id]}

    class_colors = {}
    if os.path.exists(args.class_colors):
        with open(args.class_colors) as f:
            class_colors = {int(k): v for k, v in json.load(f).items()}

    feature_files = sorted(glob.glob(os.path.join(args.rendered_features, '*.npy')))
    if args.num_images:
        feature_files = feature_files[:args.num_images]
    image_names = [Path(f).stem for f in feature_files]

    # Load GTs using parts_merge_map only
    gt_data_fused = load_ground_truth_layer(dir_fused, image_names, class_mapping, logger, parts_merge_map, "Fused")
    gt_data_zaha = load_ground_truth_layer(dir_zaha, image_names, class_mapping, logger, parts_merge_map, "Zaha")
    gt_data_ai = load_ground_truth_layer(dir_ai, image_names, class_mapping, logger, parts_merge_map, "AI")

    common_images = sorted(list(set(gt_data_fused.keys()) & set(image_names)))
    logger.info(f"Evaluating on {len(common_images)} images that have Fused GT and features.")

    building_id = int(args.building_id)
    part_ids = [int(x) for x in args.part_ids]
    nonbuild_ids = [int(x) for x in args.nonbuild_ids]

    # Building queries
    if building_id not in class_mapping:
        raise ValueError(f"Building id {building_id} not found in class_mapping.")
    b_lbl = class_mapping[building_id]
    building_queries = b_lbl if isinstance(b_lbl, list) else [b_lbl]
    building_query_ids_np = np.array([building_id] * len(building_queries), dtype=np.int32)

    # Parts queries
    parts_queries, parts_query_ids = [], []
    for cid in part_ids:
        if cid not in class_mapping:
            logger.warning(f"[WARN] Part id {cid} not found in class_mapping, skipped.")
            continue
        lbl = class_mapping[cid]
        if isinstance(lbl, list):
            parts_queries.extend(lbl)
            parts_query_ids.extend([cid] * len(lbl))
        else:
            parts_queries.append(lbl)
            parts_query_ids.append(cid)
    parts_query_ids_np = np.array(parts_query_ids, dtype=np.int32)

    # Non-building queries
    non_queries, non_query_ids = [], []
    for cid in nonbuild_ids:
        if cid not in class_mapping:
            logger.warning(f"[WARN] Non-building id {cid} not found in class_mapping, skipped.")
            continue
        lbl = class_mapping[cid]
        if isinstance(lbl, list):
            non_queries.extend(lbl)
            non_query_ids.extend([cid] * len(lbl))
        else:
            non_queries.append(lbl)
            non_query_ids.append(cid)
    non_query_ids_np = np.array(non_query_ids, dtype=np.int32)

    metrics_log = {
        "building": [],      # union-mask binary metrics restricted to GT coverage
        "parts": [],
        "nonbuilding": []
    }

    def get_gt_pack(img_name: str, row: str):
        if row == "building":
            layer = args.gt_layer_building
        elif row == "parts":
            layer = args.gt_layer_parts
        else:
            layer = args.gt_layer_nonbuild

        if layer == "fused":
            return gt_data_fused.get(img_name)
        if layer == "zaha":
            return gt_data_zaha.get(img_name)
        if layer == "ai":
            return gt_data_ai.get(img_name)
        if layer == "layer_zaha_kept":
            return gt_data_zaha.get(img_name)
        if layer == "layer_ai_filled":
            return gt_data_ai.get(img_name)
        return gt_data_fused.get(img_name)

    for img_name in tqdm(common_images, desc="Evaluating"):
        feat_path = os.path.join(args.rendered_features, f"{img_name}.npy")
        compressed = torch.from_numpy(np.load(feat_path)).float()
        decoded = decode_features(compressed, autoencoder, device)

        # Predict Building (ID=200)
        raw_idx_b, _, conf_b = query_semantic_map(decoded, clip_model, building_queries, args.use_softmax)
        pred_building = np.full_like(raw_idx_b, -1, dtype=np.int32)
        valid_b = (raw_idx_b >= 0)
        pred_building[valid_b] = building_query_ids_np[raw_idx_b[valid_b]]
        pred_building[conf_b < args.mask_thresh] = -1
        if building_merge_map:
            pred_building = apply_label_merges(pred_building, building_merge_map)

        # Predict Parts
        raw_idx_p, _, conf_p = query_semantic_map(decoded, clip_model, parts_queries, args.use_softmax)
        pred_parts = np.full_like(raw_idx_p, -1, dtype=np.int32)
        valid_p = (raw_idx_p >= 0)
        pred_parts[valid_p] = parts_query_ids_np[raw_idx_p[valid_p]]
        pred_parts[conf_p < args.mask_thresh] = -1
        if parts_merge_map:
            pred_parts = apply_label_merges(pred_parts, parts_merge_map)

        # Predict Non-building
        raw_idx_n, _, conf_n = query_semantic_map(decoded, clip_model, non_queries, args.use_softmax)
        pred_non = np.full_like(raw_idx_n, -1, dtype=np.int32)
        valid_n = (raw_idx_n >= 0)
        pred_non[valid_n] = non_query_ids_np[raw_idx_n[valid_n]]
        pred_non[conf_n < args.mask_thresh] = -1
        if parts_merge_map:
            pred_non = apply_label_merges(pred_non, parts_merge_map)

        # Resize preds to fused size
        base_gt = gt_data_fused[img_name]['semantic_map']
        if pred_building.shape != base_gt.shape:
            pred_building = cv2.resize(pred_building.astype(np.int32),
                                       (base_gt.shape[1], base_gt.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
            pred_parts = cv2.resize(pred_parts.astype(np.int32),
                                    (base_gt.shape[1], base_gt.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
            pred_non = cv2.resize(pred_non.astype(np.int32),
                                  (base_gt.shape[1], base_gt.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

        # Building metrics: restrict to GT coverage
        gt_pack_b = get_gt_pack(img_name, "building")
        if gt_pack_b is not None:
            gt_map_b = gt_pack_b['semantic_map']
            if building_merge_map:
                gt_map_b2 = apply_label_merges(gt_map_b, building_merge_map)
            else:
                gt_map_b2 = gt_map_b

            gt_valid = (gt_map_b2 >= 0)
            gt_is_b = (gt_map_b2 == building_id)
            pred_is_b = (pred_building == building_id)

            # KEY CHANGE: only evaluate within GT coverage
            eval_mask_b = gt_valid & (gt_is_b | pred_is_b)

            bm = compute_binary_metrics_union_mask(pred_is_b, gt_is_b, eval_mask_b)
            if bm:
                metrics_log["building"].append(bm)

        # Parts metrics (only GT pixels in part_ids)
        gt_pack_p = get_gt_pack(img_name, "parts")
        if gt_pack_p is not None:
            gt_map_p = gt_pack_p['semantic_map']
            eval_mask_p = np.isin(gt_map_p, part_ids)
            pm = compute_multiclass_metrics(pred_parts, gt_map_p, eval_mask_p, class_mapping)
            if pm:
                metrics_log["parts"].append(pm)

        # Non-building metrics (only GT pixels in nonbuild_ids)
        gt_pack_n = get_gt_pack(img_name, "nonbuilding")
        if gt_pack_n is not None:
            gt_map_n = gt_pack_n['semantic_map']
            eval_mask_n = np.isin(gt_map_n, nonbuild_ids)
            nm = compute_multiclass_metrics(pred_non, gt_map_n, eval_mask_n, class_mapping)
            if nm:
                metrics_log["nonbuilding"].append(nm)

        # Visualization
        if args.save_visualizations:
            vis_dir = os.path.join(args.output_dir, "visualizations_3rows")
            os.makedirs(vis_dir, exist_ok=True)

            gt_pack_bv = get_gt_pack(img_name, "building") or gt_data_fused.get(img_name)
            gt_pack_pv = get_gt_pack(img_name, "parts") or gt_data_zaha.get(img_name) or gt_data_fused.get(img_name)
            gt_pack_nv = get_gt_pack(img_name, "nonbuilding") or gt_data_ai.get(img_name) or gt_data_fused.get(img_name)

            gt_b_map = gt_pack_bv['semantic_map']
            if building_merge_map:
                gt_b_map = apply_label_merges(gt_b_map, building_merge_map)

            visualize_three_row_error(
                pred_building, gt_b_map,
                pred_parts, gt_pack_pv['semantic_map'],
                pred_non, gt_pack_nv['semantic_map'],
                class_colors,
                os.path.join(vis_dir, f"{img_name}_3rows.png"),
                building_id=building_id,
                part_ids=part_ids,
                nonbuild_ids=nonbuild_ids,
            )

    # =========================================================================
    # Aggregate & Report
    # =========================================================================
    final_report = {}

    logger.info("\n" + "=" * 70)
    logger.info("FINAL 3-ROW EVALUATION RESULTS (Building restricted to GT coverage, no TN)")
    logger.info("=" * 70)

    # Building aggregate
    bdata = metrics_log["building"]
    if bdata:
        final_report["building"] = {
            "N": len(bdata),
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

    # Parts aggregate
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
            "N": len(pdata),
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

    # Non-building aggregate
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
            "N": len(ndata),
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

    # Save report
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "three_rows_results.json")
    with open(out_path, "w") as f:
        json.dump(final_report, f, indent=2)

    logger.info(f"\n3-row results saved to: {out_path}")
    logger.info(f"Visualizations saved to: {os.path.join(args.output_dir, 'visualizations_3rows')}")


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_split_root', type=str, default=DEFAULT_GT_SPLIT_ROOT,
                        help='Root dir containing fused, layer_zaha_kept, layer_ai_filled')
    parser.add_argument('--rendered_features', type=str, default=DEFAULT_RENDERED_FEATURES_DIR)
    parser.add_argument('--ae_checkpoint', type=str, default=DEFAULT_AE_CHECKPOINT)
    parser.add_argument('--class_mapping', type=str, default=DEFAULT_CLASS_MAPPING_PATH)
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--gt_merge_map', type=str, default=DEFAULT_GT_MERGE_MAP_PATH)
    parser.add_argument('--class_colors', type=str, default=DEFAULT_CLASS_COLORS_PATH)

    parser.add_argument('--mask_thresh', type=float, default=DEFAULT_MASK_THRESH)
    parser.add_argument('--use_softmax', action='store_true', default=DEFAULT_USE_SOFTMAX)
    parser.add_argument('--save_visualizations', action='store_true', default=DEFAULT_SAVE_VISUALIZATIONS)
    parser.add_argument('--num_images', type=int, default=DEFAULT_NUM_IMAGES)

    parser.add_argument('--encoder_dims', nargs='+', type=int, default=DEFAULT_ENCODER_DIMS)
    parser.add_argument('--decoder_dims', nargs='+', type=int, default=DEFAULT_DECODER_DIMS)

    # 3-row config
    parser.add_argument('--building_id', type=int, default=DEFAULT_BUILDING_ID)
    parser.add_argument('--parts_merge_target', type=int, default=1,
                        help='Which target id is used for parts merge (default=1). Only this key will be used as parts_merge_map.')
    parser.add_argument('--part_ids', nargs='+', type=int, default=DEFAULT_PART_IDS)
    parser.add_argument('--nonbuild_ids', nargs='+', type=int, default=DEFAULT_NONBUILD_IDS)

    # Which GT layer used per row: fused / zaha / ai (also accepts layer_zaha_kept/layer_ai_filled)
    parser.add_argument('--gt_layer_building', type=str, default=DEFAULT_GT_LAYER_FOR_BUILDING,
                        help='GT layer for Building row: fused|zaha|ai')
    parser.add_argument('--gt_layer_parts', type=str, default=DEFAULT_GT_LAYER_FOR_PARTS,
                        help='GT layer for Parts row: fused|zaha|ai')
    parser.add_argument('--gt_layer_nonbuild', type=str, default=DEFAULT_GT_LAYER_FOR_NONBUILD,
                        help='GT layer for Non-building row: fused|zaha|ai')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger('three_rows_eval', os.path.join(args.output_dir, 'eval_3rows.log'))

    evaluate(args, logger)


if __name__ == '__main__':
    main()
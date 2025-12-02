#!/usr/bin/env python
"""
3D Point Cloud Semantic Segmentation Evaluation for LangSplat.

This script evaluates semantic segmentation quality directly on 3D Gaussian point clouds
by comparing with ground truth point cloud annotations.

Usage:
    python evaluate_3d_pointcloud.py \
        --checkpoint output/building1_15_dual_eval_1/chkpnt30000.pth \
        --ae_checkpoint autoencoder/ckpt/building1_15/best_ckpt.pth \
        --gt_pointcloud ZAHA_segments/gt_ply/b1_label.ply \
        --class_mapping eval/class_mapping.json \
        --class_colors eval/class_colors.json \
        --merge_mapping eval/window_merge.json \
        --output_dir eval/output/b1_3d
"""

import argparse
import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from plyfile import PlyData, PlyElement
from scipy.spatial import KDTree

import sys
sys.path.append("..")
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from autoencoder.model import Autoencoder
from openclip_encoder import OpenCLIPNetwork


# ============================================================================
# Logger
# ============================================================================

def get_logger(name, log_file=None, log_level=logging.INFO):
    logger = logging.getLogger(name)
    logger.handlers.clear()  # Clear existing handlers

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


# ============================================================================
# Load Checkpoint and Point Cloud
# ============================================================================

def load_gaussian_pointcloud_from_checkpoint(
    checkpoint_path: str,
    logger: logging.Logger
) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Load 3D Gaussian point cloud and language features from checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint file
        logger: Logger instance

    Returns:
        xyz: (N, 3) numpy array of point positions
        language_features: (N, 3) torch tensor of compressed language features
    """
    logger.info(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
    model_params, iteration = checkpoint_data

    # Check checkpoint format
    if len(model_params) == 13:
        # Feature training checkpoint
        (active_sh_degree, xyz, features_dc, features_rest, scaling,
         rotation, opacity, language_feature, max_radii2D, xyz_gradient_accum,
         denom, opt_dict, spatial_lr_scale) = model_params
    elif len(model_params) == 12:
        # RGB training checkpoint (no language features)
        logger.error("Checkpoint does not contain language features!")
        raise ValueError("This checkpoint was saved without language features. "
                        "Use a checkpoint from feature training (--include_feature).")
    else:
        raise ValueError(f"Unexpected checkpoint format with {len(model_params)} parameters")

    # Extract data
    xyz_np = xyz.detach().cpu().numpy()  # (N, 3)
    language_features_tensor = language_feature.detach()  # (N, 3)

    logger.info(f"Loaded {xyz_np.shape[0]:,} Gaussian points")
    logger.info(f"Language feature shape: {language_features_tensor.shape}")

    return xyz_np, language_features_tensor


def load_gt_pointcloud(
    gt_path: str,
    logger: logging.Logger,
    label_merge_map: Optional[Dict[int, List[int]]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ground truth point cloud with semantic labels.

    Args:
        gt_path: Path to ground truth .ply file
        logger: Logger instance
        label_merge_map: Optional mapping to merge label IDs

    Returns:
        gt_xyz: (M, 3) numpy array of GT point positions
        gt_labels: (M,) numpy array of semantic labels
    """
    logger.info(f"Loading GT point cloud from: {gt_path}")

    plydata = PlyData.read(gt_path)
    vertex = plydata['vertex']

    gt_xyz = np.stack([
        np.asarray(vertex['x']),
        np.asarray(vertex['y']),
        np.asarray(vertex['z'])
    ], axis=1)  # (M, 3)

    gt_labels = np.asarray(vertex['scalar_Classification']).astype(np.int32)  # (M,)

    # Apply label merging
    if label_merge_map:
        logger.info(f"Applying label merge map: {label_merge_map}")
        for target_id, source_ids in label_merge_map.items():
            target_id = int(target_id)
            for src_id in source_ids:
                src_id = int(src_id)
                if src_id != target_id:
                    gt_labels[gt_labels == src_id] = target_id

    # Print label distribution
    unique_labels, counts = np.unique(gt_labels, return_counts=True)
    logger.info(f"Loaded {gt_xyz.shape[0]:,} GT points")
    logger.info(f"GT label distribution:")
    for label, count in zip(unique_labels, counts):
        percentage = count / len(gt_labels) * 100
        logger.info(f"  Label {label}: {count:,} points ({percentage:.1f}%)")

    return gt_xyz, gt_labels


# ============================================================================
# Feature Decoding and Semantic Query
# ============================================================================

def decode_features_batch(
    compressed_features: torch.Tensor,
    autoencoder: Autoencoder,
    device: torch.device,
    batch_size: int = 100000
) -> torch.Tensor:
    """
    Decode compressed 3-dim features to 512-dim CLIP features in batches.

    Args:
        compressed_features: (N, 3) tensor
        autoencoder: Trained autoencoder model
        device: torch device
        batch_size: Batch size for processing

    Returns:
        decoded_features: (N, 512) tensor
    """
    N = compressed_features.shape[0]
    decoded_list = []

    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = compressed_features[i:i+batch_size].to(device)
            decoded_batch = autoencoder.decode(batch)  # (batch, 512)
            decoded_list.append(decoded_batch.cpu())

    decoded_features = torch.cat(decoded_list, dim=0)  # (N, 512)
    return decoded_features


def query_semantic_labels(
    features_512: torch.Tensor,
    clip_model: OpenCLIPNetwork,
    queries: List[str],
    query_class_ids: np.ndarray,
    device: torch.device,
    batch_size: int = 10000,
    logger: logging.Logger = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Query semantic labels from CLIP features for each point.

    Args:
        features_512: (N, 512) decoded features
        clip_model: OpenCLIP model
        queries: List of text queries
        query_class_ids: (n_queries,) array mapping query index to class ID
        device: torch device
        batch_size: Batch size for CLIP inference
        logger: Logger instance

    Returns:
        pred_labels: (N,) array of predicted class IDs
        confidence: (N,) array of prediction confidence scores
    """
    N = features_512.shape[0]
    n_queries = len(queries)

    # Set CLIP queries
    clip_model.set_positives(queries)

    pred_labels = np.zeros(N, dtype=np.int32)
    confidence = np.zeros(N, dtype=np.float32)

    if logger:
        logger.info(f"Querying semantic labels for {N:,} points with {n_queries} queries...")

    with torch.no_grad():
        for i in tqdm(range(0, N, batch_size), desc="CLIP inference"):
            batch = features_512[i:i+batch_size].to(device)  # (batch, 512)

            # Reshape for CLIP model: expects (1, H, W, 512)
            # We treat batch as a 1D image: (1, 1, batch, 512)
            batch_reshaped = batch.unsqueeze(0).unsqueeze(0)  # (1, 1, batch, 512)

            # Get relevance scores: returns (1, n_queries, 1, batch)
            relevance = clip_model.get_max_across(batch_reshaped)  # (1, n_queries, 1, batch)
            relevance = relevance.squeeze(0).squeeze(1)  # (n_queries, batch)

            # Get best query for each point
            max_relevance, max_idx = torch.max(relevance, dim=0)  # (batch,)

            # Map query index to class ID
            batch_labels = query_class_ids[max_idx.cpu().numpy()]
            batch_confidence = max_relevance.cpu().numpy()

            pred_labels[i:i+batch_size] = batch_labels
            confidence[i:i+batch_size] = batch_confidence

    return pred_labels, confidence


# ============================================================================
# Spatial Alignment via Nearest Neighbor
# ============================================================================

def filter_pointcloud_by_bbox(
    xyz: np.ndarray,
    labels: np.ndarray,
    confidence: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    margin: float = 1.0,
    logger: logging.Logger = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter point cloud to keep only points within bounding box (with margin).

    Args:
        xyz: (N, 3) point positions
        labels: (N,) labels
        confidence: (N,) confidence scores
        bbox_min: (3,) minimum corner of bounding box
        bbox_max: (3,) maximum corner of bounding box
        margin: Additional margin around bbox (in same units as coordinates)
        logger: Logger instance

    Returns:
        filtered_xyz: (K, 3) filtered points
        filtered_labels: (K,) filtered labels
        filtered_confidence: (K,) filtered confidence
        valid_indices: (K,) indices of kept points
    """
    # Expand bbox with margin
    bbox_min_expanded = bbox_min - margin
    bbox_max_expanded = bbox_max + margin

    # Find points inside expanded bbox
    valid_mask = np.all((xyz >= bbox_min_expanded) & (xyz <= bbox_max_expanded), axis=1)
    valid_indices = np.where(valid_mask)[0]

    if logger:
        original_count = len(xyz)
        filtered_count = len(valid_indices)
        percentage = filtered_count / original_count * 100
        logger.info(f"Filtered {original_count:,} -> {filtered_count:,} points ({percentage:.1f}%)")
        logger.info(f"  BBox min: [{bbox_min[0]:.2f}, {bbox_min[1]:.2f}, {bbox_min[2]:.2f}]")
        logger.info(f"  BBox max: [{bbox_max[0]:.2f}, {bbox_max[1]:.2f}, {bbox_max[2]:.2f}]")
        logger.info(f"  Margin: {margin:.2f}")

    return xyz[valid_mask], labels[valid_mask], confidence[valid_mask], valid_indices


def align_predictions_to_gt(
    pred_xyz: np.ndarray,
    pred_labels: np.ndarray,
    pred_confidence: np.ndarray,
    gt_xyz: np.ndarray,
    gt_labels: np.ndarray,
    max_distance: float = 0.5,
    logger: logging.Logger = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Align predicted labels to GT points via nearest neighbor search.

    Args:
        pred_xyz: (N, 3) predicted point positions
        pred_labels: (N,) predicted labels
        pred_confidence: (N,) prediction confidence
        gt_xyz: (M, 3) GT point positions
        gt_labels: (M,) GT labels
        max_distance: Maximum distance for valid matches (in same units as coordinates)
        logger: Logger instance

    Returns:
        aligned_pred_labels: (K,) predicted labels for matched GT points
        aligned_gt_labels: (K,) GT labels for matched points
        matched_indices: (K,) indices of matched GT points
        distances: (K,) distances to nearest predictions
    """
    if logger:
        logger.info("Building KDTree for nearest neighbor search...")

    # Build KDTree for predictions
    tree = KDTree(pred_xyz)

    if logger:
        logger.info(f"Querying nearest neighbors for {len(gt_xyz):,} GT points...")

    # Find nearest prediction for each GT point
    distances, indices = tree.query(gt_xyz)  # (M,), (M,)

    # Filter by distance threshold
    valid_mask = distances < max_distance
    matched_indices = np.where(valid_mask)[0]

    aligned_pred_labels = pred_labels[indices[valid_mask]]
    aligned_gt_labels = gt_labels[valid_mask]
    matched_distances = distances[valid_mask]

    if logger:
        coverage = len(matched_indices) / len(gt_labels) * 100
        avg_distance = matched_distances.mean()
        logger.info(f"Matched {len(matched_indices):,} / {len(gt_labels):,} GT points ({coverage:.1f}%)")
        logger.info(f"Average match distance: {avg_distance:.4f}")
        logger.info(f"Max distance threshold: {max_distance}")

    return aligned_pred_labels, aligned_gt_labels, matched_indices, matched_distances


# ============================================================================
# Evaluation Metrics
# ============================================================================

def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union > 0 else 0.0


def evaluate_segmentation(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    class_mapping: Dict[int, Union[str, List[str]]],
    logger: logging.Logger
) -> Dict:
    """
    Compute evaluation metrics for semantic segmentation.

    Args:
        pred_labels: (N,) predicted labels
        gt_labels: (N,) ground truth labels
        class_mapping: Mapping from class_id to class_name
        logger: Logger instance

    Returns:
        Dictionary with evaluation metrics
    """
    # Get unique GT classes
    gt_classes = np.unique(gt_labels).tolist()

    # Per-class metrics
    class_iou = {}
    class_precision = {}
    class_recall = {}
    class_f1 = {}
    class_pixel_counts = {}

    for cls_id in gt_classes:
        cls_entry = class_mapping.get(cls_id, f"class_{cls_id}")
        cls_name = cls_entry[0] if isinstance(cls_entry, list) else cls_entry

        gt_mask = (gt_labels == cls_id)
        pred_mask = (pred_labels == cls_id)

        # IoU
        iou = compute_iou(pred_mask, gt_mask)
        class_iou[cls_name] = iou

        # Precision, Recall, F1
        tp = np.logical_and(pred_mask, gt_mask).sum()
        fp = np.logical_and(pred_mask, ~gt_mask).sum()
        fn = np.logical_and(~pred_mask, gt_mask).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        class_precision[cls_name] = precision
        class_recall[cls_name] = recall
        class_f1[cls_name] = f1
        class_pixel_counts[cls_name] = int(gt_mask.sum())

    # Overall metrics
    mean_iou = np.mean(list(class_iou.values()))

    # Weighted IoU (by point count)
    total_points = sum(class_pixel_counts.values())
    weighted_iou = sum(
        class_iou[cls] * class_pixel_counts[cls] / total_points
        for cls in class_iou.keys()
    )

    # Pixel accuracy
    pixel_acc = (pred_labels == gt_labels).sum() / len(gt_labels)

    # Print results
    logger.info(f"\n{'='*80}")
    logger.info("Per-Class Results:")
    logger.info(f"{'Class':<30} {'IoU':<10} {'Precision':<12} {'Recall':<10} {'F1':<10} {'Count':<12}")
    logger.info("-" * 80)

    for cls_name in sorted(class_iou.keys()):
        logger.info(
            f"{cls_name:<30} {class_iou[cls_name]:<10.4f} "
            f"{class_precision[cls_name]:<12.4f} {class_recall[cls_name]:<10.4f} "
            f"{class_f1[cls_name]:<10.4f} {class_pixel_counts[cls_name]:<12,}"
        )

    logger.info(f"\n{'='*80}")
    logger.info("Overall Metrics:")
    logger.info(f"  Mean IoU:       {mean_iou:.4f}")
    logger.info(f"  Weighted IoU:   {weighted_iou:.4f}")
    logger.info(f"  Pixel Accuracy: {pixel_acc:.4f}")
    logger.info(f"{'='*80}\n")

    return {
        'class_iou': class_iou,
        'class_precision': class_precision,
        'class_recall': class_recall,
        'class_f1': class_f1,
        'class_point_counts': class_pixel_counts,
        'mean_iou': float(mean_iou),
        'weighted_iou': float(weighted_iou),
        'pixel_accuracy': float(pixel_acc),
        'total_points': int(len(gt_labels))
    }


# ============================================================================
# Save Colored Point Cloud
# ============================================================================

def save_colored_pointcloud(
    xyz: np.ndarray,
    labels: np.ndarray,
    class_colors: Dict[int, List[int]],
    output_path: str,
    logger: logging.Logger
):
    """
    Save point cloud with semantic colors in PLY format for CloudCompare.

    Args:
        xyz: (N, 3) point positions
        labels: (N,) semantic labels
        class_colors: Mapping from class_id to RGB color [0-255]
        output_path: Output .ply path
        logger: Logger instance
    """
    logger.info(f"Saving colored point cloud to: {output_path}")

    # Assign colors based on labels
    colors = np.zeros((len(xyz), 3), dtype=np.uint8)

    for cls_id, color in class_colors.items():
        cls_id = int(cls_id)
        mask = (labels == cls_id)
        colors[mask] = color

    # Handle unlabeled points (assign black or gray)
    unlabeled_mask = ~np.isin(labels, list(class_colors.keys()))
    if unlabeled_mask.any():
        colors[unlabeled_mask] = [128, 128, 128]  # Gray for unlabeled

    # Create structured array for PLY
    vertex_data = np.zeros(
        len(xyz),
        dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ]
    )

    vertex_data['x'] = xyz[:, 0]
    vertex_data['y'] = xyz[:, 1]
    vertex_data['z'] = xyz[:, 2]
    vertex_data['red'] = colors[:, 0]
    vertex_data['green'] = colors[:, 1]
    vertex_data['blue'] = colors[:, 2]

    # Save PLY
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    PlyData([vertex_element], text=False).write(output_path)

    logger.info(f"Saved {len(xyz):,} points with semantic colors")


# ============================================================================
# Main Evaluation Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="3D Point Cloud Semantic Segmentation Evaluation for LangSplat"
    )
    parser.add_argument('--checkpoint', type=str,
                       default='/LangSplat/output/building1/chkpnt30000.pth',
                       help='Path to LangSplat checkpoint (.pth)')
    parser.add_argument('--ae_checkpoint', type=str,
                       default='/LangSplat/autoencoder/ckpt/building1/best_ckpt.pth',
                       help='Path to autoencoder checkpoint')
    parser.add_argument('--gt_pointcloud', type=str,
                       default='/LangSplat/zaha/gt_ply/b1_label.ply',
                       help='Path to ground truth point cloud with labels')
    parser.add_argument('--class_mapping', type=str,
                       default='class_mapping.json',
                       help='JSON file mapping class_id to class_name/prompts')
    parser.add_argument('--class_colors', type=str,
                       default='class_colors.json',
                       help='JSON file mapping class_id to RGB color')
    parser.add_argument('--merge_mapping', type=str,
                       default='window_merge.json',
                       help='JSON file for merging label IDs')
    parser.add_argument('--output_dir', type=str,
                       default='/output/b1_3d',
                       help='Output directory for results')
    parser.add_argument('--max_distance', type=float, default=0.5,
                       help='Maximum distance for GT-prediction matching (default: 0.5)')
    parser.add_argument('--encoder_dims', nargs='+', type=int,
                       default=[256, 128, 64, 32, 3],
                       help='Autoencoder encoder dimensions')
    parser.add_argument('--decoder_dims', nargs='+', type=int,
                       default=[16, 32, 64, 128, 256, 256, 512],
                       help='Autoencoder decoder dimensions')
    parser.add_argument('--batch_size', type=int, default=100000,
                       help='Batch size for feature decoding')
    parser.add_argument('--clip_batch_size', type=int, default=10000,
                       help='Batch size for CLIP inference')
    parser.add_argument('--bbox_margin', type=float, default=1.0,
                       help='Margin around GT bounding box for filtering predictions (default: 1.0)')
    parser.add_argument('--skip_filtering', action='store_true',
                       help='Skip spatial filtering step (process all prediction points)')

    args = parser.parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, 'evaluation_3d.log')
    logger = get_logger('eval_3d', log_file=log_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"\n{'='*80}")
    logger.info("3D Point Cloud Semantic Segmentation Evaluation")
    logger.info(f"{'='*80}")
    logger.info(f"Device: {device}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"GT Point Cloud: {args.gt_pointcloud}")
    logger.info(f"Output: {args.output_dir}")

    # Load configurations
    with open(args.class_mapping, 'r') as f:
        class_mapping_raw = json.load(f)
        class_mapping = {int(k): v for k, v in class_mapping_raw.items()}

    with open(args.class_colors, 'r') as f:
        class_colors_raw = json.load(f)
        class_colors = {int(k): v for k, v in class_colors_raw.items()}

    merge_mapping = None
    if args.merge_mapping and os.path.exists(args.merge_mapping):
        with open(args.merge_mapping, 'r') as f:
            merge_raw = json.load(f)
            merge_mapping = {int(k): [int(v) for v in vals] for k, vals in merge_raw.items()}

    # Step 1: Load Gaussian point cloud and language features
    logger.info(f"\n{'='*80}")
    logger.info("Step 1: Loading Gaussian Point Cloud")
    logger.info(f"{'='*80}")
    pred_xyz, compressed_features = load_gaussian_pointcloud_from_checkpoint(
        args.checkpoint, logger
    )

    # Step 2: Load GT point cloud
    logger.info(f"\n{'='*80}")
    logger.info("Step 2: Loading Ground Truth Point Cloud")
    logger.info(f"{'='*80}")
    gt_xyz, gt_labels = load_gt_pointcloud(args.gt_pointcloud, logger, merge_mapping)

    # Compute GT bounding box
    gt_bbox_min = gt_xyz.min(axis=0)
    gt_bbox_max = gt_xyz.max(axis=0)
    logger.info(f"GT bounding box:")
    logger.info(f"  Min: [{gt_bbox_min[0]:.2f}, {gt_bbox_min[1]:.2f}, {gt_bbox_min[2]:.2f}]")
    logger.info(f"  Max: [{gt_bbox_max[0]:.2f}, {gt_bbox_max[1]:.2f}, {gt_bbox_max[2]:.2f}]")
    logger.info(f"  Size: [{gt_bbox_max[0]-gt_bbox_min[0]:.2f}, {gt_bbox_max[1]-gt_bbox_min[1]:.2f}, {gt_bbox_max[2]-gt_bbox_min[2]:.2f}]")

    # Step 2.5: Filter predictions to GT region (optional)
    if not args.skip_filtering:
        logger.info(f"\n{'='*80}")
        logger.info("Step 2.5: Filtering Predictions to GT Region")
        logger.info(f"{'='*80}")

        # Create dummy labels and confidence for filtering
        dummy_labels = np.zeros(len(pred_xyz), dtype=np.int32)
        dummy_confidence = np.zeros(len(pred_xyz), dtype=np.float32)

        pred_xyz_filtered, _, _, filtered_indices = filter_pointcloud_by_bbox(
            pred_xyz, dummy_labels, dummy_confidence,
            gt_bbox_min, gt_bbox_max, args.bbox_margin, logger
        )

        # Filter language features correspondingly
        compressed_features = compressed_features[filtered_indices]

        # Update pred_xyz for subsequent steps
        pred_xyz = pred_xyz_filtered

        logger.info(f"Reduced prediction points: {len(compressed_features):,}")
    else:
        logger.info("\nSkipping spatial filtering (processing all points)")

    # Step 3: Load models
    logger.info(f"\n{'='*80}")
    logger.info("Step 3: Loading Models")
    logger.info(f"{'='*80}")

    logger.info("Loading CLIP model...")
    clip_model = OpenCLIPNetwork(device)

    logger.info(f"Loading autoencoder from: {args.ae_checkpoint}")
    ae_checkpoint = torch.load(args.ae_checkpoint, map_location=device)
    autoencoder = Autoencoder(args.encoder_dims, args.decoder_dims).to(device)
    autoencoder.load_state_dict(ae_checkpoint)
    autoencoder.eval()

    # Step 4: Decode features
    logger.info(f"\n{'='*80}")
    logger.info("Step 4: Decoding Language Features (3-dim -> 512-dim)")
    logger.info(f"{'='*80}")
    features_512 = decode_features_batch(
        compressed_features, autoencoder, device, args.batch_size
    )
    logger.info(f"Decoded features shape: {features_512.shape}")

    # Step 5: Prepare queries
    logger.info(f"\n{'='*80}")
    logger.info("Step 5: Preparing Semantic Queries")
    logger.info(f"{'='*80}")

    # Get unique GT classes
    unique_gt_labels = np.unique(gt_labels).tolist()
    logger.info(f"Classes in GT: {unique_gt_labels}")

    # Build queries only for classes in GT
    queries = []
    query_class_ids = []
    for cls_id in unique_gt_labels:
        if cls_id not in class_mapping:
            logger.warning(f"Class ID {cls_id} not in class_mapping, skipping")
            continue

        phrases = class_mapping[cls_id]
        if isinstance(phrases, list):
            prompt_list = phrases
        else:
            prompt_list = [phrases]

        for phrase in prompt_list:
            queries.append(phrase)
            query_class_ids.append(cls_id)

    query_class_ids = np.array(query_class_ids, dtype=np.int32)
    logger.info(f"Total queries: {len(queries)}")
    logger.info(f"Queries: {queries[:5]}... (showing first 5)")

    # Step 6: Query semantic labels
    logger.info(f"\n{'='*80}")
    logger.info("Step 6: Querying Semantic Labels")
    logger.info(f"{'='*80}")
    pred_labels, pred_confidence = query_semantic_labels(
        features_512, clip_model, queries, query_class_ids,
        device, args.clip_batch_size, logger
    )

    # Apply label merging to predictions
    if merge_mapping:
        logger.info("Applying label merge to predictions...")
        for target_id, source_ids in merge_mapping.items():
            target_id = int(target_id)
            for src_id in source_ids:
                src_id = int(src_id)
                if src_id != target_id:
                    pred_labels[pred_labels == src_id] = target_id

    # Step 7: Align predictions to GT
    logger.info(f"\n{'='*80}")
    logger.info("Step 7: Aligning Predictions to Ground Truth")
    logger.info(f"{'='*80}")
    aligned_pred, aligned_gt, matched_indices, distances = align_predictions_to_gt(
        pred_xyz, pred_labels, pred_confidence, gt_xyz, gt_labels,
        args.max_distance, logger
    )

    # Step 8: Evaluate
    logger.info(f"\n{'='*80}")
    logger.info("Step 8: Computing Evaluation Metrics")
    logger.info(f"{'='*80}")
    metrics = evaluate_segmentation(aligned_pred, aligned_gt, class_mapping, logger)

    # Add alignment info
    metrics['alignment'] = {
        'matched_points': int(len(matched_indices)),
        'total_gt_points': int(len(gt_labels)),
        'coverage_percentage': float(len(matched_indices) / len(gt_labels) * 100),
        'avg_match_distance': float(distances.mean()),
        'max_distance_threshold': float(args.max_distance)
    }

    # Step 9: Save results
    logger.info(f"\n{'='*80}")
    logger.info("Step 9: Saving Results")
    logger.info(f"{'='*80}")

    # Save metrics as JSON
    results_path = os.path.join(args.output_dir, 'results_3d.json')
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to: {results_path}")

    # Save colored point cloud (full prediction)
    pred_colored_path = os.path.join(args.output_dir, 'predicted_semantic.ply')
    save_colored_pointcloud(pred_xyz, pred_labels, class_colors, pred_colored_path, logger)

    # Save GT colored point cloud
    gt_colored_path = os.path.join(args.output_dir, 'gt_semantic_colored.ply')
    save_colored_pointcloud(gt_xyz, gt_labels, class_colors, gt_colored_path, logger)

    logger.info(f"\n{'='*80}")
    logger.info("Evaluation Complete!")
    logger.info(f"{'='*80}")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"  - Metrics: results_3d.json")
    logger.info(f"  - Predicted point cloud: predicted_semantic.ply")
    logger.info(f"  - GT point cloud: gt_semantic_colored.ply")
    logger.info(f"\nOpen the .ply files in CloudCompare to visualize semantic segmentation!")


if __name__ == '__main__':
    main()

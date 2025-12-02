#!/usr/bin/env python
"""
Generate 3D Error Map Point Cloud from Evaluation Results.

This script reads ground truth and predicted semantic point clouds,
compares them, and generates an error map visualization where:
- Green points: Correct predictions
- Red points: Incorrect predictions
- Optional: Color-coded by confusion type

Usage:
    python generate_error_map_pointcloud.py \
        --gt_pointcloud eval/output/b1_3d_dual/gt_semantic_colored.ply \
        --pred_pointcloud eval/output/b1_3d_dual/predicted_semantic.ply \
        --output_dir eval/output/b1_3d_dual \
        --max_distance 0.5
"""

import argparse
import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm
from plyfile import PlyData, PlyElement
from scipy.spatial import KDTree


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

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


# ============================================================================
# Load Point Clouds
# ============================================================================

def load_semantic_pointcloud(
    ply_path: str,
    logger: logging.Logger
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load semantic point cloud from PLY file.

    Args:
        ply_path: Path to .ply file with semantic colors
        logger: Logger instance

    Returns:
        xyz: (N, 3) point positions
        colors: (N, 3) RGB colors (0-255)
        labels: (N,) reconstructed semantic labels from colors
    """
    logger.info(f"Loading point cloud from: {ply_path}")

    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']

    # Extract positions
    xyz = np.stack([
        np.asarray(vertex['x']),
        np.asarray(vertex['y']),
        np.asarray(vertex['z'])
    ], axis=1)  # (N, 3)

    # Extract colors
    colors = np.stack([
        np.asarray(vertex['red']),
        np.asarray(vertex['green']),
        np.asarray(vertex['blue'])
    ], axis=1).astype(np.uint8)  # (N, 3)

    # Try to extract labels if available
    if 'scalar_Classification' in vertex.data.dtype.names:
        labels = np.asarray(vertex['scalar_Classification']).astype(np.int32)
    else:
        # Reconstruct labels from colors (inverse color mapping)
        labels = None

    logger.info(f"Loaded {len(xyz):,} points")
    logger.info(f"Position range: X=[{xyz[:,0].min():.2f}, {xyz[:,0].max():.2f}], "
                f"Y=[{xyz[:,1].min():.2f}, {xyz[:,1].max():.2f}], "
                f"Z=[{xyz[:,2].min():.2f}, {xyz[:,2].max():.2f}]")

    return xyz, colors, labels


def reconstruct_labels_from_colors(
    colors: np.ndarray,
    class_colors: Dict[int, List[int]],
    logger: logging.Logger
) -> np.ndarray:
    """
    Reconstruct semantic labels from RGB colors using class_colors mapping.

    Args:
        colors: (N, 3) RGB colors (0-255)
        class_colors: Dict mapping class_id to [R, G, B]
        logger: Logger instance

    Returns:
        labels: (N,) reconstructed class labels
    """
    logger.info("Reconstructing labels from colors...")

    N = len(colors)
    labels = np.full(N, -1, dtype=np.int32)  # -1 for unknown

    # Build color to label mapping
    color_to_label = {}
    for cls_id, rgb in class_colors.items():
        color_tuple = tuple(rgb)
        color_to_label[color_tuple] = int(cls_id)

    # Map each point color to label
    for i in tqdm(range(N), desc="Mapping colors to labels"):
        color_tuple = tuple(colors[i])
        if color_tuple in color_to_label:
            labels[i] = color_to_label[color_tuple]

    # Statistics
    unique_labels, counts = np.unique(labels, return_counts=True)
    logger.info(f"Reconstructed label distribution:")
    for label, count in zip(unique_labels, counts):
        percentage = count / N * 100
        logger.info(f"  Label {label}: {count:,} points ({percentage:.1f}%)")

    unknown_count = (labels == -1).sum()
    if unknown_count > 0:
        logger.warning(f"  {unknown_count:,} points with unknown labels ({unknown_count/N*100:.1f}%)")

    return labels


# ============================================================================
# Point Cloud Alignment
# ============================================================================

def align_pointclouds(
    pred_xyz: np.ndarray,
    pred_labels: np.ndarray,
    gt_xyz: np.ndarray,
    gt_labels: np.ndarray,
    max_distance: float = 0.5,
    logger: logging.Logger = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Align predicted point cloud to ground truth via nearest neighbor.

    Args:
        pred_xyz: (N, 3) predicted point positions
        pred_labels: (N,) predicted labels
        gt_xyz: (M, 3) GT point positions
        gt_labels: (M,) GT labels
        max_distance: Maximum distance for valid matches
        logger: Logger instance

    Returns:
        matched_pred_xyz: (K, 3) matched prediction positions
        matched_pred_labels: (K,) matched prediction labels
        matched_gt_xyz: (K, 3) matched GT positions
        matched_gt_labels: (K,) matched GT labels
        distances: (K,) distances between matches
    """
    if logger:
        logger.info("Aligning point clouds via nearest neighbor...")

    # Build KDTree for predictions
    tree = KDTree(pred_xyz)

    if logger:
        logger.info(f"Querying nearest neighbors for {len(gt_xyz):,} GT points...")

    # Find nearest prediction for each GT point
    distances, indices = tree.query(gt_xyz)

    # Filter by distance threshold
    valid_mask = distances < max_distance
    matched_indices = np.where(valid_mask)[0]

    matched_pred_xyz = pred_xyz[indices[valid_mask]]
    matched_pred_labels = pred_labels[indices[valid_mask]]
    matched_gt_xyz = gt_xyz[valid_mask]
    matched_gt_labels = gt_labels[valid_mask]
    matched_distances = distances[valid_mask]

    if logger:
        coverage = len(matched_indices) / len(gt_labels) * 100
        avg_distance = matched_distances.mean()
        logger.info(f"Matched {len(matched_indices):,} / {len(gt_labels):,} GT points ({coverage:.1f}%)")
        logger.info(f"Average match distance: {avg_distance:.4f}")
        logger.info(f"Max distance threshold: {max_distance}")

    return matched_pred_xyz, matched_pred_labels, matched_gt_xyz, matched_gt_labels, matched_distances


# ============================================================================
# Error Map Generation
# ============================================================================

def generate_error_map(
    matched_pred_labels: np.ndarray,
    matched_gt_labels: np.ndarray,
    error_mode: str = 'simple'
) -> Tuple[np.ndarray, Dict]:
    """
    Generate error labels for visualization.

    Args:
        matched_pred_labels: (N,) predicted labels
        matched_gt_labels: (N,) ground truth labels
        error_mode: 'simple' (correct/incorrect) or 'detailed' (per-class confusion)

    Returns:
        error_labels: (N,) error type labels
        error_stats: Dictionary with error statistics
    """
    N = len(matched_pred_labels)
    error_labels = np.zeros(N, dtype=np.int32)

    if error_mode == 'simple':
        # 0: Correct (Green)
        # 1: Incorrect (Red)
        correct_mask = (matched_pred_labels == matched_gt_labels)
        error_labels[correct_mask] = 0
        error_labels[~correct_mask] = 1

        correct_count = correct_mask.sum()
        incorrect_count = (~correct_mask).sum()

        error_stats = {
            'total_points': N,
            'correct': int(correct_count),
            'incorrect': int(incorrect_count),
            'accuracy': float(correct_count / N) if N > 0 else 0.0
        }

    elif error_mode == 'detailed':
        # Create confusion-based error types
        # 0: Correct
        # 1+: Different types of errors (e.g., window->wall, wall->window, etc.)
        unique_gt_labels = np.unique(matched_gt_labels)
        unique_pred_labels = np.unique(matched_pred_labels)

        error_type_id = 1
        confusion_map = {}  # (gt_label, pred_label) -> error_type_id

        correct_mask = (matched_pred_labels == matched_gt_labels)
        error_labels[correct_mask] = 0

        # Assign error types for confusions
        for gt_label in unique_gt_labels:
            for pred_label in unique_pred_labels:
                if gt_label == pred_label:
                    continue

                # Find points with this confusion
                confusion_mask = (matched_gt_labels == gt_label) & (matched_pred_labels == pred_label)
                if confusion_mask.any():
                    error_labels[confusion_mask] = error_type_id
                    confusion_map[(int(gt_label), int(pred_label))] = error_type_id
                    error_type_id += 1

        error_stats = {
            'total_points': N,
            'correct': int(correct_mask.sum()),
            'incorrect': int((~correct_mask).sum()),
            'accuracy': float(correct_mask.sum() / N) if N > 0 else 0.0,
            'confusion_types': confusion_map,
            'n_error_types': error_type_id - 1
        }

    else:
        raise ValueError(f"Unknown error_mode: {error_mode}")

    return error_labels, error_stats


def assign_error_colors(
    error_labels: np.ndarray,
    error_stats: Dict,
    error_mode: str = 'simple'
) -> np.ndarray:
    """
    Assign RGB colors to error labels.

    Args:
        error_labels: (N,) error type labels
        error_stats: Error statistics from generate_error_map
        error_mode: 'simple' or 'detailed'

    Returns:
        colors: (N, 3) RGB colors (0-255)
    """
    N = len(error_labels)
    colors = np.zeros((N, 3), dtype=np.uint8)

    if error_mode == 'simple':
        # 0: Correct -> Green
        # 1: Incorrect -> Red
        colors[error_labels == 0] = [0, 255, 0]  # Green
        colors[error_labels == 1] = [255, 0, 0]  # Red

    elif error_mode == 'detailed':
        # 0: Correct -> Green
        colors[error_labels == 0] = [0, 255, 0]

        # Different error types -> Different colors
        n_error_types = error_stats['n_error_types']
        if n_error_types > 0:
            # Generate distinct colors for error types
            import matplotlib.pyplot as plt
            cmap = plt.cm.get_cmap('tab20', n_error_types)

            for error_type in range(1, n_error_types + 1):
                rgb = cmap(error_type - 1)[:3]
                colors[error_labels == error_type] = (np.array(rgb) * 255).astype(np.uint8)

    return colors


# ============================================================================
# Save Point Cloud
# ============================================================================

def save_error_pointcloud(
    xyz: np.ndarray,
    colors: np.ndarray,
    error_labels: np.ndarray,
    output_path: str,
    logger: logging.Logger
):
    """
    Save error map point cloud in PLY format.

    Args:
        xyz: (N, 3) point positions
        colors: (N, 3) RGB colors (0-255)
        error_labels: (N,) error type labels
        output_path: Output .ply path
        logger: Logger instance
    """
    logger.info(f"Saving error map point cloud to: {output_path}")

    # Create structured array
    vertex_data = np.zeros(
        len(xyz),
        dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('error_type', 'i4')
        ]
    )

    vertex_data['x'] = xyz[:, 0]
    vertex_data['y'] = xyz[:, 1]
    vertex_data['z'] = xyz[:, 2]
    vertex_data['red'] = colors[:, 0]
    vertex_data['green'] = colors[:, 1]
    vertex_data['blue'] = colors[:, 2]
    vertex_data['error_type'] = error_labels

    # Save PLY
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    PlyData([vertex_element], text=False).write(output_path)

    logger.info(f"Saved {len(xyz):,} points with error labels")


# ============================================================================
# Compute Error Statistics
# ============================================================================

def compute_class_wise_accuracy(
    matched_pred_labels: np.ndarray,
    matched_gt_labels: np.ndarray,
    class_mapping: Optional[Dict[int, str]] = None,
    logger: logging.Logger = None
) -> Dict:
    """
    Compute per-class accuracy statistics.

    Args:
        matched_pred_labels: (N,) predicted labels
        matched_gt_labels: (N,) ground truth labels
        class_mapping: Optional mapping from class_id to class_name
        logger: Logger instance

    Returns:
        Dictionary with per-class statistics
    """
    unique_gt_labels = np.unique(matched_gt_labels)

    class_stats = {}

    for cls_id in unique_gt_labels:
        if class_mapping:
            cls_entry = class_mapping.get(int(cls_id), f"class_{cls_id}")
            cls_name = cls_entry[0] if isinstance(cls_entry, list) else cls_entry
        else:
            cls_name = f"class_{cls_id}"

        # Points belonging to this class in GT
        gt_mask = (matched_gt_labels == cls_id)
        n_total = gt_mask.sum()

        # Correctly predicted points
        correct_mask = (matched_pred_labels == cls_id) & gt_mask
        n_correct = correct_mask.sum()

        # False positives (predicted as this class but GT is different)
        fp_mask = (matched_pred_labels == cls_id) & (~gt_mask)
        n_fp = fp_mask.sum()

        # Compute metrics
        accuracy = n_correct / n_total if n_total > 0 else 0.0
        precision = n_correct / (n_correct + n_fp) if (n_correct + n_fp) > 0 else 0.0
        recall = accuracy  # Same as accuracy for single class

        class_stats[cls_name] = {
            'class_id': int(cls_id),
            'total_points': int(n_total),
            'correct': int(n_correct),
            'false_positive': int(n_fp),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall)
        }

    return class_stats


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D error map point cloud from evaluation results"
    )
    parser.add_argument('--gt_pointcloud', type=str,
                       default='/home/qilin/Documents/LangSplat/eval/output/b1_3d_dual/gt_semantic_colored.ply',
                       help='Path to ground truth semantic point cloud (.ply)')
    parser.add_argument('--pred_pointcloud', type=str,
                       default='/home/qilin/Documents/LangSplat/eval/output/b1_3d_dual/predicted_semantic.ply',
                       help='Path to predicted semantic point cloud (.ply)')
    parser.add_argument('--output_dir', type=str,
                       default='/home/qilin/Documents/LangSplat/eval/output/b1_3d_dual',
                       help='Output directory for error map')
    parser.add_argument('--max_distance', type=float, default=0.5,
                       help='Maximum distance for point matching (default: 0.5)')
    parser.add_argument('--error_mode', type=str, choices=['simple', 'detailed'],
                       default='simple',
                       help='Error visualization mode (default: simple)')
    parser.add_argument('--class_colors', type=str,
                       default='/home/qilin/Documents/LangSplat/eval/class_colors.json',
                       help='Path to class_colors.json for label reconstruction')
    parser.add_argument('--class_mapping', type=str,
                       default='/home/qilin/Documents/LangSplat/eval/class_mapping.json',
                       help='Path to class_mapping.json for statistics')

    args = parser.parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, 'error_map_generation.log')
    logger = get_logger('error_map', log_file=log_file)

    logger.info(f"\n{'='*80}")
    logger.info("3D Error Map Point Cloud Generation")
    logger.info(f"{'='*80}")
    logger.info(f"GT Point Cloud: {args.gt_pointcloud}")
    logger.info(f"Pred Point Cloud: {args.pred_pointcloud}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Max Distance: {args.max_distance}")
    logger.info(f"Error Mode: {args.error_mode}")

    # Load class colors for label reconstruction
    class_colors = None
    if args.class_colors and os.path.exists(args.class_colors):
        with open(args.class_colors, 'r') as f:
            colors_raw = json.load(f)
            class_colors = {int(k): v for k, v in colors_raw.items()}
        logger.info(f"Loaded class colors from: {args.class_colors}")

    # Load class mapping for statistics
    class_mapping = None
    if args.class_mapping and os.path.exists(args.class_mapping):
        with open(args.class_mapping, 'r') as f:
            mapping_raw = json.load(f)
            class_mapping = {int(k): v for k, v in mapping_raw.items()}
        logger.info(f"Loaded class mapping from: {args.class_mapping}")

    # Step 1: Load point clouds
    logger.info(f"\n{'='*80}")
    logger.info("Step 1: Loading Point Clouds")
    logger.info(f"{'='*80}")

    gt_xyz, gt_colors, gt_labels_direct = load_semantic_pointcloud(
        args.gt_pointcloud, logger
    )
    pred_xyz, pred_colors, pred_labels_direct = load_semantic_pointcloud(
        args.pred_pointcloud, logger
    )

    # Step 2: Reconstruct labels from colors if not available
    logger.info(f"\n{'='*80}")
    logger.info("Step 2: Reconstructing Labels")
    logger.info(f"{'='*80}")

    if gt_labels_direct is None:
        if class_colors is None:
            logger.error("Cannot reconstruct GT labels: class_colors.json not provided!")
            return
        gt_labels = reconstruct_labels_from_colors(gt_colors, class_colors, logger)
    else:
        gt_labels = gt_labels_direct
        logger.info("GT labels loaded directly from PLY file")

    if pred_labels_direct is None:
        if class_colors is None:
            logger.error("Cannot reconstruct prediction labels: class_colors.json not provided!")
            return
        pred_labels = reconstruct_labels_from_colors(pred_colors, class_colors, logger)
    else:
        pred_labels = pred_labels_direct
        logger.info("Prediction labels loaded directly from PLY file")

    # Step 3: Align point clouds
    logger.info(f"\n{'='*80}")
    logger.info("Step 3: Aligning Point Clouds")
    logger.info(f"{'='*80}")

    matched_pred_xyz, matched_pred_labels, matched_gt_xyz, matched_gt_labels, distances = \
        align_pointclouds(pred_xyz, pred_labels, gt_xyz, gt_labels, args.max_distance, logger)

    # Step 4: Generate error map
    logger.info(f"\n{'='*80}")
    logger.info("Step 4: Generating Error Map")
    logger.info(f"{'='*80}")

    error_labels, error_stats = generate_error_map(
        matched_pred_labels, matched_gt_labels, args.error_mode
    )

    logger.info(f"Error Statistics:")
    logger.info(f"  Total Points: {error_stats['total_points']:,}")
    logger.info(f"  Correct: {error_stats['correct']:,} ({error_stats['accuracy']*100:.2f}%)")
    logger.info(f"  Incorrect: {error_stats['incorrect']:,} ({(1-error_stats['accuracy'])*100:.2f}%)")

    if args.error_mode == 'detailed' and 'confusion_types' in error_stats:
        logger.info(f"\n  Confusion Types ({error_stats['n_error_types']} types):")
        for (gt_id, pred_id), error_type in error_stats['confusion_types'].items():
            count = (error_labels == error_type).sum()
            gt_name = class_mapping.get(gt_id, f"class_{gt_id}") if class_mapping else f"class_{gt_id}"
            pred_name = class_mapping.get(pred_id, f"class_{pred_id}") if class_mapping else f"class_{pred_id}"
            if isinstance(gt_name, list):
                gt_name = gt_name[0]
            if isinstance(pred_name, list):
                pred_name = pred_name[0]
            logger.info(f"    Type {error_type}: GT={gt_name} -> Pred={pred_name}: {count:,} points")

    # Step 5: Assign error colors
    logger.info(f"\n{'='*80}")
    logger.info("Step 5: Assigning Error Colors")
    logger.info(f"{'='*80}")

    error_colors = assign_error_colors(error_labels, error_stats, args.error_mode)

    # Step 6: Compute class-wise accuracy
    logger.info(f"\n{'='*80}")
    logger.info("Step 6: Computing Class-wise Statistics")
    logger.info(f"{'='*80}")

    class_stats = compute_class_wise_accuracy(
        matched_pred_labels, matched_gt_labels, class_mapping, logger
    )

    logger.info("\nPer-Class Accuracy:")
    logger.info(f"{'Class':<20} {'Total':<10} {'Correct':<10} {'Accuracy':<10} {'Precision':<10}")
    logger.info("-" * 70)
    for cls_name in sorted(class_stats.keys()):
        stats = class_stats[cls_name]
        logger.info(
            f"{cls_name:<20} {stats['total_points']:<10,} {stats['correct']:<10,} "
            f"{stats['accuracy']:<10.4f} {stats['precision']:<10.4f}"
        )

    # Step 7: Save error map point cloud
    logger.info(f"\n{'='*80}")
    logger.info("Step 7: Saving Error Map")
    logger.info(f"{'='*80}")

    output_path = os.path.join(args.output_dir, 'error_map.ply')
    save_error_pointcloud(matched_gt_xyz, error_colors, error_labels, output_path, logger)

    # Step 8: Save statistics as JSON
    logger.info(f"\n{'='*80}")
    logger.info("Step 8: Saving Statistics")
    logger.info(f"{'='*80}")

    results = {
        'error_statistics': error_stats,
        'class_statistics': class_stats,
        'alignment': {
            'total_gt_points': int(len(gt_labels)),
            'total_pred_points': int(len(pred_labels)),
            'matched_points': int(len(matched_gt_labels)),
            'coverage': float(len(matched_gt_labels) / len(gt_labels)),
            'avg_distance': float(distances.mean()),
            'max_distance_threshold': float(args.max_distance)
        }
    }

    results_path = os.path.join(args.output_dir, 'error_map_stats.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved statistics to: {results_path}")

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("Error Map Generation Complete!")
    logger.info(f"{'='*80}")
    logger.info(f"Output files:")
    logger.info(f"  - Error map PLY: {output_path}")
    logger.info(f"  - Statistics JSON: {results_path}")
    logger.info(f"  - Log file: {log_file}")
    logger.info(f"\nVisualization:")
    logger.info(f"  Green points: Correct predictions")
    logger.info(f"  Red points: Incorrect predictions" if args.error_mode == 'simple'
                else f"  Other colors: Different confusion types")
    logger.info(f"\nOpen {output_path} in CloudCompare to visualize the error map!")


if __name__ == '__main__':
    main()

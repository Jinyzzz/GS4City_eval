#!/usr/bin/env python
"""
Evaluate LangSplat semantic segmentation using point cloud-derived ground truth.

This script evaluates semantic segmentation quality on regions where we have
ground truth from annotated point clouds. Supports partial coverage evaluation.

Usage:
    python evaluate_with_pointcloud_gt.py \
        --rendered_features output/scene_3/test/ours_None/renders_npy \
        --gt_semantic_dir /path/to/projected_gt_masks \
        --ae_checkpoint autoencoder/ckpt/scene/best_ckpt.pth \
        --class_mapping class_mapping.json \
        --output_dir eval_results \
        --mask_thresh 0.5

Ground Truth Format:
    - Each GT file: {image_name}.npy, shape=(H, W), dtype=int32
    - Background pixels: -1
    - Valid class labels: 0, 1, 2, ...
    - Coverage mask automatically computed as (semantic_map >= 0)

Class Mapping Format:
    {
        "0": "wall",
        "1": "window",
        "2": "door",
        "3": "roof"
    }
"""

import argparse
import json
import os
import glob
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict

import numpy as np
import torch
import cv2
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
import colormaps
from autoencoder.model import Autoencoder
from openclip_encoder import OpenCLIPNetwork


# ============================================================================
# Logger
# ============================================================================

def get_logger(name, log_file=None, log_level=logging.INFO):
    logger = logging.getLogger(name)
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
# Ground Truth Loading
# ============================================================================

def load_ground_truth(
    gt_dir: str,
    image_names: List[str],
    class_mapping: Dict[int, Union[str, List[str]]],
    logger: logging.Logger,
    label_merge_map: Optional[Dict[int, List[int]]] = None
) -> Dict[str, Dict]:
    """
    Load ground truth semantic maps from point cloud projection.

    Args:
        gt_dir: Directory with {image_name}.npy files
        image_names: List of image names to load (without extension)
        class_mapping: Mapping from class_id to class_name
        logger: Logger instance
        label_merge_map: Optional mapping from target class IDs to lists of source IDs to merge

    Returns:
        Dictionary mapping image_name to:
        {
            'semantic_map': (H, W) array with class labels,
            'coverage_mask': (H, W) boolean mask of valid pixels,
            'classes': list of class IDs present in this image
        }
    """
    logger.info(f"\n[INFO] Loading ground truth from: {gt_dir}")

    gt_data = {}
    total_coverage = []
    class_counts = defaultdict(int)

    for img_name in tqdm(image_names, desc="Loading GT"):
        gt_path = os.path.join(gt_dir, f"{img_name}.npy")

        if not os.path.exists(gt_path):
            logger.warning(f"GT file not found: {gt_path}, skipping")
            continue

        # Load semantic map
        semantic_map = np.load(gt_path)  # (H, W), int32

        if label_merge_map:
            semantic_map = apply_label_merges(semantic_map, label_merge_map)

        # Extract coverage mask (valid pixels)
        coverage_mask = semantic_map >= 0

        # Get unique classes in this image (excluding -1)
        unique_classes = np.unique(semantic_map[coverage_mask]).tolist()

        # Compute coverage
        coverage = coverage_mask.sum() / coverage_mask.size
        total_coverage.append(coverage)

        # Count class pixels
        for cls in unique_classes:
            class_counts[cls] += (semantic_map == cls).sum()

        gt_data[img_name] = {
            'semantic_map': semantic_map,
            'coverage_mask': coverage_mask,
            'classes': unique_classes
        }

    # Print statistics
    avg_coverage = np.mean(total_coverage) * 100
    logger.info(f"\n[INFO] Loaded {len(gt_data)} ground truth images")
    logger.info(f"[INFO] Average GT coverage: {avg_coverage:.1f}%")
    logger.info(f"[INFO] Class distribution:")

    total_pixels = sum(class_counts.values())
    for cls_id in sorted(class_counts.keys()):
        cls_entry = class_mapping.get(cls_id, f"class_{cls_id}")
        cls_name = cls_entry[0] if isinstance(cls_entry, list) else cls_entry
        count = class_counts[cls_id]
        percentage = count / total_pixels * 100
        logger.info(f"       {cls_name} (ID={cls_id}): {count:,} pixels ({percentage:.1f}%)")

    return gt_data


# ============================================================================
# Feature Decoding and Query
# ============================================================================

def decode_features(
    compressed_features: torch.Tensor,
    autoencoder: Autoencoder,
    device: torch.device
) -> torch.Tensor:
    """
    Decode compressed 3-dim features to 512-dim CLIP features.

    Args:
        compressed_features: (H, W, 3) tensor
        autoencoder: Trained autoencoder model
        device: torch device

    Returns:
        decoded_features: (H, W, 512) tensor
    """
    H, W, _ = compressed_features.shape

    with torch.no_grad():
        # Flatten and decode
        flat_features = compressed_features.reshape(-1, 3).to(device)
        decoded = autoencoder.decode(flat_features)  # (H*W, 512)
        decoded = decoded.reshape(H, W, 512)

    return decoded


def query_semantic_map(
    features_512: torch.Tensor,
    clip_model: OpenCLIPNetwork,
    queries: List[str],
    use_softmax: bool = False
) -> Tuple[np.ndarray, Dict, np.ndarray]:
    """
    Query semantic map from CLIP features.

    Args:
        features_512: (H, W, 512) decoded features
        clip_model: OpenCLIP model
        queries: List of text queries (class names)
        use_softmax: Use softmax across queries (mutual exclusion)

    Returns:
        semantic_map: (H, W) with predicted query indices
        heatmaps: Dict mapping query to (H, W) relevance heatmap
        confidence_map: (H, W) float array with per-pixel confidence
    """
    H, W, _ = features_512.shape
    clip_model.set_positives(queries)

    # Get relevance scores for each query
    # clip_model.get_max_across expects (1, H, W, 512) and returns (1, n_queries, H, W)
    features_input = features_512.unsqueeze(0)  # (1, H, W, 512)

    with torch.no_grad():
        relevance_maps = clip_model.get_max_across(features_input)  # (1, n_queries, H, W)

    relevance_maps = relevance_maps.squeeze(0)  # (n_queries, H, W)

    # Store heatmaps
    heatmaps = {}
    for i, query in enumerate(queries):
        heatmaps[query] = relevance_maps[i].cpu().numpy()

    # Generate semantic map
    if use_softmax:
        # Softmax across queries (mutual exclusion)
        probs = torch.softmax(relevance_maps, dim=0)  # (n_queries, H, W)
        pred_class = torch.argmax(probs, dim=0).cpu().numpy()  # (H, W)
        confidence = torch.max(probs, dim=0)[0].cpu().numpy()  # (H, W)
    else:
        # Independent mode: take class with highest relevance and store normalized confidence
        argmax_class = relevance_maps.argmax(dim=0).cpu().numpy()  # (H, W)
        confidence = np.zeros((H, W), dtype=np.float32)
        for i in range(len(queries)):
            rel = relevance_maps[i].cpu().numpy()
            rel_norm = (rel - rel.min()) / (rel.max() - rel.min() + 1e-9)
            mask = (argmax_class == i)
            confidence[mask] = rel_norm[mask]
        pred_class = argmax_class

    return pred_class, heatmaps, confidence


def apply_label_merges(
    label_map: np.ndarray,
    merge_map: Dict[int, List[int]]
) -> np.ndarray:
    """
    Merge multiple label IDs into canonical IDs in-place.
    """
    if merge_map is None:
        return label_map

    for target_id, source_ids in merge_map.items():
        target_id = int(target_id)
        for src_id in source_ids:
            src_id = int(src_id)
            if src_id == target_id:
                continue
            label_map[label_map == src_id] = target_id
    return label_map


# ============================================================================
# Evaluation Metrics
# ============================================================================

def compute_iou(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray
) -> float:
    """
    Compute IoU between two binary masks.

    Args:
        pred_mask: Boolean array
        gt_mask: Boolean array

    Returns:
        IoU value (0-1)
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()

    if union == 0:
        return 0.0

    return intersection / union


def compute_metrics(
    pred_semantic: np.ndarray,
    gt_semantic: np.ndarray,
    gt_mask: np.ndarray,
    class_mapping: Dict[int, Union[str, List[str]]],
    logger: logging.Logger
) -> Dict:
    """
    Compute evaluation metrics on valid GT region.

    Args:
        pred_semantic: (H, W) predicted class labels
        gt_semantic: (H, W) ground truth class labels
        gt_mask: (H, W) boolean mask of valid GT pixels
        class_mapping: Mapping from class_id to class_name
        logger: Logger instance

    Returns:
        Dictionary with metrics
    """
    # Extract valid region
    pred_valid = pred_semantic[gt_mask]
    gt_valid = gt_semantic[gt_mask]

    if len(gt_valid) == 0:
        logger.warning("No valid GT pixels for evaluation!")
        return {}

    # Get unique classes in GT
    gt_classes = np.unique(gt_valid).tolist()

    # Per-class IoU
    class_iou = {}
    class_precision = {}
    class_recall = {}
    class_pixel_counts = {}

    for cls_id in gt_classes:
        cls_entry = class_mapping.get(cls_id, f"class_{cls_id}")
        cls_name = cls_entry[0] if isinstance(cls_entry, list) else cls_entry

        gt_cls_mask = (gt_valid == cls_id)
        pred_cls_mask = (pred_valid == cls_id)

        # IoU
        iou = compute_iou(pred_cls_mask, gt_cls_mask)
        class_iou[cls_name] = iou

        # Precision and Recall
        tp = np.logical_and(pred_cls_mask, gt_cls_mask).sum()
        fp = np.logical_and(pred_cls_mask, ~gt_cls_mask).sum()
        fn = np.logical_and(~pred_cls_mask, gt_cls_mask).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        class_precision[cls_name] = precision
        class_recall[cls_name] = recall
        class_pixel_counts[cls_name] = gt_cls_mask.sum()

    # Mean IoU
    mean_iou = np.mean(list(class_iou.values()))

    # Weighted IoU (by pixel count)
    total_pixels = sum(class_pixel_counts.values())
    weighted_iou = sum(
        class_iou[cls] * class_pixel_counts[cls] / total_pixels
        for cls in class_iou.keys()
    )

    # Pixel Accuracy
    pixel_acc = (pred_valid == gt_valid).sum() / len(gt_valid)

    return {
        'class_iou': class_iou,
        'class_precision': class_precision,
        'class_recall': class_recall,
        'class_pixel_counts': class_pixel_counts,
        'mean_iou': mean_iou,
        'weighted_iou': weighted_iou,
        'pixel_accuracy': pixel_acc,
        'total_valid_pixels': len(gt_valid),
        'coverage': gt_mask.sum() / gt_mask.size
    }


# ============================================================================
# Visualization
# ============================================================================

def visualize_comparison(
    pred_semantic: np.ndarray,
    gt_semantic: np.ndarray,
    gt_mask: np.ndarray,
    class_mapping: Dict[int, Union[str, List[str]]],
    rgb_image: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    class_colors: Optional[Dict[int, List[float]]] = None
):
    """
    Visualize prediction vs ground truth comparison.

    Args:
        pred_semantic: (H, W) predicted labels
        gt_semantic: (H, W) ground truth labels
        gt_mask: (H, W) valid GT mask
        class_mapping: Class ID to name mapping
        rgb_image: Optional (H, W, 3) RGB image
        save_path: Path to save figure
    """
    # Generate color map using provided colors if available
    combined_classes = np.unique(np.concatenate([
        gt_semantic[gt_mask],
        pred_semantic[gt_mask]
    ]))

    cmap = plt.cm.get_cmap('tab20', max(len(combined_classes), 1))
    dynamic_colors = {}
    normalized_class_colors = None

    if class_colors is not None:
        normalized_class_colors = {
            int(cls_id): np.array(color, dtype=np.float32) / 255.0
            for cls_id, color in class_colors.items()
        }

    for idx, cls_id in enumerate(combined_classes):
        if cls_id < 0:
            continue
        if normalized_class_colors is not None and int(cls_id) in normalized_class_colors:
            dynamic_colors[int(cls_id)] = normalized_class_colors[int(cls_id)]
        else:
            dynamic_colors[int(cls_id)] = np.array(cmap(idx)[:3])

    # Ensure background color exists
    if normalized_class_colors is not None and -1 in normalized_class_colors:
        background_color = normalized_class_colors[-1]
    else:
        background_color = np.array([0.0, 0.0, 0.0])

    # Create colored semantic maps
    def colorize(semantic_map, mask):
        colored = np.ones((*semantic_map.shape, 3))  # Initialize to white
        for cls_id in np.unique(semantic_map):
            if cls_id < 0:
                continue
            color = dynamic_colors.get(int(cls_id))
            if color is None and normalized_class_colors is not None:
                color = normalized_class_colors.get(int(cls_id))
            if color is None:
                color = np.array([0.5, 0.5, 0.5])
            colored[semantic_map == cls_id] = color
        colored[semantic_map < 0] = background_color
        # Set invalid regions to white
        colored[~mask] = 1.0
        return colored

    gt_colored = colorize(gt_semantic, gt_mask)
    pred_colored = colorize(pred_semantic, gt_mask)

    # Error map (only in valid region)
    error_map = np.ones((*gt_semantic.shape, 3))  # Initialize to white
    correct = (pred_semantic == gt_semantic) & gt_mask
    incorrect = (pred_semantic != gt_semantic) & gt_mask
    error_map[correct] = [0, 1, 0]  # Green
    error_map[incorrect] = [1, 0, 0]  # Red
    # Invalid regions remain white (1.0)

    # Plot
    n_plots = 4 if rgb_image is not None else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))

    idx = 0
    if rgb_image is not None:
        axes[idx].imshow(rgb_image)
        axes[idx].set_title('RGB Image')
        axes[idx].axis('off')
        idx += 1

    axes[idx].imshow(gt_colored)
    axes[idx].set_title('Ground Truth')
    axes[idx].axis('off')
    idx += 1

    axes[idx].imshow(pred_colored)
    axes[idx].set_title('Prediction')
    axes[idx].axis('off')
    idx += 1

    axes[idx].imshow(error_map)
    axes[idx].set_title('Error Map (Green=Correct, Red=Wrong)')
    axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_heatmaps(
    heatmaps: Dict[str, np.ndarray],
    rgb_image: Optional[np.ndarray],
    output_dir: str,
    img_name: str
):
    """Save relevance heatmaps for each query."""
    os.makedirs(output_dir, exist_ok=True)

    for query, heatmap in heatmaps.items():
        # Normalize heatmap
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-9)

        # Apply colormap
        heatmap_colored = plt.cm.turbo(heatmap_norm)[:, :, :3]

        # Overlay on RGB if available
        if rgb_image is not None:
            alpha = 0.5
            overlay = alpha * heatmap_colored + (1 - alpha) * rgb_image
            overlay = np.clip(overlay, 0, 1)
        else:
            overlay = heatmap_colored

        # Save
        save_path = os.path.join(output_dir, f"{img_name}_{query}.png")
        plt.imsave(save_path, overlay)


# ============================================================================
# Main Evaluation
# ============================================================================

def evaluate(
    rendered_features_dir: str,
    gt_semantic_dir: str,
    ae_checkpoint: str,
    class_mapping: Dict[int, Union[str, List[str]]],
    output_dir: str,
    rgb_images_dir: Optional[str] = None,
    mask_thresh: float = 0.5,
    use_softmax: bool = True,
    save_visualizations: bool = True,
    num_images: Optional[int] = None,
    encoder_dims: List[int] = [256, 128, 64, 32, 3],
    decoder_dims: List[int] = [16, 32, 64, 128, 256, 256, 512],
    class_colors_path: Optional[str] = None,
    gt_merge_map: Optional[Dict[int, List[int]]] = None,
    class_thresholds: Optional[Dict[int, float]] = None,
    logger: logging.Logger = None
):
    """
    Main evaluation function.

    Args:
        rendered_features_dir: Directory with rendered 3-dim features (*.npy)
        gt_semantic_dir: Directory with ground truth semantic maps (*.npy)
        ae_checkpoint: Path to autoencoder checkpoint
        class_mapping: Mapping from class_id (int) to class_name (str)
        output_dir: Output directory for results
        rgb_images_dir: Optional directory with RGB images for visualization
        mask_thresh: Threshold for semantic queries
        use_softmax: Use softmax for mutually exclusive classes
        save_visualizations: Whether to save visualizations
        num_images: Optional limit on number of images to process
        encoder_dims: Autoencoder encoder dimensions
        decoder_dims: Autoencoder decoder dimensions
        class_colors_path: Optional path to class_colors.json (defaults to gt_semantic_dir/class_colors.json)
        gt_merge_map: Optional dict mapping target class IDs to lists of GT IDs to merge
        class_thresholds: Optional dict mapping class IDs to per-class confidence thresholds
        logger: Logger instance
    """
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting Evaluation with Point Cloud Ground Truth")
    logger.info(f"{'='*60}")
    logger.info(f"Rendered features: {rendered_features_dir}")
    logger.info(f"GT semantic maps: {gt_semantic_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Threshold: {mask_thresh}")
    logger.info(f"Use softmax: {use_softmax}")

    # Load models
    logger.info("\n[INFO] Loading models...")
    clip_model = OpenCLIPNetwork(device)

    checkpoint = torch.load(ae_checkpoint, map_location=device)
    autoencoder = Autoencoder(encoder_dims, decoder_dims).to(device)
    autoencoder.load_state_dict(checkpoint)
    autoencoder.eval()

    # Load class colors if available
    class_colors = None
    colors_path = class_colors_path or os.path.join(gt_semantic_dir, 'class_colors.json')
    if os.path.exists(colors_path):
        with open(colors_path, 'r') as f:
            colors_raw = json.load(f)
        class_colors = {int(k): v for k, v in colors_raw.items()}
        logger.info(f"[INFO] Loaded class colors from {colors_path}")
    else:
        logger.info(f"[INFO] No class_colors.json found at {colors_path}, using default palette")

    if gt_merge_map:
        logger.info(f"[INFO] Using GT label merge map: {gt_merge_map}")
    if class_thresholds:
        logger.info(f"[INFO] Using class-specific thresholds: {class_thresholds}")

    # Get list of rendered features
    feature_files = sorted(glob.glob(os.path.join(rendered_features_dir, '*.npy')))
    if num_images:
        feature_files = feature_files[:num_images]

    image_names = [Path(f).stem for f in feature_files]

    # Load ground truth first to determine which classes are actually present
    gt_data = load_ground_truth(gt_semantic_dir, image_names, class_mapping, logger, gt_merge_map)

    # Collect all unique class IDs present in GT
    gt_class_ids = set()
    for img_data in gt_data.values():
        gt_class_ids.update(img_data['classes'])

    gt_class_ids_sorted = sorted(gt_class_ids)
    logger.info(f"\n[INFO] Classes present in GT: {gt_class_ids_sorted}")
    logger.info(f"[INFO] Class names: {[class_mapping.get(cid, f'class_{cid}') for cid in gt_class_ids_sorted]}")

    # Prepare queries ONLY for classes present in GT
    queries: List[str] = []
    query_class_ids: List[int] = []
    for cls_id in gt_class_ids_sorted:
        if cls_id not in class_mapping:
            logger.warning(f"Class ID {cls_id} found in GT but not in class_mapping, skipping")
            continue

        phrases = class_mapping[cls_id]
        if isinstance(phrases, list):
            prompt_list = phrases
        else:
            prompt_list = [phrases]
        for phrase in prompt_list:
            queries.append(phrase)
            query_class_ids.append(cls_id)

    query_class_ids_np = np.array(query_class_ids, dtype=np.int32)
    logger.info(f"\n[INFO] Total semantic queries: {len(queries)}")
    logger.info(f"[INFO] Queries: {queries}")

    # Filter to images with GT
    valid_images = [name for name in image_names if name in gt_data]
    logger.info(f"\n[INFO] Processing {len(valid_images)} images with GT")

    # Aggregate metrics
    all_metrics = []

    # Process each image
    for img_name in tqdm(valid_images, desc="Evaluating"):
        # Load rendered features
        feature_path = os.path.join(rendered_features_dir, f"{img_name}.npy")
        compressed_features = np.load(feature_path)  # (H, W, 3)
        compressed_features = torch.from_numpy(compressed_features).float()

        # Decode features
        features_512 = decode_features(compressed_features, autoencoder, device)

        # Query semantic map
        pred_semantic_idx, heatmaps, confidence_map = query_semantic_map(
            features_512, clip_model, queries, use_softmax
        )

        # Map contiguous query indices back to original class IDs
        pred_semantic = np.full_like(pred_semantic_idx, -1, dtype=np.int32)
        pred_confidence = np.zeros_like(confidence_map, dtype=np.float32)
        valid_mask = (pred_semantic_idx >= 0) & (pred_semantic_idx < len(query_class_ids_np))
        if np.any(valid_mask):
            pred_semantic[valid_mask] = query_class_ids_np[pred_semantic_idx[valid_mask]]
            pred_confidence[valid_mask] = confidence_map[valid_mask]

        if gt_merge_map:
            pred_semantic = apply_label_merges(pred_semantic, gt_merge_map)

        # Apply per-class thresholds (default to global threshold)
        threshold_map = np.full(pred_semantic.shape, mask_thresh, dtype=np.float32)
        if class_thresholds:
            for cls_id, cls_thresh in class_thresholds.items():
                threshold_map[pred_semantic == cls_id] = cls_thresh
        pred_semantic[pred_confidence < threshold_map] = -1

        # Get ground truth
        gt_semantic = gt_data[img_name]['semantic_map']
        gt_mask = gt_data[img_name]['coverage_mask']

        # Debug: print unique labels present in prediction and GT
        pred_unique = np.unique(pred_semantic)
        gt_unique = np.unique(gt_semantic[gt_mask])
        logger.info(f"[DEBUG] {img_name}: pred_unique={pred_unique.tolist()}, gt_unique={gt_unique.tolist()}")

        # Compute metrics
        metrics = compute_metrics(
            pred_semantic, gt_semantic, gt_mask, class_mapping, logger
        )
        metrics['image_name'] = img_name
        all_metrics.append(metrics)

        # Visualizations
        if save_visualizations:
            # Get resolution for RGB resizing
            pred_h, pred_w = pred_semantic.shape

            # Load RGB image if available
            rgb_image = None
            if rgb_images_dir:
                # Try common extensions
                for ext in ['.png', '.jpg', '.jpeg', '.JPG']:
                    rgb_path = os.path.join(rgb_images_dir, f"{img_name}{ext}")
                    if os.path.exists(rgb_path):
                        rgb_image = np.array(Image.open(rgb_path)).astype(np.float32) / 255.0

                        # Resize RGB image to match rendered resolution if needed
                        rgb_h, rgb_w = rgb_image.shape[:2]
                        if (rgb_h, rgb_w) != (pred_h, pred_w):
                            logger.debug(
                                f"Resizing RGB image from {rgb_w}×{rgb_h} to {pred_w}×{pred_h}"
                            )
                            rgb_image = cv2.resize(
                                rgb_image,
                                (pred_w, pred_h),
                                interpolation=cv2.INTER_LINEAR
                            )
                        break

            # Comparison visualization
            vis_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            visualize_comparison(
                pred_semantic,
                gt_semantic,
                gt_mask,
                class_mapping,
                rgb_image,
                os.path.join(vis_dir, f"{img_name}_comparison.png"),
                class_colors,
            )

            # Heatmaps
            heatmap_dir = os.path.join(output_dir, 'heatmaps')
            save_heatmaps(heatmaps, rgb_image, heatmap_dir, img_name)

    # Aggregate results
    logger.info(f"\n{'='*60}")
    logger.info("Evaluation Results")
    logger.info(f"{'='*60}\n")

    # Per-class aggregated metrics
    all_class_iou = defaultdict(list)
    all_class_precision = defaultdict(list)
    all_class_recall = defaultdict(list)

    for metrics in all_metrics:
        for cls_name, iou in metrics.get('class_iou', {}).items():
            all_class_iou[cls_name].append(iou)
        for cls_name, prec in metrics.get('class_precision', {}).items():
            all_class_precision[cls_name].append(prec)
        for cls_name, rec in metrics.get('class_recall', {}).items():
            all_class_recall[cls_name].append(rec)

    # Print per-class results
    logger.info("Per-Class Results:")
    logger.info(f"{'Class':<20} {'IoU':<10} {'Precision':<12} {'Recall':<10}")
    logger.info("-" * 60)

    for cls_name in sorted(all_class_iou.keys()):
        avg_iou = np.mean(all_class_iou[cls_name])
        avg_prec = np.mean(all_class_precision[cls_name])
        avg_rec = np.mean(all_class_recall[cls_name])
        logger.info(f"{cls_name:<20} {avg_iou:<10.4f} {avg_prec:<12.4f} {avg_rec:<10.4f}")

    # Overall metrics
    mean_iou_all = np.mean([m['mean_iou'] for m in all_metrics])
    weighted_iou_all = np.mean([m['weighted_iou'] for m in all_metrics])
    pixel_acc_all = np.mean([m['pixel_accuracy'] for m in all_metrics])
    avg_coverage = np.mean([m['coverage'] for m in all_metrics])

    logger.info(f"\n{'='*60}")
    logger.info("Overall Metrics:")
    logger.info(f"  Mean IoU:       {mean_iou_all:.4f}")
    logger.info(f"  Weighted IoU:   {weighted_iou_all:.4f}")
    logger.info(f"  Pixel Accuracy: {pixel_acc_all:.4f}")
    logger.info(f"  Avg Coverage:   {avg_coverage*100:.1f}%")
    logger.info(f"{'='*60}\n")

    # Save results to JSON
    results = {
        'overall': {
            'mean_iou': float(mean_iou_all),
            'weighted_iou': float(weighted_iou_all),
            'pixel_accuracy': float(pixel_acc_all),
            'average_coverage': float(avg_coverage)
        },
        'per_class': {
            cls_name: {
                'iou': float(np.mean(all_class_iou[cls_name])),
                'precision': float(np.mean(all_class_precision[cls_name])),
                'recall': float(np.mean(all_class_recall[cls_name]))
            }
            for cls_name in sorted(all_class_iou.keys())
        },
        'per_image': all_metrics
    }

    # Convert numpy types to Python native types for JSON serialization
    def convert_to_serializable(obj):
        """Convert numpy types to Python native types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    results_serializable = convert_to_serializable(results)

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results_serializable, f, indent=2)

    logger.info(f"✅ Evaluation complete! Results saved to: {output_dir}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LangSplat with point cloud-derived ground truth"
    )
    parser.add_argument('--rendered_features', type=str, default="./LangSplat/output/building1/train/renders_npy",
                        help='Directory with rendered 3-dim features (*.npy)')
    parser.add_argument('--gt_semantic_dir', type=str, default="./zaha/outputs/b1_gt_maps",
                        help='Directory with ground truth semantic maps (*.npy)')
    parser.add_argument('--ae_checkpoint', type=str, default="./LangSplat/autoencoder/ckpt/building1/best_ckpt.pth",
                        help='Path to autoencoder checkpoint')
    parser.add_argument('--class_mapping', type=str, default="class_mapping.json",
                        help='JSON file mapping class_id (int) to class_name (str)')
    parser.add_argument('--output_dir', type=str, default="./output/b1",
                        help='Output directory for results')
    parser.add_argument('--rgb_images_dir', type=str, default="./LangSplat/data/building1/images",
                        help='Optional directory with RGB images for visualization')
    parser.add_argument('--mask_thresh', type=float, default=0.3,
                        help='Threshold for semantic mask (default: 0.1, use 0.1-0.15 for softmax mode)')
    parser.add_argument('--use_softmax', action='store_true', default=False,
                        help='Use softmax for mutually exclusive classes (default: False)')
    parser.add_argument('--no_softmax', action='store_false', dest='use_softmax',
                        help='Disable softmax (allow overlapping classes)')
    parser.add_argument('--save_visualizations', action='store_true', default=True,
                        help='Save visualization images (default: True)')
    parser.add_argument('--num_images', type=int, default=None,
                        help='Limit number of images to process (for debugging)')
    parser.add_argument('--encoder_dims', nargs='+', type=int,
                        default=[256, 128, 64, 32, 3],
                        help='Autoencoder encoder dimensions')
    parser.add_argument('--decoder_dims', nargs='+', type=int,
                        default=[16, 32, 64, 128, 256, 256, 512],
                        help='Autoencoder decoder dimensions')
    parser.add_argument('--class_colors', type=str, default="class_colors.json",
                        help='Optional path to class_colors.json (defaults to {gt_semantic_dir}/class_colors.json)')
    parser.add_argument('--gt_merge_map', type=str, default="window_merge.json",
                        help='Optional JSON mapping target class IDs to lists of GT IDs to merge (e.g., {"2": [2,5,13]})')
    parser.add_argument('--wall_thresh', type=float, default=0.3,
                        help='Optional per-class confidence threshold for wall class (ID=1)')
    parser.add_argument('--window_thresh', type=float, default=0.3,
                        help='Optional per-class confidence threshold for window class (ID=2)')
    parser.add_argument('--molding_thresh', type=float, default=0.3,
                        help='Optional per-class confidence threshold for molding class (ID=5)')
    parser.add_argument('--roof_thresh', type=float, default=0.3,
                        help='Optional per-class confidence threshold for roof class (ID=12)')

    args = parser.parse_args()

    # Load class mapping
    with open(args.class_mapping, 'r') as f:
        class_mapping_str = json.load(f)
        class_mapping = {int(k): v for k, v in class_mapping_str.items()}

    # Setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, 'evaluation.log')
    logger = get_logger('eval', log_file=log_file)

    # Run evaluation
    gt_merge_map = None
    if args.gt_merge_map:
        with open(args.gt_merge_map, 'r') as f:
            merge_raw = json.load(f)
        gt_merge_map = {
            int(k): [int(v) for v in values]
            for k, values in merge_raw.items()
        }

    class_thresholds = {}
    if args.wall_thresh is not None:
        class_thresholds[1] = args.wall_thresh
    if args.window_thresh is not None:
        class_thresholds[2] = args.window_thresh
    if args.molding_thresh is not None:
        class_thresholds[5] = args.molding_thresh
    if args.roof_thresh is not None:
        class_thresholds[12] = args.roof_thresh
    if len(class_thresholds) == 0:
        class_thresholds = None

    evaluate(
        rendered_features_dir=args.rendered_features,
        gt_semantic_dir=args.gt_semantic_dir,
        ae_checkpoint=args.ae_checkpoint,
        class_mapping=class_mapping,
        output_dir=args.output_dir,
        rgb_images_dir=args.rgb_images_dir,
        mask_thresh=args.mask_thresh,
        use_softmax=args.use_softmax,
        save_visualizations=args.save_visualizations,
        num_images=args.num_images,
        encoder_dims=args.encoder_dims,
        decoder_dims=args.decoder_dims,
        class_colors_path=args.class_colors,
        gt_merge_map=gt_merge_map,
        class_thresholds=class_thresholds,
        logger=logger
    )


if __name__ == '__main__':
    main()

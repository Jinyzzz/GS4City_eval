#!/usr/bin/env python
"""
Project 3D Point Cloud to 2D Ground Truth for LangSplat Evaluation

This script projects a classified point cloud (PLY) to 2D semantic segmentation maps
for each camera view, generating ground truth data suitable for evaluate_with_pointcloud_gt.py.

Key Features:
- Direct point projection with depth buffering (handles occlusion)
- Optional hole filling (occlusion-aware recommended)
- Outputs GT format required by evaluate_with_pointcloud_gt.py
- Saves visualization PNGs for verification

Usage:
    python project_2d_gt.py \
        --ply_path outputs/zaha_merged_labeled.ply \
        --colmap_dir ../data/building1_15/sparse/0 \
        --output_dir outputs/gt_semantic_maps \
        --fill_holes nearest \
        --save_vis

Output Format (for evaluate_with_pointcloud_gt.py):
    output_dir/
    ├── {image_name}.npy          # (H, W) int32, -1=background, 0,1,2,...=class_id
    ├── {image_name}_vis.png      # Visualization (colored semantic map)
    ├── statistics.json            # Coverage and class distribution
    └── class_colors.json          # Class color mapping
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from plyfile import PlyData
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path to import from LangSplat
# Get the absolute path to the parent directory (LangSplat root)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from scene import colmap_loader


# ============================================================================
# Point Cloud Loading
# ============================================================================

def load_ply_pointcloud(
    ply_path: str,
    class_field: str = 'scalar_Classification'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load point cloud from PLY file with classification labels.

    Args:
        ply_path: Path to PLY file
        class_field: Name of classification field in PLY

    Returns:
        points: (N, 3) array of XYZ coordinates
        classes: (N,) array of classification labels
    """
    print(f"\n{'='*70}")
    print(f"Loading Point Cloud")
    print(f"{'='*70}")
    print(f"File: {ply_path}")

    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    ply_data = PlyData.read(ply_path)
    vertex = ply_data['vertex']

    # Extract coordinates
    points = np.vstack([
        vertex['x'],
        vertex['y'],
        vertex['z']
    ]).T

    # Extract classification
    if class_field not in vertex.data.dtype.names:
        raise ValueError(
            f"Field '{class_field}' not found in PLY.\n"
            f"Available fields: {vertex.data.dtype.names}"
        )

    classes = vertex[class_field].astype(np.int32)

    print(f"✅ Loaded {len(points):,} points")
    print(f"\n📊 Class Distribution:")
    unique_classes, counts = np.unique(classes, return_counts=True)
    for cls_id, count in zip(unique_classes, counts):
        percentage = count / len(classes) * 100
        print(f"   Class {cls_id:2d}: {count:10,} points ({percentage:5.1f}%)")

    print(f"\n📐 Point Cloud Bounds:")
    print(f"   X: [{points[:,0].min():8.2f}, {points[:,0].max():8.2f}]")
    print(f"   Y: [{points[:,1].min():8.2f}, {points[:,1].max():8.2f}]")
    print(f"   Z: [{points[:,2].min():8.2f}, {points[:,2].max():8.2f}]")

    return points, classes


# ============================================================================
# COLMAP Camera Loading
# ============================================================================

def load_colmap_cameras(
    sparse_dir: str,
    target_width: int = None,
    target_height: int = None
) -> Dict[str, Dict]:
    """
    Load COLMAP camera parameters and poses.

    Args:
        sparse_dir: Path to COLMAP sparse reconstruction (sparse/0)
        target_width: Target image width (optional, for rescaling)
        target_height: Target image height (optional, for rescaling)

    Returns:
        Dictionary mapping image_name to camera parameters:
        {
            'K': (3, 3) intrinsic matrix,
            'R': (3, 3) rotation matrix (world to camera),
            'T': (3,) translation vector,
            'width': image width,
            'height': image height
        }
    """
    print(f"\n{'='*70}")
    print(f"Loading COLMAP Cameras")
    print(f"{'='*70}")
    print(f"Directory: {sparse_dir}")

    # Try binary first, fallback to text
    cameras_file_bin = os.path.join(sparse_dir, 'cameras.bin')
    images_file_bin = os.path.join(sparse_dir, 'images.bin')
    cameras_file_txt = os.path.join(sparse_dir, 'cameras.txt')
    images_file_txt = os.path.join(sparse_dir, 'images.txt')

    if os.path.exists(cameras_file_bin) and os.path.exists(images_file_bin):
        print("Format: Binary")
        cameras = colmap_loader.read_intrinsics_binary(cameras_file_bin)
        images = colmap_loader.read_extrinsics_binary(images_file_bin)
    elif os.path.exists(cameras_file_txt) and os.path.exists(images_file_txt):
        print("Format: Text")
        cameras = colmap_loader.read_intrinsics_text(cameras_file_txt)
        images = colmap_loader.read_extrinsics_text(images_file_txt)
    else:
        raise FileNotFoundError(
            f"COLMAP data not found in {sparse_dir}\n"
            f"Expected files: cameras.bin + images.bin OR cameras.txt + images.txt"
        )

    print(f"✅ Loaded {len(cameras)} camera models, {len(images)} images")

    # Build camera dictionary
    camera_dict = {}

    for img_data in images.values():
        cam_id = img_data.camera_id
        cam = cameras[cam_id]

        # Build intrinsic matrix K
        if cam.model in ['PINHOLE', 'SIMPLE_PINHOLE']:
            if cam.model == 'PINHOLE':
                fx, fy, cx, cy = cam.params
            else:  # SIMPLE_PINHOLE
                f, cx, cy = cam.params
                fx = fy = f

            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float64)
        else:
            raise ValueError(f"Unsupported camera model: {cam.model}")

        # Build extrinsics (world to camera)
        R = colmap_loader.qvec2rotmat(img_data.qvec)
        T = img_data.tvec

        # Rescale camera if target resolution is specified
        original_width = cam.width
        original_height = cam.height

        if target_width is not None and target_height is not None:
            # Compute scale factors
            scale_x = target_width / original_width
            scale_y = target_height / original_height

            # Adjust intrinsic matrix
            K_rescaled = K.copy()
            K_rescaled[0, 0] *= scale_x  # fx
            K_rescaled[1, 1] *= scale_y  # fy
            K_rescaled[0, 2] *= scale_x  # cx
            K_rescaled[1, 2] *= scale_y  # cy

            camera_dict[img_data.name] = {
                'K': K_rescaled,
                'R': R,
                'T': T,
                'width': target_width,
                'height': target_height
            }
        else:
            camera_dict[img_data.name] = {
                'K': K,
                'R': R,
                'T': T,
                'width': original_width,
                'height': original_height
            }

    # Print rescaling info
    if target_width is not None and target_height is not None:
        print(f"📐 Rescaling: {original_width}×{original_height} → {target_width}×{target_height}")
        print(f"   Scale: {target_width/original_width:.3f}x (width), {target_height/original_height:.3f}x (height)")

    return camera_dict


# ============================================================================
# Resolution Detection
# ============================================================================

def detect_resolution_from_rendered_features(rendered_dir: str) -> Tuple[int, int]:
    """
    Automatically detect target resolution from rendered feature files.

    Args:
        rendered_dir: Directory containing rendered .npy feature files

    Returns:
        (width, height) tuple
    """
    import glob

    npy_files = glob.glob(os.path.join(rendered_dir, '*.npy'))
    if not npy_files:
        raise FileNotFoundError(
            f"No .npy files found in {rendered_dir}\n"
            f"Please specify --target_width and --target_height manually."
        )

    # Load first file to get resolution
    first_file = npy_files[0]
    features = np.load(first_file)  # (H, W, C)

    if len(features.shape) != 3:
        raise ValueError(
            f"Expected 3D array (H, W, C), got shape {features.shape} from {first_file}"
        )

    height, width = features.shape[:2]

    print(f"\n{'='*70}")
    print(f"Auto-detected Resolution from Rendered Features")
    print(f"{'='*70}")
    print(f"Source: {rendered_dir}")
    print(f"Sample file: {os.path.basename(first_file)}")
    print(f"Resolution: {width}×{height}")
    print(f"Total files: {len(npy_files)}")

    return width, height


# ============================================================================
# Point Projection
# ============================================================================

def project_points_to_camera(
    points: np.ndarray,
    classes: np.ndarray,
    camera: Dict,
    min_depth: float = 0.01,
    max_depth: float = 100.0,
    background_class: int = -1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project 3D points to 2D image using depth buffer for occlusion handling.

    Args:
        points: (N, 3) point cloud coordinates
        classes: (N,) classification labels
        camera: Camera parameters dict
        min_depth: Minimum valid depth (meters)
        max_depth: Maximum valid depth (meters)
        background_class: Class ID for background pixels

    Returns:
        semantic_map: (H, W) semantic labels (-1 for background)
        depth_map: (H, W) depth values (inf for invalid)
        coverage_mask: (H, W) boolean mask of valid pixels
    """
    K = camera['K']
    R = camera['R']
    T = camera['T']
    width = camera['width']
    height = camera['height']

    # Transform points to camera coordinate system
    # P_cam = R @ P_world + T
    points_cam = (R @ points.T).T + T

    # Filter points behind camera or too far
    valid_depth = (points_cam[:, 2] > min_depth) & (points_cam[:, 2] < max_depth)
    points_cam = points_cam[valid_depth]
    classes_valid = classes[valid_depth]

    if len(points_cam) == 0:
        # No valid points for this camera
        semantic_map = np.full((height, width), background_class, dtype=np.int32)
        depth_map = np.full((height, width), np.inf, dtype=np.float32)
        coverage_mask = np.zeros((height, width), dtype=bool)
        return semantic_map, depth_map, coverage_mask

    # Project to image plane
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    depths = points_cam[:, 2]
    u = (fx * points_cam[:, 0] / depths + cx).astype(np.int32)
    v = (fy * points_cam[:, 1] / depths + cy).astype(np.int32)

    # Filter points outside image bounds
    valid_uv = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[valid_uv]
    v = v[valid_uv]
    depths = depths[valid_uv]
    classes_valid = classes_valid[valid_uv]

    # Initialize depth buffer and semantic map
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
    semantic_map = np.full((height, width), background_class, dtype=np.int32)

    # Depth buffering: keep closest point per pixel
    for i in range(len(u)):
        pixel_u, pixel_v = u[i], v[i]
        pixel_depth = depths[i]
        pixel_class = classes_valid[i]

        if pixel_depth < depth_buffer[pixel_v, pixel_u]:
            depth_buffer[pixel_v, pixel_u] = pixel_depth
            semantic_map[pixel_v, pixel_u] = pixel_class

    # Coverage mask
    coverage_mask = depth_buffer < np.inf

    return semantic_map, depth_buffer, coverage_mask


# ============================================================================
# Hole Filling
# ============================================================================

def fill_holes_nearest_neighbor(
    semantic_map: np.ndarray,
    coverage_mask: np.ndarray,
    max_distance: int = 5
) -> np.ndarray:
    """
    Fill holes using nearest neighbor interpolation.

    Args:
        semantic_map: (H, W) semantic labels
        coverage_mask: (H, W) boolean mask of valid pixels
        max_distance: Maximum distance (in pixels) to search

    Returns:
        filled_semantic_map: (H, W) with holes filled
    """
    from scipy.ndimage import distance_transform_edt

    if coverage_mask.all():
        return semantic_map.copy()

    holes = ~coverage_mask
    distance, indices = distance_transform_edt(holes, return_indices=True)

    filled_semantic_map = semantic_map.copy()
    fill_mask = holes & (distance <= max_distance)

    nearest_y = indices[0][fill_mask]
    nearest_x = indices[1][fill_mask]
    filled_semantic_map[fill_mask] = semantic_map[nearest_y, nearest_x]

    return filled_semantic_map


def fill_holes_occlusion_aware(
    semantic_map: np.ndarray,
    depth_map: np.ndarray,
    coverage_mask: np.ndarray,
    max_distance: int = 5,
    depth_discontinuity_threshold: float = 0.1
) -> np.ndarray:
    """
    Fill holes while respecting occlusion boundaries.

    This method detects depth discontinuities and prevents filling across
    occlusion boundaries, ensuring only small holes are filled.

    Args:
        semantic_map: (H, W) semantic labels
        depth_map: (H, W) depth values
        coverage_mask: (H, W) boolean mask of valid pixels
        max_distance: Maximum distance to fill
        depth_discontinuity_threshold: Relative depth change for boundaries

    Returns:
        filled_semantic_map: (H, W) with occlusion-aware filling
    """
    from scipy.ndimage import sobel, grey_dilation, binary_dilation
    from scipy.spatial import cKDTree

    if coverage_mask.all():
        return semantic_map.copy()

    # Detect occlusion boundaries using depth gradients
    depth_valid = depth_map.copy()
    depth_valid[~coverage_mask] = np.nan

    gradient_x = np.abs(sobel(depth_valid, axis=1, mode='constant'))
    gradient_y = np.abs(sobel(depth_valid, axis=0, mode='constant'))
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    local_depth = grey_dilation(depth_valid, size=5)
    local_depth = np.where(local_depth > 0, local_depth, 1.0)
    relative_gradient = gradient_magnitude / local_depth

    occlusion_boundaries = relative_gradient > depth_discontinuity_threshold
    occlusion_boundaries = binary_dilation(occlusion_boundaries, iterations=1)

    # Only use pixels away from boundaries as fill sources
    fill_source_mask = coverage_mask & ~occlusion_boundaries

    if not fill_source_mask.any():
        return semantic_map.copy()

    # Build KD-tree of fill sources
    source_coords = np.argwhere(fill_source_mask)
    tree = cKDTree(source_coords)

    # Find holes
    holes = ~coverage_mask
    hole_coords = np.argwhere(holes)

    if len(hole_coords) == 0:
        return semantic_map.copy()

    # Query nearest neighbors
    distances, indices = tree.query(hole_coords, k=1)

    # Fill within max_distance
    filled_semantic_map = semantic_map.copy()
    for i, (hole_y, hole_x) in enumerate(hole_coords):
        if distances[i] <= max_distance:
            source_y, source_x = source_coords[indices[i]]
            filled_semantic_map[hole_y, hole_x] = semantic_map[source_y, source_x]

    return filled_semantic_map


# ============================================================================
# Visualization
# ============================================================================

def get_class_colors(classes: np.ndarray) -> Dict[int, Tuple[int, int, int]]:
    """
    Generate color mapping for classes.

    Args:
        classes: Array of unique class IDs

    Returns:
        Dictionary mapping class_id to (R, G, B) tuple
    """
    unique_classes = np.unique(classes[classes >= 0])
    n_classes = len(unique_classes)

    if n_classes <= 20:
        cmap = plt.cm.get_cmap('tab20', n_classes)
    else:
        cmap = plt.cm.get_cmap('hsv', n_classes)

    class_colors = {}
    for i, cls in enumerate(unique_classes):
        color = cmap(i)[:3]
        class_colors[int(cls)] = tuple(int(c * 255) for c in color)

    class_colors[-1] = (0, 0, 0)  # Background = black

    return class_colors


def visualize_semantic_map(
    semantic_map: np.ndarray,
    class_colors: Dict[int, Tuple[int, int, int]]
) -> np.ndarray:
    """
    Convert semantic map to RGB visualization.

    Args:
        semantic_map: (H, W) integer class labels
        class_colors: Mapping from class_id to (R, G, B)

    Returns:
        rgb_image: (H, W, 3) uint8 RGB image
    """
    H, W = semantic_map.shape
    rgb_image = np.zeros((H, W, 3), dtype=np.uint8)

    for class_id, color in class_colors.items():
        mask = semantic_map == class_id
        rgb_image[mask] = color

    return rgb_image


# ============================================================================
# Main Processing
# ============================================================================

def process_all_cameras(
    points: np.ndarray,
    classes: np.ndarray,
    cameras: Dict[str, Dict],
    output_dir: str,
    fill_holes: str = 'nearest',
    fill_distance: int = 5,
    depth_discontinuity_threshold: float = 0.1,
    save_vis: bool = True,
    min_depth: float = 0.01,
    max_depth: float = 100.0
):
    """
    Process all cameras and save ground truth semantic maps.

    Args:
        points: (N, 3) point cloud
        classes: (N,) class labels
        cameras: Camera dictionary
        output_dir: Output directory
        fill_holes: Hole filling method ('none', 'nearest', 'occlusion_aware')
        fill_distance: Maximum distance for hole filling
        depth_discontinuity_threshold: Threshold for occlusion detection
        save_vis: Save visualization PNGs
        min_depth: Minimum valid depth
        max_depth: Maximum valid depth
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Projecting Points to 2D Ground Truth")
    print(f"{'='*70}")
    print(f"Output: {output_dir}")
    print(f"Cameras: {len(cameras)}")
    print(f"Fill method: {fill_holes}")
    print(f"Fill distance: {fill_distance} pixels")
    print()

    # Generate class colors
    class_colors = get_class_colors(classes)

    # Statistics
    statistics = {
        'total_images': len(cameras),
        'fill_method': fill_holes,
        'per_image': {}
    }

    for img_name, camera in tqdm(cameras.items(), desc="Processing views"):
        # Project points
        semantic_map, depth_map, coverage_mask = project_points_to_camera(
            points, classes, camera, min_depth, max_depth
        )

        coverage_before = coverage_mask.sum() / coverage_mask.size

        # Apply hole filling
        if fill_holes == 'nearest':
            semantic_map_filled = fill_holes_nearest_neighbor(
                semantic_map, coverage_mask, fill_distance
            )
        elif fill_holes == 'occlusion_aware':
            semantic_map_filled = fill_holes_occlusion_aware(
                semantic_map, depth_map, coverage_mask,
                fill_distance, depth_discontinuity_threshold
            )
        else:  # 'none'
            semantic_map_filled = semantic_map.copy()

        # Use filled version for GT
        semantic_map_gt = semantic_map_filled

        # Compute statistics
        valid_pixels = semantic_map_gt >= 0
        coverage_after = valid_pixels.sum() / semantic_map_gt.size
        unique_classes, counts = np.unique(semantic_map_gt[valid_pixels], return_counts=True)

        statistics['per_image'][img_name] = {
            'coverage_before_fill': float(coverage_before),
            'coverage_after_fill': float(coverage_after),
            'classes': {int(c): int(n) for c, n in zip(unique_classes, counts)}
        }

        # Save ground truth semantic map (required format for evaluation)
        base_name = Path(img_name).stem
        np.save(
            os.path.join(output_dir, f"{base_name}.npy"),
            semantic_map_gt
        )

        # Save visualization PNG
        if save_vis:
            rgb_vis = visualize_semantic_map(semantic_map_gt, class_colors)
            Image.fromarray(rgb_vis).save(
                os.path.join(output_dir, f"{base_name}_vis.png")
            )

    # Compute average statistics
    avg_coverage_before = np.mean([s['coverage_before_fill'] for s in statistics['per_image'].values()])
    avg_coverage_after = np.mean([s['coverage_after_fill'] for s in statistics['per_image'].values()])

    statistics['average_coverage_before_fill'] = float(avg_coverage_before)
    statistics['average_coverage_after_fill'] = float(avg_coverage_after)

    # Save statistics
    with open(os.path.join(output_dir, 'statistics.json'), 'w') as f:
        json.dump(statistics, f, indent=2)

    # Save class colors
    with open(os.path.join(output_dir, 'class_colors.json'), 'w') as f:
        json.dump({str(k): list(v) for k, v in class_colors.items()}, f, indent=2)

    print(f"\n{'='*70}")
    print(f"✅ Processing Complete!")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"Total images: {len(cameras)}")
    print(f"Average coverage (before fill): {avg_coverage_before*100:.1f}%")
    print(f"Average coverage (after fill):  {avg_coverage_after*100:.1f}%")
    print(f"\nOutput files:")
    print(f"  - {{image_name}}.npy: Ground truth semantic maps (H, W) int32")
    print(f"  - {{image_name}}_vis.png: Visualization images")
    print(f"  - statistics.json: Coverage and class statistics")
    print(f"  - class_colors.json: Class color mapping")
    print(f"\n💡 Use these files with evaluate_with_pointcloud_gt.py")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Project 3D point cloud to 2D ground truth for LangSplat evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python project_2d_gt.py \\
      --ply_path outputs/zaha_merged_labeled.ply \\
      --colmap_dir ../data/building1_15/sparse/0 \\
      --output_dir outputs/gt_semantic_maps

  # With occlusion-aware hole filling (recommended)
  python project_2d_gt.py \\
      --ply_path outputs/zaha_merged_labeled.ply \\
      --colmap_dir ../data/building1_15/sparse/0 \\
      --output_dir outputs/gt_semantic_maps \\
      --fill_holes occlusion_aware \\
      --fill_distance 5

  # Without hole filling (preserve exact point cloud coverage)
  python project_2d_gt.py \\
      --ply_path outputs/zaha_merged_labeled.ply \\
      --colmap_dir ../data/building1_15/sparse/0 \\
      --output_dir outputs/gt_semantic_maps \\
      --fill_holes none

  # Auto-detect resolution from rendered features (recommended)
  python project_2d_gt.py \\
      --ply_path outputs/zaha_merged_labeled.ply \\
      --colmap_dir ../data/building1_15/sparse/0 \\
      --output_dir outputs/gt_semantic_maps \\
      --rendered_features_dir ../output/building1_15_dual_eval_1/test/renders_npy_eval

  # Manually specify target resolution
  python project_2d_gt.py \\
      --ply_path outputs/zaha_merged_labeled.ply \\
      --colmap_dir ../data/building1_15/sparse/0 \\
      --output_dir outputs/gt_semantic_maps \\
      --target_width 1492 \\
      --target_height 1080
        """
    )

    # Required arguments
    parser.add_argument('--ply_path', type=str, default="./gt_ply/b1_label.ply",
                        help='Path to input PLY point cloud with classification')
    parser.add_argument('--colmap_dir', type=str, default="./data/building1_15/sparse/0",
                        help='Path to COLMAP sparse directory (sparse/0)')
    parser.add_argument('--output_dir', type=str, default="./outputs/b1_gt_maps",
                        help='Output directory for ground truth semantic maps')

    # Resolution options
    parser.add_argument('--rendered_features_dir', type=str, default="/LangSplat/output/building1_15/test/ours_None/renders_npy",
                        help='Directory with rendered features to auto-detect resolution')
    parser.add_argument('--target_width', type=int, default=None,
                        help='Target output width (overrides auto-detection)')
    parser.add_argument('--target_height', type=int, default=None,
                        help='Target output height (overrides auto-detection)')

    # Point cloud options
    parser.add_argument('--class_field', type=str, default='scalar_Classification',
                        help='Name of classification field in PLY (default: scalar_Classification)')

    # Hole filling options
    parser.add_argument('--fill_holes', type=str, default='nearest',
                        choices=['none', 'nearest', 'occlusion_aware'],
                        help='Hole filling method (default: nearest)')
    parser.add_argument('--fill_distance', type=int, default=5,
                        help='Maximum distance for hole filling in pixels (default: 5)')
    parser.add_argument('--depth_discontinuity_threshold', type=float, default=0.1,
                        help='Relative depth change threshold for occlusion detection (default: 0.1)')

    # Projection options
    parser.add_argument('--min_depth', type=float, default=0.01,
                        help='Minimum valid depth in meters (default: 0.01)')
    parser.add_argument('--max_depth', type=float, default=100.0,
                        help='Maximum valid depth in meters (default: 100.0)')

    # Output options
    parser.add_argument('--save_vis', action='store_true', default=True,
                        help='Save visualization PNGs (default: True)')
    parser.add_argument('--no_vis', action='store_false', dest='save_vis',
                        help='Do not save visualization PNGs')

    args = parser.parse_args()

    # Print configuration
    print(f"\n{'='*70}")
    print(f"Project 2D Ground Truth for LangSplat Evaluation")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  PLY path:        {args.ply_path}")
    print(f"  COLMAP dir:      {args.colmap_dir}")
    print(f"  Output dir:      {args.output_dir}")
    print(f"  Class field:     {args.class_field}")
    print(f"  Fill method:     {args.fill_holes}")
    print(f"  Fill distance:   {args.fill_distance} pixels")
    print(f"  Save vis:        {args.save_vis}")

    # Determine target resolution
    target_width = args.target_width
    target_height = args.target_height

    if target_width is not None and target_height is not None:
        # Manual resolution specified
        print(f"  Target resolution: {target_width}×{target_height} (manual)")
    elif args.rendered_features_dir is not None:
        # Auto-detect from rendered features
        target_width, target_height = detect_resolution_from_rendered_features(
            args.rendered_features_dir
        )
    else:
        # Use original COLMAP resolution
        print(f"  Target resolution: Original COLMAP resolution (no rescaling)")

    # Load point cloud
    points, classes = load_ply_pointcloud(args.ply_path, args.class_field)

    # Load cameras (with optional rescaling)
    cameras = load_colmap_cameras(args.colmap_dir, target_width, target_height)

    # Process all cameras
    process_all_cameras(
        points, classes, cameras, args.output_dir,
        fill_holes=args.fill_holes,
        fill_distance=args.fill_distance,
        depth_discontinuity_threshold=args.depth_discontinuity_threshold,
        save_vis=args.save_vis,
        min_depth=args.min_depth,
        max_depth=args.max_depth
    )


if __name__ == '__main__':
    main()

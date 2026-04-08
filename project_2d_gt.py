#!/usr/bin/env python
"""
Project 3D Point Cloud to 2D Ground Truth for LangSplat Evaluation

This script projects a classified point cloud (PLY) to 2D semantic segmentation maps
for each camera view, generating ground truth data suitable for evaluate_with_pointcloud_gt.py.

Key Features:
- Direct point projection with depth buffering (handles occlusion)  [OPTIMIZED: vectorized z-buffer]
- Optional hole filling (occlusion-aware recommended)              [OPTIMIZED: vectorized hole assignment]
- Optional global point filtering by distance to a reference camera center (with enable/disable switch)
- Outputs GT format required by evaluate_with_pointcloud_gt.py
- Saves visualization PNGs for verification
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


# ============================================================================
# DEFAULT PARAMETERS (used when no CLI args are provided)
# ============================================================================

DEFAULT_PLY_PATH = "zaha/zaha_goldcoast_33.ply"
DEFAULT_COLMAP_DIR = "/workspace/LangSplat/data/subset_goldcoast/sparse/0"
DEFAULT_OUTPUT_DIR = "gt/subset_goldcoast_33"

# Resolution options
# NOTE: Set to None by default so we won't auto-detect resolution unless you pass it.
DEFAULT_RENDERED_FEATURES_DIR = None
DEFAULT_TARGET_WIDTH = None
DEFAULT_TARGET_HEIGHT = None

# Point cloud options
DEFAULT_CLASS_FIELD = "scalar_Classification"

# Hole filling options
DEFAULT_FILL_HOLES = "DEFAULT_FILL_HOLES"
DEFAULT_FILL_DISTANCE = 10
DEFAULT_DEPTH_DISCONTINUITY_THRESHOLD = 0.05

# Projection options
DEFAULT_MIN_DEPTH = 0.01
DEFAULT_MAX_DEPTH = 1000.0

# Output options
DEFAULT_SAVE_VIS = True

# New: global distance-based point filtering (using one reference camera)
DEFAULT_ENABLE_DISTANCE_FILTER = False # <-- NEW SWITCH DEFAULT
DEFAULT_DISTANCE_FILTER_CAMERA = "DJI_20241217095813_0011_D.JPG"
DEFAULT_DISTANCE_FILTER_MIN = None
DEFAULT_DISTANCE_FILTER_MAX = 60.0


# ============================================================================
# Add LangSplat to PYTHONPATH
# ============================================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
langsplat_root = os.path.join(repo_root, "LangSplat")
langsplat_root = os.path.abspath(langsplat_root)

print("[INFO] Adding LangSplat root to PYTHONPATH:", langsplat_root)
sys.path.insert(0, langsplat_root)

from scene import colmap_loader


# ============================================================================
# Point Cloud Loading
# ============================================================================

def load_ply_pointcloud(
    ply_path: str,
    class_field: str = 'scalar_Classification'
) -> Tuple[np.ndarray, np.ndarray]:
    print(f"\n{'='*70}")
    print(f"Loading Point Cloud")
    print(f"{'='*70}")
    print(f"File: {ply_path}")

    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    ply_data = PlyData.read(ply_path)
    vertex = ply_data['vertex']

    points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T

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
    print(f"\n{'='*70}")
    print(f"Loading COLMAP Cameras")
    print(f"{'='*70}")
    print(f"Directory: {sparse_dir}")

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

    camera_dict = {}
    original_width = None
    original_height = None

    for img_data in images.values():
        cam_id = img_data.camera_id
        cam = cameras[cam_id]

        if cam.model in ['PINHOLE', 'SIMPLE_PINHOLE']:
            if cam.model == 'PINHOLE':
                fx, fy, cx, cy = cam.params
            else:
                f, cx, cy = cam.params
                fx = fy = f

            K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float64)
        else:
            raise ValueError(f"Unsupported camera model: {cam.model}")

        R = colmap_loader.qvec2rotmat(img_data.qvec)
        T = img_data.tvec

        original_width = cam.width
        original_height = cam.height

        if target_width is not None and target_height is not None:
            scale_x = target_width / original_width
            scale_y = target_height / original_height

            K_rescaled = K.copy()
            K_rescaled[0, 0] *= scale_x
            K_rescaled[1, 1] *= scale_y
            K_rescaled[0, 2] *= scale_x
            K_rescaled[1, 2] *= scale_y

            camera_dict[img_data.name] = {
                'K': K_rescaled, 'R': R, 'T': T,
                'width': target_width, 'height': target_height
            }
        else:
            camera_dict[img_data.name] = {
                'K': K, 'R': R, 'T': T,
                'width': original_width, 'height': original_height
            }

    if target_width is not None and target_height is not None and original_width is not None:
        print(f"📐 Rescaling: {original_width}×{original_height} → {target_width}×{target_height}")
        print(f"   Scale: {target_width/original_width:.3f}x (width), {target_height/original_height:.3f}x (height)")

    return camera_dict


# ============================================================================
# Resolution Detection
# ============================================================================

def detect_resolution_from_rendered_features(rendered_dir: str) -> Tuple[int, int]:
    import glob

    npy_files = glob.glob(os.path.join(rendered_dir, '*.npy'))
    if not npy_files:
        raise FileNotFoundError(
            f"No .npy files found in {rendered_dir}\n"
            f"Please specify --target_width and --target_height manually."
        )

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
# Distance-based global point filtering (by reference camera center)
# ============================================================================

def get_camera_center_world(camera: Dict) -> np.ndarray:
    R = camera['R']
    T = camera['T'].reshape(3)
    return -R.T @ T


def filter_points_by_camera_distance(
    points: np.ndarray,
    classes: np.ndarray,
    camera: Dict,
    min_distance: float = None,
    max_distance: float = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    C = get_camera_center_world(camera)
    d = np.linalg.norm(points - C[None, :], axis=1)

    mask = np.ones(len(points), dtype=bool)
    if min_distance is not None:
        mask &= (d >= float(min_distance))
    if max_distance is not None:
        mask &= (d <= float(max_distance))

    return points[mask], classes[mask], mask


def find_camera_by_name(cameras: Dict[str, Dict], name: str) -> Dict:
    if name in cameras:
        return cameras[name]

    target_stem = Path(name).stem
    for k, cam in cameras.items():
        if Path(k).stem == target_stem:
            return cam

    raise KeyError(f"Reference camera '{name}' not found in COLMAP images.")


# ============================================================================
# Point Projection (OPTIMIZED z-buffer)
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
    OPTIMIZED: vectorized z-buffer (no Python per-point loop).
    """
    K = camera['K']
    R = camera['R']
    T = camera['T']
    width = camera['width']
    height = camera['height']

    # World -> camera
    points_cam = (R @ points.T).T + T

    # Depth filter
    z = points_cam[:, 2]
    valid_depth = (z > min_depth) & (z < max_depth)
    if not np.any(valid_depth):
        semantic_map = np.full((height, width), background_class, dtype=np.int32)
        depth_map = np.full((height, width), np.inf, dtype=np.float32)
        coverage_mask = np.zeros((height, width), dtype=bool)
        return semantic_map, depth_map, coverage_mask

    points_cam = points_cam[valid_depth]
    classes_valid = classes[valid_depth]
    z = points_cam[:, 2]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = (fx * points_cam[:, 0] / z + cx).astype(np.int32)
    v = (fy * points_cam[:, 1] / z + cy).astype(np.int32)

    # Image bounds filter
    valid_uv = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(valid_uv):
        semantic_map = np.full((height, width), background_class, dtype=np.int32)
        depth_map = np.full((height, width), np.inf, dtype=np.float32)
        coverage_mask = np.zeros((height, width), dtype=bool)
        return semantic_map, depth_map, coverage_mask

    u = u[valid_uv]
    v = v[valid_uv]
    depths = z[valid_uv].astype(np.float32)
    classes_valid = classes_valid[valid_uv]

    # Vectorized z-buffer: keep smallest depth per pixel
    lin = v.astype(np.int64) * int(width) + u.astype(np.int64)  # (M,)

    # Sort by (pixel, depth) so first occurrence per pixel is closest
    order = np.lexsort((depths, lin))  # primary key: lin, secondary: depths
    lin_s = lin[order]

    keep = np.empty(len(order), dtype=bool)
    keep[0] = True
    keep[1:] = lin_s[1:] != lin_s[:-1]
    sel = order[keep]

    semantic_flat = np.full(int(height) * int(width), background_class, dtype=np.int32)
    depth_flat = np.full(int(height) * int(width), np.inf, dtype=np.float32)

    semantic_flat[lin[sel]] = classes_valid[sel].astype(np.int32)
    depth_flat[lin[sel]] = depths[sel]

    semantic_map = semantic_flat.reshape(height, width)
    depth_map = depth_flat.reshape(height, width)
    coverage_mask = depth_map < np.inf

    return semantic_map, depth_map, coverage_mask


# ============================================================================
# Hole Filling
# ============================================================================

def fill_holes_nearest_neighbor(
    semantic_map: np.ndarray,
    coverage_mask: np.ndarray,
    max_distance: int = 5
) -> np.ndarray:
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
    OPTIMIZED: vectorized filling assignment (no Python loop over holes).
    """
    from scipy.ndimage import sobel, grey_dilation, binary_dilation
    from scipy.spatial import cKDTree

    if coverage_mask.all():
        return semantic_map.copy()

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

    fill_source_mask = coverage_mask & ~occlusion_boundaries
    if not fill_source_mask.any():
        return semantic_map.copy()

    source_coords = np.argwhere(fill_source_mask)  # (S,2)
    tree = cKDTree(source_coords)

    holes = ~coverage_mask
    hole_coords = np.argwhere(holes)  # (Hn,2)
    if len(hole_coords) == 0:
        return semantic_map.copy()

    distances, nn_idx = tree.query(hole_coords, k=1)

    within = distances <= max_distance
    if not np.any(within):
        return semantic_map.copy()

    holes_in = hole_coords[within]              # (M,2)
    src = source_coords[nn_idx[within]]         # (M,2)

    filled_semantic_map = semantic_map.copy()
    filled_semantic_map[holes_in[:, 0], holes_in[:, 1]] = semantic_map[src[:, 0], src[:, 1]]

    return filled_semantic_map


# ============================================================================
# Visualization
# ============================================================================

def get_class_colors(classes: np.ndarray) -> Dict[int, Tuple[int, int, int]]:
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

    class_colors[-1] = (0, 0, 0)
    return class_colors


def visualize_semantic_map(
    semantic_map: np.ndarray,
    class_colors: Dict[int, Tuple[int, int, int]]
) -> np.ndarray:
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
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Projecting Points to 2D Ground Truth")
    print(f"{'='*70}")
    print(f"Output: {output_dir}")
    print(f"Cameras: {len(cameras)}")
    print(f"Fill method: {fill_holes}")
    print(f"Fill distance: {fill_distance} pixels")
    print()

    class_colors = get_class_colors(classes)

    statistics = {
        'total_images': len(cameras),
        'fill_method': fill_holes,
        'per_image': {}
    }

    for img_name, camera in tqdm(cameras.items(), desc="Processing views"):
        semantic_map, depth_map, coverage_mask = project_points_to_camera(
            points, classes, camera, min_depth, max_depth
        )

        coverage_before = coverage_mask.sum() / coverage_mask.size

        if fill_holes == 'nearest':
            semantic_map_filled = fill_holes_nearest_neighbor(
                semantic_map, coverage_mask, fill_distance
            )
        elif fill_holes == 'occlusion_aware':
            semantic_map_filled = fill_holes_occlusion_aware(
                semantic_map, depth_map, coverage_mask,
                fill_distance, depth_discontinuity_threshold
            )
        else:
            semantic_map_filled = semantic_map.copy()

        semantic_map_gt = semantic_map_filled

        valid_pixels = semantic_map_gt >= 0
        coverage_after = valid_pixels.sum() / semantic_map_gt.size
        if np.any(valid_pixels):
            unique_classes, counts = np.unique(semantic_map_gt[valid_pixels], return_counts=True)
            class_hist = {int(c): int(n) for c, n in zip(unique_classes, counts)}
        else:
            class_hist = {}

        statistics['per_image'][img_name] = {
            'coverage_before_fill': float(coverage_before),
            'coverage_after_fill': float(coverage_after),
            'classes': class_hist
        }

        base_name = Path(img_name).stem
        np.save(os.path.join(output_dir, f"{base_name}.npy"), semantic_map_gt)

        if save_vis:
            rgb_vis = visualize_semantic_map(semantic_map_gt, class_colors)
            Image.fromarray(rgb_vis).save(os.path.join(output_dir, f"{base_name}_vis.png"))

    avg_coverage_before = float(np.mean([s['coverage_before_fill'] for s in statistics['per_image'].values()]))
    avg_coverage_after = float(np.mean([s['coverage_after_fill'] for s in statistics['per_image'].values()]))

    statistics['average_coverage_before_fill'] = avg_coverage_before
    statistics['average_coverage_after_fill'] = avg_coverage_after

    with open(os.path.join(output_dir, 'statistics.json'), 'w') as f:
        json.dump(statistics, f, indent=2)

    with open(os.path.join(output_dir, 'class_colors.json'), 'w') as f:
        json.dump({str(k): list(v) for k, v in class_colors.items()}, f, indent=2)

    print(f"\n{'='*70}")
    print(f"✅ Processing Complete!")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"Total images: {len(cameras)}")
    print(f"Average coverage (before fill): {avg_coverage_before*100:.1f}%")
    print(f"Average coverage (after fill):  {avg_coverage_after*100:.1f}%")
    print(f"\n💡 Use these files with evaluate_with_pointcloud_gt.py")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Project 3D point cloud to 2D ground truth for LangSplat evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--ply_path', type=str, default=DEFAULT_PLY_PATH)
    parser.add_argument('--colmap_dir', type=str, default=DEFAULT_COLMAP_DIR)
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument('--rendered_features_dir', type=str, default=DEFAULT_RENDERED_FEATURES_DIR)
    parser.add_argument('--target_width', type=int, default=DEFAULT_TARGET_WIDTH)
    parser.add_argument('--target_height', type=int, default=DEFAULT_TARGET_HEIGHT)

    parser.add_argument('--class_field', type=str, default=DEFAULT_CLASS_FIELD)

    # NEW: switch to enable/disable distance filtering (other defaults unchanged)
    parser.add_argument('--enable_distance_filter', action='store_true', default=DEFAULT_ENABLE_DISTANCE_FILTER,
                        help='Enable global distance filtering before generating GT (default: enabled).')
    parser.add_argument('--disable_distance_filter', action='store_false', dest='enable_distance_filter',
                        help='Disable global distance filtering.')

    parser.add_argument('--distance_filter_camera', type=str, default=DEFAULT_DISTANCE_FILTER_CAMERA)
    parser.add_argument('--distance_filter_min', type=float, default=DEFAULT_DISTANCE_FILTER_MIN)
    parser.add_argument('--distance_filter_max', type=float, default=DEFAULT_DISTANCE_FILTER_MAX)

    parser.add_argument('--fill_holes', type=str, default=DEFAULT_FILL_HOLES,
                        choices=['none', 'nearest', 'occlusion_aware'])
    parser.add_argument('--fill_distance', type=int, default=DEFAULT_FILL_DISTANCE)
    parser.add_argument('--depth_discontinuity_threshold', type=float, default=DEFAULT_DEPTH_DISCONTINUITY_THRESHOLD)

    parser.add_argument('--min_depth', type=float, default=DEFAULT_MIN_DEPTH)
    parser.add_argument('--max_depth', type=float, default=DEFAULT_MAX_DEPTH)

    parser.add_argument('--save_vis', action='store_true', default=DEFAULT_SAVE_VIS)
    parser.add_argument('--no_vis', action='store_false', dest='save_vis')

    args = parser.parse_args()

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
    print(f"  Enable distance filter: {args.enable_distance_filter}")
    print(f"  Distance filter camera: {args.distance_filter_camera}")
    print(f"  Distance filter min:    {args.distance_filter_min}")
    print(f"  Distance filter max:    {args.distance_filter_max}")

    target_width = args.target_width
    target_height = args.target_height

    if target_width is not None and target_height is not None:
        print(f"  Target resolution: {target_width}×{target_height} (manual)")
    elif args.rendered_features_dir is not None:
        target_width, target_height = detect_resolution_from_rendered_features(args.rendered_features_dir)
    else:
        print(f"  Target resolution: Original COLMAP resolution (no rescaling)")

    points, classes = load_ply_pointcloud(args.ply_path, args.class_field)
    cameras = load_colmap_cameras(args.colmap_dir, target_width, target_height)

    # Global filtering of points by distance to one reference camera (now switchable)
    if args.enable_distance_filter and (args.distance_filter_camera is not None) and (
        args.distance_filter_min is not None or args.distance_filter_max is not None
    ):
        ref_cam = find_camera_by_name(cameras, args.distance_filter_camera)

        points_before = len(points)
        points, classes, _mask = filter_points_by_camera_distance(
            points, classes, ref_cam,
            min_distance=args.distance_filter_min,
            max_distance=args.distance_filter_max
        )

        print(f"\n{'='*70}")
        print("Distance-based Point Filtering (Global)")
        print(f"{'='*70}")
        print(f"Reference camera: {args.distance_filter_camera}")
        print(f"Min dist (m):     {args.distance_filter_min}")
        print(f"Max dist (m):     {args.distance_filter_max}")
        if points_before > 0:
            print(f"Kept points:      {len(points):,} / {points_before:,} ({len(points)/points_before*100:.1f}%)")
        else:
            print("Kept points:      0 / 0")
    elif not args.enable_distance_filter:
        print(f"\n[INFO] Distance filtering disabled by switch (--disable_distance_filter).")

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

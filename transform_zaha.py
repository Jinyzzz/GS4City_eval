"""
Convert LAZ/LAS point cloud(s) from a locally shifted CRS to
ETRS89 / UTM32N (EPSG:25832) by applying a translation, and optionally
reproject it to another CRS (e.g. Gauss–Krüger EPSG:31468).

The script preserves XYZ coordinates and the classification label,
and writes the result as a binary PLY file with fields: x, y, z, class.

Features:
- Multi-file support: Process multiple LAZ files and merge into one PLY
- Streaming conversion using chunks (no full file loading in memory)
- Robust point counting even if LAS header reports 0
- Live progress bar using tqdm (both per-file and overall)
"""

import argparse
import json
import os
import tempfile
import glob
import numpy as np
import laspy
from collections import Counter
from pyproj import Transformer
from tqdm import tqdm


# ---------- Voxel Grid Downsampling ----------
def voxel_grid_filter(xyz, cls, voxel_size=0.05):
    """
    Apply voxel grid filtering to reduce point density uniformly.

    For each voxel, keeps one point (preferring points from rare classes).

    OPTIMIZED: Uses vectorized operations instead of Python loops.

    Args:
        xyz: (N, 3) array of XYZ coordinates
        cls: (N,) array of classification labels
        voxel_size: size of voxel in meters

    Returns:
        mask: (N,) boolean array indicating which points to keep
    """
    if voxel_size <= 0:
        return np.ones(len(xyz), dtype=bool)

    # Compute voxel indices
    voxel_indices = np.floor(xyz / voxel_size).astype(np.int32)

    # Use hash to identify unique voxels
    # Combine (x, y, z) indices into a single hash
    voxel_hash = (
        voxel_indices[:, 0].astype(np.int64) * 73856093 ^
        voxel_indices[:, 1].astype(np.int64) * 19349663 ^
        voxel_indices[:, 2].astype(np.int64) * 83492791
    )

    # Sort by hash and then by class (to prefer rare classes)
    sort_idx = np.lexsort((cls, voxel_hash))
    sorted_hash = voxel_hash[sort_idx]

    # Find first occurrence of each unique hash
    _, unique_idx = np.unique(sorted_hash, return_index=True)

    # Map back to original indices
    selected_points = sort_idx[unique_idx]

    # Create mask
    mask = np.zeros(len(xyz), dtype=bool)
    mask[selected_points] = True

    return mask


# ---------- Class-Aware Stratified Downsampling ----------
def class_aware_downsample(xyz, cls, class_retention_rates=None, default_rate=0.1):
    """
    Apply class-aware downsampling: keep different retention rates for each class.

    Args:
        xyz: (N, 3) array of XYZ coordinates (not used, kept for API consistency)
        cls: (N,) array of classification labels
        class_retention_rates: dict mapping class_id -> retention_rate (0.0-1.0)
        default_rate: default retention rate for unlisted classes

    Returns:
        mask: (N,) boolean array indicating which points to keep
    """
    if class_retention_rates is None:
        class_retention_rates = {}

    mask = np.zeros(len(cls), dtype=bool)

    for class_id in np.unique(cls):
        class_mask = cls == class_id
        class_points = np.where(class_mask)[0]

        # Get retention rate for this class
        retention_rate = class_retention_rates.get(int(class_id), default_rate)
        retention_rate = np.clip(retention_rate, 0.0, 1.0)

        # Randomly sample points to keep
        n_keep = int(len(class_points) * retention_rate)
        if n_keep > 0:
            keep_indices = np.random.choice(class_points, size=n_keep, replace=False)
            mask[keep_indices] = True

    return mask


# ---------- Uniform Downsampling ----------
def uniform_downsample(xyz, cls, downsample_ratio=0.2):
    """
    Apply uniform random downsampling to all points.

    Args:
        xyz: (N, 3) array of XYZ coordinates (not used, kept for API consistency)
        cls: (N,) array of classification labels (not used, kept for API consistency)
        downsample_ratio: fraction of points to keep (0.0-1.0)

    Returns:
        mask: (N,) boolean array indicating which points to keep
    """
    n_points = len(cls)
    n_keep = int(n_points * np.clip(downsample_ratio, 0.0, 1.0))

    keep_indices = np.random.choice(n_points, size=n_keep, replace=False)
    mask = np.zeros(n_points, dtype=bool)
    mask[keep_indices] = True

    return mask


# ---------- Combined Downsampling ----------
def downsample_points(xyz, cls, strategy='none', voxel_size=0.0,
                     class_retention_rates=None, downsample_ratio=0.2,
                     default_class_rate=0.1):
    """
    Apply downsampling based on strategy.

    Args:
        xyz: (N, 3) array of XYZ coordinates
        cls: (N,) array of classification labels
        strategy: 'none', 'uniform', 'voxel', 'class_aware', 'voxel+class_aware'
        voxel_size: voxel size for voxel grid filtering (meters)
        class_retention_rates: dict for class-aware downsampling
        downsample_ratio: ratio for uniform downsampling
        default_class_rate: default retention rate for unlisted classes in class-aware strategy

    Returns:
        mask: (N,) boolean array indicating which points to keep
        stats: dict with downsampling statistics
    """
    n_original = len(cls)

    if strategy == 'none':
        mask = np.ones(n_original, dtype=bool)
    elif strategy == 'uniform':
        mask = uniform_downsample(xyz, cls, downsample_ratio)
    elif strategy == 'voxel':
        mask = voxel_grid_filter(xyz, cls, voxel_size)
    elif strategy == 'class_aware':
        mask = class_aware_downsample(xyz, cls, class_retention_rates, default_class_rate)
    elif strategy == 'voxel+class_aware':
        # First apply voxel filtering
        mask_voxel = voxel_grid_filter(xyz, cls, voxel_size)
        xyz_voxel = xyz[mask_voxel]
        cls_voxel = cls[mask_voxel]

        # Then apply class-aware downsampling
        mask_class = class_aware_downsample(xyz_voxel, cls_voxel, class_retention_rates, default_class_rate)

        # Combine masks
        mask = np.zeros(n_original, dtype=bool)
        mask[np.where(mask_voxel)[0][mask_class]] = True
    else:
        raise ValueError(f"Unknown downsampling strategy: {strategy}")

    # Compute statistics
    n_kept = np.sum(mask)
    stats = {
        'n_original': n_original,
        'n_kept': n_kept,
        'retention_rate': n_kept / n_original if n_original > 0 else 0.0,
        'class_counts_original': Counter(cls),
        'class_counts_kept': Counter(cls[mask])
    }

    return mask, stats


# ---------- Write PLY header ----------
def write_ply_header(f, n_verts: int):
    """Write a simple binary little-endian PLY header with x, y, z, classification."""
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n_verts}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar scalar_Classification\n"
        "end_header\n"
    )
    f.write(header.encode("ascii"))


# ---------- Robust point count ----------
def robust_point_count(in_path: str, chunk_size: int) -> int:
    """
    Try to obtain the total number of points in a LAZ/LAS file.

    Some files (especially LAS 1.4) may have header.point_count = 0.
    In that case, perform a quick pre-scan using chunk iteration to count points.
    """
    with laspy.open(in_path) as rd:
        n = getattr(rd.header, "point_count", 0) or 0
        if n > 0:
            return int(n)

        print("[INFO] Header reports 0 points. Pre-scanning to count them ...")
        total = 0
        for pts in rd.chunk_iterator(chunk_size):
            total += len(pts)
        print(f"[INFO] Pre-scan total points: {total:,}")
        return total


# ---------- Get all LAZ files from directory or pattern ----------
def get_laz_files(input_spec: str) -> list:
    """
    Get list of LAZ/LAS files from input specification.

    Args:
        input_spec: Can be:
            - A single .laz/.las file
            - A directory containing .laz/.las files
            - A glob pattern (e.g., "/path/*.laz")

    Returns:
        List of absolute file paths
    """
    # Check if it's a single file
    if os.path.isfile(input_spec):
        return [os.path.abspath(input_spec)]

    # Check if it's a directory
    if os.path.isdir(input_spec):
        laz_files = glob.glob(os.path.join(input_spec, "*.laz"))
        las_files = glob.glob(os.path.join(input_spec, "*.las"))
        files = sorted(laz_files + las_files)
        if not files:
            raise ValueError(f"No .laz or .las files found in directory: {input_spec}")
        return files

    # Try as glob pattern
    files = sorted(glob.glob(input_spec))
    if not files:
        raise ValueError(f"No files found matching pattern: {input_spec}")

    # Filter for .laz/.las files only
    files = [f for f in files if f.lower().endswith(('.laz', '.las'))]
    if not files:
        raise ValueError(f"No .laz or .las files found in pattern: {input_spec}")

    return [os.path.abspath(f) for f in files]


# ---------- Main processing ----------
def process_laz_to_ply(
    in_path: str,
    out_path: str,
    shift_xyz=(690826.0, 5335877.0, 500.0),
    target_epsg=25832,
    chunk_size=2_000_000,
    downsample_strategy='none',
    downsample_ratio=0.2,
    voxel_size=0.05,
    class_retention_rates=None,
    default_class_rate=0.1,
    scene_ref_shift=None,
    file_progress_bar=None
):
    """
    Convert LAZ/LAS to PLY with:
    1. Optional downsampling (uniform, voxel, or class-aware)
    2. A fixed translation to restore ETRS89/UTM32 (EPSG:25832),
    3. Optional reprojection to another CRS (if target_epsg != 25832),
    4. Optional scene reference frame transformation (final shift),
    5. Streaming binary PLY writing with progress display.

    Args:
        file_progress_bar: Optional tqdm progress bar for multi-file processing
    """

    # Obtain the total number of points (robustly)
    n_points = robust_point_count(in_path, chunk_size)

    print(f"\n[INFO] Input file: {in_path}")
    print(f"[INFO] Total points (confirmed): {n_points:,}")
    print(f"[INFO] Translation (local → UTM32 / EPSG:25832): {shift_xyz}")
    if target_epsg == 25832:
        print(f"[INFO] No CRS reprojection (staying in EPSG:25832 after translation)")
    else:
        print(f"[INFO] Reprojection: EPSG:25832 → EPSG:{target_epsg}")
    if scene_ref_shift:
        print(f"[INFO] Scene reference shift: {scene_ref_shift}")
    print(f"[INFO] Downsampling strategy: {downsample_strategy}")
    if downsample_strategy != 'none':
        if downsample_strategy == 'uniform':
            print(f"[INFO]   - Uniform ratio: {downsample_ratio:.2%}")
        if 'voxel' in downsample_strategy:
            print(f"[INFO]   - Voxel size: {voxel_size}m")
        if 'class_aware' in downsample_strategy:
            print(f"[INFO]   - Class-aware rates: {class_retention_rates or 'default'}")
            print(f"[INFO]   - Default class rate: {default_class_rate:.2%}")
    print(f"[INFO] Output file: {out_path}\n")

    # Build CRS transformer only if requested
    if target_epsg == 25832:
        transformer = None
    else:
        transformer = Transformer.from_crs(25832, target_epsg, always_xy=True)

    # Define binary layout for each vertex in PLY
    ply_dtype = np.dtype([
        ('x', '<f4'),
        ('y', '<f4'),
        ('z', '<f4'),
        ('class', 'u1')
    ])

    processed_points = 0
    downsampled_points = 0
    total_stats = {
        'class_counts_original': Counter(),
        'class_counts_kept': Counter()
    }

    # Use a temporary file to store binary data, then assemble final PLY with correct header
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.bin') as temp_bin:
        temp_bin_path = temp_bin.name

        # Open LAZ file and process chunks, writing binary data to temp file
        with laspy.open(in_path) as reader, \
             tqdm(
                 total=n_points if n_points > 0 else None,
                 unit="pts",
                 desc="Converting",
                 ncols=100,
                 miniters=1,
                 smoothing=0.1
             ) as pbar:

            for idx, points in enumerate(reader.chunk_iterator(chunk_size), start=1):
                # Convert ScaledArrayView → numpy arrays
                x = np.asarray(points.x, dtype=np.float64)
                y = np.asarray(points.y, dtype=np.float64)
                z = np.asarray(points.z, dtype=np.float64)
                cls = np.asarray(points.classification, dtype=np.uint8)

                processed_points += len(x)

                # Apply downsampling BEFORE coordinate transformation
                if downsample_strategy != 'none':
                    xyz = np.column_stack([x, y, z])
                    mask, chunk_stats = downsample_points(
                        xyz, cls,
                        strategy=downsample_strategy,
                        voxel_size=voxel_size,
                        class_retention_rates=class_retention_rates,
                        downsample_ratio=downsample_ratio,
                        default_class_rate=default_class_rate
                    )

                    # Update global statistics
                    for class_id, count in chunk_stats['class_counts_original'].items():
                        total_stats['class_counts_original'][class_id] += count
                    for class_id, count in chunk_stats['class_counts_kept'].items():
                        total_stats['class_counts_kept'][class_id] += count

                    # Apply mask
                    x = x[mask]
                    y = y[mask]
                    z = z[mask]
                    cls = cls[mask]

                n = len(x)
                if n == 0:
                    pbar.update(chunk_size)
                    continue

                downsampled_points += n

                # (1) Apply translation to get geo-registered UTM32 coordinates (EPSG:25832)
                x += shift_xyz[0]
                y += shift_xyz[1]
                z += shift_xyz[2]

                # (2) Optional reprojection (horizontal only)
                if transformer is None:
                    x_out, y_out, z_out = x, y, z
                else:
                    x_out, y_out = transformer.transform(x, y)
                    z_out = z  # keep vertical value as is

                # (3) Apply scene reference frame shift (if provided)
                if scene_ref_shift is not None:
                    x_out += scene_ref_shift[0]
                    y_out += scene_ref_shift[1]
                    z_out += scene_ref_shift[2]

                # Prepare structured array for binary writing
                vert = np.empty(n, dtype=ply_dtype)
                vert['x'] = x_out.astype(np.float32, copy=False)
                vert['y'] = y_out.astype(np.float32, copy=False)
                vert['z'] = z_out.astype(np.float32, copy=False)
                vert['class'] = cls

                # Write this chunk to the temporary binary file
                temp_bin.write(vert.tobytes())

                pbar.update(len(points.x))  # Update by original chunk size

                # Additional textual progress (in case tqdm output freezes)
                if n_points > 0 and (idx % 10 == 0):
                    pct = 100.0 * processed_points / n_points
                    print(f"[PROGRESS] {processed_points:,} / {n_points:,} ({pct:5.1f}%)")

            # Update file-level progress bar if provided
            if file_progress_bar is not None:
                file_progress_bar.update(1)

    # Now assemble the final PLY file with correct header + binary data
    try:
        with open(out_path, "wb") as fout:
            # Write correct header with actual point count
            write_ply_header(fout, downsampled_points)

            # Append binary data from temporary file
            with open(temp_bin_path, "rb") as temp_in:
                chunk = temp_in.read(8192 * 1024)  # Read in 8MB chunks
                while chunk:
                    fout.write(chunk)
                    chunk = temp_in.read(8192 * 1024)
    finally:
        # Clean up temporary file
        if os.path.exists(temp_bin_path):
            os.remove(temp_bin_path)

    print(f"\n Done! Converted {processed_points:,} points.")
    if downsample_strategy != 'none':
        print(f" Downsampling Statistics:")
        print(f"   - Original points: {processed_points:,}")
        print(f"   - Output points: {downsampled_points:,}")
        print(f"   - Retention rate: {100.0 * downsampled_points / processed_points:.2f}%")
        print(f"\n   Per-class statistics:")
        all_classes = sorted(set(total_stats['class_counts_original'].keys()) |
                           set(total_stats['class_counts_kept'].keys()))
        print(f"   {'Class':<10} {'Original':<12} {'Kept':<12} {'Rate':<10}")
        print(f"   {'-'*10} {'-'*12} {'-'*12} {'-'*10}")
        for class_id in all_classes:
            orig = total_stats['class_counts_original'].get(class_id, 0)
            kept = total_stats['class_counts_kept'].get(class_id, 0)
            rate = 100.0 * kept / orig if orig > 0 else 0.0
            print(f"   {class_id:<10} {orig:<12,} {kept:<12,} {rate:<9.2f}%")

    print(f"\n  Output written to: {out_path}")
    print(f"  Points in output: {downsampled_points:,}")
    print(f"  CRS: EPSG:{target_epsg}")
    if n_points == 0:
        print("  Header contained 0 total points; progress percentages were approximate.")

    # Return statistics for multi-file processing
    return {
        'processed_points': processed_points,
        'downsampled_points': downsampled_points,
        'class_counts_original': total_stats['class_counts_original'],
        'class_counts_kept': total_stats['class_counts_kept']
    }


# ---------- Multi-file processing with merged output ----------
def process_multiple_laz_to_ply(
    input_files: list,
    out_path: str,
    shift_xyz=(690826.0, 5335877.0, 500.0),
    target_epsg=25832,
    chunk_size=2_000_000,
    downsample_strategy='none',
    downsample_ratio=0.2,
    voxel_size=0.05,
    class_retention_rates=None,
    default_class_rate=0.1,
    scene_ref_shift=None
):
    """
    Process multiple LAZ/LAS files and merge them into a single PLY file.

    Args:
        input_files: List of LAZ/LAS file paths
        out_path: Output PLY file path
        ... (other parameters same as process_laz_to_ply)
    """
    print(f"\n{'='*80}")
    print(f" Multi-File LAZ to PLY Conversion")
    print(f"{'='*80}")
    print(f" Input files: {len(input_files)}")
    print(f" Output file: {out_path}")
    print(f" Downsampling strategy: {downsample_strategy}")
    print(f" Translation (local → UTM32 / EPSG:25832): {shift_xyz}")
    if target_epsg == 25832:
        print(f" No CRS reprojection (staying in EPSG:25832 after translation)")
    else:
        print(f" Reprojection: EPSG:25832 → EPSG:{target_epsg}")
    print(f"{'='*80}\n")

    # Display all input files
    total_size = 0
    for i, f in enumerate(input_files, 1):
        size_mb = os.path.getsize(f) / (1024 * 1024)
        total_size += size_mb
        print(f"  [{i:2d}] {os.path.basename(f):<40} ({size_mb:7.1f} MB)")
    print(f"\n  Total size: {total_size:,.1f} MB\n")

    # Build CRS transformer (shared across all files)
    if target_epsg == 25832:
        transformer = None
    else:
        transformer = Transformer.from_crs(25832, target_epsg, always_xy=True)

    # Define binary layout for PLY
    ply_dtype = np.dtype([
        ('x', '<f4'),
        ('y', '<f4'),
        ('z', '<f4'),
        ('class', 'u1')
    ])

    # Global statistics
    global_stats = {
        'total_processed': 0,
        'total_kept': 0,
        'class_counts_original': Counter(),
        'class_counts_kept': Counter()
    }

    # Use a temporary file to accumulate all binary data
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.bin') as temp_bin:
        temp_bin_path = temp_bin.name

        # Process each file with a file-level progress bar
        with tqdm(total=len(input_files), desc="Overall Progress", unit="file", position=0) as file_pbar:
            for file_idx, in_file in enumerate(input_files, 1):
                print(f"\n{'─'*80}")
                print(f"Processing file {file_idx}/{len(input_files)}: {os.path.basename(in_file)}")
                print(f"{'─'*80}")

                # Count points in this file
                n_points = robust_point_count(in_file, chunk_size)
                print(f"[INFO] Points in file: {n_points:,}")

                # Process this file in streaming mode
                processed_points = 0
                downsampled_points = 0
                file_stats = {
                    'class_counts_original': Counter(),
                    'class_counts_kept': Counter()
                }

                with laspy.open(in_file) as reader, \
                     tqdm(
                         total=n_points if n_points > 0 else None,
                         unit="pts",
                         desc=f"  File {file_idx}",
                         ncols=100,
                         position=1,
                         leave=False
                     ) as pbar:

                    for idx, points in enumerate(reader.chunk_iterator(chunk_size), start=1):
                        # Convert to numpy arrays
                        x = np.asarray(points.x, dtype=np.float64)
                        y = np.asarray(points.y, dtype=np.float64)
                        z = np.asarray(points.z, dtype=np.float64)
                        cls = np.asarray(points.classification, dtype=np.uint8)

                        processed_points += len(x)

                        # Apply downsampling
                        if downsample_strategy != 'none':
                            xyz = np.column_stack([x, y, z])
                            mask, chunk_stats = downsample_points(
                                xyz, cls,
                                strategy=downsample_strategy,
                                voxel_size=voxel_size,
                                class_retention_rates=class_retention_rates,
                                downsample_ratio=downsample_ratio,
                                default_class_rate=default_class_rate
                            )

                            # Update file statistics
                            for class_id, count in chunk_stats['class_counts_original'].items():
                                file_stats['class_counts_original'][class_id] += count
                            for class_id, count in chunk_stats['class_counts_kept'].items():
                                file_stats['class_counts_kept'][class_id] += count

                            # Apply mask
                            x = x[mask]
                            y = y[mask]
                            z = z[mask]
                            cls = cls[mask]
                        else:
                            # No downsampling - count all points
                            for class_id, count in Counter(cls).items():
                                file_stats['class_counts_original'][class_id] += count
                                file_stats['class_counts_kept'][class_id] += count

                        n = len(x)
                        if n == 0:
                            pbar.update(chunk_size)
                            continue

                        downsampled_points += n

                        # Apply transformations
                        # (1) Translation (local → EPSG:25832)
                        x += shift_xyz[0]
                        y += shift_xyz[1]
                        z += shift_xyz[2]

                        # (2) Optional reprojection
                        if transformer is None:
                            x_out, y_out, z_out = x, y, z
                        else:
                            x_out, y_out = transformer.transform(x, y)
                            z_out = z

                        # (3) Scene reference shift
                        if scene_ref_shift is not None:
                            x_out += scene_ref_shift[0]
                            y_out += scene_ref_shift[1]
                            z_out += scene_ref_shift[2]

                        # Prepare structured array
                        vert = np.empty(n, dtype=ply_dtype)
                        vert['x'] = x_out.astype(np.float32, copy=False)
                        vert['y'] = y_out.astype(np.float32, copy=False)
                        vert['z'] = z_out.astype(np.float32, copy=False)
                        vert['class'] = cls

                        # Write to temporary file
                        temp_bin.write(vert.tobytes())

                        pbar.update(len(points.x))

                # Update global statistics
                global_stats['total_processed'] += processed_points
                global_stats['total_kept'] += downsampled_points
                for class_id, count in file_stats['class_counts_original'].items():
                    global_stats['class_counts_original'][class_id] += count
                for class_id, count in file_stats['class_counts_kept'].items():
                    global_stats['class_counts_kept'][class_id] += count

                print(f"  ✓ Processed: {processed_points:,} → {downsampled_points:,} points")

                # Update file progress bar
                file_pbar.update(1)

    # Write final PLY file
    print(f"\n{'='*80}")
    print(f" Writing final PLY file...")
    print(f"{'='*80}")

    try:
        with open(out_path, "wb") as fout:
            # Write header with total point count
            write_ply_header(fout, global_stats['total_kept'])

            # Append binary data
            with open(temp_bin_path, "rb") as temp_in:
                chunk = temp_in.read(8192 * 1024)
                while chunk:
                    fout.write(chunk)
                    chunk = temp_in.read(8192 * 1024)
    finally:
        # Clean up temporary file
        if os.path.exists(temp_bin_path):
            os.remove(temp_bin_path)

    # Print final statistics
    print(f"\n{'='*80}")
    print(f" CONVERSION COMPLETE!")
    print(f"{'='*80}")
    print(f" Global Statistics:")
    print(f"   - Input files: {len(input_files)}")
    print(f"   - Total original points: {global_stats['total_processed']:,}")
    print(f"   - Total output points: {global_stats['total_kept']:,}")
    if global_stats['total_processed'] > 0:
        retention = 100.0 * global_stats['total_kept'] / global_stats['total_processed']
        print(f"   - Overall retention rate: {retention:.2f}%")

    print(f"\n Per-Class Statistics:")
    all_classes = sorted(set(global_stats['class_counts_original'].keys()) |
                        set(global_stats['class_counts_kept'].keys()))
    print(f"   {'Class':<10} {'Original':<15} {'Kept':<15} {'Rate':<10}")
    print(f"   {'-'*10} {'-'*15} {'-'*15} {'-'*10}")
    for class_id in all_classes:
        orig = global_stats['class_counts_original'].get(class_id, 0)
        kept = global_stats['class_counts_kept'].get(class_id, 0)
        rate = 100.0 * kept / orig if orig > 0 else 0.0
        print(f"   {class_id:<10} {orig:<15,} {kept:<15,} {rate:<9.2f}%")

    print(f"\n  Output file: {out_path}")
    print(f"  Output size: {os.path.getsize(out_path) / (1024*1024):,.1f} MB")
    print(f"  CRS: EPSG:{target_epsg}")
    print(f"{'='*80}\n")


# ---------- CLI entry ----------
def main():
    parser = argparse.ArgumentParser(
        description="Apply translation to restore ETRS89/UTM32 (EPSG:25832), "
                    "optionally reproject to another CRS and export PLY with progress. "
                    "Supports single file, directory, or glob pattern for multi-file processing."
    )
    parser.add_argument("--in", dest="in_path", default="./zaha/laz_clouds_for_visualisation",
                       help="Input specification: single .laz/.las file, directory, or glob pattern (e.g., '/path/*.laz')")
    parser.add_argument("--out", dest="out_path", default="./outputs/zaha_merged_labeled.ply", help="Output .ply file path")
    parser.add_argument("--shift_x", type=float, default=690826.0, help="Translation (X) in meters")
    parser.add_argument("--shift_y", type=float, default=5335877.0, help="Translation (Y) in meters")
    parser.add_argument("--shift_z", type=float, default=500.0, help="Translation (Z) in meters")
    parser.add_argument("--target_epsg", type=int, default=25832,
                        help="Target CRS EPSG code (default: 25832 = ETRS89 / UTM32N; "
                             "if different from 25832, a reprojection from 25832 will be applied)")
    parser.add_argument("--chunk", type=int, default=2_000_000,
                        help="Number of points per processing chunk")

    # Downsampling options
    parser.add_argument("--downsample_strategy", type=str, default="none",
                        choices=["none", "uniform", "voxel", "class_aware", "voxel+class_aware"],
                        help="Downsampling strategy: none (default), uniform, voxel, class_aware, or voxel+class_aware")
    parser.add_argument("--downsample_ratio", type=float, default=0.2,
                        help="Retention ratio for uniform downsampling (0.0-1.0, default: 0.2)")
    parser.add_argument("--voxel_size", type=float, default=0.05,
                        help="Voxel size in meters for voxel grid filtering (default: 0.05)")
    parser.add_argument("--class_config", type=str, default="config/class_retention_building.json",
                        help="JSON file with class retention rates (for class_aware strategy)")
    parser.add_argument("--default_class_rate", type=float, default=0.1,
                        help="Default retention rate for unlisted classes (default: 0.1)")

    # Scene reference frame transformation
    parser.add_argument("--scene_ref_json", type=str, default="config/scene_reference_frame.json",
                        help="Path to scene_reference_frame.json for final coordinate transformation")

    args = parser.parse_args()

    # Load class retention rates from JSON if provided
    class_retention_rates = None
    if args.class_config:
        try:
            with open(args.class_config, 'r') as f:
                class_retention_rates = json.load(f)
            # Convert string keys to integers, skip metadata fields (keys starting with '_')
            class_retention_rates = {
                int(k): float(v)
                for k, v in class_retention_rates.items()
                if not k.startswith('_')
            }
            print(f"[INFO] Loaded class retention rates from {args.class_config}")
            print(f"[INFO]   Rates: {class_retention_rates}")
        except Exception as e:
            print(f"[WARNING] Failed to load class config: {e}")
            print(f"[WARNING] Using default retention rate for all classes")

    # Load scene reference frame transformation if provided
    scene_ref_shift = None
    if args.scene_ref_json:
        try:
            with open(args.scene_ref_json, 'r') as f:
                scene_ref = json.load(f)
            scene_ref_shift = scene_ref['base_to_canonical']['shift']
            print(f"[INFO] Loaded scene reference shift from {args.scene_ref_json}")
            print(f"[INFO]   Shift: {scene_ref_shift}")
        except Exception as e:
            print(f"[WARNING] Failed to load scene reference frame: {e}")
            print(f"[WARNING] Skipping scene reference transformation")

    # Get list of input files
    try:
        input_files = get_laz_files(args.in_path)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return

    # Choose processing mode based on number of input files
    if len(input_files) == 1:
        # Single file mode - use original function with detailed progress
        print(f"[INFO] Single file mode: {input_files[0]}")
        process_laz_to_ply(
            in_path=input_files[0],
            out_path=args.out_path,
            shift_xyz=(args.shift_x, args.shift_y, args.shift_z),
            target_epsg=args.target_epsg,
            chunk_size=args.chunk,
            downsample_strategy=args.downsample_strategy,
            downsample_ratio=args.downsample_ratio,
            voxel_size=args.voxel_size,
            class_retention_rates=class_retention_rates,
            default_class_rate=args.default_class_rate,
            scene_ref_shift=scene_ref_shift
        )
    else:
        # Multi-file mode - use batch processing function
        print(f"[INFO] Multi-file mode: {len(input_files)} files")
        process_multiple_laz_to_ply(
            input_files=input_files,
            out_path=args.out_path,
            shift_xyz=(args.shift_x, args.shift_y, args.shift_z),
            target_epsg=args.target_epsg,
            chunk_size=args.chunk,
            downsample_strategy=args.downsample_strategy,
            downsample_ratio=args.downsample_ratio,
            voxel_size=args.voxel_size,
            class_retention_rates=class_retention_rates,
            default_class_rate=args.default_class_rate,
            scene_ref_shift=scene_ref_shift
        )


if __name__ == "__main__":
    main()

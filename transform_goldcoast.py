import argparse
import os
import tempfile
import glob
import numpy as np
import laspy
from pyproj import Transformer
from tqdm import tqdm


# -----------------------------
# PLY header
# -----------------------------
def write_ply_header(f, n_verts: int):
    """
    Write binary little-endian PLY header.
    Output fields:
      - x
      - y
      - z
      - scalar_Classification
    """
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


# -----------------------------
# Robust point count
# -----------------------------
def robust_point_count(in_path: str, chunk_size: int) -> int:
    """
    Safely obtain total point count.
    Some LAS/LAZ headers may report 0 points, so we pre-scan if needed.
    """
    with laspy.open(in_path) as rd:
        n = getattr(rd.header, "point_count", 0) or 0
        if n > 0:
            return int(n)

        print(f"[INFO] Header reports 0 points for: {in_path}")
        print("[INFO] Pre-scanning file to count points...")
        total = 0
        for pts in rd.chunk_iterator(chunk_size):
            total += len(pts)
        print(f"[INFO] Pre-scan total points: {total:,}")
        return total


# -----------------------------
# Input file resolver
# -----------------------------
def get_laz_files(input_spec: str) -> list:
    """
    Resolve input specification into a list of LAS/LAZ files.

    input_spec can be:
      - single .laz/.las file
      - directory
      - glob pattern, e.g. "./data/*.laz"
    """
    if os.path.isfile(input_spec):
        return [os.path.abspath(input_spec)]

    if os.path.isdir(input_spec):
        laz_files = glob.glob(os.path.join(input_spec, "*.laz"))
        las_files = glob.glob(os.path.join(input_spec, "*.las"))
        files = sorted(laz_files + las_files)
        if not files:
            raise ValueError(f"No .laz or .las files found in directory: {input_spec}")
        return [os.path.abspath(f) for f in files]

    files = sorted(glob.glob(input_spec))
    if not files:
        raise ValueError(f"No files found matching pattern: {input_spec}")

    files = [f for f in files if f.lower().endswith((".laz", ".las"))]
    if not files:
        raise ValueError(f"No .laz or .las files found in pattern: {input_spec}")

    return [os.path.abspath(f) for f in files]


# -----------------------------
# Class remapping
# -----------------------------
def remap_classification(classification: np.ndarray) -> np.ndarray:
    """
    Remap classes before writing to PLY.

    Mapping:
      - original 2 -> 12
      - original 3 -> 2
      - others unchanged
    """
    cls_out = classification.copy()
    cls_out[classification == 2] = 12
    cls_out[classification == 3] = 2
    return cls_out


# -----------------------------
# Core point transform
# -----------------------------
def transform_points(
    x,
    y,
    z,
    transformer,
    z_foot_to_meter=True,
    scale=(1.0, 1.0, 1.0),
    shift=(-435158.0, -4593748.0, -316.0),
    extra_z_shift=0.0
):
    """
    Coordinate transformation pipeline:

    1) Transform XY from EPSG:6549 -> EPSG:32617
    2) Convert Z from foot -> meter
    3) Apply scale
    4) Apply shift
    5) Apply extra user-defined Z shift

    Returns:
        x_out, y_out, z_out
    """
    # Step 1: XY reprojection
    x_out, y_out = transformer.transform(x, y)

    # Step 2: Z unit conversion
    if z_foot_to_meter:
        z_out = z * 0.3048
    else:
        z_out = z.copy()

    # Step 3: scale
    x_out = x_out * scale[0]
    y_out = y_out * scale[1]
    z_out = z_out * scale[2]

    # Step 4: shift
    x_out = x_out + shift[0]
    y_out = y_out + shift[1]
    z_out = z_out + shift[2]

    # Step 5: extra user-defined z shift
    z_out = z_out + extra_z_shift

    return x_out, y_out, z_out


# -----------------------------
# Single file conversion
# -----------------------------
def process_laz_to_ply(
    in_path: str,
    out_path: str,
    chunk_size=2_000_000,
    source_epsg=6549,
    target_epsg=32617,
    z_foot_to_meter=True,
    scale=(1.0, 1.0, 1.0),
    shift=(-435158.0, -4593748.0, -316.0),
    extra_z_shift=0.0,
    file_progress_bar=None
):
    """
    Convert one LAS/LAZ file to binary PLY in streaming mode.
    """
    n_points = robust_point_count(in_path, chunk_size)

    print(f"\n[INFO] Input file: {in_path}")
    print(f"[INFO] Total points (confirmed): {n_points:,}")
    print(f"[INFO] CRS transform: EPSG:{source_epsg} -> EPSG:{target_epsg}")
    print(f"[INFO] Z conversion: {'foot -> meter' if z_foot_to_meter else 'disabled'}")
    print(f"[INFO] Scale: {scale}")
    print(f"[INFO] Shift: {shift}")
    print(f"[INFO] Extra Z shift: {extra_z_shift}")
    print(f"[INFO] Class remapping: 2 -> 12, 3 -> 2")
    print(f"[INFO] Output file: {out_path}\n")

    transformer = Transformer.from_crs(source_epsg, target_epsg, always_xy=True)

    ply_dtype = np.dtype([
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("scalar_Classification", "u1")
    ])

    processed_points = 0

    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".bin") as temp_bin:
        temp_bin_path = temp_bin.name

        with laspy.open(in_path) as reader, tqdm(
            total=n_points if n_points > 0 else None,
            unit="pts",
            desc="Converting",
            ncols=100,
            miniters=1,
            smoothing=0.1
        ) as pbar:

            for idx, points in enumerate(reader.chunk_iterator(chunk_size), start=1):
                x = np.asarray(points.x, dtype=np.float64)
                y = np.asarray(points.y, dtype=np.float64)
                z = np.asarray(points.z, dtype=np.float64)

                if not hasattr(points, "classification"):
                    raise AttributeError(
                        f"Input file does not contain 'classification' dimension: {in_path}"
                    )

                classification = np.asarray(points.classification, dtype=np.uint8)
                classification = remap_classification(classification)

                processed_points += len(x)
                n = len(x)

                if n == 0:
                    pbar.update(0)
                    continue

                x_out, y_out, z_out = transform_points(
                    x=x,
                    y=y,
                    z=z,
                    transformer=transformer,
                    z_foot_to_meter=z_foot_to_meter,
                    scale=scale,
                    shift=shift,
                    extra_z_shift=extra_z_shift
                )

                vert = np.empty(n, dtype=ply_dtype)
                vert["x"] = x_out.astype(np.float32, copy=False)
                vert["y"] = y_out.astype(np.float32, copy=False)
                vert["z"] = z_out.astype(np.float32, copy=False)
                vert["scalar_Classification"] = classification

                temp_bin.write(vert.tobytes())
                pbar.update(len(points.x))

                if n_points > 0 and (idx % 10 == 0):
                    pct = 100.0 * processed_points / n_points
                    print(f"[PROGRESS] {processed_points:,} / {n_points:,} ({pct:5.1f}%)")

            if file_progress_bar is not None:
                file_progress_bar.update(1)

    try:
        with open(out_path, "wb") as fout:
            write_ply_header(fout, processed_points)
            with open(temp_bin_path, "rb") as temp_in:
                chunk = temp_in.read(8192 * 1024)
                while chunk:
                    fout.write(chunk)
                    chunk = temp_in.read(8192 * 1024)
    finally:
        if os.path.exists(temp_bin_path):
            os.remove(temp_bin_path)

    print(f"\n[DONE] Converted {processed_points:,} points.")
    print(f"[DONE] Output written to: {out_path}")
    print(f"[DONE] Points in output: {processed_points:,}")
    print(f"[DONE] Final horizontal CRS: EPSG:{target_epsg}")


# -----------------------------
# Multi-file merge conversion
# -----------------------------
def process_multiple_laz_to_ply(
    input_files: list,
    out_path: str,
    chunk_size=2_000_000,
    source_epsg=6549,
    target_epsg=32617,
    z_foot_to_meter=True,
    scale=(1.0, 1.0, 1.0),
    shift=(-435158.0, -4593748.0, -316.0),
    extra_z_shift=0.0
):
    """
    Process multiple LAS/LAZ files and merge them into a single PLY.
    """
    print(f"\n{'='*80}")
    print(" Multi-File LAZ/LAS -> PLY Conversion")
    print(f"{'='*80}")
    print(f" Input files: {len(input_files)}")
    print(f" Output file: {out_path}")
    print(f" CRS transform: EPSG:{source_epsg} -> EPSG:{target_epsg}")
    print(f" Z conversion: {'foot -> meter' if z_foot_to_meter else 'disabled'}")
    print(f" Scale: {scale}")
    print(f" Shift: {shift}")
    print(f" Extra Z shift: {extra_z_shift}")
    print(f" Class remapping: 2 -> 12, 3 -> 2")
    print(f"{'='*80}\n")

    total_size = 0
    for i, f in enumerate(input_files, 1):
        size_mb = os.path.getsize(f) / (1024 * 1024)
        total_size += size_mb
        print(f"  [{i:2d}] {os.path.basename(f):<40} ({size_mb:8.1f} MB)")
    print(f"\n  Total size: {total_size:,.1f} MB\n")

    transformer = Transformer.from_crs(source_epsg, target_epsg, always_xy=True)

    ply_dtype = np.dtype([
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("scalar_Classification", "u1")
    ])

    total_processed = 0

    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".bin") as temp_bin:
        temp_bin_path = temp_bin.name

        with tqdm(total=len(input_files), desc="Overall Progress", unit="file", position=0) as file_pbar:
            for file_idx, in_file in enumerate(input_files, 1):
                print(f"\n{'─'*80}")
                print(f"Processing file {file_idx}/{len(input_files)}: {os.path.basename(in_file)}")
                print(f"{'─'*80}")

                n_points = robust_point_count(in_file, chunk_size)
                print(f"[INFO] Points in file: {n_points:,}")

                processed_points = 0

                with laspy.open(in_file) as reader, tqdm(
                    total=n_points if n_points > 0 else None,
                    unit="pts",
                    desc=f"  File {file_idx}",
                    ncols=100,
                    position=1,
                    leave=False
                ) as pbar:

                    for idx, points in enumerate(reader.chunk_iterator(chunk_size), start=1):
                        x = np.asarray(points.x, dtype=np.float64)
                        y = np.asarray(points.y, dtype=np.float64)
                        z = np.asarray(points.z, dtype=np.float64)

                        if not hasattr(points, "classification"):
                            raise AttributeError(
                                f"Input file does not contain 'classification' dimension: {in_file}"
                            )

                        classification = np.asarray(points.classification, dtype=np.uint8)
                        classification = remap_classification(classification)

                        n = len(x)
                        processed_points += n

                        if n == 0:
                            pbar.update(0)
                            continue

                        x_out, y_out, z_out = transform_points(
                            x=x,
                            y=y,
                            z=z,
                            transformer=transformer,
                            z_foot_to_meter=z_foot_to_meter,
                            scale=scale,
                            shift=shift,
                            extra_z_shift=extra_z_shift
                        )

                        vert = np.empty(n, dtype=ply_dtype)
                        vert["x"] = x_out.astype(np.float32, copy=False)
                        vert["y"] = y_out.astype(np.float32, copy=False)
                        vert["z"] = z_out.astype(np.float32, copy=False)
                        vert["scalar_Classification"] = classification

                        temp_bin.write(vert.tobytes())
                        pbar.update(len(points.x))

                total_processed += processed_points
                print(f"  ✓ Processed: {processed_points:,} points")
                file_pbar.update(1)

    print(f"\n{'='*80}")
    print(" Writing final PLY file...")
    print(f"{'='*80}")

    try:
        with open(out_path, "wb") as fout:
            write_ply_header(fout, total_processed)
            with open(temp_bin_path, "rb") as temp_in:
                chunk = temp_in.read(8192 * 1024)
                while chunk:
                    fout.write(chunk)
                    chunk = temp_in.read(8192 * 1024)
    finally:
        if os.path.exists(temp_bin_path):
            os.remove(temp_bin_path)

    print(f"\n{'='*80}")
    print(" CONVERSION COMPLETE!")
    print(f"{'='*80}")
    print(f" Total input files: {len(input_files)}")
    print(f" Total output points: {total_processed:,}")
    print(f" Output file: {out_path}")
    print(f" Output size: {os.path.getsize(out_path) / (1024*1024):,.1f} MB")
    print(f" Final horizontal CRS: EPSG:{target_epsg}")
    print(f"{'='*80}\n")


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert LAZ/LAS point clouds from EPSG:6549 to EPSG:32617, "
            "convert Z from foot to meter, apply fixed scale/shift and "
            "an extra user-defined Z shift, remap classes, preserve classification, "
            "and export to binary PLY."
        )
    )

    parser.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help="Input specification: single .laz/.las file, directory, or glob pattern"
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        required=True,
        help="Output .ply file path"
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=2_000_000,
        help="Number of points per processing chunk"
    )

    # CRS
    parser.add_argument(
        "--source_epsg",
        type=int,
        default=6549,
        help="Source CRS EPSG code (default: 6549)"
    )
    parser.add_argument(
        "--target_epsg",
        type=int,
        default=32617,
        help="Target CRS EPSG code (default: 32617)"
    )

    # Z conversion
    parser.add_argument(
        "--z_foot_to_meter",
        action="store_true",
        help="Convert Z from foot to meter (default behavior if not disabled explicitly in code path)"
    )
    parser.add_argument(
        "--no_z_foot_to_meter",
        action="store_true",
        help="Disable Z foot-to-meter conversion"
    )

    # Scale
    parser.add_argument("--scale_x", type=float, default=1.0, help="Scale X")
    parser.add_argument("--scale_y", type=float, default=1.0, help="Scale Y")
    parser.add_argument("--scale_z", type=float, default=1.0, help="Scale Z")

    # Shift
    parser.add_argument("--shift_x", type=float, default=-435158.0, help="Shift X")
    parser.add_argument("--shift_y", type=float, default=-4593748.0, help="Shift Y")
    parser.add_argument("--shift_z", type=float, default=-316.0, help="Shift Z")

    # Extra user-defined z shift
    parser.add_argument(
        "--extra_z_shift",
        type=float,
        default=0.0,
        help="Additional Z shift applied after all other transformations"
    )

    args = parser.parse_args()

    z_foot_to_meter = True
    if args.no_z_foot_to_meter:
        z_foot_to_meter = False
    elif args.z_foot_to_meter:
        z_foot_to_meter = True

    scale = (args.scale_x, args.scale_y, args.scale_z)
    shift = (args.shift_x, args.shift_y, args.shift_z)

    try:
        input_files = get_laz_files(args.in_path)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return

    out_dir = os.path.dirname(os.path.abspath(args.out_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if len(input_files) == 1:
        print(f"[INFO] Single file mode: {input_files[0]}")
        process_laz_to_ply(
            in_path=input_files[0],
            out_path=args.out_path,
            chunk_size=args.chunk,
            source_epsg=args.source_epsg,
            target_epsg=args.target_epsg,
            z_foot_to_meter=z_foot_to_meter,
            scale=scale,
            shift=shift,
            extra_z_shift=args.extra_z_shift
        )
    else:
        print(f"[INFO] Multi-file mode: {len(input_files)} files")
        process_multiple_laz_to_ply(
            input_files=input_files,
            out_path=args.out_path,
            chunk_size=args.chunk,
            source_epsg=args.source_epsg,
            target_epsg=args.target_epsg,
            z_foot_to_meter=z_foot_to_meter,
            scale=scale,
            shift=shift,
            extra_z_shift=args.extra_z_shift
        )


if __name__ == "__main__":
    main()
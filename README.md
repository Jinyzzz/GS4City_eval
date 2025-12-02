# ZAHA Semantic Segmentation Evaluation Pipeline

This repository contains scripts for evaluating semantic segmentation quality on the ZAHA architectural dataset, supporting both 2D (image-based) and 3D (point cloud) evaluation of LangSplat outputs.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Data Preparation](#data-preparation)
- [Evaluation Pipeline](#evaluation-pipeline)
  - [Step 1: Point Cloud Preprocessing](#step-1-point-cloud-preprocessing)
  - [Step 2: Manual Cropping (Optional)](#step-2-manual-cropping-optional)
  - [Step 3: 2D Ground Truth Projection](#step-3-2d-ground-truth-projection)
  - [Step 4: 2D Evaluation](#step-4-2d-evaluation)
  - [Step 5: 3D Evaluation](#step-5-3d-evaluation)
  - [Optional: Error Visualization](#optional-error-visualization)
- [Configuration Files](#configuration-files)
- [Directory Structure](#directory-structure)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

## Overview

The evaluation pipeline transforms raw LAZ point clouds from the ZAHA dataset into labeled ground truth, projects them to 2D camera views, and evaluates semantic segmentation results from LangSplat in both 2D and 3D spaces.

**Pipeline Workflow:**
1. **Preprocessing**: Convert and downsample LAZ point clouds with coordinate transformation
2. **Cropping** (optional): Manually crop large scenes for faster processing
3. **2D GT Generation**: Project 3D labeled points to 2D camera views
4. **2D Evaluation**: Compare rendered 2D segmentation with ground truth
5. **3D Evaluation**: Direct comparison on 3D Gaussian point clouds

## Prerequisites

### Python Dependencies

```bash
pip install numpy torch torchvision
pip install open-clip-torch
pip install laspy plyfile scipy pyproj
pip install opencv-python pillow matplotlib
pip install tqdm
```

### Required Data

- **ZAHA Dataset**: Download from [https://syncandshare.lrz.de/getlink/fi7J6RXToWkf8xdrAzLCwe/ZAHA](https://syncandshare.lrz.de/getlink/fi7J6RXToWkf8xdrAzLCwe/ZAHA)
  - Extract `laz_clouds_for_visualisation.zip` - contains classified point clouds
  - Coordinate system documentation available in the dataset
- **LangSplat trained model** with language features enabled
- **COLMAP sparse reconstruction** of the scene (camera parameters)
- **Autoencoder checkpoint** for feature compression/decompression

## Data Preparation

### Configuration Files

This repository includes several JSON configuration files that control the evaluation pipeline:

- **`scene_reference_frame.json`**: Coordinate system metadata (CRS definition, shift values, scale)
  - Defines transformation from local coordinates to UTM32 (EPSG:25832) and target CRS
  - Based on ZAHA dataset documentation

- **`class_mapping.json`**: Maps class IDs to natural language descriptions
  - Supports multiple text prompts per class for better CLIP matching
  - Example: `"2": ["window transparent opening", "window clear glazed panel", ...]`

- **`class_colors.json`**: RGB color values for each semantic class (used in visualizations)

- **`class_retention_building.json`**: Class-specific retention rates for downsampling
  - Higher rates for rare/important classes (windows, doors)
  - Lower rates for abundant classes (walls, ground)

- **`window_merge.json`**: Class merge rules
  - Example: `{"2": [2, 13]}` merges class 13 (window blinds) into class 2 (windows)

## Evaluation Pipeline

### Step 1: Point Cloud Preprocessing

**Script**: `transform_zaha.py`

**Functionality**:
- Converts LAZ/LAS point clouds to PLY format
- Applies coordinate transformations: local → UTM32 (EPSG:25832) → target CRS (default: EPSG:32632)
- Downsampling strategies:
  - `voxel`: Voxel grid filtering (uniform spatial downsampling)
  - `uniform`: Random uniform downsampling
  - `class_aware`: Class-specific retention rates (recommended)
  - `voxel+class_aware`: Combined voxel + class-aware (best quality)
- Supports multi-file merging with progress tracking
- Streaming processing for large files (no memory overflow)

**Inputs**:
- LAZ/LAS point cloud files from `laz_clouds_for_visualisation.zip`
- `scene_reference_frame.json` (coordinate transformation parameters)
- `class_retention_building.json` (optional, for class-aware downsampling)

**Outputs**:
- PLY file with fields: `x`, `y`, `z`, `scalar_Classification`
- Console output showing downsampling statistics (per-class retention rates)

**Recommended Settings**:
- Use `class_aware` or `voxel+class_aware` downsampling
- Adjust `voxel_size` based on scene density (0.05m recommended)
- Adjust per-class retention rates in `class_retention_building.json`

**Example Command**:

```bash
python transform_zaha.py \
    --in ./zaha/laz_clouds_for_visualisation \
    --out ./outputs/zaha_merged_labeled.ply \
    --shift_x 690826.0 \
    --shift_y 5335877.0 \
    --shift_z 500.0 \
    --target_epsg 32632 \
    --downsample_strategy class_aware \
    --voxel_size 0.05 \
    --class_config class_retention_building.json \
    --scene_ref_json scene_reference_frame.json
```

**Key Parameters**:
- `--in`: Input LAZ file, directory, or glob pattern
- `--shift_x/y/z`: Translation to restore geo-registration (from ZAHA docs)
- `--target_epsg`: Target coordinate system (32632 = WGS84/UTM32N)
- `--downsample_strategy`: Downsampling method (none, uniform, voxel, class_aware, voxel+class_aware)
- `--voxel_size`: Voxel size in meters for voxel-based downsampling
- `--class_config`: JSON file with per-class retention rates
- `--default_class_rate`: Default retention rate for unlisted classes (default: 0.1)

---

### Step 2: Manual Cropping (Optional)

**Purpose**: Improve computation speed for `project_2d_gt.py` and evaluation scripts

**Process**:
For large scenes, manually crop the full labeled point cloud to the region covered by the COLMAP reconstruction. This significantly reduces processing time without affecting evaluation quality.

**Tools**: Use CloudCompare, MeshLab, or similar point cloud software to:
1. Load the full scene PLY from Step 1
2. Identify the region covered by camera views
3. Crop to bounding box
4. Save as cropped PLY

**Input**: `zaha_merged_labeled.ply` (from Step 1)
**Output**: `zaha_cropped_labeled.ply`

---

### Step 3: 2D Ground Truth Projection

**Script**: `project_2d_gt.py`

**Functionality**:
- Projects 3D labeled point cloud to 2D semantic segmentation maps per camera view
- Uses COLMAP camera parameters (intrinsics and extrinsics)
- Depth buffering to handle occlusion correctly
- Optional hole filling (nearest neighbor or occlusion-aware)
- Generates visualization PNGs for verification
- Outputs statistics (coverage, class distribution)

**Inputs**:
- Labeled PLY point cloud (from Step 1, optionally cropped in Step 2)
- COLMAP sparse reconstruction directory (contains `cameras.bin`, `images.bin`, `points3D.bin`)
- `class_colors.json` (for visualization)

**Outputs**:
- `{image_name}.npy`: Ground truth semantic map (H, W) int32 array
  - `-1` = background (no point cloud coverage)
  - `0, 1, 2, ...` = class IDs
- `{image_name}_vis.png`: Colored visualization of semantic map
- `statistics.json`: Coverage statistics and class distribution
- `class_colors.json`: Copy of color mapping used

**Example Command**:

```bash
python project_2d_gt.py \
    --ply_path outputs/zaha_cropped_labeled.ply \
    --colmap_dir ../data/building1_15/sparse/0 \
    --output_dir outputs/gt_semantic_maps \
    --fill_holes nearest \
    --save_vis
```

**Key Parameters**:
- `--ply_path`: Path to labeled PLY point cloud
- `--colmap_dir`: COLMAP sparse reconstruction directory (contains cameras.bin, images.bin)
- `--output_dir`: Output directory for ground truth maps
- `--fill_holes`: Hole filling strategy (none, nearest, occlusion_aware)
- `--save_vis`: Save visualization PNGs

**Output Format**:
Ground truth `.npy` files have shape `(H, W)` with `dtype=int32`:
- Background pixels: `-1`
- Valid class labels: `0, 1, 2, ..., 16`
- Coverage mask computed as `(semantic_map >= 0)`

---

### Step 4: 2D Evaluation

**Script**: `evaluate_2d.py`

**Functionality**:
- Evaluates rendered semantic segmentation against projected 2D ground truth
- **Feature decoding pipeline**:
  1. Loads compressed features from LangSplat rendering (.npy files)
  2. Decodes to 512-dim CLIP space using autoencoder
  3. Queries CLIP with semantic class descriptions from `class_mapping.json`
- Computes metrics: **IoU, Accuracy, Precision, Recall, F1** (per-class and overall)
- Supports partial coverage evaluation (only evaluates where GT exists)
- Optional class merging (e.g., merge window blinds into windows)
- Generates confusion matrix and visualization images

**Inputs**:
- Rendered feature maps (.npy files from LangSplat, typically in `output/scene/test/ours_None/renders_npy/`)
- 2D ground truth semantic maps (from Step 3)
- Autoencoder checkpoint (.pth file)
- `class_mapping.json` (class descriptions for CLIP text encoder)
- `class_colors.json` (for visualization)
- `window_merge.json` (optional, for class merging)

**Outputs**:
- `metrics.json`: Per-class and overall metrics (IoU, accuracy, precision, recall, F1)
- `confusion_matrix.png`: Confusion matrix heatmap
- `per_class_metrics.csv`: Detailed per-class statistics
- `vis_{image_name}.png`: Visualization images (GT vs Prediction)
- `eval.log`: Detailed evaluation log

**Example Command**:

```bash
python evaluate_2d.py \
    --rendered_features ../output/building1_15/test/ours_None/renders_npy \
    --gt_semantic_dir outputs/gt_semantic_maps \
    --ae_checkpoint autoencoder/ckpt/building1_15/best_ckpt.pth \
    --class_mapping class_mapping.json \
    --class_colors class_colors.json \
    --merge_mapping window_merge.json \
    --output_dir eval_results/2d \
    --mask_thresh 0.5
```

**Key Parameters**:
- `--rendered_features`: Directory containing rendered feature .npy files
- `--gt_semantic_dir`: Directory with ground truth .npy files from Step 3
- `--ae_checkpoint`: Autoencoder checkpoint for feature decoding
- `--class_mapping`: JSON mapping class IDs to text descriptions
- `--merge_mapping`: Optional JSON for merging classes
- `--mask_thresh`: Minimum GT coverage threshold for including a pixel (default: 0.5)
- `--output_dir`: Directory for evaluation results

**Notes**:
- Feature files must match GT files by name (e.g., `IMG_001.npy`)
- Autoencoder decodes compressed features (e.g., 3-dim or 256-dim) back to 512-dim CLIP embeddings
- CLIP queries use all text prompts from `class_mapping.json` and selects best match

---

### Step 5: 3D Evaluation

**Script**: `evaluate_3d.py`

**Functionality**:
- Direct evaluation on 3D Gaussian point clouds (no 2D projection)
- Loads LangSplat checkpoint containing:
  - 3D Gaussian positions (xyz)
  - Compressed language features (3-dim or 256-dim)
- **Feature decoding pipeline** (same as 2D):
  1. Extracts compressed features from checkpoint
  2. Decodes to 512-dim CLIP embeddings using autoencoder
  3. Queries CLIP with text descriptions to get semantic labels
- **Nearest-neighbor matching**: Matches predicted Gaussians to GT points using KDTree
- Computes 3D metrics: **IoU, Accuracy, Precision, Recall, F1** (per-class and overall)
- Generates colored PLY files for visualization

**Inputs**:
- LangSplat checkpoint (.pth file with Gaussian parameters and language features)
- Ground truth labeled point cloud (from Step 1)
- Autoencoder checkpoint (.pth)
- `class_mapping.json`
- `class_colors.json`
- `window_merge.json` (optional)

**Outputs**:
- `metrics_3d.json`: Per-class and overall 3D metrics
- `confusion_matrix_3d.png`: 3D confusion matrix heatmap
- `gt_semantic_colored.ply`: Ground truth point cloud with class colors
- `predicted_semantic.ply`: Predicted Gaussian positions with class colors
- `matched_predictions.ply`: Predictions matched to GT points (for error analysis)
- `eval_3d.log`: Detailed evaluation log with statistics

**Example Command**:

```bash
python evaluate_3d.py \
    --checkpoint ../output/building1_15_dual_eval_1/chkpnt30000.pth \
    --ae_checkpoint autoencoder/ckpt/building1_15/best_ckpt.pth \
    --gt_pointcloud outputs/zaha_merged_labeled.ply \
    --class_mapping class_mapping.json \
    --class_colors class_colors.json \
    --merge_mapping window_merge.json \
    --output_dir eval_results/3d \
    --max_distance 0.5
```

**Key Parameters**:
- `--checkpoint`: LangSplat checkpoint with Gaussian parameters and features
- `--ae_checkpoint`: Autoencoder checkpoint for feature decoding
- `--gt_pointcloud`: Ground truth labeled PLY (from Step 1)
- `--class_mapping`: JSON mapping class IDs to text descriptions
- `--class_colors`: JSON with RGB colors for visualization
- `--merge_mapping`: Optional JSON for merging classes
- `--max_distance`: Maximum matching distance in meters (default: 0.5m)
  - Points beyond this threshold are excluded from evaluation
- `--output_dir`: Directory for evaluation results

**Checkpoint Format**:
LangSplat checkpoints must contain language features. Checkpoint structure:
```python
checkpoint_data = torch.load(checkpoint_path)
model_params, iteration = checkpoint_data
# model_params[7] = language_feature (N, 3) or (N, 256)
```

**Notes**:
- Use `--max_distance` to filter spurious matches (0.5m works well for building-scale scenes)
- GT point cloud should use the same coordinate system as the trained LangSplat model
- Evaluation only includes points within `max_distance` of GT points

---

### Optional: Error Visualization

**Script**: `generate_error_map_pointcloud.py`

**Functionality**:
- Generates 3D error map from evaluation results
- Color-codes points by prediction correctness:
  - **Green**: Correct predictions
  - **Red**: Incorrect predictions
- Uses KDTree nearest-neighbor matching between GT and predictions
- Outputs error statistics (overall accuracy, per-class errors)

**Inputs**:
- `gt_semantic_colored.ply` (from Step 5: `evaluate_3d.py`)
- `predicted_semantic.ply` (from Step 5: `evaluate_3d.py`)

**Outputs**:
- `error_map.ply`: Point cloud with error colors (green/red)
- `error_statistics.json`: Per-class error rates and confusion types

**Example Command**:

```bash
python generate_error_map_pointcloud.py \
    --gt_pointcloud eval_results/3d/gt_semantic_colored.ply \
    --pred_pointcloud eval_results/3d/predicted_semantic.ply \
    --output_dir eval_results/3d \
    --max_distance 0.5
```

**Key Parameters**:
- `--gt_pointcloud`: GT point cloud with colored classes
- `--pred_pointcloud`: Predicted point cloud with colored classes
- `--max_distance`: Maximum matching distance (should match Step 5)
- `--output_dir`: Directory for error map output

**Visualization**:
Open `error_map.ply` in CloudCompare or MeshLab to visualize spatial distribution of errors.

---

## Configuration Files

### class_mapping.json

Maps class IDs (integers) to natural language descriptions used for CLIP text encoding. Supports multiple prompts per class for better semantic matching.

**Format**:
```json
{
  "0": "void empty background",
  "1": [
    "plain vertical exterior wall surface",
    "large stone or plaster facade plane",
    "continuous vertical building wall not molding"
  ],
  "2": [
    "window transparent opening",
    "window clear glazed panel",
    "dark reflective glass window panel"
  ]
}
```

**Usage**: All prompts for a class are used as separate CLIP queries, and the best match determines the prediction.

---

### class_colors.json

RGB color values (0-255) for each class, used in all visualization outputs.

**Format**:
```json
{
  "0": [207, 207, 207],
  "1": [255, 242, 204],
  "2": [142, 169, 219],
  "-1": [0, 0, 0]
}
```

---

### class_retention_building.json

Per-class retention rates for `class_aware` downsampling. Higher values retain more points for that class.

**Format**:
```json
{
  "_comment": "Class retention rates for building point cloud downsampling",
  "12": 0.9,   // Rare class - high retention
  "3": 0.9,    // Windows - important class
  "6": 0.4,    // Moderate retention
  "1": 0.05    // Very common class (ground) - low retention
}
```

**Rationale**: Rare or semantically important classes get higher retention to maintain evaluation quality.

---

### window_merge.json

Defines class merging rules (source classes → target class). Applied during evaluation.

**Format**:
```json
{
  "2": [2, 13]
}
```

This merges class 13 (window blinds) into class 2 (windows) during evaluation.

---

### scene_reference_frame.json

Coordinate system metadata from ZAHA dataset. Defines the transformation from local coordinates to geo-registered coordinates.

**Format**:
```json
{
  "base_to_canonical": {
    "shift": [-690955.0, -5336042.0, -604.0],
    "scale": [1.0, 1.0, 1.0]
  },
  "crs": {
    "definition": "EPSG:32632 (WGS84/UTM32N)"
  }
}
```

---

## Directory Structure

Recommended directory layout:

```
zaha_eval/
├── README.md
├── CLAUDE.md
├── transform_zaha.py
├── project_2d_gt.py
├── evaluate_2d.py
├── evaluate_3d.py
├── generate_error_map_pointcloud.py
├── openclip_encoder.py
├── colormaps.py
├── colors.py
├── autoencoder/
│   └── model.py
├── class_mapping.json
├── class_colors.json
├── class_retention_building.json
├── window_merge.json
├── scene_reference_frame.json
├── zaha/                              # Downloaded from ZAHA dataset
│   └── laz_clouds_for_visualisation/ # Extract here
│       ├── building_01.laz
│       ├── building_02.laz
│       └── ...
├── outputs/
│   ├── zaha_merged_labeled.ply       # Output from Step 1
│   ├── zaha_cropped_labeled.ply      # Output from Step 2 (optional)
│   └── gt_semantic_maps/             # Output from Step 3
│       ├── IMG_001.npy
│       ├── IMG_001_vis.png
│       └── ...
└── eval_results/
    ├── 2d/                           # Output from Step 4
    │   ├── metrics.json
    │   ├── confusion_matrix.png
    │   └── ...
    └── 3d/                           # Output from Step 5
        ├── metrics_3d.json
        ├── confusion_matrix_3d.png
        ├── gt_semantic_colored.ply
        ├── predicted_semantic.ply
        └── error_map.ply
```

---

## Troubleshooting

### COLMAP Path Issues

**Problem**: `project_2d_gt.py` fails to load COLMAP reconstruction

**Solution**:
- Ensure `--colmap_dir` points to the directory containing `cameras.bin`, `images.bin`, `points3D.bin`
- COLMAP sparse reconstruction should be in binary format (not text)
- Check that Python path includes LangSplat parent directory for `scene.colmap_loader` import

---

### Missing Dependencies

**Problem**: `ModuleNotFoundError` for `laspy`, `open_clip`, etc.

**Solution**:
```bash
pip install laspy plyfile open-clip-torch scipy pyproj
```

For CLIP, ensure you install the correct package:
```bash
pip install open-clip-torch  # NOT openai-clip
```

---

### Memory Issues with Large Point Clouds

**Problem**: Out of memory when processing large LAZ files

**Solution**:
- `transform_zaha.py` uses streaming processing (no full file load)
- Reduce `--chunk` size (default: 2,000,000 points per chunk)
- Use aggressive downsampling:
  ```bash
  --downsample_strategy voxel+class_aware --voxel_size 0.1
  ```
- For Step 2, crop the scene to camera coverage area

---

### Coordinate System Mismatches

**Problem**: Point cloud doesn't align with COLMAP cameras in `project_2d_gt.py`

**Solution**:
- Verify `--shift_x/y/z` values match ZAHA dataset documentation
- Check `scene_reference_frame.json` shift values
- Ensure target EPSG matches LangSplat training data coordinate system
- If using custom scenes, you may need to adjust coordinate transformations in `transform_zaha.py`

---

### Autoencoder Checkpoint Issues

**Problem**: `evaluate_2d.py` or `evaluate_3d.py` fails to load autoencoder

**Solution**:
- Ensure autoencoder checkpoint matches the feature dimension in LangSplat checkpoint
- Check encoder/decoder dimensions in `autoencoder/model.py`
- Autoencoder should decode to 512-dim CLIP embeddings

---

### Feature Dimension Mismatch

**Problem**: Checkpoint contains 3-dim features but autoencoder expects 256-dim

**Solution**:
- Retrain autoencoder with correct input dimension, OR
- Use checkpoint with matching feature dimension
- Check LangSplat training configuration for feature dimension

---


## Additional Notes

- All scripts support `--help` flag for detailed parameter descriptions
- Logging is saved to `eval.log` or `eval_3d.log` in output directories
- Visualization images are automatically generated for all evaluation steps
- For questions about the ZAHA dataset coordinate system, refer to documentation in the dataset download

---

**Last Updated**: December 2024

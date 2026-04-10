# Unified Evaluation Pipeline

## Introduction

This repository provides a unified evaluation pipeline for three semantic segmentation methods on urban 3D scenes:

* [GS4City](https://github.com/Jinyzzz/GS4City)
* [LangSplat](https://github.com/minghanqin/LangSplat)
* [Gaga](https://github.com/weijielyu/Gaga) / [Gaussian Grouping](https://github.com/lkeab/gaussian-grouping)

The pipeline resolves differences in output formats (semantic labels, instance masks, open-vocabulary predictions) and enables consistent evaluation under a shared label space.

This codebase is developed based on [zaha_eval](https://github.com/zqlin0521/zaha_eval).

The Zaha dataset used in this project is sourced from [Zaha Dataset Repository](https://github.com/OloOcki/zaha).



## Pipeline Overview

```
LAZ/LAS Point Cloud
        ↓
transform_goldcoast.py / transform_zaha.py
        ↓
PLY Point Cloud
        ↓
project_2d_gt.py
        ↓
gt_fusion.py
        ↓
query_mask.py (Gaga only)
        ↓
run_all_evals.py
```



## Dataset Processing

### Scripts

* `transform_goldcoast.py`
* `transform_zaha.py`

### Usage

```bash
python transform_goldcoast.py --in <input> --out <output.ply>
```

or

```bash
python transform_zaha.py --in <input> --out <output.ply>
```

### Input

* `.laz` / `.las` files
* directory or glob pattern

### Output

* `.ply` point cloud with:

  * x, y, z
  * scalar_Classification

### Key Parameters

| Parameter             | Description                          |
| --------------------- | ------------------------------------ |
| --in                  | Input file / directory / pattern     |
| --out                 | Output PLY path                      |
| --chunk               | Chunk size                           |
| --target_epsg         | Target CRS                           |
| --downsample_strategy | none / voxel / uniform / class_aware |
| --voxel_size          | Voxel size                           |



## GT Generation

### 1. Point Cloud Projection

Script:

* `project_2d_gt.py`

Usage:

```bash
python project_2d_gt.py \
  --ply_path <input.ply> \
  --colmap_dir <sparse/0> \
  --output_dir <output_dir>
```

Input:

* `.ply` point cloud
* COLMAP:

  * cameras.bin
  * images.bin

Output:

```
output_dir/
├── *.npy
├── *_vis.png
├── statistics.json
```

Key Parameters:

| Parameter               | Description                      |
| ----------------------- | -------------------------------- |
| --ply_path              | Input point cloud                |
| --colmap_dir            | Camera parameters                |
| --target_width / height | Output resolution                |
| --fill_holes            | none / nearest / occlusion_aware |
| --fill_distance         | Fill radius                      |
| --min_depth             | Minimum depth                    |
| --max_depth             | Maximum depth                    |



### 2. GT Fusion

Script:

* `gt_fusion.py`

Usage:

```bash
python gt_fusion.py \
  --rgb_dir <images> \
  --zaha_gt_dir <gt_raw> \
  --output_dir <output_dir>
```

Input:

* Raw GT (building-only `.npy`)
* RGB images

Output:

```
output_dir/
├── fused/
├── layer_zaha_kept/
├── layer_ai_filled/
├── visualization/
```

Key Parameters:

| Parameter     | Description         |
| ------------- | ------------------- |
| --rgb_dir     | RGB images          |
| --zaha_gt_dir | Raw GT directory    |
| --output_dir  | Output directory    |
| --model_repo  | SegFormer model     |
| --window_size | Sliding window size |
| --stride      | Sliding stride      |



## Method Adaptation

Different methods require different preprocessing steps to align with the unified evaluation format.

### GS4City

* Uses CityGML semantics for building components
* Uses CLIP for non-building categories

No additional processing is required.



### LangSplat

* Fully open-vocabulary method
* Semantic labels are generated through text queries

No additional processing is required.



### Gaga (Gaussian Grouping)

* Provides only instance segmentation
* Does not contain semantic labels

#### Semantic Completion

Script:

* `query_mask.py`

Usage:

```bash
python query_mask.py \
  --anchor_root <anchor> \
  --target_root <target> \
  --json_path <class_mapping.json> \
  --save_root <output_dir>
```

Input:

```
anchor_root/
├── renders/
├── objects_test/

target_root/
├── objects_test/
```

* `class_mapping.json`

Output:

```
output_dir/
├── <class_id>/
│   ├── *.png
├── _debug_anchors/
```

Key Parameters:

| Parameter     | Description     |
| ------------- | --------------- |
| --anchor_root | Anchor data     |
| --target_root | Target data     |
| --json_path   | Class mapping   |
| --sam_ckpt    | SAM checkpoint  |
| --dino_ckpt   | DINO checkpoint |
| --device      | cuda / cpu      |



## Unified Evaluation

Script:

* `run_all_evals.py`

Usage:

```bash
python run_all_evals.py \
  --run_citygml_clip \
  --run_langsplat \
  --run_gaga_dino
```

Input:

```
gt/
├── layer_zaha_kept/
├── layer_ai_filled/
```

Predictions:

* GS4City
* LangSplat
* Gaga (after semantic completion)

Output:

```
output/
├── citygml_clip/
├── langsplat/
├── gaga_dino/
├── summary_all_methods.json
├── cross_method_prediction_*/
```

Key Parameters:

| Parameter                 | Description      |
| ------------------------- | ---------------- |
| --class_mapping_path      | Class definition |
| --gt_split_root           | GT root          |
| --root_output_dir         | Output directory |
| --num_images              | Number of images |
| --whole_building_fine_ids | Building classes |
| --run_citygml_clip        | Enable GS4City   |
| --run_langsplat           | Enable LangSplat |
| --run_gaga_dino           | Enable Gaga      |

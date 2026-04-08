#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate one combined visualization per image from:
  1) evaluation result folders (predictions/*.npz)
  2) RGB image folder
  3) GT npy folder

Layout:
    columns:
        col 1: RGB / GT
        col 2: row titles
        col 3: LangSplat
        col 4: GaGa
        col 5: Ours
        col 6: legends

    rows:
        row 1: Whole level
        row 2: Part level

A short downward arrow is placed ONLY in the gap between the two rows
for each method column.

Input folders are predefined in the CONFIG section at the top.

Output:
    EVAL_ROOT/
        combined_panels/
            <image_name>.png
"""

import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.gridspec import GridSpec

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image
except ImportError:
    Image = None


# =========================================================
# ====================== CONFIG START =====================
# =========================================================

EVAL_ROOT = "/workspace/zaha_eval/all_eval_results_1"
RGB_DIR = "/workspace/CityGMLGaussian/dataset/subset_building1/images"
GT_DIR = "/workspace/zaha_eval/gt/subset1_499_test/fused"

OUTPUT_DIR_NAME = "combined_panels"
NUM_IMAGES = None

METHODS = ["langsplat", "gaga_dino", "citygml_clip"]
METHOD_DISPLAY_NAMES = {
    "langsplat": "LangSplat",
    "gaga_dino": "GaGa",
    "citygml_clip": "Ours",
}

BUILDING_ID = 200
PART_IDS = [1, 2, 3, 12]
NONBUILD_IDS = [101, 103, 104]

WHOLE_BUILDING_COLOR = [168, 123, 204]
WHOLE_NONBUILD_COLOR = [182, 208, 167]
WHOLE_NOMASK_COLOR = [255, 255, 255]

CLASS_COLORS = {
    1:   [238, 229, 195],   # wall
    2:   [140, 170, 220],   # window
    3:   [210, 105, 20],    # door
    12:  [192, 0, 0],       # roof
    101: [95, 145, 65],     # tree
    103: [255, 240, 0],     # person
    104: [120, 120, 120],   # car
}
PART_BACKGROUND_COLOR = [255, 255, 255]

FIG_W = 18.8
ROW_H = 3.95

TITLE_FONT_SIZE = 16
METHOD_TITLE_FONT_SIZE = 14
SIDE_TITLE_FONT_SIZE = 13
LEGEND_TITLE_FONT_SIZE = 12.5
LEGEND_SECTION_FONT_SIZE = 12
LEGEND_FONT_SIZE = 12

TEXT_COLOR = "black"

# smaller gaps
HSPACE = 0.032
WSPACE = 0.045

# legend strips: larger blocks, tighter layout
LEGEND_STRIP_W = 0.125
LEGEND_STRIP_H = 0.040
LEGEND_TEXT_GAP = 0.008

# very short centered arrow
ARROW_CENTER_X = 0.50
ARROW_TOP_Y = 0.53
ARROW_BOTTOM_Y = 0.47
ARROW_MUTATION_SCALE = 16
ARROW_LINEWIDTH = 2.0

# =========================================================
# ======================= CONFIG END ======================
# =========================================================


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def find_files_by_stem(root_dir: str, extensions: Tuple[str, ...], recursive: bool = True) -> Dict[str, str]:
    path_map = {}
    patterns = []
    for ext in extensions:
        if recursive:
            patterns.append(os.path.join(root_dir, "**", f"*{ext}"))
        else:
            patterns.append(os.path.join(root_dir, f"*{ext}"))

    for pattern in patterns:
        for p in glob.glob(pattern, recursive=recursive):
            stem = Path(p).stem
            if stem not in path_map:
                path_map[stem] = p
    return path_map


def list_common_prediction_names(eval_root: str, methods: List[str]) -> List[str]:
    common = None
    for method in methods:
        pred_dir = os.path.join(eval_root, method, "predictions")
        if not os.path.exists(pred_dir):
            print(f"[WARN] Missing predictions dir: {pred_dir}")
            continue
        names = {Path(p).stem for p in glob.glob(os.path.join(pred_dir, "*.npz"))}
        common = names if common is None else (common & names)

    if common is None:
        return []
    return sorted(common)


def load_method_prediction(eval_root: str, method: str, image_name: str) -> Optional[Dict[str, np.ndarray]]:
    npz_path = os.path.join(eval_root, method, "predictions", f"{image_name}.npz")
    if not os.path.exists(npz_path):
        return None

    data = np.load(npz_path)
    required = ["pred_building", "pred_parts", "pred_non"]
    for k in required:
        if k not in data:
            print(f"[WARN] Missing key '{k}' in {npz_path}")
            return None

    return {
        "pred_building": data["pred_building"].astype(np.int32),
        "pred_parts": data["pred_parts"].astype(np.int32),
        "pred_non": data["pred_non"].astype(np.int32),
    }


def load_rgb_image(image_path: str) -> np.ndarray:
    if Image is not None:
        img = Image.open(image_path).convert("RGB")
        return np.array(img)
    if cv2 is not None:
        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    raise RuntimeError("Need either PIL or cv2 installed to load RGB images.")


def load_gt_npy(gt_path: str) -> np.ndarray:
    return np.load(gt_path).astype(np.int32)


def colorize_whole(pred_building: np.ndarray, pred_parts: np.ndarray, pred_non: np.ndarray) -> np.ndarray:
    h, w = pred_building.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[:, :] = np.array(WHOLE_NOMASK_COLOR, dtype=np.uint8)

    building_mask = (pred_building == BUILDING_ID) | np.isin(pred_parts, PART_IDS)
    nonbuild_mask = np.isin(pred_non, NONBUILD_IDS)

    vis[building_mask] = np.array(WHOLE_BUILDING_COLOR, dtype=np.uint8)
    vis[nonbuild_mask] = np.array(WHOLE_NONBUILD_COLOR, dtype=np.uint8)

    return vis

def merge_part_and_nonbuilding(pred_parts: np.ndarray, pred_non: np.ndarray) -> np.ndarray:
    merged = np.full_like(pred_parts, -1, dtype=np.int32)

    part_mask = np.isin(pred_parts, PART_IDS)
    non_mask = np.isin(pred_non, NONBUILD_IDS)

    merged[part_mask] = pred_parts[part_mask]
    merged[non_mask] = pred_non[non_mask]

    return merged


def colorize_part(label_map: np.ndarray) -> np.ndarray:
    h, w = label_map.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[:, :] = np.array(PART_BACKGROUND_COLOR, dtype=np.uint8)

    for cid in np.unique(label_map):
        if cid < 0:
            continue
        vis[label_map == cid] = np.array(CLASS_COLORS.get(int(cid), [160, 160, 160]), dtype=np.uint8)
    return vis


def draw_side_title(ax, text: str):
    ax.axis("off")
    ax.text(
        0.88, 0.5,
        text,
        ha="right",
        va="center",
        fontsize=SIDE_TITLE_FONT_SIZE,
        fontweight="bold",
        color=TEXT_COLOR,
        linespacing=1.0,
    )


def draw_strip_item(ax, cx, y, color_rgb, label, fontsize=11.5):
    x0 = cx - LEGEND_STRIP_W / 2
    y0 = y - LEGEND_STRIP_H / 2

    rect = Rectangle(
        (x0, y0),
        LEGEND_STRIP_W,
        LEGEND_STRIP_H,
        facecolor=np.array(color_rgb) / 255.0,
        edgecolor="black",
        linewidth=0.8,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(rect)

    ax.text(
        cx,
        y0 - LEGEND_TEXT_GAP,
        label,
        transform=ax.transAxes,
        va="top",
        ha="center",
        fontsize=fontsize,
        color="black",
    )


def draw_whole_legend(ax):
    ax.axis("off")
    ax.text(
        0.01, 0.96, "Whole legend",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=LEGEND_TITLE_FONT_SIZE,
        fontweight="bold",
        color=TEXT_COLOR,
    )

    draw_strip_item(ax, 0.23, 0.64, WHOLE_BUILDING_COLOR, "building", LEGEND_FONT_SIZE)
    draw_strip_item(ax, 0.23, 0.40, WHOLE_NONBUILD_COLOR, "non-building", LEGEND_FONT_SIZE)


def draw_part_legend(ax):
    ax.axis("off")
    ax.text(
        0.01, 0.98, "Part legend",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=LEGEND_TITLE_FONT_SIZE,
        fontweight="bold",
        color=TEXT_COLOR,
    )

    left_items = [(2, "window"), (3, "door"), (1, "wall"), (12, "roof")]
    left_ys = [0.73, 0.56, 0.39, 0.22]
    for (cid, label), y in zip(left_items, left_ys):
        draw_strip_item(ax, 0.23, y, CLASS_COLORS[cid], label, LEGEND_FONT_SIZE)

    right_items = [(104, "car"), (103, "person"), (101, "tree")]
    right_ys = [0.73, 0.48, 0.23]
    for (cid, label), y in zip(right_items, right_ys):
        draw_strip_item(ax, 0.62, y, CLASS_COLORS[cid], label, LEGEND_FONT_SIZE)


def draw_gap_arrow_on_overlay(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    arrow = FancyArrowPatch(
        (ARROW_CENTER_X, ARROW_TOP_Y),
        (ARROW_CENTER_X, ARROW_BOTTOM_Y),
        arrowstyle="-|>",
        mutation_scale=ARROW_MUTATION_SCALE,
        linewidth=ARROW_LINEWIDTH,
        color="black",
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(arrow)


def build_valid_image_list(eval_root: str, rgb_dir: str, gt_dir: str, methods: List[str]) -> List[Tuple[str, str, str]]:
    pred_names = list_common_prediction_names(eval_root, methods)

    rgb_map = find_files_by_stem(
        rgb_dir,
        extensions=(".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"),
        recursive=True,
    )
    gt_map = find_files_by_stem(
        gt_dir,
        extensions=(".npy",),
        recursive=True,
    )

    valid = []
    missing_rgb = 0
    missing_gt = 0

    for name in pred_names:
        rgb_path = rgb_map.get(name)
        gt_path = gt_map.get(name)

        if rgb_path is None:
            missing_rgb += 1
            continue
        if gt_path is None:
            missing_gt += 1
            continue

        valid.append((name, rgb_path, gt_path))

    if missing_rgb > 0:
        print(f"[WARN] {missing_rgb} images skipped because RGB not found.")
    if missing_gt > 0:
        print(f"[WARN] {missing_gt} images skipped because GT not found.")
    return valid


def generate_one_combined_figure(eval_root: str, output_dir: str, image_name: str, rgb_path: str, gt_path: str):
    method_data = []

    for method in METHODS:
        pred = load_method_prediction(eval_root, method, image_name)
        if pred is None:
            print(f"[WARN] Missing prediction for {method}, image={image_name}")
            return

        whole_vis = colorize_whole(pred["pred_building"], pred["pred_parts"], pred["pred_non"])
        part_vis = colorize_part(merge_part_and_nonbuilding(pred["pred_parts"], pred["pred_non"]))

        method_data.append({
            "display_name": METHOD_DISPLAY_NAMES.get(method, method),
            "whole_vis": whole_vis,
            "part_vis": part_vis,
        })

    rgb_img = load_rgb_image(rgb_path)
    gt_vis = colorize_part(load_gt_npy(gt_path))

    fig_h = ROW_H * 2
    fig = plt.figure(figsize=(FIG_W, fig_h), facecolor="white")

    gs = GridSpec(
        nrows=2,
        ncols=6,
        width_ratios=[1.0, 0.18, 1.0, 1.0, 1.0, 0.46],
        height_ratios=[1.0, 1.0],
        figure=fig,
    )

    fig.suptitle(image_name, fontsize=TITLE_FONT_SIZE, fontweight="bold", y=0.975)

    # RGB / GT
    ax_rgb = fig.add_subplot(gs[0, 0])
    ax_rgb.imshow(rgb_img, interpolation="nearest")
    ax_rgb.axis("off")
    ax_rgb.set_title("RGB", fontsize=METHOD_TITLE_FONT_SIZE, pad=4, fontweight="bold")

    ax_gt = fig.add_subplot(gs[1, 0])
    ax_gt.imshow(gt_vis, interpolation="nearest")
    ax_gt.axis("off")
    ax_gt.set_title("Ground Truth", fontsize=METHOD_TITLE_FONT_SIZE, pad=4, fontweight="bold")

    # row titles
    ax_title_whole = fig.add_subplot(gs[0, 1])
    draw_side_title(ax_title_whole, "Whole\nlevel")

    ax_title_part = fig.add_subplot(gs[1, 1])
    draw_side_title(ax_title_part, "Part\nlevel")

    # method columns
    method_overlay_axes = []
    for col_idx, row in enumerate(method_data, start=2):
        ax_whole = fig.add_subplot(gs[0, col_idx])
        ax_part = fig.add_subplot(gs[1, col_idx])

        ax_whole.imshow(row["whole_vis"], interpolation="nearest")
        ax_whole.axis("off")
        ax_whole.set_facecolor("white")
        ax_whole.set_title(row["display_name"], fontsize=METHOD_TITLE_FONT_SIZE, pad=4, fontweight="bold")

        ax_part.imshow(row["part_vis"], interpolation="nearest")
        ax_part.axis("off")
        ax_part.set_facecolor("white")

        ax_overlay = fig.add_subplot(gs[:, col_idx], frameon=False)
        ax_overlay.set_zorder(10)
        ax_overlay.patch.set_alpha(0.0)
        method_overlay_axes.append(ax_overlay)

    # legends
    ax_leg_whole = fig.add_subplot(gs[0, 5])
    draw_whole_legend(ax_leg_whole)

    ax_leg_part = fig.add_subplot(gs[1, 5])
    draw_part_legend(ax_leg_part)

    plt.subplots_adjust(
        left=0.02,
        right=0.99,
        top=0.89,
        bottom=0.06,
        wspace=WSPACE,
        hspace=HSPACE,
    )

    for ax_overlay in method_overlay_axes:
        draw_gap_arrow_on_overlay(ax_overlay)

    save_path = os.path.join(output_dir, f"{image_name}.png")
    plt.savefig(save_path, dpi=180, bbox_inches="tight", pad_inches=0.018, facecolor="white")
    plt.close(fig)


def main():
    if not os.path.exists(EVAL_ROOT):
        raise FileNotFoundError(f"EVAL_ROOT not found: {EVAL_ROOT}")
    if not os.path.exists(RGB_DIR):
        raise FileNotFoundError(f"RGB_DIR not found: {RGB_DIR}")
    if not os.path.exists(GT_DIR):
        raise FileNotFoundError(f"GT_DIR not found: {GT_DIR}")

    output_dir = os.path.join(EVAL_ROOT, OUTPUT_DIR_NAME)
    ensure_dir(output_dir)

    valid_items = build_valid_image_list(EVAL_ROOT, RGB_DIR, GT_DIR, METHODS)
    if NUM_IMAGES is not None:
        valid_items = valid_items[:NUM_IMAGES]

    print(f"[INFO] Valid images found: {len(valid_items)}")
    print(f"[INFO] Output directory: {output_dir}")

    for idx, (image_name, rgb_path, gt_path) in enumerate(valid_items, start=1):
        generate_one_combined_figure(EVAL_ROOT, output_dir, image_name, rgb_path, gt_path)
        if idx % 20 == 0 or idx == len(valid_items):
            print(f"[INFO] {idx}/{len(valid_items)} done")

    print("Done.")


if __name__ == "__main__":
    main()
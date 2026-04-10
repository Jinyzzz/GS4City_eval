#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import glob
import json
import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================

DEFAULT_RGB_DIR = "/workspace/CityGMLGaussian/output/subset_goldcoast_gaga_30000/test/ours_30000/gt"
DEFAULT_ZAHA_GT_DIR = "/workspace/zaha_eval/gt/subset_goldcoast_33"
DEFAULT_OUTPUT_DIR = "/workspace/zaha_eval/gt/subset_goldcoast_test"
DEFAULT_GT_MERGE_MAP_PATH = "config/gt_merge_map.json"

DEFAULT_MODEL_REPO = "nvidia/segformer-b5-finetuned-ade-640-640"
DEFAULT_NUM_CLASSES = 150
DEFAULT_WINDOW_SIZE = 800
DEFAULT_STRIDE = 600

DEFAULT_KEEP_ZAHA_IDS = [1, 2, 3, 12]

DEFAULT_ADE20K_MAPPING = {
    2: 100,   # Sky
    4: 101,   # Tree/Vegetation
    6: 102,   # Road/Route
    13: 102,  # Earth/Ground
    12: 103,  # Person
    20: 104,  # Car
}


def parse_int_list(value: str):
    if value is None:
        return None
    if isinstance(value, list):
        return [int(v) for v in value]
    value = value.strip()
    if not value:
        return []
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_json_mapping(value: str):
    if value is None:
        return None
    if os.path.isfile(value):
        with open(value, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raw = json.loads(value)
    return {int(k): int(v) for k, v in raw.items()}


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def colorize_mask(mask):
    unique_ids = np.unique(mask)
    vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    np.random.seed(42)
    for uid in unique_ids:
        if uid <= 0:
            continue
        color = np.random.randint(0, 255, 3)
        vis[mask == uid] = color
    return vis


def apply_label_merges(label_map, merge_map):
    if not merge_map:
        return label_map
    out = label_map.copy()
    for target, sources in merge_map.items():
        target = int(target)
        for src in sources:
            src = int(src)
            if src == target:
                continue
            out[out == src] = target
    return out


def predict_sliding_window(
    model,
    processor,
    image_pil,
    num_classes,
    window_size,
    stride,
    device="cuda",
):
    w, h = image_pil.size
    image_np = np.array(image_pil)

    probs_map = torch.zeros((num_classes, h, w), dtype=torch.float16, device="cpu")
    count_map = torch.zeros((1, h, w), dtype=torch.float16, device="cpu")

    rows = sorted(list(set([x for x in range(0, h, stride)] + [max(h - window_size, 0)])))
    cols = sorted(list(set([x for x in range(0, w, stride)] + [max(w - window_size, 0)])))

    model.eval()

    for r in rows:
        for c in cols:
            r_end = min(r + window_size, h)
            c_end = min(c + window_size, w)
            r_start, c_start = r, c

            crop = image_np[r_start:r_end, c_start:c_end, :]
            crop_pil = Image.fromarray(crop)

            inputs = processor(images=crop_pil, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

                upsampled_logits = F.interpolate(
                    logits,
                    size=(crop.shape[0], crop.shape[1]),
                    mode="bilinear",
                    align_corners=False,
                )

                probs_map[:, r_start:r_end, c_start:c_end] += upsampled_logits.squeeze(0).cpu().to(torch.float16)
                count_map[:, r_start:r_end, c_start:c_end] += 1.0

    count_map = torch.clamp(count_map, min=1.0)
    probs_map /= count_map
    final_pred = probs_map.argmax(dim=0).numpy().astype(np.int32)
    return final_pred


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Fuse Zaha GT with SegFormer predictions using CLI arguments."
    )

    parser.add_argument("--rgb_dir", type=str, default=DEFAULT_RGB_DIR, help="Directory containing RGB images.")
    parser.add_argument("--zaha_gt_dir", type=str, default=DEFAULT_ZAHA_GT_DIR, help="Directory containing input Zaha GT .npy files.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output root directory.")
    parser.add_argument("--gt_merge_map_path", type=str, default=DEFAULT_GT_MERGE_MAP_PATH, help="Path to GT merge map JSON.")
    parser.add_argument("--disable_merge_map", action="store_true", help="Disable label merge map loading.")

    parser.add_argument("--model_repo", type=str, default=DEFAULT_MODEL_REPO, help="Hugging Face SegFormer model repo.")
    parser.add_argument("--num_classes", type=int, default=DEFAULT_NUM_CLASSES, help="Number of SegFormer output classes.")
    parser.add_argument("--window_size", type=int, default=DEFAULT_WINDOW_SIZE, help="Sliding window size.")
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE, help="Sliding window stride.")
    parser.add_argument("--device", type=str, default=None, help="Execution device. Default: auto-detect.")

    parser.add_argument(
        "--keep_zaha_ids",
        type=str,
        default=",".join(map(str, DEFAULT_KEEP_ZAHA_IDS)),
        help="Comma-separated class IDs that must be preserved from Zaha GT.",
    )
    parser.add_argument(
        "--ade20k_mapping",
        type=str,
        default=json.dumps(DEFAULT_ADE20K_MAPPING),
        help="JSON string or JSON file path mapping ADE20K IDs to project IDs.",
    )

    parser.add_argument("--save_visualization", action="store_true", default=True, help="Save visualization images.")
    parser.add_argument("--no_save_visualization", action="store_false", dest="save_visualization", help="Do not save visualization images.")
    parser.add_argument("--strict_rgb_match", action="store_true", help="Raise an error if an RGB image is missing for a GT file.")
    parser.add_argument("--image_extensions", type=str, default=".png,.jpg,.JPG,.jpeg,.JPEG", help="Comma-separated candidate image extensions.")

    return parser


def find_rgb_path(base_name, rgb_dir, exts):
    candidates = [base_name, base_name.replace("_D", ""), base_name.replace("_depth", "")]
    rgb_path = None

    for name in candidates:
        for ext in exts:
            p = os.path.join(rgb_dir, name + ext)
            if os.path.exists(p):
                rgb_path = p
                break
        if rgb_path:
            break

    if not rgb_path:
        potential = glob.glob(os.path.join(rgb_dir, f"{candidates[-1]}*"))
        potential = [f for f in potential if os.path.splitext(f)[1] in exts]
        if potential:
            rgb_path = potential[0]

    return rgb_path


def main():
    parser = build_argparser()
    args = parser.parse_args()

    keep_zaha_ids = parse_int_list(args.keep_zaha_ids)
    ade20k_mapping = parse_json_mapping(args.ade20k_mapping)
    device = args.device or get_device()
    image_exts = [e.strip() for e in args.image_extensions.split(",") if e.strip()]

    dir_fused = os.path.join(args.output_dir, "fused")
    dir_zaha = os.path.join(args.output_dir, "layer_zaha_kept")
    dir_ai = os.path.join(args.output_dir, "layer_ai_filled")
    dir_vis = os.path.join(args.output_dir, "visualization")

    for d in [dir_fused, dir_zaha, dir_ai, dir_vis]:
        os.makedirs(d, exist_ok=True)

    gt_merge_map = None
    if not args.disable_merge_map and args.gt_merge_map_path:
        if os.path.exists(args.gt_merge_map_path):
            print(f"Loading merge map from {args.gt_merge_map_path}...")
            with open(args.gt_merge_map_path, "r", encoding="utf-8") as f:
                merge_raw = json.load(f)
            gt_merge_map = {int(k): [int(v) for v in values] for k, values in merge_raw.items()}
        else:
            print(f"Merge map not found: {args.gt_merge_map_path}. Proceeding without merging.")
    else:
        print("Merge map loading is disabled.")

    print(f"Loading SegFormer model: {args.model_repo}...")
    processor = SegformerImageProcessor.from_pretrained(args.model_repo)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_repo)
    model.to(device)

    zaha_files = sorted(glob.glob(os.path.join(args.zaha_gt_dir, "*.npy")))
    print(f"Found {len(zaha_files)} Zaha GT files.")
    print(f"Device: {device}")
    print(f"Window size: {args.window_size}")
    print(f"Stride: {args.stride}")
    print(f"Keep Zaha IDs: {keep_zaha_ids}")
    print(f"ADE20K mapping: {ade20k_mapping}")

    skipped_missing_rgb = []

    for gt_path in tqdm(zaha_files, desc="Processing images"):
        base_name = os.path.splitext(os.path.basename(gt_path))[0]
        rgb_path = find_rgb_path(base_name, args.rgb_dir, image_exts)

        if rgb_path is None:
            msg = f"RGB image not found for {base_name}"
            if args.strict_rgb_match:
                raise FileNotFoundError(msg)
            skipped_missing_rgb.append(base_name)
            continue

        zaha_mask = np.load(gt_path)
        image = Image.open(rgb_path).convert("RGB")

        if gt_merge_map:
            zaha_mask = apply_label_merges(zaha_mask, gt_merge_map)

        seg_pred_raw = predict_sliding_window(
            model=model,
            processor=processor,
            image_pil=image,
            num_classes=args.num_classes,
            window_size=args.window_size,
            stride=args.stride,
            device=device,
        )

        if seg_pred_raw.shape != zaha_mask.shape:
            seg_pred_raw = cv2.resize(
                seg_pred_raw.astype(np.float32),
                (zaha_mask.shape[1], zaha_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int32)

        segformer_remapped = np.zeros_like(seg_pred_raw, dtype=np.int32)
        for ade_id, project_id in ade20k_mapping.items():
            segformer_remapped[seg_pred_raw == ade_id] = project_id

        is_zaha_kept = np.isin(zaha_mask, keep_zaha_ids)
        has_valid_ai = segformer_remapped > 0

        layer_zaha_kept = np.zeros_like(zaha_mask)
        layer_zaha_kept[is_zaha_kept] = zaha_mask[is_zaha_kept]

        should_use_ai = (~is_zaha_kept) & has_valid_ai
        layer_ai_filled = np.zeros_like(zaha_mask)
        layer_ai_filled[should_use_ai] = segformer_remapped[should_use_ai]

        final_mask = zaha_mask.copy()
        final_mask[should_use_ai] = segformer_remapped[should_use_ai]

        np.save(os.path.join(dir_fused, f"{base_name}.npy"), final_mask)
        np.save(os.path.join(dir_zaha, f"{base_name}.npy"), layer_zaha_kept)
        np.save(os.path.join(dir_ai, f"{base_name}.npy"), layer_ai_filled)

        if args.save_visualization:
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            axes[0].imshow(image)
            axes[0].set_title("RGB")
            axes[0].axis("off")

            axes[1].imshow(colorize_mask(zaha_mask))
            axes[1].set_title("Original Zaha GT")
            axes[1].axis("off")

            axes[2].imshow(colorize_mask(layer_ai_filled))
            axes[2].set_title("AI Fill Only")
            axes[2].axis("off")

            axes[3].imshow(colorize_mask(final_mask))
            axes[3].set_title("Fused GT")
            axes[3].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(dir_vis, f"{base_name}_vis.png"), dpi=150)
            plt.close(fig)

    print(f"Done. Files saved to subdirectories in: {args.output_dir}")
    if skipped_missing_rgb:
        print(f"Skipped {len(skipped_missing_rgb)} files due to missing RGB images.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import traceback
import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser


# ==============================================================================
# Default configuration
# ==============================================================================

DEFAULT_ANCHOR_ROOT = "/workspace/CityGMLGaussian/output/gaga_level1_30000/train/ours_30000"
DEFAULT_TARGET_ROOT = "/workspace/CityGMLGaussian/output/gaga_level1_30000/test/ours_30000"
DEFAULT_JSON_PATH = "/workspace/zaha_eval/class_mapping.json"
DEFAULT_SAVE_ROOT = "/workspace/zaha_eval/eval_results/gaga_query/objects_prompt"

DEFAULT_SAM_CHECKPOINT = "/workspace/CityGMLGaussian/weight/sam_vit_h_4b8939.pth"
DEFAULT_DINO_CONFIG = "GroundingDINO_SwinB.cfg.py"
DEFAULT_DINO_CKPT = "groundingdino_swinb_cogcoor.pth"

DEFAULT_DEVICE = "cuda"
DEFAULT_ANCHOR_FRAME_NAME = None
DEFAULT_SAVE_DEBUG_ANCHORS = True
DEFAULT_DEBUG_DIR_NAME = "_debug_anchors"
DEFAULT_TARGET_SUFFIX = ".png"


# Path setup
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ext.grounded_sam import grouned_sam_output, load_model_hf, select_obj_ioa
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("[Error] Could not import 'ext.grounded_sam'. Make sure the ext folder is located next to this script.")
    sys.exit(1)


def build_argparser():
    parser = ArgumentParser(
        description="Extract object masks from target ID maps using anchor-frame Grounded-SAM detection and object ID matching."
    )

    parser.add_argument("--anchor_root", type=str, default=DEFAULT_ANCHOR_ROOT, help="Anchor root directory. Must contain renders/ and objects_test/.")
    parser.add_argument("--target_root", type=str, default=DEFAULT_TARGET_ROOT, help="Target root directory. Must contain objects_test/.")
    parser.add_argument("--json_path", type=str, default=DEFAULT_JSON_PATH, help="Path to class_mapping JSON.")
    parser.add_argument("--save_root", type=str, default=DEFAULT_SAVE_ROOT, help="Directory to save extracted masks.")

    parser.add_argument("--sam_ckpt", type=str, default=DEFAULT_SAM_CHECKPOINT, help="Path to SAM checkpoint.")
    parser.add_argument("--dino_config", type=str, default=DEFAULT_DINO_CONFIG, help="GroundingDINO config path or filename.")
    parser.add_argument("--dino_ckpt", type=str, default=DEFAULT_DINO_CKPT, help="GroundingDINO checkpoint path or filename.")

    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="Execution device, e.g. cuda or cpu.")
    parser.add_argument("--anchor_frame_name", type=str, default=DEFAULT_ANCHOR_FRAME_NAME, help="Specific anchor frame filename to use. Default: first file in renders/.")
    parser.add_argument("--target_suffix", type=str, default=DEFAULT_TARGET_SUFFIX, help="Only target files ending with this suffix will be processed.")

    parser.add_argument("--save_debug_anchors", action="store_true", default=DEFAULT_SAVE_DEBUG_ANCHORS, help="Save debug anchor visualizations.")
    parser.add_argument("--no_save_debug_anchors", action="store_false", dest="save_debug_anchors", help="Do not save debug anchor visualizations.")
    parser.add_argument("--debug_dir_name", type=str, default=DEFAULT_DEBUG_DIR_NAME, help="Subdirectory name for debug anchor outputs.")

    return parser


def resolve_anchor_frame(anchor_render_dir: str, anchor_frame_name: str = None) -> str:
    files = sorted(os.listdir(anchor_render_dir))
    if not files:
        raise RuntimeError(f"Anchor renders directory is empty: {anchor_render_dir}")

    if anchor_frame_name is not None:
        candidate = os.path.join(anchor_render_dir, anchor_frame_name)
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"Specified anchor frame was not found: {candidate}")
        return anchor_frame_name

    return files[0]


def run_extraction(args):
    anchor_render_dir = os.path.join(args.anchor_root, "renders")
    anchor_id_dir = os.path.join(args.anchor_root, "objects_test")
    target_id_dir = os.path.join(args.target_root, "objects_test")

    if not os.path.exists(anchor_render_dir):
        raise FileNotFoundError(f"Anchor RGB directory not found: {anchor_render_dir}")
    if not os.path.exists(anchor_id_dir):
        raise FileNotFoundError(f"Anchor ID directory not found: {anchor_id_dir}")
    if not os.path.exists(target_id_dir):
        raise FileNotFoundError(f"Target ID directory not found: {target_id_dir}")
    if not os.path.exists(args.json_path):
        raise FileNotFoundError(f"JSON file not found: {args.json_path}")

    os.makedirs(args.save_root, exist_ok=True)

    with open(args.json_path, "r", encoding="utf-8") as f:
        class_mapping = json.load(f)
    print(f"[INFO] Loaded class mapping: {len(class_mapping)} classes")

    print("[INFO] Loading Grounding DINO and SAM ...")
    dino_model = load_model_hf("ShilongLiu/GroundingDINO", args.dino_ckpt, args.dino_config)

    if not os.path.exists(args.sam_ckpt):
        raise FileNotFoundError(f"SAM checkpoint not found: {args.sam_ckpt}")

    sam = sam_model_registry["vit_h"](checkpoint=args.sam_ckpt)
    sam.to(device=args.device)
    sam_predictor = SamPredictor(sam)

    anchor_name = resolve_anchor_frame(anchor_render_dir, args.anchor_frame_name)
    anchor_rgb_path = os.path.join(anchor_render_dir, anchor_name)
    anchor_id_path = os.path.join(anchor_id_dir, anchor_name)

    print(f"[INFO] Using anchor frame: {anchor_name}")

    anchor_img_pil = Image.open(anchor_rgb_path).convert("RGB")
    anchor_img_np = np.array(anchor_img_pil)

    if not os.path.exists(anchor_id_path):
        raise RuntimeError(f"Corresponding anchor ID map does not exist: {anchor_id_path}")
    anchor_id_map = cv2.imread(anchor_id_path, cv2.IMREAD_UNCHANGED)

    target_files = sorted(os.listdir(target_id_dir))
    if args.target_suffix:
        target_files = [f for f in target_files if f.endswith(args.target_suffix)]

    debug_dir = os.path.join(args.save_root, args.debug_dir_name)
    if args.save_debug_anchors:
        os.makedirs(debug_dir, exist_ok=True)

    for class_id_str, prompts in class_mapping.items():
        if isinstance(prompts, list):
            text_prompt = " . ".join([p.strip() for p in prompts if str(p).strip()])
        else:
            text_prompt = str(prompts).strip()

        class_id = int(class_id_str)
        print("\n------------------------------------------------")
        print(f"[CLASS {class_id}] Prompt: '{text_prompt}'")

        try:
            text_mask_2d, annotated_frame = grouned_sam_output(
                dino_model,
                sam_predictor,
                text_prompt,
                anchor_img_np,
            )
        except Exception as e:
            print(f"  [Error] DINO/SAM inference failed: {e}")
            continue

        if args.save_debug_anchors:
            Image.fromarray(annotated_frame).save(os.path.join(debug_dir, f"class_{class_id}.png"))

        if text_mask_2d.sum() == 0:
            print("  [Warn] No object detected in anchor frame. Skipping this class.")
            continue

        try:
            anchor_id_tensor = torch.from_numpy(anchor_id_map.astype(np.int64)).to(args.device)

            if isinstance(text_mask_2d, np.ndarray):
                text_mask_tensor = torch.from_numpy(text_mask_2d).bool().to(args.device)
            else:
                text_mask_tensor = text_mask_2d.bool().to(args.device)

            selected_ids = select_obj_ioa(anchor_id_tensor, text_mask_tensor)

            if isinstance(selected_ids, torch.Tensor):
                selected_ids = selected_ids.detach().cpu().tolist()
            elif isinstance(selected_ids, np.ndarray):
                selected_ids = selected_ids.tolist()

            print(f"  [Match] Matched 3D object IDs: {selected_ids}")

        except Exception as e:
            print(f"  [Error] ID matching failed: {e}")
            traceback.print_exc()
            continue

        if len(selected_ids) == 0:
            print("  [Warn] No 3D IDs matched. Skipping this class.")
            continue

        class_save_dir = os.path.join(args.save_root, str(class_id))
        os.makedirs(class_save_dir, exist_ok=True)

        target_id_set = np.array(selected_ids, dtype=anchor_id_map.dtype)

        print(f"  Extracting masks for {len(target_files)} target images ...")
        for t_file in tqdm(target_files, leave=False):
            t_id_path = os.path.join(target_id_dir, t_file)
            t_id_map = cv2.imread(t_id_path, cv2.IMREAD_UNCHANGED)
            if t_id_map is None:
                continue

            mask = np.isin(t_id_map, target_id_set)
            binary_mask = (mask * 255).astype(np.uint8)

            Image.fromarray(binary_mask).save(os.path.join(class_save_dir, t_file))

    print(f"\n[Success] All masks have been saved to: {args.save_root}")


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    run_extraction(args)
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate instance-based semantic segmentation (CityGML + CLIP) using Split Ground Truth,
but report results in the SAME 3-row format as the GroundingDINO script:

  1) building    (binary, id=200)
  2) parts       (multiclass, only GT pixels in part_ids)
  3) nonbuilding (multiclass, only GT pixels in nonbuild_ids)

GT structure:
    /fused
    /layer_zaha_kept
    /layer_ai_filled

Prediction logic:
  - Building: CityGML closed-vocabulary only
      instances whose CityGML semantic type is Building (and descendants) -> label building_id
  - Parts / Nonbuilding:
      full semantic prediction from CityGML closed-vocab + CLIP open-vocab,
      then filtered to part_ids / nonbuild_ids for evaluation and visualization.

Outputs:
  - JSON schema aligned with your DINO 3-row evaluator:
      {
        "building": {...},
        "parts": {...},
        "nonbuilding": {...}
      }
  - visualization: 3 rows x 3 cols
"""

# ============================================================================
# DEFAULT PARAMETERS
# ============================================================================
DEFAULT_INSTANCE_IMAGES_DIR = "/workspace/CityGMLGaussian/output/subset_building4_30000/test/ours_10000/objects_test"
DEFAULT_MODEL_ROOT = "/workspace/CityGMLGaussian/output/subset_building4_30000"

DEFAULT_GT_SPLIT_ROOT = "/workspace/zaha_eval/gt/subset4_499_test"
DEFAULT_OUTPUT_DIR = "/workspace/zaha_eval/eval_results/subset4_gml_sem_test"
DEFAULT_CLASS_MAPPING_PATH = "/workspace/zaha_eval/class_mapping.json"

DEFAULT_RGB_IMAGES_DIR = None
DEFAULT_CLIP_THRESHOLD = 0.0
DEFAULT_SAVE_VISUALIZATIONS = True
DEFAULT_NUM_IMAGES = None

DEFAULT_CLASS_COLORS_PATH = "class_colors.json"
DEFAULT_GT_MERGE_MAP_PATH = "gt_merge_map.json"
DEFAULT_CITYGML_CLASS_MAP_PATH = None

# CLIP model defaults
DEFAULT_OPENCLIP_MODEL_NAME = "ViT-B-32-quickgelu"
DEFAULT_OPENCLIP_PRETRAINED = "openai"

# 3-row evaluation defaults (aligned with DINO script)
DEFAULT_BUILDING_ID = 200
DEFAULT_PART_IDS = [1, 2, 3, 12]
DEFAULT_NONBUILD_IDS = [101, 103, 104]
DEFAULT_PARTS_MERGE_TARGET = 1

DEFAULT_GT_LAYER_BUILDING = "fused"   # fused|zaha|ai
DEFAULT_GT_LAYER_PARTS = "zaha"       # fused|zaha|ai
DEFAULT_GT_LAYER_NONBUILD = "ai"      # fused|zaha|ai

import argparse
import json
import os
import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Set
from collections import defaultdict

import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import open_clip


# ============================================================================
# Logger
# ============================================================================
def get_logger(name, log_file=None, log_level=logging.INFO):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        handlers.append(file_handler)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


# ============================================================================
# Helper Functions: Merge, GT Load, Metrics
# ============================================================================
def apply_label_merges(label_map: np.ndarray, merge_map: Dict[int, List[int]]) -> np.ndarray:
    if not merge_map:
        return label_map
    out = label_map.copy()
    for target_id, source_ids in merge_map.items():
        target_id = int(target_id)
        for src_id in source_ids:
            src_id = int(src_id)
            if src_id == target_id:
                continue
            out[out == src_id] = target_id
    return out


def load_ground_truth_layer(
    layer_dir: str,
    image_names: List[str],
    class_mapping: Dict[int, Union[str, List[str]]],
    logger: logging.Logger,
    label_merge_map: Optional[Dict[int, List[int]]] = None,
    layer_name: str = "GT"
) -> Dict[str, Dict]:
    logger.info(f"   Loading {layer_name} from: {layer_dir}")
    gt_data = {}

    if not os.path.exists(layer_dir):
        logger.error(f"   [ERROR] Directory not found: {layer_dir}")
        return {}

    valid_class_ids = set(class_mapping.keys())

    for img_name in tqdm(image_names, desc=f"Loading {layer_name}", leave=False):
        gt_path = os.path.join(layer_dir, f"{img_name}.npy")
        if not os.path.exists(gt_path):
            continue

        semantic_map = np.load(gt_path)

        # 1. Label Merge
        if label_merge_map:
            semantic_map = apply_label_merges(semantic_map, label_merge_map)

        # 2. Filter Invalid Classes
        invalid_mask = (semantic_map >= 0) & (~np.isin(semantic_map, list(valid_class_ids)))
        semantic_map[invalid_mask] = -1

        coverage_mask = semantic_map >= 0
        unique_classes = np.unique(semantic_map[coverage_mask]).tolist()

        gt_data[img_name] = {
            'semantic_map': semantic_map,
            'coverage_mask': coverage_mask,
            'classes': unique_classes
        }

    logger.info(f"   Loaded {len(gt_data)} images for {layer_name}.")
    return gt_data


def compute_multiclass_metrics(pred_semantic, gt_semantic, eval_mask, class_mapping):
    pred_valid = pred_semantic[eval_mask]
    gt_valid = gt_semantic[eval_mask]

    if len(gt_valid) == 0:
        return None

    gt_classes = np.unique(gt_valid).tolist()
    results = {
        'class_iou': {},
        'pixel_count': int(len(gt_valid)),
        'correct_count': int((pred_valid == gt_valid).sum())
    }

    for cls_id in gt_classes:
        cls_entry = class_mapping.get(int(cls_id), f"class_{cls_id}")
        cls_name = cls_entry[0] if isinstance(cls_entry, list) else cls_entry

        gt_cls_mask = (gt_valid == cls_id)
        pred_cls_mask = (pred_valid == cls_id)

        intersection = np.logical_and(pred_cls_mask, gt_cls_mask).sum()
        union = np.logical_or(pred_cls_mask, gt_cls_mask).sum()

        results['class_iou'][cls_name] = float(intersection / union) if union > 0 else 0.0

    results['mean_iou'] = float(np.mean(list(results['class_iou'].values()))) if results['class_iou'] else 0.0
    results['pixel_acc'] = float(results['correct_count'] / results['pixel_count'])
    return results


def compute_binary_metrics_union_mask(pred_is_pos: np.ndarray, gt_is_pos: np.ndarray, eval_mask: np.ndarray):
    """
    Building-level:
      eval_mask = (GT_valid) & (GT_pos | Pred_pos)
    """
    pb = pred_is_pos[eval_mask].astype(bool)
    gb = gt_is_pos[eval_mask].astype(bool)
    if gb.size == 0:
        return None

    tp = int(np.logical_and(pb, gb).sum())
    fp = int(np.logical_and(pb, ~gb).sum())
    fn = int(np.logical_and(~pb, gb).sum())

    iou = tp / (tp + fp + fn + 1e-9)
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = (2 * prec * rec) / (prec + rec + 1e-9)

    return {
        "iou": float(iou),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tp": tp, "fp": fp, "fn": fn,
        "pixel_count": int(gb.size)
    }


# ============================================================================
# Visualization (same structure as DINO 3-row script)
# ============================================================================
def visualize_three_row_error(
    # row1 building
    pred_building: np.ndarray,
    gt_building_merged: np.ndarray,
    # row2 parts
    pred_parts: np.ndarray,
    gt_parts: np.ndarray,
    # row3 nonbuilding
    pred_non: np.ndarray,
    gt_non: np.ndarray,
    class_colors: Dict[int, List[int]],
    save_path: str,
    building_id: int,
    part_ids: List[int],
    nonbuild_ids: List[int],
):
    """
    Layout (3x3):
      Row 1: [GT Building] [Pred Building] [Error Building]
        - Error only on eval_mask = (GT_valid) & (GT_is_b | Pred_is_b)
      Row 2: [GT Parts]    [Pred Parts]    [Error Parts]  (only show/eval GT in part_ids)
      Row 3: [GT NonB]     [Pred NonB]     [Error NonB]   (only show/eval GT in nonbuild_ids)
    """

    def build_palette(unique_ids):
        palette = {}
        cmap = plt.cm.get_cmap("tab20", max(len(unique_ids), 1) + 5)
        for i, uid in enumerate(sorted(list(unique_ids))):
            if uid < 0:
                continue
            if class_colors and uid in class_colors:
                palette[uid] = np.array(class_colors[uid], dtype=np.uint8)
            else:
                palette[uid] = (np.array(cmap(i)[:3]) * 255).astype(np.uint8)
        return palette

    def colorize(mask, palette):
        h, w = mask.shape
        col = np.zeros((h, w, 3), dtype=np.uint8)
        col[mask < 0] = [30, 30, 30]
        for cid in np.unique(mask):
            if cid < 0:
                continue
            col[mask == cid] = palette.get(int(cid), np.array([128, 128, 128], dtype=np.uint8))
        return col

    def error_map_multiclass_on_mask(pred, gt, eval_mask):
        err = np.full((*gt.shape, 3), 30, dtype=np.uint8)
        correct = (pred == gt) & eval_mask
        wrong = (pred != gt) & eval_mask
        err[correct] = [0, 255, 0]
        err[wrong] = [255, 0, 0]
        return err

    def error_map_building(pred_is_b, gt_is_b, eval_mask):
        err = np.full((*gt_is_b.shape, 3), 30, dtype=np.uint8)
        correct = (pred_is_b == gt_is_b) & eval_mask
        wrong = (pred_is_b != gt_is_b) & eval_mask
        err[correct] = [0, 255, 0]
        err[wrong] = [255, 0, 0]
        return err

    unique_ids = (
        set(np.unique(gt_parts))
        | set(np.unique(gt_non))
        | set(np.unique(pred_parts))
        | set(np.unique(pred_non))
        | {building_id}
    )
    palette = build_palette(unique_ids)

    # -------- Row1: building --------
    gt_valid_b = (gt_building_merged >= 0)
    gt_is_b = (gt_building_merged == building_id)
    pred_is_b = (pred_building == building_id)
    eval_b = gt_valid_b & (gt_is_b | pred_is_b)

    gt_b_vis = np.full_like(gt_building_merged, -1)
    pr_b_vis = np.full_like(pred_building, -1)
    gt_b_vis[gt_is_b] = building_id
    pr_b_vis[pred_is_b] = building_id

    vis_gt_b = colorize(gt_b_vis, palette)
    vis_pr_b = colorize(pr_b_vis, palette)
    err_b = error_map_building(pred_is_b, gt_is_b, eval_b)

    # -------- Row2: parts --------
    gt_valid_p = (gt_parts >= 0) & np.isin(gt_parts, part_ids)
    gt_p_vis = gt_parts.copy()
    pr_p_vis = pred_parts.copy()
    gt_p_vis[~gt_valid_p] = -1
    pr_p_vis[~gt_valid_p] = -1

    vis_gt_p = colorize(gt_p_vis, palette)
    vis_pr_p = colorize(pr_p_vis, palette)
    err_p = error_map_multiclass_on_mask(pr_p_vis, gt_p_vis, gt_valid_p)

    # -------- Row3: non-building --------
    gt_valid_n = (gt_non >= 0) & np.isin(gt_non, nonbuild_ids)
    gt_n_vis = gt_non.copy()
    pr_n_vis = pred_non.copy()
    gt_n_vis[~gt_valid_n] = -1
    pr_n_vis[~gt_valid_n] = -1

    vis_gt_n = colorize(gt_n_vis, palette)
    vis_pr_n = colorize(pr_n_vis, palette)
    err_n = error_map_multiclass_on_mask(pr_n_vis, gt_n_vis, gt_valid_n)

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    axes[0, 0].imshow(vis_gt_b); axes[0, 0].set_title(f"GT Building (merged to {building_id})")
    axes[0, 1].imshow(vis_pr_b); axes[0, 1].set_title("Pred Building (CityGML closed-vocab)")
    axes[0, 2].imshow(err_b);    axes[0, 2].set_title("Error (Building) [GTcov & (GT|Pred)]\nGreen=TP, Red=FP/FN")

    axes[1, 0].imshow(vis_gt_p); axes[1, 0].set_title("GT Parts (Zaha)")
    axes[1, 1].imshow(vis_pr_p); axes[1, 1].set_title("Pred Parts")
    axes[1, 2].imshow(err_p);    axes[1, 2].set_title("Error (Parts)\nGreen=Correct, Red=Wrong")

    axes[2, 0].imshow(vis_gt_n); axes[2, 0].set_title("GT Non-building (AI)")
    axes[2, 1].imshow(vis_pr_n); axes[2, 1].set_title("Pred Non-building")
    axes[2, 2].imshow(err_n);    axes[2, 2].set_title("Error (Non-building)\nGreen=Correct, Red=Wrong")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# CityGML & CLIP Logic
# ============================================================================
class CityGMLSemanticIndex:
    def __init__(self, id_mapping_path: str, city_semantics_path: str):
        with open(id_mapping_path, "r") as f:
            self.instance_to_city_id: Dict[str, str] = json.load(f)
        with open(city_semantics_path, "r") as f:
            self.city_semantics: Dict[str, Dict] = json.load(f)

        self.city_id_to_type: Dict[str, str] = {}
        self.city_id_to_parent: Dict[str, Optional[str]] = {}
        self.parent_to_children: Dict[str, List[str]] = defaultdict(list)

        for cid, rec in self.city_semantics.items():
            ctype = rec.get("type", "")
            parent = rec.get("parent")
            self.city_id_to_type[cid] = ctype
            self.city_id_to_parent[cid] = parent
            if parent is not None:
                self.parent_to_children[parent].append(cid)

    def _collect_descendants(self, root_id: str) -> Set[str]:
        result: Set[str] = set()
        stack = [root_id]
        while stack:
            cid = stack.pop()
            if cid in result:
                continue
            result.add(cid)
            for child in self.parent_to_children.get(cid, []):
                stack.append(child)
        return result

    def build_city_ids_for_types_with_descendants(self, types: List[str]) -> Dict[str, Set[str]]:
        type_to_root_ids: Dict[str, List[str]] = defaultdict(list)
        for cid, ctype in self.city_id_to_type.items():
            if ctype in types:
                type_to_root_ids[ctype].append(cid)

        type_to_ids: Dict[str, Set[str]] = {}
        for t, roots in type_to_root_ids.items():
            all_ids: Set[str] = set()
            for root in roots:
                all_ids.update(self._collect_descendants(root))
            type_to_ids[t] = all_ids
        return type_to_ids

    def build_instance_to_class_citygml(self, citygml_class_map: Dict[int, List[str]]) -> Dict[int, int]:
        all_types: Set[str] = set()
        for types in citygml_class_map.values():
            all_types.update(types)

        type_to_ids = self.build_city_ids_for_types_with_descendants(list(all_types))

        type_priority = ["Window", "Door", "RoofSurface", "WallSurface", "GroundSurface", "Building"]
        class_id_to_sorted_types: Dict[int, List[str]] = {}
        for cid, types in citygml_class_map.items():
            class_id_to_sorted_types[cid] = sorted(
                types, key=lambda t: type_priority.index(t) if t in type_priority else len(type_priority)
            )

        instance_to_class: Dict[int, int] = {}
        for inst_str, city_id in self.instance_to_city_id.items():
            inst_id = int(inst_str)
            chosen_class = None
            chosen_priority = len(type_priority) + 1

            for cid, types in class_id_to_sorted_types.items():
                for t in types:
                    ids_for_t = type_to_ids.get(t)
                    if ids_for_t and city_id in ids_for_t:
                        prio = type_priority.index(t) if t in type_priority else len(type_priority)
                        if prio < chosen_priority:
                            chosen_priority = prio
                            chosen_class = cid

            if chosen_class is not None:
                instance_to_class[inst_id] = chosen_class

        return instance_to_class

    def build_building_instance_set(self) -> Set[int]:
        type_to_ids = self.build_city_ids_for_types_with_descendants(["Building"])
        building_city_ids = type_to_ids.get("Building", set())

        building_instances: Set[int] = set()
        for inst_str, city_id in self.instance_to_city_id.items():
            inst_id = int(inst_str)
            if inst_id == 0:
                continue
            if city_id in building_city_ids:
                building_instances.add(inst_id)
        return building_instances


class CLIPInstanceIndex:
    def __init__(self, object_clip_index_path: str, class_mapping: Dict[int, Union[List[str], str]],
                 device=None, model_name=DEFAULT_OPENCLIP_MODEL_NAME, pretrained=DEFAULT_OPENCLIP_PRETRAINED):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

        loaded = np.load(object_clip_index_path)

        if isinstance(loaded, np.lib.npyio.NpzFile):
            features = loaded["features"].astype(np.float32)
            if "instance_ids" in loaded.files:
                instance_ids = loaded["instance_ids"].astype(np.int32)
            else:
                instance_ids = loaded["ids"].astype(np.int32)

            feat_norm = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
            features = features / feat_norm

            self.instance_mean_features = {}
            instance_to_indices = defaultdict(list)
            for idx, inst_id in enumerate(instance_ids.tolist()):
                instance_to_indices[int(inst_id)].append(idx)
            for inst_id, idxs in instance_to_indices.items():
                if inst_id != 0:
                    self.instance_mean_features[inst_id] = features[idxs].mean(axis=0)

        elif isinstance(loaded, np.ndarray):
            arr = loaded.astype(np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            valid = (norms.squeeze(-1) > 1e-8)
            arr_norm = np.zeros_like(arr)
            arr_norm[valid] = arr[valid] / norms[valid]
            self.instance_mean_features = {int(i): arr_norm[i] for i in np.where(valid)[0] if int(i) != 0}
        else:
            raise ValueError(f"Unsupported feature file format: {type(loaded)}")

        self.class_prompts = {}
        for cid, v in class_mapping.items():
            self.class_prompts[int(cid)] = v if isinstance(v, list) else [v]

    def _encode_texts(self, prompts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            tokens = self.tokenizer(prompts).to(self.device)
            text_feat = self.model.encode_text(tokens).float()
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        return text_feat

    def classify_instances(self, candidate_class_ids: Set[int], instance_ids_to_classify: Set[int], similarity_threshold: float) -> Dict[int, int]:
        all_inst_ids = set(self.instance_mean_features.keys())
        inst_ids = all_inst_ids.intersection(instance_ids_to_classify) if instance_ids_to_classify else all_inst_ids

        class_ids = []
        emb_list = []
        for cid, prompts in self.class_prompts.items():
            if cid in candidate_class_ids:
                text_emb = self._encode_texts(prompts)
                mean_emb = text_emb.mean(dim=0, keepdim=True)
                mean_emb = mean_emb / mean_emb.norm(dim=-1, keepdim=True)
                class_ids.append(cid)
                emb_list.append(mean_emb)

        if not emb_list:
            return {i: -1 for i in inst_ids}

        class_emb = torch.cat(emb_list, dim=0).to(self.device)

        instance_to_class = {}
        with torch.no_grad():
            for inst_id in inst_ids:
                inst_feat = self.instance_mean_features.get(inst_id)
                if inst_feat is None:
                    instance_to_class[inst_id] = -1
                    continue
                inst_feat_t = torch.from_numpy(inst_feat).to(self.device).view(1, -1)
                inst_feat_t = inst_feat_t / (inst_feat_t.norm(dim=-1, keepdim=True) + 1e-8)
                sim = inst_feat_t @ class_emb.T
                sim_val, best_idx = sim.max(dim=1)
                if float(sim_val) < similarity_threshold:
                    instance_to_class[inst_id] = -1
                else:
                    instance_to_class[inst_id] = class_ids[int(best_idx)]
        return instance_to_class


class EvaluationInstanceEngine:
    def __init__(self, id_mapping_path, city_semantics_path, object_clip_index_path, class_mapping,
                 citygml_class_map, clip_threshold, device=None,
                 model_name=DEFAULT_OPENCLIP_MODEL_NAME, pretrained=DEFAULT_OPENCLIP_PRETRAINED):
        self.class_mapping = class_mapping

        if citygml_class_map is None:
            citygml_class_map = {
                1: ["WallSurface"],
                2: ["Window"],
                3: ["Door"],
                10: ["GroundSurface"],
                12: ["RoofSurface"]
            }

        self.city_index = CityGMLSemanticIndex(id_mapping_path, city_semantics_path)

        # CityGML closed-vocab part assignment
        self.instance_to_class_city = self.city_index.build_instance_to_class_citygml(citygml_class_map)

        # Building set
        self.building_instance_ids: Set[int] = self.city_index.build_building_instance_set()

        # CLIP assignment for instances without City mapping
        self.clip_index = CLIPInstanceIndex(object_clip_index_path, class_mapping, device, model_name, pretrained)

        clip_class_ids = set(class_mapping.keys()) - set(citygml_class_map.keys())
        all_inst_ids = set(self.clip_index.instance_mean_features.keys())
        inst_without_city = all_inst_ids - set(self.instance_to_class_city.keys())

        self.instance_to_class_clip = self.clip_index.classify_instances(
            clip_class_ids,
            inst_without_city,
            clip_threshold
        )

        # Final semantic per-instance
        self.instance_to_class = {**self.instance_to_class_clip, **self.instance_to_class_city}

    def predict_instance_image(self, instance_img: np.ndarray) -> np.ndarray:
        h, w = instance_img.shape
        pred_semantic = np.full((h, w), -1, dtype=np.int32)
        unique_inst = np.unique(instance_img)
        for inst_id in unique_inst:
            cls_id = self.instance_to_class.get(int(inst_id), -1)
            if cls_id != -1:
                pred_semantic[instance_img == inst_id] = cls_id
        return pred_semantic

    def predict_building_image(self, instance_img: np.ndarray, building_id: int = 200) -> np.ndarray:
        h, w = instance_img.shape
        pred_building = np.full((h, w), -1, dtype=np.int32)
        if not self.building_instance_ids:
            return pred_building

        for inst_id in np.unique(instance_img):
            inst_id = int(inst_id)
            if inst_id == 0:
                continue
            if inst_id in self.building_instance_ids:
                pred_building[instance_img == inst_id] = int(building_id)
        return pred_building


# ============================================================================
# Main Evaluation Loop
# ============================================================================
def evaluate(
    instance_images_dir: str,
    gt_split_root: str,
    model_root: str,
    class_mapping: Dict[int, Union[str, List[str]]],
    output_dir: str,
    rgb_images_dir: Optional[str] = None,
    clip_threshold: float = 0.25,
    save_visualizations: bool = True,
    num_images: Optional[int] = None,
    class_colors_path: Optional[str] = None,
    gt_merge_map: Optional[Dict[int, List[int]]] = None,
    citygml_class_map: Optional[Dict[int, List[str]]] = None,
    logger: logging.Logger = None,
    openclip_model_name: str = DEFAULT_OPENCLIP_MODEL_NAME,
    openclip_pretrained: str = DEFAULT_OPENCLIP_PRETRAINED,
    building_id: int = DEFAULT_BUILDING_ID,
    parts_merge_target: int = DEFAULT_PARTS_MERGE_TARGET,
    part_ids: Optional[List[int]] = None,
    nonbuild_ids: Optional[List[int]] = None,
    gt_layer_building: str = DEFAULT_GT_LAYER_BUILDING,
    gt_layer_parts: str = DEFAULT_GT_LAYER_PARTS,
    gt_layer_nonbuild: str = DEFAULT_GT_LAYER_NONBUILD,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if part_ids is None:
        part_ids = DEFAULT_PART_IDS
    if nonbuild_ids is None:
        nonbuild_ids = DEFAULT_NONBUILD_IDS

    part_ids = [int(x) for x in part_ids]
    nonbuild_ids = [int(x) for x in nonbuild_ids]
    building_id = int(building_id)
    parts_merge_target = int(parts_merge_target)

    # Directories
    dir_fused = os.path.join(gt_split_root, "fused")
    dir_zaha = os.path.join(gt_split_root, "layer_zaha_kept")
    dir_ai = os.path.join(gt_split_root, "layer_ai_filled")

    logger.info("Target Evaluation Directories:")
    logger.info(f"  > Fused: {dir_fused}")
    logger.info(f"  > Zaha : {dir_zaha}")
    logger.info(f"  > AI   : {dir_ai}")

    logger.info(f"[CONFIG] building_id={building_id}")
    logger.info(f"[CONFIG] part_ids={part_ids}")
    logger.info(f"[CONFIG] nonbuild_ids={nonbuild_ids}")
    logger.info(f"[CONFIG] gt_layer_building={gt_layer_building}")
    logger.info(f"[CONFIG] gt_layer_parts={gt_layer_parts}")
    logger.info(f"[CONFIG] gt_layer_nonbuild={gt_layer_nonbuild}")

    # split merge maps
    parts_merge_map = None
    building_merge_map = None
    if gt_merge_map:
        if parts_merge_target in gt_merge_map:
            parts_merge_map = {parts_merge_target: gt_merge_map[parts_merge_target]}
        if building_id in gt_merge_map:
            building_merge_map = {building_id: gt_merge_map[building_id]}

    if building_merge_map is None:
        logger.warning("[WARN] No merge rule for building_id found in gt_merge_map. Building GT may be empty.")

    # Load Colors
    class_colors = {}
    colors_path = class_colors_path or os.path.join(gt_split_root, 'class_colors.json')
    if os.path.exists(colors_path):
        with open(colors_path, 'r') as f:
            class_colors = {int(k): v for k, v in json.load(f).items()}
        logger.info(f"[INFO] Loaded class colors from {colors_path}")

    # Paths
    id_mapping_path = os.path.join(model_root, "id_mapping.json")
    city_semantics_path = os.path.join(model_root, "city_semantics.json")

    # Auto-detect CLIP features
    object_clip_index_path = None
    for fn in ["clip_semantics.npy", "clip_features_fused.npy", "clip_features.npy", "object_clip_index.npz"]:
        p = os.path.join(model_root, fn)
        if os.path.exists(p):
            object_clip_index_path = p
            break
    if not object_clip_index_path:
        raise FileNotFoundError(f"No CLIP feature file found in {model_root}")

    # Init engine
    engine = EvaluationInstanceEngine(
        id_mapping_path=id_mapping_path,
        city_semantics_path=city_semantics_path,
        object_clip_index_path=object_clip_index_path,
        class_mapping=class_mapping,
        citygml_class_map=citygml_class_map,
        clip_threshold=clip_threshold,
        device=device,
        model_name=openclip_model_name,
        pretrained=openclip_pretrained
    )
    logger.info(f"[INFO] CityGML Building instances detected: {len(engine.building_instance_ids)}")
    logger.info(f"[DEBUG] Is instance 0 considered Building? {0 in engine.building_instance_ids}")

    # Load instance images
    image_files = sorted(glob.glob(os.path.join(instance_images_dir, '*.png')))
    if num_images:
        image_files = image_files[:num_images]
    image_names = [Path(f).stem for f in image_files]

    # Load GT layers (same as DINO script: only apply parts_merge_map here)
    gt_fused = load_ground_truth_layer(dir_fused, image_names, class_mapping, logger, parts_merge_map, "Fused")
    gt_zaha = load_ground_truth_layer(dir_zaha, image_names, class_mapping, logger, parts_merge_map, "Zaha")
    gt_ai = load_ground_truth_layer(dir_ai, image_names, class_mapping, logger, parts_merge_map, "AI")

    common = sorted(list(set(gt_fused.keys()) & set(image_names)))
    logger.info(f"Evaluating on {len(common)} images (must have fused GT + pred).")

    def pick_gt(img_name: str, which: str) -> Optional[Dict]:
        if which == "fused":
            return gt_fused.get(img_name)
        if which == "zaha":
            return gt_zaha.get(img_name)
        if which == "ai":
            return gt_ai.get(img_name)
        if which == "layer_zaha_kept":
            return gt_zaha.get(img_name)
        if which == "layer_ai_filled":
            return gt_ai.get(img_name)
        return gt_fused.get(img_name)

    # logs aligned with DINO 3-row script
    metrics_log = {
        "building": [],
        "parts": [],
        "nonbuilding": [],
    }

    vis_dir = os.path.join(output_dir, "visualizations_3rows")
    if save_visualizations:
        os.makedirs(vis_dir, exist_ok=True)

    for name in tqdm(common, desc="Evaluating"):
        img_file = os.path.join(instance_images_dir, f"{name}.png")

        inst_img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
        if inst_img is None:
            continue
        if inst_img.ndim == 3:
            inst_img = inst_img[:, :, 0]
        inst_img = inst_img.astype(np.int32)

        # resize to fused GT size as canonical
        gt_base = gt_fused[name]["semantic_map"]
        H, W = gt_base.shape
        if inst_img.shape != (H, W):
            inst_img = cv2.resize(inst_img, (W, H), interpolation=cv2.INTER_NEAREST)

        # 1) full semantic prediction
        pred_semantic = engine.predict_instance_image(inst_img)
        if parts_merge_map:
            pred_semantic = apply_label_merges(pred_semantic, parts_merge_map)

        if pred_semantic.shape != (H, W):
            pred_semantic = cv2.resize(pred_semantic.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST)

        # 2) building prediction (only CityGML closed-vocab Building)
        pred_building = engine.predict_building_image(inst_img, building_id=building_id)
        if pred_building.shape != (H, W):
            pred_building = cv2.resize(pred_building.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST)

        # 3) derive row-specific predictions from full semantic map
        pred_parts = pred_semantic.copy()
        pred_parts[~np.isin(pred_parts, part_ids)] = -1

        pred_non = pred_semantic.copy()
        pred_non[~np.isin(pred_non, nonbuild_ids)] = -1

        # -----------------------
        # Building GT + metrics
        # -----------------------
        gt_pack_b = pick_gt(name, gt_layer_building)
        if gt_pack_b is not None:
            gt_b = gt_pack_b["semantic_map"]
            gt_bm = apply_label_merges(gt_b, building_merge_map) if building_merge_map else gt_b

            gt_valid = (gt_bm >= 0)
            gt_is_b = (gt_bm == building_id)
            pred_is_b = (pred_building == building_id)

            eval_mask_b = gt_valid & (gt_is_b | pred_is_b)
            bm = compute_binary_metrics_union_mask(pred_is_b, gt_is_b, eval_mask_b)
            if bm:
                metrics_log["building"].append(bm)

        # -----------------------
        # Parts GT + metrics
        # -----------------------
        gt_pack_p = pick_gt(name, gt_layer_parts)
        if gt_pack_p is not None:
            gt_p = gt_pack_p["semantic_map"]
            eval_mask_p = (gt_p >= 0) & np.isin(gt_p, part_ids)
            pm = compute_multiclass_metrics(pred_parts, gt_p, eval_mask_p, class_mapping)
            if pm:
                metrics_log["parts"].append(pm)

        # -----------------------
        # Non-building GT + metrics
        # -----------------------
        gt_pack_n = pick_gt(name, gt_layer_nonbuild)
        if gt_pack_n is not None:
            gt_n = gt_pack_n["semantic_map"]
            eval_mask_n = (gt_n >= 0) & np.isin(gt_n, nonbuild_ids)
            nm = compute_multiclass_metrics(pred_non, gt_n, eval_mask_n, class_mapping)
            if nm:
                metrics_log["nonbuilding"].append(nm)

        # -----------------------
        # Visualization (3 rows)
        # -----------------------
        if save_visualizations and (gt_pack_b is not None) and (gt_pack_p is not None) and (gt_pack_n is not None):
            gt_b = gt_pack_b["semantic_map"]
            gt_bm = apply_label_merges(gt_b, building_merge_map) if building_merge_map else gt_b

            visualize_three_row_error(
                pred_building=pred_building,
                gt_building_merged=gt_bm,
                pred_parts=pred_parts,
                gt_parts=gt_pack_p["semantic_map"],
                pred_non=pred_non,
                gt_non=gt_pack_n["semantic_map"],
                class_colors=class_colors,
                save_path=os.path.join(vis_dir, f"{name}_3rows.png"),
                building_id=building_id,
                part_ids=part_ids,
                nonbuild_ids=nonbuild_ids,
            )

    # =========================================================================
    # Aggregate report (aligned with DINO 3-row script)
    # =========================================================================
    final_report = {}

    logger.info("\n" + "=" * 70)
    logger.info("FINAL 3-ROW EVALUATION RESULTS (GT coverage restricted)")
    logger.info("=" * 70)

    # building
    bdata = metrics_log["building"]
    if bdata:
        final_report["building"] = {
            "N": int(len(bdata)),
            "IoU": float(np.mean([d["iou"] for d in bdata])),
            "Precision": float(np.mean([d["precision"] for d in bdata])),
            "Recall": float(np.mean([d["recall"] for d in bdata])),
            "F1": float(np.mean([d["f1"] for d in bdata])),
        }
        logger.info(f"\n[BUILDING] (ID={building_id}, GT={gt_layer_building}, N={len(bdata)})")
        logger.info(f"  IoU:       {final_report['building']['IoU']:.4f}")
        logger.info(f"  Precision: {final_report['building']['Precision']:.4f}")
        logger.info(f"  Recall:    {final_report['building']['Recall']:.4f}")
        logger.info(f"  F1:        {final_report['building']['F1']:.4f}")
    else:
        logger.warning("\n[BUILDING] No valid metrics.")

    # parts
    pdata = metrics_log["parts"]
    if pdata:
        mean_iou = float(np.mean([d["mean_iou"] for d in pdata]))
        mean_acc = float(np.mean([d["pixel_acc"] for d in pdata]))
        class_ious = defaultdict(list)
        for d in pdata:
            for c, iou in d["class_iou"].items():
                class_ious[c].append(iou)
        per_class_iou = {c: float(np.mean(v)) for c, v in class_ious.items()}

        final_report["parts"] = {
            "N": int(len(pdata)),
            "mIoU": mean_iou,
            "pixel_acc": mean_acc,
            "per_class_iou": per_class_iou,
        }

        logger.info(f"\n[PARTS] (IDs={part_ids}, GT={gt_layer_parts}, N={len(pdata)})")
        logger.info(f"  mIoU:      {mean_iou:.4f}")
        logger.info(f"  Pixel Acc: {mean_acc:.4f}")
        logger.info("  Per Class IoU:")
        for c, iou in sorted(per_class_iou.items()):
            logger.info(f"    {c:<25}: {iou:.4f}")
    else:
        logger.warning("\n[PARTS] No valid metrics.")

    # nonbuilding
    ndata = metrics_log["nonbuilding"]
    if ndata:
        mean_iou = float(np.mean([d["mean_iou"] for d in ndata]))
        mean_acc = float(np.mean([d["pixel_acc"] for d in ndata]))
        class_ious = defaultdict(list)
        for d in ndata:
            for c, iou in d["class_iou"].items():
                class_ious[c].append(iou)
        per_class_iou = {c: float(np.mean(v)) for c, v in class_ious.items()}

        final_report["nonbuilding"] = {
            "N": int(len(ndata)),
            "mIoU": mean_iou,
            "pixel_acc": mean_acc,
            "per_class_iou": per_class_iou,
        }

        logger.info(f"\n[NON-BUILDING] (IDs={nonbuild_ids}, GT={gt_layer_nonbuild}, N={len(ndata)})")
        logger.info(f"  mIoU:      {mean_iou:.4f}")
        logger.info(f"  Pixel Acc: {mean_acc:.4f}")
        logger.info("  Per Class IoU:")
        for c, iou in sorted(per_class_iou.items()):
            logger.info(f"    {c:<25}: {iou:.4f}")
    else:
        logger.warning("\n[NON-BUILDING] No valid metrics.")

    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj

    out_json = os.path.join(output_dir, "three_rows_results.json")
    with open(out_json, "w") as f:
        json.dump(convert_to_serializable(final_report), f, indent=2)

    logger.info(f"\n3-row results saved to: {out_json}")
    logger.info(f"Visualizations saved to: {vis_dir}")


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate CityGML+CLIP with DINO-aligned 3-row output")

    parser.add_argument('--instance_images_dir', type=str, default=DEFAULT_INSTANCE_IMAGES_DIR)
    parser.add_argument('--gt_split_root', type=str, default=DEFAULT_GT_SPLIT_ROOT)
    parser.add_argument('--model_root', type=str, default=DEFAULT_MODEL_ROOT)
    parser.add_argument('--class_mapping', type=str, default=DEFAULT_CLASS_MAPPING_PATH)
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--rgb_images_dir', type=str, default=DEFAULT_RGB_IMAGES_DIR)
    parser.add_argument('--clip_threshold', type=float, default=DEFAULT_CLIP_THRESHOLD)
    parser.add_argument('--save_visualizations', action='store_true', default=DEFAULT_SAVE_VISUALIZATIONS)
    parser.add_argument('--num_images', type=int, default=DEFAULT_NUM_IMAGES)

    parser.add_argument('--class_colors', type=str, default=DEFAULT_CLASS_COLORS_PATH)
    parser.add_argument('--gt_merge_map', type=str, default=DEFAULT_GT_MERGE_MAP_PATH)
    parser.add_argument('--citygml_class_map', type=str, default=DEFAULT_CITYGML_CLASS_MAP_PATH)

    parser.add_argument('--openclip_model', type=str, default=DEFAULT_OPENCLIP_MODEL_NAME)
    parser.add_argument('--openclip_pretrained', type=str, default=DEFAULT_OPENCLIP_PRETRAINED)

    # 3-row params
    parser.add_argument('--building_id', type=int, default=DEFAULT_BUILDING_ID)
    parser.add_argument('--parts_merge_target', type=int, default=DEFAULT_PARTS_MERGE_TARGET)
    parser.add_argument('--part_ids', nargs="+", type=int, default=DEFAULT_PART_IDS)
    parser.add_argument('--nonbuild_ids', nargs="+", type=int, default=DEFAULT_NONBUILD_IDS)

    parser.add_argument('--gt_layer_building', type=str, default=DEFAULT_GT_LAYER_BUILDING, help="fused|zaha|ai")
    parser.add_argument('--gt_layer_parts', type=str, default=DEFAULT_GT_LAYER_PARTS, help="fused|zaha|ai")
    parser.add_argument('--gt_layer_nonbuild', type=str, default=DEFAULT_GT_LAYER_NONBUILD, help="fused|zaha|ai")

    args = parser.parse_args()

    with open(args.class_mapping, 'r') as f:
        class_mapping_str = json.load(f)
        class_mapping = {int(k): v for k, v in class_mapping_str.items()}

    os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger('citygml_clip_3rows_eval', log_file=os.path.join(args.output_dir, 'eval.log'))

    gt_merge_map = None
    if args.gt_merge_map and os.path.exists(args.gt_merge_map):
        with open(args.gt_merge_map, 'r') as f:
            merge_raw = json.load(f)
        gt_merge_map = {int(k): [int(v) for v in values] for k, values in merge_raw.items()}

    citygml_class_map = None
    if args.citygml_class_map and os.path.exists(args.citygml_class_map):
        with open(args.citygml_class_map, 'r') as f:
            raw = json.load(f)
        citygml_class_map = {int(k): v for k, v in raw.items()}

    evaluate(
        instance_images_dir=args.instance_images_dir,
        gt_split_root=args.gt_split_root,
        model_root=args.model_root,
        class_mapping=class_mapping,
        output_dir=args.output_dir,
        rgb_images_dir=args.rgb_images_dir,
        clip_threshold=args.clip_threshold,
        save_visualizations=args.save_visualizations,
        num_images=args.num_images,
        class_colors_path=args.class_colors,
        gt_merge_map=gt_merge_map,
        citygml_class_map=citygml_class_map,
        logger=logger,
        openclip_model_name=args.openclip_model,
        openclip_pretrained=args.openclip_pretrained,
        building_id=args.building_id,
        parts_merge_target=args.parts_merge_target,
        part_ids=args.part_ids,
        nonbuild_ids=args.nonbuild_ids,
        gt_layer_building=args.gt_layer_building,
        gt_layer_parts=args.gt_layer_parts,
        gt_layer_nonbuild=args.gt_layer_nonbuild,
    )


if __name__ == '__main__':
    main()
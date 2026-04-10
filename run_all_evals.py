#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run unified 2-level evaluation for multiple methods in one command.

Levels:
  1) whole: building-only binary evaluation
  2) part : global fine-grained semantic classes from class_mapping / ignore

Methods:
  1) CityGML + CLIP
  2) LangSplat
  3) Gaussian Grouping + GroundingDINO

GT logic:
  - do not use fused GT
  - use zaha GT + ai GT
  - fuse them with zaha priority
  - whole GT is derived from fused fine GT as building-only binary GT
  - part GT is the fused fine GT itself
"""

import argparse
import os
import json
import glob
from pathlib import Path
from typing import Dict, List, Optional, Union, Set, Tuple
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from common_eval import (
    UnifiedTwoLevelEvaluator,
    get_logger,
    load_json_int_keys,
    convert_to_serializable,
    create_cross_method_prediction_panels,
    require_complete_gt_pair,
)

import open_clip
from autoencoder.model import Autoencoder
from GS_eval.autoencoder.openclip_encoder import OpenCLIPNetwork
from GroundingDINO.groundingdino.util.inference import load_model as dino_load_model
from GroundingDINO.groundingdino.util.inference import predict as dino_predict


# =========================================================
# Default configuration
# =========================================================

DEFAULT_CLASS_MAPPING_PATH = "/workspace/zaha_eval/config/class_mapping.json"
DEFAULT_GT_SPLIT_ROOT = "/workspace/zaha_eval/gt/subset1_499_test"
DEFAULT_CLASS_COLORS_PATH = "/workspace/zaha_eval/config/class_colors.json"
DEFAULT_ROOT_OUTPUT_DIR = "/workspace/zaha_eval/all_eval_1_ab5"

DEFAULT_SAVE_VISUALIZATIONS = True
DEFAULT_SAVE_CROSS_METHOD_PANELS = True
DEFAULT_NUM_IMAGES = None

DEFAULT_WHOLE_BUILDING_ID = 200
DEFAULT_WHOLE_NONBUILDING_ID = 201

DEFAULT_WHOLE_BUILDING_FINE_IDS = [1, 2, 3, 12]
DEFAULT_WHOLE_NONBUILD_FINE_IDS = [101, 103, 104]

DEFAULT_RGB_IMAGE_DIR = "/workspace/CityGMLGaussian/dataset/subset_building1/images"

DEFAULT_RUN_CITYGML_CLIP = True
DEFAULT_CITY_INSTANCE_IMAGES_DIR = "/workspace/CityGMLGaussian/output/subset_building1_ab5_30000/test/ours_30000/objects_test"
DEFAULT_CITY_MODEL_ROOT = "/workspace/CityGMLGaussian/output/subset_building1_ab5_30000"
DEFAULT_CITY_CLIP_THRESHOLD = 0.2
DEFAULT_CITY_OPENCLIP_MODEL_NAME = "ViT-B-16"
DEFAULT_CITY_OPENCLIP_PRETRAINED = "laion2b_s34b_b88k"
DEFAULT_CITYGML_CLASS_MAP_PATH = None

DEFAULT_RUN_LANGSPLAT = False
DEFAULT_LANG_RENDERED_FEATURES_DIR = "/workspace/LangSplat/output/subset_building8_lang_30000/test/ours_None/renders_npy"
DEFAULT_LANG_AE_CHECKPOINT = "/workspace/LangSplat/autoencoder/ckpt/subset_building8/best_ckpt.pth"
DEFAULT_LANG_WHOLE_MASK_THRESH = 0.35
DEFAULT_LANG_PART_MASK_THRESH = 0.5
DEFAULT_LANG_USE_SOFTMAX = False
DEFAULT_LANG_ENCODER_DIMS = [256, 128, 64, 32, 3]
DEFAULT_LANG_DECODER_DIMS = [16, 32, 64, 128, 256, 256, 512]

DEFAULT_RUN_GAGA_DINO = False
DEFAULT_GAGA_IMAGES_DIR = "/workspace/CityGMLGaussian/dataset/subset_building8_gaga/images"
DEFAULT_GAGA_PRED_INST_DIR = "/workspace/CityGMLGaussian/output/subset_building8_gaga_30000/test/ours_30000/objects_test"
DEFAULT_DINO_CONFIG = "/workspace/zaha_eval/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DEFAULT_DINO_CHECKPOINT = "/workspace/zaha_eval/GroundingDINO/weights/groundingdino_swint_ogc.pth"
DEFAULT_DINO_BOX_THRESH = 0.25
DEFAULT_DINO_TEXT_THRESH = 0.20
DEFAULT_DINO_INSTANCE_MIN_OVERLAP = 0.30
DEFAULT_DINO_INSTANCE_MIN_SCORE = 0.20

DEFAULT_SUMMARY_FILENAME = "summary_all_methods.json"
DEFAULT_STABILITY_IDS = [101, 103, 104]


# =========================================================
# Helpers
# =========================================================

def parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return []
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_bool_flag_pair(enable_flag: bool, disable_flag: bool, default: bool) -> bool:
    if enable_flag and disable_flag:
        raise ValueError("Conflicting flags: both enable and disable were specified.")
    if enable_flag:
        return True
    if disable_flag:
        return False
    return default


# =========================================================
# Shared predictor base
# =========================================================
class BasePredictor:
    method_name = "base"
    vis_titles = {}

    def setup(self):
        pass

    def list_image_names(self) -> List[str]:
        raise NotImplementedError

    def required_paths(self, image_name: str) -> List[Dict]:
        raise NotImplementedError

    def predict(self, image_name: str) -> Dict[str, np.ndarray]:
        raise NotImplementedError


# =========================================================
# CityGML + CLIP internals
# =========================================================
class CityGMLSemanticIndex:
    def __init__(self, id_mapping_path: str, city_semantics_path: str):
        with open(id_mapping_path, "r", encoding="utf-8") as f:
            self.instance_to_city_id: Dict[str, str] = json.load(f)
        with open(city_semantics_path, "r", encoding="utf-8") as f:
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
                types,
                key=lambda t: type_priority.index(t) if t in type_priority else len(type_priority),
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


class CLIPInstanceIndex:
    def __init__(
        self,
        object_clip_index_path: str,
        class_mapping: Dict[int, Union[List[str], str]],
        device=None,
        model_name="ViT-B-32-quickgelu",
        pretrained="openai",
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = self.model.to(self.device)

        if self.device.type == "cpu":
            self.model = self.model.float()

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
            valid = norms.squeeze(-1) > 1e-8
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

    def classify_instances(
        self,
        candidate_class_ids: Set[int],
        instance_ids_to_classify: Optional[Set[int]],
        similarity_threshold: float,
    ) -> Dict[int, int]:
        all_inst_ids = set(self.instance_mean_features.keys())
        inst_ids = all_inst_ids.intersection(instance_ids_to_classify) if instance_ids_to_classify is not None else all_inst_ids

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
    def __init__(
        self,
        id_mapping_path,
        city_semantics_path,
        object_clip_index_path,
        class_mapping,
        citygml_class_map,
        clip_threshold,
        device=None,
        model_name="ViT-B-32-quickgelu",
        pretrained="openai",
    ):
        self.class_mapping = class_mapping

        if citygml_class_map is None:
            citygml_class_map = {
                1: ["WallSurface"],
                2: ["Window"],
                3: ["Door"],
                10: ["GroundSurface"],
                12: ["RoofSurface"],
            }

        self.city_index = CityGMLSemanticIndex(id_mapping_path, city_semantics_path)
        self.instance_to_class_city = self.city_index.build_instance_to_class_citygml(citygml_class_map)

        self.clip_index = CLIPInstanceIndex(object_clip_index_path, class_mapping, device, model_name, pretrained)

        clip_class_ids = set(class_mapping.keys()) - set(citygml_class_map.keys())
        all_inst_ids = set(self.clip_index.instance_mean_features.keys())
        inst_without_city = all_inst_ids - set(self.instance_to_class_city.keys())

        self.instance_to_class_clip = self.clip_index.classify_instances(
            clip_class_ids,
            inst_without_city,
            clip_threshold,
        )

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


class CityGMLClipPredictor(BasePredictor):
    method_name = "citygml_clip"

    def __init__(
        self,
        instance_images_dir,
        model_root,
        class_mapping,
        whole_building_fine_ids,
        clip_threshold=0.0,
        citygml_class_map=None,
        openclip_model_name="ViT-B-32-quickgelu",
        openclip_pretrained="openai",
        num_images=None,
        logger=None,
    ):
        self.instance_images_dir = instance_images_dir
        self.model_root = model_root
        self.class_mapping = class_mapping
        self.whole_building_fine_ids = set(int(x) for x in whole_building_fine_ids)
        self.clip_threshold = clip_threshold
        self.citygml_class_map = citygml_class_map
        self.openclip_model_name = openclip_model_name
        self.openclip_pretrained = openclip_pretrained
        self.num_images = num_images
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.engine = None

    def setup(self):
        if self.engine is not None:
            return

        id_mapping_path = os.path.join(self.model_root, "id_mapping.json")
        city_semantics_path = os.path.join(self.model_root, "city_semantics.json")

        object_clip_index_path = None
        for fn in ["clip_semantics.npy", "object_clip_index.npz"]:
            p = os.path.join(self.model_root, fn)
            if os.path.exists(p):
                object_clip_index_path = p
                break
        if not object_clip_index_path:
            raise FileNotFoundError(f"No CLIP feature file found in {self.model_root}")

        self.engine = EvaluationInstanceEngine(
            id_mapping_path=id_mapping_path,
            city_semantics_path=city_semantics_path,
            object_clip_index_path=object_clip_index_path,
            class_mapping=self.class_mapping,
            citygml_class_map=self.citygml_class_map,
            clip_threshold=self.clip_threshold,
            device=self.device,
            model_name=self.openclip_model_name,
            pretrained=self.openclip_pretrained,
        )

    def list_image_names(self):
        files = sorted(glob.glob(os.path.join(self.instance_images_dir, "*.png")))
        if self.num_images:
            files = files[:self.num_images]
        return [Path(f).stem for f in files]

    def required_paths(self, image_name: str) -> List[Dict]:
        return [
            {
                "label": "instance_image",
                "path": os.path.join(self.instance_images_dir, f"{image_name}.png"),
            }
        ]

    def predict(self, image_name):
        img_file = os.path.join(self.instance_images_dir, f"{image_name}.png")
        inst_img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
        if inst_img is None:
            raise FileNotFoundError(img_file)
        if inst_img.ndim == 3:
            inst_img = inst_img[:, :, 0]
        inst_img = inst_img.astype(np.int32)

        pred_part = self.engine.predict_instance_image(inst_img)

        pred_whole = np.full_like(pred_part, -1, dtype=np.int32)
        pred_whole[np.isin(pred_part, list(self.whole_building_fine_ids))] = 1

        return {
            "pred_whole": pred_whole,
            "pred_part": pred_part,
        }


# =========================================================
# LangSplat predictor
# =========================================================
class LangSplatPredictor(BasePredictor):
    method_name = "langsplat"

    def __init__(
        self,
        rendered_features,
        ae_checkpoint,
        class_mapping,
        whole_mask_thresh=0.35,
        part_mask_thresh=0.4,
        use_softmax=False,
        encoder_dims=None,
        decoder_dims=None,
        num_images=None,
    ):
        self.rendered_features = rendered_features
        self.ae_checkpoint = ae_checkpoint
        self.class_mapping = class_mapping
        self.whole_mask_thresh = float(whole_mask_thresh)
        self.part_mask_thresh = float(part_mask_thresh)
        self.use_softmax = bool(use_softmax)
        self.encoder_dims = encoder_dims or [256, 128, 64, 32, 3]
        self.decoder_dims = decoder_dims or [16, 32, 64, 128, 256, 256, 512]
        self.num_images = num_images
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.clip_model = None
        self.autoencoder = None
        self.whole_queries = ["building"]
        self.part_queries = None
        self.part_query_ids_np = None

    def setup(self):
        if self.clip_model is not None:
            return

        self.clip_model = OpenCLIPNetwork(self.device)
        self.autoencoder = Autoencoder(self.encoder_dims, self.decoder_dims).to(self.device)
        self.autoencoder.load_state_dict(torch.load(self.ae_checkpoint, map_location=self.device))
        self.autoencoder.eval()

        self.part_queries, self.part_query_ids_np = self._build_queries_from_mapping(self.class_mapping)

        if len(self.part_queries) == 0:
            raise ValueError("No part-level prompts found.")

    def list_image_names(self):
        files = sorted(glob.glob(os.path.join(self.rendered_features, "*.npy")))
        if self.num_images:
            files = files[:self.num_images]
        return [Path(f).stem for f in files]

    def required_paths(self, image_name: str) -> List[Dict]:
        return [
            {
                "label": "rendered_feature",
                "path": os.path.join(self.rendered_features, f"{image_name}.npy"),
            }
        ]

    def _build_queries_from_mapping(self, mapping):
        queries, qids = [], []
        for cid, lbl in mapping.items():
            if isinstance(lbl, list):
                queries.extend(lbl)
                qids.extend([int(cid)] * len(lbl))
            else:
                queries.append(lbl)
                qids.append(int(cid))
        return queries, np.array(qids, dtype=np.int32)

    def decode_features(self, compressed_features):
        h, w, _ = compressed_features.shape
        with torch.no_grad():
            flat = compressed_features.reshape(-1, 3).to(self.device)
            decoded = self.autoencoder.decode(flat).reshape(h, w, 512)
        return decoded

    def query_semantic_map(self, features_512, queries):
        h, w, _ = features_512.shape
        self.clip_model.set_positives(queries)
        features_input = features_512.unsqueeze(0)

        with torch.no_grad():
            relevance_maps = self.clip_model.get_max_across(features_input).squeeze(0)

        if self.use_softmax:
            probs = torch.softmax(relevance_maps, dim=0)
            pred_class = torch.argmax(probs, dim=0).cpu().numpy()
            confidence = torch.max(probs, dim=0)[0].cpu().numpy()
        else:
            argmax_class = relevance_maps.argmax(dim=0).cpu().numpy()
            confidence = np.zeros((h, w), dtype=np.float32)
            for i in range(len(queries)):
                rel = relevance_maps[i].cpu().numpy()
                rel_norm = (rel - rel.min()) / (rel.max() - rel.min() + 1e-9)
                mask = argmax_class == i
                confidence[mask] = rel_norm[mask]
            pred_class = argmax_class

        return pred_class, confidence

    def predict(self, image_name):
        feat_path = os.path.join(self.rendered_features, f"{image_name}.npy")
        if not os.path.exists(feat_path):
            raise FileNotFoundError(feat_path)

        compressed = torch.from_numpy(np.load(feat_path)).float()
        decoded = self.decode_features(compressed)

        raw_idx_w, conf_w = self.query_semantic_map(decoded, self.whole_queries)
        pred_whole = np.full_like(raw_idx_w, -1, dtype=np.int32)
        pred_whole[conf_w >= self.whole_mask_thresh] = 1

        raw_idx_p, conf_p = self.query_semantic_map(decoded, self.part_queries)
        pred_part = np.full_like(raw_idx_p, -1, dtype=np.int32)
        valid_p = raw_idx_p >= 0
        pred_part[valid_p] = self.part_query_ids_np[raw_idx_p[valid_p]]
        pred_part[conf_p < self.part_mask_thresh] = -1

        return {
            "pred_whole": pred_whole,
            "pred_part": pred_part,
        }


# =========================================================
# Gaussian Grouping + DINO predictor
# =========================================================
class GaussianGroupingDINOPredictor(BasePredictor):
    method_name = "gaga_dino"

    def __init__(
        self,
        pred_inst_dir,
        images_dir,
        class_mapping,
        dino_config,
        dino_checkpoint,
        box_thresh=0.35,
        text_thresh=0.25,
        inst_min_overlap=0.30,
        inst_min_score=0.20,
        num_images=None,
        device=None,
    ):
        self.pred_inst_dir = pred_inst_dir
        self.images_dir = images_dir
        self.class_mapping = class_mapping
        self.dino_config = dino_config
        self.dino_checkpoint = dino_checkpoint
        self.box_thresh = float(box_thresh)
        self.text_thresh = float(text_thresh)
        self.inst_min_overlap = float(inst_min_overlap)
        self.inst_min_score = float(inst_min_score)
        self.num_images = num_images
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.dino_model = None
        self.whole_queries = ["building"]
        self.whole_qids = np.array([1], dtype=np.int32)
        self.part_queries = None
        self.part_qids = None

    def setup(self):
        if self.dino_model is not None:
            return

        if not os.path.exists(self.dino_config):
            raise FileNotFoundError(f"DINO config not found: {self.dino_config}")
        if not os.path.exists(self.dino_checkpoint):
            raise FileNotFoundError(f"DINO checkpoint not found: {self.dino_checkpoint}")

        self.dino_model = dino_load_model(self.dino_config, self.dino_checkpoint).to(self.device)
        self.dino_model.eval()

        self.part_queries, self.part_qids = self._build_queries_from_mapping(self.class_mapping)

    def list_image_names(self):
        files = sorted(glob.glob(os.path.join(self.pred_inst_dir, "*.png")))
        if self.num_images:
            files = files[:self.num_images]
        return [Path(f).stem for f in files]

    def _rgb_candidates(self, image_name: str) -> List[str]:
        return [
            os.path.join(self.images_dir, image_name + ext)
            for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]
        ]

    def required_paths(self, image_name: str) -> List[Dict]:
        return [
            {
                "label": "pred_instance_image",
                "path": os.path.join(self.pred_inst_dir, f"{image_name}.png"),
            },
            {
                "label": "rgb_image",
                "any_of": True,
                "paths": self._rgb_candidates(image_name),
            },
        ]

    def _build_queries_from_mapping(self, mapping):
        queries, qids = [], []
        for cid, lbl in mapping.items():
            if isinstance(lbl, list):
                if len(lbl) > 0:
                    queries.append(str(lbl[0]))
                    qids.append(int(cid))
            else:
                queries.append(str(lbl))
                qids.append(int(cid))
        return queries, np.array(qids, dtype=np.int32)

    def _load_rgb(self, image_name):
        for p in self._rgb_candidates(image_name):
            if os.path.exists(p):
                bgr = cv2.imread(p, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise FileNotFoundError(p)
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        raise FileNotFoundError(f"RGB image not found for {image_name}")

    def _preprocess_for_dino(self, image_rgb: np.ndarray, target_size=384, max_size=640) -> Tuple[torch.Tensor, Tuple[int, int]]:
        pil = Image.fromarray(image_rgb)
        w, h = pil.size

        scale = target_size / min(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        if max(new_h, new_w) > max_size:
            scale = max_size / max(new_h, new_w)
            new_h, new_w = int(round(new_h * scale)), int(round(new_w * scale))

        pil = pil.resize((new_w, new_h), resample=Image.BILINEAR)

        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return tfm(pil), (new_h, new_w)

    def dino_detect(self, image_rgb: np.ndarray, prompt: str):
        img_t, (new_h, new_w) = self._preprocess_for_dino(image_rgb, target_size=800, max_size=1333)

        def _run(dev: str, img_tensor: torch.Tensor):
            with torch.no_grad():
                if dev.startswith("cuda"):
                    with torch.cuda.amp.autocast(True):
                        return dino_predict(
                            model=self.dino_model,
                            image=img_tensor.to(dev),
                            caption=prompt,
                            box_threshold=self.box_thresh,
                            text_threshold=self.text_thresh,
                            device=dev,
                        )
                return dino_predict(
                    model=self.dino_model,
                    image=img_tensor.to(dev),
                    caption=prompt,
                    box_threshold=self.box_thresh,
                    text_threshold=self.text_thresh,
                    device=dev,
                )

        try:
            boxes, logits, _phrases = _run(self.device, img_t)
        except torch.cuda.OutOfMemoryError:
            if str(self.device).startswith("cuda"):
                torch.cuda.empty_cache()
            img_small, (new_h, new_w) = self._preprocess_for_dino(image_rgb, target_size=640, max_size=1024)
            try:
                boxes, logits, _phrases = _run(self.device, img_small)
            except torch.cuda.OutOfMemoryError:
                boxes, logits, _phrases = _run("cpu", img_t)

        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(logits, dtype=np.float32).reshape(-1)

        if boxes.size == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        h0, w0 = image_rgb.shape[:2]

        if np.max(boxes) <= 1.5:
            boxes_scaled = boxes.copy()
            boxes_scaled[:, [0, 2]] *= float(new_w)
            boxes_scaled[:, [1, 3]] *= float(new_h)
        else:
            boxes_scaled = boxes.copy()

        bad_xyxy = np.mean((boxes_scaled[:, 2] < boxes_scaled[:, 0]) | (boxes_scaled[:, 3] < boxes_scaled[:, 1]))
        if bad_xyxy > 0.3:
            cx = boxes_scaled[:, 0]
            cy = boxes_scaled[:, 1]
            w = boxes_scaled[:, 2]
            h = boxes_scaled[:, 3]
            x1 = cx - 0.5 * w
            y1 = cy - 0.5 * h
            x2 = cx + 0.5 * w
            y2 = cy + 0.5 * h
            boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        else:
            boxes_xyxy = boxes_scaled

        sx = w0 / float(new_w)
        sy = h0 / float(new_h)
        boxes_xyxy[:, [0, 2]] *= sx
        boxes_xyxy[:, [1, 3]] *= sy

        x1 = np.minimum(boxes_xyxy[:, 0], boxes_xyxy[:, 2])
        y1 = np.minimum(boxes_xyxy[:, 1], boxes_xyxy[:, 3])
        x2 = np.maximum(boxes_xyxy[:, 0], boxes_xyxy[:, 2])
        y2 = np.maximum(boxes_xyxy[:, 1], boxes_xyxy[:, 3])

        x1 = np.clip(x1, 0, w0 - 1)
        y1 = np.clip(y1, 0, h0 - 1)
        x2 = np.clip(x2, 0, w0 - 1)
        y2 = np.clip(y2, 0, h0 - 1)

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        return boxes_xyxy.astype(np.float32), scores.astype(np.float32)

    def overlap_ratio_instance_in_box(self, inst_mask: np.ndarray, box_xyxy: np.ndarray) -> float:
        x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy.tolist()]
        h, w = inst_mask.shape
        x1 = max(0, min(w, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inst_area = float(inst_mask.sum())
        if inst_area <= 0:
            return 0.0
        inter = float(inst_mask[y1:y2, x1:x2].sum())
        return inter / inst_area

    def label_instances_with_dino(
        self,
        pred_inst: np.ndarray,
        image_rgb: np.ndarray,
        queries: List[str],
        query_ids: np.ndarray,
    ) -> Dict[int, int]:
        inst_ids = np.unique(pred_inst)
        inst_ids = inst_ids[(inst_ids != 0) & (inst_ids != -1)]

        if len(inst_ids) == 0 or len(queries) == 0:
            return {}

        hp, wp = pred_inst.shape
        hr, wr = image_rgb.shape[:2]

        sx = wp / float(wr)
        sy = hp / float(hr)

        best_class = {int(iid): -1 for iid in inst_ids}
        best_score = {int(iid): 0.0 for iid in inst_ids}

        for qi, prompt in enumerate(queries):
            cid = int(query_ids[qi])
            boxes, scores = self.dino_detect(image_rgb, prompt)

            if boxes.shape[0] == 0:
                continue

            boxes_small = boxes.copy()
            boxes_small[:, [0, 2]] *= sx
            boxes_small[:, [1, 3]] *= sy

            for bi in range(boxes_small.shape[0]):
                b = boxes_small[bi]
                s = float(scores[bi])

                for iid in inst_ids:
                    iid = int(iid)
                    m = pred_inst == iid
                    overlap = self.overlap_ratio_instance_in_box(m, b)
                    if overlap < self.inst_min_overlap:
                        continue

                    sc = s * overlap
                    if sc > best_score[iid]:
                        best_score[iid] = sc
                        best_class[iid] = cid

        out = {}
        for iid in inst_ids:
            iid = int(iid)
            if best_class[iid] >= 0 and best_score[iid] >= self.inst_min_score:
                out[iid] = int(best_class[iid])

        return out

    def rasterize_instance_labels(self, pred_inst: np.ndarray, inst2cls: Dict[int, int]) -> np.ndarray:
        pred_sem = np.full_like(pred_inst, -1, dtype=np.int32)
        for iid, cid in inst2cls.items():
            pred_sem[pred_inst == iid] = int(cid)
        return pred_sem

    def predict(self, image_name):
        pred_path = os.path.join(self.pred_inst_dir, f"{image_name}.png")
        pred_inst = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)
        if pred_inst is None:
            raise FileNotFoundError(pred_path)
        if pred_inst.ndim == 3:
            pred_inst = pred_inst[:, :, 0]
        pred_inst = pred_inst.astype(np.int32)

        image_rgb = self._load_rgb(image_name)

        inst2whole = self.label_instances_with_dino(pred_inst, image_rgb, self.whole_queries, self.whole_qids)
        inst2part = self.label_instances_with_dino(pred_inst, image_rgb, self.part_queries, self.part_qids)

        pred_whole = self.rasterize_instance_labels(pred_inst, inst2whole)
        pred_part = self.rasterize_instance_labels(pred_inst, inst2part)

        return {
            "pred_whole": pred_whole,
            "pred_part": pred_part,
        }


# =========================================================
# Utilities
# =========================================================
def maybe_load_colors(path: Optional[str]):
    if path is None or not os.path.exists(path):
        return {}
    return load_json_int_keys(path)


def maybe_load_citygml_class_map(path: Optional[str]):
    if path is None:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    raw = load_json_int_keys(path)
    return {int(k): v for k, v in raw.items()}


def save_summary_table(root_output_dir: str, summary_filename: str, all_reports: Dict[str, Dict]):
    os.makedirs(root_output_dir, exist_ok=True)
    out_path = os.path.join(root_output_dir, summary_filename)

    existing_reports = {}
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                existing_reports = json.load(f)
        except Exception:
            existing_reports = {}

    existing_reports.update(convert_to_serializable(all_reports))

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(existing_reports, f, indent=2)

    return out_path


def validate_predictor_inputs_or_raise(
    predictor: BasePredictor,
    image_names: List[str],
    logger,
):
    missing_msgs = []

    for name in image_names:
        for req in predictor.required_paths(name):
            label = req.get("label", "input")
            if req.get("any_of", False):
                candidates = req.get("paths", [])
                if not any(os.path.exists(p) for p in candidates):
                    missing_msgs.append(
                        f"[{predictor.method_name} missing] image='{name}' missing {label}; checked: {candidates}"
                    )
            else:
                path = req.get("path")
                if path is None or not os.path.exists(path):
                    missing_msgs.append(
                        f"[{predictor.method_name} missing] image='{name}' missing {label}: {path}"
                    )

    if missing_msgs:
        for msg in missing_msgs[:200]:
            logger.error(msg)
        if len(missing_msgs) > 200:
            logger.error(f"... and {len(missing_msgs) - 200} more missing predictor records.")
        raise FileNotFoundError(
            f"Input validation failed for method '{predictor.method_name}'. See the log for details."
        )


# =========================================================
# Argument parser
# =========================================================
def build_argparser():
    parser = argparse.ArgumentParser(
        description="Run unified 2-level evaluation for GS4City, LangSplat, and Gaga + GroundingDINO."
    )

    parser.add_argument("--class_mapping_path", type=str, default=DEFAULT_CLASS_MAPPING_PATH)
    parser.add_argument("--gt_split_root", type=str, default=DEFAULT_GT_SPLIT_ROOT)
    parser.add_argument("--class_colors_path", type=str, default=DEFAULT_CLASS_COLORS_PATH)
    parser.add_argument("--root_output_dir", type=str, default=DEFAULT_ROOT_OUTPUT_DIR)

    parser.add_argument("--save_visualizations", action="store_true", default=DEFAULT_SAVE_VISUALIZATIONS)
    parser.add_argument("--no_save_visualizations", action="store_false", dest="save_visualizations")

    parser.add_argument("--save_cross_method_panels", action="store_true", default=DEFAULT_SAVE_CROSS_METHOD_PANELS)
    parser.add_argument("--no_save_cross_method_panels", action="store_false", dest="save_cross_method_panels")

    parser.add_argument("--num_images", type=int, default=DEFAULT_NUM_IMAGES)

    parser.add_argument("--whole_building_id", type=int, default=DEFAULT_WHOLE_BUILDING_ID)
    parser.add_argument("--whole_nonbuilding_id", type=int, default=DEFAULT_WHOLE_NONBUILDING_ID)
    parser.add_argument(
        "--whole_building_fine_ids",
        type=str,
        default=",".join(map(str, DEFAULT_WHOLE_BUILDING_FINE_IDS)),
    )
    parser.add_argument(
        "--whole_nonbuild_fine_ids",
        type=str,
        default=",".join(map(str, DEFAULT_WHOLE_NONBUILD_FINE_IDS)),
    )

    parser.add_argument("--rgb_image_dir", type=str, default=DEFAULT_RGB_IMAGE_DIR)

    parser.add_argument("--run_citygml_clip", action="store_true")
    parser.add_argument("--skip_citygml_clip", action="store_true")
    parser.add_argument("--city_instance_images_dir", type=str, default=DEFAULT_CITY_INSTANCE_IMAGES_DIR)
    parser.add_argument("--city_model_root", type=str, default=DEFAULT_CITY_MODEL_ROOT)
    parser.add_argument("--city_clip_threshold", type=float, default=DEFAULT_CITY_CLIP_THRESHOLD)
    parser.add_argument("--city_openclip_model_name", type=str, default=DEFAULT_CITY_OPENCLIP_MODEL_NAME)
    parser.add_argument("--city_openclip_pretrained", type=str, default=DEFAULT_CITY_OPENCLIP_PRETRAINED)
    parser.add_argument("--citygml_class_map_path", type=str, default=DEFAULT_CITYGML_CLASS_MAP_PATH)

    parser.add_argument("--run_langsplat", action="store_true")
    parser.add_argument("--skip_langsplat", action="store_true")
    parser.add_argument("--lang_rendered_features_dir", type=str, default=DEFAULT_LANG_RENDERED_FEATURES_DIR)
    parser.add_argument("--lang_ae_checkpoint", type=str, default=DEFAULT_LANG_AE_CHECKPOINT)
    parser.add_argument("--lang_whole_mask_thresh", type=float, default=DEFAULT_LANG_WHOLE_MASK_THRESH)
    parser.add_argument("--lang_part_mask_thresh", type=float, default=DEFAULT_LANG_PART_MASK_THRESH)
    parser.add_argument("--lang_use_softmax", action="store_true")
    parser.add_argument("--lang_no_softmax", action="store_true")
    parser.add_argument(
        "--lang_encoder_dims",
        type=str,
        default=",".join(map(str, DEFAULT_LANG_ENCODER_DIMS)),
    )
    parser.add_argument(
        "--lang_decoder_dims",
        type=str,
        default=",".join(map(str, DEFAULT_LANG_DECODER_DIMS)),
    )

    parser.add_argument("--run_gaga_dino", action="store_true")
    parser.add_argument("--skip_gaga_dino", action="store_true")
    parser.add_argument("--gaga_images_dir", type=str, default=DEFAULT_GAGA_IMAGES_DIR)
    parser.add_argument("--gaga_pred_inst_dir", type=str, default=DEFAULT_GAGA_PRED_INST_DIR)
    parser.add_argument("--dino_config", type=str, default=DEFAULT_DINO_CONFIG)
    parser.add_argument("--dino_checkpoint", type=str, default=DEFAULT_DINO_CHECKPOINT)
    parser.add_argument("--dino_box_thresh", type=float, default=DEFAULT_DINO_BOX_THRESH)
    parser.add_argument("--dino_text_thresh", type=float, default=DEFAULT_DINO_TEXT_THRESH)
    parser.add_argument("--dino_instance_min_overlap", type=float, default=DEFAULT_DINO_INSTANCE_MIN_OVERLAP)
    parser.add_argument("--dino_instance_min_score", type=float, default=DEFAULT_DINO_INSTANCE_MIN_SCORE)
    parser.add_argument("--gaga_device", type=str, default=None)

    parser.add_argument("--summary_filename", type=str, default=DEFAULT_SUMMARY_FILENAME)
    parser.add_argument(
        "--stability_ids",
        type=str,
        default=",".join(map(str, DEFAULT_STABILITY_IDS)),
    )

    return parser


# =========================================================
# Main
# =========================================================
def main():
    parser = build_argparser()
    args = parser.parse_args()

    run_citygml_clip = parse_bool_flag_pair(args.run_citygml_clip, args.skip_citygml_clip, DEFAULT_RUN_CITYGML_CLIP)
    run_langsplat = parse_bool_flag_pair(args.run_langsplat, args.skip_langsplat, DEFAULT_RUN_LANGSPLAT)
    run_gaga_dino = parse_bool_flag_pair(args.run_gaga_dino, args.skip_gaga_dino, DEFAULT_RUN_GAGA_DINO)
    lang_use_softmax = parse_bool_flag_pair(args.lang_use_softmax, args.lang_no_softmax, DEFAULT_LANG_USE_SOFTMAX)

    whole_building_fine_ids = parse_int_list(args.whole_building_fine_ids) or []
    whole_nonbuild_fine_ids = parse_int_list(args.whole_nonbuild_fine_ids) or []
    lang_encoder_dims = parse_int_list(args.lang_encoder_dims) or []
    lang_decoder_dims = parse_int_list(args.lang_decoder_dims) or []
    stability_ids = parse_int_list(args.stability_ids) or []

    if not os.path.exists(args.class_mapping_path):
        raise FileNotFoundError(args.class_mapping_path)
    if not os.path.exists(args.gt_split_root):
        raise FileNotFoundError(args.gt_split_root)

    class_mapping = load_json_int_keys(args.class_mapping_path)
    class_colors = maybe_load_colors(args.class_colors_path)
    citygml_class_map = maybe_load_citygml_class_map(args.citygml_class_map_path)

    os.makedirs(args.root_output_dir, exist_ok=True)
    global_logger = get_logger("all_eval_validation", os.path.join(args.root_output_dir, "validation.log"))

    dir_zaha = os.path.join(args.gt_split_root, "layer_zaha_kept")
    dir_ai = os.path.join(args.gt_split_root, "layer_ai_filled")

    gt_names = require_complete_gt_pair(
        dir_zaha=dir_zaha,
        dir_ai=dir_ai,
        logger=global_logger,
    )

    if args.num_images is not None:
        gt_names = gt_names[:args.num_images]

    predictors = {}

    if run_citygml_clip:
        predictors["citygml_clip"] = CityGMLClipPredictor(
            instance_images_dir=args.city_instance_images_dir,
            model_root=args.city_model_root,
            class_mapping=class_mapping,
            whole_building_fine_ids=whole_building_fine_ids,
            clip_threshold=args.city_clip_threshold,
            citygml_class_map=citygml_class_map,
            openclip_model_name=args.city_openclip_model_name,
            openclip_pretrained=args.city_openclip_pretrained,
            num_images=args.num_images,
            logger=None,
        )

    if run_langsplat:
        predictors["langsplat"] = LangSplatPredictor(
            rendered_features=args.lang_rendered_features_dir,
            ae_checkpoint=args.lang_ae_checkpoint,
            class_mapping=class_mapping,
            whole_mask_thresh=args.lang_whole_mask_thresh,
            part_mask_thresh=args.lang_part_mask_thresh,
            use_softmax=lang_use_softmax,
            encoder_dims=lang_encoder_dims,
            decoder_dims=lang_decoder_dims,
            num_images=args.num_images,
        )

    if run_gaga_dino:
        predictors["gaga_dino"] = GaussianGroupingDINOPredictor(
            pred_inst_dir=args.gaga_pred_inst_dir,
            images_dir=args.gaga_images_dir,
            class_mapping=class_mapping,
            dino_config=args.dino_config,
            dino_checkpoint=args.dino_checkpoint,
            box_thresh=args.dino_box_thresh,
            text_thresh=args.dino_text_thresh,
            inst_min_overlap=args.dino_instance_min_overlap,
            inst_min_score=args.dino_instance_min_score,
            num_images=args.num_images,
            device=args.gaga_device or ("cuda" if torch.cuda.is_available() else "cpu"),
        )

    for _, predictor in predictors.items():
        validate_predictor_inputs_or_raise(
            predictor=predictor,
            image_names=gt_names,
            logger=global_logger,
        )

    global_logger.info(f"All validation checks passed. Total images to evaluate: {len(gt_names)}")

    all_reports = {}

    if run_citygml_clip:
        city_output_dir = os.path.join(args.root_output_dir, "citygml_clip")
        os.makedirs(city_output_dir, exist_ok=True)
        city_logger = get_logger("citygml_clip_eval", os.path.join(city_output_dir, "eval.log"))

        predictor = CityGMLClipPredictor(
            instance_images_dir=args.city_instance_images_dir,
            model_root=args.city_model_root,
            class_mapping=class_mapping,
            whole_building_fine_ids=whole_building_fine_ids,
            clip_threshold=args.city_clip_threshold,
            citygml_class_map=citygml_class_map,
            openclip_model_name=args.city_openclip_model_name,
            openclip_pretrained=args.city_openclip_pretrained,
            num_images=args.num_images,
            logger=city_logger,
        )

        evaluator = UnifiedTwoLevelEvaluator(
            predictor=predictor,
            gt_split_root=args.gt_split_root,
            class_mapping=class_mapping,
            output_dir=city_output_dir,
            logger=city_logger,
            class_colors=class_colors,
            whole_building_id=args.whole_building_id,
            whole_nonbuilding_id=args.whole_nonbuilding_id,
            whole_building_fine_ids=whole_building_fine_ids,
            whole_nonbuilding_fine_ids=whole_nonbuild_fine_ids,
            save_visualizations=args.save_visualizations,
            validated_image_names=gt_names,
        )
        all_reports["citygml_clip"] = evaluator.run()

    if run_langsplat:
        lang_output_dir = os.path.join(args.root_output_dir, "langsplat")
        os.makedirs(lang_output_dir, exist_ok=True)
        lang_logger = get_logger("langsplat_eval", os.path.join(lang_output_dir, "eval.log"))

        predictor = LangSplatPredictor(
            rendered_features=args.lang_rendered_features_dir,
            ae_checkpoint=args.lang_ae_checkpoint,
            class_mapping=class_mapping,
            whole_mask_thresh=args.lang_whole_mask_thresh,
            part_mask_thresh=args.lang_part_mask_thresh,
            use_softmax=lang_use_softmax,
            encoder_dims=lang_encoder_dims,
            decoder_dims=lang_decoder_dims,
            num_images=args.num_images,
        )

        evaluator = UnifiedTwoLevelEvaluator(
            predictor=predictor,
            gt_split_root=args.gt_split_root,
            class_mapping=class_mapping,
            output_dir=lang_output_dir,
            logger=lang_logger,
            class_colors=class_colors,
            whole_building_id=args.whole_building_id,
            whole_nonbuilding_id=args.whole_nonbuilding_id,
            whole_building_fine_ids=whole_building_fine_ids,
            whole_nonbuilding_fine_ids=whole_nonbuild_fine_ids,
            save_visualizations=args.save_visualizations,
            validated_image_names=gt_names,
        )
        all_reports["langsplat"] = evaluator.run()

    if run_gaga_dino:
        gaga_output_dir = os.path.join(args.root_output_dir, "gaga_dino")
        os.makedirs(gaga_output_dir, exist_ok=True)
        gaga_logger = get_logger("gaga_dino_eval", os.path.join(gaga_output_dir, "eval.log"))

        predictor = GaussianGroupingDINOPredictor(
            pred_inst_dir=args.gaga_pred_inst_dir,
            images_dir=args.gaga_images_dir,
            class_mapping=class_mapping,
            dino_config=args.dino_config,
            dino_checkpoint=args.dino_checkpoint,
            box_thresh=args.dino_box_thresh,
            text_thresh=args.dino_text_thresh,
            inst_min_overlap=args.dino_instance_min_overlap,
            inst_min_score=args.dino_instance_min_score,
            num_images=args.num_images,
            device=args.gaga_device or ("cuda" if torch.cuda.is_available() else "cpu"),
        )

        evaluator = UnifiedTwoLevelEvaluator(
            predictor=predictor,
            gt_split_root=args.gt_split_root,
            class_mapping=class_mapping,
            output_dir=gaga_output_dir,
            logger=gaga_logger,
            class_colors=class_colors,
            whole_building_id=args.whole_building_id,
            whole_nonbuilding_id=args.whole_nonbuilding_id,
            whole_building_fine_ids=whole_building_fine_ids,
            whole_nonbuilding_fine_ids=whole_nonbuild_fine_ids,
            save_visualizations=args.save_visualizations,
            validated_image_names=gt_names,
        )
        all_reports["gaga_dino"] = evaluator.run()

    if all_reports:
        summary_path = save_summary_table(args.root_output_dir, args.summary_filename, all_reports)

        stability_results = {}
        for method, report in all_reports.items():
            if "part" not in report:
                continue

            part_rep = report["part"]
            method_stability = {
                "overall_part_mIoU": part_rep["mIoU"],
                "target_classes": {}
            }

            for cid in stability_ids:
                cname = class_mapping.get(cid)
                if isinstance(cname, list):
                    cname = cname[0]

                if cname in part_rep["per_class_precision"]:
                    method_stability["target_classes"][cname] = {
                        "Precision": part_rep["per_class_precision"][cname],
                        "Recall": part_rep["per_class_recall"][cname],
                        "IoU": part_rep["per_class_iou"][cname],
                        "Counts": part_rep["per_class_raw_counts"][cname],
                    }

            stability_results[method] = method_stability

        stability_file = os.path.join(args.root_output_dir, "non_building_stability_analysis.json")
        with open(stability_file, "w", encoding="utf-8") as f:
            json.dump(stability_results, f, indent=4)
        print(f"Stability analysis saved to: {stability_file}")

    else:
        summary_path = os.path.join(args.root_output_dir, args.summary_filename)

    if args.save_cross_method_panels:
        method_output_dirs = {
            "langsplat": os.path.join(args.root_output_dir, "langsplat"),
            "gaga_dino": os.path.join(args.root_output_dir, "gaga_dino"),
            "citygml_clip": os.path.join(args.root_output_dir, "citygml_clip"),
        }

        create_cross_method_prediction_panels(
            root_output_dir=args.root_output_dir,
            rgb_dir=args.rgb_image_dir,
            gt_split_root=args.gt_split_root,
            class_mapping=class_mapping,
            class_colors=class_colors,
            whole_building_id=args.whole_building_id,
            whole_nonbuilding_id=args.whole_nonbuilding_id,
            whole_building_fine_ids=whole_building_fine_ids,
            whole_nonbuilding_fine_ids=whole_nonbuild_fine_ids,
            method_output_dirs=method_output_dirs,
            masked_by_gt=False,
            output_subdir="cross_method_prediction_unmasked",
        )

        create_cross_method_prediction_panels(
            root_output_dir=args.root_output_dir,
            rgb_dir=args.rgb_image_dir,
            gt_split_root=args.gt_split_root,
            class_mapping=class_mapping,
            class_colors=class_colors,
            whole_building_id=args.whole_building_id,
            whole_nonbuilding_id=args.whole_nonbuilding_id,
            whole_building_fine_ids=whole_building_fine_ids,
            whole_nonbuilding_fine_ids=whole_nonbuild_fine_ids,
            method_output_dirs=method_output_dirs,
            masked_by_gt=True,
            output_subdir="cross_method_prediction_masked",
        )

    print("\n" + "=" * 80)
    print("All evaluations finished.")
    print(f"Summary saved to: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
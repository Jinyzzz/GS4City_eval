#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# ============================================================================
# CONFIGURATION
# ============================================================================

# RGB_DIR = "/workspace/test_subset4"
RGB_DIR = "/workspace/CityGMLGaussian/output/subset_goldcoast_gaga_30000/test/ours_30000/gt"
ZAHA_GT_DIR = "/workspace/zaha_eval/gt/subset_goldcoast_33"

# 总输出目录
OUTPUT_DIR = "/workspace/zaha_eval/gt/subset_goldcoast_test"

# [新增] Merge Map 路径
GT_MERGE_MAP_PATH = "gt_merge_map.json"

# 模型配置
MODEL_REPO = "nvidia/segformer-b5-finetuned-ade-640-640"
NUM_CLASSES = 150
WINDOW_SIZE = 800 
STRIDE = 600

# [关键配置] Zaha 数据集中必须保留的 "核心标签" (合并后的 ID)
# 只要 Zaha mask 的值在这个列表里，就绝对不修改。
KEEP_ZAHA_IDS = [1, 2, 3, 12] 

# SegFormer (ADE20K) 映射
ADE20K_MAPPING = {
    2: 100,  # Sky -> 100
    4: 101,  # Tree/Vegetation -> 101
    6: 102,  # Road/Route -> 102
    13: 102, # Earth/Ground -> 102
    12: 103, # Person -> 103
    20: 104, # Car -> 104
}

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def colorize_mask(mask):
    unique_ids = np.unique(mask)
    vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    np.random.seed(42)
    for uid in unique_ids:
        if uid <= 0: continue
        color = np.random.randint(0, 255, 3)
        vis[mask == uid] = color
    return vis

def apply_label_merges(label_map, merge_map):
    """
    根据 json 映射合并标签。
    """
    if not merge_map: return label_map
    out = label_map.copy()
    for target, sources in merge_map.items():
        target = int(target)
        for src in sources:
            src = int(src)
            if src == target: continue
            out[out == src] = target
    return out

# ============================================================================
# 滑窗推理
# ============================================================================
def predict_sliding_window(model, processor, image_pil, window_size=WINDOW_SIZE, stride=STRIDE, device="cuda"):
    w, h = image_pil.size
    image_np = np.array(image_pil)
    
    probs_map = torch.zeros((NUM_CLASSES, h, w), dtype=torch.float16, device="cpu")
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
                    align_corners=False
                )
                
                probs_map[:, r_start:r_end, c_start:c_end] += upsampled_logits.squeeze(0).cpu().to(torch.float16)
                count_map[:, r_start:r_end, c_start:c_end] += 1.0

    probs_map /= count_map
    final_pred = probs_map.argmax(dim=0).numpy().astype(np.int32)
    return final_pred

# ============================================================================
# MAIN
# ============================================================================
def main():
    # 1. 创建输出目录结构
    dir_fused = os.path.join(OUTPUT_DIR, "fused")        # 最终融合结果
    dir_zaha = os.path.join(OUTPUT_DIR, "layer_zaha_kept") # 仅 Zaha保留部分
    dir_ai = os.path.join(OUTPUT_DIR, "layer_ai_filled")   # 仅 AI填补部分
    dir_vis = os.path.join(OUTPUT_DIR, "visualization")    # 可视化

    for d in [dir_fused, dir_zaha, dir_ai, dir_vis]:
        os.makedirs(d, exist_ok=True)

    # 2. 加载 Merge Map
    gt_merge_map = None
    if os.path.exists(GT_MERGE_MAP_PATH):
        print(f"Loading Merge Map from {GT_MERGE_MAP_PATH}...")
        with open(GT_MERGE_MAP_PATH, 'r') as f:
            merge_raw = json.load(f)
        gt_merge_map = {int(k): [int(v) for v in values] for k, values in merge_raw.items()}
    else:
        print("No merge map found, proceeding without merging.")

    print(f"Loading SegFormer model: {MODEL_REPO}...")
    processor = SegformerImageProcessor.from_pretrained(MODEL_REPO)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_REPO)
    device = get_device()
    model.to(device)

    zaha_files = sorted(glob.glob(os.path.join(ZAHA_GT_DIR, "*.npy")))
    print(f"Found {len(zaha_files)} Zaha GT files.")

    for gt_path in tqdm(zaha_files, desc="Processing Images"):
        base_name = os.path.splitext(os.path.basename(gt_path))[0]
        
        # 匹配 RGB
        candidates = [base_name, base_name.replace("_D", ""), base_name.replace("_depth", "")]
        rgb_path = None
        for name in candidates:
            for ext in ['.png', '.jpg', '.JPG', '.jpeg']:
                p = os.path.join(RGB_DIR, name + ext)
                if os.path.exists(p):
                    rgb_path = p; break
            if rgb_path: break
        
        if not rgb_path:
            potential = glob.glob(os.path.join(RGB_DIR, f"{candidates[-1]}*"))
            potential = [f for f in potential if f.lower().endswith(('.png', '.jpg'))]
            if potential: rgb_path = potential[0]

        if rgb_path is None:
            continue

        # 加载数据
        zaha_mask = np.load(gt_path)
        image = Image.open(rgb_path).convert("RGB")

        # 1. 类别合并
        if gt_merge_map:
            zaha_mask = apply_label_merges(zaha_mask, gt_merge_map)

        # 2. SegFormer 推理
        seg_pred_raw = predict_sliding_window(
            model, processor, image, 
            window_size=WINDOW_SIZE, 
            stride=STRIDE, 
            device=device
        )
        
        # 尺寸对齐
        if seg_pred_raw.shape != zaha_mask.shape:
             seg_pred_raw = cv2.resize(
                 seg_pred_raw.astype(np.float32), 
                 (zaha_mask.shape[1], zaha_mask.shape[0]), 
                 interpolation=cv2.INTER_NEAREST
             ).astype(np.int32)

        # 3. 映射 AI 结果到项目 ID
        segformer_remapped = np.zeros_like(seg_pred_raw, dtype=np.int32)
        for ade_id, project_id in ADE20K_MAPPING.items():
            segformer_remapped[seg_pred_raw == ade_id] = project_id
            
        # ====================================================================
        # [修改] 核心融合与分离逻辑
        # ====================================================================
        
        # 条件 A: 该像素是否属于 Zaha 必须保留的核心区域
        is_zaha_kept = np.isin(zaha_mask, KEEP_ZAHA_IDS)
        
        # 条件 B: AI 是否有有效的预测结果
        has_valid_ai = (segformer_remapped > 0)
        
        # 计算 Mask 1: Zaha 保留层 (Layer Zaha)
        # 只保留 Zaha Mask 中属于核心 ID 的部分，其余为 0
        layer_zaha_kept = np.zeros_like(zaha_mask)
        layer_zaha_kept[is_zaha_kept] = zaha_mask[is_zaha_kept]
        
        # 计算 Mask 2: AI 填补层 (Layer AI)
        # 逻辑：(不属于 Zaha 核心区域) AND (AI 有有效预测)
        should_use_ai = (~is_zaha_kept) & has_valid_ai
        layer_ai_filled = np.zeros_like(zaha_mask)
        layer_ai_filled[should_use_ai] = segformer_remapped[should_use_ai]
        
        # 计算 Mask 3: 最终融合层 (Fused)
        # 基础是 Zaha Mask (包含非核心区域的噪音，如果你想保留的话)，然后用 AI 覆盖
        # 或者为了干净，直接由 layer_zaha_kept + layer_ai_filled 组成
        # 这里我们采用稳健的覆盖逻辑：
        final_mask = zaha_mask.copy()
        final_mask[should_use_ai] = segformer_remapped[should_use_ai]
        
        # ====================================================================
        # 保存三个不同的 NPY
        # ====================================================================
        
        # 1. 保存最终融合
        np.save(os.path.join(dir_fused, f"{base_name}.npy"), final_mask)
        
        # 2. 保存 Zaha 保留层
        np.save(os.path.join(dir_zaha, f"{base_name}.npy"), layer_zaha_kept)
        
        # 3. 保存 AI 填补层
        np.save(os.path.join(dir_ai, f"{base_name}.npy"), layer_ai_filled)

        # ====================================================================
        # 可视化 (保持原样，展示融合结果)
        # ====================================================================
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(image); axes[0].set_title("RGB"); axes[0].axis('off')
        axes[1].imshow(colorize_mask(zaha_mask)); axes[1].set_title("Original Zaha GT"); axes[1].axis('off')
        # 这里为了展示清晰，可以展示 AI 填补层
        axes[2].imshow(colorize_mask(layer_ai_filled)); axes[2].set_title("AI Fill Only"); axes[2].axis('off')
        axes[3].imshow(colorize_mask(final_mask)); axes[3].set_title("Fused GT"); axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(dir_vis, f"{base_name}_vis.png"), dpi=150)
        plt.close(fig)

    print(f"Done! Files saved to subdirectories in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
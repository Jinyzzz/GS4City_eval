import os
import sys
import json
import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser

# ==============================================================================
# [用户配置区域] 请直接在这里修改路径，不需要去命令行输参数
# ==============================================================================

# 1. 锚点数据路径 (通常是 train 文件夹，用于 DINO+SAM 识别物体 ID)
#    必须包含子文件夹: renders/ (RGB图) 和 objects_test/ (ID图)
DEFAULT_ANCHOR_ROOT = "/workspace/CityGMLGaussian/output/gaga_level1_30000/train/ours_30000"

# 2. 目标数据路径 (通常是 test 文件夹，我们要从这里提取 Mask)
#    必须包含子文件夹: objects_test/ (ID图)
DEFAULT_TARGET_ROOT = "/workspace/CityGMLGaussian/output/gaga_level1_30000/test/ours_30000"

# 3. class_mapping.json 的绝对路径 (定义了类别和提示词)
DEFAULT_JSON_PATH = "/workspace/zaha_eval/class_mapping.json"

# 4. 结果保存路径 (生成的 Mask 会保存在这里)
DEFAULT_SAVE_ROOT = "/workspace/zaha_eval/eval_results/gaga_query/objects_prompt"

# 5. 模型权重路径
DEFAULT_SAM_CHECKPOINT = "/workspace/CityGMLGaussian/weight/sam_vit_h_4b8939.pth"
DEFAULT_DINO_CONFIG = "GroundingDINO_SwinB.cfg.py"  # 如果在当前目录没找到，会自动下载
DEFAULT_DINO_CKPT = "groundingdino_swinb_cogcoor.pth"

# ==============================================================================

# 路径 Hack: 确保能找到 ext.grounded_sam
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ext.grounded_sam import grouned_sam_output, load_model_hf, select_obj_ioa
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("[Error] 找不到 'ext.grounded_sam'。请确保 ext 文件夹在当前脚本同级目录下。")
    sys.exit(1)

def run_extraction(args):
    # --- 1. 路径检查与构建 ---
    anchor_render_dir = os.path.join(args.anchor_root, "renders")
    anchor_id_dir = os.path.join(args.anchor_root, "objects_test")
    target_id_dir = os.path.join(args.target_root, "objects_test")

    if not os.path.exists(anchor_render_dir):
        raise FileNotFoundError(f"找不到锚点RGB文件夹: {anchor_render_dir}")
    if not os.path.exists(anchor_id_dir):
        raise FileNotFoundError(f"找不到锚点ID文件夹: {anchor_id_dir}")
    if not os.path.exists(target_id_dir):
        raise FileNotFoundError(f"找不到目标ID文件夹: {target_id_dir}")
    if not os.path.exists(args.json_path):
        raise FileNotFoundError(f"找不到 JSON 文件: {args.json_path}")

    os.makedirs(args.save_root, exist_ok=True)

    # --- 2. 加载数据 ---
    with open(args.json_path, 'r') as f:
        class_mapping = json.load(f)
    print(f"[INFO] 类别配置已加载: {len(class_mapping)} 个类别")

    print("[INFO] 正在加载 Grounding DINO & SAM ...")
    dino_model = load_model_hf("ShilongLiu/GroundingDINO", DEFAULT_DINO_CKPT, DEFAULT_DINO_CONFIG)
    
    if not os.path.exists(args.sam_ckpt):
        raise FileNotFoundError(f"SAM 权重未找到: {args.sam_ckpt}")
    
    sam = sam_model_registry["vit_h"](checkpoint=args.sam_ckpt)
    sam.to(device='cuda')
    sam_predictor = SamPredictor(sam)

    # --- 3. 准备锚点帧 (Anchor) ---
    files = sorted(os.listdir(anchor_render_dir))
    if not files:
        print("[Error] 锚点 renders 文件夹是空的！")
        return

    anchor_name = files[0]
    anchor_rgb_path = os.path.join(anchor_render_dir, anchor_name)
    anchor_id_path = os.path.join(anchor_id_dir, anchor_name)

    print(f"[INFO] 使用第一帧作为锚点: {anchor_name}")
    
    # 读取锚点 RGB
    anchor_img_pil = Image.open(anchor_rgb_path).convert("RGB")
    anchor_img_np = np.array(anchor_img_pil)
    
    # 读取锚点 ID 图 (16bit)
    if not os.path.exists(anchor_id_path):
        raise RuntimeError(f"对应的 ID 图不存在: {anchor_id_path}")
    anchor_id_map = cv2.imread(anchor_id_path, cv2.IMREAD_UNCHANGED)

    # --- 4. 核心循环 ---
    for class_id_str, prompts in class_mapping.items():
        if isinstance(prompts, list):
            text_prompt = " . ".join([p.strip() for p in prompts if p.strip()])
        else:
            text_prompt = prompts
        
        class_id = int(class_id_str)
        print(f"\n------------------------------------------------")
        print(f"[处理类别 {class_id}] Prompt: '{text_prompt}'")

        # Step A: 2D 检测 (DINO + SAM)
        try:
            text_mask_2d, annotated_frame = grouned_sam_output(dino_model, sam_predictor, text_prompt, anchor_img_np)
        except Exception as e:
            print(f"  [Error] DINO/SAM 推理出错: {e}")
            continue

        # 保存锚点检测图 (Debug)
        debug_dir = os.path.join(args.save_root, "_debug_anchors")
        os.makedirs(debug_dir, exist_ok=True)
        Image.fromarray(annotated_frame).save(os.path.join(debug_dir, f"class_{class_id}.png"))

        if text_mask_2d.sum() == 0:
            print(f"  [Warn] 锚点帧未检测到物体，跳过此类别。")
            continue

        # Step B: 2D -> 3D ID 映射
        # [关键修复] 将 Numpy 转换为 Tensor，解决 .unique() 报错问题
        try:
            # 1. 转换 anchor_id_map (Numpy -> Tensor)
            # 确保类型是 long (int64) 或 int32
            anchor_id_tensor = torch.from_numpy(anchor_id_map.astype(np.int64)).cuda()

            # 2. 转换 text_mask_2d (Numpy -> Tensor)
            if isinstance(text_mask_2d, np.ndarray):
                text_mask_tensor = torch.from_numpy(text_mask_2d).bool().cuda()
            else:
                text_mask_tensor = text_mask_2d.bool().cuda()

            # 3. 调用函数 (现在传入的是 Tensor 了)
            selected_ids = select_obj_ioa(anchor_id_tensor, text_mask_tensor)

            # 4. 如果返回的是 Tensor，转回 Python List
            if isinstance(selected_ids, torch.Tensor):
                selected_ids = selected_ids.cpu().tolist()
            elif isinstance(selected_ids, np.ndarray):
                selected_ids = selected_ids.tolist()
            
            print(f"  [Match] 匹配到的 3D Object IDs: {selected_ids}")

        except Exception as e:
            print(f"  [Error] ID 匹配步骤出错: {e}")
            import traceback
            traceback.print_exc()
            continue

        if len(selected_ids) == 0:
            print("  [Warn] 未匹配到任何 3D ID (IoA 过低)，跳过。")
            continue

        # Step C: 批量提取 Target Mask
        class_save_dir = os.path.join(args.save_root, str(class_id))
        os.makedirs(class_save_dir, exist_ok=True)

        target_files = sorted(os.listdir(target_id_dir))
        
        # ID Set 用于加速查找 (使用 Numpy 处理)
        target_id_set = np.array(selected_ids, dtype=anchor_id_map.dtype)

        print(f"  正在提取 {len(target_files)} 张测试集图片...")
        for t_file in tqdm(target_files, leave=False):
            if not t_file.endswith(".png"): continue
            
            t_id_path = os.path.join(target_id_dir, t_file)
            t_id_map = cv2.imread(t_id_path, cv2.IMREAD_UNCHANGED) # uint16

            # 核心提取逻辑: 像素值属于 selected_ids 的设为 255
            mask = np.isin(t_id_map, target_id_set)
            binary_mask = (mask * 255).astype(np.uint8)

            Image.fromarray(binary_mask).save(os.path.join(class_save_dir, t_file))

    print(f"\n[Success] 所有 Mask 已保存在: {args.save_root}")


if __name__ == "__main__":
    parser = ArgumentParser()

    # 使用 defaults=... 将顶部的配置传入，去掉 required=True
    parser.add_argument("--anchor_root", type=str, default=DEFAULT_ANCHOR_ROOT)
    parser.add_argument("--target_root", type=str, default=DEFAULT_TARGET_ROOT)
    parser.add_argument("--json_path", type=str, default=DEFAULT_JSON_PATH)
    parser.add_argument("--save_root", type=str, default=DEFAULT_SAVE_ROOT)
    parser.add_argument("--sam_ckpt", type=str, default=DEFAULT_SAM_CHECKPOINT)

    args = parser.parse_args()
    
    run_extraction(args)
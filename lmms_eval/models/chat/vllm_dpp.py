
from lmms_eval.models.simple.vllm import VLLM as VLLM_Simple
import base64
import re
from io import BytesIO
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)



import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple, Union

from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics

from lmms_eval.protocol import ChatMessages

try:
    from vllm import LLM, SamplingParams
except ImportError:
    vllm = None

from lmms_eval import utils
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)
from sentence_transformers import SentenceTransformer
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")

from PIL import Image
import os


TEST_MIN_TOKEN_MAX_COVERAGE=False

def cdpruner_dpp_dynamic_resolution(
    frame_features,
    frame_embeds,
    text_embed,
    total_token_budget,
    tau=8.0,
    min_token_per_frame=1024,
    max_token_per_frame=None,
    alpha=1.0 
):  
    if TEST_MIN_TOKEN_MAX_COVERAGE:
        min_token_per_frame = max_token_per_frame//4
    min_token_per_frame=max_token_per_frame//4
    #print(min_token_per_frame)
    device = frame_features.device
    eps = 1e-6
    T = frame_features.shape[0]
   

    feat = frame_features / (frame_features.norm(dim=-1, keepdim=True) + eps)
    similarity = (feat @ feat.T).clamp(min=0)

    frame_indices = torch.arange(T, device=device)
    d = frame_indices[:, None] - frame_indices[None, :]
    time_decay = torch.exp(-(d.float() ** 2) / (tau ** 2))
    similarity = similarity * time_decay

    img = frame_embeds / (frame_embeds.norm(dim=-1, keepdim=True) + eps)
    txt = text_embed / (text_embed.norm() + eps)
    relevance = (img @ txt).squeeze()
    
    #r_min, r_max = relevance.min(), relevance.max()
    #relevance = (relevance - r_min) / (r_max - r_min + eps)
    temp = 0.02 
    relevance = (relevance - relevance.mean()) / temp
    relevance = torch.sigmoid(relevance)

    # ---------- 3. 构建 DPP Kernel ----------
    kernel = relevance[:, None] * similarity * relevance[None, :]

    # ---------- 4. Greedy DPP 选帧 (Gram-Schmidt) ----------
    K_max = total_token_budget // min_token_per_frame

  
    
    cis = torch.zeros((K_max, T), device=device)
    di2s = torch.diag(kernel).clone()
    selected_indices = []
    selected_energies = []

    for i in range(K_max):
        j = torch.argmax(di2s)
        current_energy = di2s[j]
        
        # 如果能量过低或已经处理完，停止
        if current_energy <= 0:
            break
            
        selected_indices.append(j.item())
        selected_energies.append(current_energy.item())
        
        # 更新残余能量 (扣除已选帧的相似度贡献)
        if i == 0:
            eis = kernel[j] / torch.sqrt(current_energy + eps)
        else:
            eis = (kernel[j] - cis[:i, j] @ cis[:i]) / torch.sqrt(current_energy + eps)
        
        cis[i] = eis
        di2s = di2s - eis ** 2
        di2s = torch.clamp(di2s, min=0)
        di2s[j] = -float("inf") # 避免重复选中同一帧

    # 转化为 Tensor 传入分配函数
    selected_indices = torch.tensor(selected_indices, device=device)
    selected_energies = torch.tensor(selected_energies, device=device)

    if TEST_MIN_TOKEN_MAX_COVERAGE:
        # 强制分配最小值
        final_indices = selected_indices
        selected_token_counts = torch.full_like(selected_energies, min_token_per_frame).int()
        return final_indices, selected_token_counts
    # ---------- 5. 动态 Token 分配 ----------
    final_indices, selected_token_counts = dynamic_token_allocation_v2(
        importance=selected_energies,
        total_token_budget=total_token_budget,
        min_token_per_frame=min_token_per_frame,
        max_token_per_frame=max_token_per_frame,
        original_indices=selected_indices
    )

    return final_indices, selected_token_counts



def dynamic_token_allocation_v2(
    importance: torch.Tensor,
    total_token_budget: int,
    min_token_per_frame: int = 1024,
    max_token_per_frame: int = 4096,
    original_indices: torch.Tensor = None
):
    device = importance.device
    n_frames = len(importance)
    
    # 记录原始候选人的完整状态
    active_mask = torch.ones(n_frames, dtype=torch.bool, device=device)
    fixed_mask = torch.zeros(n_frames, dtype=torch.bool, device=device)
    allocations = torch.zeros(n_frames, device=device)

    # --- 第一阶段：收缩（踢人直到所有人都能拿低保） ---
    while active_mask.any():
        to_assign_mask = active_mask & (~fixed_mask)
        if not to_assign_mask.any(): break
            
        remaining_budget = total_token_budget - allocations[fixed_mask].sum()
        curr_importance = importance[to_assign_mask]
        
        norm = curr_importance.sum() + 1e-8
        current_allocs = (curr_importance / norm) * remaining_budget
        
        # 踢人逻辑：只要有人不到 min，就踢掉最不重要的一个，重新循环
        if (current_allocs < min_token_per_frame).any():
            last_active_idx = torch.where(active_mask)[0][-1]
            active_mask[last_active_idx] = False
            fixed_mask.fill_(False)
            allocations.fill_(0)
            continue
            
        # 封顶逻辑：超过 max 的先锁死，多出来的钱再分给剩下的 active 帧
        too_large = current_allocs > max_token_per_frame
        if too_large.any():
            global_indices = torch.where(to_assign_mask)[0]
            over_limit_indices = global_indices[too_large]
            allocations[over_limit_indices] = float(max_token_per_frame)
            fixed_mask[over_limit_indices] = True
            continue
        else:
            allocations[to_assign_mask] = current_allocs
            break

    # --- 第二阶段：救回（核心修改点！） ---
    # 此时，所有 active 的帧都已经 >= min 且 <= max
    # 如果此时还有钱（diff 很大），且还有被踢掉的帧，我们尝试“捞人”
    
    # 找到所有被踢掉的帧，按重要性排序
    evicted_mask = ~active_mask
    if evicted_mask.any():
        evicted_indices = torch.where(evicted_mask)[0]
        # 重要性从高到低排序，优先救“更有用”的帧
        sorted_evicted = evicted_indices[torch.argsort(importance[evicted_indices], descending=True)]
        
        for revive_idx in sorted_evicted:
            current_total = allocations[active_mask].sum()
            # 如果剩下的钱足够给这一帧一个“入场券”（min_token）
            if total_token_budget - current_total >= min_token_per_frame:
                active_mask[revive_idx] = True
                allocations[revive_idx] = float(min_token_per_frame)
                # 救回后，如果还有钱，可以在下一轮循环或者微调中分配，这里简单处理
            else:
                break # 钱不够发入场券了，收工

    # --- 第三阶段：取整与最终微调 ---
    final_indices = original_indices[active_mask] if original_indices is not None else torch.where(active_mask)[0]
    final_alloc_values = allocations[active_mask]
    rounded_alloc = final_alloc_values.round().int()

    # 最后的安全检查：禁止上采样
    diff = total_token_budget - rounded_alloc.sum()
    if diff > 0:
        # 只有还没到 max 的帧才能拿剩下的钱
        adjust_mask = rounded_alloc < max_token_per_frame
        if adjust_mask.any():
            adjust_indices = torch.where(adjust_mask)[0]
            for i in range(int(diff)):
                idx = adjust_indices[i % len(adjust_indices)]
                if rounded_alloc[idx] < max_token_per_frame:
                    rounded_alloc[idx] += 1

    return final_indices, rounded_alloc




def estimate_hw_from_resolution(orig_h, orig_w, target_token_count, patch_size=14):
    aspect_ratio = orig_h / orig_w
    grid_w = max(1, int(round((target_token_count / aspect_ratio) ** 0.5)))
    grid_h = max(1, int(round(grid_w * aspect_ratio)))
    

    new_h = grid_h * patch_size
    new_w = grid_w * patch_size
    return new_h, new_w

def load_and_resize_images(frame_paths, resolutions, patch_size=14):
    images = []
    metadata = []
    for idx, (path, token_count) in enumerate(zip(frame_paths, resolutions)):
        img = Image.open(path).convert("RGB")
        orig_w, orig_h = img.size
        new_h, new_w = estimate_hw_from_resolution(orig_h, orig_w, token_count, patch_size)
        img_resized = img.resize((new_w, new_h), resample=Image.BICUBIC)
        images.append(img_resized)
        actual_patches = (new_h // 14) * (new_w // 14)
        #print(f"Path: {os.path.basename(path)} Org:{orig_w}X{orig_h}| Res: {new_w}x{new_h} | Actual Patches: {actual_patches}")

        # Collect metadata for logging
        metadata.append({
            "frame_idx": idx,
            "path": os.path.basename(path),
            "resolution": f"{new_w}x{new_h}",
            "patches": actual_patches,
            "token_count": token_count,
        })

    return images, metadata
@register_model("vllm_dpp")
class VLLM_DPP(VLLM_Simple):
    def generate_until(self, requests) -> List[str]:

        
        clip_model = SentenceTransformer("/usr/project/xtmp/jc923/EACL/ckpt/clip2/sentence-transformers/clip-ViT-L-14", device="cuda:0")
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        batch_size = self.batch_size_per_gpu
        batched_requests = [requests[i : i + batch_size] for i in range(0, len(requests), batch_size)]
        for batch_requests in batched_requests:
            batched_messages = []
            for idx in range(len(batch_requests)):
                contexts, gen_kwargs, doc_to_visual, doc_id, task, split = batch_requests[idx].arguments
                params = {
                    "max_tokens": gen_kwargs.get("max_new_tokens", 1024),
                    "temperature": gen_kwargs.get("temperature", 0.0),
                    "top_p": gen_kwargs.get("top_p", 0.95),
                }
                sampling_params = SamplingParams(**params)

                doc = self.task_dict[task][split][doc_id]
     
                # 从 doc_to_visual 中取出视频路径或图片对象
                visual = doc_to_visual(doc)[0]
                frame_dir = visual[:-4]
                frame_paths = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir)])
                


                # 1. 拿出原始问题
                question = contexts.split("\nA.")[0]

                # 2. 提取 frame 和 question 的 CLIP 特征
                with torch.no_grad():
                    feats = clip_model.encode(
                        frame_paths, batch_size=32, convert_to_tensor=True,
                        normalize_embeddings=True, device="cuda:0"
                    ).float()
                    text_embed = clip_model.encode(
                        [question], convert_to_tensor=True, normalize_embeddings=True, device="cuda:0"
                    )[0].float()

                # 3. 获取 patch 上限（用第1帧估算）
                with Image.open(frame_paths[0]) as sample:
                    w, h = sample.size
                patch_limit = (h // 14) * (w // 14)

                # 4. 执行 DPP + token 动态分配
                idxs, tokens = cdpruner_dpp_dynamic_resolution(
                    feats, feats, text_embed,
                    total_token_budget=self.max_frame_num * patch_limit,
                    max_token_per_frame=patch_limit
                )
                sorted_pairs = sorted(zip(idxs.tolist(), tokens.tolist()))
                selected_paths = [frame_paths[i] for i, _ in sorted_pairs]
                resolutions = [r for _, r in sorted_pairs]

                # 5. resize 图像
                resized_imgs, _ = load_and_resize_images(selected_paths, resolutions)

                # 6. base64 encode 作为 input image_url
                imgs = []
                for img in resized_imgs:
                    output = BytesIO()
                    img.save(output, format="PNG")
                    b64 = base64.b64encode(output.getvalue()).decode("utf-8")
                    imgs.append(b64)

                # 7. 构造 chat message 输入（text + image_url）
                messages = [{"role": "user", "content": []}]
                if self.image_first:
                    for img in imgs:
                        messages[0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})
                    messages[0]["content"].append({"type": "text", "text": contexts})
                else:
                    messages[0]["content"].append({"type": "text", "text": contexts})
                    for img in imgs:
                        messages[0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})

                batched_messages.append(messages)

            response = self.client.chat(sampling_params=sampling_params, messages=batched_messages, chat_template=self.chat_template if self.chat_template else None)
            res.extend([o.outputs[0].text for o in response])
            pbar.update(len(batch_requests))

        pbar.close()
        return res

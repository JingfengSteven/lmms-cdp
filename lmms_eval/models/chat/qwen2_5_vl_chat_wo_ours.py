import time
from typing import List, Optional, Tuple, Union

import numpy as np
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)
from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL as Qwen2_5_VLSimple
from lmms_eval.protocol import ChatMessages
from lmms_eval.metadata_manager import metadata_manager

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")

TEST_MIN_TOKEN_MAX_COVERAGE = True

## ===============================================================================
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
# from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import json
from tqdm import tqdm
import argparse
import re
from PIL import Image

def cdpruner_dpp_dynamic_resolution(
    frame_features,
    frame_embeds,
    text_embed,
    total_token_budget,
    min_token_per_frame=512,
    max_token_per_frame=4096,
    alpha=1.0 
):  
    device = frame_features.device
    eps = 1e-6
    T = frame_features.shape[0]
    min_token_per_frame=max_token_per_frame//8
    
    K_max = total_token_budget // min_token_per_frame

    # ---------- 1. 自动计算局部抑制半径 R ----------
    # R 决定了选出一帧后，其周围多大范围会被抑制。设为预期采样间隔。
    R = max(2.0, float(T) / (K_max + eps))

    # ---------- 2. 构建特征相似度矩阵 ----------
    feat = frame_features / (frame_features.norm(dim=-1, keepdim=True) + eps)
    content_sim = (feat @ feat.T).clamp(min=0)

    # ---------- 3. 构建位置先验 (Local DPP) ----------
    frame_indices = torch.arange(T, device=device)
    dist_sq = (frame_indices[:, None] - frame_indices[None, :]).float() ** 2
    location_prior = torch.exp(-dist_sq / (R ** 2))

    # 融合相似度：保证矩阵不是稀疏的，且具有局部排他性
    # 0.7 和 0.3 是平衡点，既看内容也看位置
    similarity = content_sim #+ 0.2 * location_prior

    # ---------- 4. 计算并平滑文本相关性 (Relevance) ----------
    img = frame_embeds / (frame_embeds.norm(dim=-1, keepdim=True) + eps)
    txt = text_embed / (text_embed.norm() + eps)
    relevance = (img @ txt).squeeze()
    
    # 优雅的平滑逻辑：
    # 1. 使用 Sigmoid 将原始得分映射到 (0, 1)
    # 2. 缩放到 (0.5, 1.0)，消除“能量断层”
    # 这样相关性最高和最低的帧，能量差距最多只有 2-4 倍，而不是成百上千倍
    gamma = 0.5 
    kernel = torch.pow(relevance[:, None], gamma) * similarity * torch.pow(relevance[None, :], gamma)

    

    # ---------- 6. Greedy DPP 选帧 (Gram-Schmidt) ----------
    cis = torch.zeros((K_max, T), device=device)
    di2s = torch.diag(kernel).clone()
    selected_indices = []
    selected_energies = []

    for i in range(K_max):
        j = torch.argmax(di2s)
        current_energy = di2s[j]
        
        # 能量过低则停止
        if current_energy <= 1e-7:
            break
            
        selected_indices.append(j.item())
        
        # 【关键优化】能量对数平滑：
        # 用于 Token 分配的“重要性”不应直接使用物理能量，
        # 取对数能让 Token 分配更平均，避免一帧独占所有预算。
        importance_score = torch.log1p(current_energy) 
        selected_energies.append(importance_score.item())
        
        # 标准 Gram-Schmidt 更新
        if i == 0:
            eis = kernel[j] / torch.sqrt(current_energy + eps)
        else:
        
            eis = (kernel[j] - cis[:i, j] @ cis[:i]) / torch.sqrt(current_energy + eps)
        
        cis[i] = eis
        di2s = di2s - eis ** 2
        di2s = torch.clamp(di2s, min=0)
        di2s[j] = -float("inf") 

    # 转化为 Tensor
    selected_indices = torch.tensor(selected_indices, device=device)
    selected_energies = torch.tensor(selected_energies, device=device)

    # ---------- 7. 动态 Token 分配 ----------
    # 如果处于强制覆盖模式
    '''if 'TEST_MIN_TOKEN_MAX_COVERAGE' in globals() and TEST_MIN_TOKEN_MAX_COVERAGE:
        final_indices = selected_indices
        selected_token_counts = torch.full_like(selected_energies, min_token_per_frame).int()
        return final_indices, selected_token_counts'''
    if TEST_MIN_TOKEN_MAX_COVERAGE:
        # 按照 K_max 进行均匀采样
        indices = np.linspace(0, T - 1, K_max, dtype=int)
        indices = np.unique(indices)
        
        final_indices = torch.tensor(indices, device=device)
        # 每一帧分配相等的最小 token 预算
        selected_token_counts = torch.full((len(final_indices),), min_token_per_frame, device=device).int()
        
        return final_indices, selected_token_counts

    # 调用分配函数
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
        # 只有还没到_max_的帧才能拿剩下的钱
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
        #print(f"Path: {os.path.basename(path)} Org:{orig_w}X{orig_h}| Res: {img_resized.size} | Actual Patches: {actual_patches}")

        # Collect metadata for logging
        metadata.append({
            "frame_idx": idx,
            "path": os.path.basename(path),
            "resolution": f"{new_w}x{new_h}",
            "patches": actual_patches,
            "token_count": token_count,
        })

    return images, metadata

## ===============================================================================


@register_model("qwen2_5_vl_chat_wo_ours")
class Qwen2_5_VL_Chat_WO_Ours(Qwen2_5_VLSimple):
    is_simple = False

    def generate_until(self, requests: List[Instance]) -> List[str]:
        
        print(f"[INFO] Total dataset samples: {len(requests)}")
        res = []

        # A dummy collate here to sort by doc id
        def _collate(x):
            return x[0], x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator(
            [reg.args for reg in requests],
            _collate,
            group_fn=lambda x: x[2],
            grouping=True,
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        # Create a mapping from ctx id to the original instance
        # This allows us to attach metadata back to the correct instances later
        # We use ctx (first element of args) as the key since chunks are grouped by args
        req_to_instance = {id(req.args[0]): req for req in requests}

        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        e2e_latency = 0
        total_tokens = 0
        for chunk in chunks:
            # Handle both single-item and multi-item chunks
            # Always convert to list for consistent handling
            if len(chunk) == 1:
                # Single item in chunk
                ctx = [chunk[0][0]]
                doc_to_messages = [chunk[0][1]]
                all_gen_kwargs = [chunk[0][2]]
                doc_id = [chunk[0][3]]
                task = [chunk[0][4]]
                split = [chunk[0][5]]
            else:
                # Multiple items in chunk - unzip and convert to list
                ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
                ctx = list(ctx)
                doc_to_messages = list(doc_to_messages)
                all_gen_kwargs = list(all_gen_kwargs)
                doc_id = list(doc_id)
                task = list(task)
                split = list(split)

            # Get actual docs to retrieve real doc IDs
            actual_docs = [self.task_dict[task[idx]][split[idx]][doc_id[idx]] for idx in range(len(doc_id))]

            chat_messages = [doc_to_messages[idx](actual_docs[idx]) for idx in range(len(doc_id))]
            chat_messages: List[ChatMessages] = [ChatMessages(**{"messages": message}) for message in chat_messages]
            visuals = []
            videos = []
            for messages in chat_messages:
                visual, video, _ = messages.extract_media()
                visuals.append(visual)
                videos.append(video)
            visuals = self.flatten(visuals)
            videos = self.flatten(videos)
            gen_kwargs = all_gen_kwargs[0]
            
            # TODO: 根据视频帧去从本地采样视频帧
            video_frame_paths = []
            for video in videos:
                frame_dir = video[:-4]
                video_frame_path = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir)])
                video_frame_paths.append(video_frame_path)
            
            # Apply chat template
            video_kwargs = {
                "max_pixels": self.max_pixels,
                "min_pixels": self.min_pixels,
            }
            if self.fps is not None:
                video_kwargs["fps"] = self.fps
            else:
                video_kwargs["nframes"] = self.max_num_frames
            
            batched_messages = [chat_message.to_hf_messages(video_kwargs=video_kwargs) for chat_message in chat_messages]
                        # batched_messages
            ''''content' =
[{'type': 'video', 'video': '/workspace/GX_Project/data/Video-MME/data/24i4ncHuf6A.mp4', 'max_pixels': 1605632, 'min_pixels': 200704, 'nframes': 32}, {'type': 'text', 'text': 'Select the best answer to the following multiple-choice question based on the video a...s letter from the given choices directly."}]
special variables
function variables
0 =
{'type': 'video', 'video': '/workspace/GX_Project/data/Video-MME/data/24i4ncHuf6A.mp4', 'max_pixels': 1605632, 'min_pixels': 200704, 'nframes': 32}
1 =
{'type': 'text', 'text': 'Select the best answer to the following multiple-choice question based on the video a...s letter from the given choices directly."}'''
            
            # "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.\nAccording to the video, how many individuals were in the car when Archduke Franz Ferdinand was assassinated?\nA. Three.\nB. Two.\nC. One.\nD. Four.\n\nAnswer with the option's letter from the given choices directly."
            # TODO: 获取一下原始问题的内容
            original_questions = []
            for msg in batched_messages:
                for content in msg[0]['content']:
                    if content['type'] == 'text':
                        text = content['text']
                        text = text.replace("Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.\n","")
                        question = text.split("\nA.")[0]
                        original_questions.append(question)
                        
            full_feats_list = []
            text_embed_list = []
            with torch.no_grad():
                for frame_paths, question in zip(video_frame_paths, original_questions):
                    
                    full_feats = self.clip_encoder.encode(
                        frame_paths,
                        batch_size=32,
                        convert_to_tensor=True,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        device='cuda:0'
                    ).float()
                    full_feats_list.append(full_feats)

                    text_embed = self.clip_encoder.encode(
                        [question],
                        convert_to_tensor=True,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        device='cuda:0'
                    )[0].float()
                    text_embed_list.append(text_embed)
          
            
            # TODO: 首先根据最大帧数去计算总共的token预算
            physical_patch_limits = []
            for video_frame_path in video_frame_paths:
                with Image.open(video_frame_path[0]) as sample_img:
                    orig_w, orig_h = sample_img.size
                physical_patch_limit = (orig_h // 14) * (orig_w // 14)
                physical_patch_limits.append(physical_patch_limit)
            
            print("="*20)
            
            print("Video Frame Paths and Physical Patch Limits:")
            print(f"physical_patch_limits: {physical_patch_limits}")
            
            print("="*20)
            
            print("Start DPP pruning and dynamic token allocation...")
            
            
            
            final_daynamic_video_frame_paths = []
            # Collect all frame metadata for this doc_id
            all_frame_metadata = []

            # TODO: 执行dpp选帧和动态分配token
            for full_feats, text_embed, video_frame_path, physical_patch_limit in zip(full_feats_list, text_embed_list, video_frame_paths, physical_patch_limits):
                selected_idx, selected_resolution = cdpruner_dpp_dynamic_resolution(
                    frame_features=full_feats,
                    frame_embeds=full_feats,
                    text_embed=text_embed,
                    total_token_budget=self.max_num_frames * physical_patch_limit,
                    min_token_per_frame=256,
                    max_token_per_frame=physical_patch_limit,
                    alpha=1.0
                )
                selected_indices = selected_idx.tolist()
                sorted_pairs = sorted(zip(selected_indices, selected_resolution.tolist()), key=lambda x: x[0])
                selected_indices = [i for i, _ in sorted_pairs]
                selected_resolution = [r for _, r in sorted_pairs]
                selected_frames = [video_frame_path[i] for i in selected_indices]
                selected_images, frame_metadata = load_and_resize_images(selected_frames, selected_resolution)
                final_daynamic_video_frame_paths.append(selected_images)
              

        
        
            new_batched_messages = []
            for msg_idx, msg in enumerate(batched_messages):
                
                content = []
                for frame_path, token_cnt in zip(
                        final_daynamic_video_frame_paths[msg_idx],
                        selected_resolution
                    ):
                    content.append({
                        "type": "image",
                        "image": frame_path,         
                    })
                
                content.append(batched_messages[msg_idx][0]['content'][-1])
                new_batched_messages.append([{
                    "role": "user",
                    "content": content
                }])

                
            
            texts = self.processor.apply_chat_template(new_batched_messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(new_batched_messages)
            
            
            if video_inputs is not None:
                total_frames = video_inputs[0].shape[0]
                # Only resample if we have more frames than needed
                if total_frames > self.max_num_frames:
                    indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                    # Append the last frame index if not already included
                    if total_frames - 1 not in indices:
                        indices = np.append(indices, total_frames - 1)
                    indices = np.unique(indices)  # Ensure uniqueness
                    video_inputs[0] = video_inputs[0][indices]
            else:
                # TODO: 处理多图的逻辑
                pass
                
            padding_side = "left" if self.batch_size > 1 else "right"
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                do_resize=False,
                videos=video_inputs,
                padding=True,
                padding_side=padding_side,
                return_tensors="pt",
            )

            image_grid_thw = inputs['image_grid_thw']

            #print("--- 每个视觉单元的 Token 分布 ---")
            for i, (t, h, w) in enumerate(image_grid_thw):
                num_tokens = int(h * w)
                #print(f"视觉单元 {i}: 时间维(T)={t}, 高度(H)={h}, 宽度(W)={w}, Token总数={num_tokens}")

            # 如果你想验证总数
            total_calculated = torch.sum(image_grid_thw[:, 1] * image_grid_thw[:, 2])
            print(f"验证总 Token: {total_calculated.item()}")

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            # Set default generation kwargs
            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,  # Set to 0 for greedy default
                "top_p": None,
                "num_beams": 1,
            }
            # Update with provided kwargs
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None
                current_gen_kwargs["top_k"] = None

            start_time = time.time()
            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                top_k=current_gen_kwargs.get("top_k", None),
                use_cache=self.use_cache,
            )
            end_time = time.time()

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            # Calculate timing metrics for batch
            e2e_latency += end_time - start_time
            total_tokens += sum(len(ids) for ids in generated_ids_trimmed)

            for ans, context in zip(answers, texts):
                clean_ans = parse_reasoning_model_answer(ans)
                res.append(clean_ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), clean_ans)

                eval_logger.debug(f"Question: {context}")
                eval_logger.debug(f"Model Raw Response: {ans}")
                eval_logger.debug(f"Model Clean Response: {clean_ans}")
            # reorder this group of results back to original unsorted form
            pbar.update(1)
        res = re_ords.get_original(res)

        # Calculate average speed
        avg_speed = total_tokens / e2e_latency if e2e_latency > 0 else 0
        # Log metrics
        metric_dict = {
            "total_tokens": total_tokens,
            "e2e_latency": e2e_latency,
            "avg_speed": avg_speed,
            "additional_metrics": {
                "rank": self.rank,
            },
        }
        log_metrics(**metric_dict)

        pbar.close()
        return res

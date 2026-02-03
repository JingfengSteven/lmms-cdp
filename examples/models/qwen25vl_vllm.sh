#!/bin/bash

# ---------------------- 基本配置 -----------------------
export PYTHONNOUSERSITE=1
export HF_HUB_ENABLE_HF_TRANSFER=False
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=18000000
export NCCL_DEBUG=DEBUG
export PYTHONPATH=$PYTHONPATH:$(pwd)
export HF_TOKEN="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"
python3 -m lmms_eval \
    --model vllm_dpp \
    --model_args model=/usr/project/xtmp/jc923/EACL/ckpt/qwen/Qwen/Qwen2.5-VL-7B-Instruct,tensor_parallel_size=4,max_model_len=42768,max_frame_num=8 \
    --tasks videomme \
    --batch_size 2 \
    --log_samples \
    --log_samples_suffix vllm \
    --output_path /usr/project/xtmp/jc923/EACL/eval/videomme \


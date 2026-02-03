# Run VideoMME evaluation with offline mode to prevent video download
# export HF_HOME="/workspace/GX_Project/data"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_TOKEN="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"
# export NCCL_P2P_DISABLE=1

# export CUDA_VISIBLE_DEVICES=2,7
export HF_HUB_ENABLE_HF_TRANSFER=False

accelerate launch --num_processes=1 --main_process_port=12342 -m lmms_eval \
    --model qwen2_5_vl_chat_multi_image \
    --model_args=pretrained=/usr/project/xtmp/jc923/EACL/ckpt/qwen/Qwen/Qwen2.5-VL-3B-Instruct,attn_implementation=flash_attention_2,interleave_visuals=False,max_num_frames=8 \
    --tasks videomme \
    --verbosity=DEBUG \
    --batch_size 2 \
    --output_path /workspace/GX_Project/ECCV/small_trail/eval/videomme \
    --log_samples

# pip3 install vllm
# pip3 install qwen_vl_utils

# cd ~/prod/lmms-eval-public
# pip3 install -e .
export HF_TOKEN="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"
# export NCCL_P2P_DISABLE=1

# export CUDA_VISIBLE_DEVICES=2,7
export PYTHONNOUSERSITE=1
export HF_HUB_ENABLE_HF_TRANSFER=False
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=18000000
export NCCL_DEBUG=DEBUG
export PYTHONNOUSERSITE=1

python3 -m lmms_eval \
    --model vllm \
    --model_args model=/usr/project/xtmp/jc923/EACL/ckpt/qwen/Qwen/Qwen2.5-VL-3B-Instruct,tensor_parallel_size=4,max_model_len=32768 \
    --tasks videomme \
    --batch_size 4 \
    --log_samples \
    --log_samples_suffix vllm \
    --output_path ./logs11 \
    --verbosity=DEBUG
    
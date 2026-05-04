#!/bin/bash
# Finetune DiffusionVL-QwenVL on NAVIDA-style JSONL (LazyNavidaJsonlDataset).
# Model: diffusionvl_qwenvl (Qwen2.5-VL + BD3-LM)
#
# Run from DiffusionVL/train:
#   bash scripts/diffusionvl_qwenvl_finetune_navida.sh <num_nodes> <gpus_per_node> [run_name] [bd3lm_block_size]
#
# Optional env overrides:
#   NAVIDA_JSONL=/path/to/data.jsonl
#   NAVIDA_MAX_HISTORY_FRAMES=8
#   PRETRAINED_CHECKPOINT=/path/to/converted/checkpoint
#   OUTPUT_DIR=./outputs/diffusionvl_qwenvl_navida
#   PRECISION=bf16|fp16   (default bf16; use fp16 only if you must; H100 prefers bf16 once CUDA works)
#
# If you see "CUDA initialization: The NVIDIA driver on your system is too old"
# or DeepSpeed "Setting accelerator to CPU" and then "doesn't support bf16/gpu":
#   Your installed PyTorch was built for a newer CUDA than your driver exposes (or
#   GPUs are not visible in this job). Fix by EITHER upgrading the host NVIDIA driver
#   OR reinstalling PyTorch wheels that match `nvidia-smi` "CUDA Version" (see
#   https://pytorch.org/get-started/locally/). In Kubernetes, also verify
#   nvidia.com/gpu limits and the NVIDIA device plugin.

set -euo pipefail

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL

export WANDB_DIR="${WANDB_DIR:-./wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-diffusionvl}"

# TODO: path to Qwen2.5-VL checkpoint in DiffusionVL / converted format
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-/mnt/data/vmo-ai-task/dungpq6/Qwen2.5-VL-7B-Instruct-DiffusionVL}"

# NAVIDA dataset (absolute image paths in jsonl are OK; --image_folder is a dummy)
NAVIDA_JSONL="${NAVIDA_JSONL:-/mnt/data/vmo-ai-task/dungpq6/navida/navida_train_data_r2r.jsonl}"
NAVIDA_MAX_HISTORY_FRAMES="${NAVIDA_MAX_HISTORY_FRAMES:-8}"
DATA_PATH="${NAVIDA_JSONL}"
IMAGE_FOLDER="."

OUTPUT_DIR="${OUTPUT_DIR:-/mnt/data/vmo-ai-task/dungpq6/diffusionvl_qwenvl_navida}"

num_node=${1:?usage: num_nodes gpus_per_node [run_name] [bd3lm_block_size]}
gpu_num=${2:?usage: num_nodes gpus_per_node [run_name] [bd3lm_block_size]}
custom_run_name=${3:-"diffusionvl_qwenvl_navida"}
BD3LM_BLOCK_SIZE=${4:-8}

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29199"}
RANK=${RANK:-"0"}

echo "=========================================="
echo "DiffusionVL-QwenVL finetune on NAVIDA JSONL"
echo "=========================================="
echo "NAVIDA_JSONL: ${NAVIDA_JSONL}"
echo "NAVIDA_MAX_HISTORY_FRAMES: ${NAVIDA_MAX_HISTORY_FRAMES}"
echo "Checkpoint: ${PRETRAINED_CHECKPOINT}"
echo "master_addr ${MASTER_ADDR}  master_port ${MASTER_PORT}  node_rank ${RANK}"
echo "num_node ${num_node}  gpu_num ${gpu_num}  BD3LM_BLOCK_SIZE ${BD3LM_BLOCK_SIZE}"
echo "BASE_RUN_NAME: ${custom_run_name}"

LLM_VERSION=${PRETRAINED_CHECKPOINT}
VISION_MODEL_VERSION=${PRETRAINED_CHECKPOINT}
PROMPT_VERSION=qwen_2_5

PRECISION="${PRECISION:-bf16}"
BF16_FLAG="False"
FP16_FLAG="False"
case "${PRECISION}" in
  bf16) BF16_FLAG="True" ;;
  fp16) FP16_FLAG="True" ;;
  *)
    echo "ERROR: PRECISION must be bf16 or fp16, got: ${PRECISION}"
    exit 1
    ;;
esac

echo "Precision: ${PRECISION} (bf16=${BF16_FLAG} fp16=${FP16_FLAG})"

python - <<'PY'
import sys
try:
    import torch
except Exception as e:
    print("ERROR: cannot import torch:", e)
    sys.exit(1)
if not torch.cuda.is_available():
    print("ERROR: torch.cuda.is_available() is False — training will not see GPUs.")
    print("  Fix: align PyTorch build with your NVIDIA driver (nvidia-smi 'CUDA Version')")
    print("  or fix container/device plugin so GPUs are visible to this process.")
    sys.exit(1)
n = torch.cuda.device_count()
print(f"OK: torch {torch.__version__}  cuda_runtime={torch.version.cuda}  gpu_count={n}")
for i in range(min(n, 4)):
    print("  ", i, torch.cuda.get_device_name(i))
PY

torchrun --nproc_per_node=${gpu_num} --nnodes=${num_node} --master_addr=${MASTER_ADDR} --master_port ${MASTER_PORT} --node_rank=${RANK} \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --dataset_format navida_jsonl \
    --navida_max_history_frames "${NAVIDA_MAX_HISTORY_FRAMES}" \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type qwen_merger \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio pad \
    --bf16 ${BF16_FLAG} \
    --fp16 ${FP16_FLAG} \
    --run_name "${custom_run_name}" \
    --output_dir "${OUTPUT_DIR}/${custom_run_name}" \
    --num_train_epochs 1 \
    --max_steps -1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 800 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --force_model_type "diffusionvl_qwenvl" \
    --bd3lm_block_aligned_eos True \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --dataloader_drop_last True \
    --attn_implementation sdpa \
    --use_conversation_mask False \
    --enable_bd3lm True \
    --bd3lm_block_size ${BD3LM_BLOCK_SIZE}

#!/bin/bash
# Finetune DiffusionVL-QwenVL on NAVIDA-style JSONL (LazyNavidaJsonlDataset).
# Model: diffusionvl_qwenvl (Qwen2.5-VL + BD3-LM)
#
# Run from DiffusionVL/train or repo root:
#   bash scripts/diffusionvl_qwenvl_finetune_navida.sh [num_nodes] [gpus_per_node] [run_name] [bd3lm_block_size]
#   bash train/scripts/diffusionvl_qwenvl_finetune_navida.sh
#
# Optional env overrides:
#   DATALOADER_NUM_WORKERS=8         (default; raise gradually if GPU is starved, not first knob)
#   OMP_NUM_THREADS=2                (default; raise only if CPU prep is clearly the bottleneck)
#   USE_FLASH_ATTN=1                 (use flash_attention_2 if flash-attn is installed)
#   REPORT_TO=none|wandb|tensorboard   (default wandb; use none to skip wandb entirely)
#   WANDB_MODE=offline|online          (default offline — local runs under WANDB_DIR, no sync prompt)
#   NAVIDA_JSONL=/path/to/data.jsonl
#   NAVIDA_MAX_HISTORY_FRAMES=8
#   PRETRAINED_CHECKPOINT=/path/to/converted/checkpoint
#   OUTPUT_DIR=./outputs/diffusionvl_qwenvl_navida
#   PRECISION=bf16|fp16   (default bf16; use fp16 only if you must; H100 prefers bf16 once CUDA works)
#
# Throughput / memory (defaults tuned for ~80GB H100 + NAVIDA + max_len 8192 + ZeRO-3):
#   Effective batch on N GPUs ≈ N * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
#   Default on 4 GPUs: 4 * 1 * 8 = 32 (VRAM-safe; bs=2+GC was ~3× slower in practice on this workload).
#   DATALOADER_NUM_WORKERS: each *rank* spawns this many processes — e.g. 24×4 ranks = 96 loaders → CPU/I/O thrash.
#       Start 4–8; try 12–16 only if GPUs wait on data (watch htop / nvitop). Do not set “high” blindly.
#   Optional try (after profiling): PER_DEVICE_TRAIN_BATCH_SIZE=2 GRADIENT_ACCUMULATION_STEPS=4 (keep GC=True).
#   USE_FLASH_ATTN=1 → flash_attention_2 (requires flash-attn); else sdpa.
#   MAX_STEPS=5000 for smoke runs
#
# Wall time: tqdm total steps ≈ dataset_size / (num_gpus * per_device_bs * grad_accum).
#   First ~50 steps can be noisy (compile/cache); compare s/it after that.
#
# If you see "CUDA initialization: The NVIDIA driver on your system is too old"
# or DeepSpeed "Setting accelerator to CPU" and then "doesn't support bf16/gpu":
#   Your installed PyTorch was built for a newer CUDA than your driver exposes (or
#   GPUs are not visible in this job). Fix by EITHER upgrading the host NVIDIA driver
#   OR reinstalling PyTorch wheels that match `nvidia-smi` "CUDA Version" (see
#   https://pytorch.org/get-started/locally/). In Kubernetes, also verify
#   nvidia.com/gpu limits and the NVIDIA device plugin.

set -euo pipefail

# DiffusionVL/train (parent of scripts/) — absolute paths for torchrun / deepspeed.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_TRAIN_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${_TRAIN_ROOT}"
export PYTHONPATH="${_TRAIN_ROOT}:${PYTHONPATH:-}"
TRAIN_MEM_PY="${_TRAIN_ROOT}/llava/train/train_mem.py"
DEEPSPEED_JSON="${_TRAIN_ROOT}/scripts/zero3.json"
if [ ! -f "${TRAIN_MEM_PY}" ]; then
  echo "ERROR: missing ${TRAIN_MEM_PY}" >&2
  exit 1
fi
if [ ! -f "${DEEPSPEED_JSON}" ]; then
  echo "ERROR: missing ${DEEPSPEED_JSON}" >&2
  exit 1
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Keep small: dataloader workers × ranks × OMP threads can oversubscribe CPU and slow every step.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# Do not set NCCL_DEBUG_SUBSYS=ALL here — very chatty and can slow multi-GPU jobs.
# For deep NCCL traces: export NCCL_DEBUG_SUBSYS=ALL (and often NCCL_DEBUG=INFO).

export WANDB_DIR="${WANDB_DIR:-./wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-diffusionvl}"
# Non-interactive runs (Slurm / no TTY): avoid wandb "Create account / ..." prompt blocking training.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
# Hugging Face / wandb pick up WANDB_* before Trainer starts.
REPORT_TO="${REPORT_TO:-wandb}"

# TODO: path to Qwen2.5-VL checkpoint in DiffusionVL / converted format
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-/mnt/data/vmo-ai-task/dungpq6/Qwen2.5-VL-7B-Instruct-DiffusionVL}"

# NAVIDA dataset (absolute image paths in jsonl are OK; --image_folder is a dummy)
NAVIDA_JSONL="${NAVIDA_JSONL:-/mnt/data/vmo-ai-task/dungpq6/navida/navida_train_data.jsonl}"
NAVIDA_MAX_HISTORY_FRAMES="${NAVIDA_MAX_HISTORY_FRAMES:-8}"
DATA_PATH="${NAVIDA_JSONL}"
IMAGE_FOLDER="."

OUTPUT_DIR="${OUTPUT_DIR:-/mnt/data/vmo-ai-task/dungpq6/diffusionvl_qwenvl_navida}"

# Training throughput defaults (override via env; conservative loaders — see header warning)
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
if [ "${USE_FLASH_ATTN:-0}" = "1" ]; then
  ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
else
  ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
fi
LOGGING_STEPS="${LOGGING_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-8192}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:--1}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"

num_node="${1:-1}"
if [ -n "${2:-}" ]; then
  gpu_num="$2"
elif [ -n "${NUM_GPUS:-}" ]; then
  gpu_num="${NUM_GPUS}"
elif command -v nvidia-smi >/dev/null 2>&1; then
  gpu_num="$(nvidia-smi -L 2>/dev/null | wc -l)"
  gpu_num="${gpu_num//[[:space:]]/}"
else
  gpu_num=1
fi
if ! [[ "${gpu_num}" =~ ^[0-9]+$ ]] || [ "${gpu_num}" -lt 1 ]; then
  echo "ERROR: invalid gpus_per_node='${gpu_num}' (pass as 2nd arg or set NUM_GPUS)" >&2
  exit 1
fi
custom_run_name=${3:-"diffusionvl_qwenvl_navida"}
BD3LM_BLOCK_SIZE=${4:-8}

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29199"}
RANK=${RANK:-"0"}

echo "=========================================="
echo "DiffusionVL-QwenVL finetune on NAVIDA JSONL"
echo "=========================================="
echo "TRAIN_ROOT: ${_TRAIN_ROOT}  (cwd: $(pwd))"
echo "NAVIDA_JSONL: ${NAVIDA_JSONL}"
echo "NAVIDA_MAX_HISTORY_FRAMES: ${NAVIDA_MAX_HISTORY_FRAMES}"
echo "Checkpoint: ${PRETRAINED_CHECKPOINT}"
echo "master_addr ${MASTER_ADDR}  master_port ${MASTER_PORT}  node_rank ${RANK}"
echo "num_node ${num_node}  gpu_num ${gpu_num}  BD3LM_BLOCK_SIZE ${BD3LM_BLOCK_SIZE}"
echo "BASE_RUN_NAME: ${custom_run_name}"
echo "Throughput: per_device_bs=${PER_DEVICE_TRAIN_BATCH_SIZE} grad_accum=${GRADIENT_ACCUMULATION_STEPS} workers=${DATALOADER_NUM_WORKERS} gc=${GRADIENT_CHECKPOINTING} attn=${ATTN_IMPLEMENTATION}"
echo "Train scope: epochs=${NUM_TRAIN_EPOCHS} max_steps=${MAX_STEPS} max_len=${MODEL_MAX_LENGTH} lr=${LEARNING_RATE}"
echo "Logging: REPORT_TO=${REPORT_TO} WANDB_MODE=${WANDB_MODE}"

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
    "${TRAIN_MEM_PY}" \
    --deepspeed "${DEEPSPEED_JSON}" \
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
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --max_steps "${MAX_STEPS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps "${SAVE_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --force_model_type "diffusionvl_qwenvl" \
    --bd3lm_block_aligned_eos True \
    --lr_scheduler_type "cosine" \
    --logging_steps "${LOGGING_STEPS}" \
    --tf32 True \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --lazy_preprocess True \
    --report_to "${REPORT_TO}" \
    --dataloader_drop_last True \
    --attn_implementation "${ATTN_IMPLEMENTATION}" \
    --use_conversation_mask False \
    --enable_bd3lm True \
    --bd3lm_block_size ${BD3LM_BLOCK_SIZE}

#!/bin/bash
# Finetune DiffusionVL-QwenVL on NAVIDA-style JSONL (LazyNavidaJsonlDataset).
# Model: diffusionvl_qwenvl (Qwen2.5-VL + BD3-LM)
#
# Run from DiffusionVL/train or repo root:
#   bash scripts/diffusionvl_qwenvl_finetune_navida.sh [num_nodes] [gpus_per_node] [run_name] [bd3lm_block_size]
#   bash train/scripts/diffusionvl_qwenvl_finetune_navida.sh
# Omitted args: num_nodes=1; gpus from arg 2, else NUM_GPUS / nvidia-smi count, else 1.
#
# Optional env overrides:
#   NAVIDA_JSONL, PRETRAINED_CHECKPOINT, OUTPUT_DIR, PRECISION=bf16|fp16
#   PER_DEVICE_TRAIN_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, DATALOADER_NUM_WORKERS
#   GRADIENT_CHECKPOINTING=True|False
#   USE_FLASH_ATTN=1 → try flash_attention_2 (can be unstable on some stacks; default is 0 = sdpa)
#   NAVIDA_MICRO_BS2=0 → fallback bs=1 accum=8 (default is bs=2 accum=4 for speed)
#   MODEL_MAX_LENGTH=4096 (default below; faster and lower VRAM than 8192)
#   REPORT_TO=none|wandb, WANDB_MODE=offline|online
#
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

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

export WANDB_DIR="${WANDB_DIR:-./wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-diffusionvl}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
REPORT_TO="${REPORT_TO:-wandb}"

PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-/mnt/data/vmo-ai-task/dungpq6/Qwen2.5-VL-3B-Instruct-DiffusionVL}"
NAVIDA_JSONL="${NAVIDA_JSONL:-/mnt/data/vmo-ai-task/dungpq6/navida/navida_train_data.jsonl}"
NAVIDA_MAX_HISTORY_FRAMES="${NAVIDA_MAX_HISTORY_FRAMES:-8}"
DATA_PATH="${NAVIDA_JSONL}"
IMAGE_FOLDER="."
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/data/vmo-ai-task/dungpq6/diffusionvl_qwenvl_navida}"

NAVIDA_MICRO_BS2="${NAVIDA_MICRO_BS2:-1}"
if [ "${NAVIDA_MICRO_BS2}" = "1" ]; then
  PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
  GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
else
  PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
  GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
fi
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-12}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
USE_FLASH_ATTN="${USE_FLASH_ATTN:-0}"
FLASH_ATTN_AVAILABLE="no"
if python3 -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('flash_attn') else 1)" >/dev/null 2>&1; then
  FLASH_ATTN_AVAILABLE="yes"
fi
if [ -n "${ATTN_IMPLEMENTATION:-}" ]; then
  :
elif [ "${USE_FLASH_ATTN}" = "1" ] && [ "${FLASH_ATTN_AVAILABLE}" = "yes" ]; then
  ATTN_IMPLEMENTATION=flash_attention_2
else
  ATTN_IMPLEMENTATION=sdpa
fi
LOGGING_STEPS="${LOGGING_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-4096}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:--1}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"

for _arg in "${1:-}" "${2:-}" "${3:-}" "${4:-}"; do
  if [ -n "${_arg}" ] && [ "${_arg}" = "..." ]; then
    echo "ERROR: remove literal '...' from arguments (e.g. bash train/scripts/diffusionvl_qwenvl_finetune_navida.sh 1 4 run_name)" >&2
    exit 1
  fi
done

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
if ! [[ "${num_node}" =~ ^[0-9]+$ ]] || [ "${num_node}" -lt 1 ]; then
  echo "ERROR: num_nodes (1st arg) must be a positive integer, got: '${num_node}'" >&2
  exit 1
fi
if ! [[ "${gpu_num}" =~ ^[0-9]+$ ]] || [ "${gpu_num}" -lt 1 ]; then
  echo "ERROR: invalid gpus_per_node='${gpu_num}' (pass as 2nd arg or set NUM_GPUS)" >&2
  exit 1
fi
custom_run_name=${3:-"diffusionvl_qwenvl_navida"}
BD3LM_BLOCK_SIZE=${4:-8}
if ! [[ "${BD3LM_BLOCK_SIZE}" =~ ^[0-9]+$ ]] || [ "${BD3LM_BLOCK_SIZE}" -lt 1 ]; then
  echo "ERROR: bd3lm_block_size (4th arg) must be a positive integer, got: '${BD3LM_BLOCK_SIZE}'" >&2
  exit 1
fi

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
echo "Attention backend probe: flash_attn_available=${FLASH_ATTN_AVAILABLE} use_flash_attn=${USE_FLASH_ATTN}"
if [ "${USE_FLASH_ATTN}" = "1" ] && [ "${FLASH_ATTN_AVAILABLE}" = "yes" ] && [ "${ATTN_IMPLEMENTATION}" = "flash_attention_2" ]; then
  echo "Warning: flash_attention_2 enabled; if you hit device-side assert, rerun with USE_FLASH_ATTN=0"
fi
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

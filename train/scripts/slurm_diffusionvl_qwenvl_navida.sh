#!/bin/bash -e
#SBATCH --job-name=diffusionvl-navida
#SBATCH --output=logs/diffusionvl_navida_%j.log
#SBATCH --error=logs/diffusionvl_navida_%j.err
#SBATCH --nodelist=worker-0
#SBATCH --gpus=4
#SBATCH --cpus-per-task=60
#SBATCH --mem-per-cpu=8192
#
#SBATCH --container-image=/mnt/data/vmo-ai-task/dungpq6/ubuntu22-cuda128-conda-navida.sqsh
#SBATCH --container-mounts=/mnt/data/:/mnt/data/,/home/dungpq6/Project:/home/dungpq6/Project

# DiffusionVL Qwen-VL finetune on NAVIDA jsonl (see diffusionvl_qwenvl_finetune_navida.sh).
# Submit from anywhere; script cds to TRAIN_ROOT and creates logs/ there.
#
# Optional env overrides before sbatch:
#   RUN_NAME=diffusionvl_navida_r2r
#   BD3LM_BLOCK_SIZE=8
#   PRETRAINED_CHECKPOINT=/path/to/Qwen2.5-VL-7B-Instruct-DiffusionVL
#   NAVIDA_JSONL=/path/to/navida_train_data_r2r.jsonl
#   OUTPUT_DIR=/path/to/outputs
#   PRECISION=bf16|fp16
#   TRAIN_ROOT=/home/dungpq6/Project/DiffusionVL/train
#
# Throughput preset (set USE_THROUGHPUT_PRESET=0 to skip and use finetune script defaults only):
#   USE_THROUGHPUT_PRESET=1  (default) exports below unless you already exported overrides.
# For another ~2x wall-time cut vs bs=1,gc=on, try after stable run:
#   PER_DEVICE_TRAIN_BATCH_SIZE=4 GRADIENT_ACCUMULATION_STEPS=2 (same effective batch 32 on 4 GPUs)
#   ATTN_IMPLEMENTATION=flash_attention_2  (if flash-attn is installed in the container)
USE_THROUGHPUT_PRESET="${USE_THROUGHPUT_PRESET:-1}"
if [ "${USE_THROUGHPUT_PRESET}" = "1" ]; then
  export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
  export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
  export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-False}"
  export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-16}"
  export LOGGING_STEPS="${LOGGING_STEPS:-100}"
  export SAVE_STEPS="${SAVE_STEPS:-5000}"
fi
# Log paths below are relative to your sbatch submission cwd unless you use absolute paths.

set -euo pipefail

source /home/dungpq6/anaconda3/etc/profile.d/conda.sh
conda activate diffusionvl

# IMPORTANT: SLURM may execute a copied script under /var/spool/slurmd.
# Use explicit train root (directory that contains llava/ and scripts/).
TRAIN_ROOT="${TRAIN_ROOT:-/home/dungpq6/Project/DiffusionVL/train}"
cd "${TRAIN_ROOT}"
mkdir -p logs

export PYTHONPATH="${TRAIN_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_PROJECT="${WANDB_PROJECT:-diffusionvl}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# Model / data / output (defaults match typical server paths; override in sbatch env)
export PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-/mnt/data/vmo-ai-task/dungpq6/Qwen2.5-VL-7B-Instruct-DiffusionVL}"
export NAVIDA_JSONL="${NAVIDA_JSONL:-/mnt/data/vmo-ai-task/dungpq6/navida/navida_train_data.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/data/vmo-ai-task/dungpq6/diffusionvl_qwenvl_navida}"
export NAVIDA_MAX_HISTORY_FRAMES="${NAVIDA_MAX_HISTORY_FRAMES:-8}"
export PRECISION="${PRECISION:-bf16}"

NUM_NODES="${NUM_NODES:-1}"
NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-4}}"
RUN_NAME="${RUN_NAME:-diffusionvl_navida_r2r}"
BD3LM_BLOCK_SIZE="${BD3LM_BLOCK_SIZE:-8}"

echo "TRAIN_ROOT=${TRAIN_ROOT}"
echo "NUM_NODES=${NUM_NODES} NUM_GPUS=${NUM_GPUS}"
echo "RUN_NAME=${RUN_NAME} BD3LM_BLOCK_SIZE=${BD3LM_BLOCK_SIZE}"
echo "PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT}"
echo "NAVIDA_JSONL=${NAVIDA_JSONL}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "PRECISION=${PRECISION}"
echo "WANDB_PROJECT=${WANDB_PROJECT} WANDB_MODE=${WANDB_MODE}"

# Single-node torchrun: num_node gpu_num [run_name] [bd3lm_block_size]
bash scripts/diffusionvl_qwenvl_finetune_navida.sh \
  "${NUM_NODES}" \
  "${NUM_GPUS}" \
  "${RUN_NAME}" \
  "${BD3LM_BLOCK_SIZE}"

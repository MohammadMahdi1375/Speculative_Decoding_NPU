#!/usr/bin/env bash
# =============================================================================
# DFlash training — SGLang target backend, multinode, Ascend NPU
# Launches torchrun in multi-node mode while preserving the SGLang-backend
# env-var contract from run_qwen3_8b_dflash_npu_sglang.sh.
# =============================================================================
set -euo pipefail

# ---- Repo root --------------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." &>/dev/null && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

# ---- Multinode rendezvous (must be exported by caller) ----------------------
: "${MASTER_ADDR:?MASTER_ADDR must be set (IP of NODE_RANK=0)}"
: "${MASTER_PORT:=29500}"
: "${NNODES:=1}"
: "${NODE_RANK:=0}"
: "${HCCL_SOCKET_IFNAME:?HCCL_SOCKET_IFNAME must be set to the inter-node NIC}"
export MASTER_ADDR MASTER_PORT NNODES NODE_RANK HCCL_SOCKET_IFNAME

# ---- Device + SGLang knobs (overridable) ------------------------------------
export SPECFORGE_DEVICE=${SPECFORGE_DEVICE:-npu}
export SPECFORGE_DIST_BACKEND=${SPECFORGE_DIST_BACKEND:-hccl}

NUM_NPUS=${NUM_NPUS:-8}
TP_SIZE=${TP_SIZE:-2}
SEED=${SEED:-42}
BATCH_SIZE=${BATCH_SIZE:-1}
MAX_LENGTH=${MAX_LENGTH:-2048}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-sdpa}

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-$(seq -s, 0 $((NUM_NPUS-1)))}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
export SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.10}
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=${SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN:-1}

# ---- Paths (overridable) ----------------------------------------------------
TARGET_MODEL=${TARGET_MODEL:-Qwen/Qwen3-8B}
TRAIN_DATA=${TRAIN_DATA:?TRAIN_DATA must be set}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/qwen3-8b-dflash-sglang-multinode}
DRAFT_CONFIG=${DRAFT_CONFIG:-$ROOT_DIR/configs/qwen3-8b-dflash.json}

# ---- Hyperparams (overridable) ----------------------------------------------
NUM_EPOCHS=${NUM_EPOCHS:-2}
LEARNING_RATE=${LEARNING_RATE:-6e-4}
WARMUP_RATIO=${WARMUP_RATIO:-0.04}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
NUM_ANCHORS=${NUM_ANCHORS:-512}
LOSS_DECAY_GAMMA=${LOSS_DECAY_GAMMA:-7.0}
LOG_INTERVAL=${LOG_INTERVAL:-50}
SAVE_INTERVAL=${SAVE_INTERVAL:-44386}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-qwen}

# ---- HF cache (offline-first) -----------------------------------------------
export HF_HOME=${HF_HOME:-/share/canada_group_folder/hf_cache/}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export HF_HUB_DISABLE_TELEMETRY=1

# ---- WandB (optional) -------------------------------------------------------
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_PROJECT=${WANDB_PROJECT:-specforge-qwen3-8b-dflash}
export WANDB_NAME=${WANDB_NAME:-qwen3-8b-dflash-sglang-2node}

# ---- Banner -----------------------------------------------------------------
WORLD_SIZE=$((NNODES * NUM_NPUS))
echo "================================================================"
echo "[INFO] DFlash SGLang-backend multinode training"
echo "[INFO] NNODES=$NNODES NODE_RANK=$NODE_RANK NUM_NPUS=$NUM_NPUS WORLD_SIZE=$WORLD_SIZE"
echo "[INFO] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "[INFO] HCCL_SOCKET_IFNAME=$HCCL_SOCKET_IFNAME"
echo "[INFO] ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
echo "[INFO] TP_SIZE=$TP_SIZE  -> $((WORLD_SIZE / TP_SIZE)) SGLang engine group(s) total"
echo "[INFO] SGLANG_MEM_FRACTION_STATIC=$SGLANG_MEM_FRACTION_STATIC"
echo "[INFO] TARGET_MODEL=$TARGET_MODEL"
echo "[INFO] TRAIN_DATA=$TRAIN_DATA"
echo "[INFO] OUTPUT_DIR=$OUTPUT_DIR"
echo "[INFO] BATCH_SIZE=$BATCH_SIZE MAX_LENGTH=$MAX_LENGTH"
echo "================================================================"

mkdir -p "$OUTPUT_DIR"

# ---- Launch -----------------------------------------------------------------
torchrun \
  --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --nproc_per_node="$NUM_NPUS" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  "$ROOT_DIR/scripts/train_dflash.py" \
    --target-model-path "$TARGET_MODEL" \
    --target-model-backend sglang \
    --tp-size "$TP_SIZE" \
    --seed "$SEED" \
    --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC" \
    --draft-config-path "$DRAFT_CONFIG" \
    --train-data-path "$TRAIN_DATA" \
    --output-dir "$OUTPUT_DIR" \
    --num-epochs "$NUM_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --warmup-ratio "$WARMUP_RATIO" \
    --max-grad-norm "$MAX_GRAD_NORM" \
    --max-length "$MAX_LENGTH" \
    --chat-template "$CHAT_TEMPLATE" \
    --attention-backend "$ATTENTION_BACKEND" \
    --num-anchors "$NUM_ANCHORS" \
    --loss-decay-gamma "$LOSS_DECAY_GAMMA" \
    --log-interval "$LOG_INTERVAL" \
    --save-interval "$SAVE_INTERVAL" \
    --report-to wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-name "$WANDB_NAME"